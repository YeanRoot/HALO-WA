#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlinf.data.replay_buffer import TrajectoryReplayBuffer


SLIDE_PATTERN = re.compile(r"(?:^|_)slide\d+(?:_|$)", re.IGNORECASE)
RERANK_SUFFIX_PATTERN = re.compile(r"_rerank\d+$", re.IGNORECASE)


def make_buffer_for_load(
    seed: int = 1234,
    cache_size: int = 64,
    sample_window_size: int = 1,
) -> TrajectoryReplayBuffer:
    # auto_save=False requires cache to be enabled in TrajectoryReplayBuffer.
    return TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=True,
        cache_size=max(1, cache_size),
        sample_window_size=max(1, sample_window_size),
        auto_save=False,
        trajectory_format="pt",
    )


def make_buffer_for_save(path: str, seed: int, trajectory_format: str = "pt") -> TrajectoryReplayBuffer:
    Path(path).mkdir(parents=True, exist_ok=True)
    return TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=False,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=path,
        trajectory_format=trajectory_format,
    )


def discover_source_shards(input_root: Path, bucket: str) -> List[Path]:
    shards = []
    for p in sorted(input_root.glob("rank_*")):
        shard = p / bucket
        if shard.is_dir() and (shard / "metadata.json").exists():
            shards.append(shard)
    if not shards:
        raise FileNotFoundError(
            f"No source shards found under {input_root} matching rank_*/{bucket}/metadata.json"
        )
    return shards


def open_source_buffers(shards: List[Path], seed: int, cache_size: int) -> Dict[str, TrajectoryReplayBuffer]:
    buffers: Dict[str, TrajectoryReplayBuffer] = {}
    for shard_idx, shard in enumerate(shards):
        buf = make_buffer_for_load(seed + shard_idx, cache_size=cache_size)
        buf.load_checkpoint(str(shard), is_distributed=False)
        buffers[str(shard)] = buf
    return buffers


def normalize_storage_name(name: Optional[Any]) -> Optional[str]:
    if name is None:
        return None
    name = str(name)
    if name.endswith(".pt"):
        return name[:-3]
    return name


def default_source_storage_name(trajectory_id: int, model_weights_id: str) -> str:
    return f"trajectory_{trajectory_id}_{model_weights_id}"


def normalize_for_mode_classification(name: Optional[str]) -> str:
    normalized = normalize_storage_name(name) or ""
    normalized = RERANK_SUFFIX_PATTERN.sub("", normalized)
    return normalized


def infer_data_mode_from_name(source_storage_name: Optional[str]) -> str:
    name = normalize_for_mode_classification(source_storage_name)
    if SLIDE_PATTERN.search(name):
        return "sliding"
    return "original"


def filter_handles_by_data_mode(handles: List[Dict], data_mode: str) -> List[Dict]:
    if data_mode == "all":
        return handles

    expected = "original" if data_mode == "original" else "sliding"
    filtered = [h for h in handles if infer_data_mode_from_name(h.get("source_storage_name")) == expected]
    return filtered


def summarize_handle_modes(handles: List[Dict]) -> Dict[str, int]:
    summary = {"original": 0, "sliding": 0}
    for h in handles:
        mode = infer_data_mode_from_name(h.get("source_storage_name"))
        summary[mode] = summary.get(mode, 0) + 1
    return summary


def collect_trajectory_handles(
    source_buffers: Dict[str, TrajectoryReplayBuffer],
    shuffle: bool,
    seed: int,
) -> List[Dict]:
    handles: List[Dict] = []
    for shard_path, buf in source_buffers.items():
        traj_ids = list(buf._trajectory_id_list)
        traj_ids.sort()
        for tid in traj_ids:
            info = buf._trajectory_index[tid]
            source_storage_name = normalize_storage_name(info.get("storage_name"))
            if not source_storage_name:
                source_storage_name = normalize_storage_name(info.get("source_storage_name"))
            if not source_storage_name:
                source_storage_name = default_source_storage_name(int(tid), info["model_weights_id"])
            handles.append(
                {
                    "shard_path": shard_path,
                    "trajectory_id": int(tid),
                    "model_weights_id": info["model_weights_id"],
                    "source_storage_name": source_storage_name,
                    "data_mode": infer_data_mode_from_name(source_storage_name),
                }
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(handles)
    return handles


def load_one_trajectory(
    source_buffers: Dict[str, TrajectoryReplayBuffer],
    shard_path: str,
    trajectory_id: int,
    model_weights_id: str,
):
    buf = source_buffers[shard_path]
    return buf._load_trajectory(trajectory_id, model_weights_id)


def close_buffers(buffers: Dict[str, TrajectoryReplayBuffer]) -> None:
    for buf in buffers.values():
        try:
            buf.close(wait=True)
        except Exception:
            pass


def write_manifest(output_root: Path, manifest: Dict) -> None:
    with open(output_root / "reshard_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def find_saved_file(rank_dir: Path, trajectory_id: int, model_weights_id: str, entry: Dict, ext: str) -> Path:
    candidates = []
    storage_name = entry.get("storage_name")
    if storage_name:
        candidates.append(rank_dir / f"{storage_name}{ext}")
    candidates.append(rank_dir / f"trajectory_{trajectory_id}_{model_weights_id}{ext}")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find saved file for trajectory_id={trajectory_id}, model_weights_id={model_weights_id} under {rank_dir}"
    )


def make_unique_name(rank_dir: Path, base_name: str, ext: str) -> str:
    candidate = base_name
    counter = 1
    while (rank_dir / f"{candidate}{ext}").exists():
        candidate = f"{base_name}_dup{counter}"
        counter += 1
    return candidate


def rename_output_files(
    output_root: Path,
    target_records: List[List[Dict]],
    target_world_size: int,
    trajectory_format: str = "pt",
) -> None:
    ext = ".pt" if trajectory_format == "pt" else ".pkl"

    for rank in range(target_world_size):
        rank_dir = output_root / f"rank_{rank}"
        index_path = rank_dir / "trajectory_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing trajectory_index.json under {rank_dir}")

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        trajectory_index = index_data.get("trajectory_index", {})
        for rec in target_records[rank]:
            tid_key = str(rec["trajectory_id"])
            if tid_key not in trajectory_index:
                raise KeyError(f"trajectory_id={tid_key} not found in {index_path}")

            entry = trajectory_index[tid_key]
            old_path = find_saved_file(rank_dir, rec["trajectory_id"], rec["model_weights_id"], entry, ext)

            source_storage_name = normalize_storage_name(rec.get("source_storage_name"))
            if not source_storage_name:
                source_storage_name = default_source_storage_name(rec["trajectory_id"], rec["model_weights_id"])
            desired_base = f"{source_storage_name}_rerank{rank}"
            desired_base = make_unique_name(rank_dir, desired_base, ext) if old_path.name != f"{desired_base}{ext}" else desired_base
            new_path = rank_dir / f"{desired_base}{ext}"

            if old_path != new_path:
                old_path.rename(new_path)

            entry["storage_name"] = desired_base
            entry["source_storage_name"] = source_storage_name
            entry["rerank_target"] = rank
            entry["data_mode"] = rec.get("data_mode", infer_data_mode_from_name(source_storage_name))

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Merge/re-shard offline_collection rank shards into a new world-size layout."
    )
    parser.add_argument("--input-root", required=True, help="Path like .../offline_collection")
    parser.add_argument(
        "--bucket",
        required=True,
        choices=["all", "success", "failure"],
        help="Which bucket under rank_i to reshard",
    )
    parser.add_argument(
        "--data-mode",
        default="all",
        choices=["all", "original", "sliding"],
        help=(
            "Filter trajectories by source name pattern before re-sharding: "
            "'original' keeps non-slide files only, 'sliding' keeps *_slide* files only, "
            "'all' keeps everything."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root. Script creates output_root/rank_0 ... output_root/rank_{N-1}",
    )
    parser.add_argument(
        "--target-world-size",
        required=True,
        type=int,
        help="How many target rank shards to produce for later training",
    )
    parser.add_argument(
        "--trajectory-format",
        default="pt",
        choices=["pt"],
        help="Trajectory format for output buffer",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle trajectories before repartitioning (recommended)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="How many trajectories to buffer in memory per output shard before flushing",
    )
    parser.add_argument(
        "--source-cache-size",
        type=int,
        default=64,
        help="CPU cache size for each opened source shard buffer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print summary, do not write output",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    target_world_size = int(args.target_world_size)
    if target_world_size <= 0:
        raise ValueError("--target-world-size must be > 0")

    source_shards = discover_source_shards(input_root, args.bucket)
    source_buffers = open_source_buffers(source_shards, seed=args.seed, cache_size=args.source_cache_size)
    all_handles = collect_trajectory_handles(source_buffers, shuffle=args.shuffle, seed=args.seed)
    all_mode_summary = summarize_handle_modes(all_handles)
    handles = filter_handles_by_data_mode(all_handles, data_mode=args.data_mode)
    filtered_mode_summary = summarize_handle_modes(handles)
    total = len(handles)
    if total == 0:
        close_buffers(source_buffers)
        raise RuntimeError(
            f"No trajectories found to reshard after applying --data-mode={args.data_mode}. "
            f"Available summary: {all_mode_summary}"
        )

    per_target = math.ceil(total / target_world_size)
    print(f"[reshard] input_root={input_root}")
    print(f"[reshard] bucket={args.bucket}")
    print(f"[reshard] data_mode={args.data_mode}")
    print(f"[reshard] source_shards={len(source_shards)}")
    print(f"[reshard] total_before_filter={len(all_handles)}")
    print(f"[reshard] mode_summary_before_filter={all_mode_summary}")
    print(f"[reshard] total_trajectories={total}")
    print(f"[reshard] mode_summary_after_filter={filtered_mode_summary}")
    print(f"[reshard] target_world_size={target_world_size}")
    print(f"[reshard] approx_trajectories_per_target={per_target}")
    print(f"[reshard] output_root={output_root}")
    if handles:
        print(f"[reshard] example source name -> target pattern: {handles[0]['source_storage_name']}_rerank0.pt")

    manifest = {
        "input_root": str(input_root),
        "bucket": args.bucket,
        "data_mode": args.data_mode,
        "source_shards": [str(x) for x in source_shards],
        "target_world_size": target_world_size,
        "total_before_filter": len(all_handles),
        "total_trajectories": total,
        "mode_summary_before_filter": all_mode_summary,
        "mode_summary_after_filter": filtered_mode_summary,
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "source_cache_size": int(args.source_cache_size),
        "output_naming": "{original_name}_rerank{target_rank}.pt",
        "data_mode_rule": {
            "original": "source_storage_name without *_slide* pattern",
            "sliding": "source_storage_name containing *_slide* pattern",
        },
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        close_buffers(source_buffers)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    target_buffers: List[TrajectoryReplayBuffer] = []
    pending = [[] for _ in range(target_world_size)]
    target_counts = [0 for _ in range(target_world_size)]
    target_records: List[List[Dict]] = [[] for _ in range(target_world_size)]

    for rank in range(target_world_size):
        rank_dir = output_root / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        target_buffers.append(
            make_buffer_for_save(
                str(rank_dir),
                seed=args.seed + rank,
                trajectory_format=args.trajectory_format,
            )
        )

    try:
        for idx, h in enumerate(handles):
            target_rank = idx % target_world_size
            target_local_trajectory_id = target_counts[target_rank]

            traj = load_one_trajectory(
                source_buffers=source_buffers,
                shard_path=h["shard_path"],
                trajectory_id=h["trajectory_id"],
                model_weights_id=h["model_weights_id"],
            )
            pending[target_rank].append(traj)
            target_records[target_rank].append(
                {
                    "trajectory_id": target_local_trajectory_id,
                    "model_weights_id": h["model_weights_id"],
                    "source_storage_name": h["source_storage_name"],
                    "data_mode": h["data_mode"],
                }
            )
            target_counts[target_rank] += 1

            if len(pending[target_rank]) >= args.chunk_size:
                target_buffers[target_rank].add_trajectories(pending[target_rank])
                pending[target_rank].clear()

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[reshard] processed {idx + 1}/{total} trajectories")

        for rank in range(target_world_size):
            if pending[rank]:
                target_buffers[rank].add_trajectories(pending[rank])
                pending[rank].clear()
            target_buffers[rank].close(wait=True)
    finally:
        close_buffers(source_buffers)

    rename_output_files(
        output_root=output_root,
        target_records=target_records,
        target_world_size=target_world_size,
        trajectory_format=args.trajectory_format,
    )

    manifest["target_counts"] = target_counts
    write_manifest(output_root, manifest)
    print("[reshard] done")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
