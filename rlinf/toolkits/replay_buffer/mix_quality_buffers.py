import argparse
import os
import random
from pathlib import Path

from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _load_buffer(path: str, seed: int) -> TrajectoryReplayBuffer:
    buf = TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=False,
        cache_size=1,
        sample_window_size=1,
        auto_save=False,
        trajectory_format="pt",
    )
    buf.load_checkpoint(path, is_distributed=False)
    return buf


def _collect_ids(buf: TrajectoryReplayBuffer):
    return list(buf._trajectory_id_list)


def _copy_selected(src: TrajectoryReplayBuffer, ids: list[int], dst: TrajectoryReplayBuffer):
    trajectories = []
    for tid in ids:
        info = src._trajectory_index[tid]
        traj = src._load_trajectory(tid, info["model_weights_id"])
        trajectories.append(traj)
    dst.add_trajectories(trajectories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-path", required=True)
    parser.add_argument("--failure-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--success-ratio", type=float, default=0.8)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    success_buf = _load_buffer(args.success_path, args.seed)
    failure_buf = _load_buffer(args.failure_path, args.seed + 1)

    success_ids = _collect_ids(success_buf)
    failure_ids = _collect_ids(failure_buf)
    random.shuffle(success_ids)
    random.shuffle(failure_ids)

    if args.max_trajectories > 0:
        n_success = int(round(args.max_trajectories * args.success_ratio))
        n_failure = max(args.max_trajectories - n_success, 0)
        success_ids = success_ids[: min(n_success, len(success_ids))]
        failure_ids = failure_ids[: min(n_failure, len(failure_ids))]

    out = TrajectoryReplayBuffer(
        seed=args.seed,
        enable_cache=False,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=args.output_path,
        trajectory_format="pt",
    )
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    _copy_selected(success_buf, success_ids, out)
    _copy_selected(failure_buf, failure_ids, out)
    out.close(wait=True)
    print({
        "output_path": args.output_path,
        "num_success": len(success_ids),
        "num_failure": len(failure_ids),
        "total": len(success_ids) + len(failure_ids),
    })


if __name__ == "__main__":
    main()
