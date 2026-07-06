# Copyright 2026 The RLinf Authors.

import json
import time
from collections import defaultdict

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)




def _aggregate_collection_stats(stats):
    if isinstance(stats, dict):
        return stats
    if not isinstance(stats, list):
        return {}
    agg = {"enabled": False, "quality_mode": None, "success_threshold": None}
    for item in stats:
        if not isinstance(item, dict):
            continue
        agg["enabled"] = agg["enabled"] or bool(item.get("enabled", False))
        if agg["quality_mode"] is None:
            agg["quality_mode"] = item.get("quality_mode", None)
        if agg["success_threshold"] is None:
            agg["success_threshold"] = item.get("success_threshold", None)
        for key in ["all", "success", "failure"]:
            sub = item.get(key, None)
            if not isinstance(sub, dict):
                continue
            if key not in agg:
                agg[key] = {"num_trajectories": 0, "total_samples": 0}
            agg[key]["num_trajectories"] += int(sub.get("num_trajectories", 0))
            agg[key]["total_samples"] += int(sub.get("total_samples", 0))
    return agg

def _avg_env_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in (metrics or {}).items():
        try:
            if hasattr(v, "float"):
                out[k] = float(v.float().mean().item())
            else:
                out[k] = float(v)
        except Exception:
            continue
    return out


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_place_empty_cup_collect_gigawa_offline_fixed",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    rollout_placement = component_placement.get_strategy("rollout")
    env_placement = component_placement.get_strategy("env")

    if cfg.algorithm.loss_type != "embodied_gigawa":
        raise ValueError(
            f"collect_embodied_agent_gigawa.py requires algorithm.loss_type=embodied_gigawa, got {cfg.algorithm.loss_type}"
        )

    from rlinf.workers.actor.fsdp_gigawa_policy_worker import EmbodiedGigaWAFSDPPolicy

    actor_group = EmbodiedGigaWAFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    env_channel = Channel.create("Env")
    rollout_channel = Channel.create("Rollout")
    actor_channel = Channel.create("Actor")

    rollout_group.init_worker().wait()
    env_group.init_worker().wait()
    actor_group.init_worker().wait()

    resume_dir = cfg.runner.get("resume_dir", None)
    if resume_dir:
        actor_checkpoint_path = f"{resume_dir}/actor"
        actor_group.load_checkpoint(actor_checkpoint_path).wait()

    def sync_weights():
        rollout_handle = rollout_group.sync_model_from_actor()
        actor_handle = actor_group.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    collect_cfg = cfg.algorithm.offline_collection
    target_num_trajectories = int(collect_cfg.get("target_num_trajectories", 0) or 0)
    target_total_samples = int(collect_cfg.get("target_total_samples", 0) or 0)
    log_interval = int(collect_cfg.get("log_interval", 10))
    max_collection_steps = int(collect_cfg.get("max_collection_steps", 10**9))

    start_time = time.time()
    last_log_time = start_time
    step = 0
    sync_weights()

    while step < max_collection_steps:
        env_handle = env_group.interact(
            input_channel=env_channel,
            output_channel=rollout_channel,
            actor_channel=actor_channel,
        )
        rollout_handle = rollout_group.generate(
            input_channel=rollout_channel,
            output_channel=env_channel,
        )
        actor_group.recv_rollout_trajectories(input_channel=actor_channel).wait()
        env_metrics_list = env_handle.wait()
        rollout_handle.wait()
        step += 1

        stats = _aggregate_collection_stats(
            actor_group.get_offline_collection_stats().wait()
        )
        all_stats = stats.get("all", {})
        collected_traj = int(all_stats.get("num_trajectories", 0))
        collected_samples = int(all_stats.get("total_samples", 0))

        should_stop = False
        if target_num_trajectories > 0 and collected_traj >= target_num_trajectories:
            should_stop = True
        if target_total_samples > 0 and collected_samples >= target_total_samples:
            should_stop = True

        if step == 1 or step % log_interval == 0 or should_stop:
            env_metrics = compute_evaluate_metrics(env_metrics_list) if env_metrics_list else {}
            env_metrics = _avg_env_metrics(env_metrics)
            success_msg = ""
            if len(env_metrics) > 0:
                parts = []
                for key in ["success_once", "success_at_end", "episode_reward"]:
                    if key in env_metrics:
                        parts.append(f"{key}={env_metrics[key]:.4f}")
                if parts:
                    success_msg = " | " + " | ".join(parts)
            elapsed = time.time() - start_time
            print(
                f"[collect] step={step} | elapsed={elapsed:.1f}s | all.traj={collected_traj} | all.samples={collected_samples}"
                f" | success.traj={int(stats.get('success', {}).get('num_trajectories', 0))}"
                f" | failure.traj={int(stats.get('failure', {}).get('num_trajectories', 0))}"
                f"{success_msg}"
            )
            last_log_time = time.time()

        if step % int(cfg.runner.weight_sync_interval) == 0:
            sync_weights()

        if should_stop:
            break

    final_stats = _aggregate_collection_stats(
        actor_group.finalize_offline_collection().wait()
    )
    print("[collect] finalized offline collection:")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
