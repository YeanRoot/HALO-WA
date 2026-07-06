import json
import os
import time

import hydra
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


def _pick_group_metrics(metrics):
    if isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, dict):
                return item
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _save_actor_checkpoint(actor_group, cfg, global_step: int):
    base_output_dir = os.path.join(
        cfg.runner.logger.log_path,
        cfg.runner.logger.experiment_name,
        f"checkpoints/global_step_{global_step}",
    )
    actor_save_path = os.path.join(base_output_dir, "actor")
    os.makedirs(actor_save_path, exist_ok=True)
    actor_group.save_checkpoint(actor_save_path, global_step).wait()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="offline_rl_pretrain_mergeall_12chunk",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.get("per_worker_log_path", None)
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")

    from rlinf.workers.actor.fsdp_gigawa_policy_worker import EmbodiedGigaWAFSDPPolicy

    actor_group = EmbodiedGigaWAFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    actor_group.init_offline_rl_worker().wait()

    resume_dir = cfg.runner.get("resume_dir", None)
    global_step = 0
    if resume_dir:
        actor_resume_path = (
            os.path.join(resume_dir, "actor")
            if os.path.isdir(os.path.join(resume_dir, "actor"))
            else resume_dir
        )
        actor_group.load_checkpoint(actor_resume_path).wait()
        if "global_step_" in resume_dir:
            try:
                global_step = int(resume_dir.split("global_step_")[-1].split("/")[0])
            except Exception:
                global_step = 0

    tb_dir = os.path.join(
        cfg.runner.logger.log_path,
        cfg.runner.logger.experiment_name,
        "tensorboard",
    )
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    max_epochs = int(cfg.runner.max_epochs)
    save_interval = int(cfg.runner.get("save_interval", 100))
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        metrics = _pick_group_metrics(actor_group.run_offline_rl_epoch().wait())
        global_step += 1
        elapsed = time.time() - start_time

        train_loss = metrics.get("offline_rl/train_critic_loss", None)
        val_loss = metrics.get("offline_rl/val/critic_loss", None)
        gap = metrics.get("offline_rl/critic_overfit_gap", None)
        train_q = metrics.get("offline_rl/train_q_logged_mean", None)
        val_q = metrics.get("offline_rl/val/q_logged_mean", None)
        train_gap = metrics.get("offline_rl/train_success_failure_q_gap", None)
        val_gap = metrics.get("offline_rl/val_success_failure_q_gap", None)
        grad_norm = metrics.get("offline_rl/critic_grad_norm", None)
        lr = metrics.get("critic/lr", None)

        actor_loss = metrics.get("offline_rl/train_actor_loss", None)
        bc_loss = metrics.get("offline_rl/train_bc_loss", None)
        q_pi = metrics.get("offline_rl/train_q_pi_mean", None)
        actor_grad_norm = metrics.get("offline_rl/actor_grad_norm", None)
        critic_grad_norm = metrics.get("offline_rl/critic_grad_norm", None)
        actor_lr = metrics.get("actor/lr", None)
        critic_lr = metrics.get("critic/lr", None)

        print(
            f"[offline_rl] epoch={epoch:04d} | step={global_step:06d} | elapsed={elapsed:.1f}s | "
            f"train_critic_loss={train_loss:.8f} | val_critic_loss={val_loss:.8f} | critic_gap={gap:.8f} | "
            f"train_q={train_q:.8f} | val_q={val_q:.8f} | "
            f"train_sf_gap={train_gap:.8f} | val_sf_gap={val_gap:.8f} | "
            f"actor_loss={actor_loss:.8f} | bc_loss={bc_loss:.8f} | q_pi={q_pi:.8f} | "
            f"critic_grad_norm={critic_grad_norm:.6f} | actor_grad_norm={actor_grad_norm:.6f} | "
            f"critic_lr={critic_lr:.8e} | actor_lr={actor_lr:.8e}"
        )
        print(
            "[offline_rl][success_failure] "
            f"train_success_q={metrics.get('offline_rl/train_success/q_logged_mean', 0.0):.8f} | "
            f"train_failure_q={metrics.get('offline_rl/train_failure/q_logged_mean', 0.0):.8f} | "
            f"val_success_q={metrics.get('offline_rl/val_success/q_logged_mean', 0.0):.8f} | "
            f"val_failure_q={metrics.get('offline_rl/val_failure/q_logged_mean', 0.0):.8f}"
        )

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, global_step)
        writer.add_scalar("offline_rl/epoch", epoch, global_step)
        writer.flush()

        if epoch % save_interval == 0 or epoch == max_epochs:
            _save_actor_checkpoint(actor_group, cfg, global_step)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
