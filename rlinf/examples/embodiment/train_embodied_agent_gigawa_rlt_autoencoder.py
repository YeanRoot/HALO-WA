import json
import os
import time

import hydra
import torch.multiprocessing as mp
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
    config_name="offline_rlt_autoencoder_pretrain",
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
    actor_group.init_offline_rlt_autoencoder_worker().wait()

    resume_dir = cfg.runner.get("resume_dir", None)
    global_step = 0
    if resume_dir:
        actor_resume_path = os.path.join(resume_dir, "actor") if os.path.isdir(os.path.join(resume_dir, "actor")) else resume_dir
        actor_group.load_checkpoint(actor_resume_path).wait()
        if "global_step_" in resume_dir:
            try:
                global_step = int(resume_dir.split("global_step_")[-1].split("/")[0])
            except Exception:
                global_step = 0

    max_epochs = int(cfg.runner.max_epochs)
    save_interval = int(cfg.runner.get("save_interval", 100))
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        metrics = _pick_group_metrics(actor_group.run_offline_rlt_autoencoder_epoch().wait())
        global_step += 1
        elapsed = time.time() - start_time
        train_loss = metrics.get("offline_rlt_autoencoder/train_recon_loss", 0.0)
        val_loss = metrics.get("offline_rlt_autoencoder/val_recon_loss", 0.0)
        train_rel = metrics.get("offline_rlt_autoencoder/train_relative_mse", 0.0)
        val_rel = metrics.get("offline_rlt_autoencoder/val_relative_mse", 0.0)
        gap = metrics.get("offline_rlt_autoencoder/overfit_gap", 0.0)
        grad_norm = metrics.get("actor/grad_norm", 0.0)
        lr = metrics.get("actor/lr", 0.0)
        print(
            f"[offline_rlt_autoencoder] epoch={epoch:04d} | step={global_step:06d} | elapsed={elapsed:.1f}s | "
            f"train_recon={train_loss:.8f} | val_recon={val_loss:.8f} | "
            f"train_rel={train_rel:.8f} | val_rel={val_rel:.8f} | gap={gap:.8f} | "
            f"grad_norm={grad_norm:.6f} | lr={lr:.8e}"
        )

        if epoch % save_interval == 0 or epoch == max_epochs:
            _save_actor_checkpoint(actor_group, cfg, global_step)


if __name__ == "__main__":
    main()
