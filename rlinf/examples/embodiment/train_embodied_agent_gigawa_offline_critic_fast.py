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




def _fmt_metric(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.8f}"
    except Exception:
        return str(value)
@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="offline_critic_pretrain_mergeall_12chunk",
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
    actor_group.init_offline_critic_worker().wait()

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
    interval_cfg = cfg.algorithm.get("offline_critic_pretrain", {})
    val_interval = max(1, int(interval_cfg.get("val_interval", 5)))
    class_eval_interval = max(1, int(interval_cfg.get("class_eval_interval", val_interval)))
    tb_flush_interval = max(1, int(interval_cfg.get("tb_flush_interval", val_interval)))
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        do_validation = (epoch == 1) or (epoch % val_interval == 0) or (epoch == max_epochs)
        do_class_eval = (epoch == 1) or (epoch % class_eval_interval == 0) or (epoch == max_epochs)
        metrics = _pick_group_metrics(
            actor_group.run_offline_critic_epoch(
                do_validation=do_validation,
                do_class_eval=do_class_eval,
            ).wait()
        )
        global_step += 1
        elapsed = time.time() - start_time

        train_loss = metrics.get("offline_critic/train_critic_loss", None)
        val_loss = metrics.get("offline_critic/val_critic_loss", None)
        gap = metrics.get("offline_critic/overfit_gap", None)
        train_q = metrics.get("offline_critic/train_q_logged_mean", None)
        val_q = metrics.get("offline_critic/val_q_logged_mean", None)
        train_gap = metrics.get("offline_critic/train_success_failure_q_gap", None)
        val_gap = metrics.get("offline_critic/val_success_failure_q_gap", None)
        grad_norm = metrics.get("critic/grad_norm", None)
        lr = metrics.get("critic/lr", None)

        print(
            f"[offline_critic] epoch={epoch:04d} | step={global_step:06d} | elapsed={elapsed:.1f}s | "
            f"train_loss={_fmt_metric(train_loss)} | val_loss={_fmt_metric(val_loss)} | gap={_fmt_metric(gap)} | "
            f"train_q={_fmt_metric(train_q)} | val_q={_fmt_metric(val_q)} | "
            f"train_sf_gap={_fmt_metric(train_gap)} | val_sf_gap={_fmt_metric(val_gap)} | "
            f"grad_norm={float(grad_norm) if grad_norm is not None else float('nan'):.6f} | "
            f"lr={float(lr) if lr is not None else float('nan'):.8e} | "
            f"did_val={int(bool(metrics.get('offline_critic/did_validation', 0)))} | "
            f"did_class_eval={int(bool(metrics.get('offline_critic/did_class_eval', 0)))}"
        )
        if do_class_eval:
            print(
                "[offline_critic][success_failure] "
                f"train_success_q={_fmt_metric(metrics.get('offline_critic/train_success/q_logged_mean', None))} | "
                f"train_failure_q={_fmt_metric(metrics.get('offline_critic/train_failure/q_logged_mean', None))} | "
                f"val_success_q={_fmt_metric(metrics.get('offline_critic/val_success/q_logged_mean', None))} | "
                f"val_failure_q={_fmt_metric(metrics.get('offline_critic/val_failure/q_logged_mean', None))}"
            )

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, global_step)
        writer.add_scalar("offline_critic/epoch", epoch, global_step)
        if (epoch % tb_flush_interval == 0) or do_validation or do_class_eval or (epoch % save_interval == 0) or (epoch == max_epochs):
            writer.flush()

        if epoch % save_interval == 0 or epoch == max_epochs:
            _save_actor_checkpoint(actor_group, cfg, global_step)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
