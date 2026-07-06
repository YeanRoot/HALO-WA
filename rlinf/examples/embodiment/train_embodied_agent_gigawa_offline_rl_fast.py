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




import math

def _fmt_metric(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.8f}"
    except Exception:
        return str(value)

def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _add_scalar_if_valid(writer, tag: str, value, step: int):
    value = _safe_float(value)
    if value is None:
        return
    if value != value:  # nan
        return
    if value == float("inf") or value == float("-inf"):
        return
    writer.add_scalar(tag, value, step)


def _log_key_offline_rl_metrics(writer, metrics: dict, epoch: int, global_step: int):
    train_critic_loss = _safe_float(metrics.get("offline_rl/train_critic_loss"))
    val_critic_loss = _safe_float(metrics.get("offline_rl/val/critic_loss"))
    critic_gap = _safe_float(metrics.get("offline_rl/critic_overfit_gap"))
    train_q = _safe_float(metrics.get("offline_rl/train_q_logged_mean"))
    val_q = _safe_float(metrics.get("offline_rl/val/q_logged_mean"))
    train_sf_gap = _safe_float(metrics.get("offline_rl/train_success_failure_q_gap"))
    val_sf_gap = _safe_float(metrics.get("offline_rl/val_success_failure_q_gap"))
    actor_loss = _safe_float(metrics.get("offline_rl/train_actor_loss"))
    bc_loss = _safe_float(metrics.get("offline_rl/train_bc_loss"))
    q_pi = _safe_float(metrics.get("offline_rl/train_q_pi_mean"))
    critic_grad_norm = _safe_float(metrics.get("offline_rl/critic_grad_norm"))
    actor_grad_norm = _safe_float(metrics.get("offline_rl/actor_grad_norm"))
    critic_lr = _safe_float(metrics.get("critic/lr"))
    actor_lr = _safe_float(metrics.get("actor/lr"))

    core_pairs = {
        "tb_core/critic_train_loss": train_critic_loss,
        "tb_core/critic_val_loss": val_critic_loss,
        "tb_core/critic_overfit_gap": critic_gap,
        "tb_core/actor_loss": actor_loss,
        "tb_core/bc_loss": bc_loss,
        "tb_core/q_pi": q_pi,
        "tb_core/train_q": train_q,
        "tb_core/val_q": val_q,
        "tb_core/train_success_failure_gap": train_sf_gap,
        "tb_core/val_success_failure_gap": val_sf_gap,
        "tb_core/critic_grad_norm": critic_grad_norm,
        "tb_core/actor_grad_norm": actor_grad_norm,
        "tb_core/critic_lr": critic_lr,
        "tb_core/actor_lr": actor_lr,
        "tb_core/epoch": float(epoch),
    }
    for tag, value in core_pairs.items():
        _add_scalar_if_valid(writer, tag, value, global_step)

    # 额外给你几组更好判断“崩没崩”的派生指标
    if critic_grad_norm is not None and critic_grad_norm > 0:
        _add_scalar_if_valid(writer, "tb_stability/log10_critic_grad_norm", math.log10(critic_grad_norm + 1e-12), global_step)
    if actor_grad_norm is not None and actor_grad_norm > 0:
        _add_scalar_if_valid(writer, "tb_stability/log10_actor_grad_norm", math.log10(actor_grad_norm + 1e-12), global_step)
    if bc_loss is not None and bc_loss > 0:
        _add_scalar_if_valid(writer, "tb_stability/log10_bc_loss", math.log10(bc_loss + 1e-12), global_step)
    if train_critic_loss is not None and train_critic_loss > 0:
        _add_scalar_if_valid(writer, "tb_stability/log10_critic_train_loss", math.log10(train_critic_loss + 1e-12), global_step)

    if q_pi is not None and bc_loss is not None:
        _add_scalar_if_valid(writer, "tb_ratio/abs_q_pi_over_bc", abs(q_pi) / max(abs(bc_loss), 1e-12), global_step)
    if actor_grad_norm is not None and critic_grad_norm is not None:
        _add_scalar_if_valid(writer, "tb_ratio/actor_grad_over_critic_grad", actor_grad_norm / max(critic_grad_norm, 1e-12), global_step)

    # 保留关键 success/failure Q 方便看判别能力是否还在
    sf_pairs = {
        "tb_sf/train_success_q": metrics.get("offline_rl/train_success/q_logged_mean"),
        "tb_sf/train_failure_q": metrics.get("offline_rl/train_failure/q_logged_mean"),
        "tb_sf/val_success_q": metrics.get("offline_rl/val_success/q_logged_mean"),
        "tb_sf/val_failure_q": metrics.get("offline_rl/val_failure/q_logged_mean"),
    }
    for tag, value in sf_pairs.items():
        _add_scalar_if_valid(writer, tag, value, global_step)

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
    interval_cfg = cfg.algorithm.get("offline_rl_pretrain", {})
    val_interval = max(1, int(interval_cfg.get("val_interval", 5)))
    class_eval_interval = max(1, int(interval_cfg.get("class_eval_interval", val_interval)))
    tb_flush_interval = max(1, int(interval_cfg.get("tb_flush_interval", val_interval)))
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        do_validation = (epoch == 1) or (epoch % val_interval == 0) or (epoch == max_epochs)
        do_class_eval = (epoch == 1) or (epoch % class_eval_interval == 0) or (epoch == max_epochs)
        metrics = _pick_group_metrics(
            actor_group.run_offline_rl_epoch(
                do_validation=do_validation,
                do_class_eval=do_class_eval,
            ).wait()
        )
        global_step += 1
        elapsed = time.time() - start_time

        train_loss = metrics.get("offline_rl/train_critic_loss", None)
        val_loss = metrics.get("offline_rl/val/critic_loss", None)
        gap = metrics.get("offline_rl/critic_overfit_gap", None)
        train_q = metrics.get("offline_rl/train_q_logged_mean", None)
        val_q = metrics.get("offline_rl/val/q_logged_mean", None)
        train_gap = metrics.get("offline_rl/train_success_failure_q_gap", None)
        val_gap = metrics.get("offline_rl/val_success_failure_q_gap", None)

        actor_loss = metrics.get("offline_rl/train_actor_loss", None)
        bc_loss = metrics.get("offline_rl/train_bc_loss", None)
        q_pi = metrics.get("offline_rl/train_q_pi_mean", None)
        actor_grad_norm = metrics.get("offline_rl/actor_grad_norm", None)
        critic_grad_norm = metrics.get("offline_rl/critic_grad_norm", None)
        actor_lr = metrics.get("actor/lr", None)
        critic_lr = metrics.get("critic/lr", None)

        print(
            f"[offline_rl] epoch={epoch:04d} | step={global_step:06d} | elapsed={elapsed:.1f}s | "
            f"train_critic_loss={_fmt_metric(train_loss)} | val_critic_loss={_fmt_metric(val_loss)} | critic_gap={_fmt_metric(gap)} | "
            f"train_q={_fmt_metric(train_q)} | val_q={_fmt_metric(val_q)} | "
            f"train_sf_gap={_fmt_metric(train_gap)} | val_sf_gap={_fmt_metric(val_gap)} | "
            f"actor_loss={_fmt_metric(actor_loss)} | bc_loss={_fmt_metric(bc_loss)} | q_pi={_fmt_metric(q_pi)} | "
            f"critic_grad_norm={float(critic_grad_norm) if critic_grad_norm is not None else float('nan'):.6f} | "
            f"actor_grad_norm={float(actor_grad_norm) if actor_grad_norm is not None else float('nan'):.6f} | "
            f"critic_lr={float(critic_lr) if critic_lr is not None else float('nan'):.8e} | "
            f"actor_lr={float(actor_lr) if actor_lr is not None else float('nan'):.8e} | "
            f"did_val={int(bool(metrics.get('offline_rl/did_validation', 0)))} | "
            f"did_class_eval={int(bool(metrics.get('offline_rl/did_class_eval', 0)))}"
        )
        if do_class_eval:
            print(
                "[offline_rl][success_failure] "
                f"train_success_q={_fmt_metric(metrics.get('offline_rl/train_success/q_logged_mean', None))} | "
                f"train_failure_q={_fmt_metric(metrics.get('offline_rl/train_failure/q_logged_mean', None))} | "
                f"val_success_q={_fmt_metric(metrics.get('offline_rl/val_success/q_logged_mean', None))} | "
                f"val_failure_q={_fmt_metric(metrics.get('offline_rl/val_failure/q_logged_mean', None))}"
            )

        _log_key_offline_rl_metrics(writer, metrics, epoch, global_step)
        if (epoch % tb_flush_interval == 0) or do_validation or do_class_eval or (epoch % save_interval == 0) or (epoch == max_epochs):
            writer.flush()

        if epoch % save_interval == 0 or epoch == max_epochs:
            _save_actor_checkpoint(actor_group, cfg, global_step)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
