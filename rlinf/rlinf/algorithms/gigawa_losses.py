from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CriticLossConfig:
    discount: float
    q_upper_bound: float = 1.0
    clamp_target_to_upper_bound: bool = True
    enable_q_upper_bound: bool = True
    overshoot_penalty_coef: float = 1.0


@dataclass(frozen=True)
class ActorLossConfig:
    bc_coef: float
    q_upper_bound: float = 1.0
    clamp_q_to_upper_bound: bool = True
    enable_q_upper_bound: bool = True


def compute_bc_loss(
    pred_action: torch.Tensor,
    ref_action: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    diff = (pred_action.float() - ref_action.float()) ** 2
    if valid_mask is not None:
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        valid_mask = valid_mask.to(device=diff.device, dtype=diff.dtype)
        diff = diff * valid_mask

        if reduction == "mean":
            denom = valid_mask.sum() * diff.shape[-1]
            return diff.sum() / denom.clamp_min(1.0)
        if reduction == "sum":
            return diff.sum()
        if reduction == "none":
            return diff
        raise ValueError(f"Unknown reduction: {reduction}")

    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    raise ValueError(f"Unknown reduction: {reduction}")


def compute_td3_critic_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    rewards: torch.Tensor,
    done_mask: torch.Tensor,
    target_q: torch.Tensor,
    cfg: CriticLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if cfg.enable_q_upper_bound and cfg.clamp_target_to_upper_bound:
        target_q = torch.clamp(target_q, max=cfg.q_upper_bound)

    target_q_values = rewards + (1.0 - done_mask) * cfg.discount * target_q
    if cfg.enable_q_upper_bound and cfg.clamp_target_to_upper_bound:
        target_q_values = torch.clamp(target_q_values, max=cfg.q_upper_bound)
    target_q_values = target_q_values.to(dtype=q1.dtype)

    critic_loss = F.mse_loss(q1, target_q_values) + F.mse_loss(q2, target_q_values)
    q1_overshoot = torch.clamp(q1 - cfg.q_upper_bound, min=0.0)
    q2_overshoot = torch.clamp(q2 - cfg.q_upper_bound, min=0.0)
    overshoot_penalty = q1.new_zeros(())
    if cfg.enable_q_upper_bound and cfg.overshoot_penalty_coef > 0.0:
        overshoot_penalty = cfg.overshoot_penalty_coef * (
            (q1_overshoot ** 2).mean() + (q2_overshoot ** 2).mean()
        )
        critic_loss = critic_loss + overshoot_penalty

    metrics = {
        "critic_overshoot_penalty": float(overshoot_penalty.detach().item()),
        "q1_overshoot": float(q1_overshoot.mean().detach().item()),
        "q2_overshoot": float(q2_overshoot.mean().detach().item()),
        "q_upper_bound": float(cfg.q_upper_bound),
    }
    return critic_loss, target_q_values, metrics


def compute_bc_regularized_actor_loss(
    q_pi: Optional[torch.Tensor],
    bc_loss: torch.Tensor,
    cfg: ActorLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_term = bc_loss.new_zeros(())
    q_pi_for_loss = None

    if q_pi is not None:
        q_pi_for_loss = q_pi
        if cfg.enable_q_upper_bound and cfg.clamp_q_to_upper_bound:
            q_pi_for_loss = torch.clamp(q_pi_for_loss, max=cfg.q_upper_bound)
        q_term = (-q_pi_for_loss).mean()

    actor_loss = q_term + float(cfg.bc_coef) * bc_loss
    metrics = {
        "bc_coef_effective": float(cfg.bc_coef),
        "q_upper_bound_enabled": float(cfg.enable_q_upper_bound),
        "q_upper_bound": float(cfg.q_upper_bound),
        "q_loss_clamped": float(
            cfg.enable_q_upper_bound and cfg.clamp_q_to_upper_bound and q_pi is not None
        ),
        "q_pi_used_for_loss": (
            float(q_pi_for_loss.mean().detach().item())
            if q_pi_for_loss is not None
            else 0.0
        ),
        "q_weight": 1.0,
    }
    return actor_loss, metrics
