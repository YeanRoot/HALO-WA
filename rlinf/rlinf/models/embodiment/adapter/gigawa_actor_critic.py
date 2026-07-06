from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if (not is_last) or activate_final:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinQHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.q1 = MLP(input_dim, [2048, 1024, 512], 1)
        self.q2 = MLP(input_dim, [2048, 1024, 512], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = self.token_norm(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        x = tokens + attn_out
        return x + self.ffn(self.ffn_norm(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        visual_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if visual_tokens is None or visual_tokens.numel() == 0:
            x = query_tokens
        else:
            q = self.query_norm(query_tokens)
            kv = self.kv_norm(visual_tokens)
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            x = query_tokens + attn_out
        return x + self.ffn(self.ffn_norm(x))


class CrossAttentionActor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        robot_state_dim: int,
        action_dim: int,
        action_chunk: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.state_proj = nn.Linear(robot_state_dim, hidden_dim)
        self.ref_action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_query_embed = nn.Parameter(torch.zeros(1, action_chunk, hidden_dim))
        nn.init.normal_(self.action_query_embed, mean=0.0, std=0.02)
        self.self_attn = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.output_head = MLP(hidden_dim, [1024, 512], action_dim)

    def forward(
        self,
        visual_tokens: Optional[torch.Tensor],
        robot_state: Optional[torch.Tensor],
        ref_action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, device, dtype = _infer_batch_device_dtype(
            visual_tokens, robot_state, ref_action
        )
        action_tokens = self.action_query_embed.expand(batch_size, -1, -1).to(
            device=device, dtype=dtype
        )
        if ref_action is not None:
            action_tokens = action_tokens + self.ref_action_proj(ref_action)

        query_tokens = []
        if robot_state is not None:
            query_tokens.append(self.state_proj(robot_state).unsqueeze(1))
        query_tokens.append(action_tokens)
        query = torch.cat(query_tokens, dim=1)

        query = self.self_attn(query)
        fused_tokens = self.cross_attn(query, visual_tokens)
        action_token_start = fused_tokens.shape[1] - self.action_chunk
        fused_action_tokens = fused_tokens[:, action_token_start:]
        action = self.output_head(fused_action_tokens)
        fused_state = fused_tokens.mean(dim=1)
        return action, fused_state


class CrossAttentionCritic(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        robot_state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.state_proj = nn.Linear(robot_state_dim, hidden_dim)
        self.ref_action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.value_head = TwinQHead(input_dim=hidden_dim)

    def forward(
        self,
        visual_tokens: Optional[torch.Tensor],
        robot_state: Optional[torch.Tensor],
        ref_action: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_tokens = []
        if robot_state is not None:
            query_tokens.append(self.state_proj(robot_state).unsqueeze(1))
        if ref_action is not None:
            query_tokens.append(self.ref_action_proj(ref_action))
        query_tokens.append(self.action_proj(action))
        query = torch.cat(query_tokens, dim=1)
        fused_tokens = self.cross_attn(query, visual_tokens)
        fused_state = fused_tokens.mean(dim=1)
        q1, q2 = self.value_head(fused_state)
        return q1, q2, fused_state


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def init_actor_output_small(actor: nn.Module) -> None:
    last_linear = None
    for module in actor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        nn.init.uniform_(last_linear.weight, -1e-3, 1e-3)
        nn.init.uniform_(last_linear.bias, -1e-3, 1e-3)


def _infer_batch_device_dtype(
    *tensors: Optional[torch.Tensor],
) -> tuple[int, torch.device, torch.dtype]:
    for tensor in tensors:
        if tensor is not None:
            return tensor.shape[0], tensor.device, tensor.dtype
    raise RuntimeError("Adapter received no inputs.")
