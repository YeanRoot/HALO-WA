#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def _add_repo_to_path(script_path: Path) -> None:
    candidates = [
        script_path.parent,
        script_path.parent.parent,
        script_path.parent.parent.parent,
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / 'rlinf').is_dir():
            sys.path.insert(0, str(cand))
            return
    raise RuntimeError('Could not locate repo root containing rlinf/. Please place this script under the RLinf repo.')


def _resolve_full_weights_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates: list[Path] = []
    patterns = [
        'model_state_dict/full_weights.pt',
        'actor/model_state_dict/full_weights.pt',
        '**/model_state_dict/full_weights.pt',
    ]
    for pat in patterns:
        candidates.extend(path.glob(pat))
    candidates = sorted(set(p for p in candidates if p.is_file()))
    if not candidates:
        raise FileNotFoundError(
            f'Could not find full_weights.pt under checkpoint path: {path}. '
            'Please pass either the full_weights.pt file or a checkpoint dir containing model_state_dict/full_weights.pt.'
        )
    return candidates[0]


def _parse_model_default_from_defaults(defaults: list[Any]) -> str | None:
    for item in defaults:
        if isinstance(item, str):
            s = item.strip()
            if s.startswith('model/') and '@actor.model' in s:
                return s.split('model/', 1)[1].split('@actor.model', 1)[0]
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(k, str) and k.endswith('@actor.model'):
                    return str(v)
    return None


def _load_model_cfg_from_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if OmegaConf.select(cfg, 'actor.model.model_type', default=None) is not None:
        return cfg, cfg.actor.model

    defaults = cfg.get('defaults', [])
    model_default = _parse_model_default_from_defaults(defaults)
    if model_default is None:
        raise KeyError(
            f'Config {config_path} does not contain actor.model.model_type and no model/*@actor.model default could be resolved.'
        )

    config_dir = config_path.parent
    model_default_path = config_dir / 'model' / f'{model_default}.yaml'
    if not model_default_path.is_file():
        raise FileNotFoundError(
            f'Could not resolve model default file: {model_default_path}. '
            'Please pass a fully composed yaml or place the script under examples/embodiment with the config tree intact.'
        )

    base_model_cfg = OmegaConf.load(model_default_path)
    merged_model_cfg = OmegaConf.merge(base_model_cfg, cfg.actor.model)
    cfg.actor.model = merged_model_cfg
    return cfg, cfg.actor.model


def _reshape_action_tensor(action: torch.Tensor, action_chunk: int, action_dim: int) -> torch.Tensor:
    if action.ndim == 3 and action.shape[-2:] == (action_chunk, action_dim):
        return action
    if action.ndim == 2 and action.shape[-1] == action_chunk * action_dim:
        return action.view(action.shape[0], action_chunk, action_dim)
    if action.ndim == 1 and action.shape[0] == action_chunk * action_dim:
        return action.view(1, action_chunk, action_dim)
    raise ValueError(f'Unsupported action shape {tuple(action.shape)} for action_chunk={action_chunk}, action_dim={action_dim}')


def _squeeze_batch_dim(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _squeeze_batch_dim(v)
        elif torch.is_tensor(v):
            if v.ndim >= 2 and v.shape[1] == 1:
                out[k] = v[:, 0].contiguous()
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _load_pt_trajectory(pt_path: Path) -> dict[str, Any]:
    data = torch.load(pt_path, map_location='cpu')
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict trajectory file, got {type(data)}')
    return _squeeze_batch_dim(data)


def _build_step_mc_returns_at_chunk_start(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    step_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Monte-Carlo Q targets with primitive-step discounting.

    The old analysis script first summed rewards inside each chunk and then applied
    gamma once between chunks:

        Q_chunk[t] = sum_i r[t, i] + gamma * Q_chunk[t + 1]

    This function instead flattens all primitive steps and applies gamma at every
    environment step:

        G[k] = r[k] + (1 - done[k]) * step_gamma * G[k + 1]

    The returned chunk-level target is G at the first primitive step of each
    chunk, so it matches a critic that evaluates one chunk action from the chunk
    start. If termination happens inside a chunk, the recurrence cuts off at
    that primitive step.
    """
    if rewards.shape != terminations.shape:
        raise ValueError(
            f'rewards and terminations must have the same shape, got '
            f'{tuple(rewards.shape)} vs {tuple(terminations.shape)}'
        )
    if rewards.ndim == 1:
        rewards_2d = rewards[:, None]
        terminations_2d = terminations[:, None]
        squeeze_step_dim = True
    elif rewards.ndim == 2:
        rewards_2d = rewards
        terminations_2d = terminations
        squeeze_step_dim = False
    else:
        raise ValueError(f'Expected rewards shape [T] or [T, H], got {tuple(rewards.shape)}')

    flat_rewards = rewards_2d.float().reshape(-1)
    flat_done = terminations_2d.bool().reshape(-1)
    q_step_flat = torch.zeros_like(flat_rewards, dtype=torch.float32)
    running = torch.zeros((), dtype=torch.float32, device=flat_rewards.device)

    for step_idx in reversed(range(flat_rewards.numel())):
        done = flat_done[step_idx].float()
        running = flat_rewards[step_idx] + (1.0 - done) * float(step_gamma) * running
        q_step_flat[step_idx] = running

    q_step = q_step_flat.view_as(rewards_2d)
    q_chunk_start = q_step[:, 0].contiguous()
    if squeeze_step_dim:
        q_step = q_step[:, 0]
    return q_chunk_start, q_step


def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _save_curve_plot(xs: np.ndarray, ys: list[tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for label, y in ys:
        plt.plot(xs, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_chunk_matrix_figure(
    chunk_idx: int,
    true_model: np.ndarray,
    pred_model: np.ndarray,
    diff_model: np.ndarray,
    true_exec: np.ndarray,
    pred_exec: np.ndarray,
    diff_exec: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    mats = [
        (true_model, 'True action (model)'),
        (pred_model, 'Pred action (model)'),
        (diff_model, 'Pred-True (model)'),
        (true_exec, 'True action (exec)'),
        (pred_exec, 'Pred action (exec)'),
        (diff_exec, 'Pred-True (exec)'),
    ]
    for ax, (mat, ttl) in zip(axes.reshape(-1), mats):
        im = ax.imshow(mat, aspect='auto')
        ax.set_title(f'{ttl} | chunk={chunk_idx}')
        ax.set_xlabel('action dim')
        ax.set_ylabel('step in chunk')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _sanitize_name(s: str) -> str:
    safe = ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in s)
    return safe.strip('._') or 'trajectory'


def _trajectory_output_dir(pt_path: Path, root_out: Path, seen: dict[str, int]) -> Path:
    base = _sanitize_name(pt_path.stem)
    parent = _sanitize_name(pt_path.parent.name)
    rel = f'{parent}__{base}'
    count = seen.get(rel, 0)
    seen[rel] = count + 1
    if count > 0:
        rel = f'{rel}__{count}'
    out = root_out / rel
    out.mkdir(parents=True, exist_ok=True)
    (out / 'chunk_matrices').mkdir(exist_ok=True)
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != '']
    return [str(value)]


def _select_list(cfg, key: str) -> list[str]:
    value = OmegaConf.select(cfg, key, default=[])
    if value is None:
        return []
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return _as_list(value)


def _select_optional_str(cfg, key: str, default: str | None = None) -> str | None:
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        return None
    return str(value)


def _select_optional_int(cfg, key: str, default: int | None = None) -> int | None:
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        return None
    return int(value)


def _resolve_pt_paths(
    pt: list[str] | None = None,
    pt_glob: list[str] | None = None,
    input_dir: list[str] | None = None,
    pt_list_file: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for p in pt or []:
        paths.append(Path(p))
    for pat in pt_glob or []:
        paths.extend(sorted(Path().glob(pat)))
    for d in input_dir or []:
        dpath = Path(d)
        if not dpath.is_dir():
            raise FileNotFoundError(f'input_dir not found: {dpath}')
        paths.extend(sorted(dpath.rglob('*.pt')))
    if pt_list_file:
        with open(pt_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(Path(line))
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(Path(rp))
            seen.add(rp)
    if not uniq:
        raise ValueError(
            'No pt files resolved. Set analysis.pt, analysis.pt_glob, analysis.input_dir, '
            'or analysis.pt_list_file in the config, or pass --pt/--pt-glob/--input-dir/--pt-list-file.'
        )
    return uniq


def _resolve_discount_from_config(cfg, action_chunk: int) -> tuple[float, str, float, float]:
    input_gamma = OmegaConf.select(cfg, 'algorithm.gamma', default=None)
    if input_gamma is None:
        raise KeyError('Missing required config field: algorithm.gamma')
    input_gamma = float(input_gamma)
    if not (0.0 <= input_gamma <= 1.0):
        raise ValueError(f'algorithm.gamma should be in [0, 1], got {input_gamma}')

    gamma_mode = OmegaConf.select(cfg, 'algorithm.gamma_mode', default=None)
    if gamma_mode is None:
        raise KeyError(
            'Missing required config field: algorithm.gamma_mode. '
            'Use gamma_mode: chunk if algorithm.gamma is chunk-level, '
            'or gamma_mode: step if algorithm.gamma is primitive-step-level.'
        )
    gamma_mode = str(gamma_mode).lower().strip()
    if gamma_mode not in ('chunk', 'step'):
        raise ValueError(f'Unsupported algorithm.gamma_mode={gamma_mode!r}. Expected "chunk" or "step".')

    if gamma_mode == 'chunk':
        step_gamma = input_gamma ** (1.0 / float(action_chunk))
        effective_chunk_gamma = input_gamma
    else:
        step_gamma = input_gamma
        effective_chunk_gamma = input_gamma ** int(action_chunk)
    return input_gamma, gamma_mode, step_gamma, effective_chunk_gamma


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or str(device_str).lower() == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(str(device_str))


def _analyze_single_trajectory(
    policy,
    step_gamma: float,
    input_gamma: float,
    gamma_mode: str,
    effective_chunk_gamma: float,
    pt_path: Path,
    out_dir: Path,
    action_chunk: int,
    action_dim: int,
    chunk_index: int | None,
    max_chunk_matrix_plots: int,
    device: torch.device,
    config_path: Path,
    weights_path: Path,
    missing: list[str],
    unexpected: list[str],
) -> dict[str, Any]:
    traj = _load_pt_trajectory(pt_path)

    curr_visual_latent = traj['curr_obs']['visual_latent'].to(device)
    curr_robot_state = traj['curr_obs']['robot_state'].to(device)
    curr_ref_action = traj['curr_obs']['ref_action'].to(device)
    true_action = _reshape_action_tensor(traj['actions'].to(device), action_chunk, action_dim)
    rewards = traj['rewards'].to(device).float()
    terminations = traj['terminations'].to(device).bool()

    with torch.no_grad():
        visual_feat = policy.encode_visual(curr_visual_latent)
        pred_action, actor_aux = policy.actor_forward(
            visual_feat=visual_feat,
            robot_state=curr_robot_state,
            ref_action=curr_ref_action,
            ref_action_dropout_p=0.0,
            use_target=False,
        )
        rl_state = actor_aux['rl_state']
        q1_true, q2_true = policy.critic_forward(
            rl_state=rl_state,
            action=true_action,
            use_target=False,
            critic_visual_tokens=actor_aux.get('critic_visual_tokens', None),
            critic_robot_state=actor_aux.get('critic_robot_state', None),
            critic_ref_action=actor_aux.get('critic_ref_action', None),
        )
        q1_pred, q2_pred = policy.critic_forward(
            rl_state=rl_state,
            action=pred_action,
            use_target=False,
            critic_visual_tokens=actor_aux.get('critic_visual_tokens', None),
            critic_robot_state=actor_aux.get('critic_robot_state', None),
            critic_ref_action=actor_aux.get('critic_ref_action', None),
        )
        q_true_action = torch.minimum(q1_true, q2_true).squeeze(-1)
        q_pred_action = torch.minimum(q1_pred, q2_pred).squeeze(-1)
        true_exec = policy.postprocess_action_model_batch(true_action, curr_robot_state)
        pred_exec = policy.postprocess_action_model_batch(pred_action, curr_robot_state)

    q_mc_true, q_mc_step_all = _build_step_mc_returns_at_chunk_start(
        rewards=rewards,
        terminations=terminations,
        step_gamma=step_gamma,
    )
    q_mc_true = q_mc_true.to(device)
    q_mc_step_all = q_mc_step_all.to(device)

    model_diff = pred_action - true_action
    exec_diff = pred_exec - true_exec
    model_sqerr = model_diff.float().pow(2)
    exec_sqerr = exec_diff.float().pow(2)
    mse_per_chunk_model = model_sqerr.mean(dim=(1, 2))
    mse_per_chunk_exec = exec_sqerr.mean(dim=(1, 2))
    mse_per_step_model = model_sqerr.mean(dim=-1)
    mse_per_step_exec = exec_sqerr.mean(dim=-1)
    mse_per_dim_model = model_sqerr.mean(dim=1)
    mse_per_dim_exec = exec_sqerr.mean(dim=1)

    xs = np.arange(true_action.shape[0])
    _save_curve_plot(
        xs=xs,
        ys=[
            ('q_mc_true', _tensor_to_np(q_mc_true)),
            ('critic_q(true_action)', _tensor_to_np(q_true_action)),
            ('critic_q(pred_action)', _tensor_to_np(q_pred_action)),
        ],
        title='Q(s,a) curves by chunk (MC target uses primitive-step discount)',
        xlabel='chunk index',
        ylabel='Q value',
        out_path=out_dir / 'q_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('critic_q(true_action)-q_mc_true', _tensor_to_np(q_true_action - q_mc_true)),
            ('critic_q(pred_action)-q_mc_true', _tensor_to_np(q_pred_action - q_mc_true)),
        ],
        title='Q error vs step-discounted Monte Carlo ground truth',
        xlabel='chunk index',
        ylabel='Q error',
        out_path=out_dir / 'q_error_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('model_mse_per_chunk', _tensor_to_np(mse_per_chunk_model)),
            ('exec_mse_per_chunk', _tensor_to_np(mse_per_chunk_exec)),
        ],
        title='Action MSE by chunk',
        xlabel='chunk index',
        ylabel='MSE',
        out_path=out_dir / 'action_mse_curves.png',
    )
    _save_heatmap(_tensor_to_np(mse_per_step_model), 'Model-space MSE per step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_step_exec), 'Exec-space MSE per step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_exec.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_model), 'Model-space MSE per action dim', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_exec), 'Exec-space MSE per action dim', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_exec.png')

    if chunk_index is not None:
        chunk_indices = [int(chunk_index)]
    else:
        total = int(true_action.shape[0])
        limit = min(total, int(max_chunk_matrix_plots))
        if limit <= 0:
            chunk_indices = []
        elif limit == total:
            chunk_indices = list(range(total))
        else:
            chunk_indices = sorted(set(np.linspace(0, total - 1, limit, dtype=int).tolist()))

    for idx in chunk_indices:
        _save_chunk_matrix_figure(
            chunk_idx=idx,
            true_model=_tensor_to_np(true_action[idx]),
            pred_model=_tensor_to_np(pred_action[idx]),
            diff_model=_tensor_to_np(model_diff[idx]),
            true_exec=_tensor_to_np(true_exec[idx]),
            pred_exec=_tensor_to_np(pred_exec[idx]),
            diff_exec=_tensor_to_np(exec_diff[idx]),
            out_path=out_dir / 'chunk_matrices' / f'chunk_{idx:03d}.png',
        )

    summary = {
        'pt_path': str(pt_path.resolve()),
        'weights_path': str(weights_path.resolve()),
        'config_path': str(config_path.resolve()),
        'device': str(device),
        'input_gamma': input_gamma,
        'gamma_mode': gamma_mode,
        'step_gamma': step_gamma,
        'effective_chunk_gamma': effective_chunk_gamma,
        'num_chunks': int(true_action.shape[0]),
        'action_chunk': action_chunk,
        'action_dim': action_dim,
        'trajectory_metadata': traj.get('metadata', {}),
        'missing_state_dict_keys': list(missing),
        'unexpected_state_dict_keys': list(unexpected),
        'global_metrics': {
            'model_mse_mean': float(model_sqerr.mean().item()),
            'exec_mse_mean': float(exec_sqerr.mean().item()),
            'q_true_action_abs_err_mean': float((q_true_action - q_mc_true).abs().mean().item()),
            'q_pred_action_abs_err_mean': float((q_pred_action - q_mc_true).abs().mean().item()),
            'q_true_action_mean': float(q_true_action.mean().item()),
            'q_pred_action_mean': float(q_pred_action.mean().item()),
            'q_mc_true_mean': float(q_mc_true.mean().item()),
        },
        'per_chunk': [],
    }
    reward_chunk = rewards.sum(dim=-1)
    done_chunk = terminations.any(dim=-1)
    for i in range(true_action.shape[0]):
        summary['per_chunk'].append({
            'chunk_index': int(i),
            'reward_sum': float(reward_chunk[i].item()),
            'done_chunk': bool(done_chunk[i].item()),
            'mc_q_true': float(q_mc_true[i].item()),
            'critic_q_true_action': float(q_true_action[i].item()),
            'critic_q_pred_action': float(q_pred_action[i].item()),
            'critic_q_true_action_err': float((q_true_action[i] - q_mc_true[i]).item()),
            'critic_q_pred_action_err': float((q_pred_action[i] - q_mc_true[i]).item()),
            'model_mse': float(mse_per_chunk_model[i].item()),
            'exec_mse': float(mse_per_chunk_exec[i].item()),
        })

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        out_dir / 'arrays.npz',
        true_action_model=_tensor_to_np(true_action),
        pred_action_model=_tensor_to_np(pred_action),
        diff_action_model=_tensor_to_np(model_diff),
        true_action_exec=_tensor_to_np(true_exec),
        pred_action_exec=_tensor_to_np(pred_exec),
        diff_action_exec=_tensor_to_np(exec_diff),
        q_mc_true=_tensor_to_np(q_mc_true),
        q_mc_step_all=_tensor_to_np(q_mc_step_all),
        q_true_action=_tensor_to_np(q_true_action),
        q_pred_action=_tensor_to_np(q_pred_action),
        mse_per_chunk_model=_tensor_to_np(mse_per_chunk_model),
        mse_per_chunk_exec=_tensor_to_np(mse_per_chunk_exec),
        mse_per_step_model=_tensor_to_np(mse_per_step_model),
        mse_per_step_exec=_tensor_to_np(mse_per_step_exec),
        mse_per_dim_model=_tensor_to_np(mse_per_dim_model),
        mse_per_dim_exec=_tensor_to_np(mse_per_dim_exec),
        reward_chunk=_tensor_to_np(reward_chunk),
        done_chunk=_tensor_to_np(done_chunk.float()),
    )

    lines = [
        f'pt_path: {pt_path.resolve()}',
        f'weights_path: {weights_path.resolve()}',
        f'config_path: {config_path.resolve()}',
        f'device: {device}',
        f'input_gamma: {input_gamma}',
        f'gamma_mode: {gamma_mode}',
        f'step_gamma: {step_gamma}',
        f'effective_chunk_gamma: {effective_chunk_gamma}',
        f'num_chunks: {true_action.shape[0]} | action_chunk: {action_chunk} | action_dim: {action_dim}',
        f'trajectory_metadata: {json.dumps(traj.get("metadata", {}), ensure_ascii=False)}',
        '',
        'Global metrics:',
    ]
    for k, v in summary['global_metrics'].items():
        lines.append(f'  {k}: {v:.8f}')
    lines.extend(['', 'Per-chunk summary:'])
    header = 'chunk | reward_sum | done | mc_q_true | critic_q(true) | critic_q(pred) | err_true | err_pred | model_mse | exec_mse'
    lines.append(header)
    lines.append('-' * len(header))
    for row in summary['per_chunk']:
        lines.append(
            f"{row['chunk_index']:5d} | {row['reward_sum']:10.4f} | {int(row['done_chunk']):4d} | "
            f"{row['mc_q_true']:9.6f} | {row['critic_q_true_action']:14.6f} | {row['critic_q_pred_action']:14.6f} | "
            f"{row['critic_q_true_action_err']:8.6f} | {row['critic_q_pred_action_err']:8.6f} | {row['model_mse']:9.6f} | {row['exec_mse']:9.6f}"
        )
    with open(out_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return {
        'name': out_dir.name,
        'pt_path': str(pt_path.resolve()),
        'num_chunks': int(true_action.shape[0]),
        **summary['global_metrics'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze one or more GigaWA trajectory pt files against actor/critic checkpoints.')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config. Discount is read from algorithm.gamma and algorithm.gamma_mode.')
    parser.add_argument('--pt', type=str, action='append', default=[], help='Optional runtime override: trajectory .pt file. Repeat for multiple files.')
    parser.add_argument('--pt-glob', type=str, action='append', default=[], help='Optional runtime override: glob pattern for .pt files. Repeatable.')
    parser.add_argument('--input-dir', type=str, action='append', default=[], help='Optional runtime override: directory to recursively search for .pt files. Repeatable.')
    parser.add_argument('--pt-list-file', type=str, default=None, help='Optional runtime override: text file with one .pt path per line.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional runtime override: full_weights.pt or checkpoint dir. Default reads analysis.checkpoint.')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional runtime override: output directory. Default reads analysis.output_dir.')
    parser.add_argument('--device', type=str, default=None, help='Optional runtime override: device. Default reads analysis.device or auto.')
    parser.add_argument('--chunk-index', type=int, default=None, help='Optional runtime override: chunk index to render detailed action matrices.')
    parser.add_argument('--max-chunk-matrix-plots', type=int, default=None, help='Optional runtime override: number of chunk matrix plots. Default reads analysis.max_chunk_matrix_plots or 8.')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    _add_repo_to_path(script_path)

    from rlinf.models import get_model

    config_path = Path(args.config)
    cfg, model_cfg = _load_model_cfg_from_config(config_path)

    action_chunk = int(model_cfg.num_action_chunks)
    action_dim = int(model_cfg.action_dim)
    input_gamma, gamma_mode, step_gamma, effective_chunk_gamma = _resolve_discount_from_config(cfg, action_chunk)

    checkpoint = args.checkpoint or _select_optional_str(cfg, 'analysis.checkpoint')
    if checkpoint is None:
        raise KeyError('Missing checkpoint. Set analysis.checkpoint in config or pass --checkpoint.')
    weights_path = _resolve_full_weights_path(checkpoint)

    output_dir = args.output_dir or _select_optional_str(cfg, 'analysis.output_dir')
    if output_dir is None:
        raise KeyError('Missing output_dir. Set analysis.output_dir in config or pass --output-dir.')
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pt = args.pt if args.pt else _select_list(cfg, 'analysis.pt')
    pt_glob = args.pt_glob if args.pt_glob else _select_list(cfg, 'analysis.pt_glob')
    input_dir = args.input_dir if args.input_dir else _select_list(cfg, 'analysis.input_dir')
    pt_list_file = args.pt_list_file or _select_optional_str(cfg, 'analysis.pt_list_file')
    pt_paths = _resolve_pt_paths(pt=pt, pt_glob=pt_glob, input_dir=input_dir, pt_list_file=pt_list_file)

    device = _resolve_device(args.device or _select_optional_str(cfg, 'analysis.device', default='auto'))
    chunk_index = args.chunk_index
    if chunk_index is None:
        chunk_index = _select_optional_int(cfg, 'analysis.chunk_index', default=None)
    max_chunk_matrix_plots = args.max_chunk_matrix_plots
    if max_chunk_matrix_plots is None:
        max_chunk_matrix_plots = _select_optional_int(cfg, 'analysis.max_chunk_matrix_plots', default=8)
    max_chunk_matrix_plots = int(max_chunk_matrix_plots)

    print('[analyze_gigawa_pt_qsa_batch] discount config:')
    print(f'  algorithm.gamma={input_gamma}')
    print(f'  algorithm.gamma_mode={gamma_mode}')
    print(f'  step_gamma={step_gamma}')
    print(f'  effective_chunk_gamma={effective_chunk_gamma}')

    policy = get_model(model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu')
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    policy = policy.to(device)
    policy.eval()

    seen: dict[str, int] = {}
    aggregate: list[dict[str, Any]] = []
    for idx, pt_path in enumerate(pt_paths, start=1):
        traj_out = _trajectory_output_dir(pt_path, out_root, seen)
        print(f'[analyze_gigawa_pt_qsa_batch] [{idx}/{len(pt_paths)}] analyzing {pt_path} -> {traj_out}')
        aggregate.append(
            _analyze_single_trajectory(
                policy=policy,
                step_gamma=step_gamma,
                input_gamma=input_gamma,
                gamma_mode=gamma_mode,
                effective_chunk_gamma=effective_chunk_gamma,
                pt_path=pt_path,
                out_dir=traj_out,
                action_chunk=action_chunk,
                action_dim=action_dim,
                chunk_index=chunk_index,
                max_chunk_matrix_plots=max_chunk_matrix_plots,
                device=device,
                config_path=config_path,
                weights_path=weights_path,
                missing=list(missing),
                unexpected=list(unexpected),
            )
        )

    with open(out_root / 'aggregate_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'num_trajectories': len(aggregate),
                'weights_path': str(weights_path.resolve()),
                'config_path': str(config_path.resolve()),
                'input_gamma': input_gamma,
                'gamma_mode': gamma_mode,
                'step_gamma': step_gamma,
                'effective_chunk_gamma': effective_chunk_gamma,
                'items': aggregate,
            },
            f,
            indent=2,
        )

    lines = [
        f'num_trajectories: {len(aggregate)}',
        f'weights_path: {weights_path.resolve()}',
        f'config_path: {config_path.resolve()}',
        f'input_gamma: {input_gamma}',
        f'gamma_mode: {gamma_mode}',
        f'step_gamma: {step_gamma}',
        f'effective_chunk_gamma: {effective_chunk_gamma}',
        '',
    ]
    header = 'name | num_chunks | model_mse_mean | exec_mse_mean | q_true_abs_err_mean | q_pred_abs_err_mean | pt_path'
    lines.append(header)
    lines.append('-' * len(header))
    for item in aggregate:
        lines.append(
            f"{item['name']} | {item['num_chunks']} | {item['model_mse_mean']:.8f} | {item['exec_mse_mean']:.8f} | "
            f"{item['q_true_action_abs_err_mean']:.8f} | {item['q_pred_action_abs_err_mean']:.8f} | {item['pt_path']}"
        )
    with open(out_root / 'aggregate_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[analyze_gigawa_pt_qsa_batch] analyzed {len(aggregate)} trajectories')
    print(f'[analyze_gigawa_pt_qsa_batch] outputs saved under: {out_root}')


if __name__ == '__main__':
    main()
