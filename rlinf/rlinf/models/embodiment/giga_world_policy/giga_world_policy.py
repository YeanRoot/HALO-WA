# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import importlib.util
import json
import os
import sys
import types
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from rlinf.algorithms.gigawa_losses import compute_bc_loss
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.adapter.gigawa_actor_critic import (
    CrossAttentionActor,
    CrossAttentionCritic,
    init_actor_output_small,
    set_requires_grad,
)


def _load_module_from_file(
    module_name: str,
    file_path: str,
    package_dir: Optional[str] = None,
):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if package_dir is None:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            file_path,
            submodule_search_locations=[package_dir],
        )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_wa_paths(wa_root: str, diffusers_src: Optional[str] = None) -> None:
    wa_root = os.path.abspath(wa_root)
    if diffusers_src:
        diffusers_src = os.path.abspath(diffusers_src)
        if diffusers_src not in sys.path:
            sys.path.insert(0, diffusers_src)

    extra_paths = [
        os.path.join(wa_root, "giga-train", "projects", "diffusion", "world_action_model"),
        os.path.join(wa_root, "giga-train", "projects", "diffusion", "world_action_model", "scripts"),
        os.path.join(wa_root, "giga-models"),
        os.path.join(wa_root, "giga-datasets"),
        os.path.join(wa_root, "giga-train"),
    ]
    for path in extra_paths:
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)


def _make_dummy_sockets_module():
    sockets_mod = types.ModuleType("giga_models.sockets")

    class _DummyRobotInferenceServer:
        pass

    class _DummyRobotInferenceClient:
        pass

    sockets_mod.RobotInferenceServer = _DummyRobotInferenceServer
    sockets_mod.RobotInferenceClient = _DummyRobotInferenceClient
    return sockets_mod


def _preload_wa_runtime(wa_root: str):
    """Load only the inference-time WA modules without training-time imports."""
    wa_root = os.path.abspath(wa_root)
    world_dir = os.path.join(
        wa_root, "giga-train", "projects", "diffusion", "world_action_model"
    )
    wa_pkg_dir = os.path.join(world_dir, "wa")
    infer_file = os.path.join(world_dir, "scripts", "inference_openloop_action_only.py")
    transformer_file = os.path.join(wa_pkg_dir, "transformer_wa_casual.py")

    if not os.path.isdir(world_dir):
        raise FileNotFoundError(f"world_action_model dir not found under wa_root: {world_dir}")
    if not os.path.isfile(transformer_file):
        raise FileNotFoundError(f"transformer_wa_casual.py not found: {transformer_file}")
    if not os.path.isfile(infer_file):
        raise FileNotFoundError(
            f"inference_openloop_action_only.py not found: {infer_file}"
        )

    if "wa" not in sys.modules:
        wa_pkg = types.ModuleType("wa")
        wa_pkg.__path__ = [wa_pkg_dir]
        sys.modules["wa"] = wa_pkg

    transformer_mod = _load_module_from_file(
        "wa.transformer_wa_casual", transformer_file
    )

    if "giga_models" not in sys.modules:
        gm_pkg = types.ModuleType("giga_models")
        gm_pkg.__path__ = [os.path.join(wa_root, "giga-models", "giga_models")]
        sys.modules["giga_models"] = gm_pkg
    if "giga_models.sockets" not in sys.modules:
        sys.modules["giga_models.sockets"] = _make_dummy_sockets_module()

    infer_module_name = "wa_inference_openloop_action_only_min"
    if infer_module_name in sys.modules:
        infer_mod = sys.modules[infer_module_name]
    else:
        with open(infer_file, "r", encoding="utf-8") as f:
            infer_src = f.read()
        sentinel = "\nfrom giga_datasets import image_utils, video_utils"
        cut = infer_src.find(sentinel)
        if cut == -1:
            raise RuntimeError(
                "Could not find dataset-import sentinel in inference_openloop_action_only.py"
            )
        infer_src = infer_src[:cut]
        infer_mod = types.ModuleType(infer_module_name)
        infer_mod.__file__ = infer_file
        infer_mod.__package__ = ""
        sys.modules[infer_module_name] = infer_mod
        exec(compile(infer_src, infer_file, "exec"), infer_mod.__dict__)

    return transformer_mod.CasualWorldActionTransformer, infer_mod.WAPipeline



class GigaWorldPolicy(BasePolicy, nn.Module):
    """
    RLinf Giga World Action policy wrapper.

    Frozen WA backbone plus trainable cross-attention actor/critic heads.
    """

    def __init__(self, cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()

        policy_cfg = cfg.giga_world_policy
        transformer_ckpt = cfg.model_path
        _setup_wa_paths(
            wa_root=policy_cfg.wa_root,
            diffusers_src=policy_cfg.get("diffusers_src", None),
        )

        from diffusers.models import AutoencoderKLWan

        CasualWorldActionTransformer, WAPipeline = _preload_wa_runtime(
            policy_cfg.wa_root
        )

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        self.cfg = cfg
        self.dtype = torch_dtype
        self.action_chunk = int(cfg.num_action_chunks)
        self.wa_action_chunk = int(policy_cfg.get("wa_num_action_chunks", self.action_chunk))
        if self.wa_action_chunk < self.action_chunk:
            raise ValueError(
                f"wa_num_action_chunks ({self.wa_action_chunk}) must be >= actor/runtime num_action_chunks ({self.action_chunk})."
            )
        self.env_action_dim = int(cfg.action_dim)
        self.num_inference_steps = int(policy_cfg.num_inference_steps)
        self.guidance_scale = float(policy_cfg.guidance_scale)
        self.num_frames = int(policy_cfg.get("num_frames", 5))
        self.prompt_override = policy_cfg.get("prompt", None)
        self.robotype = str(policy_cfg.robotype)
        self.single_view_size = (
            int(policy_cfg.get("single_view_width", 256)),
            int(policy_cfg.get("single_view_height", 192)),
        )
        self.ref_image_size = (
            int(policy_cfg.get("ref_image_width", self.single_view_size[0] * 3)),
            int(policy_cfg.get("ref_image_height", self.single_view_size[1])),
        )
        # full_image_size is the image canvas passed to the WA pipeline.
        self.full_image_size = self.ref_image_size
        self.device_ref = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_model_dir = policy_cfg.base_model_dir
        norm_json = policy_cfg.norm_json

        # rollout switch: keep base WA by default until RL worker is ready.
        # This flag must live in the state_dict so actor->rollout weight sync can carry it.
        initial_rollout_flag = 1 if bool(policy_cfg.get("use_rl_head_for_rollout", False)) else 0
        self.register_buffer(
            "use_rl_head_for_rollout_flag",
            torch.tensor(initial_rollout_flag, dtype=torch.uint8),
            persistent=True,
        )

        self.visual_feature_dim = int(policy_cfg.get("visual_feature_dim", 2048))
        self.bc_coef = float(policy_cfg.get("bc_coef", 1.0))
        self.ref_action_dropout_p = float(policy_cfg.get("ref_action_dropout_p", 0.5))
        self.target_tau = float(policy_cfg.get("target_tau", 0.005))
        self.enable_absolute_action_bound = bool(
            policy_cfg.get("enable_absolute_action_bound", True)
        )

        self.cross_attention_dim = int(policy_cfg.get("cross_attention_dim", 512))
        self.cross_attention_heads = int(policy_cfg.get("cross_attention_heads", 8))
        self.cross_attention_dropout = float(policy_cfg.get("cross_attention_dropout", 0.0))
        self.cross_attention_time_reduce = str(
            policy_cfg.get("cross_attention_time_reduce", "mean")
        ).lower()
        if self.cross_attention_time_reduce not in {"mean", "first"}:
            raise ValueError(
                f"Unsupported cross_attention_time_reduce={self.cross_attention_time_reduce}."
            )
        if self.cross_attention_dim % self.cross_attention_heads != 0:
            raise ValueError(
                f"cross_attention_dim ({self.cross_attention_dim}) must be divisible by "
                f"cross_attention_heads ({self.cross_attention_heads})."
            )
        self.max_visual_tokens = int(policy_cfg.get("max_visual_tokens", 1024))

        vae = AutoencoderKLWan.from_pretrained(
            base_model_dir,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )
        transformer = CasualWorldActionTransformer.from_pretrained(
            transformer_ckpt,
            use_safetensors=bool(policy_cfg.get("use_safetensors", False)),
        ).to(torch_dtype)

        self.pipe = WAPipeline.from_pretrained(
            base_model_dir,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device_ref)
        self._freeze_pipe_modules()

        self.model_action_dim = int(self.pipe.transformer.action_encoder[0].in_features)
        self.vae_z_dim = int(self.pipe.vae.config.z_dim)

        with open(norm_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.stats = stats["norm_stats"] if "norm_stats" in stats else stats

        self.register_buffer(
            "state_mean",
            self._load_stat("observation.state", "mean", 0.0),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            self._load_stat("observation.state", "std", 1.0),
            persistent=False,
        )
        self.register_buffer(
            "delta_mean",
            self._load_stat("action", "mean", 0.0),
            persistent=False,
        )
        self.register_buffer(
            "delta_std",
            self._load_stat("action", "std", 1.0),
            persistent=False,
        )
        self.register_buffer(
            "action_q01_raw",
            self._load_stat("action", "q01", -1.0),
            persistent=False,
        )
        self.register_buffer(
            "action_q99_raw",
            self._load_stat("action", "q99", 1.0),
            persistent=False,
        )
        action_std_safe = torch.where(
            self.delta_std.abs() < 1e-8,
            torch.ones_like(self.delta_std),
            self.delta_std,
        )
        self.register_buffer(
            "action_q01",
            (self.action_q01_raw - self.delta_mean) / action_std_safe,
            persistent=False,
        )
        self.register_buffer(
            "action_q99",
            (self.action_q99_raw - self.delta_mean) / action_std_safe,
            persistent=False,
        )
        self.register_buffer(
            "action_bound_center",
            0.5 * (self.action_q01 + self.action_q99),
            persistent=False,
        )
        self.register_buffer(
            "action_bound_half_range",
            0.5 * (self.action_q99 - self.action_q01),
            persistent=False,
        )
        self.register_buffer(
            "delta_mask",
            self._build_delta_mask(self.robotype, self.model_action_dim),
            persistent=False,
        )

        self.ref_action_flat_dim = self.action_chunk * self.model_action_dim
        self.robot_state_dim = self.model_action_dim
        self.rl_state_dim = self.cross_attention_dim
        self.visual_compressor = nn.Sequential(
            nn.Linear(self.vae_z_dim, self.cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(self.cross_attention_dim),
        )
        self.visual_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_visual_tokens, self.cross_attention_dim)
        )
        nn.init.normal_(self.visual_pos_embed, mean=0.0, std=0.02)
        self.actor_head = CrossAttentionActor(
            hidden_dim=self.cross_attention_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.cross_attention_dropout,
            robot_state_dim=self.robot_state_dim,
            action_dim=self.model_action_dim,
            action_chunk=self.action_chunk,
        )
        self.critic = CrossAttentionCritic(
            hidden_dim=self.cross_attention_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.cross_attention_dropout,
            robot_state_dim=self.robot_state_dim,
            action_dim=self.model_action_dim,
        )
        self.actor_target = copy.deepcopy(self.actor_head)
        self.critic_target = copy.deepcopy(self.critic)
        self._set_requires_grad(self.actor_target, False)
        self._set_requires_grad(self.critic_target, False)

        # small init for actor output to avoid too wild first actions
        self._init_actor_output_small()

    def _freeze_pipe_modules(self):
        """
        Freeze all trainable WA backbone submodules safely.
        self.pipe is a DiffusionPipeline container, not an nn.Module.
        We must freeze its registered nn.Module children individually.
        """
        module_names = [
            "vae",
            "text_encoder",
            "image_encoder",
            "transformer",
            "transformer_2",
        ]
        for name in module_names:
            module = getattr(self.pipe, name, None)
            if module is None:
                continue
            if hasattr(module, "eval"):
                module.eval()
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = False

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        set_requires_grad(module, requires_grad)

    def set_visual_trainable(self, trainable: bool) -> None:
        self._set_requires_grad(self.visual_compressor, trainable)
        if hasattr(self, "visual_pos_embed"):
            self.visual_pos_embed.requires_grad = trainable

    def set_actor_head_trainable(self, trainable: bool) -> None:
        self._set_requires_grad(self.actor_head, trainable)

    def set_critic_trainable(self, trainable: bool) -> None:
        self._set_requires_grad(self.critic, trainable)

    def _init_actor_output_small(self):
        init_actor_output_small(self.actor_head)

    def _load_stat(self, key1: str, key2: str, pad_value: float) -> torch.Tensor:
        x = torch.as_tensor(self.stats[key1][key2], dtype=torch.float32)
        if x.numel() >= self.model_action_dim:
            x = x[: self.model_action_dim]
        else:
            pad = torch.full(
                (self.model_action_dim - x.numel(),),
                float(pad_value),
                dtype=torch.float32,
            )
            x = torch.cat([x, pad], dim=0)
        return x

    @staticmethod
    def _robotype_to_embed_id(robotype: str) -> int:
        name = robotype.lower()
        if "agibot" in name:
            return 1
        return 0

    def _build_delta_mask(self, robotype: str, dim: int) -> torch.Tensor:
        embed_id = self._robotype_to_embed_id(robotype)
        templates = {
            0: np.array(
                [True, True, True, True, True, True, False,
                 True, True, True, True, True, True, False],
                dtype=bool,
            ),
            1: np.array(
                [True, True, True, True, True, True, True, False,
                 True, True, True, True, True, True, True, False],
                dtype=bool,
            ),
        }
        base = templates[embed_id]
        if dim > len(base):
            base = np.pad(base, (0, dim - len(base)), constant_values=False)
        else:
            base = base[:dim]
        return torch.as_tensor(base, dtype=torch.bool)

    @staticmethod
    def _to_pil(img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().float()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.numpy()
        img = np.asarray(img)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def _resize_center_crop(img: Image.Image, dst_w: int, dst_h: int) -> Image.Image:
        w, h = img.size
        if float(dst_h) / h < float(dst_w) / w:
            new_h = int(round(float(dst_w) / w * h))
            new_w = dst_w
        else:
            new_h = dst_h
            new_w = int(round(float(dst_h) / h * w))
        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        img = TF.center_crop(img, (dst_h, dst_w))
        return img

    def _blank_view(self) -> Image.Image:
        return Image.fromarray(
            np.zeros(
                (self.single_view_size[1], self.single_view_size[0], 3),
                dtype=np.uint8,
            )
        )

    def _extract_views(self, env_obs: dict[str, Any], index: int):
        main_image = env_obs["main_images"][index]
        wrist_images = env_obs.get("wrist_images", None)

        img_high = self._resize_center_crop(
            self._to_pil(main_image),
            self.single_view_size[0],
            self.single_view_size[1],
        )

        img_left = self._blank_view()
        img_right = self._blank_view()
        if wrist_images is not None:
            sample_wrist = wrist_images[index]
            n_views = int(sample_wrist.shape[0]) if hasattr(sample_wrist, "shape") else len(sample_wrist)
            if n_views >= 1:
                img_left = self._resize_center_crop(
                    self._to_pil(sample_wrist[0]),
                    self.single_view_size[0],
                    self.single_view_size[1],
                )
            if n_views >= 2:
                img_right = self._resize_center_crop(
                    self._to_pil(sample_wrist[1]),
                    self.single_view_size[0],
                    self.single_view_size[1],
                )
        return img_high, img_left, img_right

    def _build_ref_image(self, env_obs: dict[str, Any], index: int) -> Image.Image:
        img_high, img_left, img_right = self._extract_views(env_obs, index)
        cat = np.concatenate(
            [np.asarray(img_high), np.asarray(img_left), np.asarray(img_right)],
            axis=1,
        )
        ref_image = Image.fromarray(cat)

        if ref_image.size != self.full_image_size:
            ref_image = TF.resize(
                ref_image,
                (self.full_image_size[1], self.full_image_size[0]),
                interpolation=InterpolationMode.BILINEAR,
            )
        return ref_image

    def _normalize_state(self, state_raw: torch.Tensor):
        state_raw = torch.as_tensor(state_raw, dtype=torch.float32).flatten()
        if state_raw.numel() >= self.model_action_dim:
            state = state_raw[: self.model_action_dim]
        else:
            pad = torch.zeros(self.model_action_dim - state_raw.numel(), dtype=torch.float32)
            state = torch.cat([state_raw, pad], dim=0)
        state = state.to(self.device_ref)
        eps = 1e-8
        norm_state = (state - self.state_mean) / self.state_std.clamp_min(eps)
        return norm_state.unsqueeze(0), state

    def _select_prompt(self, env_obs: dict[str, Any], index: int) -> str:
        if self.prompt_override:
            return str(self.prompt_override)
        task_descriptions = env_obs.get("task_descriptions", None)
        if task_descriptions is None:
            return ""
        return str(task_descriptions[index])

    @torch.no_grad()
    def _run_pipe(
        self,
        ref_image: Image.Image,
        norm_state: torch.Tensor,
        prompt: str,
    ) -> torch.Tensor:
        _, pred_delta_norm = self.pipe(
            height=self.full_image_size[1],
            width=self.full_image_size[0],
            action_chunk=self.wa_action_chunk,
            state=norm_state,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            image=ref_image,
            prompt=prompt,
            return_dict=False,
        )
        return pred_delta_norm

    @torch.no_grad()
    def _extract_visual_latent_from_ref(
        self,
        ref_image: Image.Image,
        norm_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get on-device visual latent from the frozen WA VAE path without using
        the debug dict, so this is suitable for training.
        Returns latent_condition with shape [1, z_dim, T_lat, H_lat, W_lat].
        """
        image = self.pipe.video_processor.preprocess(
            ref_image,
            height=self.full_image_size[1],
            width=self.full_image_size[0],
        ).to(self.device_ref, dtype=torch.float32)

        prepare_kwargs = dict(
            image=image,
            batch_size=1,
            num_channels_latents=self.pipe.vae.config.z_dim,
            height=self.full_image_size[1],
            width=self.full_image_size[0],
            num_frames=self.num_frames,
            dtype=torch.float32,
            device=self.device_ref,
            generator=None,
            latents=None,
            last_image=None,
            action_chunk=self.wa_action_chunk,
            action_dim=self.model_action_dim,
        )
        try:
            latents_outputs = self.pipe.prepare_latents(
                **prepare_kwargs,
                return_latent_debug=False,
            )
        except TypeError as e:
            if "return_latent_debug" not in str(e):
                raise
            latents_outputs = self.pipe.prepare_latents(**prepare_kwargs)

        if self.pipe.config.expand_timesteps:
            # returns: latents, latent_condition, first_frame_mask, action
            _, latent_condition, _, _ = latents_outputs
        else:
            # returns: latents, packed_condition, action
            _, packed_condition, _ = latents_outputs
            latent_condition = packed_condition[:, -self.vae_z_dim:]

        return latent_condition.float()

    def _postprocess_pred_delta(
        self,
        pred_delta_norm: torch.Tensor,
        state_pad: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        pred_delta = pred_delta_norm * self.delta_std.clamp_min(eps) + self.delta_mean
        pred_action = pred_delta.clone()
        pred_action[:, :, self.delta_mask] += state_pad[self.delta_mask]
        return pred_action[:, :, : self.env_action_dim].float()

    @torch.no_grad()
    def _extract_frozen_backbone_single(
        self,
        env_obs: dict[str, Any],
        index: int,
    ) -> dict[str, torch.Tensor]:
        ref_image = self._build_ref_image(env_obs, index)
        norm_state, state_pad = self._normalize_state(env_obs["states"][index])
        prompt = self._select_prompt(env_obs, index)

        pred_delta_norm = self._run_pipe(
            ref_image=ref_image,
            norm_state=norm_state,
            prompt=prompt,
        )

        if pred_delta_norm.shape[1] < self.action_chunk:
            raise RuntimeError(
                f"WA planner returned only {pred_delta_norm.shape[1]} steps, fewer than actor/runtime num_action_chunks={self.action_chunk}."
            )

        pred_delta_norm = pred_delta_norm[:, : self.action_chunk]

        # `ref_action_model` stays in the WA model space (normalized delta space).
        # The executed action is obtained by applying the same WA post-process used
        # by the base policy.
        ref_action_model = pred_delta_norm[0].float()  # [C, A_model]
        ref_action_exec = self._postprocess_pred_delta(pred_delta_norm, state_pad)[0]  # [C, A_env]
        visual_latent = self._extract_visual_latent_from_ref(
            ref_image=ref_image,
            norm_state=norm_state,
        )[0]  # [Z, T, H, W]

        return {
            "visual_latent": visual_latent,             # [Z, T, H, W]
            "robot_state": state_pad.float(),           # [state_dim]
            "ref_action": ref_action_model,             # [C, A_model]
            "ref_action_exec": ref_action_exec,         # [C, A_env]
        }

    @torch.no_grad()
    def extract_frozen_backbone_batch(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Frozen backbone extraction for a batch of RobotWin observations.

        Returns:
            visual_latent:   [B, Z, T, H, W]
            robot_state:     [B, state_dim]
            ref_action:      [B, C, A_model]  (model-space action)
            ref_action_exec: [B, C, A_env]    (post-processed executable action)
        """
        batch_size = int(env_obs["states"].shape[0])
        outs = [self._extract_frozen_backbone_single(env_obs, i) for i in range(batch_size)]
        visual_latent = torch.stack([o["visual_latent"] for o in outs], dim=0).to(self.device_ref)
        robot_state = torch.stack([o["robot_state"] for o in outs], dim=0).to(self.device_ref)
        ref_action = torch.stack([o["ref_action"] for o in outs], dim=0).to(self.device_ref)
        ref_action_exec = torch.stack([o["ref_action_exec"] for o in outs], dim=0).to(self.device_ref)
        return {
            "visual_latent": visual_latent,
            "robot_state": robot_state,
            "ref_action": ref_action,
            "ref_action_exec": ref_action_exec,
        }

    def encode_visual(self, visual_latent: torch.Tensor) -> torch.Tensor:
        if visual_latent.ndim != 5:
            raise ValueError(
                f"Expected visual_latent [B,Z,T,H,W], got {tuple(visual_latent.shape)}"
            )

        comp_param = next(self.visual_compressor.parameters())
        comp_device = comp_param.device
        comp_dtype = comp_param.dtype
        x = visual_latent.to(device=comp_device, dtype=comp_dtype)
        if x.shape[2] == 1 or self.cross_attention_time_reduce == "first":
            x = x[:, :, 0]
        else:
            x = x.mean(dim=2)

        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.vae_z_dim)
        if x.shape[1] > self.max_visual_tokens:
            raise RuntimeError(
                f"visual token count {x.shape[1]} exceeds max_visual_tokens={self.max_visual_tokens}."
            )
        x = self.visual_compressor(x)
        pos = self.visual_pos_embed[:, : x.shape[1]].to(device=comp_device, dtype=comp_dtype)
        return x + pos

    def _apply_ref_action_dropout(
        self,
        ref_action: torch.Tensor,
        p: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Batch-wise dropout on reference action conditioning.
        If dropped, the whole reference chunk for that sample is set to zero.
        """
        if (not self.training) or p <= 0.0:
            return ref_action, None
        batch_size = ref_action.shape[0]
        keep = (
            torch.rand(batch_size, 1, 1, device=ref_action.device) > p
        ).to(dtype=ref_action.dtype)
        dropped = ref_action * keep
        return dropped, keep

    def _bound_absolute_action_model(
        self,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bound absolute model-space actions to the empirical q01/q99 range.

        Important: q01/q99 are first converted from raw action space into the
        normalized model space used by the actor, then applied here. This keeps
        the actor output in the same support as the offline action dataset while
        preserving the "absolute action" parameterization.
        """
        center = self.action_bound_center.to(device=raw_action.device, dtype=raw_action.dtype)
        half_range = self.action_bound_half_range.to(device=raw_action.device, dtype=raw_action.dtype)
        low = self.action_q01.to(device=raw_action.device, dtype=raw_action.dtype)
        high = self.action_q99.to(device=raw_action.device, dtype=raw_action.dtype)

        bounded = center + half_range * torch.tanh(raw_action)
        bounded = torch.maximum(torch.minimum(bounded, high), low)
        return bounded

    def actor_forward(
        self,
        visual_feat: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
        ref_action_dropout_p: Optional[float] = None,
        use_target: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if ref_action_dropout_p is None:
            ref_action_dropout_p = self.ref_action_dropout_p

        cond_ref_action, dropout_mask = self._apply_ref_action_dropout(
            ref_action, p=ref_action_dropout_p
        )
        head = self.actor_target if use_target else self.actor_head
        actor_param = next(head.parameters())
        actor_device = actor_param.device
        actor_dtype = actor_param.dtype

        visual_tokens = visual_feat.to(device=actor_device, dtype=actor_dtype)
        visual_summary = visual_tokens.mean(dim=1)
        robot_state_for_state = robot_state.to(device=actor_device, dtype=actor_dtype)
        ref_action_for_state = cond_ref_action.to(device=actor_device, dtype=actor_dtype)
        learned_action, fused_state = head(
            visual_tokens=visual_tokens,
            robot_state=robot_state_for_state,
            ref_action=ref_action_for_state,
        )
        action = (
            self._bound_absolute_action_model(learned_action)
            if self.enable_absolute_action_bound
            else learned_action
        )
        aux = {
            "raw_action": learned_action,
            "rl_state": fused_state,
            "cond_ref_action": cond_ref_action,
            "visual_feat_for_state": visual_summary,
            "robot_state_for_state": robot_state_for_state,
            "ref_action_flat_for_state": ref_action_for_state.reshape(
                ref_action_for_state.shape[0], -1
            ),
            "critic_visual_tokens": visual_tokens,
            "critic_robot_state": robot_state_for_state,
            "critic_ref_action": ref_action_for_state,
        }
        if dropout_mask is not None:
            aux["ref_dropout_mask"] = dropout_mask
        return action, aux

    def critic_forward(
        self,
        rl_state: Optional[torch.Tensor],
        action: torch.Tensor,
        use_target: bool = False,
        critic_visual_tokens: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_ref_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic = self.critic_target if use_target else self.critic
        critic_param = next(critic.parameters())
        critic_device = critic_param.device
        critic_dtype = critic_param.dtype

        if critic_visual_tokens is None:
            raise RuntimeError(
                "critic_forward requires actor_aux['critic_visual_tokens'] from actor_forward."
            )
        action_for_critic = (
            self._bound_absolute_action_model(action)
            if self.enable_absolute_action_bound
            else action
        ).to(device=critic_device, dtype=critic_dtype)
        visual_tokens = critic_visual_tokens.to(device=critic_device, dtype=critic_dtype)
        robot_state = (
            critic_robot_state.to(device=critic_device, dtype=critic_dtype)
            if critic_robot_state is not None
            else None
        )
        ref_action = (
            critic_ref_action.to(device=critic_device, dtype=critic_dtype)
            if critic_ref_action is not None
            else None
        )
        q1, q2, _ = critic(
            visual_tokens=visual_tokens,
            robot_state=robot_state,
            ref_action=ref_action,
            action=action_for_critic,
        )
        return q1, q2

    def target_actor_forward(
        self,
        visual_feat: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.actor_forward(
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=ref_action,
            ref_action_dropout_p=0.0,
            use_target=True,
        )

    def target_critic_forward(
        self,
        rl_state: Optional[torch.Tensor],
        action: torch.Tensor,
        critic_visual_tokens: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_ref_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic_forward(
            rl_state=rl_state,
            action=action,
            use_target=True,
            critic_visual_tokens=critic_visual_tokens,
            critic_robot_state=critic_robot_state,
            critic_ref_action=critic_ref_action,
        )

    @torch.no_grad()
    def postprocess_action_model_batch(
        self,
        action_model: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert model-space action chunks into executable env-space actions using
        the same WA post-processing as the base policy.

        Args:
            action_model: [B, C, A_model]
            robot_state:  [B, state_dim], where state_dim matches the padded state
                          used by WA post-processing.
        Returns:
            action_exec:  [B, C, A_env]
        """
        outs = []
        for i in range(action_model.shape[0]):
            outs.append(self._postprocess_pred_delta(action_model[i : i + 1], robot_state[i])[0])
        return torch.stack(outs, dim=0)

    def compute_bc_loss(
        self,
        pred_action: torch.Tensor,
        ref_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return compute_bc_loss(pred_action, ref_action, valid_mask, reduction)

    def set_use_rl_head_for_rollout(self, flag: bool):
        self.use_rl_head_for_rollout_flag.fill_(1 if flag else 0)

    def get_use_rl_head_for_rollout(self) -> bool:
        return bool(int(self.use_rl_head_for_rollout_flag.item()))

    def soft_update_targets(self, tau: Optional[float] = None):
        if tau is None:
            tau = self.target_tau

        with torch.no_grad():
            for p_tgt, p in zip(self.actor_target.parameters(), self.actor_head.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

            for p_tgt, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

    def build_training_batch(
        self,
        env_obs: dict[str, Any],
        ref_action_dropout_p: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Convenience API for future worker code.

        Returns a dict containing:
            visual_latent
            visual_feat
            robot_state
            ref_action
            rl_state
            actor_action
            q1
            q2
            bc_loss_to_ref
        """
        backbone = self.extract_frozen_backbone_batch(env_obs)
        visual_latent = backbone["visual_latent"]
        robot_state = backbone["robot_state"]
        ref_action = backbone["ref_action"]

        visual_feat = self.encode_visual(visual_latent)
        actor_action, actor_aux = self.actor_forward(
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=ref_action,
            ref_action_dropout_p=ref_action_dropout_p,
            use_target=False,
        )
        rl_state = actor_aux["rl_state"]
        q1, q2 = self.critic_forward(
            rl_state=rl_state,
            action=actor_action,
            use_target=False,
            critic_visual_tokens=actor_aux.get("critic_visual_tokens", None),
            critic_robot_state=actor_aux.get("critic_robot_state", None),
            critic_ref_action=actor_aux.get("critic_ref_action", None),
        )
        bc_loss = self.compute_bc_loss(actor_action, ref_action)

        out = {
            "visual_latent": visual_latent,
            "visual_feat": visual_feat,
            "robot_state": robot_state,
            "ref_action": ref_action,
            "rl_state": rl_state,
            "actor_action": actor_action,
            "q1": q1,
            "q2": q2,
            "bc_loss_to_ref": bc_loss,
        }
        out.update(actor_aux)
        return out

    def default_forward(self, **kwargs):
        """
        Temporary training-capable entry point.

        Supported usage:
            default_forward(env_obs=..., mode="build_training_batch")
            default_forward(visual_feat=..., robot_state=..., ref_action=..., mode="actor")
            default_forward(rl_state=..., action=..., mode="critic")
            default_forward(rl_state=..., action=..., critic_visual_tokens=..., critic_robot_state=..., critic_ref_action=..., mode="critic")
        """
        mode = kwargs.pop("mode", "build_training_batch")

        if mode == "build_training_batch":
            env_obs = kwargs.pop("env_obs")
            ref_action_dropout_p = kwargs.pop("ref_action_dropout_p", None)
            return self.build_training_batch(
                env_obs=env_obs,
                ref_action_dropout_p=ref_action_dropout_p,
            )

        if mode == "actor":
            return self.actor_forward(
                visual_feat=kwargs["visual_feat"],
                robot_state=kwargs["robot_state"],
                ref_action=kwargs["ref_action"],
                ref_action_dropout_p=kwargs.get("ref_action_dropout_p", None),
                use_target=kwargs.get("use_target", False),
            )

        if mode == "critic":
            return self.critic_forward(
                rl_state=kwargs.get("rl_state", None),
                action=kwargs["action"],
                use_target=kwargs.get("use_target", False),
                critic_visual_tokens=kwargs.get("critic_visual_tokens", None),
                critic_robot_state=kwargs.get("critic_robot_state", None),
                critic_ref_action=kwargs.get("critic_ref_action", None),
            )

        if mode == "encode_visual":
            return self.encode_visual(kwargs["visual_latent"])

        raise ValueError(f"Unsupported mode for default_forward: {mode}")

    @torch.no_grad()
    def _plan_single(self, env_obs: dict[str, Any], index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Base WA reference action for rollout / warmup.

        Returns:
            action_exec:  executable action chunk in env space
            action_model: action chunk in WA model space
        """
        ref_image = self._build_ref_image(env_obs, index)
        norm_state, state_pad = self._normalize_state(env_obs["states"][index])
        prompt = self._select_prompt(env_obs, index)

        pred_delta_norm = self._run_pipe(
            ref_image=ref_image,
            norm_state=norm_state,
            prompt=prompt,
        )

        if pred_delta_norm.shape[1] < self.action_chunk:
            raise RuntimeError(
                f"WA planner returned only {pred_delta_norm.shape[1]} steps, fewer than actor/runtime num_action_chunks={self.action_chunk}."
            )
        pred_delta_norm = pred_delta_norm[:, : self.action_chunk]

        pred_action_exec = self._postprocess_pred_delta(pred_delta_norm, state_pad)
        pred_action_model = pred_delta_norm.float()
        return pred_action_exec[0], pred_action_model[0]

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
        **kwargs,
    ):
        del mode, compute_values, kwargs
        batch_size = int(env_obs["states"].shape[0])

        rollout_uses_actor = self.get_use_rl_head_for_rollout()
        actor_actions_model = None
        actor_actions_exec = None

        if rollout_uses_actor:
            backbone = self.extract_frozen_backbone_batch(env_obs)
            visual_feat = self.encode_visual(backbone["visual_latent"])
            actor_actions_model, _ = self.actor_forward(
                visual_feat=visual_feat,
                robot_state=backbone["robot_state"],
                ref_action=backbone["ref_action"],
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            actor_actions_model = actor_actions_model.to(self.device_ref)
            actor_actions_exec = self.postprocess_action_model_batch(
                action_model=actor_actions_model,
                robot_state=backbone["robot_state"],
            ).to(self.device_ref)

        if rollout_uses_actor:
            actions_model = actor_actions_model
            actions_exec = actor_actions_exec
        else:
            exec_chunks = []
            model_chunks = []
            for idx in range(batch_size):
                action_exec, action_model = self._plan_single(env_obs, idx)
                exec_chunks.append(action_exec)
                model_chunks.append(action_model)
            actions_exec = torch.stack(exec_chunks, dim=0).to(self.device_ref)
            actions_model = torch.stack(model_chunks, dim=0).to(self.device_ref)

        result = {
            "prev_logprobs": torch.zeros(
                batch_size,
                self.action_chunk,
                device=actions_exec.device,
                dtype=torch.float32,
            ),
            "prev_values": torch.zeros(
                batch_size,
                1,
                device=actions_exec.device,
                dtype=torch.float32,
            ),
            "forward_inputs": {
                "action": actions_exec.reshape(batch_size, -1).contiguous(),
                "model_action": actions_model.reshape(batch_size, -1).contiguous(),
            },
        }
        return actions_exec, result

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.device_ref = torch.device(device)
        self.pipe.to(self.device_ref)
        self._freeze_pipe_modules()   # 双保险，防止后续 device 迁移后忘记冻结检查

        self.visual_compressor.to(self.device_ref)
        self.actor_head.to(self.device_ref)
        self.critic.to(self.device_ref)
        self.actor_target.to(self.device_ref)
        self.critic_target.to(self.device_ref)
        return self


def get_model(cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
    return GigaWorldPolicy(cfg, torch_dtype=torch_dtype)
