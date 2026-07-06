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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")

    async_rollout_training = bool(cfg.runner.get("async_rollout_training", False))
    loss_type = cfg.algorithm.loss_type
    model_type = cfg.actor.model.get("model_type", None)

    if async_rollout_training:
        if loss_type != "embodied_gigawa" and model_type != "giga_world_policy":
            raise ValueError(
                "runner.async_rollout_training=true is currently implemented for "
                f"embodied_gigawa / giga_world_policy only, got loss_type={loss_type}, "
                f"model_type={model_type}."
            )
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        from rlinf.workers.actor.async_fsdp_gigawa_policy_worker import (
            AsyncEmbodiedGigaWAFSDPPolicy,
        )
        from rlinf.workers.env.async_env_worker import AsyncEnvWorker
        from rlinf.workers.rollout.hf.async_huggingface_worker import (
            AsyncMultiStepRolloutWorker,
        )

        runner_cls = AsyncEmbodiedRunner
        actor_worker_cls = AsyncEmbodiedGigaWAFSDPPolicy
        rollout_worker_cls = AsyncMultiStepRolloutWorker
        env_worker_cls = AsyncEnvWorker
    else:
        from rlinf.runners.embodied_runner import EmbodiedRunner
        from rlinf.workers.env.env_worker import EnvWorker
        from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

        runner_cls = EmbodiedRunner
        rollout_worker_cls = MultiStepRolloutWorker
        env_worker_cls = EnvWorker

        if loss_type == "embodied_sac":
            from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
            actor_worker_cls = EmbodiedSACFSDPPolicy
        elif loss_type == "embodied_gigawa" or model_type == "giga_world_policy":
            from rlinf.workers.actor.fsdp_gigawa_policy_worker import EmbodiedGigaWAFSDPPolicy
            actor_worker_cls = EmbodiedGigaWAFSDPPolicy
        elif loss_type == "embodied_dagger":
            from rlinf.workers.actor.fsdp_dagger_policy_worker import (
                EmbodiedDAGGERFSDPPolicy,
            )
            actor_worker_cls = EmbodiedDAGGERFSDPPolicy
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
            actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
