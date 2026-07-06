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

"""Async GigaWA actor worker.

This worker overlaps rollout collection with actor/critic training.
The env/rollout side continuously pushes raw trajectories into ``input_channel``;
this worker receives them in a lightweight background thread, then drains and
converts them into the GigaWA replay format at the beginning of each training
step.

Replay-buffer writes intentionally happen in the training thread, not in the
receiver thread.  This avoids concurrent reads/writes to ``TrajectoryReplayBuffer``
while ``run_training`` samples mini-batches.
"""

import queue
import threading
import time
from typing import Any

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_gigawa_policy_worker import EmbodiedGigaWAFSDPPolicy


class AsyncEmbodiedGigaWAFSDPPolicy(EmbodiedGigaWAFSDPPolicy):
    """Off-policy GigaWA worker with asynchronous rollout ingestion."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.should_stop = False
        self._recv_queue: queue.Queue | None = None
        self._recv_rollout_thread: threading.Thread | None = None
        self._async_recv_total_raw = 0
        self._async_drain_total_raw = 0
        self._async_drain_total_gigawa = 0
        self._async_last_drain_metrics: dict[str, Any] = {}
        self._last_not_ready_log_time = 0.0

    # ------------------------------------------------------------------
    # Async rollout ingestion
    # ------------------------------------------------------------------
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Start a background receiver and return immediately.

        ``AsyncEmbodiedRunner`` calls this once after launching async env/rollout
        tasks.  The thread blocks on ``input_channel.get()`` and pushes raw
        trajectories into a bounded local queue.  If training cannot keep up,
        the bounded queue naturally back-pressures the rollout pipeline.
        """
        clear_memory(sync=False)

        if self._recv_queue is None:
            maxsize = int(self.cfg.actor.get("recv_queue_maxsize", 8))
            self._recv_queue = queue.Queue(maxsize=max(1, maxsize))

        if self._recv_rollout_thread is not None and self._recv_rollout_thread.is_alive():
            return

        self.should_stop = False
        self._recv_rollout_thread = threading.Thread(
            target=self._recv_rollout_thread_main,
            args=(input_channel,),
            daemon=True,
        )
        self._recv_rollout_thread.start()
        self.log_on_first_rank(
            "[async_gigawa] Started background rollout receiver "
            f"with recv_queue_maxsize={self._recv_queue.maxsize}."
        )

    def _recv_rollout_thread_main(self, input_channel: Channel) -> None:
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        while not self.should_stop:
            for _ in range(split_num):
                if self.should_stop:
                    break
                try:
                    trajectory: Trajectory = input_channel.get()
                except Exception as exc:  # noqa: BLE001 - keep receiver alive unless stopping
                    if not self.should_stop:
                        self.log_on_first_rank(
                            f"[async_gigawa] Receiver failed to get trajectory: {exc}"
                        )
                    time.sleep(0.1)
                    continue

                while not self.should_stop:
                    try:
                        assert self._recv_queue is not None
                        self._recv_queue.put(trajectory, timeout=1.0)
                        self._async_recv_total_raw += 1
                        break
                    except queue.Full:
                        # Training is behind collection.  Block here to create
                        # back-pressure instead of growing memory unboundedly.
                        continue

    def _pop_received_trajectories(self, max_trajectories: int | None) -> list[Trajectory]:
        if self._recv_queue is None:
            return []

        recv_list: list[Trajectory] = []
        while True:
            if max_trajectories is not None and len(recv_list) >= max_trajectories:
                break
            try:
                recv_list.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        return recv_list

    @torch.no_grad()
    def _drain_received_trajectories(
        self, max_trajectories: int | None = None
    ) -> dict[str, Any]:
        """Move queued raw trajectories into GigaWA replay/demo buffers."""
        recv_list = self._pop_received_trajectories(max_trajectories)
        queue_size_after_pop = self._recv_queue.qsize() if self._recv_queue is not None else 0

        if not recv_list:
            metrics = {
                "async_recv/raw_drained": 0.0,
                "async_recv/gigawa_added": 0.0,
                "async_recv/queue_size": float(queue_size_after_pop),
                "async_recv/raw_total_received": float(self._async_recv_total_raw),
                "async_recv/raw_total_drained": float(self._async_drain_total_raw),
                "async_recv/gigawa_total_added": float(self._async_drain_total_gigawa),
            }
            self._async_last_drain_metrics = metrics
            return metrics

        gigawa_list: list[Trajectory] = []
        for traj in recv_list:
            gigawa_list.extend(self._convert_trajectory_for_gigawa(traj))

        if self.store_online_replay and len(gigawa_list) > 0:
            self.replay_buffer.add_trajectories(gigawa_list)

        self._store_offline_collection_trajectories(gigawa_list)

        intervene_count = 0
        if self.demo_buffer is not None and self.store_online_demo_interventions:
            intervene_traj_list = []
            for traj in gigawa_list:
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)
            intervene_count = len(intervene_traj_list)
            if intervene_count > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        self._async_drain_total_raw += len(recv_list)
        self._async_drain_total_gigawa += len(gigawa_list)

        metrics = {
            "async_recv/raw_drained": float(len(recv_list)),
            "async_recv/gigawa_added": float(len(gigawa_list)),
            "async_recv/intervene_added": float(intervene_count),
            "async_recv/queue_size": float(queue_size_after_pop),
            "async_recv/raw_total_received": float(self._async_recv_total_raw),
            "async_recv/raw_total_drained": float(self._async_drain_total_raw),
            "async_recv/gigawa_total_added": float(self._async_drain_total_gigawa),
        }
        self._async_last_drain_metrics = metrics
        return metrics

    def _sync_bool_all_ranks(self, value: bool, op=torch.distributed.ReduceOp.MIN) -> bool:
        flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(flag, op=op)
        return bool(flag.item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @Worker.timer("run_training")
    def run_training(self):
        """Drain async trajectories, then run a globally synchronized update.

        Different actor ranks may receive trajectories at slightly different
        times.  A local readiness check can deadlock FSDP if one rank enters
        backward/barrier while another rank skips the step.  Therefore both
        ``can_train`` and ``train_actor`` are synchronized with an all-rank MIN
        before any optimizer step is executed.
        """
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        max_drain = self.cfg.actor.get("recv_drain_max_trajectories", 32)
        if max_drain is not None:
            max_drain = int(max_drain)
            if max_drain <= 0:
                max_drain = None

        ingest_metrics = self._drain_received_trajectories(max_trajectories=max_drain)

        readiness = self._get_training_readiness()
        can_train = self._sync_bool_all_ranks(readiness["can_train"])
        train_actor = self._sync_bool_all_ranks(readiness["train_actor"])

        if not can_train:
            now = time.time()
            log_interval = float(self.cfg.actor.get("async_skip_log_interval_s", 10.0))
            if now - self._last_not_ready_log_time >= log_interval:
                start_status = readiness["start_status"]
                self.log_on_first_rank(
                    "Skipping async training because at least one actor rank buffer is not ready: "
                    f"local_replay_ready={start_status['replay_ready']} "
                    f"(need {readiness['replay_start_size']}, have {len(self.replay_buffer)}), "
                    f"local_demo_ready={start_status['demo_ready']} "
                    f"(need {readiness['demo_start_size']}, "
                    f"have {len(self.demo_buffer) if self.demo_buffer is not None else 0}), "
                    f"local_queue={ingest_metrics.get('async_recv/queue_size', 0.0)}, "
                    f"total_received={ingest_metrics.get('async_recv/raw_total_received', 0.0)}, "
                    f"total_added={ingest_metrics.get('async_recv/gigawa_total_added', 0.0)}"
                )
                self._last_not_ready_log_time = now
            skip_sleep = float(self.cfg.actor.get("async_skip_sleep_s", 0.2))
            if skip_sleep > 0:
                time.sleep(skip_sleep)
            return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        append_to_dict(metrics, ingest_metrics)
        append_to_dict(
            metrics,
            {
                "buffer_mix/use_replay_this_step": float(readiness["start_status"]["use_replay"]),
                "buffer_mix/use_demo_this_step": float(readiness["start_status"]["use_demo"]),
                "buffer_mix/train_actor": float(train_actor),
                "buffer_mix/replay_ready": float(readiness["start_status"]["replay_ready"]),
                "buffer_mix/demo_ready": float(readiness["start_status"]["demo_ready"]),
                "buffer_mix/global_can_train": float(can_train),
                "buffer_mix/global_train_actor": float(train_actor),
                "buffer_mix/replay_required_for_training": float(self.replay_required_for_training),
                "buffer_mix/allow_train_on_demo_only": float(self.allow_train_on_demo_only),
                "stage/freeze_actor_updates": float(self.freeze_actor_updates),
                "stage/freeze_critic_updates": float(self.freeze_critic_updates),
                "stage/freeze_visual_layers": float(self.freeze_visual_layers),
            },
        )

        update_epoch = self.utd_ratio
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    async def stop(self) -> None:
        self.should_stop = True
        recv_thread = self._recv_rollout_thread
        if recv_thread is not None and recv_thread.is_alive():
            recv_thread.join(timeout=5.0)

        if getattr(self, "buffer_dataset", None) is not None:
            self.buffer_dataset.close()
        if self.offline_collection_enable:
            self.finalize_offline_collection()
