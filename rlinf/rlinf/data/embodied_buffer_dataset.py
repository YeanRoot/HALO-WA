# Copyright 2026 The RLinf Authors.
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

import queue
import threading
import time
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch
import copy

logger = get_logger()


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for key in ("actions", "rewards", "dones", "terminations", "truncations", "prev_logprobs", "prev_values", "versions"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    for parent_key in ("curr_obs", "next_obs", "forward_inputs"):
        nested = batch.get(parent_key)
        if isinstance(nested, dict):
            for value in nested.values():
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
    raise ValueError("Unable to infer batch size for compatibility padding.")


def _zeros_like_batch_dim(ref: torch.Tensor, batch_size: int) -> torch.Tensor:
    shape = list(ref.shape)
    shape[0] = batch_size
    return torch.zeros(shape, dtype=ref.dtype, device=ref.device)


def _build_forward_inputs_fallback(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    actions = batch.get("actions")
    if not isinstance(actions, torch.Tensor):
        return {}
    return {
        "action": actions.clone(),
        "model_action": actions.clone(),
    }


def _align_forward_inputs_for_concat(
    replay_batch: dict[str, Any],
    demo_batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_batch = copy.deepcopy(replay_batch)
    demo_batch = copy.deepcopy(demo_batch)

    replay_fi = replay_batch.get("forward_inputs")
    demo_fi = demo_batch.get("forward_inputs")

    if not isinstance(replay_fi, dict):
        replay_fi = _build_forward_inputs_fallback(replay_batch)
        replay_batch["forward_inputs"] = replay_fi
    if not isinstance(demo_fi, dict):
        demo_fi = _build_forward_inputs_fallback(demo_batch)
        demo_batch["forward_inputs"] = demo_fi

    if not replay_fi and not demo_fi:
        return replay_batch, demo_batch

    replay_bs = _infer_batch_size(replay_batch)
    demo_bs = _infer_batch_size(demo_batch)
    all_keys = set(replay_fi.keys()) | set(demo_fi.keys())

    for key in all_keys:
        replay_tensor = replay_fi.get(key)
        demo_tensor = demo_fi.get(key)

        if replay_tensor is None and demo_tensor is None:
            continue
        if replay_tensor is None and isinstance(demo_tensor, torch.Tensor):
            replay_fi[key] = _zeros_like_batch_dim(demo_tensor, replay_bs)
            continue
        if demo_tensor is None and isinstance(replay_tensor, torch.Tensor):
            demo_fi[key] = _zeros_like_batch_dim(replay_tensor, demo_bs)
            continue

    return replay_batch, demo_batch


class ReplayBufferDataset(IterableDataset):
    """Dataset that samples batches from replay and demonstration buffers.

    Compared with the original fixed 50/50 mixing rule, this version supports
    three regimes that are all useful for RLPD / offline-to-online training:

    - replay only
    - demo only
    - replay + demo mixed with a configurable demo sampling ratio

    If one source is temporarily unavailable, the dataset can optionally fall
    back to the other source instead of stalling forever.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        demo_sample_ratio: float = 0.5,
        allow_demo_only_fallback: bool = True,
        allow_replay_only_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = int(max(0, min_replay_buffer_size))
        self.min_demo_buffer_size = int(max(0, min_demo_buffer_size))
        self.batch_size = int(batch_size)
        self.demo_sample_ratio = float(min(max(demo_sample_ratio, 0.0), 1.0))
        self.allow_demo_only_fallback = bool(allow_demo_only_fallback)
        self.allow_replay_only_fallback = bool(allow_replay_only_fallback)

        self.replay_batch_size_target, self.demo_batch_size_target = self._compute_target_batch_sizes(
            self.batch_size,
            self.demo_sample_ratio,
            has_demo_buffer=self.demo_buffer is not None,
        )

    @staticmethod
    def _compute_target_batch_sizes(
        batch_size: int,
        demo_sample_ratio: float,
        has_demo_buffer: bool,
    ) -> tuple[int, int]:
        if batch_size <= 0:
            raise ValueError(f"{batch_size=} must be positive")

        if not has_demo_buffer or demo_sample_ratio <= 0.0:
            return batch_size, 0
        if demo_sample_ratio >= 1.0:
            return 0, batch_size

        demo_batch_size = int(round(batch_size * demo_sample_ratio))
        demo_batch_size = max(1, min(batch_size - 1, demo_batch_size))
        replay_batch_size = batch_size - demo_batch_size
        return replay_batch_size, demo_batch_size

    def _is_replay_requested(self) -> bool:
        return self.replay_batch_size_target > 0

    def _is_demo_requested(self) -> bool:
        return self.demo_buffer is not None and self.demo_batch_size_target > 0

    def _is_replay_ready(self) -> bool:
        if not self._is_replay_requested():
            return False
        return self.replay_buffer.is_ready(self.min_replay_buffer_size)

    def _is_demo_ready(self) -> bool:
        if not self._is_demo_requested():
            return False
        return self.demo_buffer.is_ready(self.min_demo_buffer_size)

    def resolve_sampling_plan(self) -> Optional[tuple[int, int]]:
        replay_requested = self._is_replay_requested()
        demo_requested = self._is_demo_requested()
        replay_ready = self._is_replay_ready()
        demo_ready = self._is_demo_ready()

        if replay_requested and demo_requested:
            if replay_ready and demo_ready:
                return self.replay_batch_size_target, self.demo_batch_size_target
            if demo_ready and self.allow_demo_only_fallback:
                return 0, self.batch_size
            if replay_ready and self.allow_replay_only_fallback:
                return self.batch_size, 0
            return None

        if replay_requested:
            return (self.batch_size, 0) if replay_ready else None

        if demo_requested:
            return (0, self.batch_size) if demo_ready else None

        return None

    def sample_once(self) -> Optional[dict[str, torch.Tensor]]:
        plan = self.resolve_sampling_plan()
        if plan is None:
            return None

        replay_batch_size, demo_batch_size = plan
        if replay_batch_size > 0 and demo_batch_size > 0:
            replay_batch = self.replay_buffer.sample(replay_batch_size)
            demo_batch = self.demo_buffer.sample(demo_batch_size)
            replay_batch, demo_batch = _align_forward_inputs_for_concat(replay_batch, demo_batch)
            return concat_batch(replay_batch, demo_batch)
        if replay_batch_size > 0:
            return self.replay_buffer.sample(replay_batch_size)
        if demo_batch_size > 0:
            return self.demo_buffer.sample(demo_batch_size)
        return None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            batch = self.sample_once()
            if batch is None:
                time.sleep(1)
                continue
            yield batch

    def close(self) -> None:
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self) -> None:
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches replay/demo batches in a background thread."""

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        demo_sample_ratio: float = 0.5,
        allow_demo_only_fallback: bool = True,
        allow_replay_only_fallback: bool = True,
        prefetch_size: int = 5,
    ) -> None:
        super().__init__(
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            batch_size=batch_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            demo_sample_ratio=demo_sample_ratio,
            allow_demo_only_fallback=allow_demo_only_fallback,
            allow_replay_only_fallback=allow_replay_only_fallback,
        )
        self._stop_event = threading.Event()
        self.prefetch_size = prefetch_size
        assert self.prefetch_size > 0, f"{self.prefetch_size=} must be greater than 0"
        self.preload_queue = queue.Queue(maxsize=prefetch_size)
        self.sample_thread = None
        self._exception = None

    def _sample_buffer(self) -> None:
        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            batch = self.sample_once()
            if batch is None:
                time.sleep(1)
                continue

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error in ReplayBufferDataset: {e}")
                self._exception = e
                self._stop_event.set()
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(target=self._sample_buffer, daemon=True)
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    if self._exception is not None:
                        raise RuntimeError("Sampling thread failed") from self._exception
                    break
                continue

    def close(self) -> None:
        self._stop_event.set()
        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )
        super().close()

    def __del__(self) -> None:
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
