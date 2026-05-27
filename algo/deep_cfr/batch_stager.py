"""Reusable pinned memory stager for Deep CFR training batches.

This module attacks the high-frequency copy + device transfer overhead
in the advantage and strategy network training loops.

Problem (before this change):
- ReservoirBuffer.sample() did 4x .copy() on every call.
- _array_to_device() did torch.from_numpy + conditional pin_memory() +
  .to(device) thousands of times per CFR iteration.
- pin_memory() attempts are expensive and fragile on ROCm.

Solution:
- PinnedBatchStager pre-allocates pinned host tensors once.
- Training code can request samples without copies.
- The stager does a single fast host-to-pinned copy, then a non-blocking
  transfer to GPU. Subsequent steps reuse the same pinned buffers.

This significantly reduces CPU overhead and improves GPU utilization
during the 4000+ advantage training steps + 8000+ strategy steps per
CFR iteration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class PinnedBatchStager:
    """Manages reusable pinned host buffers for training batches.

    Typical usage in the trainer:

        stager = PinnedBatchStager(max_batch_size, obs_dim, num_actions, device)

        for step in range(steps):
            obs_np, legal_np, target_np, weight_np = buffer.sample(bs, copy=False)
            obs, legal, target, weight = stager.stage(obs_np, legal_np, target_np, weight_np)
            # ... use tensors ...
    """

    def __init__(
        self,
        max_batch_size: int,
        obs_dim: int,
        num_actions: int,
        device: torch.device,
    ):
        self.max_batch_size = int(max_batch_size)
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = device

        # Pre-allocate pinned host tensors (float32 for all our data)
        self._pinned_obs = torch.empty(
            (self.max_batch_size, self.obs_dim), dtype=torch.float32, pin_memory=True
        )
        self._pinned_legal = torch.empty(
            (self.max_batch_size, self.num_actions), dtype=torch.float32, pin_memory=True
        )
        self._pinned_target = torch.empty(
            (self.max_batch_size, self.num_actions), dtype=torch.float32, pin_memory=True
        )
        self._pinned_weight = torch.empty(
            (self.max_batch_size,), dtype=torch.float32, pin_memory=True
        )

        self._current_batch_size = 0

    def stage(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy numpy batch into pinned buffers and return device tensors (non-blocking)."""
        bs = obs.shape[0]
        if bs > self.max_batch_size:
            raise ValueError(f"batch size {bs} exceeds stager max {self.max_batch_size}")

        # Fast host copy into pinned memory (contiguous writes are cheap)
        self._pinned_obs[:bs].copy_(torch.from_numpy(obs))
        self._pinned_legal[:bs].copy_(torch.from_numpy(legal))
        self._pinned_target[:bs].copy_(torch.from_numpy(target))
        self._pinned_weight[:bs].copy_(torch.from_numpy(weight))

        self._current_batch_size = bs

        # Non-blocking transfer to device
        obs_t = self._pinned_obs[:bs].to(self.device, non_blocking=True)
        legal_t = self._pinned_legal[:bs].to(self.device, non_blocking=True)
        target_t = self._pinned_target[:bs].to(self.device, non_blocking=True)
        weight_t = self._pinned_weight[:bs].to(self.device, non_blocking=True)

        return obs_t, legal_t, target_t, weight_t

    def stage_validation(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as stage, but for validation sets (can be smaller)."""
        return self.stage(obs, legal, target, weight)

    def get_current_device_batch_size(self) -> int:
        return self._current_batch_size


def make_stager_from_config(
    cfg,
    obs_dim: int,
    num_actions: int,
    device: torch.device,
) -> Optional[PinnedBatchStager]:
    """Convenience constructor used by DeepCFRTrainer."""
    if device.type != "cuda":
        return None
    max_bs = getattr(cfg, "train_batch_size", 4096)
    return PinnedBatchStager(max_bs, obs_dim, num_actions, device)
