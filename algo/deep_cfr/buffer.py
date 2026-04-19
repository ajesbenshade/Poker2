"""Reservoir-sampling buffer for Deep CFR.

Stores ``(observation, legal_mask, target, weight)`` tuples and supports
uniform random sampling from the entire stream (Vitter's Algorithm R) once
capacity is reached.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np


class ReservoirBuffer:
    """Thread-safe reservoir buffer backed by preallocated numpy arrays."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        num_actions: int,
        target_dim: Optional[int] = None,
        seed: int = 0,
    ):
        if target_dim is None:
            target_dim = num_actions
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.target_dim = target_dim
        self._obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._legal = np.zeros((self.capacity, num_actions), dtype=np.float32)
        self._target = np.zeros((self.capacity, target_dim), dtype=np.float32)
        self._weight = np.zeros(self.capacity, dtype=np.float32)
        self._size = 0
        self._seen = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def total_seen(self) -> int:
        return self._seen

    def add(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        target: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        with self._lock:
            self._seen += 1
            if self._size < self.capacity:
                idx = self._size
                self._size += 1
            else:
                idx = int(self._rng.integers(0, self._seen))
                if idx >= self.capacity:
                    return
            self._obs[idx] = obs
            self._legal[idx] = legal
            self._target[idx] = target
            self._weight[idx] = weight

    def add_batch(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        for i in range(n):
            self.add(obs[i], legal[i], target[i], float(weights[i]))

    def add_arrays(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Vectorized reservoir insertion of a numpy block.

        Equivalent to calling :meth:`add` once per row but acquires the lock
        only once and avoids per-row Python overhead. Uses Vitter's
        Algorithm R: the first ``capacity - size`` rows fill empty slots, the
        rest run the standard random-replacement step.
        """
        n = int(obs.shape[0])
        if n == 0:
            return
        if obs.shape[1] != self.obs_dim or legal.shape[1] != self.num_actions \
                or target.shape[1] != self.target_dim or weights.shape[0] != n:
            raise ValueError(
                f"shape mismatch: obs={obs.shape} legal={legal.shape} "
                f"target={target.shape} weights={weights.shape} "
                f"vs (obs_dim={self.obs_dim}, num_actions={self.num_actions}, "
                f"target_dim={self.target_dim})"
            )
        with self._lock:
            # Phase 1: fill empty slots if any
            free = self.capacity - self._size
            fill = min(free, n)
            if fill > 0:
                start = self._size
                end = start + fill
                self._obs[start:end] = obs[:fill]
                self._legal[start:end] = legal[:fill]
                self._target[start:end] = target[:fill]
                self._weight[start:end] = weights[:fill]
                self._size += fill
            # Phase 2: standard reservoir replacement for the rest
            remaining = n - fill
            if remaining > 0:
                # Each remaining row r has global index self._seen + fill + r,
                # and is kept (replaces a random slot) with probability
                # capacity / (global_index + 1). Implementation: draw uniform
                # candidate slot in [0, global_index]; keep if < capacity.
                global_indices = np.arange(
                    self._seen + fill,
                    self._seen + fill + remaining,
                    dtype=np.int64,
                )
                # sample uniform integer in [0, gi]  (inclusive)
                candidates = self._rng.integers(
                    0, global_indices + 1, dtype=np.int64
                )
                kept_mask = candidates < self.capacity
                if kept_mask.any():
                    src_idx = np.nonzero(kept_mask)[0] + fill
                    dst_idx = candidates[kept_mask]
                    self._obs[dst_idx] = obs[src_idx]
                    self._legal[dst_idx] = legal[src_idx]
                    self._target[dst_idx] = target[src_idx]
                    self._weight[dst_idx] = weights[src_idx]
            self._seen += n

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if self._size == 0:
                raise RuntimeError("buffer is empty")
            idx = self._rng.integers(0, self._size, size=batch_size)
            return (
                self._obs[idx].copy(),
                self._legal[idx].copy(),
                self._target[idx].copy(),
                self._weight[idx].copy(),
            )

    def clear(self) -> None:
        with self._lock:
            self._size = 0
            self._seen = 0
