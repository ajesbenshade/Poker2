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
