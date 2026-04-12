import os

import numpy as np
import torch


class Float16ReservoirBuffer:
    def __init__(self, storage_dir, name, capacity, feature_dim, target_dim, create=False):
        self.storage_dir = storage_dir
        self.name = name
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)
        self.size = 0
        self.total_seen = 0
        self._rng = np.random.default_rng()

        os.makedirs(storage_dir, exist_ok=True)
        self.features_path = os.path.join(storage_dir, f'{name}_features.dat')
        self.targets_path = os.path.join(storage_dir, f'{name}_targets.dat')
        self.weights_path = os.path.join(storage_dir, f'{name}_weights.dat')

        mode = 'w+' if create or not os.path.exists(self.features_path) else 'r+'
        self.features = np.memmap(self.features_path, dtype=np.float16, mode=mode, shape=(self.capacity, self.feature_dim))
        self.targets = np.memmap(self.targets_path, dtype=np.float16, mode=mode, shape=(self.capacity, self.target_dim))
        self.weights = np.memmap(self.weights_path, dtype=np.float32, mode=mode, shape=(self.capacity,))

        if mode == 'w+':
            self.features[:] = 0.0
            self.targets[:] = 0.0
            self.weights[:] = 0.0
            self.flush()

    def __len__(self):
        return self.size

    def flush(self):
        self.features.flush()
        self.targets.flush()
        self.weights.flush()

    def add_batch(self, features, targets, weights=None):
        features = np.asarray(features, dtype=np.float16)
        targets = np.asarray(targets, dtype=np.float16)
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError('features and targets must be rank-2 arrays')
        if features.shape[0] != targets.shape[0]:
            raise ValueError('features and targets must have matching batch sizes')
        if features.shape[1] != self.feature_dim or targets.shape[1] != self.target_dim:
            raise ValueError('feature or target dimensions do not match buffer configuration')

        if weights is None:
            weights = np.ones(features.shape[0], dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)

        finite_mask = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1) & np.isfinite(weights)
        if not finite_mask.any():
            return 0

        features = features[finite_mask]
        targets = targets[finite_mask]
        weights = weights[finite_mask]

        inserted = 0
        for row_index in range(features.shape[0]):
            if self.size < self.capacity:
                write_index = self.size
                self.size += 1
            else:
                replacement_index = int(self._rng.integers(0, self.total_seen + 1))
                if replacement_index >= self.capacity:
                    self.total_seen += 1
                    continue
                write_index = replacement_index

            self.features[write_index] = features[row_index]
            self.targets[write_index] = targets[row_index]
            self.weights[write_index] = weights[row_index]
            self.total_seen += 1
            inserted += 1

        return inserted

    def sample(self, batch_size, device):
        if self.size <= 0:
            return None

        sample_size = min(int(batch_size), self.size)
        indices = self._rng.integers(0, self.size, size=sample_size)
        features = np.asarray(self.features[indices], dtype=np.float32)
        targets = np.asarray(self.targets[indices], dtype=np.float32)
        weights = np.asarray(self.weights[indices], dtype=np.float32)

        return {
            'features': torch.as_tensor(features, dtype=torch.float32, device=device),
            'targets': torch.as_tensor(targets, dtype=torch.float32, device=device),
            'weights': torch.as_tensor(weights, dtype=torch.float32, device=device),
        }

    def state_dict(self):
        return {
            'storage_dir': self.storage_dir,
            'name': self.name,
            'capacity': self.capacity,
            'feature_dim': self.feature_dim,
            'target_dim': self.target_dim,
            'size': self.size,
            'total_seen': self.total_seen,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        buffer = cls(
            storage_dir=state_dict['storage_dir'],
            name=state_dict['name'],
            capacity=state_dict['capacity'],
            feature_dim=state_dict['feature_dim'],
            target_dim=state_dict['target_dim'],
            create=False,
        )
        buffer.size = int(state_dict['size'])
        buffer.total_seen = int(state_dict['total_seen'])
        return buffer