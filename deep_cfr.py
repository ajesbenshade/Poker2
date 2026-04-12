import json
import os
import random
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np

_ALLOCATOR_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:128"

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HIP_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11_0_0")
os.environ.setdefault("PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from abstractions import encode_infoset, feature_vector_size
from config import Config


def regret_matching(values: torch.Tensor) -> torch.Tensor:
    squeeze_output = values.dim() == 1
    if squeeze_output:
        values = values.unsqueeze(0)

    positive = torch.clamp(torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    normalizer = positive.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / positive.shape[-1])
    strategy = torch.where(normalizer > 0, positive / normalizer.clamp_min(1e-8), uniform)

    return strategy.squeeze(0) if squeeze_output else strategy


def _new_grad_scaler():
    enabled = Config.scaler_enabled()
    try:
        return torch.amp.GradScaler(Config.autocast_device_type(), enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ReplayBuffer:
    def __init__(self, capacity: int, initial_capacity=None):
        self.capacity = int(capacity)
        self.initial_capacity = max(1, min(
            self.capacity,
            int(Config.REPLAY_BUFFER_INITIAL_CAPACITY if initial_capacity is None else initial_capacity),
        ))
        self.feature_dim = feature_vector_size()
        self.features = None
        self.targets = None
        self.weights = None
        self.size = 0
        self.allocated = 0
        self.total_seen = 0

    def _ensure_capacity(self, required_size):
        if self.allocated >= required_size:
            return

        new_capacity = self.allocated or self.initial_capacity
        while new_capacity < required_size and new_capacity < self.capacity:
            new_capacity = min(
                self.capacity,
                max(new_capacity + 1, new_capacity * int(Config.REPLAY_BUFFER_GROWTH_FACTOR)),
            )

        new_features = np.zeros((new_capacity, self.feature_dim), dtype=np.float16)
        new_targets = np.zeros((new_capacity, Config.NUM_ACTIONS), dtype=np.float16)
        new_weights = np.zeros((new_capacity,), dtype=np.float32)

        if self.size > 0 and self.features is not None:
            new_features[:self.size] = self.features[:self.size]
            new_targets[:self.size] = self.targets[:self.size]
            new_weights[:self.size] = self.weights[:self.size]

        self.features = new_features
        self.targets = new_targets
        self.weights = new_weights
        self.allocated = new_capacity

    def _write(self, index, infoset, targets, weight):
        self._ensure_capacity(index + 1)
        encoded = encode_infoset(infoset).astype(np.float16, copy=False)
        self.features[index] = encoded
        self.targets[index] = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
        self.weights[index] = max(float(weight), 1e-6)

    def add(self, infoset, targets, weight=1.0):
        target_array = np.asarray(targets, dtype=np.float32)
        if target_array.shape != (Config.NUM_ACTIONS,):
            raise ValueError(f"Expected targets shaped {(Config.NUM_ACTIONS,)}, got {target_array.shape}")

        self.total_seen += 1
        if self.size < self.capacity:
            self._write(self.size, infoset, target_array, weight)
            self.size += 1
            return

        replace_idx = random.randrange(self.total_seen)
        if replace_idx < self.capacity:
            self._write(replace_idx, infoset, target_array, weight)

    def sample(self, batch_size: int, device=Config.DEVICE):
        if self.size == 0:
            return None

        batch_size = min(int(batch_size), self.size)
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        features = torch.as_tensor(self.features[indices], device=device, dtype=Config.NN_DTYPE)
        targets = torch.as_tensor(self.targets[indices], device=device, dtype=Config.NN_DTYPE)
        weights = torch.as_tensor(self.weights[indices], device=device, dtype=Config.NN_DTYPE)
        return features, targets, weights

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'initial_capacity': self.initial_capacity,
            'size': self.size,
            'allocated': self.allocated,
            'total_seen': self.total_seen,
            'features': None if self.features is None else self.features[:self.size].copy(),
            'targets': None if self.targets is None else self.targets[:self.size].copy(),
            'weights': None if self.weights is None else self.weights[:self.size].copy(),
        }

    def load_state_dict(self, state):
        self.capacity = int(state.get('capacity', self.capacity))
        self.initial_capacity = int(state.get('initial_capacity', self.initial_capacity))
        self.size = int(state.get('size', 0))
        self.total_seen = int(state.get('total_seen', self.size))
        self.allocated = 0
        self.features = None
        self.targets = None
        self.weights = None

        if self.size <= 0:
            self.size = 0
            return

        self._ensure_capacity(self.size)
        self.features[:self.size] = state['features'][:self.size]
        self.targets[:self.size] = state['targets'][:self.size]
        self.weights[:self.size] = state['weights'][:self.size]

    def __len__(self):
        return self.size


class FeedForwardPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        hidden_dim = Config.MODEL_HIDDEN_DIM
        num_hidden_layers = max(int(Config.MODEL_NUM_LAYERS), 1)

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.to(device=Config.DEVICE, dtype=Config.NN_DTYPE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.training and Config.GRADIENT_CHECKPOINTING and len(self.network) > 2:
            checkpoint_input = features
            if not checkpoint_input.requires_grad:
                checkpoint_input = checkpoint_input.detach().requires_grad_(True)
            segments = max(1, min(Config.CHECKPOINT_SEGMENTS, len(self.network)))
            try:
                return checkpoint_sequential(self.network, segments, checkpoint_input, use_reentrant=False)
            except TypeError:
                return checkpoint_sequential(self.network, segments, checkpoint_input)
        return self.network(features)


class DeepCFRAgent:
    def __init__(self):
        input_dim = feature_vector_size()
        player_count = len(Config.DEFAULT_STACK_SIZES)

        self.player_count = player_count
        self.advantage_networks = [
            FeedForwardPolicy(input_dim, Config.NUM_ACTIONS)
            for _ in range(player_count)
        ]
        self.average_network = FeedForwardPolicy(input_dim, Config.NUM_ACTIONS)
        self.advantage_optimizers = [
            torch.optim.Adam(network.parameters(), lr=Config.NN_LEARNING_RATE)
            for network in self.advantage_networks
        ]
        self.average_optimizer = torch.optim.Adam(
            self.average_network.parameters(),
            lr=Config.NN_LEARNING_RATE,
        )
        self.advantage_scalers = [_new_grad_scaler() for _ in range(player_count)]
        self.average_scaler = _new_grad_scaler()

        self.advantage_memories = [ReplayBuffer(Config.REPLAY_BUFFER_SIZE) for _ in range(player_count)]
        self.strategy_memory = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
        self.visited_infosets = OrderedDict()
        self.skipped_batches = {f'advantage_{player}': 0 for player in range(player_count)}
        self.skipped_batches['strategy'] = 0

    def register_infoset(self, infoset):
        if infoset.key in self.visited_infosets:
            self.visited_infosets.move_to_end(infoset.key)
        elif len(self.visited_infosets) >= Config.MAX_VISITED_INFOSETS:
            self.visited_infosets.popitem(last=False)
        self.visited_infosets[infoset.key] = infoset

    def _autocast_context(self):
        if not Config.AMP_ENABLED:
            return nullcontext()
        return torch.autocast(
            device_type=Config.autocast_device_type(),
            dtype=Config.AMP_DTYPE,
            enabled=True,
        )

    @staticmethod
    def _batch_is_finite(*tensors):
        return all(torch.isfinite(tensor).all().item() for tensor in tensors)

    @staticmethod
    def _sanitize_values(tensor, *, min_value=None, max_value=None):
        cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if min_value is not None or max_value is not None:
            cleaned = torch.clamp(cleaned, min=min_value, max=max_value)
        return cleaned

    @staticmethod
    def _move_optimizer_state(optimizer):
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=Config.DEVICE)

    def _encode_infoset(self, infoset):
        encoded = torch.tensor(
            encode_infoset(infoset),
            device=Config.DEVICE,
            dtype=Config.NN_DTYPE,
        )
        return self._sanitize_values(encoded).unsqueeze(0)

    def advantage_strategy(self, infoset, player: int) -> torch.Tensor:
        self.register_infoset(infoset)
        with torch.no_grad():
            advantages = self._sanitize_values(
                self.advantage_networks[player](self._encode_infoset(infoset)).squeeze(0),
                min_value=-1e4,
                max_value=1e4,
            )
        return regret_matching(advantages)

    def average_strategy(self, infoset) -> torch.Tensor:
        self.register_infoset(infoset)
        with torch.no_grad():
            logits = self._sanitize_values(
                self.average_network(self._encode_infoset(infoset)).squeeze(0),
                min_value=-1e4,
                max_value=1e4,
            )
        strategy = F.softmax(logits, dim=-1)
        return self._sanitize_values(strategy, min_value=0.0, max_value=1.0)

    def record_advantage(self, infoset, player: int, advantages, weight=1.0):
        self.register_infoset(infoset)
        if isinstance(advantages, torch.Tensor):
            advantages = advantages.detach().cpu().numpy()
        targets = np.asarray(advantages, dtype=np.float32)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        self.advantage_memories[player].add(infoset, targets, weight=weight)

    def record_strategy(self, infoset, strategy, weight=1.0):
        self.register_infoset(infoset)
        if isinstance(strategy, torch.Tensor):
            strategy = strategy.detach().cpu().numpy()
        targets = np.asarray(strategy, dtype=np.float32)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.clip(targets, 0.0, 1.0)
        total = float(targets.sum())
        if total <= 0.0:
            targets = np.full((Config.NUM_ACTIONS,), 1.0 / Config.NUM_ACTIONS, dtype=np.float32)
        else:
            targets /= total
        self.strategy_memory.add(infoset, targets, weight=weight)

    def _optimizer_step(self, loss, optimizer, scaler, network):
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                return None
            scaler.step(optimizer)
            scaler.update()
            return grad_norm

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            return None
        optimizer.step()
        return grad_norm

    def train_advantage_network(self, player: int, steps=None, batch_size=None) -> float:
        steps = Config.current_nn_train_steps() if steps is None else int(steps)
        batch_size = Config.current_nn_batch_size() if batch_size is None else int(batch_size)
        if len(self.advantage_memories[player]) == 0 or steps <= 0:
            return 0.0

        network = self.advantage_networks[player]
        optimizer = self.advantage_optimizers[player]
        scaler = self.advantage_scalers[player]
        losses = []

        network.train()
        for _ in range(steps):
            batch = self.advantage_memories[player].sample(batch_size, device=Config.DEVICE)
            if batch is None:
                break

            features, targets, weights = batch
            if not self._batch_is_finite(features, targets, weights):
                self.skipped_batches[f'advantage_{player}'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            weights = weights / weights.mean().clamp_min(1.0)
            targets = self._sanitize_values(targets, min_value=-1e4, max_value=1e4)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                predictions = self._sanitize_values(
                    network(features).to(dtype=Config.NN_DTYPE),
                    min_value=-1e4,
                    max_value=1e4,
                )
                loss = (((predictions - targets) ** 2) * weights.unsqueeze(-1)).mean()
                loss = self._sanitize_values(loss)
            if not torch.isfinite(loss):
                self.skipped_batches[f'advantage_{player}'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            grad_norm = self._optimizer_step(loss, optimizer, scaler, network)
            if grad_norm is None:
                self.skipped_batches[f'advantage_{player}'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            losses.append(float(loss.item()))

        network.eval()
        Config.clear_device_cache()
        return float(np.mean(losses)) if losses else 0.0

    def train_average_network(self, steps=None, batch_size=None) -> float:
        steps = Config.current_nn_train_steps() if steps is None else int(steps)
        batch_size = Config.current_nn_batch_size() if batch_size is None else int(batch_size)
        if len(self.strategy_memory) == 0 or steps <= 0:
            return 0.0

        self.average_network.train()
        losses = []
        for _ in range(steps):
            batch = self.strategy_memory.sample(batch_size, device=Config.DEVICE)
            if batch is None:
                break

            features, targets, weights = batch
            if not self._batch_is_finite(features, targets, weights):
                self.skipped_batches['strategy'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            weights = weights / weights.mean().clamp_min(1.0)
            targets = self._sanitize_values(targets, min_value=0.0, max_value=1.0)
            self.average_optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                logits = self._sanitize_values(
                    self.average_network(features).to(dtype=Config.NN_DTYPE),
                    min_value=-1e4,
                    max_value=1e4,
                )
                log_probs = F.log_softmax(logits, dim=-1)
                per_example_loss = -(targets * log_probs).sum(dim=-1)
                loss = self._sanitize_values((per_example_loss * weights).mean())
            if not torch.isfinite(loss):
                self.skipped_batches['strategy'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            grad_norm = self._optimizer_step(loss, self.average_optimizer, self.average_scaler, self.average_network)
            if grad_norm is None:
                self.skipped_batches['strategy'] += 1
                Config.clear_device_cache(aggressive=True)
                continue

            losses.append(float(loss.item()))

        self.average_network.eval()
        Config.clear_device_cache()
        return float(np.mean(losses)) if losses else 0.0

    def buffer_sizes(self):
        sizes = {f'advantage_{player}': len(memory) for player, memory in enumerate(self.advantage_memories)}
        sizes['strategy'] = len(self.strategy_memory)
        return sizes

    def export_average_strategies(self):
        raw_strategies = {}
        json_strategies = {}
        for key, infoset in list(self.visited_infosets.items()):
            strategy = self.average_strategy(infoset).detach().cpu().tolist()
            raw_strategies[key] = strategy
            json_strategies[json.dumps(key)] = strategy
        return raw_strategies, json_strategies

    def save_checkpoint(self, path: str, iteration: int):
        checkpoint = {
            'iteration': int(iteration),
            'config': {
                'algorithm_mode': Config.ALGORITHM_MODE,
                'infoset_key_mode': Config.INFOSET_KEY_MODE,
                'nn_dtype': str(Config.NN_DTYPE),
                'amp_dtype': str(Config.AMP_DTYPE),
                'amp_enabled': bool(Config.AMP_ENABLED),
            },
            'advantage_networks': [network.state_dict() for network in self.advantage_networks],
            'average_network': self.average_network.state_dict(),
            'advantage_optimizers': [optimizer.state_dict() for optimizer in self.advantage_optimizers],
            'average_optimizer': self.average_optimizer.state_dict(),
            'advantage_scalers': [scaler.state_dict() for scaler in self.advantage_scalers],
            'average_scaler': self.average_scaler.state_dict(),
            'advantage_memories': [memory.state_dict() for memory in self.advantage_memories],
            'strategy_memory': self.strategy_memory.state_dict(),
            'skipped_batches': dict(self.skipped_batches),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, map_location=None, load_optimizers: bool = True) -> int:
        load_kwargs = {'map_location': map_location or Config.checkpoint_map_location()}
        try:
            checkpoint = torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            checkpoint = torch.load(path, **load_kwargs)

        for network, state_dict in zip(self.advantage_networks, checkpoint.get('advantage_networks', [])):
            network.load_state_dict(state_dict)
            network.to(device=Config.DEVICE, dtype=Config.NN_DTYPE)

        self.average_network.load_state_dict(checkpoint['average_network'])
        self.average_network.to(device=Config.DEVICE, dtype=Config.NN_DTYPE)

        if load_optimizers:
            for optimizer, state_dict in zip(self.advantage_optimizers, checkpoint.get('advantage_optimizers', [])):
                optimizer.load_state_dict(state_dict)
                self._move_optimizer_state(optimizer)
            if 'average_optimizer' in checkpoint:
                self.average_optimizer.load_state_dict(checkpoint['average_optimizer'])
                self._move_optimizer_state(self.average_optimizer)

        for scaler, state_dict in zip(self.advantage_scalers, checkpoint.get('advantage_scalers', [])):
            scaler.load_state_dict(state_dict)
        if 'average_scaler' in checkpoint:
            self.average_scaler.load_state_dict(checkpoint['average_scaler'])

        for memory, state_dict in zip(self.advantage_memories, checkpoint.get('advantage_memories', [])):
            memory.load_state_dict(state_dict)
        if 'strategy_memory' in checkpoint:
            self.strategy_memory.load_state_dict(checkpoint['strategy_memory'])

        self.skipped_batches.update(checkpoint.get('skipped_batches', {}))
        Config.clear_device_cache(aggressive=True)
        return int(checkpoint.get('iteration', -1))