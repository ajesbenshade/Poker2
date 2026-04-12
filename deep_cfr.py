import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.entries = []
        self.total_seen = 0

    def add(self, infoset, targets, weight=1.0):
        target_array = np.asarray(targets, dtype=np.float32)
        if target_array.shape != (Config.NUM_ACTIONS,):
            raise ValueError(f"Expected targets shaped {(Config.NUM_ACTIONS,)}, got {target_array.shape}")

        sample = {
            'infoset': infoset,
            'key': infoset.key,
            'features': encode_infoset(infoset),
            'targets': np.nan_to_num(target_array, nan=0.0, posinf=0.0, neginf=0.0),
            'weight': max(float(weight), 1e-6),
        }

        self.total_seen += 1
        if len(self.entries) < self.capacity:
            self.entries.append(sample)
            return

        replace_idx = random.randrange(self.total_seen)
        if replace_idx < self.capacity:
            self.entries[replace_idx] = sample

    def sample(self, batch_size: int, device=Config.DEVICE):
        if not self.entries:
            return None

        batch_size = min(int(batch_size), len(self.entries))
        indices = np.random.choice(len(self.entries), size=batch_size, replace=False)
        batch = [self.entries[idx] for idx in indices]

        features = torch.tensor(
            np.stack([entry['features'] for entry in batch]),
            device=device,
            dtype=Config.NN_DTYPE,
        )
        targets = torch.tensor(
            np.stack([entry['targets'] for entry in batch]),
            device=device,
            dtype=Config.NN_DTYPE,
        )
        weights = torch.tensor(
            [entry['weight'] for entry in batch],
            device=device,
            dtype=Config.NN_DTYPE,
        )
        return features, targets, weights

    def __len__(self):
        return len(self.entries)


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

        self.advantage_memories = [ReplayBuffer(Config.REPLAY_BUFFER_SIZE) for _ in range(player_count)]
        self.strategy_memory = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
        self.visited_infosets = {}

    def register_infoset(self, infoset):
        self.visited_infosets[infoset.key] = infoset

    def _encode_infoset(self, infoset):
        encoded = torch.tensor(
            encode_infoset(infoset),
            device=Config.DEVICE,
            dtype=Config.NN_DTYPE,
        )
        return encoded.unsqueeze(0)

    def advantage_strategy(self, infoset, player: int) -> torch.Tensor:
        self.register_infoset(infoset)
        with torch.no_grad():
            advantages = self.advantage_networks[player](self._encode_infoset(infoset)).squeeze(0)
        return regret_matching(advantages)

    def average_strategy(self, infoset) -> torch.Tensor:
        self.register_infoset(infoset)
        with torch.no_grad():
            logits = self.average_network(self._encode_infoset(infoset)).squeeze(0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return F.softmax(logits, dim=-1)

    def record_advantage(self, infoset, player: int, advantages, weight=1.0):
        self.register_infoset(infoset)
        if isinstance(advantages, torch.Tensor):
            advantages = advantages.detach().cpu().numpy()
        self.advantage_memories[player].add(infoset, advantages, weight=weight)

    def record_strategy(self, infoset, strategy, weight=1.0):
        self.register_infoset(infoset)
        if isinstance(strategy, torch.Tensor):
            strategy = strategy.detach().cpu().numpy()
        self.strategy_memory.add(infoset, strategy, weight=weight)

    def train_advantage_network(self, player: int, steps=None, batch_size=None) -> float:
        steps = Config.NN_TRAIN_STEPS if steps is None else int(steps)
        batch_size = Config.NN_BATCH_SIZE if batch_size is None else int(batch_size)
        if len(self.advantage_memories[player]) == 0 or steps <= 0:
            return 0.0

        network = self.advantage_networks[player]
        optimizer = self.advantage_optimizers[player]
        losses = []

        network.train()
        for _ in range(steps):
            batch = self.advantage_memories[player].sample(batch_size, device=Config.DEVICE)
            if batch is None:
                break

            features, targets, weights = batch
            weights = weights / weights.mean().clamp_min(1.0)
            predictions = network(features)
            loss = (((predictions - targets) ** 2) * weights.unsqueeze(-1)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else 0.0

    def train_average_network(self, steps=None, batch_size=None) -> float:
        steps = Config.NN_TRAIN_STEPS if steps is None else int(steps)
        batch_size = Config.NN_BATCH_SIZE if batch_size is None else int(batch_size)
        if len(self.strategy_memory) == 0 or steps <= 0:
            return 0.0

        self.average_network.train()
        losses = []
        for _ in range(steps):
            batch = self.strategy_memory.sample(batch_size, device=Config.DEVICE)
            if batch is None:
                break

            features, targets, weights = batch
            weights = weights / weights.mean().clamp_min(1.0)
            logits = self.average_network(features)
            log_probs = F.log_softmax(logits, dim=-1)
            per_example_loss = -(targets * log_probs).sum(dim=-1)
            loss = (per_example_loss * weights).mean()

            self.average_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.average_network.parameters(), max_norm=5.0)
            self.average_optimizer.step()

            losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else 0.0

    def buffer_sizes(self):
        sizes = {f'advantage_{player}': len(memory) for player, memory in enumerate(self.advantage_memories)}
        sizes['strategy'] = len(self.strategy_memory)
        return sizes

    def export_average_strategies(self):
        raw_strategies = {}
        json_strategies = {}
        for key, infoset in self.visited_infosets.items():
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
            },
            'advantage_networks': [network.state_dict() for network in self.advantage_networks],
            'average_network': self.average_network.state_dict(),
            'advantage_optimizers': [optimizer.state_dict() for optimizer in self.advantage_optimizers],
            'average_optimizer': self.average_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)