from collections import defaultdict

import torch

from config import Config

try:
    import ray
except ImportError:
    ray = None


def _empty_node():
    return {
        'regret_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu'),
        'strategy_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu'),
    }


class LocalNodeStore:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.nodes = defaultdict(_empty_node)

    def get_regret_sum(self, key):
        return self.nodes[key]['regret_sum'].to(self.device)

    def get_strategy_sum(self, key):
        return self.nodes[key]['strategy_sum'].to(self.device)

    def update_regret_sum(self, key, delta):
        self.nodes[key]['regret_sum'] += delta.detach().cpu()

    def update_strategy_sum(self, key, delta):
        self.nodes[key]['strategy_sum'] += delta.detach().cpu()

    def get_all_keys(self):
        return list(self.nodes.keys())


class RemoteNodeStore:
    def __init__(self, actor, device=Config.DEVICE):
        if ray is None:
            raise RuntimeError("Ray is not installed; remote node storage is unavailable.")
        self.actor = actor
        self.device = device

    def get_regret_sum(self, key):
        return ray.get(self.actor.get_regret_sum.remote(key)).to(self.device)

    def get_strategy_sum(self, key):
        return ray.get(self.actor.get_strategy_sum.remote(key)).to(self.device)

    def update_regret_sum(self, key, delta):
        self.actor.update_regret_sum.remote(key, delta.detach().cpu())

    def update_strategy_sum(self, key, delta):
        self.actor.update_strategy_sum.remote(key, delta.detach().cpu())

    def get_all_keys(self):
        return ray.get(self.actor.get_all_keys.remote())


def coerce_node_store(node_store=None, actor=None, default_store=None):
    if node_store is not None:
        return node_store
    if actor is not None:
        return RemoteNodeStore(actor)
    return default_store if default_store is not None else LocalNodeStore()