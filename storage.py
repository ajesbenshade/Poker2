from collections import defaultdict

import torch

from config import Config

try:
    import ray
except ImportError:
    ray = None


NODE_REGRET_INDEX = 0
NODE_STRATEGY_INDEX = 1


def _empty_node():
    return torch.zeros((2, Config.NUM_ACTIONS), dtype=Config.STORAGE_DTYPE, device='cpu')


class LocalNodeStore:
    def __init__(self, device=None):
        self.device = Config.DEVICE if device is None else device
        self.nodes = defaultdict(_empty_node)

    def _read(self, key, index):
        return self.nodes[key][index].to(device=self.device, dtype=Config.DTYPE)

    def get_regret_sum(self, key):
        return self._read(key, NODE_REGRET_INDEX)

    def get_strategy_sum(self, key):
        return self._read(key, NODE_STRATEGY_INDEX)

    def update_regret_sum(self, key, delta):
        self.nodes[key][NODE_REGRET_INDEX].add_(
            delta.detach().to(device='cpu', dtype=Config.STORAGE_DTYPE)
        )

    def update_strategy_sum(self, key, delta):
        self.nodes[key][NODE_STRATEGY_INDEX].add_(
            delta.detach().to(device='cpu', dtype=Config.STORAGE_DTYPE)
        )

    def get_all_keys(self):
        return list(self.nodes.keys())


class RemoteNodeStore:
    def __init__(self, actor, device=None):
        if ray is None:
            raise RuntimeError("Ray is not installed; remote node storage is unavailable.")
        self.actor = actor
        self.device = Config.DEVICE if device is None else device

    def get_regret_sum(self, key):
        return ray.get(self.actor.get_regret_sum.remote(key)).to(device=self.device, dtype=Config.DTYPE)

    def get_strategy_sum(self, key):
        return ray.get(self.actor.get_strategy_sum.remote(key)).to(device=self.device, dtype=Config.DTYPE)

    def update_regret_sum(self, key, delta):
        self.actor.update_regret_sum.remote(
            key,
            delta.detach().to(device='cpu', dtype=Config.STORAGE_DTYPE),
        )

    def update_strategy_sum(self, key, delta):
        self.actor.update_strategy_sum.remote(
            key,
            delta.detach().to(device='cpu', dtype=Config.STORAGE_DTYPE),
        )

    def get_all_keys(self):
        return ray.get(self.actor.get_all_keys.remote())


def coerce_node_store(node_store=None, actor=None, default_store=None):
    if node_store is not None:
        return node_store
    if actor is not None:
        return RemoteNodeStore(actor)
    return default_store if default_store is not None else LocalNodeStore()