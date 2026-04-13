import csv
import os
from dataclasses import dataclass

import numpy as np
import torch

from environment import clear_runtime_caches, get_memory_snapshot


@dataclass
class BatchAdaptationResult:
    batch_size: int
    simulations: int
    reason: str


def select_amp_dtype(config):
    prefer_bf16 = getattr(config, "AMP_PREFER_BF16", False) or getattr(config, "AMP_BF16_OPT_IN", False)
    if prefer_bf16 and torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def ensure_numpy_float32(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def adapt_batch_and_sims(config, current_batch_size, current_simulations):
    snapshot = get_memory_snapshot()
    over_vram = snapshot["used_gb"] > config.VRAM_SOFT_LIMIT_GB
    over_ram = snapshot["ram_pct"] > config.RAM_SOFT_LIMIT_PCT

    if over_vram or over_ram:
        ladder = list(config.BATCH_SIZE_LADDER)
        if current_batch_size in ladder:
            idx = ladder.index(current_batch_size)
            new_batch = ladder[min(idx + 1, len(ladder) - 1)]
        else:
            new_batch = max(ladder[-1], current_batch_size // 2)
        new_sims = max(256, int(current_simulations * 0.75))
        clear_runtime_caches()
        return BatchAdaptationResult(new_batch, new_sims, "memory-pressure")

    recovery_vram = config.VRAM_SOFT_LIMIT_GB * config.RECOVERY_VRAM_PCT
    recovery_ram = config.RAM_SOFT_LIMIT_PCT - config.RECOVERY_RAM_MARGIN
    if snapshot["used_gb"] < recovery_vram and snapshot["ram_pct"] < recovery_ram:
        ladder = list(config.BATCH_SIZE_LADDER)
        if current_batch_size in ladder:
            idx = ladder.index(current_batch_size)
            new_batch = ladder[max(0, idx - 1)]
        else:
            new_batch = min(ladder[0], current_batch_size * 2)
        new_sims = min(config.MAX_NUM_SIMULATIONS, int(current_simulations * 1.1))
        reason = "recovered" if (new_batch != current_batch_size or new_sims != current_simulations) else "stable"
        return BatchAdaptationResult(new_batch, new_sims, reason)

    return BatchAdaptationResult(current_batch_size, current_simulations, "stable")


class CsvMetricLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._fields = [
            "iteration",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "avg_reward",
            "elo",
            "ram_pct",
            "vram_used_gb",
            "vram_pct",
            "batch_size",
            "simulations",
            "num_opponents",
            "mcts_rate",
        ]
        if not os.path.exists(output_path):
            with open(output_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fields)
                writer.writeheader()

    def log(self, row):
        with open(self.output_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fields)
            writer.writerow(row)
