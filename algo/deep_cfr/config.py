"""Configuration for Deep CFR training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from engine.actions import DEFAULT_BET_FRACTIONS


@dataclass
class DeepCFRConfig:
    # Game
    num_players: int = 2
    starting_stack: int = 200          # 100 BB
    small_blind: int = 1
    big_blind: int = 2
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS

    # CFR loop
    num_iterations: int = 200
    traversals_per_iter: int = 1500    # external-sampling traversals per player per CFR iter
    linear_cfr: bool = True            # weight regret samples by iter t

    # Network
    hidden_size: int = 256
    num_blocks: int = 4
    dropout: float = 0.0

    # Buffers
    advantage_buffer_size: int = 1_000_000
    strategy_buffer_size: int = 2_000_000

    # Network training
    advantage_train_steps: int = 4000
    strategy_train_steps: int = 8000
    train_batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    reset_advantage_net_each_iter: bool = True   # standard Deep CFR resets V each iter
    train_strategy_every: int = 1                # train avg policy every N iters

    # Eval
    eval_interval: int = 5
    eval_hands: int = 2000

    # LBR exploitability eval (Phase G)
    lbr_interval: int = 0              # 0 disables; otherwise compute LBR every N iters
    lbr_hands: int = 1000
    lbr_equity_samples: int = 100

    # Runtime
    device: str = "cuda"
    amp_dtype: str = "bfloat16"        # "bfloat16" | "float16" | "float32"
    seed: int = 0

    # Checkpointing
    checkpoint_dir: str = "checkpoints/deep_cfr"
    log_dir: str = "runs/deep_cfr"
    log_interval: int = 1

    # Parallel traversal (Phase A)
    num_workers: int = 0               # 0 => serial (legacy path); >0 => process pool
    worker_chunk_min: int = 8          # min traversals per dispatched chunk
    use_torch_compile: bool = False    # apply torch.compile to GPU train nets

    # Async pipeline (Phase B): overlap CPU traversals with GPU training.
    # Iter t's traversals are dispatched while iter t-1's GPU training runs.
    # Requires num_workers > 0. Adds 1-iter staleness to traversal nets.
    async_pipeline: bool = False
