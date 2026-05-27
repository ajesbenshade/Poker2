"""Configuration for Deep CFR training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

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
    # Discounted CFR exponents (Brown et al. 2019). Powers are applied to
    # iter_t to compute the per-sample weight when linear_cfr=True.
    # alpha => advantage (regret) sample weight = t**alpha.
    # gamma => strategy sample weight = t**gamma. Setting gamma=2.0 is the
    # standard "DCFR" recipe and is essentially free CPU.
    discounted_cfr_alpha: float = 1.0
    discounted_cfr_gamma: float = 2.0
    # CFR+ regret clipping for the advantage net training target. When True,
    # the regression target is clamped to be non-negative before computing
    # MSE. Conservative (default off) but a known stabilizer.
    cfr_plus: bool = False

    # Network
    hidden_size: int = 256
    num_blocks: int = 4
    dropout: float = 0.0

    # Buffers
    advantage_buffer_size: int = 3_000_000
    strategy_buffer_size: int = 8_000_000

    # Network training
    advantage_train_steps: int = 4000
    strategy_train_steps: int = 8000
    train_batch_size: int = 4096
    learning_rate: float = 1e-3
    advantage_learning_rate: Optional[float] = None
    strategy_learning_rate: Optional[float] = None
    lr_schedule: str = "constant"       # "constant" | "cosine" | "one_cycle"
    lr_min_mult: float = 0.1
    lr_warmup_frac: float = 0.0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    reset_advantage_net_each_iter: bool = True   # standard Deep CFR resets V each iter
    train_strategy_every: int = 1                # train avg policy every N iters
    pin_training_batches: bool = False           # pin sampled CPU batches before GPU transfer
    loss_log_interval: int = 1                   # sample loss every N steps (1=exact per-step avg)
    concurrent_advantage_training: bool = False  # train per-seat advantage nets on CUDA streams
    # Early-stop advantage training when held-out validation loss plateaus.
    # 0 disables (run the full advantage_train_steps). >0 stops after this many
    # consecutive evaluations with no improvement. Saves time AND reduces
    # advantage-net overfit on stale buffer samples.
    adv_early_stop_patience: int = 0
    adv_early_stop_eval_every: int = 200         # steps between val evaluations
    adv_early_stop_min_steps: int = 500          # never stop before this many steps
    adv_early_stop_val_fraction: float = 0.05    # held-out fraction of training batch

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
    latest_checkpoint_interval: int = 1          # save latest.pt every N iters (1=every iter)
    init_checkpoint: Optional[str] = None         # load weights only; train for num_iterations
    resume_checkpoint: Optional[str] = None       # load iter/buffers when present; continue to num_iterations
    save_buffer_state: bool = False              # can make checkpoints many GB for large buffers

    # Parallel traversal (Phase A)
    # RX 7900 XT ROCm guidance: keep 0 for serial smoke tests, use 8 workers
    # for first multiprocessing validation, 10-12 for stable long runs, and
    # 14-16 only for aggressive runs while monitoring RAM/VRAM.
    num_workers: int = 0               # 0 => serial (legacy path); >0 => process pool
    worker_chunk_min: int = 0          # 0=auto; >0 target traversals per dispatched chunk
    min_tasks_per_worker: int = 4      # When auto chunking, aim for at least this many tasks per worker for load balancing (post fast-sim + sharedmem era)
    worker_torch_threads: int = 1      # per-worker torch CPU thread count
    script_worker_nets: bool = False   # torch.jit.script CPU worker advantage nets
    traversal_backend: str = "recursive"  # "recursive" | "vectorized"
    vectorized_traversal_batch_size: int = 128  # Inner batch size for vectorized traversal (larger is often better after fast simulator)
    worker_result_transport: str = "ipc"  # "ipc" | "file" | "sharedmem" (recommended for local multi-worker)
    use_proxy_nets: bool = False       # distill smaller traversal-only advantage nets
    proxy_hidden_size: int = 128
    proxy_num_blocks: int = 2
    proxy_refresh_interval: int = 5
    proxy_training_steps: int = 2000
    proxy_distill_strategy_weight: float = 0.4   # Weight on strategy (regret-matching) matching loss vs pure advantage MSE during proxy distillation. 0 = advantage only, higher = better strategy fidelity for traversal.
    traversal_inference_mode: str = "worker_cpu"  # "worker_cpu" | "server"
    inference_server_batch_size: int = 128
    inference_server_timeout_ms: float = 2.0
    inference_server_queue_size: int = 4096
    use_torch_compile: bool = False    # apply torch.compile to GPU train nets

    # Async pipeline (Phase B): overlap CPU traversals with GPU training.
    # Iter t's traversals are dispatched while iter t-1's GPU training runs.
    # Requires num_workers > 0. Adds 1-iter staleness to traversal nets.
    async_pipeline: bool = False
    async_pipeline_depth: int = 1       # max traversal batches queued ahead when async_pipeline is on
