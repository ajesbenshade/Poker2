import os

from environment import setup_rocmo

setup_rocmo()

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)


class EquityNet(nn.Module):
    # Lightweight MLP used by the CFR pipeline and the standalone equity trainer.
    def __init__(self, input_dim=106, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.net(inputs)

class Config:
    SAFE_HARDWARE_MODE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.float32
    NN_DTYPE = torch.float32
    BUFFER_DTYPE = torch.float16
    AMP_ENABLED = False
    AMP_DTYPE = torch.float32
    NUM_BUCKETS = 2048
    NUM_SIMULATIONS = 65536
    NUM_SIMS = NUM_SIMULATIONS
    ITERATIONS = 5000
    SAMPLING_RATE = 0.5
    DISCOUNT = 0.99
    NUM_ACTIONS = 3
    NUM_OPPONENTS = 1
    BATCH_SIZE = 2048
    MAX_BATCH_SIZE = 8192
    MIN_BATCH_SIZE = 512
    NN_HIDDEN_SIZE = 1024
    MODEL_HIDDEN_DIM = NN_HIDDEN_SIZE
    MODEL_DEPTH = 4
    MODEL_DROPOUT = 0.0
    DEEP_CFR_FEATURE_DIM = 16
    DEEP_CFR_TRAVERSALS_PER_ITER = 512
    NUM_TRAVERSALS = DEEP_CFR_TRAVERSALS_PER_ITER
    MAX_NUM_TRAVERSALS = 4096
    MIN_NUM_TRAVERSALS = 128
    ADVANTAGE_TRAIN_STEPS = 4
    STRATEGY_TRAIN_STEPS = 2
    REPLAY_BUFFER_SIZE = 2_000_000
    REPLAY_WARMUP_SAMPLES = 4096
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM = 5.0
    LOSS_CLAMP = 25.0
    UTILITY_CLAMP = 4.0
    POT_SIZE = 100.0
    CALL_AMOUNT = 20.0
    RAISE_MULTIPLIER = 3.0
    FOLD_EQUITY_MEAN = 0.4
    FOLD_EQUITY_STD = 0.3
    EQUITY_STD = 0.1
    EQUITY_ROLLOUTS = 32
    BLUFF_FACTOR = 0.3
    FOLD_PENALTY = 0.5

    RAY_NUM_CPUS = min(22, max(20, os.cpu_count() or 24))
    RAY_WORKER_LIMIT = min(22, RAY_NUM_CPUS)
    RAY_TASK_BATCH = 256
    MIN_RAY_TASK_BATCH = 32

    MP_PROCESSES = min(12, max(4, (os.cpu_count() or 24) // 2))

    START_MAX_DEPTH = 2
    MAX_CFR_DEPTH = 6
    CURRICULUM_INTERVAL = 5000
    MAX_CURRICULUM_OPPONENTS = 4

    VRAM_SOFT_LIMIT_GB = 15.5
    RAM_SOFT_LIMIT_PCT = 78.0
    LOG_INTERVAL = 25
    MIN_EQUITY_ROLLOUTS = 8
    BACKOFF_COOLDOWN = 5
    CACHE_CLEAR_INTERVAL = 10

    CHECKPOINT_INTERVAL = 100
    CHECKPOINT_DIR = 'checkpoints'
    LATEST_CHECKPOINT_NAME = 'checkpoint_latest.pt'
    BEST_CHECKPOINT_NAME = 'checkpoint_best.pt'
    STORAGE_DIR = 'storage'
    ABSTRACTION_CACHE_DIR = os.path.join(STORAGE_DIR, 'abstractions')

    HYBRID_UPDATE_INTERVAL = 25000
    HYBRID_BOOST_WEIGHT = 0.05

    SMOKE_TEST_ITERATIONS = 2
    SMOKE_TEST_NUM_SIMS = 2048
    SMOKE_TEST_NUM_BUCKETS = 32
    SMOKE_TEST_BATCH_SIZE = 256
    SMOKE_TEST_TRAVERSALS = 64
    SMOKE_TEST_REPLAY_WARMUP = 64
    SMOKE_TEST_REPLAY_BUFFER_SIZE = 16384
    SMOKE_TEST_LOG_INTERVAL = 1
    SMOKE_TEST_CHECKPOINT_INTERVAL = 1
    SMOKE_TEST_EQUITY_ROLLOUTS = 4

    LONG_RUN_ITERATIONS = 250000

    USE_LMDB_TABLES = False
    TABLE_STORAGE_PATH = 'cfr_tables.lmdb'
    TABLE_MAP_SIZE_BYTES = 16 * 1024 ** 3

    EQUITY_FEATURE_DIM = 106
    EQUITY_HIDDEN_DIM = 256
    EQUITY_MODEL = EquityNet(EQUITY_FEATURE_DIM, EQUITY_HIDDEN_DIM).to(DEVICE)


if os.path.exists('best_equity_model.pth'):
    try:
        Config.EQUITY_MODEL.load_state_dict(torch.load('best_equity_model.pth', map_location=Config.DEVICE, weights_only=True))
    except Exception:
        pass