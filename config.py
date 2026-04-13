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
    # Runtime and hardware profile.
    SAFE_HARDWARE_MODE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.float32
    NN_DTYPE = torch.float32
    AMP_ENABLED = True
    AMP_DTYPE = torch.float16
    AMP_BF16_OPT_IN = False
    USE_TORCH_COMPILE = False

    # Thread safety on ROCm + multiprocessing.
    OMP_NUM_THREADS = 1
    TORCH_NUM_THREADS = 1

    # PPO core.
    NUM_ACTIONS = 3
    STATE_DIM = 169
    MODEL_HIDDEN_DIM = 1024
    MODEL_DEPTH = 4
    MODEL_DROPOUT = 0.05
    GRADIENT_CHECKPOINTING = True
    MAX_GRAD_NORM = 5.0
    PPO_EPOCHS = 6
    PPO_MINIBATCHES = 8
    CLIP_EPS = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.015
    GAE_LAMBDA = 0.95
    GAMMA = 0.995
    LEARNING_RATE = 2.5e-4
    WEIGHT_DECAY = 1e-4

    # Throughput and simulation.
    BATCH_SIZE = 16384
    BATCH_SIZE_LADDER = (16384, 12288, 8192, 6144, 4096)
    NUM_SIMULATIONS = 512
    MAX_NUM_SIMULATIONS = 768
    ROLLOUT_STEPS = 256
    ITERATIONS = 100000
    LOG_INTERVAL = 10

    # Curriculum and population training.
    START_OPPONENTS = 2
    TARGET_OPPONENTS = 6
    NUM_OPPONENTS = START_OPPONENTS
    CURRICULUM_INTERVAL = 2000
    POPULATION_SIZE = 6
    VALIDATION_INTERVAL = 250
    PBT_MUTATION_SCALE = 0.15
    ELO_K_FACTOR = 24.0

    # Hybrid action logic.
    HYBRID_EQUITY_WEIGHT = 0.35
    MCTS_EQUITY_THRESHOLD = 0.6
    MCTS_TRIGGER_PROB = 0.75
    MCTS_MAX_DEPTH = 3
    PREFLOP_CHART_ENABLED = True

    # Game simulation constants.
    POT_SIZE = 100.0
    CALL_AMOUNT = 20.0
    RAISE_MULTIPLIER = 3.0
    FOLD_EQUITY_MEAN = 0.4
    FOLD_EQUITY_STD = 0.3
    EQUITY_STD = 0.1
    EQUITY_ROLLOUTS = 32
    BLUFF_FACTOR = 0.2
    FOLD_PENALTY = 0.5
    UTILITY_CLAMP = 4.0

    # Memory guardrails for this 7900 XT rig.
    VRAM_SOFT_LIMIT_GB = 18.0
    RAM_SOFT_LIMIT_PCT = 82.0
    RECOVERY_VRAM_PCT = 0.72
    RECOVERY_RAM_MARGIN = 8.0

    # Storage and checkpointing.
    CHECKPOINT_INTERVAL = 100
    CHECKPOINT_DIR = 'checkpoints'
    LATEST_CHECKPOINT_NAME = 'checkpoint_latest.pt'
    BEST_CHECKPOINT_NAME = 'checkpoint_best.pt'
    STORAGE_DIR = 'storage'
    REPLAY_BUFFER_SIZE = 5_000_000
    REPLAY_WARMUP_SAMPLES = 4096

    # Multiprocessing and workers.
    MP_PROCESSES = min(24, max(20, os.cpu_count() or 24))

    # Smoke-test profile.
    SMOKE_TEST_ITERATIONS = 4
    SMOKE_TEST_BATCH_SIZE = 4096
    SMOKE_TEST_NUM_SIMULATIONS = 128
    SMOKE_TEST_ROLLOUT_STEPS = 64
    SMOKE_TEST_VALIDATION_INTERVAL = 2
    SMOKE_TEST_LOG_INTERVAL = 1
    SMOKE_TEST_CHECKPOINT_INTERVAL = 2

    # Optional long unattended run profile.
    LONG_RUN_ITERATIONS = 500000

    EQUITY_FEATURE_DIM = 106
    EQUITY_HIDDEN_DIM = 256
    EQUITY_MODEL = EquityNet(EQUITY_FEATURE_DIM, EQUITY_HIDDEN_DIM).to(DEVICE)


if os.path.exists('best_equity_model.pth'):
    try:
        Config.EQUITY_MODEL.load_state_dict(torch.load('best_equity_model.pth', map_location=Config.DEVICE, weights_only=True))
    except Exception:
        pass