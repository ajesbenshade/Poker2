import os

# Keep ROCm environment defaults close to the configuration so standalone scripts
# inherit the same AMD-friendly allocator and device settings.
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:256')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn


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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DTYPE = torch.bfloat16
    NUM_BUCKETS = 5000
    # Raise abstraction sampling so the 7900X stays busier during feature generation.
    NUM_SIMULATIONS = 200000
    NUM_SIMS = NUM_SIMULATIONS
    # Extend long-running CFR training slightly for stronger convergence on the target rig.
    ITERATIONS = 1250000
    SAMPLING_RATE = 0.5
    DISCOUNT = 0.99
    NUM_ACTIONS = 3
    NUM_OPPONENTS = 1
    # Start with a larger GPU-friendly batch and shrink dynamically under memory pressure.
    BATCH_SIZE = 2048
    MAX_BATCH_SIZE = 2048
    MIN_BATCH_SIZE = 256
    POT_SIZE = 100.0
    CALL_AMOUNT = 20.0
    RAISE_MULTIPLIER = 3.0
    FOLD_EQUITY_MEAN = 0.4
    FOLD_EQUITY_STD = 0.3
    EQUITY_STD = 0.1
    # Increase rollout count for stronger equity labels while still fitting the 7900XT.
    EQUITY_ROLLOUTS = 64
    BLUFF_FACTOR = 0.3
    FOLD_PENALTY = 0.5

    # Cap Ray around 22 workers so the 7900X still has headroom for ROCm/runtime threads.
    RAY_NUM_CPUS = min(22, max(20, os.cpu_count() or 24))
    RAY_WORKER_LIMIT = min(22, RAY_NUM_CPUS)
    RAY_TASK_BATCH = 256
    MIN_RAY_TASK_BATCH = 128

    # Reuse a capped multiprocessing pool inside equity evaluation to stay cache-friendly.
    MP_PROCESSES = min(12, max(4, (os.cpu_count() or 24) // 2))

    # Curriculum settings gradually increase traversal complexity during long runs.
    START_MAX_DEPTH = 2
    MAX_CFR_DEPTH = 6
    CURRICULUM_INTERVAL = 5000
    MAX_CURRICULUM_OPPONENTS = 4

    # Memory pressure thresholds keep the 7900XT under ~16 GB VRAM and host RAM under 80%.
    VRAM_SOFT_LIMIT_GB = 16.0
    RAM_SOFT_LIMIT_PCT = 80.0
    LOG_INTERVAL = 250
    MIN_EQUITY_ROLLOUTS = 8

    # Save periodic recovery checkpoints so long runs can resume from recent strategy snapshots.
    CHECKPOINT_INTERVAL = 1000
    CHECKPOINT_DIR = 'checkpoints'

    # Lightweight hybrid regret boost cadence for periodic strategy stabilization.
    HYBRID_UPDATE_INTERVAL = 25000
    HYBRID_BOOST_WEIGHT = 0.05

    # Allow small smoke-test runs without editing source files.
    TEST_ITERATIONS = 1
    TEST_NUM_SIMS = 128
    TEST_NUM_BUCKETS = 32

    # Optional future disk-backed table storage for large regret tables under 64 GB RAM.
    USE_LMDB_TABLES = False
    TABLE_STORAGE_PATH = 'cfr_tables.lmdb'
    TABLE_MAP_SIZE_BYTES = 16 * 1024 ** 3

    # Shared equity model configuration for CFR rollouts and standalone equity training.
    EQUITY_FEATURE_DIM = 106
    EQUITY_HIDDEN_DIM = 256
    EQUITY_MODEL = EquityNet(EQUITY_FEATURE_DIM, EQUITY_HIDDEN_DIM).to(DEVICE)


if os.path.exists('best_equity_model.pth'):
    # Automatically reload the latest trained equity weights for production runs.
    Config.EQUITY_MODEL.load_state_dict(torch.load('best_equity_model.pth', map_location=Config.DEVICE, weights_only=True))