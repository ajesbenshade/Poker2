import gc
import os

import torch


_ALLOCATOR_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:128"

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HIP_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11_0_0")
os.environ.setdefault("PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF))


def _cuda_available():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _bf16_supported(has_cuda):
    if not has_cuda:
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


class Config:
    HAS_CUDA = _cuda_available()
    IS_HIP = bool(getattr(torch.version, 'hip', None))
    DEVICE = 'cuda' if HAS_CUDA else 'cpu'
    SIMULATION_DEVICE = 'cpu'
    BF16_SUPPORTED = _bf16_supported(HAS_CUDA)
    DTYPE = torch.float32
    STORAGE_DTYPE = torch.float32
    NN_DTYPE = torch.float32
    AMP_ENABLED = HAS_CUDA
    AMP_DTYPE = torch.bfloat16 if BF16_SUPPORTED else torch.float16

    SAFE_HARDWARE_MODE = True
    SAFE_MODE = True
    SAVE_BEST_MODEL = True
    RUN_UNTIL_STOP = False

    MAX_VRAM_BEFORE_BACKOFF_GB = 15.5
    MAX_RAM_UTILIZATION_PERCENT = 78.0
    MIN_SIM_BATCH_SIZE = 256
    MIN_EQUITY_ROLLOUTS = 4
    MIN_NN_BATCH_SIZE = 128
    MIN_NN_TRAIN_STEPS = 1
    MIN_DEEP_TRAVERSALS_PER_ITER = 1
    MAX_VISITED_INFOSETS = 16384

    TORCH_THREADS_MAIN = max(1, min(8, (os.cpu_count() or 8) // 2))
    TORCH_THREADS_PER_WORKER = 1
    GRADIENT_CHECKPOINTING = True
    CHECKPOINT_SEGMENTS = 2

    SEED = 42
    ALGORITHM_MODE = 'tabular'
    ENVIRONMENT_MODE = 'simplified'
    INFOSET_KEY_MODE = 'legacy'

    RAY_NUM_CPUS = 8
    RAY_NUM_GPUS = 1 if HAS_CUDA else 0
    HAND_EVAL_PROCESSES = 4
    LOG_INTERVAL = 25
    CHECKPOINT_INTERVAL = 500
    EVAL_INTERVAL = 1000
    MAX_DEPTH = 2

    NUM_BUCKETS = 1024
    NUM_SIMS = 4096
    ITERATIONS = 100000
    SAMPLING_RATE = 0.5
    DISCOUNT = 0.99
    NUM_ACTIONS = 3
    NUM_OPPONENTS = 1
    BATCH_SIZE = 4096
    MAX_BATCH_SIZE = 8192

    INITIAL_STACK = 1000.0
    POT_SIZE = 100.0
    CALL_AMOUNT = 20.0
    DEFAULT_STACK_SIZES = (INITIAL_STACK, INITIAL_STACK)

    RAISE_MULTIPLIER = 3.0
    FOLD_EQUITY_MEAN = 0.4
    FOLD_EQUITY_STD = 0.2
    EQUITY_STD = 0.02
    EQUITY_ROLLOUTS = 16
    BLUFF_FACTOR = 0.2
    FOLD_PENALTY = 0.5

    HISTORY_FEATURES = 16
    CARD_FEATURES = 52
    MODEL_HIDDEN_DIM = 2048
    MODEL_NUM_LAYERS = 2
    REPLAY_BUFFER_SIZE = 1000000
    REPLAY_BUFFER_INITIAL_CAPACITY = 65536
    REPLAY_BUFFER_GROWTH_FACTOR = 2
    NN_BATCH_SIZE = 2048
    MAX_NN_BATCH_SIZE = 8192
    NN_TRAIN_STEPS = 16
    NN_LEARNING_RATE = 3e-4
    DEEP_CFR_TRAVERSALS_PER_ITER = 1

    _runtime_batch_size = None
    _runtime_equity_rollouts = None
    _runtime_nn_batch_size = None
    _runtime_nn_train_steps = None
    _runtime_deep_traversals_per_iter = None

    @classmethod
    def autocast_device_type(cls):
        return 'cuda' if cls.HAS_CUDA else 'cpu'

    @classmethod
    def checkpoint_map_location(cls):
        return torch.device(cls.DEVICE)

    @classmethod
    def scaler_enabled(cls):
        return cls.AMP_ENABLED and cls.AMP_DTYPE == torch.float16

    @classmethod
    def current_batch_size(cls):
        return cls.BATCH_SIZE if cls._runtime_batch_size is None else cls._runtime_batch_size

    @classmethod
    def current_equity_rollouts(cls):
        return cls.EQUITY_ROLLOUTS if cls._runtime_equity_rollouts is None else cls._runtime_equity_rollouts

    @classmethod
    def current_nn_batch_size(cls):
        return cls.NN_BATCH_SIZE if cls._runtime_nn_batch_size is None else cls._runtime_nn_batch_size

    @classmethod
    def current_nn_train_steps(cls):
        return cls.NN_TRAIN_STEPS if cls._runtime_nn_train_steps is None else cls._runtime_nn_train_steps

    @classmethod
    def current_deep_traversals_per_iter(cls):
        if cls._runtime_deep_traversals_per_iter is None:
            return cls.DEEP_CFR_TRAVERSALS_PER_ITER
        return cls._runtime_deep_traversals_per_iter

    @classmethod
    def apply_runtime_limits(
        cls,
        *,
        batch_size=None,
        equity_rollouts=None,
        nn_batch_size=None,
        nn_train_steps=None,
        deep_traversals_per_iter=None,
    ):
        cls._runtime_batch_size = batch_size
        cls._runtime_equity_rollouts = equity_rollouts
        cls._runtime_nn_batch_size = nn_batch_size
        cls._runtime_nn_train_steps = nn_train_steps
        cls._runtime_deep_traversals_per_iter = deep_traversals_per_iter

    @classmethod
    def reset_runtime_limits(cls):
        cls.apply_runtime_limits()

    @classmethod
    def clear_device_cache(cls, aggressive=False):
        gc.collect()
        if not cls.HAS_CUDA:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if aggressive:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
