import argparse
import gc
import logging
import multiprocessing as mp
import os
import subprocess
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512",
)
os.environ.setdefault(
    "PYTORCH_HIP_ALLOC_CONF",
    os.environ["PYTORCH_ALLOC_CONF"],
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_HIP_ALLOC_CONF"])
os.environ.setdefault("PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING", "1")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("TORCH_CUDNN_ENABLE", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from environment import get_memory_snapshot, initialize_rocm_runtime

ROCM_STARTUP_INFO = initialize_rocm_runtime()

import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from rl import ActorCriticAgent
from utils import CsvMetricLogger, select_amp_dtype


def _configure_torch_backends():
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = Config.CUDNN_DETERMINISTIC
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

try:
    mp.set_start_method("forkserver", force=True)
except RuntimeError:
    pass

_configure_torch_backends()
torch.set_num_threads(Config.TORCH_NUM_THREADS)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def _parse_optional_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_lmdb_map_size(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("LMDB map size must be positive")
    if parsed > 1024:
        gib = 1024 ** 3
        return max(1, (parsed + gib - 1) // gib)
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="PPO-first Poker2 trainer tuned for RX 7900 XT ROCm.")
    parser.add_argument("--profile", choices=["stable", "stable-plus", "smoke", "medium", "aggressive", "long-run", "custom"], default=None, help="Apply a named PPO runtime profile before individual overrides")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short PPO validation profile")
    parser.add_argument("--smoke_test", dest="smoke_test_compat", nargs="?", const=True, default=None, type=_parse_optional_bool, help=argparse.SUPPRESS)
    parser.add_argument("--long-run", action="store_true", help="Run the long unattended profile")
    parser.add_argument("--long_run", dest="long_run_compat", nargs="?", const=True, default=None, type=_parse_optional_bool, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", "--num-hands", "--num_hands", dest="iterations", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", "--batch", dest="batch_size", type=int, default=None)
    parser.add_argument("--rollout-steps", "--rollout_steps", "--rollout", dest="rollout_steps", type=int, default=None, help="Override PPO rollout size per iteration")
    parser.add_argument("--hidden-size", "--hidden_size", "--hidden", dest="hidden_size", type=int, default=None, help="Override PPO model hidden width before model construction")
    parser.add_argument("--num-simulations", "--num_simulations", "--sims", dest="num_simulations", type=int, default=None)
    parser.add_argument("--mp-processes", "--mp_processes", "--mp", dest="mp_processes", type=int, default=None, help="Override multiprocessing worker count for hand ranking")
    parser.add_argument("--learning-rate", "--learning_rate", "--lr", dest="learning_rate", type=float, default=None, help="Override AdamW learning rate")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp", dest="amp_enabled", nargs="?", const=True, default=None, type=_parse_optional_bool, help="Explicitly enable or disable AMP")
    parser.add_argument("--amp-bf16", "--bf16", dest="amp_bf16", nargs="?", const=True, default=None, type=_parse_optional_bool, help="Enable BF16 autocast if supported")
    parser.add_argument("--amp-dtype", "--amp_dtype", dest="amp_dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default=None, help="Override autocast dtype selection")
    parser.add_argument("--enable-ppo-replay", "--use-per", "--use_per", "--replay", dest="enable_ppo_replay", nargs="?", const=True, default=None, type=_parse_optional_bool, help="Enable auxiliary PPO replay writes")
    parser.add_argument("--replay-buffer-size", "--replay_capacity", dest="replay_buffer_size", type=int, default=None, help="Set PPO replay capacity when replay is enabled")
    parser.add_argument("--replay-warmup-samples", "--replay_warmup_samples", dest="replay_warmup_samples", type=int, default=None, help="Set PPO replay warmup threshold")
    parser.add_argument("--lmdb-map-size-gb", "--lmdb_map_size", dest="lmdb_map_size_gb", type=_parse_lmdb_map_size, default=None, help="Set LMDB map size for PPO replay, in GB or raw bytes")
    parser.add_argument("--log-interval", "--log-every", "--log_every", dest="log_interval", type=int, default=None)
    parser.add_argument("--train-steps", "--training-steps", "--num-training-steps", "--num_training_steps", dest="num_training_steps", type=int, default=None, help="Set PPO training steps per iteration")
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--validation-interval", "--validation_interval", dest="validation_interval", type=int, default=None)
    parser.add_argument("--entropy-coef", "--entropy-beta", "--entropy_coef", "--entropy", dest="entropy_coef", type=float, default=None, help="Override PPO entropy coefficient")
    parser.add_argument("--entropy-final-coef", "--entropy_final_coef", dest="entropy_final_coef", type=float, default=None, help="Set the final entropy coefficient for linear decay")
    parser.add_argument("--entropy-decay-iterations", "--entropy_decay_iterations", dest="entropy_decay_iterations", type=int, default=None, help="Iterations over which to linearly decay entropy")
    parser.add_argument("--value-clipping", "--value_clipping", "--value-clip", "--value_clip", dest="value_clipping", nargs="?", const=True, default=None, type=_parse_optional_bool, help="Enable PPO value clipping around rollout values")
    parser.add_argument("--mcts-depth", "--mcts_depth", "--mcts-max-depth", "--mcts_max_depth", dest="mcts_depth", type=int, default=None, help="Set depth for PPO MCTS search")
    parser.add_argument("--population-mix-prob", "--population_mix_prob", dest="population_mix_prob", type=float, default=None, help="Probability of using a mutated population clone for PPO behavior sampling")
    parser.add_argument("--enable-equity-training", dest="enable_equity_training", nargs="?", const=True, default=None, type=_parse_optional_bool, help="Refresh EquityNet in a background subprocess during PPO training")
    parser.add_argument("--equity-train-interval", "--equity_refresh_interval", dest="equity_train_interval", type=int, default=None, help="Iterations between background EquityNet refresh jobs")
    parser.add_argument("--equity-train-epochs", type=int, default=None, help="Background EquityNet epochs per refresh job")
    parser.add_argument("--equity-train-steps-per-epoch", type=int, default=None, help="Background EquityNet steps per epoch")
    parser.add_argument("--equity-train-batch-size", type=int, default=None, help="Background EquityNet batch size")
    parser.add_argument("--equity-train-validation-batches", type=int, default=None, help="Background EquityNet validation batches")
    parser.add_argument("--players", type=int, default=None, help="Pin number of opponents")
    # --- Deep CFR options ---
    parser.add_argument("--algo", choices=["ppo", "deep-cfr"], default="ppo",
                        help="Training algorithm")
    parser.add_argument("--cfr-traversals", dest="cfr_traversals", type=int, default=None,
                        help="External-sampling traversals per player per CFR iter")
    parser.add_argument("--cfr-hidden", dest="cfr_hidden", type=int, default=None,
                        help="Deep CFR network hidden width")
    parser.add_argument("--cfr-blocks", dest="cfr_blocks", type=int, default=None,
                        help="Deep CFR network residual block count")
    parser.add_argument("--cfr-eval-hands", dest="cfr_eval_hands", type=int, default=None,
                        help="Hands per baseline match in Deep CFR eval")
    parser.add_argument("--cfr-eval-interval", dest="cfr_eval_interval", type=int, default=None,
                        help="CFR iterations between baseline evals")
    parser.add_argument("--cfr-stack", dest="cfr_stack", type=int, default=None,
                        help="Starting stack in chips (BB units = stack / big_blind)")
    parser.add_argument("--cfr-adv-steps", dest="cfr_adv_steps", type=int, default=None)
    parser.add_argument("--cfr-strat-steps", dest="cfr_strat_steps", type=int, default=None)
    parser.add_argument("--cfr-batch-size", dest="cfr_batch_size", type=int, default=None)
    parser.add_argument("--cfr-lr", dest="cfr_lr", type=float, default=None)
    parser.add_argument("--cfr-num-workers", dest="cfr_num_workers", type=int, default=None,
                        help="Worker processes for parallel CFR traversals (0=serial)")
    parser.add_argument("--cfr-worker-chunk", dest="cfr_worker_chunk", type=int, default=None,
                        help="Min traversals per worker chunk")
    parser.add_argument("--cfr-compile", dest="cfr_compile", action="store_true",
                        help="Apply torch.compile to GPU train nets")
    parser.add_argument("--cfr-async", dest="cfr_async", action="store_true",
                        help="Overlap CPU traversals with GPU training (Phase B)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def apply_runtime_profile(args):
    smoke_selected = bool(args.smoke_test)
    if args.profile is None and args.smoke_test_compat is not None:
        smoke_selected = bool(args.smoke_test_compat)

    long_run_selected = bool(args.long_run)
    if args.profile is None and args.long_run_compat is not None:
        long_run_selected = bool(args.long_run_compat)

    if smoke_selected and long_run_selected:
        raise ValueError("Use either --smoke-test or --long-run, not both")
    if args.profile in {"smoke", "long-run"} and (smoke_selected or long_run_selected):
        raise ValueError("Use either --profile for named profiles or the legacy smoke/long-run flags, not both")

    profile_name = 'stable'
    custom_overrides = False

    if args.profile == 'smoke' or smoke_selected:
        profile_name = 'smoke'
        Config.ITERATIONS = Config.SMOKE_TEST_ITERATIONS
        Config.BATCH_SIZE = Config.SMOKE_TEST_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.SMOKE_TEST_NUM_SIMULATIONS
        Config.ROLLOUT_STEPS = Config.SMOKE_TEST_ROLLOUT_STEPS
        Config.ROLLOUT_SCALE_WITH_BATCH = False
        Config.WARMUP_ENABLED = False
        Config.VALIDATION_INTERVAL = Config.SMOKE_TEST_VALIDATION_INTERVAL
        Config.LOG_INTERVAL = Config.SMOKE_TEST_LOG_INTERVAL
        Config.CHECKPOINT_INTERVAL = Config.SMOKE_TEST_CHECKPOINT_INTERVAL

    if args.profile == 'stable-plus':
        profile_name = 'stable-plus'
        Config.ITERATIONS = Config.STABLE_PLUS_ITERATIONS
        Config.BATCH_SIZE = Config.STABLE_PLUS_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.STABLE_PLUS_NUM_SIMULATIONS
        Config.MODEL_HIDDEN_DIM = Config.STABLE_PLUS_HIDDEN_DIM
        Config.ROLLOUT_STEPS = Config.STABLE_PLUS_ROLLOUT_STEPS
        Config.ROLLOUT_SCALE_WITH_BATCH = False
        Config.WARMUP_ENABLED = False
        Config.LOG_INTERVAL = Config.STABLE_PLUS_LOG_INTERVAL
        Config.VALIDATION_INTERVAL = Config.STABLE_PLUS_VALIDATION_INTERVAL
        Config.NUM_TRAINING_STEPS = Config.STABLE_PLUS_NUM_TRAINING_STEPS
        Config.PPO_MINIBATCHES = Config.STABLE_PLUS_PPO_MINIBATCHES
        Config.GRADIENT_ACCUMULATION_STEPS = Config.STABLE_PLUS_GRADIENT_ACCUMULATION_STEPS
        Config.CACHE_CLEAR_INTERVAL_STEPS = Config.STABLE_PLUS_CACHE_CLEAR_INTERVAL_STEPS
        Config.GRADIENT_CHECKPOINTING = Config.STABLE_PLUS_GRADIENT_CHECKPOINTING
        Config.MCTS_MAX_DEPTH = Config.STABLE_PLUS_MCTS_DEPTH
        Config.PPO_REPLAY_ENABLED = Config.STABLE_PLUS_PPO_REPLAY_ENABLED
        Config.VRAM_SOFT_LIMIT_GB = Config.STABLE_PLUS_VRAM_SOFT_LIMIT_GB
        Config.MP_PROCESSES = Config.STABLE_PLUS_MP_PROCESSES

    if args.profile == 'medium':
        profile_name = 'medium'
        Config.ITERATIONS = Config.MEDIUM_RUN_ITERATIONS
        Config.BATCH_SIZE = Config.MEDIUM_RUN_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.MEDIUM_RUN_NUM_SIMULATIONS
        Config.MODEL_HIDDEN_DIM = Config.MEDIUM_RUN_HIDDEN_DIM
        Config.ROLLOUT_STEPS = Config.MEDIUM_RUN_ROLLOUT_STEPS
        Config.ROLLOUT_SCALE_WITH_BATCH = False
        Config.WARMUP_ENABLED = False
        Config.LOG_INTERVAL = Config.MEDIUM_RUN_LOG_INTERVAL
        Config.VALIDATION_INTERVAL = Config.MEDIUM_RUN_VALIDATION_INTERVAL
        Config.NUM_TRAINING_STEPS = Config.MEDIUM_RUN_NUM_TRAINING_STEPS
        Config.GRADIENT_ACCUMULATION_STEPS = Config.MEDIUM_RUN_GRADIENT_ACCUMULATION_STEPS
        Config.REPLAY_BUFFER_SIZE = Config.MEDIUM_RUN_REPLAY_BUFFER_SIZE
        Config.LMDB_MAP_SIZE_GB = Config.MEDIUM_RUN_LMDB_MAP_SIZE_GB
        Config.ENTROPY_COEF = Config.MEDIUM_RUN_ENTROPY_COEF
        Config.ENTROPY_FINAL_COEF = Config.MEDIUM_RUN_ENTROPY_FINAL_COEF
        Config.ENTROPY_DECAY_ITERS = Config.MEDIUM_RUN_ENTROPY_DECAY_ITERS
        Config.POPULATION_MIX_PROB = Config.MEDIUM_RUN_POPULATION_MIX_PROB
        Config.MCTS_MAX_DEPTH = Config.MEDIUM_RUN_MCTS_DEPTH
        Config.PPO_REPLAY_ENABLED = Config.MEDIUM_RUN_PPO_REPLAY_ENABLED
        Config.VALUE_CLIP_ENABLED = Config.MEDIUM_RUN_VALUE_CLIP_ENABLED
        Config.EQUITY_TRAIN_INTERVAL = Config.MEDIUM_RUN_EQUITY_TRAIN_INTERVAL
        Config.MP_PROCESSES = Config.MEDIUM_RUN_MP_PROCESSES

    if args.profile == 'aggressive':
        profile_name = 'aggressive'
        Config.ITERATIONS = Config.AGGRESSIVE_RUN_ITERATIONS
        Config.BATCH_SIZE = Config.AGGRESSIVE_RUN_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.AGGRESSIVE_RUN_NUM_SIMULATIONS
        Config.MODEL_HIDDEN_DIM = Config.AGGRESSIVE_RUN_HIDDEN_DIM
        Config.ROLLOUT_STEPS = Config.AGGRESSIVE_RUN_ROLLOUT_STEPS
        Config.ROLLOUT_SCALE_WITH_BATCH = Config.AGGRESSIVE_RUN_ROLLOUT_SCALE_WITH_BATCH
        Config.MAX_ROLLOUT_STEPS = Config.AGGRESSIVE_RUN_MAX_ROLLOUT_STEPS
        Config.LOG_INTERVAL = Config.AGGRESSIVE_RUN_LOG_INTERVAL
        Config.VALIDATION_INTERVAL = Config.AGGRESSIVE_RUN_VALIDATION_INTERVAL
        Config.NUM_TRAINING_STEPS = Config.AGGRESSIVE_RUN_NUM_TRAINING_STEPS
        Config.GRADIENT_ACCUMULATION_STEPS = Config.AGGRESSIVE_RUN_GRADIENT_ACCUMULATION_STEPS
        Config.REPLAY_BUFFER_SIZE = Config.AGGRESSIVE_RUN_REPLAY_BUFFER_SIZE
        Config.LMDB_MAP_SIZE_GB = Config.AGGRESSIVE_RUN_LMDB_MAP_SIZE_GB
        Config.LEARNING_RATE = Config.AGGRESSIVE_RUN_LEARNING_RATE
        Config.ENTROPY_COEF = Config.AGGRESSIVE_RUN_ENTROPY_COEF
        Config.ENTROPY_FINAL_COEF = Config.AGGRESSIVE_RUN_ENTROPY_FINAL_COEF
        Config.ENTROPY_DECAY_ITERS = Config.AGGRESSIVE_RUN_ENTROPY_DECAY_ITERS
        Config.POPULATION_MIX_PROB = Config.AGGRESSIVE_RUN_POPULATION_MIX_PROB
        Config.MCTS_MAX_DEPTH = Config.AGGRESSIVE_RUN_MCTS_DEPTH
        Config.MCTS_TRIGGER_PROB = Config.AGGRESSIVE_RUN_MCTS_TRIGGER_PROB
        Config.PPO_REPLAY_ENABLED = Config.AGGRESSIVE_RUN_PPO_REPLAY_ENABLED
        Config.VALUE_CLIP_ENABLED = Config.AGGRESSIVE_RUN_VALUE_CLIP_ENABLED
        Config.EQUITY_TRAIN_INTERVAL = Config.AGGRESSIVE_RUN_EQUITY_TRAIN_INTERVAL
        Config.MP_PROCESSES = Config.AGGRESSIVE_RUN_MP_PROCESSES
        Config.PPO_MINIBATCHES = Config.AGGRESSIVE_RUN_PPO_MINIBATCHES
        Config.GRADIENT_CHECKPOINTING = Config.AGGRESSIVE_RUN_GRADIENT_CHECKPOINTING
        Config.USE_TORCH_COMPILE = Config.AGGRESSIVE_RUN_USE_TORCH_COMPILE
        Config.VRAM_SOFT_LIMIT_GB = Config.AGGRESSIVE_RUN_VRAM_SOFT_LIMIT_GB
        Config.RAM_SOFT_LIMIT_PCT = Config.AGGRESSIVE_RUN_RAM_SOFT_LIMIT_PCT
        Config.WARMUP_ENABLED = True
        Config.WARMUP_TOTAL_ITERS = Config.AGGRESSIVE_WARMUP_ITERS
        Config.WARMUP_MIN_ROLLOUT_STEPS = Config.AGGRESSIVE_WARMUP_MIN_ROLLOUT_STEPS
        Config.WARMUP_MIN_NUM_SIMULATIONS = Config.AGGRESSIVE_WARMUP_MIN_NUM_SIMULATIONS
        Config.WARMUP_MIN_TRAINING_STEPS = Config.AGGRESSIVE_WARMUP_MIN_TRAINING_STEPS

    if args.profile == 'long-run' or long_run_selected:
        profile_name = 'long-run'
        Config.ITERATIONS = Config.LONG_RUN_ITERATIONS

    if args.iterations is not None:
        Config.ITERATIONS = args.iterations
        custom_overrides = True
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        custom_overrides = True
    if args.rollout_steps is not None:
        Config.ROLLOUT_STEPS = max(1, int(args.rollout_steps))
        custom_overrides = True
    if args.hidden_size is not None:
        Config.MODEL_HIDDEN_DIM = max(128, int(args.hidden_size))
        custom_overrides = True
    if args.num_simulations is not None:
        Config.NUM_SIMULATIONS = args.num_simulations
        custom_overrides = True
    if args.learning_rate is not None:
        Config.LEARNING_RATE = max(1e-7, float(args.learning_rate))
        custom_overrides = True
    if args.mp_processes is not None:
        Config.MP_PROCESSES = max(1, int(args.mp_processes))
        custom_overrides = True
    if args.log_interval is not None:
        Config.LOG_INTERVAL = args.log_interval
        custom_overrides = True
    if args.num_training_steps is not None:
        Config.NUM_TRAINING_STEPS = max(1, int(args.num_training_steps))
        custom_overrides = True
    if args.checkpoint_interval is not None:
        Config.CHECKPOINT_INTERVAL = args.checkpoint_interval
        custom_overrides = True
    if args.validation_interval is not None:
        Config.VALIDATION_INTERVAL = args.validation_interval
        custom_overrides = True
    if args.entropy_coef is not None:
        Config.ENTROPY_COEF = float(args.entropy_coef)
        custom_overrides = True
    if args.entropy_final_coef is not None:
        Config.ENTROPY_FINAL_COEF = float(args.entropy_final_coef)
        custom_overrides = True
    if args.entropy_decay_iterations is not None:
        Config.ENTROPY_DECAY_ITERS = max(0, int(args.entropy_decay_iterations))
        custom_overrides = True
    if args.replay_buffer_size is not None:
        Config.REPLAY_BUFFER_SIZE = max(1024, int(args.replay_buffer_size))
        custom_overrides = True
    if args.replay_warmup_samples is not None:
        Config.REPLAY_WARMUP_SAMPLES = max(1, int(args.replay_warmup_samples))
        custom_overrides = True
    if args.lmdb_map_size_gb is not None:
        Config.LMDB_MAP_SIZE_GB = max(1, min(Config.MAX_LMDB_MAP_SIZE_GB, int(args.lmdb_map_size_gb)))
        custom_overrides = True
    if args.players is not None:
        Config.START_OPPONENTS = max(2, min(Config.TARGET_OPPONENTS, args.players))
        Config.TARGET_OPPONENTS = Config.START_OPPONENTS
        custom_overrides = True
    if args.mcts_depth is not None:
        Config.MCTS_MAX_DEPTH = max(0, int(args.mcts_depth))
        custom_overrides = True
    if args.population_mix_prob is not None:
        Config.POPULATION_MIX_PROB = float(max(0.0, min(1.0, args.population_mix_prob)))
        custom_overrides = True

    if args.disable_amp:
        Config.AMP_ENABLED = False
        custom_overrides = True
    if args.amp_enabled is not None:
        Config.AMP_ENABLED = bool(args.amp_enabled)
        custom_overrides = True
    if args.amp_bf16 is not None:
        Config.AMP_BF16_OPT_IN = bool(args.amp_bf16)
        Config.AMP_PREFER_BF16 = bool(args.amp_bf16)
        custom_overrides = True
    if args.amp_dtype is not None:
        normalized_amp_dtype = args.amp_dtype.lower()
        if normalized_amp_dtype in {"bf16", "bfloat16"}:
            Config.AMP_BF16_OPT_IN = True
            Config.AMP_PREFER_BF16 = True
        elif normalized_amp_dtype in {"fp16", "float16"}:
            Config.AMP_BF16_OPT_IN = False
            Config.AMP_PREFER_BF16 = False
            Config.AMP_DTYPE = torch.float16
        elif normalized_amp_dtype in {"fp32", "float32"}:
            Config.AMP_ENABLED = False
            Config.AMP_DTYPE = torch.float32
        custom_overrides = True
    if args.enable_ppo_replay is not None:
        Config.PPO_REPLAY_ENABLED = bool(args.enable_ppo_replay)
        custom_overrides = True
    if args.value_clipping is not None:
        Config.VALUE_CLIP_ENABLED = bool(args.value_clipping)
        custom_overrides = True
    if args.enable_equity_training is not None:
        Config.EQUITY_TRAINING_ENABLED = bool(args.enable_equity_training)
        custom_overrides = True
    if args.equity_train_interval is not None:
        Config.EQUITY_TRAIN_INTERVAL = max(1, int(args.equity_train_interval))
        custom_overrides = True
    if args.equity_train_epochs is not None:
        Config.EQUITY_TRAIN_EPOCHS = max(1, int(args.equity_train_epochs))
        custom_overrides = True
    if args.equity_train_steps_per_epoch is not None:
        Config.EQUITY_TRAIN_STEPS_PER_EPOCH = max(1, int(args.equity_train_steps_per_epoch))
        custom_overrides = True
    if args.equity_train_batch_size is not None:
        Config.EQUITY_TRAIN_BATCH_SIZE = max(16, int(args.equity_train_batch_size))
        custom_overrides = True
    if args.equity_train_validation_batches is not None:
        Config.EQUITY_TRAIN_VALIDATION_BATCHES = max(1, int(args.equity_train_validation_batches))
        custom_overrides = True

    if custom_overrides:
        profile_name = f'{profile_name}-custom' if profile_name != 'stable' else 'custom'

    Config.AMP_ENABLED = Config.AMP_ENABLED and Config.DEVICE == 'cuda'
    Config.RUNTIME_PROFILE = profile_name
    if Config.AMP_ENABLED:
        Config.AMP_DTYPE = select_amp_dtype(Config)
    else:
        Config.AMP_DTYPE = torch.float32
    _configure_torch_backends()
    Config.RECOVERY_BATCH_CAP = Config.BATCH_SIZE
    Config.NUM_SIMS = Config.NUM_SIMULATIONS
    Config.PRESSURE_SIMULATION_CAP = Config.NUM_SIMULATIONS
    Config.RECOVERY_SIMULATION_CAP = Config.NUM_SIMULATIONS
    torch.set_num_threads(Config.TORCH_NUM_THREADS)


def _log_startup(args):
    snapshot = get_memory_snapshot()
    logger.info("DEVICE: %s", ROCM_STARTUP_INFO["device"])
    logger.info("GPU Detected: %s", ROCM_STARTUP_INFO["gpu_name"])
    logger.info("VRAM Total: %.2f GB", ROCM_STARTUP_INFO["vram_total_gb"])
    logger.info("ROCm Fallback Applied: %s", ROCM_STARTUP_INFO["fallback_applied"])
    if ROCM_STARTUP_INFO.get("probe_error"):
        logger.warning("ROCm probe error: %s", ROCM_STARTUP_INFO["probe_error"])
    logger.info(
        "profile=%s | smoke=%s | iterations=%s | rollout=%s | batch=%s | hidden=%s | sims=%s | mp=%s | train_steps=%s | amp=%s | amp_dtype=%s | replay=%s | entropy=%.4f -> %.4f/%s | value_clip=%s | mcts_depth=%s | warmup=%s/%s",
        Config.RUNTIME_PROFILE,
        args.smoke_test,
        Config.ITERATIONS,
        Config.ROLLOUT_STEPS,
        Config.BATCH_SIZE,
        Config.MODEL_HIDDEN_DIM,
        Config.NUM_SIMULATIONS,
        Config.MP_PROCESSES,
        Config.NUM_TRAINING_STEPS,
        Config.AMP_ENABLED,
        Config.AMP_DTYPE,
        Config.PPO_REPLAY_ENABLED,
        Config.ENTROPY_COEF,
        Config.ENTROPY_FINAL_COEF,
        Config.ENTROPY_DECAY_ITERS,
        Config.VALUE_CLIP_ENABLED,
        Config.MCTS_MAX_DEPTH,
        int(getattr(Config, "WARMUP_ENABLED", False)),
        int(getattr(Config, "WARMUP_TOTAL_ITERS", 0)),
    )
    logger.info(
        "memory: vram %.2f/%.2f GB (%.1f%%) | ram %.1f%% | vram soft limit %.1f GB | lmdb map %s GB | replay cap %s",
        snapshot["used_gb"],
        snapshot["total_gb"],
        snapshot["used_pct"],
        snapshot["ram_pct"],
        Config.VRAM_SOFT_LIMIT_GB,
        Config.LMDB_MAP_SIZE_GB,
        Config.REPLAY_BUFFER_SIZE,
    )
    logger.info(
        "ROCm env: HIP_VISIBLE_DEVICES=%s | HSA_OVERRIDE_GFX_VERSION=%s | PYTORCH_ALLOC_CONF=%s | HSA_ENABLE_SDMA=%s | TORCH_CUDNN_ENABLE=%s | OMP_NUM_THREADS=%s | MKL_NUM_THREADS=%s",
        os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        os.environ.get("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        os.environ.get("PYTORCH_ALLOC_CONF", ""),
        os.environ.get("HSA_ENABLE_SDMA", "0"),
        os.environ.get("TORCH_CUDNN_ENABLE", "0"),
        os.environ.get("OMP_NUM_THREADS", "1"),
        os.environ.get("MKL_NUM_THREADS", "1"),
    )


def train_ppo(args):
    writer = SummaryWriter()
    csv_logger = CsvMetricLogger(os.path.join("runs", "ppo_metrics.csv"))
    agent = ActorCriticAgent(writer=writer)
    equity_process = None
    equity_log_handle = None

    def maybe_start_equity_training(current_iteration, num_opponents):
        nonlocal equity_process, equity_log_handle
        if not Config.EQUITY_TRAINING_ENABLED or current_iteration <= 0:
            return
        if current_iteration % Config.EQUITY_TRAIN_INTERVAL != 0:
            return
        if equity_process is not None and equity_process.poll() is None:
            return
        if equity_process is not None and equity_process.poll() is not None and equity_log_handle is not None:
            logger.info("equity trainer exited with code %s", equity_process.returncode)
            equity_log_handle.close()
            equity_process = None
            equity_log_handle = None

        equity_log_handle = open("equity_training.log", "a", encoding="utf-8")
        command = [
            sys.executable,
            "equity_trainer.py",
            "--epochs", str(Config.EQUITY_TRAIN_EPOCHS),
            "--steps-per-epoch", str(Config.EQUITY_TRAIN_STEPS_PER_EPOCH),
            "--batch-size", str(Config.EQUITY_TRAIN_BATCH_SIZE),
            "--validation-batch-size", str(Config.EQUITY_TRAIN_VALIDATION_BATCH_SIZE),
            "--validation-batches", str(Config.EQUITY_TRAIN_VALIDATION_BATCHES),
            "--num-opponents", str(num_opponents),
        ]
        equity_process = subprocess.Popen(
            command,
            stdout=equity_log_handle,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
        logger.info("started background equity trainer at iteration %s (pid=%s)", current_iteration, equity_process.pid)

    if args.resume_checkpoint:
        start_iter = agent.load_checkpoint(args.resume_checkpoint)
        logger.info("resumed checkpoint from iteration %s", start_iter)
        logger.info(
            "resume state | batch=%s | hidden=%s | sims=%s | players=%s | elo=%.1f | replay=%s",
            agent.current_batch_size,
            Config.MODEL_HIDDEN_DIM,
            agent.current_simulations,
            agent.num_opponents,
            agent.elo_tracker.rating,
            Config.PPO_REPLAY_ENABLED,
        )

    best_elo = agent.elo_tracker.rating

    try:
        for _ in range(Config.ITERATIONS):
            metrics = agent.train_iteration()
            maybe_start_equity_training(agent.iteration, agent.num_opponents)

            # Periodic cache cleanup to reduce long-run ROCm VRAM ratcheting.
            clear_interval = max(1, int(getattr(Config, "CACHE_CLEAR_INTERVAL_ITERS", 10)))
            proactive_clear_gb = float(getattr(Config, "VRAM_PROACTIVE_CLEAR_GB", Config.VRAM_SOFT_LIMIT_GB * 0.95))
            if agent.iteration > 0 and agent.iteration % clear_interval == 0 and Config.DEVICE == "cuda":
                snapshot = get_memory_snapshot()
                if snapshot["used_gb"] >= proactive_clear_gb:
                    torch.cuda.empty_cache()
                    gc.collect()
                    snapshot = get_memory_snapshot()

                # Only force a batch downshift if we still exceed soft limit after cleanup.
                if snapshot["used_gb"] > Config.VRAM_SOFT_LIMIT_GB:
                    torch.cuda.empty_cache()
                    gc.collect()
                    snapshot = get_memory_snapshot()
                    new_batch = max(4096, int(agent.current_batch_size) // 2)
                    if new_batch < agent.current_batch_size:
                        agent.current_batch_size = new_batch
                        Config.BATCH_SIZE = new_batch
                        Config.RECOVERY_BATCH_CAP = min(Config.RECOVERY_BATCH_CAP, new_batch)
                        logger.warning(
                            "vram soft limit hit after cache clear (%.2f GB) -> reducing batch to %s",
                            snapshot["used_gb"],
                            new_batch,
                        )

            if metrics["elo"] > best_elo:
                best_elo = metrics["elo"]
                agent.save_checkpoint(agent.iteration, metrics, is_best=True)

            csv_logger.log(
                {
                    "iteration": agent.iteration,
                    "loss": metrics["loss"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "entropy_coef": metrics["entropy_coef"],
                    "avg_reward": metrics["avg_reward"],
                    "elo": metrics["elo"],
                    "ram_pct": metrics["ram_pct"],
                    "vram_used_gb": metrics["vram_used_gb"],
                    "vram_pct": metrics["vram_pct"],
                    "batch_size": metrics["batch_size"],
                    "rollout_steps": metrics["rollout_steps"],
                    "simulations": metrics["simulations"],
                    "training_steps": metrics["training_steps"],
                    "num_opponents": metrics["num_opponents"],
                    "mcts_rate": metrics["mcts_rate"],
                    "population_rate": metrics["population_rate"],
                }
            )

            if agent.iteration % Config.LOG_INTERVAL == 0:
                logger.info(
                    "iter %s | reward %.4f | elo %.1f | loss %.4f | entropy_coef %.4f | vram %.2f GB | ram %.1f%% | batch %s | rollout %s | sims %s | train_steps %s | players %s | mcts %.2f | pop %.2f | backoff %s",
                    agent.iteration,
                    metrics["avg_reward"],
                    metrics["elo"],
                    metrics["loss"],
                    metrics["entropy_coef"],
                    metrics["vram_used_gb"],
                    metrics["ram_pct"],
                    metrics["batch_size"],
                    metrics["rollout_steps"],
                    metrics["simulations"],
                    metrics["training_steps"],
                    metrics["num_opponents"],
                    metrics["mcts_rate"],
                    metrics["population_rate"],
                    metrics["backoff"],
                )
    finally:
        if equity_process is not None and equity_process.poll() is None:
            equity_process.terminate()
            try:
                equity_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                equity_process.kill()
        if equity_log_handle is not None:
            equity_log_handle.close()
        writer.close()


def train_deep_cfr(args):
    from algo.deep_cfr import DeepCFRConfig, DeepCFRTrainer

    cfg = DeepCFRConfig()
    if args.iterations is not None:
        cfg.num_iterations = int(args.iterations)
    if args.cfr_traversals is not None:
        cfg.traversals_per_iter = int(args.cfr_traversals)
    if args.cfr_hidden is not None:
        cfg.hidden_size = int(args.cfr_hidden)
    if args.cfr_blocks is not None:
        cfg.num_blocks = int(args.cfr_blocks)
    if args.cfr_eval_hands is not None:
        cfg.eval_hands = int(args.cfr_eval_hands)
    if args.cfr_eval_interval is not None:
        cfg.eval_interval = int(args.cfr_eval_interval)
    if args.cfr_stack is not None:
        cfg.starting_stack = int(args.cfr_stack)
    if args.cfr_adv_steps is not None:
        cfg.advantage_train_steps = int(args.cfr_adv_steps)
    if args.cfr_strat_steps is not None:
        cfg.strategy_train_steps = int(args.cfr_strat_steps)
    if args.cfr_batch_size is not None:
        cfg.train_batch_size = int(args.cfr_batch_size)
    if args.cfr_lr is not None:
        cfg.learning_rate = float(args.cfr_lr)
    if args.cfr_num_workers is not None:
        cfg.num_workers = int(args.cfr_num_workers)
    if args.cfr_worker_chunk is not None:
        cfg.worker_chunk_min = int(args.cfr_worker_chunk)
    if args.cfr_compile:
        cfg.use_torch_compile = True
    if args.cfr_async:
        cfg.async_pipeline = True
    if args.players is not None:
        cfg.num_players = int(args.players)
    if args.log_interval is not None:
        cfg.log_interval = int(args.log_interval)
    if args.amp_dtype is not None:
        cfg.amp_dtype = (
            "bfloat16" if args.amp_dtype in ("auto", "bf16", "bfloat16")
            else "float16" if args.amp_dtype in ("fp16", "float16") else "float32"
        )
    cfg.seed = int(args.seed)

    logger.info(
        "DEEP CFR | players=%d | iters=%d | traversals/iter/p=%d | hidden=%d | blocks=%d | "
        "stack=%d (%d BB) | adv_steps=%d | strat_steps=%d | bs=%d | lr=%.4g | eval_every=%d (%d hands) | seed=%d",
        cfg.num_players, cfg.num_iterations, cfg.traversals_per_iter,
        cfg.hidden_size, cfg.num_blocks, cfg.starting_stack,
        cfg.starting_stack // cfg.big_blind, cfg.advantage_train_steps,
        cfg.strategy_train_steps, cfg.train_batch_size, cfg.learning_rate,
        cfg.eval_interval, cfg.eval_hands, cfg.seed,
    )
    DeepCFRTrainer(cfg).train()


def main():
    args = parse_args()
    if args.algo == "deep-cfr":
        train_deep_cfr(args)
        return
    apply_runtime_profile(args)
    _log_startup(args)
    train_ppo(args)


if __name__ == "__main__":
    main()
