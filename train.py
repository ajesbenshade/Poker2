import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import sys

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
    parser.add_argument("--profile", choices=["stable", "smoke", "medium", "long-run", "custom"], default=None, help="Apply a named PPO runtime profile before individual overrides")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short PPO validation profile")
    parser.add_argument("--smoke_test", dest="smoke_test_compat", nargs="?", const=True, default=None, type=_parse_optional_bool, help=argparse.SUPPRESS)
    parser.add_argument("--long-run", action="store_true", help="Run the long unattended profile")
    parser.add_argument("--long_run", dest="long_run_compat", nargs="?", const=True, default=None, type=_parse_optional_bool, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", "--num-hands", "--num_hands", dest="iterations", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", "--batch", dest="batch_size", type=int, default=None)
    parser.add_argument("--hidden-size", "--hidden_size", "--hidden", dest="hidden_size", type=int, default=None, help="Override PPO model hidden width before model construction")
    parser.add_argument("--num-simulations", "--num_simulations", "--sims", dest="num_simulations", type=int, default=None)
    parser.add_argument("--mp-processes", "--mp_processes", "--mp", dest="mp_processes", type=int, default=None, help="Override multiprocessing worker count for hand ranking")
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
        Config.VALIDATION_INTERVAL = Config.SMOKE_TEST_VALIDATION_INTERVAL
        Config.LOG_INTERVAL = Config.SMOKE_TEST_LOG_INTERVAL
        Config.CHECKPOINT_INTERVAL = Config.SMOKE_TEST_CHECKPOINT_INTERVAL

    if args.profile == 'medium':
        profile_name = 'medium'
        Config.ITERATIONS = Config.MEDIUM_RUN_ITERATIONS
        Config.BATCH_SIZE = Config.MEDIUM_RUN_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.MEDIUM_RUN_NUM_SIMULATIONS
        Config.MODEL_HIDDEN_DIM = Config.MEDIUM_RUN_HIDDEN_DIM
        Config.LOG_INTERVAL = Config.MEDIUM_RUN_LOG_INTERVAL
        Config.VALIDATION_INTERVAL = Config.MEDIUM_RUN_VALIDATION_INTERVAL
        Config.NUM_TRAINING_STEPS = Config.MEDIUM_RUN_NUM_TRAINING_STEPS
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

    if args.profile == 'long-run' or long_run_selected:
        profile_name = 'long-run'
        Config.ITERATIONS = Config.LONG_RUN_ITERATIONS

    if args.iterations is not None:
        Config.ITERATIONS = args.iterations
        custom_overrides = True
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        custom_overrides = True
    if args.hidden_size is not None:
        Config.MODEL_HIDDEN_DIM = max(128, int(args.hidden_size))
        custom_overrides = True
    if args.num_simulations is not None:
        Config.NUM_SIMULATIONS = args.num_simulations
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
        "profile=%s | smoke=%s | iterations=%s | rollout=%s | batch=%s | hidden=%s | sims=%s | mp=%s | train_steps=%s | amp=%s | amp_dtype=%s | replay=%s | entropy=%.4f -> %.4f/%s | value_clip=%s | mcts_depth=%s",
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
        "ROCm env: HIP_VISIBLE_DEVICES=%s | HSA_OVERRIDE_GFX_VERSION=%s | PYTORCH_ALLOC_CONF=%s | HSA_ENABLE_SDMA=%s | TORCH_CUDNN_ENABLE=%s | OMP_NUM_THREADS=%s",
        os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        os.environ.get("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        os.environ.get("PYTORCH_ALLOC_CONF", ""),
        os.environ.get("HSA_ENABLE_SDMA", "0"),
        os.environ.get("TORCH_CUDNN_ENABLE", "0"),
        os.environ.get("OMP_NUM_THREADS", "1"),
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
                    "simulations": metrics["simulations"],
                    "num_opponents": metrics["num_opponents"],
                    "mcts_rate": metrics["mcts_rate"],
                    "population_rate": metrics["population_rate"],
                }
            )

            if agent.iteration % Config.LOG_INTERVAL == 0:
                logger.info(
                    "iter %s | reward %.4f | elo %.1f | loss %.4f | entropy_coef %.4f | vram %.2f GB | ram %.1f%% | batch %s | sims %s | players %s | mcts %.2f | pop %.2f | backoff %s",
                    agent.iteration,
                    metrics["avg_reward"],
                    metrics["elo"],
                    metrics["loss"],
                    metrics["entropy_coef"],
                    metrics["vram_used_gb"],
                    metrics["ram_pct"],
                    metrics["batch_size"],
                    metrics["simulations"],
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


def main():
    args = parse_args()
    apply_runtime_profile(args)
    _log_startup(args)
    train_ppo(args)


if __name__ == "__main__":
    main()
