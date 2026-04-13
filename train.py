import argparse
import logging
import multiprocessing as mp
import os

# Apply ROCm-critical env vars before importing torch.
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collect_threshold:0.6,expandable_segment:True,max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"

from environment import apply_hsa_fallback, get_memory_snapshot, setup_rocmo

setup_rocmo()

import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from rl import ActorCriticAgent
from utils import CsvMetricLogger

try:
    mp.set_start_method("forkserver", force=True)
except RuntimeError:
    pass

torch.set_num_threads(Config.TORCH_NUM_THREADS)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO-first Poker2 trainer tuned for RX 7900 XT ROCm.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short PPO validation profile")
    parser.add_argument("--long-run", action="store_true", help="Run the long unattended profile")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp-bf16", action="store_true", help="Enable BF16 autocast if supported")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--players", type=int, default=None, help="Pin number of opponents")
    return parser.parse_args()


def apply_runtime_profile(args):
    if args.smoke_test and args.long_run:
        raise ValueError("Use either --smoke-test or --long-run, not both")

    if args.smoke_test:
        Config.ITERATIONS = Config.SMOKE_TEST_ITERATIONS
        Config.BATCH_SIZE = Config.SMOKE_TEST_BATCH_SIZE
        Config.NUM_SIMULATIONS = Config.SMOKE_TEST_NUM_SIMULATIONS
        Config.ROLLOUT_STEPS = Config.SMOKE_TEST_ROLLOUT_STEPS
        Config.VALIDATION_INTERVAL = Config.SMOKE_TEST_VALIDATION_INTERVAL
        Config.LOG_INTERVAL = Config.SMOKE_TEST_LOG_INTERVAL
        Config.CHECKPOINT_INTERVAL = Config.SMOKE_TEST_CHECKPOINT_INTERVAL

    if args.long_run:
        Config.ITERATIONS = Config.LONG_RUN_ITERATIONS

    if args.iterations is not None:
        Config.ITERATIONS = args.iterations
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.num_simulations is not None:
        Config.NUM_SIMULATIONS = args.num_simulations
    if args.log_interval is not None:
        Config.LOG_INTERVAL = args.log_interval
    if args.checkpoint_interval is not None:
        Config.CHECKPOINT_INTERVAL = args.checkpoint_interval
    if args.validation_interval is not None:
        Config.VALIDATION_INTERVAL = args.validation_interval
    if args.players is not None:
        Config.START_OPPONENTS = max(2, min(Config.TARGET_OPPONENTS, args.players))
        Config.TARGET_OPPONENTS = Config.START_OPPONENTS

    if args.disable_amp:
        Config.AMP_ENABLED = False
    if args.amp_bf16:
        Config.AMP_BF16_OPT_IN = True


def _log_startup(args):
    snapshot = get_memory_snapshot()
    logger.info(
        "smoke=%s | iterations=%s | batch=%s | sims=%s | amp=%s | bf16_opt_in=%s",
        args.smoke_test,
        Config.ITERATIONS,
        Config.BATCH_SIZE,
        Config.NUM_SIMULATIONS,
        Config.AMP_ENABLED,
        Config.AMP_BF16_OPT_IN,
    )
    logger.info(
        "memory: vram %.2f/%.2f GB (%.1f%%) | ram %.1f%%",
        snapshot["used_gb"],
        snapshot["total_gb"],
        snapshot["used_pct"],
        snapshot["ram_pct"],
    )
    logger.info(
        "ROCm env: HIP_VISIBLE_DEVICES=%s | HSA_OVERRIDE_GFX_VERSION=%s | PYTORCH_HIP_ALLOC_CONF=%s | OMP_NUM_THREADS=%s",
        os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        os.environ.get("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        os.environ.get("PYTORCH_HIP_ALLOC_CONF", ""),
        os.environ.get("OMP_NUM_THREADS", "1"),
    )


def _rocm_sanity_or_fallback():
    if not torch.cuda.is_available():
        logger.warning("CUDA/HIP backend not available; running on CPU")
        return
    try:
        torch.cuda.current_device()
        logger.info("HIP device initialized: %s", torch.cuda.get_device_name(0))
    except Exception as exc:
        logger.warning("HIP init failed with current HSA override: %s", exc)
        fallback = apply_hsa_fallback()
        logger.warning("Applied HSA fallback %s; restart recommended if failures persist", fallback)


def train_ppo(args):
    writer = SummaryWriter()
    csv_logger = CsvMetricLogger(os.path.join("runs", "ppo_metrics.csv"))
    agent = ActorCriticAgent(writer=writer)

    if args.resume_checkpoint:
        start_iter = agent.load_checkpoint(args.resume_checkpoint)
        logger.info("resumed checkpoint from iteration %s", start_iter)

    best_elo = agent.elo_tracker.rating

    try:
        for _ in range(Config.ITERATIONS):
            metrics = agent.train_iteration()
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
                    "avg_reward": metrics["avg_reward"],
                    "elo": metrics["elo"],
                    "ram_pct": metrics["ram_pct"],
                    "vram_used_gb": metrics["vram_used_gb"],
                    "vram_pct": metrics["vram_pct"],
                    "batch_size": metrics["batch_size"],
                    "simulations": metrics["simulations"],
                    "num_opponents": metrics["num_opponents"],
                    "mcts_rate": metrics["mcts_rate"],
                }
            )

            if agent.iteration % Config.LOG_INTERVAL == 0:
                logger.info(
                    "iter %s | reward %.4f | elo %.1f | loss %.4f | vram %.2f GB | ram %.1f%% | batch %s | sims %s | players %s | mcts %.2f | backoff %s",
                    agent.iteration,
                    metrics["avg_reward"],
                    metrics["elo"],
                    metrics["loss"],
                    metrics["vram_used_gb"],
                    metrics["ram_pct"],
                    metrics["batch_size"],
                    metrics["simulations"],
                    metrics["num_opponents"],
                    metrics["mcts_rate"],
                    metrics["backoff"],
                )
    finally:
        writer.close()


def main():
    args = parse_args()
    apply_runtime_profile(args)
    _log_startup(args)
    _rocm_sanity_or_fallback()
    train_ppo(args)


if __name__ == "__main__":
    main()
