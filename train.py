infosets = None
actor = None

import argparse
import itertools
import json
import logging
import os
import random
import signal
import sys
from collections import defaultdict

_ALLOCATOR_CONF = "garbage_collection_threshold:0.6,max_split_size_mb:128"

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HIP_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11_0_0")
os.environ.setdefault("PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_HIP_ALLOC_CONF", _ALLOCATOR_CONF))

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from cfr import average_strategy, deep_cfr_traverse, mccfr
from abstractions import simulate_features, create_buckets
from config import Config
from datatypes import Infoset
from deep_cfr import DeepCFRAgent
from game import close_hand_eval_pool
from storage import LocalNodeStore, NODE_REGRET_INDEX, NODE_STRATEGY_INDEX

try:
    import ray
except ImportError:
    ray = None

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])


if ray is not None:
    @ray.remote
    class NodesActor:
        def __init__(self):
            self.nodes = defaultdict(
                lambda: torch.zeros((2, Config.NUM_ACTIONS), dtype=Config.STORAGE_DTYPE, device='cpu')
            )

        def get_regret_sum(self, key):
            return self.nodes[key][NODE_REGRET_INDEX]

        def get_strategy_sum(self, key):
            return self.nodes[key][NODE_STRATEGY_INDEX]

        def update_regret_sum(self, key, delta):
            self.nodes[key][NODE_REGRET_INDEX] += delta.to(device='cpu', dtype=Config.STORAGE_DTYPE)

        def update_strategy_sum(self, key, delta):
            self.nodes[key][NODE_STRATEGY_INDEX] += delta.to(device='cpu', dtype=Config.STORAGE_DTYPE)

        def get_all_keys(self):
            return list(self.nodes.keys())


    @ray.remote
    def run_mccfr(infoset, iteration, actor, max_depth):
        return mccfr(infoset, iteration, actor=actor, max_depth=max_depth)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the poker agent in tabular or deep mode.")
    parser.add_argument('--mode', choices=['tabular', 'deep'], default=None)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--long-run', action='store_true')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--num-sims', type=int, default=None)
    parser.add_argument('--num-buckets', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--equity-rollouts', type=int, default=None)
    parser.add_argument('--hand-eval-processes', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ray-cpus', type=int, default=None)
    parser.add_argument('--ray-gpus', type=int, default=None)
    parser.add_argument('--key-mode', choices=['legacy', 'state'], default=None)
    parser.add_argument('--nn-batch-size', type=int, default=None)
    parser.add_argument('--nn-train-steps', type=int, default=None)
    parser.add_argument('--nn-learning-rate', type=float, default=None)
    parser.add_argument('--deep-traversals-per-iter', type=int, default=None)
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    return parser.parse_args()


def apply_runtime_overrides(args):
    if args.smoke_test:
        smoke_defaults = {
            'iterations': 1,
            'num_sims': 8,
            'num_buckets': 2,
            'batch_size': 16,
            'equity_rollouts': 1,
            'hand_eval_processes': 1,
            'log_interval': 1,
            'checkpoint_interval': 1,
            'max_depth': 1,
            'nn_batch_size': 8,
            'nn_train_steps': 1,
            'deep_traversals_per_iter': 1,
        }
        for field_name, default_value in smoke_defaults.items():
            if getattr(args, field_name) is None:
                setattr(args, field_name, default_value)

    override_map = {
        'ALGORITHM_MODE': args.mode,
        'ITERATIONS': args.iterations,
        'NUM_SIMS': args.num_sims,
        'NUM_BUCKETS': args.num_buckets,
        'BATCH_SIZE': args.batch_size,
        'EQUITY_ROLLOUTS': args.equity_rollouts,
        'HAND_EVAL_PROCESSES': args.hand_eval_processes,
        'LOG_INTERVAL': args.log_interval,
        'CHECKPOINT_INTERVAL': args.checkpoint_interval,
        'MAX_DEPTH': args.max_depth,
        'SEED': args.seed,
        'RAY_NUM_CPUS': args.ray_cpus,
        'RAY_NUM_GPUS': args.ray_gpus,
        'INFOSET_KEY_MODE': args.key_mode,
        'NN_BATCH_SIZE': args.nn_batch_size,
        'NN_TRAIN_STEPS': args.nn_train_steps,
        'NN_LEARNING_RATE': args.nn_learning_rate,
        'DEEP_CFR_TRAVERSALS_PER_ITER': args.deep_traversals_per_iter,
    }
    for attr, value in override_map.items():
        if value is not None:
            setattr(Config, attr, value)

    if Config.ALGORITHM_MODE == 'deep':
        Config.INFOSET_KEY_MODE = 'state'

    Config.RUN_UNTIL_STOP = bool(args.long_run and args.iterations is None)
    Infoset.KEY_MODE = Config.INFOSET_KEY_MODE


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_runtime():
    configure_threading()
    if Config.ALGORITHM_MODE != 'tabular':
        return
    if ray is not None and not ray.is_initialized():
        ray.init(num_cpus=Config.RAY_NUM_CPUS, num_gpus=Config.RAY_NUM_GPUS, ignore_reinit_error=True)


def configure_threading():
    try:
        torch.set_num_threads(Config.TORCH_THREADS_MAIN)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def get_vram_usage():
    usage = {
        'allocated_gb': 0.0,
        'reserved_gb': 0.0,
        'driver_used_gb': 0.0,
        'free_gb': 0.0,
        'total_gb': 0.0,
        'utilization_percent': 0.0,
    }
    if not Config.HAS_CUDA:
        return usage

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        usage['total_gb'] = props.total_memory / (1024 ** 3)
        usage['allocated_gb'] = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        usage['reserved_gb'] = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            usage['free_gb'] = free_bytes / (1024 ** 3)
            usage['driver_used_gb'] = max(total_bytes - free_bytes, 0) / (1024 ** 3)
        except Exception:
            usage['driver_used_gb'] = max(usage['allocated_gb'], usage['reserved_gb'])
        peak = max(usage['allocated_gb'], usage['reserved_gb'], usage['driver_used_gb'])
        usage['utilization_percent'] = (peak / usage['total_gb']) * 100.0 if usage['total_gb'] > 0 else 0.0
    except Exception as exc:
        logging.debug("Unable to read VRAM usage: %s", exc)
    return usage


def get_ram_usage():
    usage = {
        'percent': 0.0,
        'used_gb': 0.0,
        'available_gb': 0.0,
        'process_gb': 0.0,
    }
    try:
        vm = psutil.virtual_memory()
        usage['percent'] = float(vm.percent)
        usage['used_gb'] = vm.used / (1024 ** 3)
        usage['available_gb'] = vm.available / (1024 ** 3)
        usage['process_gb'] = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception as exc:
        logging.debug("Unable to read RAM usage: %s", exc)
    return usage


def clear_runtime_caches(reason=None, aggressive=False):
    if reason:
        logging.debug("Clearing device caches: %s", reason)
    Config.clear_device_cache(aggressive=aggressive)


def get_training_budget():
    vram_usage = get_vram_usage()
    ram_usage = get_ram_usage()
    scale_divisor = 1
    reasons = []

    if max(vram_usage['allocated_gb'], vram_usage['reserved_gb'], vram_usage['driver_used_gb']) > Config.MAX_VRAM_BEFORE_BACKOFF_GB:
        scale_divisor *= 2
        reasons.append(f"VRAM>{Config.MAX_VRAM_BEFORE_BACKOFF_GB:.1f}GB")
    if ram_usage['percent'] > Config.MAX_RAM_UTILIZATION_PERCENT:
        scale_divisor *= 2
        reasons.append(f"RAM>{Config.MAX_RAM_UTILIZATION_PERCENT:.0f}%")

    budget = {
        'batch_size': Config.BATCH_SIZE,
        'equity_rollouts': Config.EQUITY_ROLLOUTS,
        'nn_batch_size': Config.NN_BATCH_SIZE,
        'nn_train_steps': Config.NN_TRAIN_STEPS,
        'deep_traversals_per_iter': Config.DEEP_CFR_TRAVERSALS_PER_ITER,
        'throttled': scale_divisor > 1,
        'reason': '; '.join(reasons),
    }
    if budget['throttled']:
        budget['batch_size'] = max(Config.MIN_SIM_BATCH_SIZE, Config.BATCH_SIZE // scale_divisor)
        budget['equity_rollouts'] = max(Config.MIN_EQUITY_ROLLOUTS, Config.EQUITY_ROLLOUTS // scale_divisor)
        budget['nn_batch_size'] = max(Config.MIN_NN_BATCH_SIZE, Config.NN_BATCH_SIZE // scale_divisor)
        budget['nn_train_steps'] = max(Config.MIN_NN_TRAIN_STEPS, Config.NN_TRAIN_STEPS // scale_divisor)
        budget['deep_traversals_per_iter'] = max(
            Config.MIN_DEEP_TRAVERSALS_PER_ITER,
            Config.DEEP_CFR_TRAVERSALS_PER_ITER // scale_divisor,
        )

    return budget, vram_usage, ram_usage


def serialize_strategy_key(key):
    return json.dumps(key)


def collect_strategies(infosets=None, actor=None, node_store=None, deep_agent=None):
    if deep_agent is not None:
        return deep_agent.export_average_strategies()

    raw_strategies = {}
    json_strategies = {}
    for inf in infosets:
        if actor is not None:
            strategy_sum = ray.get(actor.get_strategy_sum.remote(inf.key))
            strategy = average_strategy(inf, actor=actor).cpu().tolist() if strategy_sum.sum() > 0 else None
        else:
            strategy_sum = node_store.get_strategy_sum(inf.key)
            strategy = average_strategy(inf, node_store=node_store).cpu().tolist() if strategy_sum.sum() > 0 else None
        if strategy_sum.sum() > 0:
            raw_strategies[inf.key] = strategy
            json_strategies[serialize_strategy_key(inf.key)] = strategy
    return raw_strategies, json_strategies


def save_strategies(infosets, prefix, actor=None, node_store=None, deep_agent=None):
    if not infosets:
        return
    raw_strategies, json_strategies = collect_strategies(
        infosets,
        actor=actor,
        node_store=node_store,
        deep_agent=deep_agent,
    )
    np.save(f'{prefix}.npy', raw_strategies)
    with open(f'{prefix}.json', 'w') as handle:
        json.dump(json_strategies, handle)


def save_checkpoint_bundle(infosets, prefix, iteration, actor=None, node_store=None, deep_agent=None):
    save_strategies(infosets, prefix, actor=actor, node_store=node_store, deep_agent=deep_agent)
    if deep_agent is not None:
        deep_agent.save_checkpoint(f'{prefix}.pt', iteration=iteration)


def iteration_iterator(start_iteration):
    if Config.RUN_UNTIL_STOP:
        return itertools.count(start_iteration)
    return range(start_iteration, Config.ITERATIONS)


def build_root_infosets():
    features = simulate_features()
    buckets, _ = create_buckets(features)
    unique_buckets = np.unique(buckets)
    return [
        Infoset(
            int(bid),
            pot_size=Config.POT_SIZE,
            stack_sizes=Config.DEFAULT_STACK_SIZES,
            current_bet=Config.CALL_AMOUNT,
        )
        for bid in unique_buckets
    ]


def train(args=None):
    global infosets, actor

    initialize_runtime()
    set_global_seeds(Config.SEED)
    writer = SummaryWriter()
    infosets = build_root_infosets()
    actor = None
    node_store = None
    deep_agent = None
    start_iteration = 0
    last_iteration = start_iteration - 1
    best_metric = float('-inf')
    last_budget_signature = None

    logging.info(
        "Starting training | mode=%s env=%s key_mode=%s backend=%s device=%s sim_device=%s hip=%s safe_hw=%s buckets=%s iterations=%s depth=%s seed=%s long_run=%s",
        Config.ALGORITHM_MODE,
        Config.ENVIRONMENT_MODE,
        Config.INFOSET_KEY_MODE,
        'ray' if Config.ALGORITHM_MODE == 'tabular' and ray is not None else 'local',
        Config.DEVICE,
        Config.SIMULATION_DEVICE,
        Config.IS_HIP,
        Config.SAFE_HARDWARE_MODE,
        Config.NUM_BUCKETS,
        Config.ITERATIONS,
        Config.MAX_DEPTH,
        Config.SEED,
        Config.RUN_UNTIL_STOP,
    )

    def local_sigterm_handler(signum, frame):
        logging.error("SIGTERM received; saving partial strategies and exiting.")
        save_checkpoint_bundle(
            infosets,
            'partial_strategies',
            iteration=last_iteration,
            actor=actor,
            node_store=node_store,
            deep_agent=deep_agent,
        )
        sys.exit(1)

    signal.signal(signal.SIGTERM, local_sigterm_handler)

    try:
        if Config.ALGORITHM_MODE == 'deep':
            deep_agent = DeepCFRAgent()
            if args is not None and args.resume_checkpoint:
                loaded_iteration = deep_agent.load_checkpoint(args.resume_checkpoint)
                start_iteration = max(0, loaded_iteration + 1)
                last_iteration = loaded_iteration
                logging.info(
                    "Loaded Deep CFR checkpoint %s at iteration %s.",
                    args.resume_checkpoint,
                    loaded_iteration,
                )
            logging.info("Deep mode uses local replay buffers and state-key infosets.")
        else:
            if args is not None and args.resume_checkpoint:
                logging.warning("Ignoring --resume-checkpoint in tabular mode; only Deep CFR checkpoints are supported.")
            actor = NodesActor.remote() if ray is not None else None
            node_store = None if actor is not None else LocalNodeStore()
            if actor is not None:
                ray.get(actor.get_all_keys.remote())
                logging.info("NodesActor initialized successfully.")
            else:
                logging.info("Ray unavailable; using local node storage.")
    except Exception as exc:
        logging.error(f"Actor initialization failed: {exc}")
        raise

    try:
        for it in iteration_iterator(start_iteration):
            last_iteration = it
            training_budget, _, _ = get_training_budget()
            Config.apply_runtime_limits(
                batch_size=training_budget['batch_size'],
                equity_rollouts=training_budget['equity_rollouts'],
                nn_batch_size=training_budget['nn_batch_size'],
                nn_train_steps=training_budget['nn_train_steps'],
                deep_traversals_per_iter=training_budget['deep_traversals_per_iter'],
            )
            budget_signature = (
                Config.current_batch_size(),
                Config.current_equity_rollouts(),
                Config.current_nn_batch_size(),
                Config.current_nn_train_steps(),
                Config.current_deep_traversals_per_iter(),
            )
            if training_budget['throttled'] and budget_signature != last_budget_signature:
                logging.warning(
                    "Memory backoff applied at iter %s: sim_batch=%s rollouts=%s nn_batch=%s nn_steps=%s traversals=%s (%s)",
                    it,
                    Config.current_batch_size(),
                    Config.current_equity_rollouts(),
                    Config.current_nn_batch_size(),
                    Config.current_nn_train_steps(),
                    Config.current_deep_traversals_per_iter(),
                    training_budget['reason'],
                )
                clear_runtime_caches(reason=training_budget['reason'], aggressive=True)
            last_budget_signature = budget_signature

            if deep_agent is not None:
                utils = []
                for _ in range(Config.current_deep_traversals_per_iter()):
                    for inf in infosets:
                        utils.append(deep_cfr_traverse(inf, it, deep_agent=deep_agent, max_depth=Config.MAX_DEPTH))
                advantage_losses = [
                    deep_agent.train_advantage_network(
                        player,
                        steps=Config.current_nn_train_steps(),
                        batch_size=Config.current_nn_batch_size(),
                    )
                    for player in range(len(Config.DEFAULT_STACK_SIZES))
                ]
                average_loss = deep_agent.train_average_network(
                    steps=Config.current_nn_train_steps(),
                    batch_size=Config.current_nn_batch_size(),
                )
            else:
                if actor is not None:
                    futures = [run_mccfr.remote(inf, it, actor, Config.MAX_DEPTH) for inf in infosets]
                    utils = ray.get(futures)
                else:
                    utils = [mccfr(inf, it, node_store=node_store, max_depth=Config.MAX_DEPTH) for inf in infosets]

            total_util = float(np.mean(utils))
            if not np.isfinite(total_util):
                logging.warning("Non-finite utility observed at iter %s; substituting 0.0 for logging stability.", it)
                total_util = 0.0

            should_log = (it % Config.LOG_INTERVAL == 0)
            should_checkpoint = (it > 0 and it % Config.CHECKPOINT_INTERVAL == 0)
            approx_hands = (it + 1) * len(infosets) * (
                Config.current_deep_traversals_per_iter() if deep_agent is not None else 1
            )

            if should_log:
                vram_usage = get_vram_usage()
                ram_usage = get_ram_usage()
                cpu_percent = psutil.cpu_percent(interval=None)
                if deep_agent is not None:
                    buffer_sizes = deep_agent.buffer_sizes()
                    writer.add_scalar('Run/ApproxHands', approx_hands, it)
                    writer.add_scalar('Util/Avg', total_util, it)
                    writer.add_scalar('Loss/AdvantagePlayer0', advantage_losses[0], it)
                    writer.add_scalar('Loss/AdvantagePlayer1', advantage_losses[1], it)
                    writer.add_scalar('Loss/AverageStrategy', average_loss, it)
                    writer.add_scalar('Replay/AdvantagePlayer0', buffer_sizes['advantage_0'], it)
                    writer.add_scalar('Replay/AdvantagePlayer1', buffer_sizes['advantage_1'], it)
                    writer.add_scalar('Replay/AverageStrategy', buffer_sizes['strategy'], it)
                    writer.add_scalar('Runtime/SimulationBatchSize', Config.current_batch_size(), it)
                    writer.add_scalar('Runtime/EquityRollouts', Config.current_equity_rollouts(), it)
                    writer.add_scalar('Training/AdaptiveBatchSize', Config.current_nn_batch_size(), it)
                    writer.add_scalar('Training/AdaptiveTrainSteps', Config.current_nn_train_steps(), it)
                    writer.add_scalar('Training/DeepTraversals', Config.current_deep_traversals_per_iter(), it)
                    writer.add_scalar('Training/SkippedAdvantagePlayer0', deep_agent.skipped_batches['advantage_0'], it)
                    writer.add_scalar('Training/SkippedAdvantagePlayer1', deep_agent.skipped_batches['advantage_1'], it)
                    writer.add_scalar('Training/SkippedAverage', deep_agent.skipped_batches['strategy'], it)
                    writer.add_scalar('Memory/VRAMAllocatedGB', vram_usage['allocated_gb'], it)
                    writer.add_scalar('Memory/VRAMReservedGB', vram_usage['reserved_gb'], it)
                    writer.add_scalar('Memory/VRAMDriverUsedGB', vram_usage['driver_used_gb'], it)
                    writer.add_scalar('Memory/RAMPercent', ram_usage['percent'], it)
                    writer.add_scalar('Memory/ProcessRSSGB', ram_usage['process_gb'], it)
                    logging.info(
                        "Iter %s | hands~%s | util %.2f | adv_loss [%.4f, %.4f] | avg_loss %.4f | replay %s | skipped %s | sim_batch=%s rollouts=%s nn_batch=%s nn_steps=%s traversals=%s | VRAM %.2f/%.2f GB reserved %.2f driver %.2f | RAM %.1f%% (proc %.2f GB) | CPU %.1f%%",
                        it,
                        approx_hands,
                        total_util,
                        advantage_losses[0],
                        advantage_losses[1],
                        average_loss,
                        buffer_sizes,
                        deep_agent.skipped_batches,
                        Config.current_batch_size(),
                        Config.current_equity_rollouts(),
                        Config.current_nn_batch_size(),
                        Config.current_nn_train_steps(),
                        Config.current_deep_traversals_per_iter(),
                        vram_usage['allocated_gb'],
                        vram_usage['total_gb'],
                        vram_usage['reserved_gb'],
                        vram_usage['driver_used_gb'],
                        ram_usage['percent'],
                        ram_usage['process_gb'],
                        cpu_percent,
                    )
                else:
                    if actor is not None:
                        regret_sums = ray.get([actor.get_regret_sum.remote(inf.key) for inf in infosets])
                    else:
                        regret_sums = [node_store.get_regret_sum(inf.key).cpu() for inf in infosets]
                    regrets = [rs.mean().item() for rs in regret_sums]
                    avg_regret = float(np.mean(regrets)) if regrets else 0.0
                    writer.add_scalar('Run/ApproxHands', approx_hands, it)
                    writer.add_scalar('Util/Avg', total_util, it)
                    writer.add_scalar('Regret/Avg', avg_regret, it)
                    writer.add_scalar('Runtime/SimulationBatchSize', Config.current_batch_size(), it)
                    writer.add_scalar('Runtime/EquityRollouts', Config.current_equity_rollouts(), it)
                    writer.add_scalar('Memory/VRAMAllocatedGB', vram_usage['allocated_gb'], it)
                    writer.add_scalar('Memory/VRAMReservedGB', vram_usage['reserved_gb'], it)
                    writer.add_scalar('Memory/VRAMDriverUsedGB', vram_usage['driver_used_gb'], it)
                    writer.add_scalar('Memory/RAMPercent', ram_usage['percent'], it)
                    writer.add_scalar('Memory/ProcessRSSGB', ram_usage['process_gb'], it)
                    logging.info(
                        f"Iter {it} | hands~{approx_hands} | Util {total_util:.2f}, Regret {avg_regret:.4f} | "
                        f"sim_batch={Config.current_batch_size()} rollouts={Config.current_equity_rollouts()} | "
                        f"VRAM {vram_usage['allocated_gb']:.2f}/{vram_usage['total_gb']:.2f} GB reserved {vram_usage['reserved_gb']:.2f} driver {vram_usage['driver_used_gb']:.2f} | "
                        f"RAM {ram_usage['percent']:.1f}% (proc {ram_usage['process_gb']:.2f} GB) | CPU: {cpu_percent:.1f}% | "
                        f"Ray Dashboard: http://127.0.0.1:8265"
                    )

            if Config.SAVE_BEST_MODEL and np.isfinite(total_util) and total_util > best_metric:
                best_metric = total_util
                save_checkpoint_bundle(
                    infosets,
                    'best_strategies',
                    iteration=it,
                    actor=actor,
                    node_store=node_store,
                    deep_agent=deep_agent,
                )
                if deep_agent is not None:
                    deep_agent.save_checkpoint('best_model.pt', iteration=it)

            if should_checkpoint:
                save_checkpoint_bundle(
                    infosets,
                    f'checkpoint_{it}',
                    iteration=it,
                    actor=actor,
                    node_store=node_store,
                    deep_agent=deep_agent,
                )

            if should_log or should_checkpoint or training_budget['throttled']:
                clear_runtime_caches(
                    reason='iteration_maintenance',
                    aggressive=bool(training_budget['throttled'] or should_checkpoint),
                )
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received; finalizing current artifacts.")
    finally:
        save_checkpoint_bundle(
            infosets,
            'strategies',
            iteration=last_iteration,
            actor=actor,
            node_store=node_store,
            deep_agent=deep_agent,
        )
        logging.info("Training complete; strategies saved.")
        writer.close()
        Config.reset_runtime_limits()
        close_hand_eval_pool()
        clear_runtime_caches(reason='shutdown', aggressive=True)
        if ray is not None and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_overrides(args)
    torch.set_default_dtype(Config.DTYPE if Config.ALGORITHM_MODE == 'tabular' else Config.NN_DTYPE)
    train(args=args)
