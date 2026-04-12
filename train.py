infosets = None
actor = None
active_agent = None
writer = None

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')

from environment import clear_runtime_caches, get_memory_snapshot, setup_rocmo

setup_rocmo()

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)

from abstractions import create_buckets, simulate_features
from cfr import apply_regret_matching_boost, average_strategy, mccfr
from config import Config
from datatypes import Infoset
from deep_cfr import DeepCFRAgent, resolve_checkpoint_path


try:
    mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Poker2 with Deep CFR as the primary AMD-safe backend.')
    parser.add_argument('--mode', choices=['deep', 'mccfr'], default='deep')
    parser.add_argument('--smoke-test', dest='smoke_test', action='store_true', help='Run a short validation profile.')
    parser.add_argument('--test-run', dest='smoke_test', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--long-run', action='store_true', help='Apply the unattended long-run profile.')
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--num-sims', type=int, default=None)
    parser.add_argument('--num-buckets', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--task-batch', type=int, default=None)
    parser.add_argument('--traversals', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    return parser.parse_args()


def apply_runtime_profile(args):
    if args.smoke_test and args.long_run:
        raise ValueError('Use either --smoke-test or --long-run, not both.')

    if args.smoke_test:
        Config.ITERATIONS = Config.SMOKE_TEST_ITERATIONS
        Config.NUM_SIMS = Config.SMOKE_TEST_NUM_SIMS
        Config.NUM_SIMULATIONS = Config.SMOKE_TEST_NUM_SIMS
        Config.NUM_BUCKETS = Config.SMOKE_TEST_NUM_BUCKETS
        Config.BATCH_SIZE = Config.SMOKE_TEST_BATCH_SIZE
        Config.MAX_BATCH_SIZE = max(Config.MAX_BATCH_SIZE, Config.BATCH_SIZE)
        Config.NUM_TRAVERSALS = Config.SMOKE_TEST_TRAVERSALS
        Config.REPLAY_WARMUP_SAMPLES = Config.SMOKE_TEST_REPLAY_WARMUP
        Config.REPLAY_BUFFER_SIZE = Config.SMOKE_TEST_REPLAY_BUFFER_SIZE
        Config.LOG_INTERVAL = Config.SMOKE_TEST_LOG_INTERVAL
        Config.CHECKPOINT_INTERVAL = Config.SMOKE_TEST_CHECKPOINT_INTERVAL
        Config.EQUITY_ROLLOUTS = Config.SMOKE_TEST_EQUITY_ROLLOUTS
        Config.RAY_TASK_BATCH = min(Config.RAY_TASK_BATCH, 32)
        Config.MP_PROCESSES = min(Config.MP_PROCESSES, 4)
    elif args.long_run:
        Config.ITERATIONS = Config.LONG_RUN_ITERATIONS

    if args.iterations is not None:
        Config.ITERATIONS = args.iterations
    if args.num_sims is not None:
        Config.NUM_SIMS = args.num_sims
        Config.NUM_SIMULATIONS = args.num_sims
    if args.num_buckets is not None:
        Config.NUM_BUCKETS = args.num_buckets
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.task_batch is not None:
        Config.RAY_TASK_BATCH = args.task_batch
    if args.traversals is not None:
        Config.NUM_TRAVERSALS = args.traversals
    if args.log_interval is not None:
        Config.LOG_INTERVAL = args.log_interval
    if args.checkpoint_interval is not None:
        Config.CHECKPOINT_INTERVAL = args.checkpoint_interval


def get_abstraction_cache_paths():
    os.makedirs(Config.ABSTRACTION_CACHE_DIR, exist_ok=True)
    cache_key = f'{Config.NUM_BUCKETS}b_{Config.NUM_SIMS}s'
    buckets_path = os.path.join(Config.ABSTRACTION_CACHE_DIR, f'buckets_{cache_key}.npy')
    centroids_path = os.path.join(Config.ABSTRACTION_CACHE_DIR, f'centroids_{cache_key}.npy')
    return buckets_path, centroids_path


def load_or_create_abstractions():
    buckets_path, centroids_path = get_abstraction_cache_paths()
    legacy_centroids_path = 'centroids.npy'
    centroids = None
    if os.path.exists(centroids_path):
        try:
            loaded_centroids = np.load(centroids_path)
            if loaded_centroids.ndim == 2 and loaded_centroids.shape[0] == Config.NUM_BUCKETS:
                centroids = loaded_centroids.astype(np.float32)
                logger.info('Loaded cached centroids from %s', centroids_path)
        except Exception as load_error:
            logger.warning('Ignoring cached centroids because they failed to load: %s', load_error)

    if centroids is None and os.path.exists(legacy_centroids_path):
        try:
            loaded_centroids = np.load(legacy_centroids_path)
            if loaded_centroids.ndim == 2 and loaded_centroids.shape[0] == Config.NUM_BUCKETS:
                centroids = loaded_centroids.astype(np.float32)
                logger.info('Loaded legacy centroids from %s', legacy_centroids_path)
        except Exception as load_error:
            logger.warning('Ignoring legacy centroids because they failed to load: %s', load_error)

    if centroids is None:
        logger.info('Generating abstractions with %s simulations and %s buckets', Config.NUM_SIMS, Config.NUM_BUCKETS)
        features = simulate_features(num_sims=Config.NUM_SIMS)
        buckets, centroids = create_buckets(features, num_buckets=Config.NUM_BUCKETS)
        np.save(buckets_path, buckets)
        np.save(centroids_path, centroids)

    root_infosets = [Infoset(bucket_id) for bucket_id in range(len(centroids))]
    return root_infosets, np.asarray(centroids, dtype=np.float32)


def load_deep_resume_state(checkpoint_path):
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint_state = torch.load(resolved_path, map_location='cpu', weights_only=False)
    if checkpoint_state.get('mode') not in (None, 'deep'):
        raise ValueError(f'Checkpoint {resolved_path} is not a Deep CFR checkpoint.')

    centroids = checkpoint_state.get('bucket_centroids')
    if centroids is None:
        return resolved_path, checkpoint_state, None, None

    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[0] <= 0:
        return resolved_path, checkpoint_state, None, None

    infosets = [Infoset(bucket_id) for bucket_id in range(len(centroids))]
    logger.info('Loaded Deep CFR abstraction metadata from %s', resolved_path)
    return resolved_path, checkpoint_state, infosets, centroids


def init_ray():
    if ray.is_initialized():
        return
    ray.init(
        num_cpus=Config.RAY_NUM_CPUS,
        num_gpus=1 if torch.cuda.is_available() else 0,
        ignore_reinit_error=True,
        runtime_env={
            'env_vars': {
                'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', '0'),
                'PYTORCH_HIP_ALLOC_CONF': os.environ.get('PYTORCH_HIP_ALLOC_CONF', ''),
                'HSA_OVERRIDE_GFX_VERSION': os.environ.get('HSA_OVERRIDE_GFX_VERSION', '11.0.0'),
                'PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING': os.environ.get('PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING', '1'),
                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '1'),
            }
        },
    )


def save_strategies(output_path):
    if actor is None or infosets is None or not ray.is_initialized():
        return

    strategies = {}
    for infoset in infosets:
        strategy_sum = ray.get(actor.get_strategy_sum.remote(infoset.key))
        if strategy_sum.sum() > 0:
            strategies[infoset.key] = average_strategy(infoset, actor=actor).cpu().tolist()
    np.save(output_path, strategies)


def save_mccfr_checkpoint(iteration):
    if actor is None or infosets is None or not ray.is_initialized():
        return

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    strategies = {}
    for infoset in infosets:
        strategy_sum = ray.get(actor.get_strategy_sum.remote(infoset.key))
        if strategy_sum.sum() > 0:
            strategies[infoset.key] = average_strategy(infoset, actor=actor).cpu().tolist()

    checkpoint_state = {'mode': 'mccfr', 'iteration': iteration, 'strategies': strategies}
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'mccfr_checkpoint_iter_{iteration:07d}.pt')
    latest_path = os.path.join(Config.CHECKPOINT_DIR, 'mccfr_checkpoint_latest.pt')
    torch.save(checkpoint_state, checkpoint_path)
    torch.save(checkpoint_state, latest_path)


def sigterm_handler(signum, frame):
    logger.error('SIGTERM received; attempting to save runtime state before exit.')
    if active_agent is not None:
        try:
            active_agent.save_checkpoint(active_agent.iteration, {'signal': 'sigterm', 'avg_utility': 0.0, 'exploitability_proxy': 0.0})
        except Exception as checkpoint_error:
            logger.error('Failed to save Deep CFR checkpoint on SIGTERM: %s', checkpoint_error)
    else:
        save_strategies('partial_strategies.npy')
    sys.exit(1)


signal.signal(signal.SIGTERM, sigterm_handler)


@ray.remote(num_cpus=0)
class NodesActor:
    def __init__(self):
        self.nodes = defaultdict(lambda: {
            'regret_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu'),
            'strategy_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu'),
        })

    def get_regret_sum(self, key):
        return self.nodes[key]['regret_sum']

    def get_strategy_sum(self, key):
        return self.nodes[key]['strategy_sum']

    def update_regret_sum(self, key, delta):
        self.nodes[key]['regret_sum'] += delta.cpu()

    def update_strategy_sum(self, key, delta):
        self.nodes[key]['strategy_sum'] += delta.cpu()

    def apply_regret_matching_boost(self, weight):
        for node in self.nodes.values():
            node['strategy_sum'] = apply_regret_matching_boost(node['regret_sum'], node['strategy_sum'], weight).cpu()


@ray.remote(num_cpus=1)
def run_mccfr(infoset, iteration, actor_handle, max_depth, num_opponents):
    Config.NUM_OPPONENTS = max(1, num_opponents)
    return mccfr(infoset, iteration, actor=actor_handle, max_depth=max_depth)


def train_mccfr():
    global infosets, actor

    if writer is None:
        raise RuntimeError('TensorBoard writer not initialized')

    if active_agent is not None:
        logger.warning('Deep CFR state is active; MCCFR will run as a fallback from a fresh tree state.')

    if infosets is None:
        infosets, _ = load_or_create_abstractions()

    init_ray()
    actor = NodesActor.remote()
    current_task_batch = min(Config.RAY_TASK_BATCH, Config.RAY_WORKER_LIMIT * 2)
    current_batch_size = Config.BATCH_SIZE
    current_rollouts = Config.EQUITY_ROLLOUTS
    backoff_events = 0
    last_backoff_iteration = -Config.BACKOFF_COOLDOWN

    logger.info('Starting MCCFR fallback with %s infosets', len(infosets))

    avg_utility = 0.0
    avg_regret = 0.0
    exploitability_proxy = 0.0

    for iteration in range(Config.ITERATIONS):
        current_max_depth = min(Config.MAX_CFR_DEPTH, Config.START_MAX_DEPTH + (iteration // Config.CURRICULUM_INTERVAL))
        current_num_opponents = min(Config.MAX_CURRICULUM_OPPONENTS, 1 + (iteration // Config.CURRICULUM_INTERVAL))
        all_utilities = []

        for start in range(0, len(infosets), current_task_batch):
            infoset_chunk = infosets[start:start + current_task_batch]
            futures = [run_mccfr.remote(infoset, iteration, actor, current_max_depth, current_num_opponents) for infoset in infoset_chunk]
            all_utilities.extend(ray.get(futures))

        avg_utility = float(np.mean(all_utilities)) if all_utilities else 0.0
        regret_sums = ray.get([actor.get_regret_sum.remote(infoset.key) for infoset in infosets])
        avg_regret = float(np.mean([regret_sum.mean().item() for regret_sum in regret_sums])) if regret_sums else 0.0
        exploitability_proxy = float(
            np.mean([max(0.0, regret_sum.max().item() - regret_sum.mean().item()) for regret_sum in regret_sums])
        ) if regret_sums else 0.0
        snapshot = get_memory_snapshot()
        backoff_label = 'none'

        if snapshot['used_gb'] > Config.VRAM_SOFT_LIMIT_GB or snapshot['ram_pct'] > Config.RAM_SOFT_LIMIT_PCT:
            current_task_batch = max(Config.MIN_RAY_TASK_BATCH, current_task_batch // 2)
            current_batch_size = max(Config.MIN_BATCH_SIZE, current_batch_size // 2)
            current_rollouts = max(Config.MIN_EQUITY_ROLLOUTS, current_rollouts // 2)
            Config.BATCH_SIZE = current_batch_size
            Config.EQUITY_ROLLOUTS = current_rollouts
            backoff_events += 1
            last_backoff_iteration = iteration
            backoff_label = 'memory-pressure'
            clear_runtime_caches()
        elif backoff_events > 0 and iteration - last_backoff_iteration >= Config.BACKOFF_COOLDOWN:
            if snapshot['used_gb'] < (Config.VRAM_SOFT_LIMIT_GB * 0.75) and snapshot['ram_pct'] < (Config.RAM_SOFT_LIMIT_PCT - 8.0):
                current_task_batch = min(Config.RAY_TASK_BATCH, current_task_batch * 2)
                current_batch_size = min(Config.MAX_BATCH_SIZE, current_batch_size * 2)
                current_rollouts = min(64, current_rollouts * 2)
                Config.BATCH_SIZE = current_batch_size
                Config.EQUITY_ROLLOUTS = current_rollouts
                backoff_label = 'recovered'

        if iteration % Config.LOG_INTERVAL == 0:
            writer.add_scalar('MCCFR/AvgUtility', avg_utility, iteration)
            writer.add_scalar('MCCFR/AvgRegret', avg_regret, iteration)
            writer.add_scalar('MCCFR/ExploitabilityProxy', exploitability_proxy, iteration)
            writer.add_scalar('MCCFR/BatchSize', current_batch_size, iteration)
            writer.add_scalar('MCCFR/BackoffEvents', backoff_events, iteration)
            writer.add_scalar('System/VRAMPct', snapshot['used_pct'], iteration)
            writer.add_scalar('System/RAMPct', snapshot['ram_pct'], iteration)
            logger.info(
                'Iter %s | avg_utility %.4f | exploitability %.4f | vram %.1f%% (%.2f/%.2f GB) | ram %.1f%% | batch %s | backoff %s',
                iteration,
                avg_utility,
                exploitability_proxy,
                snapshot['used_pct'],
                snapshot['used_gb'],
                snapshot['total_gb'],
                snapshot['ram_pct'],
                current_batch_size,
                backoff_label,
            )

        if iteration > 0 and iteration % Config.HYBRID_UPDATE_INTERVAL == 0:
            ray.get(actor.apply_regret_matching_boost.remote(Config.HYBRID_BOOST_WEIGHT))

        if iteration % Config.CHECKPOINT_INTERVAL == 0:
            save_mccfr_checkpoint(iteration)

    save_strategies('strategies.npy')
    return {
        'avg_utility': avg_utility,
        'avg_regret': avg_regret,
        'exploitability_proxy': exploitability_proxy,
        'backoff_events': backoff_events,
    }


def train_deep_cfr(args):
    global active_agent, infosets

    start_iteration = 0
    checkpoint_state = None
    resolved_resume_path = None

    if args.resume_checkpoint:
        resolved_resume_path, checkpoint_state, resumed_infosets, resumed_centroids = load_deep_resume_state(args.resume_checkpoint)
        if resumed_infosets is not None and resumed_centroids is not None:
            infosets, centroids = resumed_infosets, resumed_centroids
        else:
            infosets, centroids = load_or_create_abstractions()
    else:
        infosets, centroids = load_or_create_abstractions()

    active_agent = DeepCFRAgent(infosets=infosets, bucket_centroids=centroids, writer=writer, resume=checkpoint_state is not None)
    if checkpoint_state is not None:
        start_iteration = active_agent.load_checkpoint_state(checkpoint_state, resolved_resume_path)

    logger.info(
        'Starting Deep CFR with %s infosets | iterations=%s | batch=%s | traversals=%s | safe_mode=%s',
        len(infosets),
        Config.ITERATIONS,
        Config.BATCH_SIZE,
        Config.NUM_TRAVERSALS,
        Config.SAFE_HARDWARE_MODE,
    )
    return active_agent.train(iterations=Config.ITERATIONS, start_iteration=start_iteration)


def main():
    global writer, active_agent

    args = parse_args()
    apply_runtime_profile(args)
    writer = SummaryWriter()
    initial_snapshot = get_memory_snapshot()
    logger.info(
        'Launcher mode=%s | smoke_test=%s | long_run=%s | vram=%.2f/%.2f GB | ram=%.1f%%',
        args.mode,
        args.smoke_test,
        args.long_run,
        initial_snapshot['used_gb'],
        initial_snapshot['total_gb'],
        initial_snapshot['ram_pct'],
    )

    try:
        if args.mode == 'deep':
            try:
                metrics = train_deep_cfr(args)
                logger.info('Deep CFR finished with metrics: %s', metrics)
            except Exception as deep_error:
                logger.exception('Deep CFR failed; falling back to MCCFR: %s', deep_error)
                active_agent = None
                metrics = train_mccfr()
                logger.info('MCCFR fallback finished with metrics: %s', metrics)
        else:
            if args.resume_checkpoint:
                logger.warning('Resume checkpoints are only supported for Deep CFR in this pass; MCCFR will start fresh.')
            metrics = train_mccfr()
            logger.info('MCCFR finished with metrics: %s', metrics)
    finally:
        if writer is not None:
            writer.close()
        if ray.is_initialized():
            ray.shutdown()
        clear_runtime_caches()


if __name__ == '__main__':
    main()