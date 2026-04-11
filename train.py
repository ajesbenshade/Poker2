infosets = None
actor = None

import os
import signal
import sys
import logging
import multiprocessing as mp
from collections import defaultdict

# Set ROCm environment defaults before importing torch/ray so worker processes inherit them.
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:256')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ['PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING'] = '1'

import numpy as np
import psutil
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from abstractions import simulate_features, create_buckets
from cfr import apply_regret_matching_boost, average_strategy, mccfr
from config import Config
from datatypes import Infoset

try:
    # Forkserver avoids copying a large ROCm state into child workers during long runs.
    mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

writer = SummaryWriter()

# Keep logging on both stdout and file for overnight training sessions.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def maybe_compile_models():
    if torch.cuda.is_available() and hasattr(Config, 'EQUITY_MODEL'):
        try:
            # ROCm-optimized torch.compile for 7900XT speed boost on the shared equity model.
            Config.EQUITY_MODEL = torch.compile(Config.EQUITY_MODEL, mode='max-autotune', fullgraph=False)
            logger.info('Compiled Config.EQUITY_MODEL with torch.compile(max-autotune).')
        except Exception as compile_error:
            logger.warning(f'Falling back to eager EquityNet execution: {compile_error}')


def init_ray():
    if not ray.is_initialized():
        # Limit Ray CPU slots so the nested equity-evaluation pool does not oversubscribe the 7900X.
        ray.init(
            num_cpus=Config.RAY_NUM_CPUS,
            num_gpus=1 if torch.cuda.is_available() else 0,
            ignore_reinit_error=True,
            runtime_env={
                'env_vars': {
                    'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', '0'),
                    'PYTORCH_HIP_ALLOC_CONF': os.environ.get('PYTORCH_HIP_ALLOC_CONF', ''),
                    'HSA_OVERRIDE_GFX_VERSION': os.environ.get('HSA_OVERRIDE_GFX_VERSION', '11.0.0'),
                }
            },
        )


def get_vram_usage_gb():
    # Track allocated VRAM so the loop can back off before ROCm OOMs.
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def get_ram_usage_pct():
    # Host RAM tracking keeps large regret tables stable during day-long training.
    return psutil.virtual_memory().percent


def save_strategies(output_path):
    if actor is None or infosets is None:
        return

    strategies = {}
    for infoset in infosets:
        strategy_sum = ray.get(actor.get_strategy_sum.remote(infoset.key))
        if strategy_sum.sum() > 0:
            strategies[infoset.key] = average_strategy(infoset, actor=actor).cpu().tolist()
    np.save(output_path, strategies)


def sigterm_handler(signum, frame):
    logger.error('SIGTERM received; saving partial strategies and exiting.')
    save_strategies('partial_strategies.npy')
    sys.exit(1)


signal.signal(signal.SIGTERM, sigterm_handler)


@ray.remote
class NodesActor:
    def __init__(self):
        self.nodes = defaultdict(lambda: {
            'regret_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu'),
            'strategy_sum': torch.zeros(Config.NUM_ACTIONS, dtype=Config.DTYPE, device='cpu')
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
        # Blend average strategies toward current regret matching as a lightweight hybrid update.
        for node in self.nodes.values():
            node['strategy_sum'] = apply_regret_matching_boost(node['regret_sum'], node['strategy_sum'], weight).cpu()

    def get_all_keys(self):
        return list(self.nodes.keys())


@ray.remote(num_cpus=1)
def run_mccfr(infoset, iteration, actor_handle, max_depth, num_opponents):
    # Push the curriculum opponent count into each worker before traversal.
    Config.NUM_OPPONENTS = max(1, num_opponents)
    return mccfr(infoset, iteration, actor=actor_handle, max_depth=max_depth)


def train():
    global infosets, actor

    init_ray()
    maybe_compile_models()

    features = simulate_features()
    buckets, _ = create_buckets(features)
    unique_buckets = np.unique(buckets)
    infosets = [Infoset(bid) for bid in unique_buckets]
    actor = NodesActor.remote()

    def local_sigterm_handler(signum, frame):
        logger.error('SIGTERM received; saving partial strategies and exiting.')
        save_strategies('partial_strategies.npy')
        sys.exit(1)

    signal.signal(signal.SIGTERM, local_sigterm_handler)

    try:
        ray.get(actor.get_all_keys.remote())
        logger.info('NodesActor initialized successfully.')
    except Exception as error:
        logger.error(f'Actor initialization failed: {error}')
        raise

    current_task_batch = Config.RAY_TASK_BATCH
    current_batch_size = Config.BATCH_SIZE

    try:
        for it in range(Config.ITERATIONS):
            # Ramp traversal depth and opponent count gradually to stabilize early learning.
            current_max_depth = min(Config.MAX_CFR_DEPTH, Config.START_MAX_DEPTH + (it // Config.CURRICULUM_INTERVAL))
            current_num_opponents = min(Config.MAX_CURRICULUM_OPPONENTS, 1 + (it // Config.CURRICULUM_INTERVAL))

            all_utils = []
            for start in range(0, len(infosets), current_task_batch):
                infoset_chunk = infosets[start:start + current_task_batch]
                futures = [
                    run_mccfr.remote(infoset, it, actor, current_max_depth, current_num_opponents)
                    for infoset in infoset_chunk
                ]
                all_utils.extend(ray.get(futures))

            total_util = float(np.mean(all_utils)) if all_utils else 0.0

            if it % Config.LOG_INTERVAL == 0:
                regret_sums = ray.get([actor.get_regret_sum.remote(infoset.key) for infoset in infosets])
                regrets = [rs.mean().item() for rs in regret_sums]
                avg_regret = float(np.mean(regrets)) if regrets else 0.0
                vram_gb = get_vram_usage_gb()
                ram_pct = get_ram_usage_pct()

                # Automatically back off batch/task pressure before memory becomes unstable.
                if vram_gb > Config.VRAM_SOFT_LIMIT_GB or ram_pct > Config.RAM_SOFT_LIMIT_PCT:
                    current_task_batch = max(Config.MIN_RAY_TASK_BATCH, current_task_batch // 2)
                    current_batch_size = max(Config.MIN_BATCH_SIZE, current_batch_size // 2)
                    Config.BATCH_SIZE = current_batch_size
                    logger.warning(
                        f'Memory pressure detected; reducing task batch to {current_task_batch} and BATCH_SIZE to {current_batch_size}.'
                    )
                elif current_task_batch < Config.RAY_TASK_BATCH and vram_gb < (Config.VRAM_SOFT_LIMIT_GB * 0.75) and ram_pct < (Config.RAM_SOFT_LIMIT_PCT - 10.0):
                    current_task_batch = min(Config.RAY_TASK_BATCH, current_task_batch * 2)
                    current_batch_size = min(Config.MAX_BATCH_SIZE, current_batch_size * 2)
                    Config.BATCH_SIZE = current_batch_size

                writer.add_scalar('Util/Avg', total_util, it)
                writer.add_scalar('Regret/Avg', avg_regret, it)
                writer.add_scalar('System/VRAM_GB', vram_gb, it)
                writer.add_scalar('System/RAM_Pct', ram_pct, it)
                writer.add_scalar('Curriculum/MaxDepth', current_max_depth, it)
                writer.add_scalar('Curriculum/NumOpponents', current_num_opponents, it)
                log_msg = (
                    f'Iter {it}: Util {total_util:.4f}, Regret {avg_regret:.6f} | '
                    f'VRAM {vram_gb:.2f} GB | RAM {ram_pct:.1f}% | '
                    f'Depth {current_max_depth} | Opponents {current_num_opponents} | Chunk {current_task_batch}'
                )
                logger.info(log_msg)
                print(log_msg)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if it > 0 and it % Config.HYBRID_UPDATE_INTERVAL == 0:
                # Periodic regret-matching boost acts like a lightweight hybrid refinement step.
                logger.info('Applying lightweight regret-matching boost across nodes...')
                ray.get(actor.apply_regret_matching_boost.remote(Config.HYBRID_BOOST_WEIGHT))

    except KeyboardInterrupt:
        local_sigterm_handler(None, None)
    finally:
        save_strategies('strategies.npy')
        logger.info('Training complete; strategies saved.')
        print('Training complete; strategies saved.')


if __name__ == '__main__':
    torch.set_default_dtype(Config.DTYPE)
    train()
    writer.close()
    if ray.is_initialized():
        ray.shutdown()