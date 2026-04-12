infosets = None
actor = None

import argparse
import os
import signal
import sys
import json
import logging
import random
from collections import defaultdict

import numpy as np
import torch
import psutil
from torch.utils.tensorboard import SummaryWriter

from cfr import average_strategy, deep_cfr_traverse, mccfr
from abstractions import simulate_features, create_buckets
from config import Config
from datatypes import Infoset
from deep_cfr import DeepCFRAgent
from storage import LocalNodeStore

try:
    import ray
except ImportError:
    ray = None

os.environ["PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])


if ray is not None:
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

        def get_all_keys(self):
            return list(self.nodes.keys())


    @ray.remote
    def run_mccfr(infoset, iteration, actor, max_depth):
        return mccfr(infoset, iteration, actor=actor, max_depth=max_depth)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the poker agent in tabular or deep mode.")
    parser.add_argument('--mode', choices=['tabular', 'deep'], default=None)
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
    return parser.parse_args()


def apply_runtime_overrides(args):
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

    Infoset.KEY_MODE = Config.INFOSET_KEY_MODE


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_runtime():
    if Config.ALGORITHM_MODE != 'tabular':
        return
    if ray is not None and not ray.is_initialized():
        ray.init(num_cpus=Config.RAY_NUM_CPUS, num_gpus=Config.RAY_NUM_GPUS, ignore_reinit_error=True)


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
    raw_strategies, json_strategies = collect_strategies(
        infosets,
        actor=actor,
        node_store=node_store,
        deep_agent=deep_agent,
    )
    np.save(f'{prefix}.npy', raw_strategies)
    with open(f'{prefix}.json', 'w') as handle:
        json.dump(json_strategies, handle)


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


def train():
    global infosets, actor

    initialize_runtime()
    set_global_seeds(Config.SEED)
    writer = SummaryWriter()
    infosets = build_root_infosets()
    actor = None
    node_store = None
    deep_agent = None

    logging.info(
        "Starting training | mode=%s env=%s key_mode=%s backend=%s buckets=%s iterations=%s depth=%s seed=%s",
        Config.ALGORITHM_MODE,
        Config.ENVIRONMENT_MODE,
        Config.INFOSET_KEY_MODE,
        'ray' if Config.ALGORITHM_MODE == 'tabular' and ray is not None else 'local',
        Config.NUM_BUCKETS,
        Config.ITERATIONS,
        Config.MAX_DEPTH,
        Config.SEED,
    )

    def local_sigterm_handler(signum, frame):
        logging.error("SIGTERM received; saving partial strategies and exiting.")
        save_strategies(infosets, 'partial_strategies', actor=actor, node_store=node_store, deep_agent=deep_agent)
        if deep_agent is not None:
            deep_agent.save_checkpoint('partial_strategies.pt', iteration=-1)
        sys.exit(1)

    signal.signal(signal.SIGTERM, local_sigterm_handler)

    try:
        if Config.ALGORITHM_MODE == 'deep':
            deep_agent = DeepCFRAgent()
            logging.info("Deep mode uses local replay buffers and state-key infosets.")
        else:
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
        for it in range(Config.ITERATIONS):
            if deep_agent is not None:
                utils = []
                for _ in range(Config.DEEP_CFR_TRAVERSALS_PER_ITER):
                    for inf in infosets:
                        utils.append(deep_cfr_traverse(inf, it, deep_agent=deep_agent, max_depth=Config.MAX_DEPTH))
                advantage_losses = [
                    deep_agent.train_advantage_network(player)
                    for player in range(len(Config.DEFAULT_STACK_SIZES))
                ]
                average_loss = deep_agent.train_average_network()
            else:
                if actor is not None:
                    futures = [run_mccfr.remote(inf, it, actor, Config.MAX_DEPTH) for inf in infosets]
                    utils = ray.get(futures)
                else:
                    utils = [mccfr(inf, it, node_store=node_store, max_depth=Config.MAX_DEPTH) for inf in infosets]

            total_util = float(np.mean(utils))
            if it % Config.LOG_INTERVAL == 0:
                if deep_agent is not None:
                    buffer_sizes = deep_agent.buffer_sizes()
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    cpu_percent = psutil.cpu_percent() if 'psutil' in globals() else 'N/A'
                    writer.add_scalar('Util/Avg', total_util, it)
                    writer.add_scalar('Loss/AdvantagePlayer0', advantage_losses[0], it)
                    writer.add_scalar('Loss/AdvantagePlayer1', advantage_losses[1], it)
                    writer.add_scalar('Loss/AverageStrategy', average_loss, it)
                    writer.add_scalar('Replay/AdvantagePlayer0', buffer_sizes['advantage_0'], it)
                    writer.add_scalar('Replay/AdvantagePlayer1', buffer_sizes['advantage_1'], it)
                    writer.add_scalar('Replay/AverageStrategy', buffer_sizes['strategy'], it)
                    logging.info(
                        "Iter %s: Util %.2f, AdvLoss [%.4f, %.4f], AvgLoss %.4f | Replay %s | VRAM: %.2f GB | CPU: %s%%",
                        it,
                        total_util,
                        advantage_losses[0],
                        advantage_losses[1],
                        average_loss,
                        buffer_sizes,
                        vram_gb,
                        cpu_percent,
                    )
                else:
                    if actor is not None:
                        regret_sums = ray.get([actor.get_regret_sum.remote(inf.key) for inf in infosets])
                    else:
                        regret_sums = [node_store.get_regret_sum(inf.key).cpu() for inf in infosets]
                    regrets = [rs.mean().item() for rs in regret_sums]
                    avg_regret = float(np.mean(regrets)) if regrets else 0.0
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    cpu_percent = psutil.cpu_percent() if 'psutil' in globals() else 'N/A'
                    writer.add_scalar('Util/Avg', total_util, it)
                    writer.add_scalar('Regret/Avg', avg_regret, it)
                    logging.info(
                        f"Iter {it}: Util {total_util:.2f}, Regret {avg_regret:.4f} | "
                        f"VRAM: {vram_gb:.2f} GB | CPU: {cpu_percent}% | Ray Dashboard: http://127.0.0.1:8265"
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if it > 0 and it % Config.CHECKPOINT_INTERVAL == 0:
                save_strategies(infosets, f'checkpoint_{it}', actor=actor, node_store=node_store, deep_agent=deep_agent)
                if deep_agent is not None:
                    deep_agent.save_checkpoint(f'checkpoint_{it}.pt', iteration=it)
    except KeyboardInterrupt:
        local_sigterm_handler(None, None)
    finally:
        save_strategies(infosets, 'strategies', actor=actor, node_store=node_store, deep_agent=deep_agent)
        if deep_agent is not None:
            deep_agent.save_checkpoint('strategies.pt', iteration=Config.ITERATIONS)
        logging.info("Training complete; strategies saved.")
        writer.close()
        if ray is not None and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_overrides(args)
    torch.set_default_dtype(Config.DTYPE if Config.ALGORITHM_MODE == 'tabular' else Config.NN_DTYPE)
    train()
