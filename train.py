infosets = None
actor = None

import os
import signal
import sys
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import ray
import psutil
from torch.utils.tensorboard import SummaryWriter

from cfr import mccfr, average_strategy
from abstractions import simulate_features, create_buckets
from config import Config
from datatypes import Infoset

os.environ["PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING"] = "1"
ray.init(num_cpus=24, num_gpus=1 if torch.cuda.is_available() else 0)

writer = SummaryWriter()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])


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
def run_mccfr(infoset, iteration, actor):
    return mccfr(infoset, iteration, actor=actor)


def train():
    global infosets, actor

    features = simulate_features()
    buckets, _ = create_buckets(features)
    unique_buckets = np.unique(buckets)
    infosets = [Infoset(int(bid)) for bid in unique_buckets]
    actor = NodesActor.remote()

    def local_sigterm_handler(signum, frame):
        logging.error("SIGTERM received; saving partial strategies and exiting.")
        strategies = {}
        for inf in infosets:
            strategy_sum = ray.get(actor.get_strategy_sum.remote(inf.key))
            if strategy_sum.sum() > 0:
                strategies[inf.key] = average_strategy(inf, actor=actor).cpu().tolist()
        np.save('partial_strategies.npy', strategies)
        with open('partial_strategies.json', 'w') as f:
            json.dump(strategies, f)
        sys.exit(1)

    signal.signal(signal.SIGTERM, local_sigterm_handler)

    try:
        ray.get(actor.get_all_keys.remote())
        logging.info("NodesActor initialized successfully.")
    except Exception as e:
        logging.error(f"Actor initialization failed: {e}")
        raise

    try:
        for it in range(Config.ITERATIONS):
            futures = [run_mccfr.remote(inf, it, actor) for inf in infosets]
            utils = ray.get(futures)
            total_util = float(np.mean(utils))
            if it % 1000 == 0:
                regret_sums = ray.get([actor.get_regret_sum.remote(inf.key) for inf in infosets])
                regrets = [rs.mean().item() for rs in regret_sums]
                avg_regret = float(np.mean(regrets)) if regrets else 0.0
                vram_gb = torch.cuda.memory_allocated() / 1e9
                cpu_percent = psutil.cpu_percent() if 'psutil' in globals() else 'N/A'
                writer.add_scalar('Util/Avg', total_util, it)
                writer.add_scalar('Regret/Avg', avg_regret, it)
                logging.info(
                    f"Iter {it}: Util {total_util:.2f}, Regret {avg_regret:.4f} | "
                    f"VRAM: {vram_gb:.2f} GB | CPU: {cpu_percent}% | Ray Dashboard: http://127.0.0.1:8265"
                )
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        local_sigterm_handler(None, None)
    finally:
        strategies = {}
        for inf in infosets:
            strategy_sum = ray.get(actor.get_strategy_sum.remote(inf.key))
            if strategy_sum.sum() > 0:
                strategies[inf.key] = average_strategy(inf, actor=actor).cpu().tolist()
        np.save('strategies.npy', strategies)
        with open('strategies.json', 'w') as f:
            json.dump(strategies, f)
        logging.info("Training complete; strategies saved.")
        writer.close()
        ray.shutdown()


if __name__ == "__main__":
    torch.set_default_dtype(Config.DTYPE)
    train()
