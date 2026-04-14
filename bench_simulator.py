import argparse
import statistics
import time

import eval7
import numpy as np
import torch

from environment import initialize_rocm_runtime

initialize_rocm_runtime()

from config import Config
from game import simulate_action_batch, simulate_equity_batch
from datatypes import Action, Infoset


def _build_hands(batch_size, board_len):
    holes = []
    boards = []
    for _ in range(batch_size):
        deck = eval7.Deck()
        deck.shuffle()
        holes.append(deck.deal(2))
        boards.append(deck.deal(board_len))
    return holes, boards


def _build_infosets(batch_size):
    infosets = []
    actions = []
    action_values = [Action.FOLD, Action.CALL, Action.RAISE]
    for index in range(batch_size):
        history_len = index % 4
        history = tuple((index + offset) % Config.NUM_ACTIONS for offset in range(history_len))
        infosets.append(Infoset(bucket_id=index % 4096, history=history))
        actions.append(action_values[index % len(action_values)])
    return infosets, actions


def _measure(fn, iterations):
    samples = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if Config.DEVICE == 'cuda':
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return samples, result


def main():
    parser = argparse.ArgumentParser(description="Benchmark simulator hotspots for Poker2.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--board-len", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-opponents", type=int, default=None)
    args = parser.parse_args()

    num_opponents = max(1, args.num_opponents or Config.NUM_OPPONENTS)
    holes, boards = _build_hands(args.batch_size, args.board_len)
    infosets, actions = _build_infosets(args.batch_size)

    equity_samples, equities = _measure(
        lambda: simulate_equity_batch(holes, boards, num_opponents=num_opponents),
        args.iterations,
    )
    action_samples, utilities = _measure(
        lambda: simulate_action_batch(infosets, actions),
        args.iterations,
    )

    print({
        "device": Config.DEVICE,
        "batch_size": args.batch_size,
        "board_len": args.board_len,
        "num_opponents": num_opponents,
        "equity_avg_s": round(statistics.mean(equity_samples), 4),
        "equity_min_s": round(min(equity_samples), 4),
        "action_avg_s": round(statistics.mean(action_samples), 4),
        "action_min_s": round(min(action_samples), 4),
        "equity_preview": [round(float(value), 4) for value in np.asarray(equities)[:5]],
        "action_preview": [round(float(value), 4) for value in np.asarray(utilities)[:5]],
    })


if __name__ == "__main__":
    main()