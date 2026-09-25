"""Benchmarks the river solver on the GPU and the CPU.

    python -m search.bench_river [--iterations N] [--cpu-threads N]

Spot: pot 1000, 20,000 stacks, the blueprint's river sizes (1/3 to 1.5 pot plus
all-in, up to 3 raises), uniform ranges on a coordinated board.
"""

import argparse
import time

import numpy as np
import torch

from .cards import NUM_COMBOS, parse_cards
from .river_solver import RiverSolver
from .tree import RiverSpot


def run(device, dtype, iterations):
    spot = RiverSpot(contrib=(500, 500), stack=20000, first_to_act=1)
    solver = RiverSolver(spot, parse_cards("Ts9s8d4h2c"), [np.ones(NUM_COMBOS)] * 2, device=device, dtype=dtype)
    solver.iterate(2)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    solver.iterate(iterations)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    decisions = sum(1 for n in solver.nodes if n.kind == "decision")
    return seconds, decisions, len(solver.nodes), solver.exploitability()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    configs = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        configs.insert(0, ("cuda", torch.float32))
    for device, dtype in configs:
        seconds, decisions, nodes, expl = run(device, dtype, args.iterations)
        label = f"{device} {str(dtype).replace('torch.', '')}"
        if device == "cpu":
            label += f" ({args.cpu_threads} threads)"
        print(f"{label:24s} {args.iterations} iterations in {seconds:6.2f}s = {args.iterations / seconds:7.1f} it/s"
              f"  | tree {decisions} decisions / {nodes} nodes | exploitability {expl:.3%} of pot")


if __name__ == "__main__":
    main()
