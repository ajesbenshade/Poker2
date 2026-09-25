# Poker2
Aaron's latest attempt at a GTO poker player.

The goal is a heads-up no-limit hold'em agent built the way Libratus and
Pluribus were: a real game engine, card and action abstraction, an MCCFR
blueprint trained on the CPU, and real-time subgame solving at play time.
It targets one machine (Ryzen 9 7900X, 64 GB RAM, RTX 3080) over about a month.
See [docs/ROADMAP.md](docs/ROADMAP.md) for the full plan and current status.

## Engine (C++)

`engine/` is a dependency-free C++17 library. It currently contains:

- **Hold'em engine** (`engine/src/holdem/`): heads-up no-limit rules with Slumbot's
  200bb setup and action strings, a 7-card hand evaluator, a suit-isomorphism hand indexer,
  and the blueprint's action abstraction.
- **Card abstraction** (`engine/src/abstraction/`): exact river equity and opponent-cluster
  hand strength, distribution-aware turn and flop features, multithreaded k-means, and the
  runtime bucket lookup.
- **Toy games:** Kuhn poker and Leduc hold'em (`engine/src/games/`), used to validate solvers exactly.
- **Solvers:** CFR+ as the exact reference solver, and external-sampling MCCFR with optional
  Linear CFR weighting, which is the algorithm the HUNL blueprint will use (`engine/src/cfr/`).
- **Evaluation:** exact best response, exploitability and profile value (`engine/src/cfr/evaluation.h`).

Build and test inside WSL (Ubuntu 24.04, g++ 13):

```bash
make -C engine test
```

`make -C engine test-slow` adds the exhaustive checks: all 133,784,560 seven-card hands, every
flop deal, and full turn and river isomorphism round trips (about 35 seconds).

Size the blueprint's betting tree and memory, and benchmark the building blocks:

```bash
engine/build/poker2_tree
```

```bash
engine/build/poker2_bench
```

Build the card abstraction (writes about 2.5 GB to `~/poker2-data/abstraction` on the WSL disk):

```bash
engine/build/poker2_abstraction
```

It runs the stages `preflop,equity,river,turn,flop,report`; `--stages` reruns a subset, and
`--buckets FLOP,TURN,RIVER` changes bucket counts.

Plot a convergence curve:

```bash
engine/build/poker2_solve --game leduc --algo mccfr --iters 3000000 --eval-every 300000 --csv leduc.csv
```

Reference results (all checked by the test suite):

| Check | Result |
|---|---|
| Leduc infosets | 288 (textbook count) |
| Kuhn best response vs uniform | 1/2 and 5/12, exact |
| Kuhn CFR+ game value | -1/18, P1 strategy matches the unique equilibrium |
| Leduc CFR+ game value, 2k iterations | -0.08560 (published: -0.0856), exploitability 8e-5 |
| Leduc ES-MCCFR, 3M iterations (about 8 s) | exploitability ~0.006-0.008 chips/hand |
| 5-card hands | Published category counts; exactly 7,462 distinct values |
| 7-card hands (all 133.8M) | Published category counts; exactly 4,824 distinct values |
| Isomorphism classes | 169 / 1,286,792 / 13,960,050 / 123,156,254, plus the perfect-recall 55,190,538 and 2,428,287,420 |
| Blueprint tree | 1.03B infosets, 11.5 GB of tables |

## Legacy Python trainer

The Python files at the repo root (`train.py`, `cfr.py`, `game.py`, ...) are the previous
trainer. They are kept for reference but should not be used for training: payoffs come from a
heuristic formula rather than poker outcomes, infosets never see their own cards, and the tree
stops after two actions. See the roadmap for details. They will move to `legacy/` once the
C++ engine can play full hands.
