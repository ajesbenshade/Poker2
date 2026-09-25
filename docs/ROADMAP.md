# Roadmap

Target: the strongest heads-up no-limit hold'em agent this machine can train in about
a month. Ryzen 9 7900X (24 threads), 64 GB RAM, RTX 3080 10 GB, training in WSL2
(Ubuntu 24.04).

## Game and abstraction

- **Rules:** Slumbot-compatible HUNL, 200 bb stacks (50/100 blinds, 20,000 chips), so
  results can be measured head-to-head against Slumbot's public API.
- **Actions:** `ActionAbstraction::blueprint()` in `engine/src/holdem/action_abstraction.h`.
  Preflop opens of 0.5/0.75/1/1.5/2.5x pot, postflop 1/3 to 1.5x pot, fewer sizes for re-raises,
  all-in always available, at most 3 raises per street.
- **Cards:** preflop uses the 169 lossless classes. Flop and turn use about 2,000 buckets each
  (k-means on equity histograms). River uses about 1,500 buckets (opponent-cluster hand
  strength). Isomorphic hands are merged first (`engine/src/holdem/isomorphism.h`: 169 /
  1,286,792 / 13,960,050 / 123,156,254 classes). The CPU is fast enough for this (see below).
- **Blueprint:** external-sampling MCCFR (`engine/src/cfr/mccfr.h`) on dense `int32` regret
  arrays, lock-free multithreaded, with negative-regret pruning after warm-up. Average
  strategy stored for preflop and flop; later streets use real-time search.
- **Play time:** depth-limited subgame solving, with a GPU-trained value network at the leaves.

## Role of the RTX 3080

1. Leaf value network for real-time search (main job).
2. Training and evaluation on older checkpoints while the CPU trains the blueprint.
3. Optional: clustering for the card abstraction, if CPU k-means turns out slow.

The blueprint and the equity computations are CPU work. The original plan put the card
abstraction on the GPU, but the evaluator runs at 59M hands/s per thread: river equity vs a
random hand for all 123M river classes is about 1.2e11 evaluations, roughly 1.5 minutes on
24 threads.

## Measured sizes and speeds

Blueprint abstraction with 169 / 2,000 / 2,000 / 1,500 buckets (`poker2_tree`):

| Street | Betting nodes | Infosets | Table GB |
|---|---|---|---|
| Preflop | 144 | 24K | 0.00 |
| Flop | 6,428 | 12.9M | 0.30 |
| Turn | 72,904 | 146M | 1.64 |
| River | 578,652 | 868M | 9.59 |
| **Total** | | **1.03B** | **11.5** |

Memory is not the constraint; iterations per infoset are. The pilot run decides whether to
trim sizes or buckets.

Single-thread throughput (`poker2_bench`): 7-card evaluation 59M/s, river isomorphism index
13.5M/s, flop index 15.8M/s, random deals 30M/s, random abstract hand playouts 4.3M/s.

## Card abstraction results

`poker2_abstraction` builds everything in about 8.5 minutes on 24 threads, writing 2.4 GB to
`~/poker2-data/abstraction`:

| Stage | Time | Result |
|---|---|---|
| Preflop | 2 s | 169 lossless buckets; 8 opponent clusters by preflop equity (AA 85.2%, 32o 32.3%) |
| River equity + OCHS | 113 s | Exact equity vs all 990 opponent hands for 123,156,254 classes |
| River | 52 s | 1,500 buckets on OCHS vectors; k-means explains 98.5% (hit the 50-iteration cap) |
| Turn | 154 s | 2,000 buckets on 46-value equity distributions; explains 99.6% |
| Flop | 164 s | 2,000 buckets on 50-bin runout distributions; explains 99.5% |

Quality report on 500K random deals per street: every bucket occupied; largest bucket 0.20% /
0.51% / 1.21% of hands (flop / turn / river); buckets explain 99.90% / 99.90% / 99.66% of
equity variance. Bucket ids are sorted by strength, and spot checks land where expected:
royal flush 1499/1500, top set 1999/2000, a straight-flush draw mid-range (1010/2000),
air near the bottom.

The equity R^2 is a sanity check rather than proof of quality, since the features are built
from the same equities. The real test is the pilot run's exploitability and head-to-head results.

## Blueprint trainer

`engine/src/blueprint/`: the betting tree is flattened into arrays (658,128 decision nodes);
regrets and averages live in per-street float tables indexed by (node, bucket, action), updated
by all threads without locks through relaxed `std::atomic_ref`. External-sampling MCCFR as
validated on Leduc, plus a regret floor, Pluribus-style pruning (skip actions with regret below
-3e8 with probability 0.95, never on the river or for hand-ending actions), and Linear CFR as
k/(k+1) discounting between epochs.

Validation:
- **Push/fold at 10bb** (exact exploitability over all 1326 x 1225 hand pairs): 141 -> ~3 mbb/hand
  in 4M iterations; AA pushes and calls, 72o folds to a shove.
- **Small postflop game:** LBR falls from ~1,036 to ~590 mbb/hand with 1M iterations; the trained
  strategy beats random and always-call bots.
- Terminal payoffs match the rules engine; tree statistics match `count_tree`; checkpoints
  round-trip exactly and reject a mismatched layout; single-thread runs are reproducible.

Throughput with the full blueprint bet sizes and 200-bucket abstraction: ~310,000 iterations/s
on 24 threads (each iteration traverses once per player), ~27 billion iterations per day.

## Schedule

| Days | Phase | Done when | Status |
|---|---|---|---|
| 1-2 | CFR core validated on Kuhn and Leduc | Exact exploitability goes to ~0; published game values match | **Done** |
| 3-6 | HUNL engine: rules, hand evaluator, isomorphism, action abstraction, tests | Evaluator and isomorphism match published counts exactly; random games replay correctly | **Done** |
| 5-8 | Card abstraction (CPU equity, k-means clustering) | Bucket files written and quality-checked | **Done** |
| 8-9 | 24-hour pilot with a small abstraction | Local-best-response exploitability falling | Trainer done; pilot running |
| 9-27 | Full blueprint run (~18 days x 24 threads) | Checkpoint every 12 h, each evaluated automatically | |
| 12-25 | Real-time search and value network | Search beats the blueprint alone head-to-head | |
| 27-30 | Final evaluation vs Slumbot, 20k+ hands with AIVAT | Checkpoint chosen by exploitability and head-to-head, never by average utility | |

## Findings so far

- **CFR+ needs deferred regret updates.** An infoset is reached through several histories
  per traversal, so its strategy must stay fixed for the whole traversal, and the CFR+ floor
  applies to the summed regret. Applying updates per history left Leduc exploitability at 0.05
  after 1,000 iterations; deferring them gives 0.00026.
- **Linear CFR weighting made no measurable difference** for ES-MCCFR on Leduc (3 seeds, 3M
  iterations; both end at ~0.006-0.008). It stays a setting (`--linear-cap`) and gets
  decided by the HUNL pilot run.
- **Isomorphism counts depend on grouping.** Treating the board as one group gives 13,960,050
  turn and 123,156,254 river classes. Dealing it street by street gives 55,190,538 and
  2,428,287,420. The abstraction uses one board group, since hand strength doesn't depend
  on which street a card arrived on. The tests check all six published counts.

## Machine setup checklist

- [ ] Install the NVIDIA driver on Windows (the RTX 3080 currently has none). Don't install a
      Linux driver inside WSL; check with `nvidia-smi` from Ubuntu.
- [ ] `C:\Users\Aaron\.wslconfig`: `memory=54GB`, `processors=24`, `swap=16GB`, then `wsl --shutdown`.
- [ ] Turn off sleep (`powercfg /change standby-timeout-ac 0`) and pause Windows Update for the run.
- [ ] Optional: Eco mode (105 W) for the 7900X in the BIOS for a cooler month-long run.
- [ ] Keep checkpoints on the WSL ext4 disk during training and copy them to E: or A: periodically.
      Writing large files straight to `/mnt/*` is slow.
