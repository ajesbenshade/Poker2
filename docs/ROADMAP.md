# Roadmap

Target: the strongest heads-up no-limit hold'em agent this machine can train in about
a month. Ryzen 9 7900X (24 threads), 64 GB RAM, RTX 3080 10 GB, training in WSL2
(Ubuntu 24.04).

## Game and abstraction

- **Rules:** Slumbot-compatible HUNL, 200 bb stacks (50/100 blinds, 20,000 chips), so
  results can be measured head-to-head against Slumbot's public API.
- **Actions:** about 6 preflop raise sizes; postflop 1/3, 1/2, 3/4, 1x, 1.5x pot and all-in;
  at most 3 raises per street.
- **Cards:** preflop uses the 169 lossless classes. Flop and turn use about 2,000 buckets each
  (k-means on equity histograms). River uses about 1,500 buckets (opponent-cluster hand
  strength). Computed on the GPU.
- **Blueprint:** external-sampling MCCFR (`engine/src/cfr/mccfr.h`) on dense `int32` regret
  arrays, lock-free multithreaded, with negative-regret pruning after warm-up. Average
  strategy stored for preflop and flop; later streets use real-time search.
- **Play time:** depth-limited subgame solving, with a GPU-trained value network at the leaves.

## Role of the RTX 3080

1. Card abstraction: equity for about 1.3M flops, 14M turns and 123M rivers (after
   isomorphism), plus clustering.
2. Leaf value network for real-time search.
3. Training and evaluation on older checkpoints while the CPU trains the blueprint.

The blueprint itself is CPU work.

## Schedule

| Days | Phase | Done when | Status |
|---|---|---|---|
| 1-2 | CFR core validated on Kuhn and Leduc | Exact exploitability goes to ~0; published game values match | **Done** |
| 3-6 | HUNL engine: rules, hand evaluator, isomorphism, tests | Random games replay correctly; evaluator matches a reference | Next |
| 5-8 | Card abstraction on the GPU | Bucket files written and quality-checked | |
| 8-9 | 24-hour pilot with a small abstraction | Local-best-response exploitability falling | |
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

## Machine setup checklist

- [ ] Install the NVIDIA driver on Windows (the RTX 3080 currently has none). Don't install a
      Linux driver inside WSL; check with `nvidia-smi` from Ubuntu.
- [ ] `C:\Users\Aaron\.wslconfig`: `memory=54GB`, `processors=24`, `swap=16GB`, then `wsl --shutdown`.
- [ ] Turn off sleep (`powercfg /change standby-timeout-ac 0`) and pause Windows Update for the run.
- [ ] Optional: Eco mode (105 W) for the 7900X in the BIOS for a cooler month-long run.
- [ ] Keep checkpoints on the WSL ext4 disk during training and copy them to E: or A: periodically.
      Writing large files straight to `/mnt/*` is slow.
