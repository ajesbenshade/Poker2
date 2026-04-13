# Poker2
Aaron's latest attempt at a GTO poker player

## Abstractions

The project groups similar game states into buckets to make solving manageable.
`abstractions.py` currently builds those buckets with ``k``-means clustering on
simulated equity feature vectors for each board class.  Other schemes such as
effective hand strength (EHS/EHS2) or more elaborate potential-aware methods can
be substituted, but the simple equity-vector approach keeps the example light.

The default bucket counts are:

| Street   | Buckets |
|----------|---------|
| Preflop  | 169     |
| Flop     | 1000    |
| Turn     | 500     |
| River    | 200     |

Hands and boards are canonicalised prior to feature extraction so that suit or
order permutations map to the same bucket, preventing "bucket bleed" where near
equivalent states fall into different clusters.

## Current Training Baseline

The repo still trains a tabular MCCFR baseline, but the foundation has been refactored so future Deep CFR work can plug into explicit seams instead of rewriting the whole trainer.

- Richer infosets now carry acting player, street, pot, stack sizes, board cards, private cards, and current bet while preserving a legacy key mode for the existing tabular path.
- Environment transitions and payoff sampling are separated behind a simplified poker environment adapter.
- Node storage is pluggable, with local execution as a fallback when Ray is unavailable.
- Neural-ready observation encoding lives in [abstractions.py](./abstractions.py) via `encode_infoset` and `encode_infosets`.
- A first local-only Deep CFR path now exists with per-player advantage replay, an average-strategy replay buffer, two advantage networks, and one average-strategy network.

This workspace does not contain the larger Actor-Critic, MCTS, or EquityNet stack mentioned in earlier design notes. The live code is a heads-up MCCFR plus Deep CFR trainer, and the current runtime hardening is scoped to that code path.

## Smoke Test

Use a tiny local run to validate the trainer end to end before launching a large job:

```bash
.venv/bin/python train.py --smoke-test
```

This produces `strategies.npy` and `strategies.json` using the local fallback path if Ray is not installed.

## Deep CFR Preview

Use deep mode to exercise the first neural slice. This path forces state-key infosets automatically and writes both strategy exports and a model checkpoint.

```bash
.venv/bin/python train.py --mode deep --smoke-test --max-depth 2
```

This produces `strategies.npy`, `strategies.json`, and `strategies.pt` from the local Deep CFR trainer.

## ROCm Safe Mode

The trainer now defaults to a hardware-safe profile aimed at long, stable ROCm runs on a 20 GB GPU with large system RAM. Key changes:

- ROCm-friendly allocator environment variables are set at startup in both [train.py](./train.py) and [deep_cfr.py](./deep_cfr.py).
- Monte Carlo equity simulation stays on CPU via `Config.SIMULATION_DEVICE`, which keeps rollout work off the GPU.
- Deep mode uses array-backed replay buffers, optional gradient checkpointing, scaler-aware AMP, and aggressive cache clearing after large steps.
- Runtime backoff automatically halves simulation batch size, equity rollouts, neural batch size, train steps, and Deep CFR traversals when VRAM exceeds 15.5 GB or RAM exceeds 78 percent.
- Checkpoints now include model, optimizer, scaler, and replay-buffer state. The trainer also writes `best_strategies.*` and `best_model.pt` when it reaches a new best average utility.

Resume a Deep CFR run from a saved checkpoint with:

```bash
.venv/bin/python train.py --mode deep --resume-checkpoint checkpoint_10000.pt
```

Run indefinitely until interrupted with:

```bash
.venv/bin/python train.py --mode deep --long-run
```

Recommended environment variables before launching a long run:

```bash
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11_0_0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF"
```
