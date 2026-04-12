# Poker2
Aaron's latest attempt at a GTO poker player

## Current Training Baseline

The repo still trains a tabular MCCFR baseline, but the foundation has been refactored so future Deep CFR work can plug into explicit seams instead of rewriting the whole trainer.

- Richer infosets now carry acting player, street, pot, stack sizes, board cards, private cards, and current bet while preserving a legacy key mode for the existing tabular path.
- Environment transitions and payoff sampling are separated behind a simplified poker environment adapter.
- Node storage is pluggable, with local execution as a fallback when Ray is unavailable.
- Neural-ready observation encoding lives in [abstractions.py](./abstractions.py) via `encode_infoset` and `encode_infosets`.
- A first local-only Deep CFR path now exists with per-player advantage replay, an average-strategy replay buffer, two advantage networks, and one average-strategy network.

## Smoke Test

Use a tiny local run to validate the trainer end to end before launching a large job:

```bash
.venv/bin/python train.py \
	--iterations 1 \
	--num-sims 8 \
	--num-buckets 2 \
	--batch-size 2 \
	--equity-rollouts 1 \
	--hand-eval-processes 1 \
	--log-interval 1 \
	--max-depth 1
```

This produces `strategies.npy` and `strategies.json` using the local fallback path if Ray is not installed.

## Deep CFR Preview

Use deep mode to exercise the first neural slice. This path forces state-key infosets automatically and writes both strategy exports and a model checkpoint.

```bash
.venv/bin/python train.py \
	--mode deep \
	--iterations 1 \
	--num-sims 8 \
	--num-buckets 2 \
	--batch-size 2 \
	--equity-rollouts 1 \
	--hand-eval-processes 1 \
	--log-interval 1 \
	--max-depth 2 \
	--deep-traversals-per-iter 1 \
	--nn-train-steps 1 \
	--nn-batch-size 2
```

This produces `strategies.npy`, `strategies.json`, and `strategies.pt` from the local Deep CFR trainer.
