# Poker2 ROCm Optimization Guide
## For Ryzen 9 7900X + RX 7900 XT (20GB VRAM) + 64GB RAM

The primary bring-up path is now Deep CFR via `train.py --algo deep-cfr`. PPO remains available as a baseline in `train.py` and `rl.py`.

## Critical Startup Order

Call `initialize_rocm_runtime()` before any torch import. The shared runtime in `environment.py` normalizes allocator settings, applies ROCm defaults, probes the GPU, and logs fallback status before model code loads.

## Current ROCm Environment

Use `PYTORCH_ALLOC_CONF`, not `PYTORCH_HIP_ALLOC_CONF`. The current default configuration is:

- `HIP_VISIBLE_DEVICES=0`
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`
- `HSA_ENABLE_SDMA=0`
- `TORCH_CUDNN_ENABLE=0`
- `OMP_NUM_THREADS=1`

## Precision and Safety

`torch.set_default_dtype(torch.float32)` remains the safe baseline. PPO AMP is enabled on ROCm, and the default trainer configuration now prefers bf16 when supported, otherwise falls back to fp16. Any tensor exported to NumPy must cast through `.float().numpy()` first.

For Deep CFR, prefer `--amp-dtype bf16` on the RX 7900 XT. Keep `--cfr-compile` off unless you are explicitly testing it; worker state snapshots are normalized for compiled modules, but compile can still add ROCm warm-up overhead.

## Deep CFR Validation Ladder

Run these commands from the repository root with the project virtual environment available. `./run.sh` applies the ROCm startup environment and dispatches to `train.py`.

1. Serial smoke test:

`./run.sh --algo deep-cfr --iterations 2 --cfr-traversals 100 --cfr-hidden 64 --cfr-blocks 2 --cfr-batch-size 512 --cfr-adv-steps 100 --cfr-strat-steps 200 --amp-dtype bf16 --seed 0`

Expected: two iterations complete, no worker `state_dict` errors, and VRAM remains well below 6 GB.

2. Multiprocessing validation:

`./run.sh --algo deep-cfr --iterations 5 --cfr-traversals 256 --cfr-hidden 128 --cfr-blocks 3 --cfr-batch-size 1024 --cfr-adv-steps 1000 --cfr-strat-steps 2000 --cfr-num-workers 8 --cfr-worker-chunk 16 --amp-dtype bf16 --seed 0`

Expected: the worker pool starts, five iterations complete, and VRAM is typically around 10-13 GB.

3. Stable long-run profile:

`./run.sh --algo deep-cfr --iterations 200 --cfr-traversals 1500 --cfr-hidden 256 --cfr-blocks 4 --cfr-batch-size 4096 --cfr-adv-steps 4000 --cfr-strat-steps 8000 --cfr-num-workers 12 --cfr-worker-chunk 25 --cfr-eval-interval 5 --cfr-eval-hands 2000 --amp-dtype bf16 --seed 42`

Expected: steady iteration times, no `_orig_mod.` worker reload failures, and VRAM generally below 18 GB.

If VRAM approaches 18 GB or workers crash, reduce `--cfr-num-workers` to 8-10 or lower `--cfr-batch-size` to 2048. If VRAM stays under 14 GB and CPU utilization is low, try 14 workers or a larger `--cfr-worker-chunk` before increasing model size.

## Validated PPO Run Shape

The currently validated fast path on this hardware is:

- `batch=12288`
- `hidden=2560`
- `sims=16`
- `mp=20`
- `train_steps=12`
- `mcts_depth=0`
- `amp_dtype=bf16`
- `replay=true`

This path completed with stable ROCm startup, no fallback, and peak VRAM around 9 GB.

## Important Throughput Note

On the PPO path, `ROLLOUT_STEPS` is the main per-iteration workload knob. Raising `BATCH_SIZE` alone does not substantially increase GPU occupancy unless rollout size is also increased. The trainer now exposes `--rollout-steps` and an `aggressive` profile that can scale rollout size with the active batch cap.

The aggressive profile also uses an iteration-based warm-up ramp. During the first 5k iterations it starts from a lower effective rollout, simulation count, and PPO update depth, then ramps toward the configured aggressive targets. This keeps early training stable while still allowing the 7900 XT to fill out later in the run.

## Scaling Guidance

The next scale-up to test is:

- `batch=16384`
- `hidden=4096`
- `sims=32`
- `mp=20`
- `train_steps=16`
- `mcts_depth=0`

Use the new CLI override to sweep PPO update depth directly:

`python train.py --profile medium --batch 16384 --hidden 4096 --sims 32 --mp 20 --train-steps 16 --amp true --amp-dtype bf16 --replay true --mcts_max_depth 0`

For a higher-throughput ROCm run that increases actual PPO rollout work instead of only the logged batch cap:

`python train.py --profile aggressive --amp-dtype bf16 --rollout-steps 2048`

For the updated hardware-tuned aggressive baseline on the 7900X + 7900 XT rig:

`python train.py --profile aggressive --amp-dtype bf16 --rollout-steps 1024 --hidden-size 4096 --batch-size 16384 --num-simulations 1024 --train-steps 64 --iterations 10000`

## Monitoring

Use `rocm-smi --showmeminfo vram --showuse` for GPU load and `tail -f training.log` for startup/device confirmation and per-iteration timing. The trainer startup log prints `DEVICE`, `GPU Detected`, `VRAM Total`, `ROCm Fallback Applied`, and `train_steps`.
