# PPO-First ROCm Optimization Guide
## For Ryzen 9 7900X + RX 7900 XT (20GB VRAM) + 64GB RAM

This repository's active high-performance path is PPO in `train.py` and `rl.py`, not the older Deep CFR flow.

## Critical Startup Order

Call `initialize_rocm_runtime()` before any torch import. The shared runtime in `environment.py` normalizes allocator settings, applies ROCm defaults, probes the GPU, and logs fallback status before model code loads.

## Current ROCm Environment

Use `PYTORCH_ALLOC_CONF`, not `PYTORCH_HIP_ALLOC_CONF`. The current default configuration is:

- `HIP_VISIBLE_DEVICES=0`
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512`
- `HSA_ENABLE_SDMA=0`
- `TORCH_CUDNN_ENABLE=0`
- `OMP_NUM_THREADS=1`

## Precision and Safety

`torch.set_default_dtype(torch.float32)` remains the safe baseline. PPO AMP is enabled on ROCm, and the default trainer configuration now prefers bf16 when supported, otherwise falls back to fp16. Any tensor exported to NumPy must cast through `.float().numpy()` first.

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
