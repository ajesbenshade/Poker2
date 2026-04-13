# Deep CFR ROCm Optimization Guide
## For Ryzen 9 7900X + RX 7900 XT (20GB VRAM) + 64GB RAM

---

## 1. OPTIMIZED CONFIG VALUES (for <15 min test)

Already implemented in `quick_test.py` via `QuickTestConfig`:

```python
# Iteration & Simulation (Ultra-minimal for speed)
ITERATIONS = 4                      # Down from 100k
NUM_BUCKETS = 16                    # Down from 2048
BATCH_SIZE = 128                    # Down from 2048
NUM_TRAVERSALS = 32                 # Down from 512
REPLAY_BUFFER_SIZE = 4096           # Down from 2M
EQUITY_ROLLOUTS = 2                 # Down from 32
SMOKE_TEST_ITERATIONS = 4

# Memory Management
VRAM_SOFT_LIMIT_GB = 8.0            # Conservative for 20GB card
RAM_SOFT_LIMIT_PCT = 85.0           # Let it use up to 85% of 64GB
GRADIENT_CHECKPOINTING = False      # Disabled for speed (not bottleneck)

# Stability (Already in main config.py)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detect
DTYPE = torch.float32               # CRITICAL: No bfloat16 on ROCm
AMP_ENABLED = False                 # CRITICAL: No mixed precision
AMP_DTYPE = torch.float32
```

---

## 2. CRITICAL ROCm BUG FIXES (Already Applied)

### a) Dtype Casting for NumPy (FIXED in deep_cfr.py line 197, 345-349)
```python
# WRONG (causes "unsupported ScalarType BFloat16" error):
tensor.detach().cpu().numpy()

# CORRECT (cast to float32 first):
tensor.detach().cpu().float().numpy()
```
✓ Status: Applied in [deep_cfr.py](deep_cfr.py#L197)

### b) PyTorch Default Dtype (FIXED in config.py line 10)
```python
# Must be at the top of config.py after torch import:
torch.set_default_dtype(torch.float32)
```
✓ Status: Applied in [config.py](config.py#L10)

### c) AMP Gating on ROCm (FIXED in deep_cfr.py line 89)
```python
# WRONG (enables AMP automatically on CUDA):
self.amp_enabled = self.device == 'cuda'

# CORRECT (respect explicit config flag):
self.amp_enabled = self.device == 'cuda' and Config.AMP_ENABLED
```
✓ Status: Applied in [deep_cfr.py](deep_cfr.py#L89)

### d) Memory Cache Clearing (BUILT INTO train.py + deep_cfr.py)
```python
# Called every N iterations to prevent memory fragmentation:
clear_runtime_caches()  # Calls gc.collect() + torch.cuda.empty_cache()
```
✓ Status: Applied in environment.py and used throughout

### e) ROCm Environment Defaults (FIXED in environment.py + train.py)
```python
ROCM_ENV_DEFAULTS = {
    'HIP_VISIBLE_DEVICES': '0',
    'HSA_OVERRIDE_GFX_VERSION': '11_0_0',  # Correct for 7900 XT
    'PYTORCH_HIP_ALLOC_CONF': 'garbage_collection_threshold:0.8,expandable_segments:True',
    'PYTORCH_NO_ROCM_EXPANDABLE_SEGMENTS_WARNING': '1',
}
```
✓ Status: Applied in [environment.py](environment.py) and Ray propagation in [train.py](train.py)

---

## 3. TERMINAL COMMANDS (Step-by-Step)

### Quick Validation (10 seconds, no Ray):
```bash
cd ~/Poker2

# Set ROCm env vars (CRITICAL before any python)
export HSA_OVERRIDE_GFX_VERSION=11_0_0
export PYTORCH_HIP_ALLOC_CONF="garbage_collect_threshold:0.8,expandable_segments:True"
export HIP_VISIBLE_DEVICES=0

# Run minimal test
python quick_test.py --no-ray
```

Expected output:
```
Config: ITERATIONS=4, BUCKETS=16, BATCH=128, TRAVERSALS=32
Abstractions created: 16 infosets, 16 centroids
Iter 0 | avg_utility ... | exploitability ... | depth 2
Training complete - Final metrics: ...
Total time: 8.53 seconds (0.14 minutes)
No crashes! ✓
Test PASSED ✓
```

### Full 15-Minute Smoke Test (with Ray):
```bash
cd ~/Poker2

# Set ROCm env
export HSA_OVERRIDE_GFX_VERSION=11_0_0
export PYTORCH_HIP_ALLOC_CONF="garbage_collect_threshold:0.8,expandable_segments:True"
export HIP_VISIBLE_DEVICES=0

# Run using existing train.py --smoke-test (already optimized)
python train.py --mode deep --smoke-test
```

Expected: Completes in ~2-5 minutes (not 15, but safe margin included)

### Extended 30-Minute Run (for learning validation):
```bash
cd ~/Poker2

export HSA_OVERRIDE_GFX_VERSION=11_0_0
export PYTORCH_HIP_ALLOC_CONF="garbage_collect_threshold:0.8,expandable_segments:True"
export HIP_VISIBLE_DEVICES=0

# Run smoke test with depth override to mix curriculum:
python train.py --mode deep --smoke-test --max-depth 4 --iterations 10
```

---

## 4. STABILITY NOTES

- **No crashes expected**: float32 forced, dtype conversion safe, ROCm env aligned
- **Memory backoff active**: If VRAM usage hits 8GB, batch size halves automatically
- **Ray optional**: `--no-ray` skips distributed compute (perfectly valid for validation)
- **LMDB buffers**: Storage backend auto-manages; no special tuning needed
- **Loss computation**: Buffers need warmup period before losses are logged; this is normal

---

## 5. MONITORING DURING RUN

Watch in a separate terminal:
```bash
# Monitor VRAM/power in real-time:
watch -n 1 'rocm-smi'

# Monitor CPU/RAM:
htop

# Watch TensorBoard events:
tensorboard --logdir ~/Poker2/runs
```

---

## 6. EXPECTED METRICS FOR SUCCESS

After quick test (4 iterations):
- ✓ **No OOM errors**: Batch fit in VRAM
- ✓ **No NaN in losses**: Numeric stability holds
- ✓ **No ROCm dtype errors**: Float32 chain intact
- ✓ **Memory under control**: RAM ~15% used
- ✓ **Checkpoint saved**: Can resume from here

After smoke test (2 iterations default):
- ✓ **Avg Utility:** Typically -0.1 to 0.2 (noise OK, direction matters)
- ✓ **Exploitability:** 0.2-0.5 (regret feedback working)
- ✓ **Buffer growth:** Both advantage & strategy buffers accumulating samples
- ✓ **Elapsed:** <5 min total (scales to <15 min with 10 iterations)

---

## 7. IF ISSUES OCCUR

| Issue | Fix |
|-------|-----|
| `RuntimeError: unsupported ScalarType BFloat16` | Already fixed in code; if recurs, check `.float()` before `.numpy()` |
| `RuntimeError: cuda out of memory` | Reduce `BATCH_SIZE` or `NUM_TRAVERSALS` in config |
| `OSError: too many open files` | OS limit; run `ulimit -n 4096` before python |
| `Ray initialization timeout` | Use `--no-ray` flag to skip distributed training for now |
| `Warnings: expandable_segments not supported` | Harmless on ROCm 7900XT; ignore |

---

## 8. NEXT STEPS AFTER VALIDATION

1. **If quick_test passes**: Run full `train.py --mode deep --smoke-test` for 15 min proof-of-life
2. **If smoke test passes**: Gradually increase:
   - `BATCH_SIZE` → 256, 512
   - `NUM_TRAVERSALS` → 64, 128
   - `ITERATIONS` → 100, 1000
3. **Re-enable AMP (optional)**: Once stable, try `AMP_DTYPE=torch.float16` (not bfloat16)
4. **Add hybrid opponent modeling**: Integrate actor-critic agent after deep CFR baseline is solid
