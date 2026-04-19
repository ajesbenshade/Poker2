#!/usr/bin/env bash
# Quick Test Run Instructions for Deep CFR on AMD ROCm (7900X + 7900XT + 64GB)
# Target: Complete in <15 minutes, no crashes, visible learning

echo "================================"
echo "QUICK TEST: Deep CFR ROCm Validation"
echo "================================"

# Step 1: Set ROCm environment variables (MUST DO BEFORE RUNNING PYTHON)
export HSA_OVERRIDE_GFX_VERSION=11_0_0
export PYTORCH_HIP_ALLOC_CONF="garbage_collect_threshold:0.8,expandable_segments:True"
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

echo "ROCm environment configured:"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"

# Step 2: Activate Python environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Step 3: Run quick test (skip Ray for stability in validation)
echo ""
echo "Starting quick_test.py (will complete in ~10 seconds)..."
python quick_test.py --no-ray

# Step 4: Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ QUICK TEST PASSED - No crashes!"
    echo "  Next: Run full train.py --mode deep --smoke-test for 15-min validation"
else
    echo ""
    echo "✗ QUICK TEST FAILED - Check logs above"
    exit 1
fi

# Step 5: Optional - Run extended smoke test (if quick test succeeded)
echo ""
echo "Running extended 15-minute smoke test..."
python train.py --mode deep --smoke-test

echo ""
echo "================================"
echo "Test run complete!"
echo "Check logs/TensorBoard at: runs/"
echo "================================"
