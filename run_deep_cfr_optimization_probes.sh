#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

probe="${1:-}"
iterations="${ITERATIONS:-2}"
seed="${SEED:-42}"

base_args=(
  --algo deep-cfr
  --iterations "$iterations"
  --cfr-traversals 2000
  --cfr-hidden 512
  --cfr-blocks 6
  --cfr-adv-buf-size 3000000
  --cfr-strat-buf-size 8000000
  --cfr-worker-chunk 25
  --cfr-async
  --cfr-latest-interval 999999
  --cfr-eval-interval 0
  --cfr-lbr-interval 0
  --amp-dtype bf16
  --seed "$seed"
)

run_probe() {
  local name="$1"
  shift
  local ckpt="checkpoints/deep_cfr_probe_${name}"
  local logs="runs/deep_cfr_probe_${name}"
  local logfile="phase_b_probe_${name}.log"
  echo "running probe: ${name} (iterations=${iterations}, seed=${seed})"
  ./run.sh "${base_args[@]}" \
    --cfr-checkpoint-dir "$ckpt" \
    --cfr-log-dir "$logs" \
    "$@" 2>&1 | tee "$logfile"
}

case "$probe" in
  script-8x2)
    run_probe script_8x2 \
      --cfr-num-workers 8 \
      --cfr-worker-threads 2 \
      --cfr-worker-script \
      --cfr-batch-size 4096 \
      --cfr-adv-steps 4000 \
      --cfr-strat-steps 9000
    ;;
  workers-16)
    run_probe workers_16 \
      --cfr-num-workers 16 \
      --cfr-worker-threads 1 \
      --cfr-batch-size 4096 \
      --cfr-adv-steps 4000 \
      --cfr-strat-steps 9000
    ;;
  batch-8192)
    run_probe batch_8192 \
      --cfr-num-workers 12 \
      --cfr-worker-threads 1 \
      --cfr-batch-size 8192 \
      --cfr-adv-steps 2000 \
      --cfr-strat-steps 4500
    ;;
  batch-8192-concurrent)
    run_probe batch_8192_concurrent \
      --cfr-num-workers 12 \
      --cfr-worker-threads 1 \
      --cfr-pin-batches \
      --cfr-concurrent-adv \
      --cfr-batch-size 8192 \
      --cfr-adv-steps 2000 \
      --cfr-strat-steps 4500
    ;;
  lowrisk-combo)
    run_probe lowrisk_combo \
      --cfr-num-workers 8 \
      --cfr-worker-threads 2 \
      --cfr-worker-script \
      --cfr-async-depth 2 \
      --cfr-pin-batches \
      --cfr-concurrent-adv \
      --cfr-batch-size 8192 \
      --cfr-adv-steps 2000 \
      --cfr-strat-steps 4500
    ;;
  vectorized-smoke)
    run_probe vectorized_smoke \
      --cfr-traversals 64 \
      --cfr-hidden 64 \
      --cfr-blocks 1 \
      --cfr-num-workers 2 \
      --cfr-worker-threads 1 \
      --cfr-traversal-backend vectorized \
      --cfr-vectorized-batch-size 8 \
      --cfr-batch-size 256 \
      --cfr-adv-steps 8 \
      --cfr-strat-steps 8
    ;;
  vectorized-strong)
    run_probe vectorized_strong \
      --cfr-num-workers 12 \
      --cfr-worker-threads 1 \
      --cfr-traversal-backend vectorized \
      --cfr-vectorized-batch-size 16 \
      --cfr-batch-size 4096 \
      --cfr-adv-steps 4000 \
      --cfr-strat-steps 9000
    ;;
  vectorized-proxy-smoke)
    run_probe vectorized_proxy_smoke \
      --cfr-traversals 64 \
      --cfr-hidden 128 \
      --cfr-blocks 2 \
      --cfr-num-workers 2 \
      --cfr-worker-threads 1 \
      --cfr-traversal-backend vectorized \
      --cfr-vectorized-batch-size 8 \
      --cfr-proxy-nets \
      --cfr-proxy-hidden 64 \
      --cfr-proxy-blocks 1 \
      --cfr-proxy-refresh 1 \
      --cfr-proxy-steps 8 \
      --cfr-batch-size 256 \
      --cfr-adv-steps 8 \
      --cfr-strat-steps 8
    ;;
  *)
    cat <<'USAGE'
Usage: ITERATIONS=2 ./run_deep_cfr_optimization_probes.sh <probe>

Probes:
  script-8x2              8 workers, 2 torch threads, scripted worker nets
  workers-16              16 workers, 1 torch thread
  batch-8192              batch 8192 with proportionally fewer train steps
  batch-8192-concurrent   batch 8192 plus pinned batches and concurrent advantage training
  lowrisk-combo           worker script + async depth 2 + batch/concurrent/pinned combo
  vectorized-smoke        tiny vectorized traversal backend smoke probe
  vectorized-strong       strong-shape vectorized traversal timing probe
  vectorized-proxy-smoke  tiny vectorized traversal plus proxy-net smoke probe

Each probe writes isolated checkpoints, TensorBoard logs, and a phase_b_probe_*.log file.
USAGE
    exit 2
    ;;
esac
