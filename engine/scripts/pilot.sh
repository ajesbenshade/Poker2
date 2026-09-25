#!/bin/bash
# 24-hour pilot (roadmap days 8-9): full blueprint bet sizes on the 200-bucket
# abstraction. Measured ~140K iterations/s on 24 threads once the strategy is
# past uniform (a fresh run starts near 300K/s because random play ends hands early):
#   epoch 50M iterations         ~6 minutes
#   Linear CFR for 3.4B          ~first 400 minutes (as in Pluribus)
#   pruning after 1.7B           ~200 minutes (as in Pluribus)
# Build the abstraction first:
#   engine/build/poker2_abstraction --out ~/poker2-data/abstraction-small --buckets 200,200,200
# From Windows, start it detached so it survives closing terminals:
#   powershell -ExecutionPolicy Bypass -File engine\scripts\start_pilot.ps1
# Stop early (checkpoints, then exits):  pkill -TERM -x poker2_train
# Resume:                                engine/scripts/pilot.sh --resume
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RUN:-$HOME/poker2-runs/pilot}"
mkdir -p "$RUN"
# setsid: its own session, so losing the launching terminal cannot signal it.
exec setsid --wait nice -n 10 "$ENGINE/build/poker2_train" \
  --abstraction "$HOME/poker2-data/abstraction-small" \
  --run "$RUN" \
  --profile blueprint \
  --hours 24 \
  --epoch 50000000 \
  --linear-until 3400000000 \
  --prune-after 1700000000 \
  --checkpoint-minutes 60 \
  --eval-minutes 30 \
  --lbr-hands 100000 \
  --h2h-deals 50000 \
  "$@"
