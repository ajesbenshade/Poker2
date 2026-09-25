#!/bin/bash
# Full blueprint run (roadmap days 9-27): blueprint bet sizes on the full
# 169/2000/2000/1500-bucket abstraction, ~22.6 GB of tables (float regrets,
# double preflop/flop averages, float turn/river averages).
# Schedule by training time, as in Pluribus: Linear CFR for the first 400
# minutes, pruning after 200 minutes. --hours counts wall time from each start,
# so a resumed run gets a fresh budget; stop it when the month's plan says so.
# Build the abstraction first:  engine/build/poker2_abstraction
# From Windows, start it detached:
#   powershell -ExecutionPolicy Bypass -File engine\scripts\start_run.ps1 -Script blueprint
# Stop (checkpoints, then exits):  pkill -TERM -x poker2_train
# Resume:                          engine/scripts/blueprint.sh --resume
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RUN:-$HOME/poker2-runs/blueprint}"
mkdir -p "$RUN"
# setsid: its own session, so losing the launching terminal cannot signal it.
exec setsid --wait nice -n 10 "$ENGINE/build/poker2_train" \
  --abstraction "$HOME/poker2-data/abstraction" \
  --run "$RUN" \
  --profile blueprint \
  --hours 432 \
  --epoch 50000000 \
  --linear-minutes 400 \
  --prune-after-minutes 200 \
  --checkpoint-minutes 120 \
  --eval-minutes 60 \
  --lbr-hands 100000 \
  --h2h-deals 50000 \
  "$@"
