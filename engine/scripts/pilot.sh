#!/bin/bash
# 24-hour pilot (roadmap days 8-9): full blueprint bet sizes on the 200-bucket
# abstraction. At ~310K iterations/s on 24 threads:
#   epoch 100M iterations        ~5.5 minutes
#   Linear CFR for 7.2B          ~first 400 minutes (as in Pluribus)
#   pruning after 3.6B           ~200 minutes (as in Pluribus)
# Build the abstraction first:
#   engine/build/poker2_abstraction --out ~/poker2-data/abstraction-small --buckets 200,200,200
# Stop early (checkpoints, then exits):  pkill -TERM poker2_train
# Resume:                                engine/scripts/pilot.sh --resume
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RUN:-$HOME/poker2-runs/pilot}"
mkdir -p "$RUN"
exec nice -n 10 "$ENGINE/build/poker2_train" \
  --abstraction "$HOME/poker2-data/abstraction-small" \
  --run "$RUN" \
  --profile blueprint \
  --hours 24 \
  --epoch 100000000 \
  --linear-until 7200000000 \
  --prune-after 3600000000 \
  --checkpoint-minutes 60 \
  --eval-minutes 30 \
  --lbr-hands 100000 \
  --h2h-deals 50000 \
  "$@"
