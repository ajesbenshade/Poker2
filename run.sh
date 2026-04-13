#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF="garbage_collect_threshold:0.6,expandable_segment:True,max_split_size_mb:128"
export OMP_NUM_THREADS=1

exec /home/aaron/Poker2/.venv/bin/python /home/aaron/Poker2/train.py "$@"
