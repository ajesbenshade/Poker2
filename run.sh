#!/usr/bin/env bash
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
	echo "Virtual environment not found at $VENV_PYTHON"
	echo "Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
	exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/train.py" "$@"
