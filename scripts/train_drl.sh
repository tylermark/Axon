#!/usr/bin/env bash
# Axon — Launch DRL Training (panelization + placement)
# Reference: ARCHITECTURE.md §Training Pipeline Phase D, TASKS.md TR-005
#
# Usage:
#   ./scripts/train_drl.sh                                            # defaults
#   ./scripts/train_drl.sh --total-timesteps 1000000 --device cuda
#   ./scripts/train_drl.sh --no-resplan --wandb-enabled
set -euo pipefail

echo "══════════════════════════════════════════"
echo " Axon DRL Training (Panelization + Placement)"
echo "══════════════════════════════════════════"

# Check if --no-resplan flag is present
USE_RESPLAN=true
for arg in "$@"; do
    if [ "$arg" = "--no-resplan" ]; then
        USE_RESPLAN=false
        break
    fi
done

RESPLAN_PATH="${RESPLAN_PATH:-datasets/ResPlan/ResPlan.pkl}"
if [ "$USE_RESPLAN" = "true" ]; then
    if [ ! -f "$RESPLAN_PATH" ]; then
        echo "ERROR: Dataset not found: $RESPLAN_PATH" >&2
        echo "  Set RESPLAN_PATH or place the file at the default location." >&2
        exit 1
    fi
fi

python -m src.training.drl_training \
    --checkpoint-dir checkpoints/drl \
    --total-timesteps 500000 \
    --resplan-path "$RESPLAN_PATH" \
    "$@"