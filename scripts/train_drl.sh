#!/usr/bin/env bash
# Axon — Launch DRL Training (panelization + placement)
# Reference: ARCHITECTURE.md §Training Pipeline Phase D, TASKS.md TR-005
#
# Usage:
#   ./scripts/train_drl.sh                                            # defaults
#   ./scripts/train_drl.sh --total-timesteps 1000000 --device cuda
#   ./scripts/train_drl.sh --no-resplan --wandb-enabled
set -euo pipefail

echo "════════��══════════════════════════════════"
echo " Axon DRL Training (Panelization + Placement)"
echo "════════════════════════════════════��══════"

python -m src.training.drl_training \
    --checkpoint-dir checkpoints/drl \
    --total-timesteps 500000 \
    --resplan-path datasets/ResPlan/ResPlan.pkl \
    "$@"
