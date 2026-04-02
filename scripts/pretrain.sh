#!/usr/bin/env bash
# Axon — Launch SSL pre-training (Masked Primitive Modeling)
# Reference: ARCHITECTURE.md §Training Pipeline Phase A
#
# Usage:
#   ./scripts/pretrain.sh                          # defaults
#   ./scripts/pretrain.sh --epochs 200 --device cuda
#   ./scripts/pretrain.sh --data-root /data/axon --batch-size 32
set -euo pipefail

echo "═══════════════════════════════════════════"
echo " Axon MPM Pre-Training"
echo "═══════════════════════════════════════════"

python -m src.training.mpm \
    --data-root datasets/ \
    --checkpoint-dir checkpoints/mpm \
    --epochs 100 \
    --batch-size 16 \
    --mask-ratio 0.8 \
    --device auto \
    "$@"
