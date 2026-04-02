#!/usr/bin/env bash
# Axon — Launch SFT + GRPO fine-tuning
# Reference: ARCHITECTURE.md §Training Pipeline Phase B, C
#
# Usage:
#   ./scripts/finetune.sh                                       # defaults
#   ./scripts/finetune.sh --epochs 100 --device cuda
#   SKIP_GRPO=1 ./scripts/finetune.sh                           # SFT only
set -euo pipefail

echo "══════════��═══════════════════════════���════"
echo " Axon SFT + GRPO Fine-Tuning"
echo "═══════════════════════════════════════════"

# Phase 1: Supervised Fine-Tuning
echo ""
echo "── Phase 1: SFT ──"
python -m src.training.sft \
    --data-root datasets/ \
    --pretrained-checkpoint checkpoints/mpm/best.pt \
    --checkpoint-dir checkpoints/sft \
    --epochs 50 \
    "$@"

# Phase 2: GRPO Quality Annealing (uses SFT checkpoint as reference)
if [ "${SKIP_GRPO:-0}" != "1" ]; then
    echo ""
    echo "── Phase 2: GRPO ──"
    python -m src.training.grpo \
        --sft-checkpoint checkpoints/sft/best.pt \
        --checkpoint-dir checkpoints/grpo \
        --iterations 1000 \
        "$@"
else
    echo ""
    echo "Skipping GRPO (SKIP_GRPO=1)."
fi

echo ""
echo "Fine-tuning complete."
