#!/usr/bin/env bash
# Axon — Launch SFT + GRPO fine-tuning
# Reference: ARCHITECTURE.md §Training Pipeline Phase B, C
#
# Usage:
#   ./scripts/finetune.sh                                       # defaults
#   ./scripts/finetune.sh --epochs 100 --device cuda
#   SKIP_GRPO=1 ./scripts/finetune.sh                           # SFT only
set -euo pipefail

echo "══════════════════════════════════════════"
echo " Axon SFT + GRPO Fine-Tuning"
echo "══════════════════════════════════════════"

# Validate prerequisites
DATA_ROOT="${DATA_ROOT:-datasets/}"
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found: $DATA_ROOT" >&2
    echo "  Set DATA_ROOT or create the directory with training data." >&2
    exit 1
fi

MPM_CHECKPOINT="${MPM_CHECKPOINT:-checkpoints/mpm/best.pt}"
if [ ! -f "$MPM_CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $MPM_CHECKPOINT" >&2
    echo "  Set MPM_CHECKPOINT or run pretraining first." >&2
    exit 1
fi

# Phase 1: Supervised Fine-Tuning
echo ""
echo "── Phase 1: SFT ──"
python -m src.training.sft \
    --data-root "$DATA_ROOT" \
    --pretrained-checkpoint "$MPM_CHECKPOINT" \
    --checkpoint-dir checkpoints/sft \
    --epochs 50 \
    "$@"

# Phase 2: GRPO Quality Annealing (uses SFT checkpoint as reference)
if [ "${SKIP_GRPO:-0}" != "1" ]; then
    SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/sft/best.pt}"
    if [ ! -f "$SFT_CHECKPOINT" ]; then
        echo "ERROR: SFT checkpoint not found: $SFT_CHECKPOINT" >&2
        echo "  SFT may have failed or the path is wrong." >&2
        exit 1
    fi

    echo ""
    echo "── Phase 2: GRPO ──"
    python -m src.training.grpo \
        --sft-checkpoint "$SFT_CHECKPOINT" \
        --checkpoint-dir checkpoints/grpo \
        --iterations 1000 \
        "$@"
else
    echo ""
    echo "Skipping GRPO (SKIP_GRPO=1)."
fi

echo ""
echo "Fine-tuning complete."
