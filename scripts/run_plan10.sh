#!/usr/bin/env bash
# Plan-10 V1 (Option B) launcher: Qwen2VL/Qwen2VL two-tower contrastive training.
#
# Differences vs run_plan5.sh:
#   - No target cache build step. The target tower is trainable, so the gallery
#     is encoded dynamically inside train_plan10.py (startup + end-of-epoch).
#   - Defaults are tuned for Option B (two backbones, ~30 GB resident): bs=8,
#     8 GPUs, gather=ON.
#   - Default --run-dir lives under runs_local_plan10/ (server10-local;
#     the runs/ symlink only resolves on server6).
#
# Usage:
#   bash scripts/run_plan10.sh                # 8-GPU launch, bs=8, gather=ON
#   bash scripts/run_plan10.sh --batch-size 6 # smaller batch
#   bash scripts/run_plan10.sh --first-eval-step 5 [other flags...]
#
# All unrecognized args are forwarded to train_plan10.py.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure HF caches land on local SSD, not NAS home quota
export HF_HOME="${HF_HOME:-/ssd1/zhuoyuan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export WANDB_DIR="${WANDB_DIR:-/ssd1/zhuoyuan/wandb_cache}"

# Make `from src...` imports work when accelerate spawns workers
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_PYTHON="$(conda run -n fashion_retrieval which python 2>/dev/null || echo python)"
CONDA_ACCEL="$(conda run -n fashion_retrieval which accelerate 2>/dev/null || echo accelerate)"

# ── Defaults ────────────────────────────────────────────────────────────────
NUM_GPUS=8
RUN_DIR="runs_local_plan10/run_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=8
GATHER_FLAG="--gather"
EXTRA_ARGS=()

# ── Parse our flags; pass the rest to train_plan10.py ──────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)    NUM_GPUS="$2";        shift 2 ;;
        --run-dir)     RUN_DIR="$2";         shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";      shift 2 ;;
        --no-gather)   GATHER_FLAG="";       shift ;;
        *)             EXTRA_ARGS+=("$1");   shift ;;
    esac
done

mkdir -p "$RUN_DIR"

# ── Launch training ─────────────────────────────────────────────────────────
TRAIN_ARGS=(
    --run-dir   "$RUN_DIR"
    --batch-size "$BATCH_SIZE"
)
if [[ -n "$GATHER_FLAG" ]]; then
    TRAIN_ARGS+=("$GATHER_FLAG")
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "=== Multi-GPU launch (${NUM_GPUS} GPUs) ==="
    TRAIN_CMD=(
        "$CONDA_ACCEL" launch
        --num_processes "$NUM_GPUS"
        src/training/train_plan10.py
        "${TRAIN_ARGS[@]}"
    )
else
    echo "=== Single-GPU launch ==="
    TRAIN_CMD=(
        "$CONDA_PYTHON" src/training/train_plan10.py
        "${TRAIN_ARGS[@]}"
    )
fi

echo "=== Starting Plan-10 V1 (Option B) training: ${RUN_DIR} ==="
echo "    cmd: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
