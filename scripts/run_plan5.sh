#!/usr/bin/env bash
# Plan-5 orchestrator: build target cache (once), then launch training.
#
# Usage:
#   Single-GPU (Path A — A6000-class 48 GB):
#     bash scripts/run_plan5.sh --batch-size 32 [extra args...]
#
#   Multi-GPU (Path B — 8× 3090-class 24 GB with all_gather):
#     bash scripts/run_plan5.sh --multi-gpu --num-gpus 8 --batch-size 4 --gather [extra args...]
#
#   Profile only (pick batch size):
#     bash scripts/run_plan5.sh --profile
#
#   Smoke test (plumbing check):
#     bash scripts/run_plan5.sh --smoke --batch-size 4
#
# All unrecognized args are forwarded to train_plan5.py.
# Decision: Path A vs Path B is made at smoke-test time based on --profile output.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure HF caches land on local SSD, not NAS home quota
export HF_HOME="${HF_HOME:-/ssd1/zhuoyuan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export WANDB_DIR="${WANDB_DIR:-/ssd1/zhuoyuan/wandb_cache}"

# Make `from src...` imports work when accelerate spawns workers (sys.path[0]
# is the script dir, not repo root, so `src` is not importable without this)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_PYTHON="$(conda run -n fashion_retrieval which python 2>/dev/null || echo python)"
CONDA_ACCEL="$(conda run -n fashion_retrieval which accelerate 2>/dev/null || echo accelerate)"

# ── Defaults ────────────────────────────────────────────────────────────────
ENCODER_ID="marqo-fashionclip"
CACHE_DIR="runs/plan5"
RUN_DIR="runs/plan5/run_$(date +%Y%m%d_%H%M%S)"
MULTI_GPU=0
NUM_GPUS=1
EXTRA_ARGS=()

# ── Parse our flags; pass the rest to train_plan5.py ────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-gpu)   MULTI_GPU=1;          shift ;;
        --num-gpus)    NUM_GPUS="$2";        shift 2 ;;
        --encoder-id)  ENCODER_ID="$2";      shift 2 ;;
        --cache-dir)   CACHE_DIR="$2";       shift 2 ;;
        --run-dir)     RUN_DIR="$2";         shift 2 ;;
        *)             EXTRA_ARGS+=("$1");   shift ;;
    esac
done

mkdir -p "$CACHE_DIR" "$RUN_DIR"

# ── Step 1: Build target embedding cache (skip if already built) ─────────────
CACHE_NPY="${CACHE_DIR}/target_emb_cache_${ENCODER_ID}.npy"
if [[ ! -f "$CACHE_NPY" ]]; then
    echo "=== Building target embedding cache for '${ENCODER_ID}' ==="
    python -m src.training.target_cache \
        --encoder-id "$ENCODER_ID" \
        --out-dir "$CACHE_DIR"
else
    echo "=== Target cache exists, skipping build: ${CACHE_NPY} ==="
fi

# ── Step 2: Launch training ──────────────────────────────────────────────────
TRAIN_CMD=(
    "$CONDA_PYTHON" src/training/train_plan5.py
    --encoder-id "$ENCODER_ID"
    --cache-dir  "$CACHE_DIR"
    --run-dir    "$RUN_DIR"
    "${EXTRA_ARGS[@]}"
)

if [[ $MULTI_GPU -eq 1 ]]; then
    echo "=== Multi-GPU launch (${NUM_GPUS} GPUs) ==="
    TRAIN_CMD=(
        "$CONDA_ACCEL" launch
        --num_processes "$NUM_GPUS"
        src/training/train_plan5.py
        --encoder-id "$ENCODER_ID"
        --cache-dir  "$CACHE_DIR"
        --run-dir    "$RUN_DIR"
        "${EXTRA_ARGS[@]}"
    )
fi

echo "=== Starting training: ${RUN_DIR} ==="
echo "    cmd: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
