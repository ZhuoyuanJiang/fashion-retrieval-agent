#!/bin/bash
# =============================================================================
# fashion-retrieval-agent — Plan_3 baseline run (v1)
# =============================================================================
# One command to produce the headline baseline number on the FACap dress
# evaluation slice (last 1000 train triplets), using speechQwen2VL in
# text-only mode as the VLM backend.
#
# Prerequisites:
#   - bash scripts/setup_server.sh has been run successfully
#   - `fashion_retrieval` conda env is activated:
#       conda activate fashion_retrieval
#   - GPU with >= 14 GB VRAM at bf16 (Qwen2-VL-7B fits at bf16 around 15 GB)
#
# What this script does:
#   1. Pre-fetches the eval slice's reference images from Marqo/fashion200k
#      into the local image cache (no surprise network calls mid-run)
#   2. Runs the baseline: VLM caption-generation + text-to-text retrieval
#      against the 59k FACap dress target captions, computes Recall@K +
#      median/mean rank, dumps qualitative samples
#   3. Pretty-prints the metrics file
#
# Outputs:
#   runs/baseline_v1_speechqwen2vl/caption_db/      caption embedding DB
#   runs/baseline_v1_speechqwen2vl/metrics.json     Recall@K + ranks
#   runs/baseline_v1_speechqwen2vl/qualitative/results.jsonl
#                                                    per-query top-10 + caption
#
# Usage:
#   conda activate fashion_retrieval
#   bash scripts/run_baseline_v1.sh
#
# Environment variables (optional overrides):
#   N_EVAL       eval slice size (default: 1000)
#   VLM          backend: speechqwen2vl | qwen2vl | oracle | mock (default: speechqwen2vl)
#   RUN_NAME     run directory name (default: baseline_v1_${VLM})
#   HF_HOME      HuggingFace cache location (default: ~/.cache/huggingface)
#   DB_SIZE      OPT-IN: if set, build a subset DB of this size for fast debug
#                iteration (eval targets guaranteed + sampled distractors).
#                Unset (default) = full DB encoding all ~59k captions, shared
#                across runs and reused across n_eval changes.
#   SEED         subset mode only: seed for distractor sampling (default: 42)
#   PROMPT_VARIANT  Plan 9: VLM prompt template — `concise` (default, original
#                   v1 prompt) or `detailed` (longer multi-attribute output)
# =============================================================================

set -e  # exit on any error

# ---------- config ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N_EVAL="${N_EVAL:-1000}"
VLM="${VLM:-speechqwen2vl}"
RUN_NAME="${RUN_NAME:-baseline_v1_${VLM}}"
DB_SIZE="${DB_SIZE:-}"      # empty = full mode (default), int = subset mode
SEED="${SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-concise}"

if [ -z "$DB_SIZE" ]; then
    DB_MODE_DESC="full (all captions, shared at runs/caption_db/<enc>/)"
else
    DB_MODE_DESC="subset (db_size=$DB_SIZE, seed=$SEED, runs/caption_db_subset/...)"
fi

echo "=== baseline run config ==="
echo "  vlm           : $VLM"
echo "  n_eval        : $N_EVAL"
echo "  run_name      : $RUN_NAME"
echo "  db_mode       : $DB_MODE_DESC"
echo "  prompt_variant: $PROMPT_VARIANT"
echo "  HF_HOME       : ${HF_HOME:-default (~/.cache/huggingface)}"
echo ""

# ---------- env sanity ----------
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "fashion_retrieval" ]; then
    echo "WARNING: CONDA_DEFAULT_ENV is '${CONDA_DEFAULT_ENV:-<unset>}', expected 'fashion_retrieval'."
    echo "         Run: conda activate fashion_retrieval"
    echo "         (continuing anyway — assuming you know what you're doing)"
    echo ""
fi

# ---------- step 1: prefetch images for the eval slice ----------
echo "=== Step 1/3: prefetch reference images ==="
python -m src.baseline.prepare_images \
    --n-eval "$N_EVAL"

# ---------- step 2: run the baseline ----------
echo ""
echo "=== Step 2/3: run baseline ($VLM, n_eval=$N_EVAL) ==="
RUN_BASELINE_ARGS=(
    --vlm "$VLM"
    --n-eval "$N_EVAL"
    --run-name "$RUN_NAME"
    --prompt-variant "$PROMPT_VARIANT"
)
if [ -n "$DB_SIZE" ]; then
    RUN_BASELINE_ARGS+=(--db-size "$DB_SIZE" --seed "$SEED")
fi
python -m src.baseline.run_baseline "${RUN_BASELINE_ARGS[@]}"

# ---------- step 3: pretty-print metrics ----------
echo ""
echo "=== Step 3/3: metrics summary ==="
python - <<PY
import json, pathlib
m = json.load(open(pathlib.Path("runs") / "$RUN_NAME" / "metrics.json"))
print(json.dumps(m, indent=2))
PY

echo ""
echo "=== Baseline run complete ==="
echo "Artifacts:"
echo "  runs/$RUN_NAME/metrics.json"
echo "  runs/$RUN_NAME/qualitative/results.jsonl"
if [ -z "$DB_SIZE" ]; then
    echo "  runs/caption_db/<encoder>/  (full DB, shared, reused across runs)"
else
    echo "  runs/caption_db_subset/eval${N_EVAL}_db${DB_SIZE}_seed${SEED}/<encoder>/"
fi
echo ""
echo "Next: write Documentation/Baseline_1_Report_<YYYYMMDD>.md from these."
