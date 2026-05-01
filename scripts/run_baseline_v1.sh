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
#   DB_SIZE      caption DB size (default: same as full FACap dress targets)
#   VLM          backend: speechqwen2vl | qwen2vl | oracle | mock (default: speechqwen2vl)
#   RUN_NAME     run directory name (default: baseline_v1_${VLM})
#   HF_HOME      HuggingFace cache location (default: ~/.cache/huggingface)
# =============================================================================

set -e  # exit on any error

# ---------- config ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N_EVAL="${N_EVAL:-1000}"
DB_SIZE="${DB_SIZE:-59082}"           # full FACap dress target set
VLM="${VLM:-speechqwen2vl}"
RUN_NAME="${RUN_NAME:-baseline_v1_${VLM}}"

echo "=== baseline run config ==="
echo "  vlm        : $VLM"
echo "  n_eval     : $N_EVAL"
echo "  db_size    : $DB_SIZE"
echo "  run_name   : $RUN_NAME"
echo "  HF_HOME    : ${HF_HOME:-default (~/.cache/huggingface)}"
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
python -m src.baseline.run_baseline \
    --vlm "$VLM" \
    --n-eval "$N_EVAL" \
    --db-size "$DB_SIZE" \
    --run-name "$RUN_NAME"

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
echo "  runs/$RUN_NAME/caption_db/"
echo ""
echo "Next: write Documentation/Baseline_1_Report_<YYYYMMDD>.md from these."
