#!/bin/bash
# =============================================================================
# Encoder ablation: replay the speechqwen2vl baseline with each encoder in
# the zoo. VLM step is skipped (we reuse the saved generated_captions),
# only the text-encoder + retrieval is rerun per encoder.
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_RUN="${SOURCE_RUN:-runs/baseline_v1_speechqwen2vl}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Encoder slug -> assigned GPU index. 8 GPUs available; spread the heavy
# 4-8B models onto separate GPUs, batch the small ones together later.
declare -A GPU
GPU[mpnet-base]=0
GPU[bge-large]=0
GPU[e5-large-v2]=1
GPU[gte-modernbert-base]=1
GPU[clip-vit-l-14]=2
GPU[marqo-fashionclip]=2
GPU[marqo-fashionsiglip]=3
GPU[qwen3-emb-0.6b]=3
GPU[qwen3-emb-4b]=4
GPU[qwen3-emb-8b]=5
GPU[nv-embed-v2]=6

# Order: launch heavy ones first so they aren't blocked at the end.
SLUGS=(
    qwen3-emb-8b
    nv-embed-v2
    qwen3-emb-4b
    qwen3-emb-0.6b
    bge-large
    e5-large-v2
    mpnet-base
    gte-modernbert-base
    clip-vit-l-14
    marqo-fashionclip
    marqo-fashionsiglip
)

source /home/zhuoyuan/miniconda3/etc/profile.d/conda.sh
conda activate fashion_retrieval

echo "=== encoder swap orchestrator ==="
echo "  source run : $SOURCE_RUN"
echo "  log dir    : $LOG_DIR"
echo "  encoders   : ${#SLUGS[@]}"
echo ""

# Files we'll touch for status tracking.
STATUS_FILE="$LOG_DIR/encoder_swap_status.txt"
> "$STATUS_FILE"

# Track PIDs to wait on at the end.
declare -A PIDS

for slug in "${SLUGS[@]}"; do
    gpu="${GPU[$slug]:-0}"
    log="$LOG_DIR/encoder_swap_${slug}.log"
    run_name="baseline_v1_speechqwen2vl_${slug}"

    # Skip already-completed runs (so re-running is idempotent).
    if [ -f "runs/${run_name}/metrics.json" ]; then
        echo "[skip ] $slug (already done at runs/${run_name}/metrics.json)"
        echo "$slug skipped" >> "$STATUS_FILE"
        continue
    fi

    echo "[start] $slug on GPU $gpu  -> $log"
    (
        echo "=== $slug on GPU $gpu, started $(date -Iseconds) ==="
        CUDA_VISIBLE_DEVICES="$gpu" python -m src.baseline.replay_with_encoder \
            --source-run "$SOURCE_RUN" \
            --encoder-slug "$slug" \
            --run-name "$run_name"
        rc=$?
        echo "=== $slug exited with code $rc at $(date -Iseconds) ==="
        echo "$slug exit=$rc" >> "$STATUS_FILE"
    ) > "$log" 2>&1 &
    PIDS[$slug]=$!
done

echo ""
echo "all jobs launched. waiting..."
echo ""

# Wait for all background jobs.
for slug in "${!PIDS[@]}"; do
    pid="${PIDS[$slug]}"
    wait "$pid"
    echo "[done ] $slug (pid=$pid)"
done

echo ""
echo "=== all done. status summary: ==="
cat "$STATUS_FILE"
echo ""
echo "Run results in runs/baseline_v1_speechqwen2vl_<slug>/"
echo "Logs in $LOG_DIR/encoder_swap_<slug>.log"
