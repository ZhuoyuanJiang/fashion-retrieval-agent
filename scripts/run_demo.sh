#!/bin/bash
# =============================================================================
# fashion-retrieval-agent — Plan_8 demo app launcher
# =============================================================================
# Launches the Gradio demo (three-pipeline retrieval comparison).
#
# Usage:
#   bash scripts/run_demo.sh                        # v0.1 default (no GPU needed)
#   DEMO_STAGE=v0.2 bash scripts/run_demo.sh        # v0.2 (Pipeline 2 + Whisper live; needs ≥24 GB GPU)
#   DEMO_STAGE=v0.3 bash scripts/run_demo.sh        # v0.3 (Pipeline 1 also live; needs ≥49 GB GPU)
#   GRADIO_SHARE=1 bash scripts/run_demo.sh         # also expose a *.gradio.live public URL (72h)
#
# Stages:
#   v0.1: scripted demo. UI + preset clicks + cached results from JSON. No model
#         loaded at request time. Runs on a personal laptop without a GPU.
#   v0.2: Pipeline 2 + Whisper run live. Pipeline 1 still cached on presets.
#         Requires ≥24 GB GPU and the gallery cache + Plan-6 checkpoint on disk.
#   v0.3: Pipeline 1 also live. Requires ≥49 GB GPU (A6000-class).
#
# Notes:
# - Gallery image path /ssd1/zhuoyuan/facap-images/ is host-specific. v0.1 uses
#   only the bundled preset thumbs and works anywhere; v0.2+ needs the full
#   gallery on the host or a mounted equivalent.
# - HF_HOME / TRANSFORMERS_CACHE must point to a local SSD with >50 GB free
#   for v0.2+ to avoid downloading models to NAS home.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DEMO_STAGE="${DEMO_STAGE:-v0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"

# Per-user gradio scratch dir — /tmp/gradio is shared across lab users and
# tends to be locked down by whoever ran first. Override to a private location.
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-$HOME/.cache/gradio_tmp_$USER}"
mkdir -p "$GRADIO_TEMP_DIR"

# v0.2+ uses HF cache. Default to a local SSD path; honour an explicit override.
if [[ "$DEMO_STAGE" != "v0.1" ]]; then
    export HF_HOME="${HF_HOME:-/ssd1/zhuoyuan/hf_cache}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
fi

PYTHON_BIN="${PYTHON_BIN:-/home/zhuoyuan/miniconda3/envs/fashion_retrieval/bin/python}"

echo "[run_demo] stage=$DEMO_STAGE  port=$GRADIO_SERVER_PORT  share=$GRADIO_SHARE"
echo "[run_demo] repo=$REPO_ROOT"
echo "[run_demo] python=$PYTHON_BIN"

exec "$PYTHON_BIN" -m src.demo.app
