#!/bin/bash
# =============================================================================
# fashion-retrieval-agent — server setup
# =============================================================================
# Recreates a working `fashion_retrieval` conda env on a fresh server.
# Public-repo-friendly: clone fashion-retrieval-agent + speechQwen2VL
# (sibling) and run this script.
#
# Prerequisites:
#   - conda available
#   - Linux with NVIDIA GPU (>= 14 GB VRAM at bf16 for Qwen2-VL-7B)
#   - speechQwen2VL repo cloned at $SPEECHQWEN2VL_DIR (default: ../speechQwen2VL)
#
# What this script does:
#   1. Checks for the sibling speechQwen2VL repo (clones it if missing,
#      since we depend on its setup_forks.sh as the source of truth for
#      the forked transformers + qwen-vl-utils SHAs)
#   2. Creates the fashion_retrieval conda env from environment.yml
#   3. Installs requirements.txt (the hand-curated superset of
#      speechQwen2VL's pip deps + sentence-transformers + requests)
#   4. Runs speechQwen2VL/scripts/setup_forks.sh which installs the
#      forked transformers + qwen-vl-utils into the active env LAST,
#      overriding the upstream transformers
#   5. Verifies the install
#
# Usage:
#   bash scripts/setup_server.sh
#
# Environment variables (optional):
#   SPEECHQWEN2VL_DIR    Path to speechQwen2VL clone (default: ../speechQwen2VL)
#   SPEECHQWEN2VL_REPO   GitHub URL to clone if missing
#                         (default: https://github.com/ZhuoyuanJiang/speechQwen2VL.git)
#   ENV_NAME             conda env name (default: fashion_retrieval)
# =============================================================================

set -e  # exit on any error

# ---------- config ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-fashion_retrieval}"
SPEECHQWEN2VL_DIR="${SPEECHQWEN2VL_DIR:-$(cd "$REPO_ROOT/.." && pwd)/speechQwen2VL}"
SPEECHQWEN2VL_REPO="${SPEECHQWEN2VL_REPO:-https://github.com/ZhuoyuanJiang/speechQwen2VL.git}"

echo "=== fashion-retrieval-agent server setup ==="
echo "  repo root          : $REPO_ROOT"
echo "  env name           : $ENV_NAME"
echo "  speechQwen2VL dir  : $SPEECHQWEN2VL_DIR"
echo ""

# ---------- step 1: ensure speechQwen2VL is cloned alongside ----------
if [ ! -d "$SPEECHQWEN2VL_DIR" ]; then
    echo "=== Step 1/5: cloning speechQwen2VL alongside ==="
    git clone "$SPEECHQWEN2VL_REPO" "$SPEECHQWEN2VL_DIR"
else
    echo "=== Step 1/5: speechQwen2VL already cloned at $SPEECHQWEN2VL_DIR ==="
fi

if [ ! -f "$SPEECHQWEN2VL_DIR/scripts/setup_forks.sh" ]; then
    echo "ERROR: $SPEECHQWEN2VL_DIR/scripts/setup_forks.sh not found."
    echo "       Either the clone failed or speechQwen2VL's repo layout changed."
    exit 1
fi

# ---------- step 2: create conda env ----------
echo ""
echo "=== Step 2/5: creating conda env '$ENV_NAME' ==="
# Initialize conda for this shell so `conda activate` works.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "env '$ENV_NAME' already exists; skipping creation"
else
    conda env create -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml"
fi
conda activate "$ENV_NAME"
echo "active env: $(python -c 'import sys; print(sys.prefix)')"

# ---------- step 3: install requirements.txt (pip layer) ----------
echo ""
echo "=== Step 3/5: pip install -r requirements.txt ==="
pip install -r "$REPO_ROOT/requirements.txt"

# ---------- step 4: install forks LAST (override upstream transformers) ----------
echo ""
echo "=== Step 4/5: installing forked transformers + qwen-vl-utils ==="
echo "    (delegating to speechQwen2VL's setup_forks.sh)"
bash "$SPEECHQWEN2VL_DIR/scripts/setup_forks.sh"

# ---------- step 5: verify ----------
echo ""
echo "=== Step 5/5: verifying install ==="
python - <<'PY'
import sys
print(f"python: {sys.version.split()[0]}")
import torch
print(f"torch:  {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"        device 0: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
import transformers, peft, sentence_transformers
print(f"transformers: {transformers.__version__}  ({transformers.__file__})")
print(f"peft:         {peft.__version__}")
print(f"sentence_transformers: {sentence_transformers.__version__}")
try:
    import qwen_vl_utils
    print(f"qwen_vl_utils: ok ({qwen_vl_utils.__file__})")
except ImportError as e:
    print(f"qwen_vl_utils: MISSING ({e})")
PY

echo ""
echo "=== Setup complete ==="
echo "Activate the env in your shell with:"
echo "    conda activate $ENV_NAME"
echo "Then run scripts/run_baseline_v1.sh to execute the baseline."
