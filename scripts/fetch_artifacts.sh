#!/usr/bin/env bash
# scripts/fetch_artifacts.sh — pull models + data so the demo can run.
#
# Two modes:
#   bash scripts/fetch_artifacts.sh                 # public  (default)  — HF Hub + dataset clones
#   bash scripts/fetch_artifacts.sh --from-drive    # owner-only: full restore from Google Drive backup
#
# Optional flag (combine with public mode):
#   --with-images   Also download the FACap dress catalog images
#                   (~3.5 GB, ~59K JPEGs) from the public Marqo/fashion200k
#                   HF dataset mirror. Needed for the demo to display
#                   product thumbnails. After this, run
#                   `python scripts/make_demo_thumbs.py` to generate the
#                   1085 thumbnails the demo actually uses.
#
# Idempotent — safe to re-run.
#
# Demo image data (the 59K-vs-1085 note):
#   The cached demo only displays ~1085 specific thumbnails, but the
#   upstream image mirror doesn't support "pull only these IDs" — so the
#   --with-images flag pulls the full ~59K Fashion200k catalog, then
#   scripts/make_demo_thumbs.py resizes the 1085 you actually need into
#   demo_assets/preset_thumbs/.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse args: --from-drive is the mode; --with-images is an optional flag.
MODE="public"
WITH_IMAGES=0
for arg in "$@"; do
  case "$arg" in
    --from-drive)  MODE="--from-drive" ;;
    --with-images) WITH_IMAGES=1 ;;
    public)        MODE="public" ;;
    *)             printf '\033[1;31m[fetch]\033[0m Unknown arg: %s\n' "$arg" >&2; exit 1 ;;
  esac
done

log()  { printf "\033[1;36m[fetch]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[fetch]\033[0m %s\n" "$*" >&2; exit 1; }

# The example 'runs/' symlink points to an SSD path that won't exist on a
# fresh clone — replace it with a real directory so downloads land somewhere.
[[ -L runs && ! -e runs ]] && { log "Replacing dangling runs/ symlink with a real dir."; rm runs; }
mkdir -p runs

# ===================================================================== public
if [[ "$MODE" != "--from-drive" ]]; then
  log "mode = public  (HuggingFace Hub + public dataset helpers)"

  log "[1/3] Trained models -> runs/hf_models/  (also caches the base model in \$HF_HOME)"
  python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
dst = Path("runs/hf_models/audio-composed-fashion-item-retriever")
dst.parent.mkdir(parents=True, exist_ok=True)
snapshot_download("DanJZY/audio-composed-fashion-item-retriever", local_dir=str(dst))
snapshot_download("DanJZY/Qwen2-VL-7B-Speech")          # frozen base (cached in HF_HOME)
snapshot_download("DanJZY/Qwen2-VL-7B-Speech-LoRA")     # frozen Stage-2 LoRA
print(f"  models ready: {dst}")
PY

  log "[2/3] Third-party dataset annotation repos -> data_exploration/datasets/"
  bash scripts/setup_datasets.sh

  if [[ "$WITH_IMAGES" == "1" ]]; then
    FACAP_LOCAL="${FACAP_LOCAL:-$REPO_ROOT/_artifacts}"
    mkdir -p "$FACAP_LOCAL/facap-images"
    log "[3/4] --with-images : Fashion200k from public HF mirror -> $FACAP_LOCAL/facap-images/  (~3.5 GB)"
    FACAP_LOCAL="$FACAP_LOCAL" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
target = Path(os.environ["FACAP_LOCAL"]) / "facap-images"
snapshot_download(
    repo_id="Marqo/fashion200k",
    repo_type="dataset",
    local_dir=str(target),
)
print(f"  -> {target}")
PY
  fi

  STEP=$([[ "$WITH_IMAGES" == "1" ]] && echo "4/4" || echo "3/3")
  log "[$STEP] Done — what's still needed before the demo will run:"
  cat <<EOF

  Trained models   : runs/hf_models/audio-composed-fashion-item-retriever/   (audio/ + text/)
  Dataset repos    : data_exploration/datasets/                              (FACap triplets, etc.)
EOF
  if [[ "$WITH_IMAGES" == "1" ]]; then
    cat <<EOF
  FACap images     : ${FACAP_LOCAL}/facap-images/  (just pulled, ~3.5 GB)
  Demo thumbnails  : run \`python scripts/make_demo_thumbs.py --images-dir ${FACAP_LOCAL}/facap-images\`
                     to generate demo_assets/preset_thumbs/.
EOF
  else
    cat <<EOF
  FACap images     : NOT YET — re-run with --with-images to pull from the
                                public Marqo/fashion200k HF mirror (~3.5 GB).
  Demo thumbnails  : after images are local, run scripts/make_demo_thumbs.py
                     to generate demo_assets/preset_thumbs/.
EOF
  fi
  cat <<EOF
  Demo preset cache: owner-only (lives in the private Google Drive backup).

  Wire the audio checkpoint to where src/demo/config.py expects it (or set
  AUDIO_2T_CKPT env var to override the default in src/demo/config.py):

    ln -s "$PWD/runs/hf_models/audio-composed-fashion-item-retriever/audio" \\
          \$AUDIO_2T_CKPT
EOF
  exit 0
fi

# ======================================================== --from-drive (owner)
log "mode = --from-drive  (rclone restore from Google Drive backup)"
if [[ "$WITH_IMAGES" == "1" ]]; then
  log "Note: --with-images is ignored in --from-drive mode (the backup tar"
  log "      already contains the FACap images via facap-images.tar)."
fi

RCLONE="${RCLONE:-$HOME/bin/rclone}"
[[ -x "$RCLONE" ]] || RCLONE="$(command -v rclone)" || die "rclone not installed (see https://rclone.org/install/)."
SRC="${BACKUP_SRC:-vllab13:Backups/fashion-retrieval-agent_project_backup_20260525}"
"$RCLONE" lsd "$SRC" >/dev/null 2>&1 || die "Cannot access $SRC — configure the Google Drive rclone remote first."

# Local landing paths — where each artifact bucket gets restored.
# Defaults below put everything under $REPO_ROOT/_artifacts/  — works on any
# fresh machine, no /ssd1/ required. The "dev-machine example" column below
# shows the layout this project happened to use on its original training
# server (it had /ssd1, /tmp3 local SSDs); export those env vars to reproduce
# that layout.
#
#   RUNS_LOCAL   — checkpoints + caption_db + baselines + demo preset cache
#                   default             : $REPO_ROOT/_artifacts/runs
#                   dev-machine example : /ssd1/zhuoyuan/fashion-retrieval-agent_runs
#
#   AUDIO_LOCAL  — parent dir for the untarred plan14_audio/  (TTS audio data)
#                   default             : $REPO_ROOT/_artifacts   → .../plan14_audio/
#                   dev-machine example : /tmp3/zhuoyuan          → /tmp3/zhuoyuan/plan14_audio/
#
#   FACAP_LOCAL  — parent dir for the untarred facap-images/  (~59K product images)
#                   default             : $REPO_ROOT/_artifacts   → .../facap-images/
#                   dev-machine example : /ssd1/zhuoyuan          → /ssd1/zhuoyuan/facap-images/
#
#   SCRATCH      — temp dir for tarballs in-flight (auto-cleaned)
#                   default             : /tmp/_fetch_artifacts
#
# Dev-machine example one-liner (reproduce the original training-server layout):
#
#   RUNS_LOCAL=/ssd1/zhuoyuan/fashion-retrieval-agent_runs \
#   AUDIO_LOCAL=/tmp3/zhuoyuan FACAP_LOCAL=/ssd1/zhuoyuan \
#       bash scripts/fetch_artifacts.sh --from-drive
RUNS_LOCAL="${RUNS_LOCAL:-$REPO_ROOT/_artifacts/runs}"
AUDIO_LOCAL="${AUDIO_LOCAL:-$REPO_ROOT/_artifacts}"
FACAP_LOCAL="${FACAP_LOCAL:-$REPO_ROOT/_artifacts}"
SCRATCH="${SCRATCH:-/tmp/_fetch_artifacts}"
mkdir -p "$RUNS_LOCAL" "$AUDIO_LOCAL" "$FACAP_LOCAL" "$SCRATCH"

untar_from_drive() {
  local rel="$1" target="$2"
  log "  unpacking $rel -> $target"
  "$RCLONE" copy "$SRC/$rel" "$SCRATCH" --transfers=8 --stats=10s
  tar xf "$SCRATCH/$(basename "$rel")" -C "$target"
  rm "$SCRATCH/$(basename "$rel")"
}

log "[1/6] Checkpoints (6 models) -> $RUNS_LOCAL/"
"$RCLONE" copy "$SRC/checkpoints" "$RUNS_LOCAL" --transfers=8 --stats=10s

log "[2/6] TTS audio dataset"
untar_from_drive data/plan14_audio.tar "$AUDIO_LOCAL"

log "[3/6] FACap images"
untar_from_drive data/facap-images.tar "$FACAP_LOCAL"

log "[4/6] caption_db"
untar_from_drive caption_db.tar "$RUNS_LOCAL"

log "[5/6] baselines + demo preset cache"
untar_from_drive baselines.tar "$RUNS_LOCAL"
"$RCLONE" copy "$SRC/demo" "$RUNS_LOCAL/demo" --transfers=8 --stats=10s

log "[6/6] FACap triplet annotations"
"$RCLONE" copy "$SRC/data/datasets" "$REPO_ROOT/data_exploration/datasets" --transfers=8 --stats=10s

# Restore the example-style runs/ symlink so demo config.py paths resolve.
if [[ ! -e "$REPO_ROOT/runs" || -L "$REPO_ROOT/runs" ]]; then
  [[ -L "$REPO_ROOT/runs" ]] && rm "$REPO_ROOT/runs"
  ln -s "$RUNS_LOCAL" "$REPO_ROOT/runs"
  log "Recreated runs/ -> $RUNS_LOCAL  (replaces the dangling symlink from a fresh clone)."
fi

# Audio-checkpoint path expected by src/demo/config.py (AUDIO_2T_CKPT).
# Only set up the example /ssd1 path if /ssd1/zhuoyuan already exists —
# on a fresh server we don't want to silently create new top-level dirs.
if [[ -d /ssd1/zhuoyuan && ! -e /ssd1/zhuoyuan/plan15_demo/ckpt_epoch17 ]]; then
  mkdir -p /ssd1/zhuoyuan/plan15_demo
  ln -s "$RUNS_LOCAL/audio_plan15_bs32/ckpt_epoch17" /ssd1/zhuoyuan/plan15_demo/ckpt_epoch17 2>/dev/null || true
  log "Symlinked demo audio ckpt -> $RUNS_LOCAL/audio_plan15_bs32/ckpt_epoch17"
elif [[ ! -d /ssd1/zhuoyuan ]]; then
  log "Note: /ssd1/zhuoyuan does not exist on this machine — src/demo/config.py"
  log "      currently hardcodes /ssd1/zhuoyuan/plan15_demo/ckpt_epoch17 as the"
  log "      demo audio-ckpt path. Either edit AUDIO_2T_CKPT in that config, or"
  log "      symlink it: ln -s $RUNS_LOCAL/audio_plan15_bs32/ckpt_epoch17 <wherever>"
fi

rmdir "$SCRATCH" 2>/dev/null || true
log "DONE (--from-drive mode). Full local state restored — demo should run."
