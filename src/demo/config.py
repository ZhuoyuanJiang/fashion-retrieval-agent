"""Demo configuration. Read by app.py at startup; tweak via env vars or by editing here.

Stage flag controls how much of the demo is live vs cached:
  - "v0.1": no GPU; results read from PRESET_CACHE_JSON; mocked ASR
  - "v0.2": Pipeline 2 + Whisper run live; Pipeline 1 still cached on presets
  - "v0.3": Pipeline 1 also live (LIVE_PIPELINE_1=True)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE: str = os.environ.get("DEMO_STAGE", "v0.1")

# --- Demo asset paths ---
# Prefer demo_assets/ (tracked in git, present after a fresh clone). Fall
# back to runs/demo/ — the original dev-server location, kept for the
# original developer's setup.
_DEMO_REPO = REPO_ROOT / "demo_assets"
_DEMO_LEGACY = REPO_ROOT / "runs" / "demo"


def _demo_path(rel: str) -> Path:
    primary = _DEMO_REPO / rel
    if primary.exists():
        return primary
    legacy = _DEMO_LEGACY / rel
    if legacy.exists():
        return legacy
    print(
        f"[demo config] WARNING: '{rel}' not in demo_assets/ or runs/demo/. "
        f"For preset_thumbs run `python scripts/make_demo_thumbs.py`; "
        f"for cache/audio re-run `bash scripts/fetch_artifacts.sh`.",
        file=sys.stderr,
    )
    return primary


PRESET_CACHE_JSON: Path = _demo_path("preset_cache.json")
PRESET_THUMBS_DIR: Path = _demo_path("preset_thumbs")
# Live-mode paths (only used when the demo retrieves over the full catalog or
# loads the audio two-tower at request time). Defaults are the original
# dev-server's paths; on any other machine, override via env vars.
GALLERY_DIR: Path = Path(os.environ.get("GALLERY_DIR", "/ssd1/zhuoyuan/facap-images"))

CKPT_DIR: Path = REPO_ROOT / "runs" / "plan5" / "run_bs64_8xA6000_plan6_20260503_011214" / "ckpt_epoch16"
TARGET_CACHE_NPY: Path = REPO_ROOT / "runs" / "plan5" / "target_emb_cache_marqo-fashionclip.npy"
CAPTION_DB_DIR: Path = REPO_ROOT / "runs" / "baseline_v1_speechqwen2vl" / "caption_db"

WHISPER_MODEL: str = "openai/whisper-base"

K_DEFAULT: int = 10
K_MIN: int = 5
K_MAX: int = 50

LIVE_PIPELINE_1: bool = False  # v0.3 flips this to True

# ----- Live audio two-tower (Plan-15) -----
# The bottom "live" row records the user's own speech and runs the Plan-15
# audio query tower against the full gallery. Gated by LIVE_AUDIO=1 so the
# cached-only demo still launches on a CPU laptop with no checkpoints.
LIVE_AUDIO: bool = os.environ.get("LIVE_AUDIO", "0") == "1"
LIVE_AUDIO_DEVICE: str = os.environ.get("LIVE_AUDIO_DEVICE", "cuda:0")
AUDIO_2T_CKPT: Path = Path(os.environ.get(
    "AUDIO_2T_CKPT", "/ssd1/zhuoyuan/plan15_demo/ckpt_epoch17"))
AUDIO_2T_GALLERY: Path = Path(os.environ.get(
    "AUDIO_2T_GALLERY", "/ssd1/zhuoyuan/plan15_demo/gallery_emb_epoch17.npy"))
# TTS clip per preset — the spoken modification the audio rows (P4 / live) hear.
PRESET_AUDIO_DIR: Path = _demo_path("preset_audio")

GRADIO_SERVER_PORT: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE: bool = os.environ.get("GRADIO_SHARE", "0") == "1"
