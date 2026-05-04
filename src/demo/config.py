"""Demo configuration. Read by app.py at startup; tweak via env vars or by editing here.

Stage flag controls how much of the demo is live vs cached:
  - "v0.1": no GPU; results read from PRESET_CACHE_JSON; mocked ASR
  - "v0.2": Pipeline 2 + Whisper run live; Pipeline 1 still cached on presets
  - "v0.3": Pipeline 1 also live (LIVE_PIPELINE_1=True)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE: str = os.environ.get("DEMO_STAGE", "v0.1")

PRESET_CACHE_JSON: Path = REPO_ROOT / "runs" / "demo" / "preset_cache.json"
PRESET_THUMBS_DIR: Path = REPO_ROOT / "runs" / "demo" / "preset_thumbs"
GALLERY_DIR: Path = Path("/ssd1/zhuoyuan/facap-images")

CKPT_DIR: Path = REPO_ROOT / "runs" / "plan5" / "run_bs64_8xA6000_plan6_20260503_011214" / "ckpt_epoch16"
TARGET_CACHE_NPY: Path = REPO_ROOT / "runs" / "plan5" / "target_emb_cache_marqo-fashionclip.npy"
CAPTION_DB_DIR: Path = REPO_ROOT / "runs" / "baseline_v1_speechqwen2vl" / "caption_db"

WHISPER_MODEL: str = "openai/whisper-base"

K_DEFAULT: int = 10
K_MIN: int = 5
K_MAX: int = 50

LIVE_PIPELINE_1: bool = False  # v0.3 flips this to True

GRADIO_SERVER_PORT: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE: bool = os.environ.get("GRADIO_SHARE", "0") == "1"
