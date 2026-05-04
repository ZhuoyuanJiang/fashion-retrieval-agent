"""Resolve gallery target_id to image path on disk.

In v0.1 we serve preset thumbnails from runs/demo/preset_thumbs/ so the demo
runs on a laptop without /ssd1 mounted. In v0.2+ we fall back to the full
gallery on the lab GPU host (/ssd1/zhuoyuan/facap-images).
"""
from __future__ import annotations

from pathlib import Path

from . import config


def image_path(target_id: str) -> Path:
    """Return the on-disk path for a target_id, preferring the local thumb cache.

    v0.1 ships ~80 preset thumbnails (8 presets x ~10 top-K each) under
    runs/demo/preset_thumbs/. If a thumb is present we use it. Otherwise we
    point at /ssd1/.../<id>.jpeg, which only exists on the GPU host.
    """
    thumb = config.PRESET_THUMBS_DIR / f"{target_id}.jpeg"
    if thumb.exists():
        return thumb
    return config.GALLERY_DIR / f"{target_id}.jpeg"


def thumb_or_placeholder(target_id: str) -> Path | None:
    """Like image_path but returns None if neither thumb nor /ssd1 image exists.

    Used by the UI to decide whether to render the thumbnail or a placeholder
    box ("image not available on this host").
    """
    p = image_path(target_id)
    return p if p.exists() else None
