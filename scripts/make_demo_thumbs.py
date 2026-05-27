#!/usr/bin/env python3
"""Generate demo thumbnails from the user's local FACap images.

What this script does:
  1. Read demo_assets/preset_cache.json and pull every item ID the demo will
     need to display (~1085 IDs across 8 presets × 4 pipelines + references).
  2. For each ID, find the matching FACap image in the local images directory.
  3. PIL-resize down to ~300 px on the longer side (keep aspect ratio).
  4. Save the thumbnail to demo_assets/preset_thumbs/<id>.jpeg.

Usage:
  python scripts/make_demo_thumbs.py
  python scripts/make_demo_thumbs.py --images-dir /path/to/facap-images
  python scripts/make_demo_thumbs.py --size 400          # bigger thumbs

Prerequisites:
  - FACap images present locally — see scripts/fetch_artifacts.sh --with-images
    (which wraps facap-repo's downloader) or run that downloader directly.
  - Pillow installed (`pip install pillow`).

Why pull 59K just to thumbnail ~1085 of them?
  The cached demo only displays ~1085 specific items (8 presets × 4 pipelines
  × top-K + references), but the FACap upstream downloader doesn't support
  "pull only these IDs". So the practical workflow is: pull the full ~59K
  FACap dress catalog (via fetch_artifacts.sh --with-images, or facap-repo's
  own downloader at data_exploration/datasets/facap-repo/src/run/), then this
  script extracts and resizes the 1085 you actually need.

We do **not** redistribute the FACap source images (license unclear upstream;
images come from Fashion200k / DeepFashion / FashionIQ). Instead, each user
generates demo thumbnails on their own machine from images they already
have — purely local PIL resize, no redistribution involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_CACHE = REPO_ROOT / "demo_assets" / "preset_cache.json"
DEFAULT_THUMBS_OUT = REPO_ROOT / "demo_assets" / "preset_thumbs"
# Example layout from the original training server; override via --images-dir
# on a fresh machine.
DEFAULT_FACAP_IMAGES = Path("/ssd1/zhuoyuan/facap-images")


def collect_item_ids(preset_cache_path: Path) -> set[str]:
    """Pull every item ID the demo will need to display from preset_cache.json."""
    cache = json.loads(preset_cache_path.read_text())
    ids: set[str] = set()
    for preset in cache["presets"]:
        ids.add(preset["candidate_image_id"])
        ids.add(preset["true_target_id"])
        for pipeline_key in ("p1", "p2", "text2t", "audio2t"):
            ids.update(preset[pipeline_key]["target_ids"])
    return ids


def find_image(item_id: str, images_dir: Path) -> Path | None:
    """Locate the FACap image for an item ID; return None if missing."""
    direct = images_dir / f"{item_id}.jpeg"
    if direct.exists():
        return direct
    # FACap images may be nested by category subdirs — fall back to rglob.
    for match in images_dir.rglob(f"{item_id}.jpeg"):
        return match
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate demo thumbnails from local FACap images.")
    p.add_argument("--preset-cache", type=Path, default=DEFAULT_PRESET_CACHE,
                   help=f"default: {DEFAULT_PRESET_CACHE}")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_FACAP_IMAGES,
                   help=f"default: {DEFAULT_FACAP_IMAGES} (override on a "
                        f"fresh machine)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_THUMBS_OUT,
                   help=f"default: {DEFAULT_THUMBS_OUT}")
    p.add_argument("--size", type=int, default=300,
                   help="Max width/height in pixels (default: 300)")
    args = p.parse_args()

    if not args.preset_cache.exists():
        sys.exit(f"ERROR: {args.preset_cache} not found.")
    if not args.images_dir.exists():
        sys.exit(
            f"ERROR: FACap images dir {args.images_dir} not found.\n"
            f"Either pass --images-dir <path>, or fetch the FACap images "
            f"first via:\n"
            f"  bash scripts/fetch_artifacts.sh --with-images")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = collect_item_ids(args.preset_cache)
    print(f"[thumbs] {len(ids)} unique item IDs to thumbnail")

    made, already, missing = 0, 0, []
    size = (args.size, args.size)
    for item_id in sorted(ids):
        out_path = args.out_dir / f"{item_id}.jpeg"
        if out_path.exists():
            already += 1
            continue
        src = find_image(item_id, args.images_dir)
        if src is None:
            missing.append(item_id)
            continue
        try:
            img = Image.open(src).convert("RGB")
            img.thumbnail(size)
            img.save(out_path, "JPEG", quality=85)
            made += 1
        except Exception as e:
            print(f"[thumbs]   FAILED {item_id}: {e}")
            missing.append(item_id)

    print(f"[thumbs] DONE — made: {made}, already there: {already}, "
          f"missing: {len(missing)}")
    if missing:
        print(f"[thumbs] Missing IDs (first 5): {missing[:5]}")
        print(f"[thumbs] If these are FACap items, ensure --images-dir points "
              f"at the full set fetched via scripts/fetch_artifacts.sh "
              f"--with-images.")


if __name__ == "__main__":
    main()
