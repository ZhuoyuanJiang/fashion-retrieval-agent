"""FACap dress-triplet dataset for the text-modification retrieval baseline.

Returns dicts with image *paths* (not loaded PIL objects); call
`FacapDataset.load_image(item, side)` to actually open one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACAP_ROOT = REPO_ROOT / "data_exploration" / "datasets" / "facap-repo" / "data" / "facap"
DEFAULT_IMAGE_CACHE = REPO_ROOT / "data_exploration" / "datasets" / "facap-images"


def _path_to_image_id(facap_path: str) -> str:
    """`f200k_images/.../51727804_0.jpeg` -> `51727804_0`."""
    return facap_path.rsplit("/", 1)[-1].removesuffix(".jpeg")


class FacapDataset(Dataset):
    """FACap CIR triplets for one category/split.

    Each item is a dict:
        candidate_image_path: str   FACap-relative path of the reference image
        modification_text:    str   the modification instruction
        target_image_path:    str   FACap-relative path of the target image
        target_caption:       str   FACap pre-computed caption of the target image
        target_id:            str   image id (filename stem) for the target
        candidate_id:         str   image id for the candidate

    The text-only baseline only needs `modification_text`, `target_caption`,
    and `target_id` in the hot path. The path fields are there so callers can
    open a few images for sanity checks via `load_image(item, side)`.
    """

    def __init__(
        self,
        category: str = "dress",
        split: str = "train",
        facap_root: Path | str = DEFAULT_FACAP_ROOT,
        image_cache: Path | str = DEFAULT_IMAGE_CACHE,
    ) -> None:
        self.category = category
        self.split = split
        self.facap_root = Path(facap_root)
        self.image_cache = Path(image_cache)

        triplets_path = self.facap_root / "cir_triplets" / f"{category}_{split}_triplets.json"
        captions_path = self.facap_root / "image_captions" / f"{category}_{split}_captions.json"

        with open(triplets_path) as f:
            self._triplets: list[dict[str, Any]] = json.load(f)
        with open(captions_path) as f:
            self._captions: dict[str, str] = json.load(f)

    @property
    def captions(self) -> dict[str, str]:
        """Map FACap-relative image path -> pre-computed long caption."""
        return self._captions

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self._triplets[idx]
        cand_path = t["candidate"]
        tgt_path = t["target"]
        return {
            "candidate_image_path": cand_path,
            "modification_text": t["captions"][0],
            "target_image_path": tgt_path,
            "target_caption": self._captions[tgt_path],
            "target_id": _path_to_image_id(tgt_path),
            "candidate_id": _path_to_image_id(cand_path),
        }

    def load_image(self, item: dict[str, Any], side: Literal["candidate", "target"]) -> Image.Image:
        """Open the candidate or target image for a dataset item.

        Looks in the local image cache only — fetching missing images from
        the Marqo HF mirror is the job of `src/baseline/prepare_images.py`,
        kept here as a clean local read.
        """
        if side not in ("candidate", "target"):
            raise ValueError(f"side must be 'candidate' or 'target', got {side!r}")
        image_id = item[f"{side}_id"]
        local = self.image_cache / f"{image_id}.jpeg"
        if not local.exists():
            raise FileNotFoundError(
                f"image {image_id} not in cache ({local}); "
                f"run src/baseline/prepare_images.py to fetch it"
            )
        return Image.open(local)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FacapDataset")
    parser.add_argument("--print", type=int, default=20, help="number of items to print")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = FacapDataset(category=args.category, split=args.split)
    print(f"FacapDataset({args.category}/{args.split}): {len(ds)} triplets")
    print(f"image cache: {ds.image_cache}")
    print()

    for i in range(min(args.print, len(ds))):
        item = ds[i]
        print(f"[{i}] target_id={item['target_id']}  cand_id={item['candidate_id']}")
        print(f"    mod: {item['modification_text'][:90]}")
        print(f"    tgt_caption: {item['target_caption'][:90]}...")

    # Find the first item whose target image is in the local cache and open it.
    print("\nResolving one image via load_image()...")
    opened = False
    for i in range(len(ds)):
        item = ds[i]
        try:
            img = ds.load_image(item, "target")
        except FileNotFoundError:
            continue
        print(f"  opened target_id={item['target_id']}  size={img.size}  mode={img.mode}")
        opened = True
        break
    if not opened:
        print("  WARNING: no cached image found; "
              "run src/baseline/prepare_images.py to populate the cache")


if __name__ == "__main__":
    _main()
