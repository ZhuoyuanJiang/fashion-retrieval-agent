"""Plan_2 M1 — FacapDataset returns the expected per-triplet dict
and `load_image()` resolves to a cached image on disk.

Run as a script:
    conda activate fashion_retrieval
    python -m tests.test_m1_facap_dataset

Or via pytest:
    pytest tests/test_m1_facap_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from src.data.facap_dataset import FacapDataset  # noqa: E402
from tests._runner import cli_main  # noqa: E402

EXPECTED_KEYS = {
    "candidate_image_path",
    "modification_text",
    "target_image_path",
    "target_caption",
    "target_id",
    "candidate_id",
}


def test_dataset_constructs_with_dress_train() -> None:
    """Constructor reads triplets + captions JSON without error."""
    ds = FacapDataset(category="dress", split="train")
    # 59,082 is the published count; we don't pin the exact number so a
    # FACap re-release with extra triplets doesn't trip the test.
    assert len(ds) > 50_000, f"expected >50k triplets, got {len(ds)}"


def test_item_has_six_expected_keys() -> None:
    """Each item is a 6-key dict matching the documented schema."""
    ds = FacapDataset(category="dress", split="train")
    item = ds[0]
    missing = EXPECTED_KEYS - set(item.keys())
    extra = set(item.keys()) - EXPECTED_KEYS
    assert not missing, f"missing keys: {missing}"
    assert not extra, f"unexpected keys: {extra}"


def test_item_field_types() -> None:
    """All six fields are strings (paths, ids, text)."""
    ds = FacapDataset(category="dress", split="train")
    item = ds[0]
    for k in EXPECTED_KEYS:
        assert isinstance(item[k], str), f"{k} is {type(item[k]).__name__}, expected str"
        assert item[k], f"{k} is empty"


def test_target_id_matches_target_path_filename() -> None:
    """target_id should be the target image filename stem (no .jpeg)."""
    ds = FacapDataset(category="dress", split="train")
    item = ds[0]
    expected = item["target_image_path"].rsplit("/", 1)[-1].removesuffix(".jpeg")
    assert item["target_id"] == expected, (
        f"target_id={item['target_id']!r} != filename stem {expected!r}"
    )


def test_load_image_opens_at_least_one_cached_image() -> None:
    """`load_image()` must successfully open at least one cached image.

    M1's exit criterion required this — if the path resolution is wrong
    (cache layout, image_id derivation), every triplet would raise.
    """
    ds = FacapDataset(category="dress", split="train")
    last_err: Exception | None = None
    for i in range(min(len(ds), 200)):  # bounded scan to keep test fast
        item = ds[i]
        try:
            img = ds.load_image(item, "target")
            assert isinstance(img, Image.Image)
            assert img.size[0] > 0 and img.size[1] > 0
            return
        except FileNotFoundError as e:
            last_err = e
            continue
    raise AssertionError(
        f"no cached target image found in the first 200 triplets; "
        f"last error: {last_err}. Either the cache is empty (run "
        f"src/baseline/prepare_images.py) or load_image() can't resolve paths."
    )


TESTS = [
    test_dataset_constructs_with_dress_train,
    test_item_has_six_expected_keys,
    test_item_field_types,
    test_target_id_matches_target_path_filename,
    test_load_image_opens_at_least_one_cached_image,
]


if __name__ == "__main__":
    cli_main(TESTS, "M1 (FacapDataset)")
