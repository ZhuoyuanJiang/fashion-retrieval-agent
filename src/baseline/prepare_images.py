"""Pre-fetch reference images for an evaluation slice into the local cache.

Real VLM inference must not depend on surprise network calls mid-run.
This script streams Marqo/fashion200k once, picks out exactly the
candidate images needed for the slice, and saves them to the cache.

Already-cached images are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from src.data.facap_dataset import (
    DEFAULT_IMAGE_CACHE,
    FacapDataset,
    REPO_ROOT,
)


def needed_candidate_ids(category: str, split: str, n_eval: int) -> set[str]:
    """Set of candidate_ids for the last `n_eval` triplets."""
    ds = FacapDataset(category=category, split=split)
    n = len(ds)
    if n_eval > n:
        raise ValueError(f"n_eval {n_eval} > number of triplets {n}")
    return {ds[i]["candidate_id"] for i in range(n - n_eval, n)}


def fetch_images_for(
    needed: set[str], cache: Path, stream_limit: int | None = None
) -> tuple[int, int]:
    """Stream Marqo/fashion200k and save any image whose item_ID is in `needed`.

    Returns (newly_saved, missing_after).
    """
    cache.mkdir(parents=True, exist_ok=True)
    already_cached = {p.stem for p in cache.glob("*.jpeg")}
    todo = needed - already_cached
    if not todo:
        return 0, 0

    print(f"need {len(needed)} images; {len(already_cached & needed)} already cached, "
          f"{len(todo)} to fetch")
    print("streaming Marqo/fashion200k...")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    saved = 0
    for i, ex in enumerate(ds):
        if stream_limit is not None and i >= stream_limit:
            print(f"hit stream_limit {stream_limit}, stopping")
            break
        item_id = ex["item_ID"]
        if item_id in todo:
            ex["image"].save(cache / f"{item_id}.jpeg")
            todo.discard(item_id)
            saved += 1
            if saved % 50 == 0:
                print(f"  saved {saved}/{len(needed)} (still need {len(todo)})")
            if not todo:
                break
    print(f"done: saved {saved}, still missing {len(todo)}")
    return saved, len(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FACap candidate images for eval slice")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-eval", type=int, required=True,
                        help="how many trailing triplets to cover (matches run_baseline --n-eval)")
    parser.add_argument("--cache", default=str(DEFAULT_IMAGE_CACHE))
    parser.add_argument("--stream-limit", type=int, default=None,
                        help="cap on how many HF rows to scan (debug only)")
    args = parser.parse_args()

    needed = needed_candidate_ids(args.category, args.split, args.n_eval)
    saved, missing = fetch_images_for(needed, Path(args.cache), args.stream_limit)
    if missing:
        print(f"WARNING: {missing} candidate image(s) not found in HF stream; "
              f"the corresponding eval queries will fail in the real-VLM run")


if __name__ == "__main__":
    main()
