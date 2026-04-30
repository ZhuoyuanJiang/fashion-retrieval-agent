"""Build the retrieval caption DB from FACap target captions.

Smoke runs use a small, coverage-guaranteed slice: the eval queries'
target captions are *forced* into the DB along with random distractors.
The full server run later re-uses the same code with the entire FACap
target set.

Outputs (under `runs/<run_name>/caption_db/`):
    embeddings.npy   float32, shape (N, EMBED_DIM)
    metadata.jsonl   one row per embedding, in the same order
    config.json      provenance: encoder, slice description, FACap SHA, seed
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.baseline.text_encoder import DEFAULT_MODEL, EMBED_DIM, TextEncoder
from src.data.facap_dataset import (
    DEFAULT_FACAP_ROOT,
    REPO_ROOT,
    FacapDataset,
    _path_to_image_id,
)


def build_signature(
    *,
    eval_n: int,
    db_size: int,
    category: str,
    split: str,
    encoder_name: str,
    seed: int,
) -> dict[str, Any]:
    """The set of args that, if changed, invalidate an existing caption DB.

    Recorded in `config.json` and re-checked by `run_baseline.ensure_caption_db`
    so a stale DB from a prior run can never be silently reused with new
    args (especially `encoder_name` — a different encoder gives a different
    embedding space, and silent reuse would yield meaningless metrics).
    """
    return {
        "eval_n": eval_n,
        "db_size": db_size,
        "category": category,
        "split": split,
        "encoder_name": encoder_name,
        "seed": seed,
    }


def _facap_commit_sha(facap_root: Path) -> str:
    repo = facap_root.parent  # facap-repo/
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        return sha
    except Exception as e:  # pragma: no cover
        return f"unknown ({e!r})"


def build_db(
    run_name: str,
    eval_n: int,
    db_size: int,
    category: str,
    split: str,
    encoder_name: str,
    seed: int,
    out_root: Path,
) -> None:
    if eval_n > db_size:
        raise ValueError(f"eval_n ({eval_n}) must be <= db_size ({db_size})")

    ds = FacapDataset(category=category, split=split)
    n_triplets = len(ds)
    if eval_n > n_triplets:
        raise ValueError(f"eval_n ({eval_n}) > number of triplets ({n_triplets})")

    eval_target_paths: list[str] = []
    seen: set[str] = set()
    for idx in range(n_triplets - eval_n, n_triplets):
        path = ds[idx]["target_image_path"]
        if path not in seen:
            eval_target_paths.append(path)
            seen.add(path)
    n_eval_unique = len(eval_target_paths)

    captions: dict[str, str] = ds.captions
    distractor_pool = [p for p in captions if p not in seen]
    n_distractors = db_size - n_eval_unique
    if n_distractors > len(distractor_pool):
        raise ValueError(
            f"requested {n_distractors} distractors but only {len(distractor_pool)} available"
        )

    rng = random.Random(seed)
    distractor_paths = rng.sample(distractor_pool, n_distractors)

    rows: list[dict[str, Any]] = []
    for path in eval_target_paths:
        caption = captions[path]
        rows.append({
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": caption,
            "caption_length_chars": len(caption),
            "source": "eval_target",
        })
    for path in distractor_paths:
        caption = captions[path]
        rows.append({
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": caption,
            "caption_length_chars": len(caption),
            "source": "distractor",
        })

    print(f"encoding {len(rows)} captions with {encoder_name} ...")
    encoder = TextEncoder(model_name=encoder_name)
    embeddings = encoder.encode([r["caption"] for r in rows])
    assert embeddings.shape == (len(rows), EMBED_DIM), embeddings.shape

    out_dir = out_root / run_name / "caption_db"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    config = {
        "run_name": run_name,
        "encoder_name": encoder_name,
        "embedding_dim": EMBED_DIM,
        "category": category,
        "split": split,
        "subset_description": (
            f"{category}_{split}: last {eval_n} triplets' targets "
            f"({n_eval_unique} unique) + {n_distractors} random distractors "
            f"(seed={seed})"
        ),
        "n_eval_targets_unique": n_eval_unique,
        "n_distractors": n_distractors,
        "n_total": len(rows),
        "facap_commit_sha": _facap_commit_sha(DEFAULT_FACAP_ROOT),
        "seed": seed,
        # Verbatim build args, used by run_baseline to detect stale-DB reuse.
        "build_args": build_signature(
            eval_n=eval_n, db_size=db_size, category=category,
            split=split, encoder_name=encoder_name, seed=seed,
        ),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"wrote {len(rows)} rows to {out_dir}")
    print(f"  embeddings.npy  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  metadata.jsonl  {len(rows)} rows")
    print(f"  config.json     {config['subset_description']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FACap caption retrieval DB")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-n", type=int, default=50,
                        help="number of eval triplets whose targets must be in the DB")
    parser.add_argument("--db-size", type=int, default=1000,
                        help="total DB size (eval targets + distractors)")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    parser.add_argument("--encoder", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "runs"))
    args = parser.parse_args()

    build_db(
        run_name=args.run_name,
        eval_n=args.eval_n,
        db_size=args.db_size,
        category=args.category,
        split=args.split,
        encoder_name=args.encoder,
        seed=args.seed,
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()
