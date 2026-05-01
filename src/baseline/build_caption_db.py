"""Build the retrieval caption DB from FACap target captions.

Two modes — both share output format (embeddings.npy / metadata.jsonl / config.json):

  full   — encode ALL target captions for (category, split). Stable artifact
           keyed only on (category, split, encoder_name). Build once, reuse
           across every eval run with the same encoder. The production path.

  subset — guaranteed coverage of last `eval_n` triplets' targets + random
           distractors sampled from the rest, totalling `db_size` rows.
           Useful when you want a smaller DB for fast debug iteration
           (smoke tests, plumbing checks). Keyed on the full sampling args
           so reproducible across re-builds with the same seed.

Outputs (under `out_dir`):
    embeddings.npy   float32, shape (N, EMBED_DIM)
    metadata.jsonl   one row per embedding, in the same order
    config.json      provenance: encoder, mode, FACap SHA, build_args
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


def encoder_slug(encoder_name: str) -> str:
    """HuggingFace-style `org/model` -> filesystem-safe `org__model`.

    Used to scope caption DB directories by encoder so multiple encoders'
    DBs can coexist (e.g., MiniLM vs BGE) without overwriting each other.
    """
    return encoder_name.replace("/", "__")


def build_signature_full(
    *,
    category: str,
    split: str,
    encoder_name: str,
) -> dict[str, Any]:
    """Stale-check signature for full-mode DBs."""
    return {
        "mode": "full",
        "category": category,
        "split": split,
        "encoder_name": encoder_name,
    }


def build_signature_subset(
    *,
    category: str,
    split: str,
    encoder_name: str,
    eval_n: int,
    db_size: int,
    seed: int,
) -> dict[str, Any]:
    """Stale-check signature for subset-mode DBs."""
    return {
        "mode": "subset",
        "category": category,
        "split": split,
        "encoder_name": encoder_name,
        "eval_n": eval_n,
        "db_size": db_size,
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


def _write_db(
    out_dir: Path,
    rows: list[dict[str, Any]],
    encoder_name: str,
    category: str,
    split: str,
    extra_config: dict[str, Any],
    build_args: dict[str, Any],
) -> None:
    """Shared writer: encode rows, save embeddings + metadata + config."""
    print(f"encoding {len(rows)} captions with {encoder_name} ...")
    encoder = TextEncoder(model_name=encoder_name)
    embeddings = encoder.encode([r["caption"] for r in rows])
    assert embeddings.shape == (len(rows), EMBED_DIM), embeddings.shape

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    config = {
        "encoder_name": encoder_name,
        "embedding_dim": EMBED_DIM,
        "category": category,
        "split": split,
        "n_total": len(rows),
        "facap_commit_sha": _facap_commit_sha(DEFAULT_FACAP_ROOT),
        "build_args": build_args,
        **extra_config,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"wrote {len(rows)} rows to {out_dir}")
    print(f"  embeddings.npy  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  metadata.jsonl  {len(rows)} rows")


def build_db_full(
    out_dir: Path,
    category: str,
    split: str,
    encoder_name: str,
) -> None:
    """Encode ALL target captions for (category, split). Production path."""
    ds = FacapDataset(category=category, split=split)
    captions: dict[str, str] = ds.captions

    rows: list[dict[str, Any]] = [
        {
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": caption,
            "caption_length_chars": len(caption),
        }
        for path, caption in captions.items()
    ]

    _write_db(
        out_dir=out_dir,
        rows=rows,
        encoder_name=encoder_name,
        category=category,
        split=split,
        extra_config={
            "subset_description": f"full {category}/{split}: all {len(rows)} target captions",
        },
        build_args=build_signature_full(
            category=category, split=split, encoder_name=encoder_name,
        ),
    )


def build_db_subset(
    out_dir: Path,
    category: str,
    split: str,
    encoder_name: str,
    eval_n: int,
    db_size: int,
    seed: int,
) -> None:
    """Subset DB: guaranteed eval targets + randomly sampled distractors.

    Use for fast debug iteration with smaller DBs. Reproducible given (seed).
    """
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
            f"requested {n_distractors} distractors but only {len(distractor_pool)} available; "
            f"reduce db_size or use full mode"
        )

    rng = random.Random(seed)
    distractor_paths = rng.sample(distractor_pool, n_distractors)

    rows: list[dict[str, Any]] = []
    for path in eval_target_paths:
        rows.append({
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": captions[path],
            "caption_length_chars": len(captions[path]),
            "source": "eval_target",
        })
    for path in distractor_paths:
        rows.append({
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": captions[path],
            "caption_length_chars": len(captions[path]),
            "source": "distractor",
        })

    _write_db(
        out_dir=out_dir,
        rows=rows,
        encoder_name=encoder_name,
        category=category,
        split=split,
        extra_config={
            "subset_description": (
                f"subset {category}/{split}: last {eval_n} triplets' targets "
                f"({n_eval_unique} unique) + {n_distractors} random distractors "
                f"(seed={seed})"
            ),
            "n_eval_targets_unique": n_eval_unique,
            "n_distractors": n_distractors,
            "seed": seed,
        },
        build_args=build_signature_subset(
            category=category, split=split, encoder_name=encoder_name,
            eval_n=eval_n, db_size=db_size, seed=seed,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FACap caption retrieval DB")
    parser.add_argument("--out-dir", required=True,
                        help="directory to write embeddings.npy / metadata.jsonl / config.json")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    parser.add_argument("--encoder", default=DEFAULT_MODEL)
    parser.add_argument("--db-size", type=int, default=None,
                        help="if set, build a subset DB of this size (debug mode); "
                             "otherwise build a full DB of all captions (default)")
    parser.add_argument("--eval-n", type=int, default=None,
                        help="subset mode only: number of eval targets guaranteed in DB")
    parser.add_argument("--seed", type=int, default=42,
                        help="subset mode only: seed for distractor sampling")
    args = parser.parse_args()

    if args.db_size is None:
        build_db_full(
            out_dir=Path(args.out_dir),
            category=args.category,
            split=args.split,
            encoder_name=args.encoder,
        )
    else:
        if args.eval_n is None:
            parser.error("--eval-n is required when --db-size is set (subset mode)")
        build_db_subset(
            out_dir=Path(args.out_dir),
            category=args.category,
            split=args.split,
            encoder_name=args.encoder,
            eval_n=args.eval_n,
            db_size=args.db_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
