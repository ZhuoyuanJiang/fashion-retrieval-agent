"""Cosine-similarity retrieval over a pre-built caption DB.

The caption DB stores L2-normalized embeddings (see TextEncoder), so
cosine similarity == dot product == one matmul.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CaptionDB:
    embeddings: np.ndarray            # (N, D), float32, L2-normalized
    target_ids: list[str]             # length N
    image_paths: list[str]            # length N
    config: dict                      # provenance from build_caption_db

    @classmethod
    def load(cls, db_dir: Path | str) -> "CaptionDB":
        db_dir = Path(db_dir)
        emb = np.load(db_dir / "embeddings.npy")
        ids: list[str] = []
        paths: list[str] = []
        with open(db_dir / "metadata.jsonl") as f:
            for line in f:
                row = json.loads(line)
                ids.append(row["target_id"])
                paths.append(row["image_path"])
        with open(db_dir / "config.json") as f:
            cfg = json.load(f)
        if emb.shape[0] != len(ids):
            raise ValueError(f"embeddings rows ({emb.shape[0]}) != metadata rows ({len(ids)})")
        return cls(embeddings=emb, target_ids=ids, image_paths=paths, config=cfg)


def top_k(query_emb: np.ndarray, db: CaptionDB, k: int) -> list[tuple[str, float]]:
    """One query embedding (D,) or (1, D) -> [(target_id, score)] sorted desc."""
    q = query_emb.reshape(-1)
    scores = db.embeddings @ q  # (N,)
    k = min(k, scores.shape[0])
    # argpartition for the top-k, then sort just those by score desc.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(db.target_ids[i], float(scores[i])) for i in top_idx]


def rank_of(true_target_id: str, query_emb: np.ndarray, db: CaptionDB) -> int | None:
    """1-based rank of `true_target_id` in the full ranking; None if absent.

    If multiple DB rows share the same target_id (shouldn't happen in our
    smoke setup but possible in larger DBs), the *best* rank wins.
    """
    q = query_emb.reshape(-1)
    scores = db.embeddings @ q
    order = np.argsort(-scores)
    for r, idx in enumerate(order, start=1):
        if db.target_ids[idx] == true_target_id:
            return r
    return None
