"""Plan_2 M2 — `build_caption_db.build_db()` produces a self-describing
caption DB with the expected shape, normalization, and provenance.

Run as a script:
    conda activate fashion_retrieval
    python -m tests.test_m2_caption_db

Or via pytest:
    pytest tests/test_m2_caption_db.py

First run takes ~30-60 seconds (encodes 1000 captions on CPU SBERT).
Subsequent runs reuse the built DB at runs/_test_m2/caption_db/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.build_caption_db import build_db  # noqa: E402
from src.baseline.text_encoder import DEFAULT_MODEL, EMBED_DIM  # noqa: E402
from tests._runner import cli_main  # noqa: E402

TEST_RUN_NAME = "_test_m2"
TEST_RUN_DIR = REPO_ROOT / "runs" / TEST_RUN_NAME
DB_DIR = TEST_RUN_DIR / "caption_db"

EVAL_N = 50
DB_SIZE = 1000
SEED = 42


def _ensure_built() -> None:
    """Build the test caption DB once if it doesn't already exist."""
    if (DB_DIR / "embeddings.npy").exists():
        return
    if TEST_RUN_DIR.exists():
        shutil.rmtree(TEST_RUN_DIR)
    build_db(
        run_name=TEST_RUN_NAME,
        eval_n=EVAL_N,
        db_size=DB_SIZE,
        category="dress",
        split="train",
        encoder_name=DEFAULT_MODEL,
        seed=SEED,
        out_root=REPO_ROOT / "runs",
    )


def test_embeddings_shape_and_dtype() -> None:
    """embeddings.npy must be a (DB_SIZE, EMBED_DIM) float32 array."""
    _ensure_built()
    emb = np.load(DB_DIR / "embeddings.npy")
    assert emb.shape == (DB_SIZE, EMBED_DIM), f"expected ({DB_SIZE}, {EMBED_DIM}), got {emb.shape}"
    assert emb.dtype == np.float32, f"expected float32, got {emb.dtype}"


def test_embeddings_l2_normalized() -> None:
    """Every row should be L2-normalized so cosine similarity == matmul.

    If this fails, retrieval would silently compute dot products instead
    of cosine similarities and rankings would be off.
    """
    _ensure_built()
    emb = np.load(DB_DIR / "embeddings.npy")
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"row norms not L2-normalized: range [{norms.min():.4f}, {norms.max():.4f}]"
    )


def test_metadata_aligned_with_embeddings() -> None:
    """metadata.jsonl must have one row per embedding, with the documented fields."""
    _ensure_built()
    rows = [json.loads(l) for l in (DB_DIR / "metadata.jsonl").open()]
    assert len(rows) == DB_SIZE, f"expected {DB_SIZE} metadata rows, got {len(rows)}"
    expected = {"image_path", "target_id", "caption", "caption_length_chars", "source"}
    missing = expected - set(rows[0].keys())
    assert not missing, f"first metadata row missing fields: {missing}"


def test_source_split_matches_smoke_construction() -> None:
    """Smoke build = (eval-targets) + (random distractors); counts must match."""
    _ensure_built()
    rows = [json.loads(l) for l in (DB_DIR / "metadata.jsonl").open()]
    sources: dict[str, int] = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    n_eval = sources.get("eval_target", 0)
    n_dist = sources.get("distractor", 0)
    assert n_eval == EVAL_N, f"expected {EVAL_N} eval_target rows, got {n_eval}"
    assert n_dist == DB_SIZE - EVAL_N, f"expected {DB_SIZE - EVAL_N} distractor rows, got {n_dist}"


def test_config_records_provenance() -> None:
    """config.json must record encoder, build args, and FACap commit SHA.

    Without provenance, you can't tell whether a stored DB is still valid
    against the current encoder / dataset revision — the stale-DB gate in
    run_baseline.ensure_caption_db reads these fields.
    """
    _ensure_built()
    cfg = json.loads((DB_DIR / "config.json").read_text())
    assert cfg["encoder_name"] == DEFAULT_MODEL
    assert cfg["embedding_dim"] == EMBED_DIM
    assert cfg["n_eval_targets_unique"] == EVAL_N
    assert cfg["n_distractors"] == DB_SIZE - EVAL_N
    assert cfg["n_total"] == DB_SIZE
    assert cfg["seed"] == SEED
    assert "build_args" in cfg, "config.json missing build_args (stale-DB gate breaks)"
    sha = cfg.get("facap_commit_sha", "")
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
        f"facap_commit_sha looks malformed: {sha!r}"
    )


TESTS = [
    test_embeddings_shape_and_dtype,
    test_embeddings_l2_normalized,
    test_metadata_aligned_with_embeddings,
    test_source_split_matches_smoke_construction,
    test_config_records_provenance,
]


if __name__ == "__main__":
    cli_main(TESTS, "M2 (caption DB)")
