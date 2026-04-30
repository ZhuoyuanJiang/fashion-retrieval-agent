"""Plan_2 M3 — the orchestrator wires captioner -> encoder -> retrieve ->
eval correctly. Oracle should hit perfect Recall@1 (proves plumbing);
mock should run to completion and write the expected artifacts.

Run as a script:
    conda activate fashion_retrieval
    python -m tests.test_m3_pipeline

Or via pytest:
    pytest tests/test_m3_pipeline.py

Total runtime ~30-60 seconds — each backend builds its own small DB
(10 eval + 190 distractors = 200 captions) on first invocation.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.run_baseline import run as run_baseline_run  # noqa: E402
from src.baseline.text_encoder import DEFAULT_MODEL  # noqa: E402
from src.data.facap_dataset import DEFAULT_IMAGE_CACHE  # noqa: E402
from tests._runner import cli_main  # noqa: E402

N_EVAL = 10
DB_SIZE = 200
SEED = 42

ORACLE_RUN_NAME = "_test_m3_oracle"
MOCK_RUN_NAME = "_test_m3_mock"


def _run_backend(vlm: str, run_name: str) -> Path:
    """Invoke run_baseline.run() for one backend; return its run dir."""
    run_dir = REPO_ROOT / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_baseline_run(
        vlm=vlm,
        n_eval=N_EVAL,
        run_name=run_name,
        category="dress",
        split="train",
        db_size=DB_SIZE,
        encoder_name=DEFAULT_MODEL,
        seed=SEED,
        out_root=REPO_ROOT / "runs",
        image_cache=DEFAULT_IMAGE_CACHE,
    )
    return run_dir


def _read_metrics(run_dir: Path) -> dict:
    """Load metrics.json — every successful run writes it."""
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists(), f"metrics.json missing at {metrics_path}"
    return json.loads(metrics_path.read_text())


def test_oracle_recall_at_1_is_perfect() -> None:
    """Oracle returns the ground-truth caption -> R@1 must be 1.0.

    A failure here means the encoder/index/retrieval/rank pipeline has a
    bug — the input is the exact DB caption, so any rank > 1 indicates
    a plumbing problem (encoder non-determinism, sort direction, off-by-one
    in rank, DB row alignment, etc.).
    """
    run_dir = _run_backend("oracle", ORACLE_RUN_NAME)
    metrics = _read_metrics(run_dir)
    recall = metrics["recall"]
    assert recall["R@1"] == 1.0, (
        f"oracle R@1 should be 1.0, got {recall['R@1']}; "
        f"the plumbing has a bug, not the model."
    )
    assert recall["R@5"] == 1.0
    assert recall["R@10"] == 1.0
    assert metrics["median_rank"] == 1.0
    assert metrics["mean_rank"] == 1.0


def test_mock_pipeline_runs_clean_and_writes_artifacts() -> None:
    """Mock should run end-to-end without errors; all artifacts must exist.

    Numbers here don't matter (mock is a soft floor, not a true random
    baseline). What matters is that every step of the pipeline runs and
    every expected file is on disk.
    """
    run_dir = _run_backend("mock", MOCK_RUN_NAME)
    metrics = _read_metrics(run_dir)

    expected_top = {"n", "recall", "median_rank", "mean_rank", "n_unranked"}
    missing = expected_top - set(metrics.keys())
    assert not missing, f"metrics.json missing top-level fields: {missing}"
    assert metrics["n"] == N_EVAL, f"expected n={N_EVAL}, got {metrics['n']}"
    expected_recall = {"R@1", "R@5", "R@10", "R@50"}
    missing_r = expected_recall - set(metrics["recall"].keys())
    assert not missing_r, f"metrics.json recall sub-dict missing: {missing_r}"

    qual_path = run_dir / "qualitative" / "results.jsonl"
    assert qual_path.exists(), f"qualitative jsonl missing at {qual_path}"
    rows = [json.loads(l) for l in qual_path.open()]
    assert len(rows) == N_EVAL, f"expected {N_EVAL} qualitative rows, got {len(rows)}"

    # Each row should carry the documented fields including the blank
    # failure_category stub for later by-hand classification.
    expected_fields = {"query_idx", "query_id", "true_target", "modification_text",
                       "generated_caption", "top10_predicted", "top10_scores",
                       "rank", "failure_category"}
    missing = expected_fields - set(rows[0].keys())
    assert not missing, f"first qualitative row missing fields: {missing}"


def test_real_backends_are_server_only() -> None:
    """Qwen2VL and SpeechQwen2VL must raise clearly on the laptop.

    `make_captioner("qwen2vl")` and `make_captioner("speechqwen2vl")`
    eagerly check VRAM in __init__ and raise RuntimeError on machines
    that can't host the 7B model — failing here would mean a future user
    silently tries to load a 14 GB model on an 8 GB GPU and OOMs deep
    inside generation, far from the configuration that caused it.
    """
    from src.baseline.vlm_caption import make_captioner

    for name in ("qwen2vl", "speechqwen2vl"):
        try:
            make_captioner(name, image_cache_root=DEFAULT_IMAGE_CACHE)
        except RuntimeError as e:
            assert "server-only" in str(e).lower() or "vram" in str(e).lower(), (
                f"{name}: RuntimeError raised but message unclear: {e}"
            )
            continue
        # If we get here, it loaded without raising — only legitimate on a
        # ≥14 GB GPU; on the laptop this is a regression.
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if free_gb < 14:
                raise AssertionError(
                    f"{name}: did not raise on a {free_gb:.1f} GB GPU "
                    f"(threshold is 14 GB). VRAM gate has regressed."
                )


TESTS = [
    test_oracle_recall_at_1_is_perfect,
    test_mock_pipeline_runs_clean_and_writes_artifacts,
    test_real_backends_are_server_only,
]


if __name__ == "__main__":
    cli_main(TESTS, "M3 (pipeline)")
