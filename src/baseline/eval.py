"""Metrics + qualitative dump for the baseline run."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECALL_KS = (1, 5, 10, 50)


@dataclass
class EvalResult:
    n: int
    recall: dict[int, float]
    median_rank: float
    mean_rank: float
    n_unranked: int  # eval queries whose true target wasn't in the DB at all

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "recall": {f"R@{k}": v for k, v in self.recall.items()},
            "median_rank": self.median_rank,
            "mean_rank": self.mean_rank,
            "n_unranked": self.n_unranked,
        }


def compute_metrics(ranks: list[int | None], ks: tuple[int, ...] = RECALL_KS) -> EvalResult:
    n = len(ranks)
    n_unranked = sum(1 for r in ranks if r is None)
    ranked = [r for r in ranks if r is not None]
    recall = {}
    for k in ks:
        hits = sum(1 for r in ranked if r <= k)
        recall[k] = hits / n if n > 0 else 0.0
    if ranked:
        arr = np.asarray(ranked, dtype=np.int64)
        median_rank = float(np.median(arr))
        mean_rank = float(np.mean(arr))
    else:
        median_rank = float("nan")
        mean_rank = float("nan")
    return EvalResult(
        n=n, recall=recall, median_rank=median_rank,
        mean_rank=mean_rank, n_unranked=n_unranked,
    )


def write_qualitative(rows: list[dict], out_dir: Path) -> Path:
    """Write `{out_dir}/qualitative/results.jsonl`. One dict per query.

    Each row has: query_id, true_target, top10_predicted, generated_caption,
    rank, failure_category (blank by default; filled in by hand later from the
    rubric in Plan_2).
    """
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    out = qual_dir / "results.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out


def write_metrics(result: EvalResult, out_dir: Path, extra: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    if extra:
        payload["context"] = extra
    out = out_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def format_metrics_table(result: EvalResult) -> str:
    lines = [
        f"  n queries:    {result.n}",
        f"  unranked:     {result.n_unranked}  (true target absent from DB)",
        f"  median rank:  {result.median_rank}",
        f"  mean rank:    {result.mean_rank:.2f}",
    ]
    for k, v in result.recall.items():
        lines.append(f"  R@{k:<3}        {v:.4f}")
    return "\n".join(lines)
