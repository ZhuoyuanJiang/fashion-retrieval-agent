"""Aggregate per-encoder run dirs into a single comparison table.

Reads `runs/baseline_v1_speechqwen2vl_<slug>/metrics.json` for each slug in
the encoder zoo (and the anchor `runs/baseline_v1_speechqwen2vl/`), prints
a markdown table sorted by R@1, and (if --out is given) writes the same
table to a markdown file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.encoder_zoo import ENCODER_ZOO


ANCHOR_RUN = "baseline_v1_speechqwen2vl"


def _load_metrics(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def gather(out_root: Path) -> list[dict]:
    rows: list[dict] = []
    # Anchor row (the original full run; encoder = MiniLM)
    anchor_metrics = _load_metrics(out_root / ANCHOR_RUN / "metrics.json")
    if anchor_metrics is not None:
        rows.append({
            "slug": "minilm-l6",
            "hf_id": ENCODER_ZOO["minilm-l6"].hf_model_id,
            "params": "22M",
            "dim": 384,
            "notes": ENCODER_ZOO["minilm-l6"].notes,
            "metrics": anchor_metrics,
            "run_dir": ANCHOR_RUN,
            "status": "done",
        })

    for slug, cfg in ENCODER_ZOO.items():
        if slug == "minilm-l6":
            continue
        run_dir = f"baseline_v1_speechqwen2vl_{slug}"
        m = _load_metrics(out_root / run_dir / "metrics.json")
        rows.append({
            "slug": slug,
            "hf_id": cfg.hf_model_id,
            "params": "—",  # unknown / not tracked
            "dim": (m or {}).get("context", {}).get("embedding_dim", "—"),
            "notes": cfg.notes,
            "metrics": m,
            "run_dir": run_dir,
            "status": "done" if m else "missing",
        })

    return rows


def fmt_table(rows: list[dict]) -> str:
    """Markdown table sorted by R@1 desc (missing rows go to the bottom)."""
    def sort_key(r):
        m = r.get("metrics")
        if m is None:
            return (1, 0.0)
        return (0, -float(m["recall"]["R@1"]))

    rows = sorted(rows, key=sort_key)

    headers = [
        "Encoder", "HF ID", "Dim",
        "R@1", "R@5", "R@10", "R@50",
        "Median rank", "Mean rank", "Status",
    ]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r.get("metrics")
        if m is None:
            line = [r["slug"], f"`{r['hf_id']}`", "—",
                    "—", "—", "—", "—", "—", "—", "❌ missing"]
        else:
            rec = m["recall"]
            line = [
                r["slug"],
                f"`{r['hf_id']}`",
                str(r["dim"]),
                f"{rec['R@1']:.4f}",
                f"{rec['R@5']:.4f}",
                f"{rec['R@10']:.4f}",
                f"{rec['R@50']:.4f}",
                f"{m['median_rank']:.1f}",
                f"{m['mean_rank']:.1f}",
                "✅",
            ]
        out.append("| " + " | ".join(line) + " |")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate encoder swap results")
    parser.add_argument("--out-root", default="runs",
                        help="root containing baseline_v1_speechqwen2vl_<slug>/ dirs")
    parser.add_argument("--out", default=None,
                        help="optional: write the markdown table to this file")
    args = parser.parse_args()

    rows = gather(Path(args.out_root))
    table = fmt_table(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote markdown table -> {args.out}")


if __name__ == "__main__":
    main()
