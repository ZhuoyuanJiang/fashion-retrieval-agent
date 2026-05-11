"""Aggregate per-encoder run dirs into a single comparison table.

Default mode (single source prefix): reads
`runs/<source>_<slug>/metrics.json` for each slug in the encoder zoo
(and the anchor `runs/<source>/`), prints a markdown table sorted by
R@1, and (if --out is given) writes it to a markdown file.

Compare mode (--compare-prefix set): scans TWO source-run trees and
emits a side-by-side `Δ` table — Plan 9 uses this to compare concise
vs detailed prompts across the same 11+1 encoder zoo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline.encoder_zoo import ENCODER_ZOO


DEFAULT_SOURCE_PREFIX = "baseline_v1_speechqwen2vl"


def _load_metrics(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def gather(out_root: Path, source_prefix: str = DEFAULT_SOURCE_PREFIX) -> list[dict]:
    """Collect per-encoder rows for a single source-run prefix.

    The anchor row (encoder = MiniLM-L6) reads `<source_prefix>/metrics.json`
    (no slug suffix — historical convention). Each other slug reads
    `<source_prefix>_<slug>/metrics.json`.
    """
    rows: list[dict] = []
    # Anchor row (the source dir doubles as MiniLM-L6's encoder result)
    anchor_metrics = _load_metrics(out_root / source_prefix / "metrics.json")
    if anchor_metrics is not None:
        rows.append({
            "slug": "minilm-l6",
            "hf_id": ENCODER_ZOO["minilm-l6"].hf_model_id,
            "params": "22M",
            "dim": 384,
            "notes": ENCODER_ZOO["minilm-l6"].notes,
            "metrics": anchor_metrics,
            "run_dir": source_prefix,
            "status": "done",
        })

    for slug, cfg in ENCODER_ZOO.items():
        if slug == "minilm-l6":
            continue
        run_dir = f"{source_prefix}_{slug}"
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


def fmt_compare_table(
    a_rows: list[dict],
    b_rows: list[dict],
    a_label: str,
    b_label: str,
) -> str:
    """Side-by-side R@K comparison with Δ columns. Sorted by b's R@1 desc.

    a_rows / b_rows are gather() outputs from two different source-run trees.
    Slugs are matched 1:1 by encoder slug.
    """
    by_slug_a = {r["slug"]: r for r in a_rows}
    by_slug_b = {r["slug"]: r for r in b_rows}
    slugs = sorted(set(by_slug_a) | set(by_slug_b))

    def b_r1(slug: str) -> tuple[int, float]:
        rb = by_slug_b.get(slug)
        if rb is None or rb.get("metrics") is None:
            return (1, 0.0)
        return (0, -float(rb["metrics"]["recall"]["R@1"]))

    slugs = sorted(slugs, key=b_r1)

    headers = [
        "Encoder",
        f"{a_label} R@1", f"{b_label} R@1", "Δ R@1",
        f"{a_label} R@10", f"{b_label} R@10", "Δ R@10",
        f"{a_label} R@50", f"{b_label} R@50", "Δ R@50",
        f"{a_label} median rank", f"{b_label} median rank",
    ]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]

    def cell_metric(row: dict | None, key: str, fmt: str = "{:.4f}") -> str:
        if row is None or row.get("metrics") is None:
            return "—"
        return fmt.format(row["metrics"]["recall"][key])

    def cell_delta(row_a: dict | None, row_b: dict | None, key: str) -> str:
        if (row_a is None or row_a.get("metrics") is None
                or row_b is None or row_b.get("metrics") is None):
            return "—"
        d = row_b["metrics"]["recall"][key] - row_a["metrics"]["recall"][key]
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.4f}"

    def cell_median(row: dict | None) -> str:
        if row is None or row.get("metrics") is None:
            return "—"
        return f"{row['metrics']['median_rank']:.1f}"

    for slug in slugs:
        ra = by_slug_a.get(slug)
        rb = by_slug_b.get(slug)
        line = [
            slug,
            cell_metric(ra, "R@1"), cell_metric(rb, "R@1"),
            cell_delta(ra, rb, "R@1"),
            cell_metric(ra, "R@10"), cell_metric(rb, "R@10"),
            cell_delta(ra, rb, "R@10"),
            cell_metric(ra, "R@50"), cell_metric(rb, "R@50"),
            cell_delta(ra, rb, "R@50"),
            cell_median(ra), cell_median(rb),
        ]
        out.append("| " + " | ".join(line) + " |")

    return "\n".join(out)


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
                        help="root containing <source>_<slug>/ dirs")
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX,
                        help=f"source-run prefix (default: {DEFAULT_SOURCE_PREFIX!r})")
    parser.add_argument("--compare-prefix", default=None,
                        help="if set, emit a side-by-side Δ table comparing "
                             "--source-prefix vs --compare-prefix "
                             "(Plan 9: concise vs detailed)")
    parser.add_argument("--source-label", default=None,
                        help="column label for --source-prefix in compare mode "
                             "(default: short auto-derived name)")
    parser.add_argument("--compare-label", default=None,
                        help="column label for --compare-prefix in compare mode")
    parser.add_argument("--out", default=None,
                        help="optional: write the markdown table to this file")
    args = parser.parse_args()

    out_root = Path(args.out_root)

    if args.compare_prefix:
        a_rows = gather(out_root, source_prefix=args.source_prefix)
        b_rows = gather(out_root, source_prefix=args.compare_prefix)
        a_label = args.source_label or args.source_prefix.split("_")[1] \
            if "_" in args.source_prefix else args.source_prefix
        b_label = args.compare_label or args.compare_prefix.split("_")[1] \
            if "_" in args.compare_prefix else args.compare_prefix
        table = fmt_compare_table(a_rows, b_rows, a_label, b_label)
    else:
        rows = gather(out_root, source_prefix=args.source_prefix)
        table = fmt_table(rows)

    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote markdown table -> {args.out}")


if __name__ == "__main__":
    main()
