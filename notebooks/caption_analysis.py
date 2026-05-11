# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (fashion_retrieval)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Caption generation analysis
#
# Inspect the captions produced by the **speechQwen2VL** captioner on the
# Phase A baseline (`runs/baseline_v1_speechqwen2vl/`). For each of the 1000
# evaluation queries we have:
#
# - the **reference** image + the natural-language **modification**
#   instruction (input to the VLM)
# - the **generated_caption** produced by the VLM (the system's guess at
#   what the target should look like)
# - the **target** image + the FACap-provided ground-truth target
#   caption (what a "correct" caption looks like)
# - the **rank** of the true target in the encoder's top-K retrieval
#   over the gallery
#
# This notebook does **not** re-run the VLM or re-encode anything. All
# captions are loaded from the saved `qualitative/results.jsonl` artifact.
#
# ### What it does
# 1. Sanity stats (rank distribution, R@K, caption length).
# 2. Per-query gallery — side-by-side image+text view per query.
# 3. Rank-stratified gallery — what does a rank=1 caption look like vs a
#    rank=500 caption?
# 4. Caption-vs-target programmatic comparison (length ratio, token
#    overlap, optional BLEU). Does caption quality correlate with
#    retrieval rank?
# 5. Optional opt-in failure-annotation scaffold.
#
# ### What it does NOT do
# - No baseline rerun, no encoder swap.
# - No data files written to `runs/` (except optionally if you call
#   `annotate()` in the last section).

# %% [markdown]
# ## Pipeline recap — what produced the numbers in this notebook
#
# This notebook reads the `runs/baseline_v1_speechqwen2vl/` run, whose
# pipeline is:
#
# | Component | Choice |
# |---|---|
# | **VLM** (caption generator) | `speechqwen2vl` |
# | **Text encoder** (gallery + query embedding) | `sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-dim — the **weakest anchor** from the Plan 3 M4 encoder zoo) |
# | **Gallery** | All 59,082 dress captions from FACap (`dress_train_captions.json`) |
# | **Eval queries** | 1000 dress triplets from FACap (`dress_train_triplets.json`) |
#
# **End-to-end retrieval, per query:**
#
# 1. `(reference image, modification text)` → `speechqwen2vl` → `generated_caption` (1 string, median ≈ 92 chars)
# 2. `MiniLM-L6.encode(generated_caption)` → `query_emb` ∈ ℝ³⁸⁴
# 3. (built once, reused) `MiniLM-L6.encode(every dress caption)` → `gallery_embs` ∈ ℝ^{59082 × 384}
# 4. cosine similarity → 59,082 scores → sort → find the true target's position → `rank`
# 5. Aggregate: `R@K = #{rank ≤ K} / 1000`
#
# So **R@1 = 84 / 1000 = 0.084**, **R@10 = 240 / 1000 = 0.240**.
#
# **Two properties to keep in mind:**
#
# - **Text-only after step 1.** Steps 2–5 never see a pixel. Failure
#   modes split cleanly into *VLM-side* (caption is bad) vs
#   *encoder-side* (caption is fine but text→embedding mapping fails).
# - **Captions are encoder-independent.** The 1000 captions are written
#   by the VLM once and saved; swapping the encoder (e.g. to Marqo
#   FashionCLIP, R@1 = 0.258) only re-runs steps 2–4. So the
#   caption-quality findings in this notebook (length gap, token
#   overlap, etc.) transfer verbatim to any other encoder run.
#
# Why MiniLM-L6 specifically (not the strongest encoder)? It's the
# Plan 3 baseline anchor and has more failure samples to characterize
# (rank ≥ 201: 472 of 1000 vs ~250 with Marqo). To re-analyze a
# different encoder run, change `RUN_DIR` in cell 2 below.

# %% [markdown]
# ## 1. Imports + paths
#
# Change `RUN_DIR` to point at any other run directory (e.g.
# `runs/baseline_v1_speechqwen2vl_marqo-fashionclip`) to reanalyze that
# encoder's ranks. The captions are identical across encoder-swap runs,
# so only the rank-stratification cells will look different.

# %%
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

# Make `src/` importable when running from notebooks/.
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.facap_dataset import FacapDataset, _path_to_image_id  # noqa: E402

# ---- config: change RUN_DIR to analyze a different run ----
RUN_DIR = REPO_ROOT / "runs" / "baseline_v2_speechqwen2vl_20260504"
RESULTS = RUN_DIR / "qualitative" / "results.jsonl"
METRICS = RUN_DIR / "metrics.json"
ANNOTATIONS = RUN_DIR / "annotations.jsonl"  # opt-in; only created if you call annotate()

print(f"REPO_ROOT     : {REPO_ROOT}")
print(f"RUN_DIR       : {RUN_DIR}")
print(f"RESULTS exists: {RESULTS.exists()}")
print(f"METRICS exists: {METRICS.exists()}")

# %% [markdown]
# ## 2. Load data
#
# Read all 1000 query rows + the FACap dataset (for image rendering and
# target-caption lookup).

# %%
rows: list[dict] = [json.loads(line) for line in open(RESULTS)]
metrics: dict = json.loads(METRICS.read_text())
ds = FacapDataset(category="dress", split="train")

# Map image_id -> caption (FACap stores by path; flip to id for fast lookup
# from results.jsonl rows which carry true_target as an id).
ID_TO_CAPTION: dict[str, str] = {
    _path_to_image_id(path): cap for path, cap in ds.captions.items()
}

print(f"loaded {len(rows)} query rows from {RESULTS.name}")
print(f"first row keys: {sorted(rows[0].keys())}")
print(f"FacapDataset: {len(ds)} triplets, {len(ID_TO_CAPTION)} captioned images")
print(f"image cache : {ds.image_cache}")

# %% [markdown]
# ## 3. Quick stats — R@K and rank distribution

# %%
print("Headline metrics for this run:")
print(f"  N         = {metrics['n']}")
print(f"  R@1       = {metrics['recall']['R@1']:.4f}")
print(f"  R@5       = {metrics['recall']['R@5']:.4f}")
print(f"  R@10      = {metrics['recall']['R@10']:.4f}")
print(f"  R@50      = {metrics['recall']['R@50']:.4f}")
print(f"  median rk = {metrics['median_rank']}")
print(f"  mean rk   = {metrics['mean_rank']}")
print(f"  unranked  = {metrics['n_unranked']}")
print(f"  encoder   = {metrics['context'].get('encoder', '?')}")
print(f"  vlm       = {metrics['context'].get('vlm', '?')}")

# %%
ranks_present = [r["rank"] for r in rows if r["rank"] is not None]
n_missing = sum(1 for r in rows if r["rank"] is None)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(np.log10(np.clip(ranks_present, 1, None)), bins=40, color="steelblue", edgecolor="white")
ax.set_xlabel("log10(rank)")
ax.set_ylabel("count")
ax.set_title(f"Rank distribution ({len(ranks_present)} ranked, {n_missing} missing-from-DB)")
ax.axvline(0, color="green", linestyle="--", alpha=0.6, label="rank=1 (perfect)")
ax.axvline(1, color="orange", linestyle="--", alpha=0.6, label="rank=10 (R@10)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Caption length stats
#
# How verbose is the VLM versus the FACap ground-truth caption?

# %%
gen_lens = [len(r["generated_caption"]) for r in rows]
tgt_lens = [
    len(ID_TO_CAPTION.get(r["true_target"], "")) for r in rows
    if r["true_target"] in ID_TO_CAPTION
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(gen_lens, bins=40, color="indianred", edgecolor="white")
axes[0].set_title(f"Generated caption length (n={len(gen_lens)})")
axes[0].set_xlabel("characters"); axes[0].set_ylabel("count")
axes[0].axvline(np.median(gen_lens), color="black", linestyle="--", label=f"median={int(np.median(gen_lens))}")
axes[0].legend()

axes[1].hist(tgt_lens, bins=40, color="seagreen", edgecolor="white")
axes[1].set_title(f"Target caption length (n={len(tgt_lens)})")
axes[1].set_xlabel("characters")
axes[1].axvline(np.median(tgt_lens), color="black", linestyle="--", label=f"median={int(np.median(tgt_lens))}")
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"generated: median={int(np.median(gen_lens))}  mean={np.mean(gen_lens):.0f}  "
      f"min={min(gen_lens)}  max={max(gen_lens)}")
print(f"target   : median={int(np.median(tgt_lens))}  mean={np.mean(tgt_lens):.0f}  "
      f"min={min(tgt_lens)}  max={max(tgt_lens)}")

# %% [markdown]
# ## 5. Per-query gallery
#
# Each query renders as **three images**:
#
# 1. **Reference** (the input garment + the natural-language modification)
# 2. **Target** (the ground-truth answer)
# 3. **Top-1 retrieved** (what the baseline system actually returned)
#
# Below each row we print all four pieces of text: the modification, the
# VLM-generated caption (used as the retrieval query), the FACap
# ground-truth target caption, and the FACap caption for the top-1
# retrieved garment. This is the cell to scroll through and screenshot/share.
#
# When the rank is 1 the target and top-1 panels are the same image (success).

# %%
def _load_image_safe(image_id: str) -> Image.Image | None:
    """Return PIL image for a given FACap image_id, or None if not cached."""
    if not image_id:
        return None
    p = ds.image_cache / f"{image_id}.jpeg"
    if not p.exists():
        return None
    try:
        return Image.open(p)
    except Exception:
        return None


def _wrap(text: str, width: int = 80) -> str:
    """Greedy word-wrap so long captions render readably under the image."""
    if not text:
        return ""
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def render_query(row: dict) -> None:
    """Render reference + target + top-1-retrieved images with all caption text.

    Three columns:
      1. Reference (query)        — input garment
      2. Target (ground truth)    — what the system *should* retrieve
      3. Top-1 retrieved          — what the system *did* retrieve (= target if rank==1)
    """
    qid = row["query_id"]
    tgt_id = row["true_target"]
    rank = row["rank"]
    rank_str = "missing" if rank is None else f"#{rank}"
    top10 = row.get("top10_predicted") or []
    top10_scores = row.get("top10_scores") or []
    top1_id = top10[0] if top10 else None
    top1_score = top10_scores[0] if top10_scores else None
    top1_match = "✓ match" if top1_id == tgt_id else "✗ wrong"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = [
        (axes[0], qid, f"REFERENCE\n{qid}"),
        (axes[1], tgt_id, f"TARGET (ground truth)\n{tgt_id}   rank {rank_str}"),
        (
            axes[2],
            top1_id,
            (f"TOP-1 RETRIEVED  ({top1_match})\n{top1_id}"
             + (f"   score={top1_score:.3f}" if top1_score is not None else ""))
            if top1_id else "TOP-1 RETRIEVED\n(none)",
        ),
    ]
    for ax, img_id, label in panels:
        img = _load_image_safe(img_id) if img_id else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5,
                    f"image not in cache:\n{img_id}" if img_id else "(no top-1)",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    tgt_caption = ID_TO_CAPTION.get(tgt_id, "(target caption not found)")
    top1_caption = ID_TO_CAPTION.get(top1_id, "(no caption)") if top1_id else "(no top-1)"
    print("─" * 110)
    print(f"  modification:        {_wrap(row['modification_text'])}")
    print(f"  VLM generated:       {_wrap(row['generated_caption'])}")
    print(f"  ground-truth target: {_wrap(tgt_caption)}")
    print(f"  top-1 caption:       {_wrap(top1_caption)}")
    print()


# %% [markdown]
# ### 5a. Ten random queries (seed-fixed)

# %%
rng = random.Random(42)
sample = rng.sample(rows, 10)
for r in sample:
    render_query(r)

# %% [markdown]
# ### 5b. Five hand-picked queries spanning the rank spectrum

# %%
def _first_with_rank_in(rows: list[dict], lo: int, hi: int | None) -> dict | None:
    for r in rows:
        rk = r["rank"]
        if rk is None:
            continue
        if rk >= lo and (hi is None or rk <= hi):
            return r
    return None

def _first_missing(rows: list[dict]) -> dict | None:
    for r in rows:
        if r["rank"] is None:
            return r
    return None


picks: list[tuple[str, dict | None]] = [
    ("rank=1 (perfect)",       _first_with_rank_in(rows, 1, 1)),
    ("rank 2-10 (top-10)",     _first_with_rank_in(rows, 2, 10)),
    ("rank 11-50 (top-50)",    _first_with_rank_in(rows, 11, 50)),
    ("rank 51-500 (mid)",      _first_with_rank_in(rows, 51, 500)),
    ("rank 500+ (deep miss)",  _first_with_rank_in(rows, 501, None)),
]
miss = _first_missing(rows)
if miss is not None:
    picks.append(("missing-from-DB", miss))

for label, r in picks:
    if r is None:
        print(f"\n[{label}] no example found in this run\n")
        continue
    print(f"\n========== {label} ==========")
    render_query(r)

# %% [markdown]
# ## 6. Rank-stratified gallery
#
# Three random examples per rank bucket. Lets you eyeball whether the
# captions for rank-1 queries look qualitatively different from rank-500
# queries.

# %%
BUCKETS: list[tuple[str, int, int | None]] = [
    ("rank=1",       1, 1),
    ("rank 2-10",    2, 10),
    ("rank 11-50",   11, 50),
    ("rank 51-200",  51, 200),
    ("rank 201+",    201, None),
]
N_PER_BUCKET = 3
seed_rng = random.Random(2026)

bucket_rows: dict[str, list[dict]] = {}
for label, lo, hi in BUCKETS:
    in_bucket = [
        r for r in rows
        if r["rank"] is not None and r["rank"] >= lo and (hi is None or r["rank"] <= hi)
    ]
    bucket_rows[label] = in_bucket
missing_rows = [r for r in rows if r["rank"] is None]
bucket_rows["missing-from-DB"] = missing_rows

for label, in_bucket in bucket_rows.items():
    sampled = seed_rng.sample(in_bucket, min(N_PER_BUCKET, len(in_bucket))) if in_bucket else []
    print(f"\n========== {label}  ({len(in_bucket)} queries; showing {len(sampled)}) ==========")
    if not sampled:
        print("  (no queries in this bucket for this run)")
        continue
    for r in sampled:
        render_query(r)

# %% [markdown]
# ## 7. Caption-vs-target programmatic comparison
#
# For every query, compute simple agreement metrics between the
# generated caption and the FACap ground-truth target caption:
#
# - **`len_ratio`** = `len(generated) / len(target)` — is the VLM
#   under- or over-describing?
# - **`token_overlap`** = `|gen_tokens ∩ target_tokens| / |target_tokens|`
#   on lowercased whitespace tokens — fraction of target's content words
#   the VLM also said.
# - **`bleu`** (optional) — only computed if `nltk` is available.
#
# Then plot retrieval rank vs each metric and report Spearman correlation.

# %%
def tokenize(s: str) -> set[str]:
    return {t.strip(".,;:!?()'\"") for t in s.lower().split() if t.strip()}

def metrics_for(row: dict) -> dict | None:
    """Return per-row caption-vs-target agreement metrics, or None if unusable."""
    tgt = ID_TO_CAPTION.get(row["true_target"])
    if tgt is None:
        return None
    gen = row["generated_caption"]
    gen_toks, tgt_toks = tokenize(gen), tokenize(tgt)
    overlap = len(gen_toks & tgt_toks) / max(len(tgt_toks), 1)
    return {
        "rank": row["rank"],  # may be None
        "len_ratio": len(gen) / max(len(tgt), 1),
        "token_overlap": overlap,
    }


per_row = [m for m in (metrics_for(r) for r in rows) if m is not None]
ranked = [m for m in per_row if m["rank"] is not None]
print(f"computed agreement metrics on {len(per_row)} rows ({len(ranked)} have a rank).")

# Optional BLEU.
bleu_scores: list[float] | None = None
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for r in rows:
        tgt = ID_TO_CAPTION.get(r["true_target"])
        if tgt is None:
            bleu_scores.append(float("nan"))
            continue
        bleu_scores.append(sentence_bleu(
            [tgt.lower().split()], r["generated_caption"].lower().split(),
            smoothing_function=smooth,
        ))
    print(f"BLEU computed for {sum(1 for b in bleu_scores if not np.isnan(b))} rows.")
except ImportError:
    print("nltk not installed — BLEU skipped (install nltk to enable).")

# %%
ranks_arr = np.array([m["rank"] for m in ranked])
overlap_arr = np.array([m["token_overlap"] for m in ranked])
lenratio_arr = np.array([m["len_ratio"] for m in ranked])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(ranks_arr, overlap_arr, alpha=0.3, s=10, color="steelblue")
axes[0].set_xscale("symlog")
axes[0].set_xlabel("rank (log scale)"); axes[0].set_ylabel("token_overlap")
axes[0].set_title("Token overlap vs rank")
axes[0].axvline(10, color="orange", linestyle="--", alpha=0.5, label="R@10 cutoff")
axes[0].legend()

axes[1].scatter(ranks_arr, lenratio_arr, alpha=0.3, s=10, color="indianred")
axes[1].set_xscale("symlog"); axes[1].set_yscale("log")
axes[1].set_xlabel("rank (log scale)"); axes[1].set_ylabel("len_ratio (log)")
axes[1].set_title("Length ratio vs rank")
axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="parity")
axes[1].legend()
plt.tight_layout()
plt.show()

rho_overlap, p_overlap = spearmanr(ranks_arr, overlap_arr)
rho_len, p_len = spearmanr(ranks_arr, lenratio_arr)
print(f"Spearman rank vs token_overlap: rho = {rho_overlap:+.4f}  (p = {p_overlap:.2e})")
print(f"Spearman rank vs len_ratio    : rho = {rho_len:+.4f}  (p = {p_len:.2e})")
print()
print("Interpretation:")
print("  - negative rho on token_overlap means: higher overlap correlates with")
print("    smaller (better) rank — i.e. captions that share words with the")
print("    ground-truth caption tend to retrieve their target more easily.")
print("  - rho near zero would mean caption-target agreement does NOT predict")
print("    retrieval quality on this run.")

# %% [markdown]
# ## 8. Per-bucket caption-agreement summary
#
# Mean token_overlap per rank bucket — gives a one-glance answer to "are
# good-rank queries the ones with high caption agreement?"

# %%
agg_rows = []
for label, lo, hi in BUCKETS:
    overlaps = [m["token_overlap"] for m in ranked
                if m["rank"] >= lo and (hi is None or m["rank"] <= hi)]
    if overlaps:
        agg_rows.append((label, len(overlaps), np.mean(overlaps), np.median(overlaps)))
agg_rows.append((
    "missing-from-DB",
    sum(1 for r in rows if r["rank"] is None),
    np.mean([metrics_for(r)["token_overlap"]
             for r in rows if r["rank"] is None and metrics_for(r) is not None] or [np.nan]),
    np.median([metrics_for(r)["token_overlap"]
               for r in rows if r["rank"] is None and metrics_for(r) is not None] or [np.nan]),
))

print(f"{'bucket':<20} {'n':>5}  {'mean overlap':>14}  {'median overlap':>14}")
print("-" * 60)
for label, n, mean, median in agg_rows:
    mean_s = f"{mean:.3f}" if not np.isnan(mean) else "  nan"
    med_s = f"{median:.3f}" if not np.isnan(median) else "  nan"
    print(f"{label:<20} {n:>5}  {mean_s:>14}  {med_s:>14}")

# %% [markdown]
# ## 9. Optional: failure annotation scaffold
#
# If you want to hand-label some failures with the Plan_2 rubric, the
# helper below appends rows to `runs/baseline_v1_speechqwen2vl/annotations.jsonl`.
# Nothing is written until you uncomment a call.
#
# **Rubric** (from `Documentation/Plan_2_20260427.md`):
#
# - `caption_wrong` — VLM caption fails to summarize the modification correctly.
# - `embedding_mismatch` — caption looks correct, but embedding-space retrieval failed.
# - `dataset_ambiguity` — multiple gallery items match the description equally.
# - `visual_nuance_lost` — VLM omits fine visual details (texture, weave, etc.).

# %%
VALID_CATEGORIES = {
    "caption_wrong",
    "embedding_mismatch",
    "dataset_ambiguity",
    "visual_nuance_lost",
}

def annotate(query_id: str, category: str, notes: str = "") -> None:
    """Append a hand-annotation to runs/<run>/annotations.jsonl. Idempotent: each
    call appends a new row; nothing is overwritten. Safe to re-run cells."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"unknown category {category!r}; must be one of {sorted(VALID_CATEGORIES)}"
        )
    record = {
        "query_id": query_id,
        "category": category,
        "notes": notes,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended {record} -> {ANNOTATIONS}")


def show_annotation_summary() -> None:
    """Print counts per category from annotations.jsonl. No-op if file missing."""
    if not ANNOTATIONS.exists():
        print(f"no annotations yet at {ANNOTATIONS}")
        return
    cats = Counter(
        json.loads(line)["category"] for line in open(ANNOTATIONS)
    )
    print(f"annotations in {ANNOTATIONS}:")
    for k, v in cats.most_common():
        print(f"  {k:<22} {v}")
    print(f"  {'total':<22} {sum(cats.values())}")


# Example usage — uncomment to actually write:
# annotate("91306678_2", "caption_wrong", "VLM said 'light blue' but target is dark teal")
# annotate("91306678_3", "embedding_mismatch", "caption is right; encoder ranked similar dresses higher")
# show_annotation_summary()

# %% [markdown]
# ---
#
# That's it. Re-run the notebook end-to-end with
#
# ```bash
# jupyter nbconvert --to notebook --execute --inplace notebooks/caption_analysis.ipynb
# ```
#
# Or export to a self-contained HTML for sharing:
#
# ```bash
# jupyter nbconvert --to html notebooks/caption_analysis.ipynb
# ```
