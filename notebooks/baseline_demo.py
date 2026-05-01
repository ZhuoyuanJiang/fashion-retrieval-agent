# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: fashion_retrieval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fashion-retrieval baseline (Plan_2 / Plan_3)
#
# **Stage 1** (separate repo): [speechQwen2VL](https://github.com/ZhuoyuanJiang/speechQwen2VL) — Qwen2-VL-7B fine-tuned with a Whisper encoder + MLP projector + LoRA so it can take audio input.
#
# **Stage 2** (this repo): use Stage-1's model for **composed fashion retrieval**. Given a reference garment image + a modification text (e.g., *"the same dress but red and shorter"*), retrieve the matching item from a catalog.
#
# This notebook runs the **Plan_3 text-modification baseline** end-to-end:
# 1. Sanity-check the harness with the **oracle** backend (identity-path; should hit R@1 ≈ 1.0).
# 2. Run the **real** baseline with `speechQwen2VL` in text-only mode on the FACap dress evaluation slice (last 1000 train triplets).
# 3. Render metrics + qualitative success/failure samples.
#
# **Method A** (this baseline): VLM `(reference image + modification text) → "imagined target caption" → text-to-text retrieval against pre-encoded FACap target captions`. Method B (contrastive end-to-end training) is Plan_4+.

# %% [markdown]
# ## 0. Environment setup
#
# Auto-detects Colab vs server:
#
# - **Colab:** clone the public repos, install deps, install forks. This takes ~10 min on first run.
# - **Server / local with `fashion_retrieval` env active:** assume `bash scripts/setup_server.sh` has been run; just verify imports.
#
# If you're on the server, **activate the env first** in the shell that launched Jupyter: `conda activate fashion_retrieval`.

# %%
import os, sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
print(f"environment: {'Colab' if IN_COLAB else 'server/local'}")

if IN_COLAB:
    # --- Colab path: clone repos + install ---
    if not Path('/content/fashion-retrieval-agent').exists():
        # !git clone https://github.com/ZhuoyuanJiang/fashion-retrieval-agent.git /content/fashion-retrieval-agent
    if not Path('/content/speechQwen2VL').exists():
        # !git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git /content/speechQwen2VL
    # %cd /content/fashion-retrieval-agent
    # !pip install -q -r requirements.txt
    # !bash /content/speechQwen2VL/scripts/setup_forks.sh
    # FACap annotations (no images — those stream from HF)
    # !bash scripts/setup_datasets.sh
else:
    # --- Server / local path: cd into repo, assume env is active ---
    repo = Path.cwd()
    while repo != repo.parent and not (repo / 'src' / 'baseline' / 'run_baseline.py').exists():
        repo = repo.parent
    assert (repo / 'src' / 'baseline' / 'run_baseline.py').exists(), \
        f"can't find repo root from {Path.cwd()}; run from inside fashion-retrieval-agent"
    os.chdir(repo)
    print(f"repo root: {repo}")

# --- verify imports ---
import torch
import transformers
import peft
import sentence_transformers
print(f"torch:                {torch.__version__}  cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"                      device: {p.name}  ({p.total_memory/1024**3:.1f} GB)")
print(f"transformers:         {transformers.__version__}  (expect 4.56.0.dev0)")
print(f"peft:                 {peft.__version__}")
print(f"sentence_transformers:{sentence_transformers.__version__}")

# %% [markdown]
# ## 1. Run config
#
# Override these with environment variables if you want a quicker smoke (e.g., `N_EVAL=50`).

# %%
N_EVAL    = int(os.environ.get('N_EVAL', 1000))
DB_SIZE   = int(os.environ.get('DB_SIZE', 59082))
VLM       = os.environ.get('VLM', 'speechqwen2vl')
RUN_NAME  = os.environ.get('RUN_NAME', f'baseline_v1_{VLM}')

print(f"vlm        : {VLM}")
print(f"n_eval     : {N_EVAL}")
print(f"db_size    : {DB_SIZE}")
print(f"run_name   : {RUN_NAME}")

# %% [markdown]
# ## 2. Sanity 1 — Dataset class
#
# Confirms the FACap dress triplets load and the item dict has the expected 6 keys.

# %%
from src.data.facap_dataset import FacapDataset

ds = FacapDataset(category='dress', split='train')
print(f"FacapDataset(dress/train): {len(ds)} triplets")
print(f"item[0] keys: {sorted(ds[0].keys())}")
print()
print(f"item[0]:")
for k, v in ds[0].items():
    s = v if len(v) < 100 else v[:100] + '...'
    print(f"  {k:25} {s}")

# %% [markdown]
# ## 3. Sanity 2 — Oracle backend (identity-path)
#
# The oracle returns the ground-truth target caption verbatim. With the smoke caption DB (which forces every eval target into the index), this should hit **Recall@1 ≈ 1.0**. If it doesn't, the bug is in the encoder / index / retrieval / eval — not the VLM.
#
# We use a small slice (n=50) since this is a plumbing check, not a real metric.

# %%
from src.baseline.run_baseline import run as run_baseline
from src.baseline.text_encoder import DEFAULT_MODEL
from src.data.facap_dataset import DEFAULT_IMAGE_CACHE

REPO = Path.cwd()

run_baseline(
    vlm='oracle',
    n_eval=50,
    run_name='nb_smoke_oracle',
    category='dress',
    split='train',
    db_size=1000,
    encoder_name=DEFAULT_MODEL,
    seed=42,
    out_root=REPO / 'runs',
    image_cache=DEFAULT_IMAGE_CACHE,
)

# %% [markdown]
# ## 4. Real baseline run
#
# Two steps:
#
# 1. **Pre-fetch reference images** for the eval slice's `candidate_id`s into the local cache. Avoids surprise network calls mid-run.
# 2. **Run the baseline**: `(reference image + modification text) → speechQwen2VL caption → SBERT embedding → cosine top-K → metrics`.
#
# On a single GPU this takes ~20–30 min for `n_eval=1000` (most time is in the VLM forward pass).

# %%
# Step 1: prefetch images
from src.baseline.prepare_images import needed_candidate_ids, fetch_images_for

needed = needed_candidate_ids('dress', 'train', N_EVAL)
saved, missing = fetch_images_for(needed, DEFAULT_IMAGE_CACHE)
print(f"prefetch: {saved} new images saved, {missing} still missing")

# %%
# Step 2: run the real baseline
run_baseline(
    vlm=VLM,
    n_eval=N_EVAL,
    run_name=RUN_NAME,
    category='dress',
    split='train',
    db_size=DB_SIZE,
    encoder_name=DEFAULT_MODEL,
    seed=42,
    out_root=REPO / 'runs',
    image_cache=DEFAULT_IMAGE_CACHE,
)

# %% [markdown]
# ## 5. Metrics table

# %%
import json

metrics_path = REPO / 'runs' / RUN_NAME / 'metrics.json'
metrics = json.load(open(metrics_path))
print(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 6. Qualitative samples
#
# 3 success cases (rank = 1) + 3 failure cases (rank > 50 or unranked), each showing the reference image, modification text, and the top retrieved target image.

# %%
import json
import matplotlib.pyplot as plt
from PIL import Image

qual_path = REPO / 'runs' / RUN_NAME / 'qualitative' / 'results.jsonl'
rows = [json.loads(l) for l in open(qual_path)]

successes = [r for r in rows if r['rank'] == 1][:3]
failures  = [r for r in rows if r['rank'] is None or r['rank'] > 50][:3]

def render_pair(row, label):
    cand_path = DEFAULT_IMAGE_CACHE / f"{row['query_id']}.jpeg"
    pred_top1 = row['top10_predicted'][0]
    tgt_path  = DEFAULT_IMAGE_CACHE / f"{pred_top1}.jpeg"

    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    for ax, path, title in zip(axes,
                                [cand_path, tgt_path],
                                [f'reference ({row["query_id"]})',
                                 f'top-1 retrieved ({pred_top1}, rank={row["rank"]})']):
        if path.exists():
            ax.imshow(Image.open(path).convert('RGB'))
        else:
            ax.text(0.5, 0.5, f'image not in cache:\n{path.name}',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle(f'{label}: "{row["modification_text"][:80]}..."', fontsize=10)
    plt.tight_layout()
    plt.show()
    print(f'  generated caption: {row["generated_caption"][:200]}')
    print()

print('=== Successes (rank == 1) ===')
for r in successes:
    render_pair(r, 'success')

print('=== Failures (rank > 50 or unranked) ===')
for r in failures:
    render_pair(r, 'failure')

# %% [markdown]
# ## 7. Run notes
#
# *Fill this in after the run. Keep it short — 3–5 bullets.*
#
# - **Headline number:** R@1 = ___, R@5 = ___, R@10 = ___, R@50 = ___, median rank = ___.
# - **What this means:** ...
# - **Failure modes seen:** caption_wrong / embedding_mismatch / dataset_ambiguity / visual_nuance_lost — which dominates?
# - **Implication for Plan_4 (contrastive):** strong baseline → contrastive needs a more compelling angle; weak baseline → contrastive has clear motivation.
# - **Open questions / next experiments:** ...
