# fashion-retrieval-agent

Research workspace for an audio-conditioned composed fashion retrieval system:
given a reference garment image and a spoken modification request (e.g. "make
it black", "shorter sleeves"), retrieve the matching item from a fashion
catalog. The current focus is dataset exploration and method scoping; see
`Documentation/` for proposals, plans, and progress logs.

---

## Project motivation

When I'm shopping in a store and see a piece of clothing I like, if I want to find a similar version, I might first take a photo of it, then later type to ask the model whether there's a similar version, or whether it comes in another color.

But this kind of interaction is actually quite slow, because I need to keep stopping to type.

If I could just keep looking at the clothes and at the same time directly speak:

- *"Do they have it in black?"*
- *"Find me a similar one that's a bit more formal"*

Then the whole flow would be much more natural — and better suited to real-time shopping exploration.

**This project** lets a user **point a camera at a garment** and **speak (or type) a modification** ("make it black", "shorter sleeves", "something similar but more formal"). The system jointly interprets the visual scene + spoken request and returns the matching item from a 59K-product catalog in **one shot** — so people can reduce shopping friction and refine their search in the moment, without ever stopping to type.

---

## Demo

<!-- TODO: replace placeholder with screenshot/gif once Plan-10 Option A finishes and the demo preset cache is rebuilt -->
> **[ demo screenshot / gif placeholder — pending Plan-10 Option A convergence ]**
>
> Side-by-side comparison of the three retrieval recipes (Phase-A caption + Marqo-FashionCLIP, Plan-6 query-only contrastive, Plan-10 two-tower co-trained) on curated FACap dress queries, with an audio-query slot as a v0.2 placeholder.

```bash
bash scripts/run_demo.sh    # → opens http://localhost:7860
```

Demo source: [`src/demo/app.py`](src/demo/app.py).

---

## Headline result

| FACap dress, 1,000-query held-out, 59,048-item gallery |
|---|
| **R@10 = 0.637** &nbsp;·&nbsp; +10.4 pp over Marqo-FashionCLIP (0.533) &nbsp;·&nbsp; +5.1 pp over Qwen3-Embedding-8B (0.586) &nbsp;·&nbsp; +165 % over the MiniLM caption baseline (0.240) |

<sub>Numbers are mid-run on Plan-10 V1 (Option B) at epoch ~11 of 18; final pending.</sub>

**Jump to:**
[Pipelines](#pipeline-comparison) ·
[Architecture](#architecture) ·
[Results](#results) ·
[Recipes](#recipes) ·
[Phase B training](#phase-b-contrastive-training-plans-510) ·
[Demo](#demo) ·
[Docs](#documentation-index)

---

## Pipeline comparison

Seven retrieval pipelines were designed, trained, and benchmarked across the project's lifecycle. The system progressed from **text-only caption retrieval** (Phase A) to **co-trained two-tower multi-modal retrieval** (Phase B), and finally to **native audio query** (Plan-10 V4) — built on a speech-extended VLM backbone that consumes audio tokens directly, no ASR step.

> The 6 pipelines below are the **representative recipes** that anchored project decisions. Inside Family A specifically, we benchmarked **11 retrieval encoders × 2 caption regimes** (concise vs detailed VLM caption); see the [Encoder ablation](#encoder-ablation-phase-a-11-retrieval-encoders) and [Caption-prompt ablation](#caption-prompt-ablation-plan-9) tables under *Results* for the full sweep.

| # | Pipeline | Query input | Target encoder | Trainable | R@10 | Status |
|---|---|---|---|---|---|---|
| 1 | Caption + MiniLM-L6 (Plan-3 anchor) | text caption | MiniLM (frozen) | ✗ | 0.240 | baseline |
| 2 | Caption + Marqo-FashionCLIP (Phase-A best, v1 concise) | text caption | FashionCLIP (frozen) | ✗ | 0.533 | strong fashion baseline |
| 3 | Caption + Qwen3-Embedding-8B (Plan-9 v2 detailed) | detailed VLM caption | Qwen3-Embedding-8B (frozen) | ✗ | 0.586 | strong general baseline |
| 4 | Qwen2VL → frozen FashionCLIP alignment (Plan-6) | (image, text) | FashionCLIP image (frozen) | query only | 0.402 | first contrastive attempt |
| 5 | **Two-tower Qwen2VL — separate backbones (Plan-10 V1 Option B)** | (image, text) | Qwen2VL (trainable) | **both** | **0.637** | **current best, mid-run** |
| 6 | Two-tower Qwen2VL — shared backbone + 2 LoRA adapters (Plan-10 V1 Option A) | (image, text) | Qwen2VL (trainable) | both | TBD | 🟡 in progress |
| 7 | **Two-tower Qwen2VL + native audio query (Plan-10 V4 audio extension)** | **(image, audio)** | Qwen2VL (trainable) | both | TBD | **audio-native variant of Pipeline 6** 🎙️ |

**The key insight (4 → 5):** replacing a frozen target encoder with a **co-trained** target tower lets the embedding space be **constructed end-to-end** by both towers rather than inherited from a frozen teacher — improving R@10 from 0.402 to 0.637 (+58.5 % relative) on the same data.

**The audio extension (6 → 7):** since the backbone is speechQwen2VL — already natively audio-capable — swapping the text-mod input channel for a spoken-mod audio waveform requires **no architectural change**. Same two-tower system, same training loop, same eval; users can type *or* speak.

### Architecture families (deployment view)

The 6 pipelines fall into **3 architecture families** — components within a family are identical, only the specific models swap. The diagrams below show **what to deploy** for each family: every model called out is a real component you need to provision (a captioner, an encoder, a database, …). Each diagram is split into **OFFLINE INDEX** (how the 59K target embeddings get into the database) and **ONLINE QUERY** (what happens at search time).

#### Family A — Caption-based retrieval (Pipelines 1, 2, 3)

```
OFFLINE INDEX  (built once per text encoder)
─────────────
  For each of N target items in the catalog:

    target image
        │
        ├──────────[ optional: VLM Captioner ]──────────┐
        │  (skip if your catalog already provides       │
        │   captions, e.g. Fashion200K metadata)        │
        ▼                                                │
    target caption  ←───────────────────────────────────┘
        │
        ▼
    Text Encoder  (MiniLM / Marqo-FashionCLIP / Qwen3-Embedding / …)
        │
        ▼
    target embedding  (D-dim)
        │
        ▼
    Database  (N × D)


ONLINE QUERY
────────────
  reference image  +  modification text
                  │
                  ▼
           VLM Captioner  (Qwen2VL / speechQwen2VL / …)
                  │
                  ▼
        "imagined target caption"
                  │
                  ▼
           Text Encoder   ←  same model as offline index
                  │
                  ▼
           query embedding  (D-dim)
                  │
                  ▼
        cosine similarity over N target embeddings
                  │
                  ▼
                top-K
```

**Deployment checklist:** ① a VLM captioner for the query side, ② a text encoder (your pick — swap to change which Pipeline 1/2/3 you're running) used **both** at indexing and querying, ③ a vector database holding `N × D` target embeddings. Index is static — build once, reuse forever.

#### Family B — Query-tower contrastive (Pipeline 4 — Plan-6)

```
OFFLINE INDEX  (built once)
─────────────
  For each of N target items in the catalog:

    target image
        │
        ▼
    FashionCLIP image encoder  (frozen, off-the-shelf)
        │
        ▼
    target embedding  (512-d)
        │
        ▼
    Database  (N × 512)


ONLINE QUERY
────────────
  reference image  +  modification text
                  │
                  ▼
    Qwen2VL + LoRA + Projection head
       (trained contrastively against the frozen target encoder)
                  │
                  ▼
           query embedding  (512-d)
                  │
                  ▼
        cosine similarity over N target embeddings
                  │
                  ▼
                top-K
```

**Deployment checklist:** ① the frozen target image encoder (e.g. FashionCLIP — no training needed), ② a custom-trained query model (Qwen2VL + LoRA + projection), ③ a vector database. The query model is trained once and the target embeddings are fixed forever — index built only once.

#### Family C — Two-tower joint embedding (Pipelines 5, 6, 7 — Plan-10)

```
OFFLINE INDEX  (rebuilt at end of every training epoch — target tower is trainable)
─────────────
  For each of N target items in the catalog:

    Target Tower input:  target image  +  "Describe this image in detail."  (fixed prompt)
        │
        ▼
    Qwen2VL Target Tower + LoRA + Projection head
       (trained, co-evolves with the query tower)
        │
        ▼
    target embedding  (512-d)
        │
        ▼
    Database  (N × 512)   ← refreshed every epoch during training;
                            frozen after the final epoch


ONLINE QUERY
────────────
    Query Tower input:  reference image  +  modification
                                          ├── text   (Pipelines 5, 6)
                                          └── audio  (Pipeline 7, native — no ASR)
                  │
                  ▼
    Qwen2VL Query Tower + LoRA + Projection head
       (trained jointly with the target tower via symmetric InfoNCE;
        speechQwen2VL backbone natively consumes text OR audio tokens)
                  │
                  ▼
           query embedding  (512-d)
                  │
                  ▼
        cosine similarity over N target embeddings
                  │
                  ▼
                top-K
```

**Deployment checklist:** ① two custom-trained Qwen2VL towers (query + target, both with LoRA + projection), ② a vector database that gets re-encoded at the end of every training epoch (then frozen for serving), ③ a contrastive training loop with cross-GPU negatives. Both towers co-evolve during training; post-training the system serves like Family B's static index.

**Tower inputs are asymmetric.** The target tower always sees `(target image, fixed description prompt)`; the query tower sees `(reference image, user modification)` where the modification is either typed text or spoken audio. The speechQwen2VL backbone treats both modalities as token streams, so the same trained system serves both — no model swap, no separate audio pipeline.

---

## Architecture

Plan-10 V1 (Pipelines 5 & 6) — **two-tower joint embedding** trained with symmetric multi-positive InfoNCE:

```
   QUERY TOWER                                 TARGET TOWER
   ───────────                                 ────────────
   reference image  +  modification text       target image  +  "Describe this image in detail."
          │                                            │
          ▼                                            ▼
   Qwen2VL-7B backbone                          Qwen2VL-7B backbone
   (speechQwen2VL = Qwen2VL + Stage-2)          (speechQwen2VL = Qwen2VL + Stage-2)
          │                                            │
          ▼                                            ▼
   LoRA (rank 32, q/k/v/o)                      LoRA (rank 32, q/k/v/o)
          │                                            │
          ▼                                            ▼
   last-token pooling                           last-token pooling
          │                                            │
          ▼                                            ▼
   3584 → 1024 → 512 MLP head                   3584 → 1024 → 512 MLP head
          │                                            │
          ▼                                            ▼
   q_emb (L2-normalized)                        t_emb (L2-normalized)
          │                                            │
          └────────────── contrastive ───────────────┘
                   symmetric multi-positive InfoNCE,
                  cross-GPU all_gather global negatives
```

**Option A (shared backbone)** uses one Qwen2VL backbone with two PEFT LoRA adapters toggled via `set_adapter` — saves VRAM at the cost of an adapter-gradient-checkpointing footgun (mitigated by disabling gradient checkpointing).

**Option B (separate backbones)** instantiates two independent Qwen2VL instances — 2× resident VRAM but zero cross-talk risk; the variant validated first.

The same architecture is **audio-extensible** because the backbone (speechQwen2VL) natively accepts audio tokens alongside image + text. Swap the query-side modification text for an audio waveform and the system trains end-to-end on spoken queries with no architectural change.

---

## Results

### Headline (FACap dress, 1,000-query held-out, 59,048-item gallery)

| Pipeline | R@1 | R@5 | R@10 | R@50 |
|---|---|---|---|---|
| Caption + MiniLM-L6 | 0.084 | 0.191 | 0.240 | 0.384 |
| Caption + Marqo-FashionCLIP (v1 concise) | 0.258 | 0.456 | 0.533 | 0.685 |
| Caption + Qwen3-Embedding-8B (v2 detailed) | 0.290 | — | 0.586 | 0.710 |
| Qwen2VL → frozen FashionCLIP (Plan-6) | — | — | 0.402 | 0.646 |
| **Two-tower Qwen2VL (Plan-10 V1 Option B, ep ~11)** | **0.222** | **0.522** | **0.637** | **0.842** |
| Two-tower Qwen2VL (Plan-10 V1 Option A) | TBD | TBD | TBD | TBD |

### Encoder ablation (Phase A, 11 retrieval encoders)

Top of the 11-encoder sweep under the concise VLM caption regime — full table at [`Documentation/encoder_swap_table.md`](Documentation/encoder_swap_table.md):

| Encoder | Dim | R@1 | R@10 | R@50 |
|---|---|---|---|---|
| Marqo-FashionCLIP | 512 | 0.258 | **0.533** | 0.685 |
| Qwen3-Embedding-8B | 4096 | 0.174 | 0.522 | 0.704 |
| BGE-large | 1024 | 0.233 | 0.496 | 0.685 |
| E5-large-v2 | 1024 | 0.231 | 0.496 | 0.670 |
| MiniLM-L6 | 384 | 0.084 | 0.240 | 0.384 |

### Caption-prompt ablation (Plan-9)

Same 11 encoders re-evaluated under a **detailed** VLM caption vs the original **concise** caption — full table at [`Documentation/encoder_swap_table_v1_vs_v2.md`](Documentation/encoder_swap_table_v1_vs_v2.md):

| Encoder | v1 R@10 (concise) | v2 R@10 (detailed) | Δ |
|---|---|---|---|
| Qwen3-Embedding-8B | 0.522 | **0.586** | **+0.064** |
| Qwen3-Embedding-4B | 0.475 | 0.552 | +0.077 |
| Marqo-FashionCLIP | **0.533** | 0.484 | −0.049 |
| BGE-large | 0.496 | 0.458 | −0.038 |
| Marqo-FashionSigLIP | 0.455 | 0.420 | −0.035 |

**Finding:** detailed captions help high-capacity general-purpose embedders (Qwen3 family) but **regress** smaller / fashion-specialized encoders — the longer text introduces noise that BERT-family and CLIP-family retrievers can't filter.

---

## Recipes

### Recipe 1 — Caption + MiniLM-L6 (Plan-3 anchor)

The project's anchor baseline.

- **Input**: reference image + modification text → VLM caption ("imagined target")
- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2`, frozen, 384-dim
- **Code**: [`src/baseline/run_baseline.py`](src/baseline/run_baseline.py)
- **R@10**: 0.240
- **Why it matters**: anchored the lower bound that every subsequent recipe is measured against.

### Recipe 2 — Caption + Marqo-FashionCLIP (Phase-A best)

After an 11-encoder ablation, Marqo-FashionCLIP was the strongest off-the-shelf encoder on the concise caption regime.

- **Encoder**: `hf-hub:Marqo/marqo-fashionCLIP`, 512-dim
- **Code**: [`scripts/run_encoder_swap.sh`](scripts/run_encoder_swap.sh) drives the 11-encoder sweep
- **R@10**: 0.533
- **Why it matters**: locked the strongest fashion-specific frozen-encoder baseline; defines the gap the trained system must close.

### Recipe 3 — Caption + Qwen3-Embedding-8B (Plan-9 detailed prompt)

Same caption-retrieval shape as Recipe 2 but with a detailed VLM caption and the strongest general-purpose embedder.

- **Encoder**: `Qwen/Qwen3-Embedding-8B`, 4096-dim
- **R@10**: 0.586
- **Why it matters**: per-encoder × per-prompt interaction — detailed captions help Qwen3 but hurt FashionCLIP-family encoders.

### Recipe 4 — Qwen2VL → frozen FashionCLIP image alignment (Plan-6)

First contrastive recipe. Query tower is a trainable Qwen2VL; target is the frozen FashionCLIP **image** encoder.

- **Query tower**: speechQwen2VL + LoRA + projection head → 512-dim
- **Target tower**: Marqo-FashionCLIP image encoder, frozen
- **Loss**: symmetric multi-positive InfoNCE with cross-GPU `all_gather`
- **Code**: [`src/training/train_plan5.py`](src/training/train_plan5.py)
- **R@10**: 0.402 — **regressed** from the 0.533 Phase-A baseline
- **Why it matters**: the regression revealed the ceiling — the query tower was capped at FashionCLIP's discrimination. Motivated Recipe 5.

### Recipe 5 — Two-tower Qwen2VL, separate backbones (Plan-10 V1 Option B) ✅ current best

Both query and target are trainable Qwen2VL towers. Embedding space co-constructed during training instead of inherited from a frozen teacher.

- **Both towers**: speechQwen2VL + LoRA + projection head → 512-dim
- **Loss**: symmetric multi-positive InfoNCE with cross-GPU `all_gather`, dynamic database re-encoding at end of every epoch
- **Code**: [`src/training/train_plan10.py`](src/training/train_plan10.py), [`src/training/two_tower_model.py`](src/training/two_tower_model.py) (`TwoTowerSeparateBackbones`)
- **Launch**: `bash scripts/run_plan10.sh --arch separate`
- **R@10**: **0.637** (mid-run, epoch ~11 of 18)
- **Why it matters**: validates the project's core architectural hypothesis; +0.235 absolute R@10 vs Recipe 4, beats every Phase-A baseline.

### Recipe 6 — Two-tower Qwen2VL, shared backbone + 2 LoRA adapters (Plan-10 V1 Option A) 🟡 in progress

Same loss / data / eval as Recipe 5, but with **one shared Qwen2VL backbone** plus two PEFT LoRA adapters toggled via `set_adapter`. Trades 2× VRAM for adapter cross-talk risk.

- **Mitigation**: gradient checkpointing disabled (PEFT footgun — checkpoint recompute reads the *current* `active_adapter` rather than the one active during forward), documented in [`Documentation/Progress_11_20260512.md`](Documentation/Progress_11_20260512.md).
- **Code**: [`src/training/two_tower_model.py`](src/training/two_tower_model.py) (`TwoTowerSharedBackbone`)
- **Launch**: `bash scripts/run_plan10.sh --arch shared`
- **R@10**: TBD — currently training; status tracked in Progress_11.

### Recipe 7 — Two-tower Qwen2VL with native audio query (Plan-10 V4 audio extension) 🎙️

Audio-native variant of Recipe 6. Same two-tower architecture; the modification channel on the query side is replaced by a spoken-modification audio waveform consumed directly by speechQwen2VL's audio encoder — no separate ASR step.

- **Both towers**: speechQwen2VL + LoRA + projection head → 512-dim (identical to Recipe 6)
- **Query input**: `(reference image, spoken modification)` — audio tokens fed natively into the backbone
- **Target tower**: identical to Recipe 6 — `(target image, "Describe this image in detail.")`
- **Loss / training**: same symmetric multi-positive InfoNCE with cross-GPU `all_gather`, dynamic end-of-epoch gallery refresh
- **Code**: extension of [`src/training/train_plan10.py`](src/training/train_plan10.py) with an audio collator
- **R@10**: TBD (training pending) — expected to match Recipe 6 baseline at parity given identical backbone and training data, modality swap only
- **Why it matters**: closes the loop on the Motivation section's "speak as you browse" promise — the trained two-tower system serves both typed and spoken modification queries with zero architectural change.

---

## Setup (only when replicating on a fresh machine)

The steps below rebuild the dataset and venv layout on a new machine — a GPU
server, a fresh laptop, etc. **Skip this section on the dev machine that
already has these things.** Re-running is safe: each step is idempotent.

### 1. Clone third-party dataset repos

```bash
bash scripts/setup_datasets.sh
```

Clones four public repos into `data_exploration/datasets/`. Total ~240 MB of
text annotations, **no images**.

| Local path | Source | Purpose |
|---|---|---|
| `fashion-iq/` | github.com/XiaoxiaoGuo/fashion-iq | FashionIQ captions, splits, starter code |
| `fashion-iq-metadata/` | github.com/hongwang600/fashion-iq-metadata | ASIN → Amazon image URL mapping |
| `facap-repo/` | github.com/fgxaos/facap-sigir25-gennext | All FACap triplet + image-caption JSONs |
| `fashion-200k/` | github.com/xthan/fashion-200k | README only; images on HuggingFace mirror |

### 2. Python venv

```bash
python3 -m venv data_exploration/venv
data_exploration/venv/bin/pip install requests pillow jupyter ipykernel \
    tqdm matplotlib datasets
```

### 3. (Optional) Recreate the small image samples used by the notebook

`setup_datasets.sh` pulls annotations only. To match the dev machine exactly,
also run the two helpers described below — together they pull ~30 thumbnails
(<1 MB total). Skip if you don't need to re-execute the notebook.

## Data-fetching helpers

Two thin helpers materialize the small image samples used during dataset
exploration. They're designed to run on demand; nothing else in the repo
fetches images automatically.

### FashionIQ image fetcher (inside the notebook)

`data_exploration/dataset_inspection.ipynb` defines `fetch_image(asin, url, cat)`,
which downloads one Amazon-hosted image by ASIN and caches it under
`data_exploration/datasets/fashion-iq-images/<cat>/<asin>.jpg`. The notebook's
sampling cell calls this in a loop to pull ~14 triplets across dress, shirt,
and toptee.

- **Input:** ASIN strings parsed from `cap.<cat>.<split>.json` triplets, joined
  to URLs from `fashion-iq-metadata/image_url/asin2url.<cat>.txt`.
- **Output:** `.jpg` files cached locally; PIL.Image objects in memory for
  rendering.
- **When to run:** open the notebook and execute cells top to bottom — the
  helper is invoked automatically. Images already on disk are reused.
- **Failure modes:** Amazon may return 404/403 for individual ASINs; the
  helper logs and skips them rather than failing the whole batch.

### FACap dress sample fetcher (standalone script)

```bash
data_exploration/venv/bin/python data_exploration/fetch_facap_sample.py
```

Streams the `Marqo/fashion200k` HuggingFace mirror (~3.47 GB total, but only
the first ~300 records are pulled — about 5 MB over the wire) and saves any
images that match the first FACap dress triplets into
`data_exploration/datasets/facap-images/`. It also writes a manifest JSON
(`dress_sample_manifest.json`) listing the matched triplets for the
notebook's FACap rendering cell to consume.

- **Input:** `data_exploration/datasets/facap-repo/data/facap/cir_triplets/dress_train_triplets.json`
  + the streaming HuggingFace dataset.
- **Output:** ~5 `.jpeg` files (~70 KB total) and a manifest JSON.
- **When to run:** once after `setup_datasets.sh`, if you want the notebook's
  FACap section to render real images. The notebook's text cells work without
  these.
- **Knobs:** `STREAM_N` (how many HF records to scan) and `MAX_MATCHES`
  (how many triplets to keep) at the top of the script.

## Full image downloads (training-time)

The setup script and helpers above cover **annotations + small samples**, not
the full image data needed for training. When the implementation phase
starts, the following downloads need to be added (no script exists yet —
TODO: `scripts/download_full_datasets.py`):

1. **FashionIQ images** — full splits across dress / shirt / toptee (~57k images).
   - Source: Amazon CDN URLs in `fashion-iq-metadata/image_url/asin2url.*.txt`.
   - Estimated 5–10 GB.
   - Risk: Amazon URL rot. URLs were alive in 2026 but the dataset was released
     at ICCV 2019, so long-term entries may drop.

2. **Fashion200k images** — needed for FACap pretraining.
   - HuggingFace mirror: `huggingface.co/datasets/Marqo/fashion200k`.
   - ~3.47 GB, ~200k images.
   - Stream by `item_ID` (e.g. `51727804_0`); FACap triplets reference these
     by paths like `f200k_images/dresses/.../51727804_0.jpeg`.

3. **DeepFashion-MultiModal images** — auxiliary FACap source.
   - HuggingFace mirror: `huggingface.co/datasets/Marqo/deepfashion-multimodal`.

Defer until the training plan specifies which splits and categories are
actually needed — pretraining all six FACap categories vs. starting with just
dress changes the disk and bandwidth footprint significantly.

## Baseline pipeline (`src/`)

The text-modification retrieval baseline (Plan_2). Method: turn
`(reference image + modification text)` into an "imagined target caption"
via a VLM, then retrieve via text-to-text similarity against pre-encoded
target captions. Each file under `src/` does one step of that pipeline.

| File | Role |
|---|---|
| `src/data/facap_dataset.py` | `FacapDataset` — iterates FACap CIR triplets; returns 6-key dicts with image **paths** (lazy I/O via `load_image()`) |
| `src/baseline/text_encoder.py` | Sentence-BERT wrapper (`all-MiniLM-L6-v2`), CPU-only, L2-normalized 384-d output |
| `src/baseline/build_caption_db.py` | Builds the retrieval index at `runs/<run_name>/caption_db/` (embeddings + metadata + provenance config) |
| `src/baseline/vlm_caption.py` | Pluggable captioner: `Mock` / `Oracle` / `Qwen2VL` / `SpeechQwen2VL`. Real backends are server-only at ≥14 GB VRAM |
| `src/baseline/prepare_images.py` | Pre-fetches eval-slice images so real VLM runs don't depend on mid-run network calls |
| `src/baseline/retrieve.py` | Cosine similarity top-K + true-target rank lookup |
| `src/baseline/eval.py` | Recall@1/5/10/50 + median + mean rank; writes per-query qualitative JSONL |
| **`src/baseline/run_baseline.py`** | **Entry point.** Auto-builds the caption DB if missing (with stale-DB gates), runs the full eval loop, writes metrics + qualitative |

System-design diagrams + per-milestone execution log live in
[`Documentation/Progress_2_20260420.md`](Documentation/Progress_2_20260420.md);
the bird's-eye phase roadmap is in
[`Documentation/Plan_overview.md`](Documentation/Plan_overview.md).

## Running the baseline

### Conda env

The baseline code runs in a dedicated **conda** env (separate from
the dataset-exploration `data_exploration/venv/` above):

```bash
conda env create -f environment.yml
conda activate fashion_retrieval
```

Local 8 GB-VRAM laptops can run the mock and oracle backends; the
real VLM backends (`qwen2vl`, `speechqwen2vl`) raise a clear
`server-only` RuntimeError below 14 GB VRAM and are intended for the
GPU server.

### Smoke runs

Two end-to-end runs verify the pipeline:

```bash
# Oracle: identity-path sanity check; should hit Recall@1 = 1.0.
# A failure here means the encoder/index/retrieve/rank chain has a bug.
python -m src.baseline.run_baseline --vlm oracle --n-eval 50 --run-name smoke_oracle

# Mock: numbers don't matter, only that the pipeline runs to completion
# and writes the expected artifacts.
python -m src.baseline.run_baseline --vlm mock --n-eval 50 --run-name smoke_mock
```

Outputs land under `runs/smoke_{oracle,mock}/` (gitignored): the
auto-built caption DB at `caption_db/`, `metrics.json`, and
`qualitative/results.jsonl`.

### Tests

Persistent reproducibility checks for M1–M3 — 13 cases across three
files. Each file is runnable as a script (no pytest dependency) and
also discoverable by pytest:

```bash
# Each milestone individually (script mode, prints ✓/✗ per case)
python -m tests.test_m1_facap_dataset
python -m tests.test_m2_caption_db
python -m tests.test_m3_pipeline

# Or all 13 at once via pytest (optional install: pip install pytest)
pytest tests/
```

Tests build into fresh `runs/_test_*/` directories so they don't
collide with your smoke runs.

## Real VLM baseline on a server

Run the headline baseline (`speechqwen2vl` backend, 1000-query FACap
dress eval slice) on a GPU box. The smoke runs above use mock/oracle
captioners and run on CPU; the real VLM run needs a GPU.

### Hardware & setup (single-GPU caption-retrieval inference)

- **NVIDIA GPU with ≥ 14 GB VRAM at bf16.** Qwen2-VL-7B occupies
  ~15 GB once image tokens are in the mix. Single GPU is enough.
- **`HF_TOKEN`** set up so model downloads from
  `DanJZY/Qwen2-VL-7B-Speech` and `Marqo/fashion200k` don't
  rate-limit. Verify with `huggingface-cli whoami`.
- **~20 GB free disk** for the Qwen2-VL-7B base + LoRA adapter +
  caption-DB artifacts. If `~` is tight, set `HF_HOME` to a scratch
  path before running setup:
  ```bash
  export HF_HOME=/scratch/$USER/hf_cache
  ```

### One-shot setup

Clone this repo and `speechQwen2VL` as siblings, then run the setup
script:

```bash
cd ~/CSprojects   # (or wherever; just keep them siblings)
git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git
git clone https://github.com/ZhuoyuanJiang/fashion-retrieval-agent.git
cd fashion-retrieval-agent
bash scripts/setup_server.sh
conda activate fashion_retrieval
bash scripts/setup_datasets.sh   # FACap + FashionIQ + Fashion200k annotations
```

`setup_server.sh` creates the `fashion_retrieval` conda env from
`environment.yml`, installs `requirements.txt`, then shells out to
`speechQwen2VL/scripts/setup_forks.sh` to install the forked
`transformers` + `qwen-vl-utils` (these must be installed *last* so
they override the upstream `transformers` that `sentence-transformers`
brings in).

### One-shot run

```bash
bash scripts/run_baseline_v1.sh
```

Defaults: `n_eval=1000`, `db_size=59082` (full FACap dress targets),
`vlm=speechqwen2vl`, `run_name=baseline_v1_speechqwen2vl`. Override
via env vars (e.g.
`N_EVAL=50 RUN_NAME=smoke_real bash scripts/run_baseline_v1.sh` for
a quick smoke first).

The script does three things:
1. Pre-fetches the eval slice's reference images into the local cache
   (no surprise network calls mid-run).
2. Runs the baseline: VLM caption-generation + text-to-text retrieval
   against the 59k FACap dress target captions.
3. Pretty-prints `metrics.json`.

Outputs land under `runs/<run_name>/`:

```
runs/baseline_v1_speechqwen2vl/
  caption_db/
    embeddings.npy        (N=59082, dim=384) float32
    metadata.jsonl        target_id, image_path, caption per row
    config.json           encoder + build_args + facap_commit_sha
  metrics.json            Recall@1/5/10/50, median + mean rank
  qualitative/
    results.jsonl         per-query top-10 + generated caption + true rank
                          (failure_category field starts blank, fill by hand)
```

Takes ~20–30 minutes on a single GPU (most time is VLM forward passes).

### Troubleshooting

- **`RuntimeError: server-only: ... needs ≥ 14.0 GB VRAM`** — the
  selected GPU is too small. Pick a different one with
  `CUDA_VISIBLE_DEVICES=N`.
- **Fork override didn't stick.** If
  `python -c "import transformers; print(transformers.__version__)"`
  prints `5.x` instead of `4.56.0.dev0`, the `setup_forks.sh` step
  didn't run. Re-run:
  `bash ../speechQwen2VL/scripts/setup_forks.sh`.
- **HF download stalls or 429s.** Set `HF_TOKEN` env var via
  `huggingface-cli login`.
- **Stale-DB error.** `runs/<run_name>/caption_db/` was built with
  different args (encoder, eval size, FACap commit) than this run.
  Either delete the run dir or use a fresh `--run-name`. The error
  message names the offending arg(s).

## Phase B: Contrastive training (Plans 5–10)

The training pipeline for Phase B contrastive recipes (Recipes 4, 5, 6).

### Hardware & setup (multi-GPU contrastive training)

- **Hardware**: ≥ 7× A6000-class GPU (≈ 49 GB VRAM each) for the default Plan-10 config (`--arch separate`, batch 8, 8 GPUs, gather=ON). Option A (shared backbone, `--arch shared`) can run on fewer GPUs at smaller batch sizes — see `scripts/run_plan10.sh` flag overrides.
- **Disk**: ≥ 30 GB free on a fast local SSD for `HF_HOME` — Qwen2-VL-7B-Speech ≈ 17 GB + Stage-2 LoRA ≈ 650 MB + gallery embeddings ≈ 120 MB × 18 epochs.
- **FACap images**: the full ~60 K FACap dress image set must be available at `$FACAP_IMAGES_DIR`. The existing helper (`data_exploration/fetch_facap_sample.py`) pulls 5 samples only — for the full set, stream `huggingface.co/datasets/Marqo/fashion200k` and save items matching the IDs in `data_exploration/datasets/facap-repo/data/facap/cir_triplets/dress_*.json`. (A reusable script is on the project TODO list; pull requests welcome.)
- **Required env vars** (export before invoking any `run_plan*.sh`):
  ```bash
  export HF_HOME=/path/to/local/ssd/hf_cache
  export WANDB_DIR=/path/to/local/ssd/wandb_cache
  export FACAP_IMAGES_DIR=/path/to/local/ssd/facap-images
  export PYTHONUNBUFFERED=1   # for long runs under nohup / tmux
  ```
- **W&B**: project name is `fashion-retrieval-agent`. Run `wandb login` once before the first run.

### Plan-6: query-tower contrastive (frozen FashionCLIP target)

```bash
bash scripts/run_plan5.sh
```

Defaults: 8 GPUs, batch 8/GPU, gather=ON, 18 epochs. Frozen FashionCLIP image encoder serves as the target; only the Qwen2VL query tower + projection head are trainable. Outputs land under `runs/plan5/<run_name>/`.

### Plan-10: two-tower co-trained (current best)

```bash
# Option B — two separate Qwen2VL backbones (recommended; current best result)
bash scripts/run_plan10.sh --arch separate

# Option A — shared backbone + two PEFT LoRA adapters (lower VRAM, in progress)
bash scripts/run_plan10.sh --arch shared
```

Defaults: 8 GPUs, batch 8/GPU, gather=ON, 18 epochs, end-of-epoch gallery refresh. Both towers are trainable; the embedding space is co-constructed.

W&B run names auto-generate as `plan10/v1_<arch>_bs<N>_<G>x<gpu>_<date>` from `torch.cuda.get_device_name()`.

### Eval

Both training scripts run dev + headline retrieval evaluation at every 0.5 epoch automatically. Numbers logged to W&B (`fashion-retrieval-agent` project) and persisted to `runs/<run_name>/metrics.json`.

---

## Documentation index

The full design and execution history lives in [`Documentation/`](Documentation/). Reading order for someone catching up cold:

1. [`Plan_overview.md`](Documentation/Plan_overview.md) — bird's-eye roadmap (Phases A / B / C).
2. [`Plan_3_20260430.md`](Documentation/Plan_3_20260430.md) + [`Progress_3_20260430.md`](Documentation/Progress_3_20260430.md) — Phase-A baseline + 11-encoder ablation.
3. [`Plan_5_20260501.md`](Documentation/Plan_5_20260501.md) → `Plan_7_20260503.md` + corresponding Progress — Plan-6 query-tower contrastive recipe.
4. [`Plan_9_20260504.md`](Documentation/Plan_9_20260504.md) + [`Progress_9_20260505.md`](Documentation/Progress_9_20260505.md) — detailed-vs-concise VLM caption ablation.
5. [`Plan_10_20260510.md`](Documentation/Plan_10_20260510.md) + [`Progress_10_20260512.md`](Documentation/Progress_10_20260512.md) + [`Progress_11_20260512.md`](Documentation/Progress_11_20260512.md) — two-tower co-trained architecture (current best).
6. [`meeting_memo_20260503.md`](Documentation/meeting_memo_20260503.md) — mentor feedback that motivated Plan 10.

---

## Repo structure

- `Documentation/` — proposals, plans, progress reports, meeting memos.
- `data_exploration/` — inspection notebook, sample fetchers, scratch space.
- `scripts/` — reproducibility helpers.
- `src/` — baseline implementation (Plan_2 M1–M3); entry point is `src/baseline/run_baseline.py`.
- `tests/` — runnable test suite for M1–M3 (13 cases).
- `runs/` — gitignored: caption DBs, metrics, qualitative dumps.

## Licenses

- FashionIQ: CDLA-Permissive.
- FACap: not stated in upstream repo or project page (clarify before
  redistributing derived artifacts).
- Fashion200k / DeepFashion-MultiModal: see source repos.
- This repo never redistributes third-party image data; all images are fetched
  from upstream at setup time.
