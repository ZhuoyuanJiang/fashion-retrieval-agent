# fashion-retrieval-agent

**Audio-conditioned composed fashion retrieval.** Given a reference garment
image and a spoken modification request (e.g. "make it black", "shorter
sleeves"), retrieve the matching item from a fashion catalog. See
`Documentation/` for proposals, plans, and progress logs.

---

## Table of Contents

**🪧 Intro**
- [Project motivation](#project-motivation) — what the system does and why
- [Demo](#demo) — what the running app looks like
- [Quick Start](#-quick-start) — 4-step path to the running demo
- [Headline result](#headline-result)

**🔧 Install & data**
- [Download original data](#download-original-data) — dataset annotations + full image catalogs (when you want more than the cached demo)
- [Data exploration](#data-exploration) — venv + sample images for the dataset-exploration notebook
- [Synthetic training data generation](#synthetic-training-data-generation) — TTS spoken modifications for training the audio-native model

**📐 Method, pipeline & architecture overviews**
- [Pipeline comparison](#pipeline-comparison) — 7 retrieval pipelines across 3 architecture families
- [Architecture families (deployment view)](#architecture-families-deployment-view) — components to provision for each family
- [Model architecture (training view)](#model-architecture-training-view) — backbone, LoRA, pooling, projection, loss
- [Results](#results) — headline table + encoder & caption ablations

**🛠 Running & training**
- [Recipes](#recipes) — exact commands to reproduce each pipeline
- [Replicate caption baseline (deep guide)](#replicate-caption-baseline-deep-guide) — source code map + smoke + full single-GPU run (Recipes 1–3)
- [Contrastive training (deep guide)](#contrastive-training-deep-guide) — multi-GPU hardware/env + launch scripts (Recipes 4–7)

**📚 Meta**
- [Documentation index](#documentation-index)
- [Repo structure](#repo-structure)
- [Acknowledgments](#acknowledgments)
- [Licenses](#licenses)

---

## Project motivation

When I'm shopping in a store and see a piece of clothing I like, if I want to find a similar version, I might first take a photo of it, then later type to ask in a search engine (or nowadays probably an LLM, e.g. ChatGPT) whether there's a similar version, or whether it comes in another color.

But this kind of interaction is actually quite slow, because I need to keep stopping to type.

If I could just keep looking at the clothes and at the same time directly speak:

- *"Do they have it in black?"*
- *"Find me a similar one that's a bit more formal"*

Then the whole flow would be much more natural — and better suited to real-time shopping exploration.

**This project aims to develop a model** that lets a user **point a camera at a garment** and **speak (or type) a modification** ("make it black", "shorter sleeves", "something similar but more formal"). The system then jointly interprets the visual scene + spoken request and returns the matching item from a clothing database (e.g., a 59K-product catalog) in **one shot** — so people can reduce shopping friction and refine their search in the moment, without ever stopping to type.

---

## Demo

```bash
bash scripts/run_demo.sh    # opens http://localhost:7860
```

The Gradio app lets you pick a reference garment image and either **type or
speak** a modification — *"make it black"*, *"shorter sleeves"*, *"something
more formal"* — and returns the top-K matching items from the FACap dress
catalog (59K items). Source: [`src/demo/app.py`](src/demo/app.py).

> *(Screenshot/GIF in progress — try it locally in the meantime.)*

---

## 🚀 Quick Start

**Requirements:** Python 3.10+, conda, NVIDIA GPU with ≥16 GB VRAM (the demo loads a frozen 7B speech-VLM + the LoRA adapters on top).

Get the demo running in 4 steps. (For training a pipeline from scratch see
[§Recipes](#recipes); for data fetching beyond what these 4 steps cover see
[§Download original data](#download-original-data); for non-default demo
modes — your own queries, custom paths, live audio recording — see the
**Beyond cached mode** expandable at the bottom of this section.)

### 1. Clone + create conda env

```bash
git clone https://github.com/ZhuoyuanJiang/fashion-retrieval-agent.git
cd fashion-retrieval-agent
bash scripts/setup_server.sh    # clones speechQwen2VL sibling + creates the fashion_retrieval conda env
```

### 2. Log in to HuggingFace

The base model ([`DanJZY/Qwen2-VL-7B-Speech`](https://huggingface.co/DanJZY/Qwen2-VL-7B-Speech))
and the trained two-tower
([`DanJZY/audio-composed-fashion-item-retriever`](https://huggingface.co/DanJZY/audio-composed-fashion-item-retriever))
download from HF on first use.

```bash
huggingface-cli login
```

### 3. (One-time) Fetch product images needed by demo

```bash
bash scripts/fetch_artifacts.sh --with-images   # download all ~59K FACap images (upstream mirror has no "subset" option)
python scripts/make_demo_thumbs.py              # extract the ~1085 images the demo actually uses
```

### 4. Run the demo (cached version)

```bash
conda activate fashion_retrieval
bash scripts/run_demo.sh        # opens http://localhost:7860
```

Open [http://localhost:7860](http://localhost:7860) and try it.

<details>
<summary><b>Beyond cached mode</b> — env-var recipes for live audio recording, live retrieval, custom paths, custom port</summary>

<br>

Steps 1–4 above run the demo in **cached mode** — 8 preset queries with
pre-computed top-K results, no GPU at request time. You
can also run the demo with your own image + spoken query (instead of the 8
presets), enable live audio recording, override default paths on a fresh
machine, or change the Gradio port/sharing. Set env vars before launching
`bash scripts/run_demo.sh`:

**Try your own image + spoken query** (live retrieval over the full catalog;
needs a GPU + the FACap images you fetched in Quick Start step 3):

```bash
export DEMO_STAGE=v0.2
bash scripts/run_demo.sh
```

**Live audio recording** (records your own speech, runs the Plan-15 audio
two-tower on it):

```bash
export LIVE_AUDIO=1
bash scripts/run_demo.sh
```

**Override paths on a non-default machine** (e.g., if you put the model or
the FACap images somewhere other than the dev-server defaults):

```bash
export AUDIO_2T_CKPT=/your/path/to/audio/checkpoint
export AUDIO_2T_GALLERY=/your/path/to/gallery_emb.npy
export GALLERY_DIR=/your/path/to/facap-images
bash scripts/run_demo.sh
```

**Run on a different port or share publicly** (Gradio settings):

```bash
export GRADIO_SERVER_PORT=8080         # default 7860
export GRADIO_SHARE=1                  # gets a *.gradio.live public tunnel URL
bash scripts/run_demo.sh
```

Full env-var list with defaults at the top of
[`src/demo/config.py`](src/demo/config.py).

</details>

---

## Headline result

| FACap dress · 1,000-query held-out · 59,048-item gallery | R@10 | R@50 |
|---|:---:|:---:|
| **Two-tower, typed modification** — best | **0.654** | **0.866** |
| **Two-tower, native spoken modification** — flagship (audio in, no ASR) | **0.624** | **0.853** |
| Caption + Qwen3-Embedding-8B — best caption baseline | 0.586 | 0.710 |
| Caption + Marqo-FashionCLIP — best fashion baseline | 0.533 | 0.685 |
| Caption + MiniLM-L6 — anchor baseline | 0.240 | 0.384 |

<sub>The two-tower lifts R@10 **+12.1 pp** over [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (2024) — a leading fashion-domain CLIP embedder, near the top of [Marqo's public fashion-retrieval leaderboard](https://github.com/marqo-ai/marqo-FashionCLIP/blob/main/LEADERBOARD.md) (last updated 2024-08) — and **+6.8 pp** over Qwen3-Embedding-8B. Swapping the typed modification for TTS-synthesized speech — entering natively through speechQwen2VL's Whisper encoder, no ASR step — costs only ~0.03 R@10, so the audio-native query is competitive with text. (Best-checkpoint, FACap dress slice.)</sub>

**Jump to:**
[Pipelines](#pipeline-comparison) ·
[Model architecture](#model-architecture-training-view) ·
[Results](#results) ·
[Recipes](#recipes) ·
[Contrastive training](#contrastive-training-deep-guide) ·
[Demo](#demo) ·
[Docs](#documentation-index)

---

# 🔧 Install & data

*How to fetch data and set up dependencies for any goal past Quick Start.*

---

## Download original data

[§Quick Start](#-quick-start) above runs the demo in **cached mode** — 8 preset
queries with pre-computed results, no GPU, just click through. Anything past
that needs more setup. **Pick by your goal:**

- **Run the demo on your own image + query** (instead of the 8 presets) →
  needs a GPU + the FACap image catalog
  ([§Full image downloads](#full-image-downloads-training-time)) + a
  HuggingFace login. Then re-run with
  `DEMO_STAGE=v0.2 bash scripts/run_demo.sh`.

- **Train any retrieval model yourself** (caption baselines or two-towers) →
  needs the [dataset annotations](#annotations--240-mb-json-no-images) +
  the full image catalogs
  ([§Full image downloads](#full-image-downloads-training-time)) + a multi-GPU
  server ([§Contrastive training](#contrastive-training-deep-guide)).

- **Re-train the audio-native model from scratch** → same as training above,
  plus the ~56K-clip synthetic spoken-modification audio dataset
  ([§Synthetic training data generation](#synthetic-training-data-generation)
  regenerates it from the FACap modification text via VCTK voices + Chatterbox
  TTS).

- **Explore the FACap / FashionIQ data in a Jupyter notebook** (just curious
  what's in the data) → needs the
  [dataset annotations](#annotations--240-mb-json-no-images) +
  the [exploration-notebook venv](#create-the-notebook-venv-optional) +
  [~30 sample images](#fetch-about-30-sample-images-for-the-notebook-optional)
  (the latter two in [§Data exploration](#data-exploration)).

Two kinds of original data to pull from upstream: **annotations** (the JSON
files that describe the datasets) and the **image catalogs themselves**.

### Annotations (~240 MB JSON, no images)

These are JSON files that **describe** the datasets — triplet IDs, captions,
train/dev/test splits, image-filename ↔ URL mappings. A FACap dress-train
annotation row looks like:

```json
{"reference": "51727804_0", "modification": "is more fitted and shorter, with vibrant red", "target": "89255983_0"}
```

— it tells you which image IDs form the triplet and what the modification
text is. The JSONs **include the URLs** where each image lives upstream, but
**not the image bytes themselves**. The actual JPEG/PNG image files live
separately and are downloaded by the next subsection.

```bash
bash scripts/setup_datasets.sh
```

Clones four public repos into `data_exploration/datasets/`:

| Local path | Source | Purpose |
|---|---|---|
| `fashion-iq/` | github.com/XiaoxiaoGuo/fashion-iq | FashionIQ captions, splits, starter code |
| `fashion-iq-metadata/` | github.com/hongwang600/fashion-iq-metadata | ASIN → Amazon image URL mapping |
| `facap-repo/` | github.com/fgxaos/facap-sigir25-gennext | All FACap triplet + image-caption JSONs |
| `fashion-200k/` | github.com/xthan/fashion-200k | README only; images on HuggingFace mirror |

### Full image downloads (training-time)

The annotations above only contain image-filename ↔ URL mappings, not the
actual image bytes. The full image data needed for training (and for live
demo retrieval — `DEMO_STAGE=v0.2`) lives on upstream mirrors and is fetched
separately.

**FACap dress images (~3.5 GB, ~59K JPEGs).** Sourced from the
`Marqo/fashion200k` HuggingFace mirror. The reproducibility script handles
this:

```bash
bash scripts/fetch_artifacts.sh --with-images
```

Populates `facap-images/` so the demo can display product thumbnails (combined
with `scripts/make_demo_thumbs.py`) and so training runs have something to
retrieve against.

**Other catalogs (FashionIQ + DeepFashion-MultiModal).** Needed only if you
want to pretrain or evaluate across the broader CIR datasets. No bundled
script yet; pull manually:

1. **FashionIQ images** — full splits across dress / shirt / toptee (~57k images).
   - Source: Amazon CDN URLs in `fashion-iq-metadata/image_url/asin2url.*.txt`.
   - Estimated 5–10 GB.
   - Risk: Amazon URL rot. URLs were alive in 2026 but the dataset was released
     at ICCV 2019, so long-term entries may drop.

2. **DeepFashion-MultiModal images** — auxiliary FACap source.
   - HuggingFace mirror: `huggingface.co/datasets/Marqo/deepfashion-multimodal`.

Defer these until your training plan specifies which splits and categories
are actually needed — pretraining all six FACap categories vs. starting with
just dress changes the disk and bandwidth footprint significantly.

## Data exploration

The [`data_exploration/dataset_inspection.ipynb`](data_exploration/dataset_inspection.ipynb)
notebook renders a few example FACap / FashionIQ triplets inline — reference
image + modification text + target image, side by side — so you can **see
what the data actually looks like**. The notebook has saved outputs, so you
can **browse it directly on GitHub** without setting anything up.

This section is only needed if you want to **re-execute or extend** the
notebook locally on a fresh machine — e.g., to verify the saved outputs
reproduce, or to add your own cells on top of the existing exploration.
Two steps, both optional.

### Create the notebook venv (optional)

A separate venv from the main `fashion_retrieval` conda env, because the
notebook only needs lightweight Python deps (jupyter + pillow + requests).

```bash
python3 -m venv data_exploration/venv
data_exploration/venv/bin/pip install requests pillow jupyter ipykernel \
    tqdm matplotlib datasets
```

### Fetch about 30 sample images for the notebook (optional)

The notebook needs ~30 thumbnails (<1 MB total) to render its example triplets.
Two thin fetchers handle this, each serving one of the two datasets the
notebook renders:

**FashionIQ image fetcher (inside the notebook).**
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

**FACap dress sample fetcher (standalone script).**

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

## Synthetic training data generation

The query-side **spoken modifications** are TTS-synthesized from the FACap
dress-slice modification texts — there is no manually recorded speech in
training. Audio enters the model natively through speechQwen2VL's Whisper
encoder (no ASR step).

Pipeline ([`src/data/build_tts_audio.py`](src/data/build_tts_audio.py)):

1. **Source text** — FACap dress-train triplets (`reference image,
   modification text, target image`); the modification string is the text to
   voice.
2. **Voice bank** — a VCTK reference bank of ~110 speakers (gender-balanced),
   split into a ~100-speaker training pool + 10 held-out OOD speakers.
   Chatterbox is a zero-shot voice clone, so each speaker's reference clip *is*
   the voice.
3. **Synthesis** — Chatterbox TTS renders every used triplet (train + dev +
   headline) to a 16 kHz mono wav with a training-pool speaker; dev + headline
   additionally get a held-out speaker for the separate OOD-voice eval
   (~56,686 clips total).
4. **Manifest** — `manifest.json` indexes every clip by FACap triplet index →
   `{wav, speaker, split, gender, accent}`, under `in_dist` and `ood`
   sub-dicts.

Regenerate (needs VCTK extracted + a Chatterbox env):

```bash
python -m src.data.build_tts_audio bank                            # VCTK speaker bank + synthesis plan
python -m src.data.build_tts_audio synth --shard 0 --num-shards 8  # one per GPU, resumable
python -m src.data.build_tts_audio manifest                        # collate manifest.json
```

Inspect a sample of the synthesized audio in
[`notebooks/audio_dataset_demo.ipynb`](notebooks/audio_dataset_demo.ipynb)
(image + audio + Whisper transcript for 15 triplets). Full details — TTS engine
selection, WER QC, and voice control — are in
[`Documentation/Progress_14_20260515.md`](Documentation/Progress_14_20260515.md).

---

# 📐 Method, pipeline & architecture overviews

*The retrieval architectures we tried and the pipelines they produced — from caption baselines to the audio-native flagship.*

---

## Pipeline comparison

Throughout the project, we tried **three different retrieval architectures**
before adding the audio modality:

1. **Caption-based retrieval** (Phase A, no model training) — three steps:
   (a) an off-the-shelf VLM (e.g., Qwen2-VL) generates a *target* caption
   from the `(reference image, text-modification)` input — i.e., a textual
   description of what the modified garment should look like;
   (b) a frozen text encoder (MiniLM / Marqo-FashionCLIP / Qwen3-Emb / ...)
   turns that caption into a vector;
   (c) cosine-similarity nearest-neighbor over the **catalog** — a
   precomputed index of caption embeddings, one caption per item, all 59K
   items in the FACap dress slice, encoded by the same text encoder.
   → Pipelines 1, 2, 3.

2. **Two-tower with separate Qwen backbones** (Phase B, Plan-10 Option B) —
   the **query tower** takes `(reference image, text-modification)` and
   produces a 512-d query embedding; the **target tower** takes a target
   image and produces a 512-d target embedding; both are trained end-to-end
   with contrastive (InfoNCE) loss so matching pairs land close in a shared
   embedding space. Each tower has its own independent Qwen2-VL backbone +
   LoRA. At retrieval time, the **catalog** is the target-tower embeddings
   of all 59K target images, precomputed once.
   → Pipeline 5.

3. **Two-tower with shared backbone + 2 LoRA adapters** (Phase B, Plan-10
   Option A → Plan-12/13) — same query/target setup as stage 2, but instead
   of two independent Qwen backbones, **one frozen Qwen2-VL backbone serves
   both towers** via PEFT LoRA adapters (one per tower, swapped at forward
   time). Saves roughly half the GPU memory, which is what lets the batch
   grow to 24 (more in-batch negatives → better contrastive learning).
   **This became the main line and produced the best text result
   (R@10 0.654).**
   → Pipeline 6.

Plus one intermediate experiment (Pipeline 4 — Plan-6) sits between stages
1 and 2: a **query-tower-only contrastive setup** — only the query side is
trained (Qwen2-VL + projection head), with a frozen off-the-shelf
FashionCLIP image encoder as the target. Proved the contrastive paradigm
worked but capped at FashionCLIP's discrimination — which motivated
stage 2's full co-trained two-tower.

**Then the audio modality was added on top of stage 3** — same query/target
setup as stage 3, but the query input is now `(reference image,
spoken-modification waveform)` instead of `(reference image, typed-text
modification)`. The waveform goes through speechQwen2VL's Whisper audio
encoder directly, with no ASR conversion to text in between. Audio-native
flagship: **R@10 0.624**.
→ Pipeline 7 (Plan-15).

*(Engineer view: each of these 3 stages maps to a deployment Family — see
[§Architecture families](#architecture-families-deployment-view) below for
the component-level diagrams.)*

The table below lists all **7 pipelines** with their headline R@10 — the
representative recipes that anchored project decisions.

> Inside Family A specifically, we also benchmarked **11 retrieval encoders ×
> 2 caption regimes** (concise vs detailed VLM caption); see the
> [Encoder ablation](#encoder-ablation-phase-a-11-retrieval-encoders) and
> [Caption-prompt ablation](#caption-prompt-ablation-plan-9) tables under
> *Results* for the full sweep.

| # | Pipeline | Query input | Target encoder | Trainable | R@10 | Status |
|---|---|---|---|---|---|---|
| 1 | Caption + MiniLM-L6 (Plan-3 anchor) | text caption | MiniLM (frozen) | ✗ | 0.240 | baseline |
| 2 | Caption + Marqo-FashionCLIP (Phase-A best, v1 concise) | text caption | FashionCLIP (frozen) | ✗ | 0.533 | strong fashion baseline |
| 3 | Caption + Qwen3-Embedding-8B (Plan-9 v2 detailed) | detailed VLM caption | Qwen3-Embedding-8B (frozen) | ✗ | 0.586 | strong general baseline |
| 4 | Qwen2VL → frozen FashionCLIP alignment (Plan-6) | (image, text) | FashionCLIP image (frozen) | query only | 0.402 | first contrastive attempt |
| 5 | Two-tower Qwen2VL — separate backbones (Plan-10 Option B) | (image, text) | Qwen2VL (trainable) | both | 0.637 | separate-backbone, final |
| 6 | **Two-tower Qwen2VL — shared backbone + 2 LoRA adapters (Plan-10 → 13)** | (image, text) | Qwen2VL (trainable) | **both** | **0.654** | **best (text)** |
| 7 | **Two-tower Qwen2VL + native audio query (Plan-15)** | **(image, audio)** | Qwen2VL (trainable) | **both** | **0.624** | **flagship — audio-native, competitive with text** 🎙️ |

> *Note on the* Query input *column*: Pipelines 1–3 (caption-based) show
> the VLM-generated caption that the frozen text encoder actually sees —
> the user originally provides `(reference image, text-modification)`, and
> the VLM converts it to a caption upstream. Pipelines 4–7 (contrastive)
> show what the trained query tower receives directly, without that
> intermediate caption step.

**The key insight (4 → 5):** replacing a frozen target encoder with a **co-trained** target tower lets the embedding space be **constructed end-to-end** by both towers rather than inherited from a frozen teacher — improving R@10 from 0.402 to 0.637 (+58.5 % relative) on the same data.

**The audio extension (6 → 7):** since the backbone is speechQwen2VL — already natively audio-capable — swapping the text-mod input channel for a spoken-mod audio waveform requires **no architectural change**. Same two-tower system, same training loop, same eval; users can type *or* speak.

## Architecture families (deployment view)

If you wanted to **actually deploy** one of the 7 pipelines above in production, this section is the guide.

The 7 pipelines fall into **3 architecture families** — components within a family are identical, only the specific models swap. The diagrams below show **what to deploy** for each family: every model called out is a real component you need to provision (a captioner, an encoder, a database, …). Each diagram is split into **OFFLINE INDEX** (how the 59K target embeddings get into the database) and **ONLINE QUERY** (what happens at search time).

*For the **training-side internals** of the winning family (C / two-tower) — backbone, LoRA, pooling, projection head, loss — see [§Model architecture (training view)](#model-architecture-training-view) below.*

### Family A — Caption-based retrieval (Pipelines 1, 2, 3)

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

### Family B — Query-tower contrastive (Pipeline 4 — Plan-6)

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

### Family C — Two-tower joint embedding (Pipelines 5, 6, 7 — Plan-10)

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

## Model architecture (training view)

*If you wanted to **actually build and train** the winning two-tower yourself, the diagram below walks through how it's wired up internally — backbone, adapters, pooling, projection head, loss. For **what components you'd need to deploy** this in production, see [§Architecture families (deployment view)](#architecture-families-deployment-view) above.*

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
| Two-tower Qwen2VL — separate backbone (Plan-10 Option B) | 0.222 | 0.522 | 0.637 | 0.842 |
| **Two-tower Qwen2VL — shared backbone (Plan-13, bs=24)** | **0.231** | **0.528** | **0.654** | **0.866** |
| **Two-tower Qwen2VL — native audio query (Plan-15)** 🎙️ | **0.210** | **0.522** | **0.624** | **0.853** |

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

# 🛠 Running & training

*Source files, smoke tests, single-GPU baseline runs, and multi-GPU contrastive training.*

---

## Recipes

Each recipe below is a minimum reproduction guide for one of the 7 pipelines above — what to launch, what encoders/models you'll use, and the R@10 you should see.

### Recipe 1 — Caption + MiniLM-L6 (Plan-3 anchor)

The project's anchor baseline.

- **Input**: reference image + modification text → VLM caption ("imagined target")
- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2`, frozen, 384-dim
- **Code**: [`src/baseline/run_baseline.py`](src/baseline/run_baseline.py)
- **Launch**: `bash scripts/run_baseline_v1.sh` (MiniLM is the default encoder)
- **R@10**: 0.240
- **Why it matters**: anchored the lower bound that every subsequent recipe is measured against.

### Recipe 2 — Caption + Marqo-FashionCLIP (Phase-A best)

After an 11-encoder ablation, Marqo-FashionCLIP was the strongest off-the-shelf encoder on the concise caption regime.

- **Encoder**: `hf-hub:Marqo/marqo-fashionCLIP`, 512-dim
- **Code**: [`src/baseline/run_baseline.py`](src/baseline/run_baseline.py) (encoder swapped via the swap script)
- **Launch**: `bash scripts/run_encoder_swap.sh` (sweeps all 11 encoders against the v1/concise baseline; FashionCLIP is the strongest row)
- **R@10**: 0.533
- **Why it matters**: locked the strongest fashion-specific frozen-encoder baseline; defines the gap the trained system must close.

### Recipe 3 — Caption + Qwen3-Embedding-8B (Plan-9 detailed prompt)

Same caption-retrieval shape as Recipe 2 but with a detailed VLM caption and the strongest general-purpose embedder.

- **Encoder**: `Qwen/Qwen3-Embedding-8B`, 4096-dim
- **Code**: [`src/baseline/run_baseline.py`](src/baseline/run_baseline.py) (encoder swapped via the swap script, against the v2 / detailed-caption baseline)
- **Launch**: `SOURCE_RUN=runs/baseline_v2_speechqwen2vl_20260504 bash scripts/run_encoder_swap.sh`
- **R@10**: 0.586
- **Why it matters**: per-encoder × per-prompt interaction — detailed captions help Qwen3 but hurt FashionCLIP-family encoders.

### Recipe 4 — Qwen2VL → frozen FashionCLIP image alignment (Plan-6)

First contrastive recipe. Query tower is a trainable Qwen2VL; target is the frozen FashionCLIP **image** encoder.

- **Query tower**: speechQwen2VL + LoRA + projection head → 512-dim
- **Target tower**: Marqo-FashionCLIP image encoder, frozen
- **Loss**: symmetric multi-positive InfoNCE with cross-GPU `all_gather`
- **Code**: [`src/training/train_plan5.py`](src/training/train_plan5.py)
- **Launch**: `bash scripts/run_plan5.sh --multi-gpu --num-gpus 8 --batch-size 4 --gather` (single-GPU variant: `--batch-size 32` instead)
- **R@10**: 0.402 — **regressed** from the 0.533 Phase-A baseline
- **Why it matters**: the regression revealed the ceiling — the query tower was capped at FashionCLIP's discrimination. Motivated Recipe 5.

### Recipe 5 — Two-tower Qwen2VL, separate backbones (Plan-10 Option B)

Both query and target are trainable Qwen2VL towers. Embedding space co-constructed during training instead of inherited from a frozen teacher.

**In plain terms:** "separate backbones" means **two full Qwen2VL instances live in GPU memory at the same time** — one driving the query tower, one driving the target tower — each with its own LoRA adapter on top. Memory cost: ~2× 7B base weights resident. The training objective (symmetric multi-positive InfoNCE) pulls matching `(query, target)` pairs together in the embedding space and pushes non-matching pairs apart.

- **Both towers**: speechQwen2VL + LoRA + projection head → 512-dim
- **Loss**: symmetric multi-positive InfoNCE with cross-GPU `all_gather`, dynamic database re-encoding at end of every epoch
- **Code**: [`src/training/train_plan10.py`](src/training/train_plan10.py), [`src/training/two_tower_model.py`](src/training/two_tower_model.py) (`TwoTowerSeparateBackbones`)
- **Launch**: `bash scripts/run_plan10.sh --arch separate`
- **R@10**: **0.637** (peak, ckpt_epoch11)
- **Why it matters**: validates the project's core architectural hypothesis; +0.235 absolute R@10 vs Recipe 4, beats every Phase-A baseline. The shared-backbone variant (Recipe 6) later edged ahead at a larger batch.

### Recipe 6 — Two-tower Qwen2VL, shared backbone + 2 LoRA adapters (Plan-10 → 13) ✅ best (text)

Same loss / data / eval as Recipe 5, but with **one shared Qwen2VL backbone** plus two PEFT LoRA adapters toggled via `set_adapter`. Trades adapter cross-talk risk for half the VRAM — which is what later lets the batch (and the score) grow.

**In plain terms (vs Recipe 5):** only **one Qwen2VL base lives in GPU memory** — both towers share it. Each tower still has its own small LoRA adapter attached to that shared base; at forward time we call `set_adapter('query')` or `set_adapter('target')` to enable the right one. Memory: ~1× 7B base (about half of Recipe 5), which is what frees the VRAM to push the batch (and the score) up.

- **The PEFT gradient-checkpointing fix**: the first shared run (Plan-10 Option A) had to *disable* gradient checkpointing (PEFT footgun — checkpoint recompute reads the *current* `active_adapter` rather than the one active during forward; [`Progress_11`](Documentation/Progress_11_20260512.md)), forcing bs=4 → R@10 0.587. Plan-12 replaced it with a **PEFT-aware `context_fn`** that re-enables checkpointing safely → bs=8 (0.623), then bs=24 (Plan-13) → **R@10 0.654**, the project's best text result. Batch size, not architecture, drove ~74% of the earlier shared-vs-separate gap ([`Progress_12`](Documentation/Progress_12_20260513.md)).
- **Code**: [`src/training/two_tower_model.py`](src/training/two_tower_model.py) (`TwoTowerSharedBackbone`)
- **Launch**: `bash scripts/run_plan10.sh --arch shared --batch-size 24`
- **R@10**: **0.654** (Plan-13, bs=24, best text checkpoint)

### Recipe 7 — Two-tower Qwen2VL with native audio query (Plan-15) 🎙️ ✅ flagship

Once Recipe 6 (shared-backbone two-tower) emerged as the best text-side architecture, we added the audio modality **on top of that exact same model**. Recipe 7 is the audio-native variant: identical two-tower architecture, identical training loop, identical shared-backbone + 2 LoRA setup — **the only thing that changes is the query-side modification channel**, which is now a spoken-modification audio waveform consumed directly by speechQwen2VL's audio encoder (no separate ASR step).

- **Both towers**: speechQwen2VL + LoRA + projection head → 512-dim (identical to Recipe 6)
- **Query input**: `(reference image, spoken modification)` — audio tokens fed natively into the backbone
- **Target tower**: identical to Recipe 6 — `(target image, "Describe this image in detail.")`
- **Loss / training**: same symmetric multi-positive InfoNCE with cross-GPU `all_gather`, dynamic end-of-epoch gallery refresh
- **Code**: extension of [`src/training/train_plan10.py`](src/training/train_plan10.py) with an audio collator
- **Launch**: `bash scripts/run_plan10.sh --arch shared --batch-size 32 --query-modality audio`
- **R@10**: **0.624** (Plan-15, dev-selected checkpoint; 0.643 best-checkpoint) — only ~0.03 below the typed-modification Recipe 6, confirming the audio-native query is competitive with text. A 3-way sensitivity probe (real / no-audio / shuffled-audio) confirms the result is genuinely audio-driven. See [`Progress_15`](Documentation/Progress_15_20260518.md).
- **Why it matters**: closes the loop on the Motivation section's "speak as you browse" promise — the trained two-tower system serves both typed and spoken modification queries with zero architectural change.

---

## Replicate caption baseline (deep guide)

For those of you who want to replicate the baseline, this section bundles everything you need to actually run all components of the caption-based baseline pipelines (Recipes 1–3) and understand them. The code lives under `src/` — see [§Baseline code map (`src/`)](#baseline-code-map-src) below for the file-by-file breakdown (data loader, encoder, VLM captioner, retrieve, eval) — and you can run it two ways:

- **CPU-only verification using your local laptop** — see [§CPU run (laptop verification)](#cpu-run-laptop-verification) below. Runs the full pipeline end-to-end with a *mock* (returns fixed strings) or *oracle* (returns the ground-truth target caption) captioner instead of the real ~7B VLM (the real captioner we use), so the whole thing finishes in seconds on CPU. This does **not** produce real R@10 numbers; it just confirms the data → captioner → retrieve → eval wiring works. Use it to read the code while watching it execute, or to catch bugs before spending GPU time on a real run.
- **GPU verification on a GPU server to try reproducing the R@10 numbers (0.240 / 0.533 / 0.586)** — see [§Full single-GPU run on a server](#full-single-gpu-run-on-a-server) below. The real run that swaps in the actual VLM captioner (Qwen2VL) and produces the headline R@10 numbers over the 1,000-query FACap dress slice. Same run that Recipe 1–3's Launch commands kick off, with explicit hardware/env setup and troubleshooting included.

### Baseline code map (`src/`)

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

### CPU run (laptop verification)

**Optional, CPU-only.** If you cloned the repo and want to make sure the code works *before* committing to a real GPU run, this is the path. Smoke runs use mock/oracle VLMs (no real model needed) and complete in seconds; unit tests are checks that live in `tests/` and rerun on every change, to catch when a previously-working component (data loader, encoder, index, eval) silently breaks. **Skip this subsection if you only want to replicate the R@10 numbers, or if you prefer to go straight to using a GPU** — see [§Full single-GPU run on a server](#full-single-gpu-run-on-a-server) below.

The baseline code runs in a dedicated **conda** env (separate from the dataset-exploration `data_exploration/venv/` above):

```bash
conda env create -f environment.yml
conda activate fashion_retrieval
```

Local 8 GB-VRAM laptops can run the mock and oracle backends; the real VLM backends (`qwen2vl`, `speechqwen2vl`) raise a clear `server-only` RuntimeError below 14 GB VRAM and are intended for the GPU server.

#### Whole Pipeline Check / Smoke runs

This section is the lightest sanity check — running the whole pipeline end-to-end with fake (mock/oracle) captioners just to see if anything crashes. It doesn't tell you whether the numbers are right; it tells you whether the wiring (data → captioner → retrieve → eval) is still intact.

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

#### Per Component Checks / Unit tests

This section checks each piece of the pipeline individually (vs the whole-pipeline check above, which runs the pipeline as one) — does the data loader still return the documented dict schema, does the caption DB build still write the right embeddings + metadata, and does the pipeline orchestrator still wire captioner → encoder → retrieve → eval correctly?

- **Data loader (`FacapDataset`)** — each item should look like a 6-key dict with keys `candidate_image_path`, `modification_text`, `target_image_path`, `target_caption`, `target_id`, `candidate_id`. Each `candidate_image_path` and `target_image_path` should resolve to a real cached image on disk. The full dataset should contain at least 50k triplets.
- **Caption DB build (`build_db()`)** — should write an `embeddings.npy` of the right shape with L2-normalized rows, a matching `metadata.jsonl`, and a `config.json` recording encoder + build args + FACap commit (provenance).
- **Pipeline orchestrator (`run_baseline.run()`)** — should wire captioner → encoder → retrieve → eval correctly. With the **oracle** backend, it should hit **perfect Recall@1** (proves the plumbing). With the **mock** backend, it should run to completion and write the expected artifacts.

13 reproducibility cases across the three files. Each file runs as a script (no pytest dependency) or via pytest:

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

### Full single-GPU run on a server

If you prefer to go straight to using a GPU to test if you can replicate the baseline, this section walks you through the actual single-GPU run that produces R@10 = 0.240 (MiniLM) / 0.533 (FashionCLIP) / 0.586 (Qwen3-Emb-8B). The run launches the headline baseline (`speechqwen2vl` backend, 1,000-query FACap dress eval slice) on a GPU box. Unlike the [§CPU run](#cpu-run-laptop-verification) above (mock/oracle captioners on CPU), this real VLM run needs a GPU.

#### Hardware & setup requirements (single-GPU caption-retrieval inference)

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

#### Setup + run

Clone this repo and `speechQwen2VL` as siblings, then run setup once + the baseline:

```bash
# ── First-time setup ─────────────────────────────────────
cd ~/CSprojects   # (or wherever; just keep them siblings)
git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git
git clone https://github.com/ZhuoyuanJiang/fashion-retrieval-agent.git
cd fashion-retrieval-agent

# Heads up: setup_server.sh installs the forked transformers + qwen-vl-utils
# LAST, so sentence-transformers' upstream transformers doesn't override them.
bash scripts/setup_server.sh

conda activate fashion_retrieval
bash scripts/setup_datasets.sh   # FACap + FashionIQ + Fashion200k annotations

# ── Run the baseline ─────────────────────────────────────
# Defaults baked into the script: n_eval=1000, db_size=59082 (full FACap dress
# targets), vlm=speechqwen2vl, run_name=baseline_v1_speechqwen2vl.
# To override, prefix matching env vars before the command, e.g.:
#   N_EVAL=50 RUN_NAME=smoke_real bash scripts/run_baseline_v1.sh   # 50-query smoke
bash scripts/run_baseline_v1.sh
```

`run_baseline_v1.sh` does three things:
1. Pre-fetches the eval slice's reference images into local cache, so the
   main eval loop doesn't stall on image downloads mid-run.
2. Runs the baseline (VLM caption generation + text-to-text retrieval over
   the 59k FACap dress target captions), writing all artifacts to
   `runs/<run_name>/`.
3. Pretty-prints `metrics.json` to the terminal — that's the
   Recall@1/5/10/50, median + mean rank summary. All the other files in the
   output tree below were already written in step 2; only this summary is
   shown in stdout.

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

#### Troubleshooting

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

## Contrastive training (deep guide)

This section bundles the **shared infrastructure** for training any of the contrastive recipes (Recipes 4–7): hardware requirements, env vars, launch scripts, and W&B setup. **All 4 contrastive recipes need the same hardware + env vars** — the per-recipe difference is just the launch flag (see each Recipe's Launch line in [§Recipes](#recipes) above).

The per-Plan subsections below ([§Plan-6](#plan-6-query-tower-contrastive-frozen-fashionclip-target) and [§Plan-10/13](#plan-1013-two-tower-co-trained-best-text)) walk through each Recipe's actual training script — useful if you want to understand or modify the training loop. Skip them if you only need to launch.

### Hardware & setup (multi-GPU contrastive training)

- **Hardware**: ≥ 7× A6000-class GPU (≈ 49 GB VRAM each) for the default Plan-10 config (`--arch separate`, batch 8, 8 GPUs, gather=ON). Option A (shared backbone, `--arch shared`) can run on fewer GPUs at smaller batch sizes — see `scripts/run_plan10.sh` flag overrides.
- **Disk**: ≥ 30 GB free on a fast local SSD for `HF_HOME` — Qwen2-VL-7B-Speech ≈ 17 GB + Stage-2 LoRA ≈ 650 MB + gallery embeddings ≈ 120 MB × 18 epochs.
- **FACap images**: the full ~60 K FACap dress image set must be available at `$FACAP_IMAGES_DIR`. Fetch it via `bash scripts/fetch_artifacts.sh --with-images` (see [§Download original data → Full image downloads](#full-image-downloads-training-time) above).
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

### Plan-10/13: two-tower co-trained (best text)

```bash
# Shared backbone + 2 PEFT LoRA adapters (Plan-13, bs=24 → R@10 0.654 — best text)
bash scripts/run_plan10.sh --arch shared --batch-size 24

# Separate backbones (Plan-10 architecture comparison; dropped side branch)
bash scripts/run_plan10.sh --arch separate
```

Defaults: 8 GPUs, gather=ON, 18 epochs, end-of-epoch gallery refresh. Both towers are trainable; the embedding space is co-constructed. The audio-native variant (Plan-15, R@10 0.624) runs the same launcher with `--query-modality audio`. See [Recipe 6](#recipes) for the full story and the PEFT gradient-checkpointing fix that enabled bs=24.

W&B run names auto-generate as `plan10/v1_<arch>_bs<N>_<G>x<gpu>_<date>` from `torch.cuda.get_device_name()`.

### Eval

Both training scripts run dev + headline retrieval evaluation at every 0.5 epoch automatically. Numbers logged to W&B (`fashion-retrieval-agent` project) and persisted to `runs/<run_name>/metrics.json`.

---

# 📚 Meta

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
- `demo_assets/` — cached data the demo loads at startup:
  ```
  demo_assets/
  ├── preset_cache.json    8 preset queries + cached top-K results per pipeline
  ├── preset_audio/        TTS audio for each preset's spoken modification (~2.4 MB)
  ├── survey.jsonl         past-user feedback notes
  └── preset_thumbs/       gitignored — product image thumbnails the demo gallery
                           shows. Generate locally with
                           `python scripts/make_demo_thumbs.py` (needs FACap images
                           first; run `bash scripts/fetch_artifacts.sh --with-images`).
  ```
- `scripts/` — reproducibility helpers.
- `src/` — baseline implementation (Plan_2 M1–M3); entry point is `src/baseline/run_baseline.py`.
- `tests/` — runnable test suite for M1–M3 (13 cases).
- `runs/` — gitignored: caption DBs, metrics, qualitative dumps.

## Acknowledgments

Special thanks to **Nima Tajbakhsh** (Nvidia) for valuable technical guidance and feedback throughout this project.

---

## Licenses

- FashionIQ: CDLA-Permissive.
- FACap: not stated in upstream repo or project page (clarify before
  redistributing derived artifacts).
- Fashion200k / DeepFashion-MultiModal: see source repos.
- This repo never redistributes third-party image data; all images are fetched
  from upstream at setup time.
