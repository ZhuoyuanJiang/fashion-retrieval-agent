# fashion-retrieval-agent

Research workspace for an audio-conditioned composed fashion retrieval system:
given a reference garment image and a spoken modification request (e.g. "make
it black", "shorter sleeves"), retrieve the matching item from a fashion
catalog. The current focus is dataset exploration and method scoping; see
`Documentation/` for proposals, plans, and progress logs.

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

### Prerequisites

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
