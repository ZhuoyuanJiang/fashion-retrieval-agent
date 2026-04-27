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

## Repo structure

- `Documentation/` — proposals, plans, progress reports, meeting memos.
- `data_exploration/` — inspection notebook, sample fetchers, scratch space.
- `scripts/` — reproducibility helpers.

## Licenses

- FashionIQ: CDLA-Permissive.
- FACap: not stated in upstream repo or project page (clarify before
  redistributing derived artifacts).
- Fashion200k / DeepFashion-MultiModal: see source repos.
- This repo never redistributes third-party image data; all images are fetched
  from upstream at setup time.
