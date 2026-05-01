# Server handoff — Plan_3 baseline run

This document orients an agent (or human) on a GPU server to the
fashion-retrieval-agent project and walks through the steps to
produce the headline baseline number.

## What this project is

**Stage 2** of a two-stage research project:

- **Stage 1** (separate repo, Stage-1 model already trained):
  `https://github.com/ZhuoyuanJiang/speechQwen2VL` — Qwen2-VL-7B
  fine-tuned with a Whisper encoder + MLP projector + LoRA so it can
  take audio input. The trained weights are on HuggingFace as
  `DanJZY/Qwen2-VL-7B-Speech` (base) and `DanJZY/Qwen2-VL-7B-Speech-LoRA`
  (Stage-2 LoRA adapter).
- **Stage 2** (this repo): use Stage-1's model for **composed fashion
  retrieval** — given a reference garment image + a (text or
  eventually spoken) modification, retrieve the matching item from a
  catalog. Plan_2 builds the **text-modification caption-generation
  retrieval baseline**; this hand-off is for running it on a GPU box.

For the full project plan and milestone history, see:
- `Documentation/Plan_2_20260427.md` — local-scaffolding plan (M1–M4)
- `Documentation/Progress_2_20260420.md` — implementation log + system
  design diagram

## Prerequisites on the server

1. **Conda** available and on PATH.
2. **NVIDIA GPU** with **>= 14 GB VRAM** at bf16 (the speechQwen2VL
   backend loads Qwen2-VL-7B at bf16, occupying ~15 GB once image
   tokens are in the mix). Single GPU is fine; we don't need multi-GPU
   for this run.
3. **HuggingFace auth** (`HF_TOKEN`) set up so model downloads from
   `DanJZY/Qwen2-VL-7B-Speech` and `Marqo/fashion200k` don't
   rate-limit on a 1000-query run. Check with `huggingface-cli whoami`.
4. **Disk space**: budget ~20 GB for the Qwen2-VL-7B base + Stage-2
   LoRA + this repo's clones + caption DB artifacts. If `~` is tight,
   set `HF_HOME` to a scratch path before running setup.

## Getting set up

Two clones side by side, then one setup script.

```bash
# 1. clone both repos as siblings
cd ~/CSprojects   # (or wherever you keep code; just make them siblings)
git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git
git clone https://github.com/ZhuoyuanJiang/fashion-retrieval-agent.git
cd fashion-retrieval-agent

# 2. one-shot setup: creates conda env, installs deps, installs forks LAST
bash scripts/setup_server.sh
```

`setup_server.sh` will:

1. Verify (or auto-clone) the sibling `speechQwen2VL` repo.
2. Create a `fashion_retrieval` conda env from `environment.yml`
   (Python 3.10.18, cuda-toolkit 12.1.1, ffmpeg, libsndfile).
3. Install `requirements.txt` — a hand-curated **superset** of
   speechQwen2VL's pip deps, plus `sentence-transformers` and
   `requests` for the retrieval side.
4. Shell out to `~/CSprojects/speechQwen2VL/scripts/setup_forks.sh`
   to install the **forked** `transformers` (4.56.0.dev0 from
   `ZhuoyuanJiang/transformers@speech-qwen2vl`) and `qwen-vl-utils`
   in editable mode. They are installed **last** so they override the
   upstream `transformers` that `sentence-transformers` pulled in.
5. Run a verification block printing `torch.cuda.is_available()`,
   transformers version (should be `4.56.0.dev0`), peft, and
   sentence-transformers versions.

Activate the env in your shell:

```bash
conda activate fashion_retrieval
```

## About the existing `speech_qwen2vl` env on this server

If a `speech_qwen2vl` conda env already exists from prior Stage-1 work,
you have two reasonable options:

- **Keep it** if Stage-1 work might be revisited (re-evaluation,
  ablations, paper figures). Cost: ~13 GB disk. The new
  `fashion_retrieval` env will live alongside it.
- **Delete it** (`conda env remove -n speech_qwen2vl`) if Stage-1 is
  effectively shipped (the Stage-1 model is on HuggingFace, no
  re-training planned). Saves ~13 GB. The reverse is expensive
  (re-installing flash-attn alone takes ~10 minutes).

**Note:** `fashion_retrieval` is a true **superset** of `speech_qwen2vl`
in terms of the pip deps it installs (same pins, plus
sentence-transformers and requests). So *running speechQwen2VL Stage-1
inference* would technically work in `fashion_retrieval` too, except
for the audio system libs (`ffmpeg`, `libsndfile`) which we **do**
include via the conda layer. Stage-1 *training* would also work
because we bring in `bitsandbytes`, `flash-attn`, `accelerate`, `trl`.

So if disk is tight, deleting `speech_qwen2vl` is safe assuming the
trained Stage-1 model is preserved on HuggingFace — which it is.

## Where data and model weights live

Two paths the script needs to know about. Defaults work but you may
want to override on a server with separate scratch storage.

| Path                | Default                                         | What it is                               |
|---------------------|-------------------------------------------------|------------------------------------------|
| `HF_HOME`           | `~/.cache/huggingface`                          | Where Qwen2-VL-7B + LoRA download to    |
| Image cache         | `data_exploration/datasets/facap-images/`       | Where Marqo/fashion200k images cache    |
| FACap annotations   | `data_exploration/datasets/facap-repo/`         | Cloned by `scripts/setup_datasets.sh`   |

If your server has a scratch disk (e.g. `/scratch/$USER/`), set
`HF_HOME` before running:

```bash
export HF_HOME=/scratch/$USER/hf_cache
```

For FACap annotations, if not already cloned by an earlier setup pass,
run:

```bash
bash scripts/setup_datasets.sh
```

This clones FashionIQ + FACap + Fashion200k repos (annotations only,
~240 MB; no images).

## Running the baseline

One command. The script orchestrates `prepare_images` → `run_baseline`
→ pretty-print metrics.

```bash
bash scripts/run_baseline_v1.sh
```

Defaults: `n_eval=1000`, `db_size=59082` (full FACap dress targets),
`vlm=speechqwen2vl`, `run_name=baseline_v1_speechqwen2vl`.

To override (e.g., quick smoke first):

```bash
N_EVAL=50 RUN_NAME=smoke_real bash scripts/run_baseline_v1.sh
```

Outputs land in `runs/<run_name>/`:

```
runs/baseline_v1_speechqwen2vl/
  caption_db/
    embeddings.npy        (N=59082, dim=384) float32
    metadata.jsonl        target_id, image_path, caption per row
    config.json           encoder name + build_args + facap_commit_sha
  metrics.json            Recall@1/5/10/50, median rank, mean rank
  qualitative/
    results.jsonl         per-query top-10 + generated caption + true rank
                          (failure_category field starts blank, fill by hand
                           from Plan_2's rubric)
```

## What to produce after the run

Write `Documentation/Baseline_1_Report_<YYYYMMDD>.md` containing:

1. The metrics table (R@1/5/10/50, median rank, mean rank).
2. 5 qualitative success cases — show reference image, modification
   text, retrieved target image. All visually correct.
3. 5 qualitative failure cases with `failure_category` filled in from
   Plan_2's rubric:
   - `caption_wrong` — VLM caption misses the modification
   - `embedding_mismatch` — caption is fine but SBERT places it far from target
   - `dataset_ambiguity` — multiple plausible targets exist
   - `visual_nuance_lost` — modification is too subtle for text to capture
4. The `caption_db/config.json` of the run + the exact command used.
5. One-paragraph interpretation: is this baseline strong or weak?
   Strong → contrastive learning (Plan_4) needs a more compelling angle.
   Weak → contrastive learning has clear motivation.

This report is the input to the next planning round.

## Optional: presentation notebook

`notebooks/baseline_demo.ipynb` is a portable end-to-end demo of the
same pipeline — auto-detects Colab vs server, runs the same
operations, renders metrics and qualitative samples inline. Open with
`jupyter lab notebooks/baseline_demo.ipynb` to execute interactively.
The committed version has placeholder outputs; running it fills them
in.

## Troubleshooting

- **`RuntimeError: server-only: ... needs >= 14.0 GB VRAM`** — the
  selected GPU has < 14 GB VRAM. Use a different `CUDA_VISIBLE_DEVICES`.
- **Forks not overriding upstream transformers** — if
  `python -c "import transformers; print(transformers.__version__)"`
  prints `5.x` instead of `4.56.0.dev0`, the `setup_forks.sh` step
  didn't run or failed silently. Re-run:
  `bash ../speechQwen2VL/scripts/setup_forks.sh`
- **HF download stalls or 429-rate-limits** — set `HF_TOKEN` env var.
  See `huggingface-cli login`.
- **Stale-DB error** — `runs/<run_name>/caption_db/` was built with
  different args (encoder, eval size) than this run. Either delete the
  run directory or use a fresh `--run-name`. The error message names
  the offending arg(s).
