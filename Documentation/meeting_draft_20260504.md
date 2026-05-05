# Meeting Draft - Fashion Retrieval Agent Progress

**Date:** 2026-05-04  
**Scope:** Work completed from dataset exploration through the latest contrastive training and demo planning.

---

## 1. One-Minute Summary

The project goal is an audio-conditioned composed fashion retrieval system:

> reference garment image + spoken/text modification -> retrieve the target fashion item.

Since the dataset exploration phase, I completed four major pieces of work:

1. **Dataset and baseline infrastructure.** I verified FashionIQ and FACap, then built a FACap-based retrieval pipeline with dataset loading, caption DB construction, VLM caption generation, retrieval, evaluation, and tests.
2. **Caption-based baseline and encoder ablation.** The initial MiniLM baseline was weak, but swapping only the text encoder showed that Marqo FashionCLIP reaches R@10 = 0.533 on the 1000-query FACap dress headline split.
3. **Direct contrastive training.** I implemented a Qwen2-VL query encoder trained with multi-positive InfoNCE against frozen FashionCLIP image embeddings. The best contrastive checkpoint reaches headline R@10 = 0.402 and R@50 = 0.646.
4. **First demo path.** I started a Gradio demo that compares three pipelines: caption-based retrieval, contrastive embedding retrieval, and a future native-audio placeholder.

The current technical read is: the contrastive model learns useful composed embeddings and is competitive at R@50, but it still trails the caption-based FashionCLIP baseline at R@10. The next research bottleneck is not more epochs or LR tuning; it is better retrieval supervision, likely hard negatives or reranking.

---

## 2. Meeting Talk Track

This is the version I should present verbally. The key is to make every transition a decision, not just a list of experiments.

1. **Start with the task.** The product goal is reference garment image plus spoken/text modification, then retrieve the target item from a fashion catalog.
2. **Explain the dataset decision.** I checked FashionIQ and FACap. I chose FACap dress first because it gives candidate image, target image, modification text, and detailed target captions, which let me build both a caption-based baseline and a contrastive image-target training setup.
3. **Build a safe caption-based baseline first.** The baseline converts the image-plus-modification query into a generated target caption, embeds that generated caption, and searches a prebuilt database of target-caption text embeddings. This gave us a working end-to-end retrieval/eval system before doing any training.
4. **Diagnose why the first baseline was weak.** The MiniLM run was weak, but there were two possible causes: generated captions might be too short/low-quality, or MiniLM might be the wrong retrieval encoder. Encoder ablation showed encoder choice mattered a lot; caption analysis showed caption length/quality was still a real concern.
5. **Choose baselines carefully.** Marqo FashionCLIP became the primary baseline because it has the best top ranking and a usable image tower for contrastive training. Qwen3-Embedding-8B is still important because it has the best R@50 candidate recall.
6. **Do not over-optimize caption retrieval yet.** The caption path already had decent R@50 with strong encoders, so rather than spend the whole project improving generated caption length/style, I moved to the research question: can direct contrastive training remove the caption bottleneck? This was a prioritization decision, not a claim that the caption baseline had no remaining problems.
7. **Explain the contrastive model.** Qwen2-VL encodes candidate image plus modification text directly into the 512-d FashionCLIP image space. The target database is now image embeddings, not caption embeddings.
8. **Explain the major training fixes.** Plan-6 became the clean run after fixing EOS pooling and duplicate-target false negatives with multi-positive InfoNCE.
9. **Interpret the result.** Plan-6 contrastive does not beat the caption baseline at R@10, but its R@50 is close. That means it often retrieves the correct neighborhood, but the top-50 ordering is not sharp enough.
10. **End with next direction.** The most logical next step is hard negatives or reranking, because the first-stage candidate pool is already reasonably good.

---

## 3. Project Framing

The product-level task is **audio-conditioned composed fashion retrieval**.

Input:

- reference garment image
- user modification query, initially text and eventually speech

Output:

- ranked catalog items matching the reference image after applying the modification

The work has been staged deliberately:

| Stage | Goal | Status |
|---|---|---|
| Dataset exploration | Decide whether FashionIQ / FACap can support the task | Done |
| Caption-based baseline | Build a stable zero-training retrieval pipeline | Done |
| Encoder ablation | Find whether the weak baseline was VLM-bound or encoder-bound | Done |
| Contrastive training | Train Qwen2-VL to output retrieval embeddings directly | Done through Plan-7 diagnostics |
| Demo | Build a usable comparison UI | In progress |
| Native audio | Train or wire audio path directly | Not started |

---

## 4. Dataset Exploration

### FashionIQ

FashionIQ is a standard composed image retrieval dataset:

- triplet format: candidate image, target image, two short modification captions
- categories: dress, shirt, toptee
- useful for standard CIR evaluation
- TTS-friendly because modifications are short and natural

Limitations:

- no target-side detailed product captions
- image URLs come from Amazon metadata and may have long-term URL rot

### FACap

FACap became the main working dataset because it provides exactly what the baseline needed:

- triplets: candidate image, target image, modification text
- target captions: detailed target image descriptions
- large category-level training split
- dress split: 59,082 triplets

Important detail: FACap is within-category. A dress query retrieves another dress; it does not train cross-category transformations like dress -> skirt.

### Dataset Decision

I chose FACap dress as the first serious experimental slice because it gives:

- one clear category
- a large 59k gallery
- paired reference / modification / target supervision
- target captions for the caption-based baseline
- target images for contrastive training

### Dataset Class Construction

The central dataset wrapper is `src/data/facap_dataset.py`.

`FacapDataset` reads two FACap JSON files:

- `cir_triplets/{category}_{split}_triplets.json`
- `image_captions/{category}_{split}_captions.json`

Each item contains the fields needed by both pipelines:

```python
{
    "candidate_image_path": str,
    "modification_text": str,
    "target_image_path": str,
    "target_caption": str,
    "target_id": str,
    "candidate_id": str,
}
```

Important implementation details:

- `target_caption` comes from FACap's precomputed long image captions.
- `modification_text` comes from the triplet caption field.
- `target_id` and `candidate_id` are filename stems, e.g. `51727804_0`.
- Images are not loaded during plain dataset indexing. `load_image(item, "candidate")` or `load_image(item, "target")` opens the cached local JPEG only when needed.

The first four fields preserve the raw FACap triplet content. The ID fields are
derived from the paths; they are not extra supervision. The reason for returning
them is to separate **provenance** from **operational identity**:

```text
candidate_image_path = where FACap says the reference image came from
candidate_id         = the normalized key for that reference image

target_image_path    = where FACap says the target image came from
target_id            = the normalized key for that target image
```

A path can be used as a key if the whole system commits to path strings
everywhere. I chose to expose IDs because retrieval/evaluation naturally works
in terms of image identities:

```text
reference image ID + modification text -> ranked target image IDs
```

Example qualitative row:

```json
{
  "query_id": "12345678_0",
  "true_target": "87654321_0",
  "top10_predicted": ["11111111_0", "87654321_0", "22222222_0"],
  "rank": 2
}
```

This is easier to read and less storage-layout-dependent than carrying full
paths through every artifact. If the image cache layout, directory prefix, or
file extension changes, the path string can change while the image identity
remains `12345678_0`.

| Key | Why it is stored | Where it is used |
|---|---|---|
| `target_id` | Normalized key for the correct answer image | Caption baseline evaluation calls `rank_of(item["target_id"], ...)`; caption DB metadata stores target IDs; contrastive training uses it to look up the frozen FashionCLIP target embedding; multi-positive InfoNCE uses duplicate `target_id`s to build the positive mask. |
| `candidate_id` | Normalized key for the reference/query image | VLM captioning uses it to resolve the local candidate JPEG; qualitative dumps use it as `query_id`; contrastive/dev/headline splitting excludes candidate IDs to avoid train/eval leakage. |

This ID layer matters even before contrastive learning. In the caption baseline,
the database search returns ranked `target_id`s, and evaluation asks whether the
true `target_id` appears in the top-k. The VLM side also needs `candidate_id`
to identify which reference image produced the generated caption and retrieval
result.

In short, this schema keeps three things aligned in one item:

- **query input:** candidate image + `modification_text`
- **text supervision:** `target_caption`
- **retrieval identity:** `candidate_id` / `target_id`

Meeting wording:

> We keep both the raw FACap path and a normalized image ID. The path preserves provenance; the ID is the operational key. Retrieval/evaluation is naturally phrased as "reference image ID plus modification retrieves a ranked list of target image IDs." The ID is derived from the path, not extra supervision.

This design let the same base dataset support:

- text-only caption database building
- VLM caption-generation baseline
- image-loading contrastive training
- qualitative demo examples

---

## 5. Baseline Pipeline: Caption-Based Retrieval

The first implemented method was the safe baseline:

```text
candidate image + modification text
  -> VLM generates target-oriented caption
  -> text encoder embeds generated caption
  -> nearest-neighbor search over target-caption embeddings
  -> ranked retrieved items
```

The important database detail:

> The caption-based baseline searches a database of **text embeddings of FACap target captions**. It does not search image embeddings.

The pipeline has two phases.

### Offline Caption Database Build

This is done once per text encoder.

```text
all FACap target captions
  -> TextEncoder
  -> L2-normalized text embeddings
  -> save embeddings.npy + metadata.jsonl + config.json
```

Code path:

- `src/baseline/build_caption_db.py`
- `src/baseline/text_encoder.py`
- `src/baseline/retrieve.py`

Database files:

| File | Meaning |
|---|---|
| `embeddings.npy` | float32 text embeddings, one row per target caption |
| `metadata.jsonl` | target ID, image path, caption text, row order |
| `config.json` | encoder name, split, FACap commit, build args |

For the production baseline, the DB is full mode: all target captions for the FACap dress split. Subset mode exists only for smoke/debug runs.

### Online Query/Eval Path

For each eval query:

```text
candidate image + modification text
  -> VLM captioner
  -> generated target-style caption
  -> same TextEncoder
  -> query text embedding
  -> cosine search against caption DB text embeddings
  -> rank true target_id
```

Code path:

- `src/baseline/run_baseline.py` orchestrates the full run.
- `make_captioner()` in `src/baseline/vlm_caption.py` selects mock/oracle/Qwen2-VL/speechQwen2VL.
- `TextEncoder.encode()` embeds the generated caption.
- `top_k()` and `rank_of()` in `src/baseline/retrieve.py` do dot-product search over L2-normalized embeddings.
- `src/baseline/eval.py` computes R@1/R@5/R@10/R@50 and median rank.

This is the answer if someone asks: "Where did the database embeddings come from?" For the caption baseline, they came from FACap target captions encoded by the chosen text encoder.

### Code Built

| File | Role |
|---|---|
| `src/data/facap_dataset.py` | FACap triplet dataset, lazy image loading, target/candidate IDs |
| `src/baseline/text_encoder.py` | Sentence-BERT wrapper for caption embeddings |
| `src/baseline/build_caption_db.py` | Builds target-caption embedding DB |
| `src/baseline/vlm_caption.py` | Mock, oracle, Qwen2-VL, speechQwen2VL captioners |
| `src/baseline/retrieve.py` | Cosine top-k and true-rank lookup |
| `src/baseline/eval.py` | Recall@1/5/10/50, median rank, qualitative dump |
| `src/baseline/run_baseline.py` | End-to-end baseline orchestrator |
| `scripts/run_baseline_v1.sh` | Server launch wrapper |

### Tests and Sanity Checks

I added a small test suite covering the core baseline pieces:

- dataset schema and image lookup
- caption DB shape and normalization
- oracle retrieval sanity
- mock pipeline smoke test
- server-only gating for real VLM backends

Oracle sanity reached R@1 = 1.0, which confirms the encode -> retrieve -> rank path is wired correctly.

---

## 6. Server Baseline and Encoder Ablation

### Initial Real VLM Baseline

Server baseline:

- VLM: `speechQwen2VL`
- eval: last 1000 FACap dress train triplets
- gallery: full 59,082 dress targets
- initial encoder: `all-MiniLM-L6-v2`

Result:

| Metric | Score |
|---|---:|
| R@1 | 0.084 |
| R@5 | 0.191 |
| R@10 | 0.240 |
| R@50 | 0.384 |
| Median rank | 155 |

This looked weak, but it was not clear where the failure came from. There were two plausible explanations:

1. The VLM-generated captions might be too short, generic, or missing the fine visual attributes needed for fashion retrieval.
2. The generated captions might be reasonable, but MiniLM might be a weak encoder for matching generated target descriptions to long FACap product captions.

The next two analyses separated these hypotheses:

- **Encoder ablation**: hold generated captions fixed, only swap the retrieval encoder.
- **Caption analysis**: inspect generated caption length/content and correlate it with rank.

### Encoder Ablation

I replayed the same 1000 generated captions through 11 successful encoders, holding VLM output fixed.

Key results:

| Encoder | R@1 | R@10 | R@50 | Median rank |
|---|---:|---:|---:|---:|
| Marqo FashionCLIP | 0.258 | 0.533 | 0.685 | 8 |
| BGE-large | 0.233 | 0.496 | 0.685 | 11 |
| E5-large-v2 | 0.231 | 0.496 | 0.670 | 11 |
| Marqo FashionSigLIP | 0.207 | 0.455 | 0.619 | 16 |
| Qwen3-Embedding-4B | 0.192 | 0.475 | 0.656 | 13 |
| Qwen3-Embedding-8B | 0.174 | 0.522 | 0.704 | 9 |
| MiniLM-L6 | 0.084 | 0.240 | 0.384 | 155 |

Main conclusion:

> The original MiniLM result should not be interpreted as only a caption-generation failure. Holding generated captions fixed and swapping encoders produced a large gain, so encoder choice is an important factor. Caption quality and caption length still remain possible bottlenecks. Marqo FashionCLIP became the main Phase-A top-ranking reference: R@10 = 0.533.

This changed the bar for contrastive learning. The contrastive model should not be compared only to MiniLM R@10 = 0.240; it should be compared to Marqo FashionCLIP R@10 = 0.533.

### Why Marqo FashionCLIP Became the Primary Baseline

Marqo FashionCLIP was not selected because it was the unique winner on every metric. Qwen3-Embedding-8B had the best R@50:

| Encoder | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| Marqo FashionCLIP | 0.258 | 0.533 | 0.685 |
| Qwen3-Embedding-8B | 0.174 | 0.522 | 0.704 |
| BGE-large | 0.233 | 0.496 | 0.685 |

The decision was:

- Use **Marqo FashionCLIP as the primary retrieval baseline** because it gives the strongest top ranking overall: best R@1, best R@10, and median rank 8.
- Also report **Qwen3-Embedding-8B as the best R@50 candidate-recall baseline**, because it finds the correct target somewhere in the top 50 most often.
- Use **FashionCLIP image embeddings as the contrastive target space** because FashionCLIP has an image tower. BGE, E5, and Qwen3-Embedding are text encoders only in this setup; they are useful caption baselines but cannot directly encode target images for Plan-5/6 contrastive training.

So the clean framing is:

- For a user-facing top-10 grid, Marqo FashionCLIP is the main Phase-A baseline.
- For first-stage candidate generation, R@50 matters more, and Qwen3-Embedding-8B is an important comparison.
- For contrastive training, FashionCLIP is the practical frozen target tower.

---

## 7. Caption Analysis

I built a caption analysis notebook to inspect what the VLM was producing.

Important findings:

- generated captions are much shorter than FACap target captions
- median generated caption length: 92 chars
- median FACap target caption length: 554 chars
- median length ratio: about 0.167

Caption agreement correlates with rank, but weakly:

- Spearman rank vs token overlap: rho = -0.306
- Spearman rank vs length ratio: rho = -0.216

Interpretation:

- Caption quality matters.
- But literal caption overlap is far from the whole story.
- Even rank-1 examples only have about 23.8% token overlap with target captions.
- The encoder is doing semantic matching, not just word matching.

Decision after this analysis:

- I did not try to perfect the caption-generation prompt immediately.
- The strong encoders already gave decent R@50, so the caption pipeline was useful as a baseline and candidate generator.
- But the caption route still had a structural bottleneck: it compresses candidate image plus modification into one short generated text string.
- To test whether that bottleneck mattered, I moved to direct contrastive learning instead of spending the next phase only on caption prompt engineering.
- This was a sequencing decision: first see whether contrastive learning can produce a meaningful result, then revisit caption-specific improvements if needed.

---

## 8. Contrastive Training: Goal

The direct contrastive method removes the caption bottleneck:

```text
candidate image + modification text
  -> trainable Qwen2-VL query encoder
  -> 512-d query embedding
  -> cosine search against frozen FashionCLIP image embeddings
```

Target side:

- frozen Marqo FashionCLIP image tower
- precomputed target image embeddings for the full gallery
- cache: about 59k x 512 float32 vectors

Query side:

- Qwen2-VL / speechQwen2VL backbone
- LoRA-adapted LLM decoder
- projection head from hidden state to 512-d FashionCLIP image space

The training objective is to align the query embedding with the correct target image embedding.

The important database detail:

> The contrastive model searches a database of **FashionCLIP image embeddings**. This is different from the caption baseline, which searches text embeddings of target captions.

Contrastive target-cache build:

```text
all FACap gallery images
  -> frozen Marqo FashionCLIP image tower
  -> L2-normalized image embeddings
  -> target_emb_cache_marqo-fashionclip.npy
```

Code path:

- `src/training/target_cache.py` builds and loads the image embedding cache.
- `load_target_cache()` returns embeddings, gallery IDs, and embedding dimension.
- `make_gallery_lookup()` maps `target_id -> frozen image embedding` for training batches.
- `make_gallery_db()` in `src/training/online_eval.py` wraps the image cache so the same retrieval/rank code can be reused.

---

## 9. Contrastive Dataset Construction

I wrapped `FacapDataset` with a contrastive dataset class:

`src/data/contrastive_dataset.py`

It creates:

- training split
- 500-query dev slice
- 1000-query headline slice
- exclusion set to avoid train/eval leakage

Dataset item:

```python
{
    "cand_image": PIL.Image,
    "mod_text": str,
    "target_id": str,
}
```

Important split decision:

- headline slice is the same 1000-query slice used in the baseline
- dev slice is separate and used for online monitoring
- training excludes target IDs and candidate IDs from both dev and headline sets

This prevents the contrastive model from learning eval target images through other triplets.

Final sizes:

| Split | Size |
|---|---:|
| Train | 53,686 |
| Dev | 500 |
| Headline | 1,000 |
| Excluded IDs | 2,663 |

### How It Is Built From `FacapDataset`

Code path:

- `src/data/contrastive_dataset.py`
- class: `FacapContrastiveDataset`
- collate function: `contrastive_collate`

Construction steps:

1. Load the base `FacapDataset`.
2. Reserve the last 1000 triplets as the headline eval slice, matching the caption baseline's eval setup.
3. Collect both candidate IDs and target IDs from that headline slice.
4. Apply L2 filtering: remove any training triplet whose candidate or target image appears in the headline exclusion set.
5. Sample a deterministic 500-query dev slice from the remaining clean pool.
6. Remove dev candidate/target IDs from training too.
7. Save the dev slice JSON in the run directory for reproducibility.

Training items return only what the contrastive loop needs:

```python
{
    "cand_image": PIL.Image,
    "mod_text": str,
    "target_id": str,
}
```

The target image itself is not loaded in each training sample. Instead, the train loop uses `target_id` to look up the frozen FashionCLIP target embedding from the cache. This keeps training cheaper and makes the target tower truly frozen.

---

## 10. Model Architecture and Trainable Parameters

### Base Model

Model:

- `DanJZY/Qwen2-VL-7B-Speech`
- Stage-2 LoRA loaded and merged
- audio path unused for these runs

Why merge Stage-2 LoRA:

- keeps any useful Stage-2 representation changes
- avoids double PEFT adapter wrapping
- gives Plan-5/6 a clean fresh LoRA training surface

### Query Embedding

Forward path:

```text
candidate image + text prompt
  -> Qwen2-VL processor
  -> Qwen2-VL forward with hidden states
  -> last attended token hidden state, dim 3584
  -> projection head: 3584 -> 1024 -> 512
  -> L2 normalize
```

Important bug fixed:

- Qwen uses left padding.
- The old pooling code used `attention_mask.sum() - 1`, which points into padding for shorter sequences.
- Fixed to rightmost attended token with `flip + argmax`.

### Trainable Layers

Frozen:

- Qwen2-VL vision tower
- merged Qwen2-VL base weights
- frozen FashionCLIP image tower
- target embedding cache

Trainable:

- LoRA on Qwen2-VL LLM decoder attention projections:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- projection head
- InfoNCE temperature parameter `logit_scale`

LoRA config:

| Parameter | Value |
|---|---:|
| Rank | 32 |
| Alpha | 64 |
| Dropout | 0 |
| Target modules | q/k/v/o projections |

Projection head:

```text
3584 -> 1024 -> 512
GELU + LayerNorm
```

Approximate trainable parameters:

- LoRA: about 28M
- projection head: about 4.2M
- total: about 32M

---

## 11. Loss Function

### Initial Loss

Plan-5 used symmetric InfoNCE:

```text
q -> target image embedding
target image embedding -> q
```

with:

- learned temperature
- `logit_scale = log(1 / tau)`
- initialized at `log(1 / 0.07)`
- clamped so inverse temperature <= 100

### Multi-Positive Fix

FACap has many duplicate target IDs:

- 53,686 train triplets
- 28,252 unique target images

That means different queries can legitimately point to the same target image. Diagonal-only InfoNCE incorrectly treats same-target items as negatives.

Plan-6 fixed this with multi-positive InfoNCE:

```text
positive(i, j) = target_id[i] == target_id[j]
```

For each query:

```text
loss_i = logsumexp(all targets) - logsumexp(all positives)
```

This also gathers `target_ids` across GPUs, so cross-rank duplicates are handled correctly.

---

## 12. Training Loop

Implemented in:

`src/training/train_plan5.py`

Main loop:

```text
for batch:
    load candidate images and modification text
    look up frozen target embeddings from cache
    encode query with Qwen2-VL + projection head
    compute symmetric multi-positive InfoNCE
    backward with Accelerate
    AdamW step
    clamp logit_scale
    periodically run dev/headline eval
    save checkpoint at epoch end
```

Code review pointers:

| File | What to show |
|---|---|
| `src/training/train_plan5.py` | dataloader, cache loading, optimizer groups, eval/checkpoint loop |
| `src/training/contrastive_model.py` | Qwen2-VL forward path, prompt, pooling, projection head, LoRA targets |
| `src/training/loss.py` | symmetric multi-positive InfoNCE and cross-GPU gather |
| `src/training/online_eval.py` | dev/headline retrieval eval and sensitivity probes |
| `src/training/target_cache.py` | frozen FashionCLIP image cache construction |

Distributed training:

- HuggingFace Accelerate
- DDP across 8 GPUs
- optional `--gather` for global in-batch negatives
- effective batch = per-GPU batch x number of GPUs

Optimizer:

| Param group | LR |
|---|---:|
| LoRA | 2e-5 |
| Projection head | 1e-4 |
| logit_scale | 2e-5 |

Scheduler:

- Plan-6 used constant LR
- Plan-7 tested cosine LR
- cosine is now opt-in, not default

Evaluation:

- dev slice: 500 queries
- headline slice: 1000 queries
- metrics: R@1, R@5, R@10, R@50, median rank
- sensitivity probe:
  - normal query
  - stripped modification
  - shuffled modification

The sensitivity gap stayed strongly positive in good runs, confirming that the model uses the modification text rather than doing pure visual nearest-neighbor retrieval.

---

## 13. Engineering Bugs Found and Fixed

Major issues fixed during training:

| Issue | Cause | Fix |
|---|---|---|
| Wrong conda env | `accelerate` resolved outside `fashion_retrieval` | use correct env / wrapper |
| All ranks loaded on GPU 0 | wrong `device_map` semantics | use per-rank device map |
| PEFT adapter loaded to GPU 0 | used `device_map` instead of `torch_device` | pass `torch_device` |
| DDP bucket mismatch | gradient checkpointing interaction | `use_reentrant=False`, `find_unused_parameters=True` |
| `lm_head` OOM | logits `(B, seq, vocab)` allocated but unused | replace `lm_head` with 1-output stub |
| EOS pooling bug | left padding plus `sum - 1` | rightmost attended token |
| duplicate target false negatives | diagonal-only InfoNCE | multi-positive mask |
| rank-0-only clamp | local data mutation not synced | clamp on all ranks |
| missing headline metrics | only logged to W&B | save in `metrics.json` |
| epoch logging bug | double-counted epoch after epoch 0 | `global_step / steps_per_epoch` |

These fixes are part of the reason Plan-6 is the clean contrastive run.

---

## 14. Contrastive Experiment Results

### Plan-5: First Implementation Runs

Plan-5 proved the infrastructure worked but trained under two major bugs:

- wrong EOS pooling
- duplicate target false negatives

Best meaningful Plan-5 result before clean fixes:

| Run | Effective batch | Best dev R@10 | Notes |
|---|---:|---:|---|
| bs=8, 8x3090 | 64 | 0.226 | plateaued early |
| bs=64, 8xA6000 | 512 | 0.240 | trained under pooling bug |

Re-evaluating one checkpoint after the pooling fix showed a large jump:

- dev R@10 `0.212 -> 0.310`

This confirmed the pooling bug was suppressing performance.

### Plan-6: Clean Run

Plan-6 trained from epoch 0 with:

- fixed EOS pooling
- multi-positive InfoNCE
- constant LR

Results:

| Run | Effective batch | Best headline R@10 | Best headline R@50 | Interpretation |
|---|---:|---:|---:|---|
| server 10, bs=64 x 8 | 512 | 0.402 | 0.646 | best contrastive checkpoint |
| server 6, bs=16 x 8 | 128 | 0.387 | 0.641 | overfit after epoch 10 |

Best checkpoint:

```text
runs/plan5/run_bs64_8xA6000_plan6_20260503_011214/ckpt_epoch16
```

Key read:

- Plan-6 substantially improves over Plan-5.
- It still does not beat Phase-A Marqo R@10 = 0.533.
- It gets close on R@50: 0.646 vs 0.685.

### Plan-7: Dev Loss and Cosine LR

Plan-7 added:

- fixed-set dev InfoNCE loss
- optional cosine LR schedule

Server 10 result:

| Run | Best headline R@10 | Notes |
|---|---:|---|
| Plan-6 constant LR | 0.402 | best |
| Plan-7 cosine LR | 0.378 | worse |

Dev loss decreased monotonically, which means the model was not clearly overfitting on server 10. The plateau is more likely caused by saturated random in-batch negatives.

Conclusion:

- keep dev/loss logging
- keep cosine LR as optional
- default back to constant LR

---

## 15. Current Interpretation

The strongest current results depend on the metric:

| Method | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| Caption + Marqo FashionCLIP | 0.258 | 0.533 | 0.685 |
| Caption + Qwen3-Embedding-8B | 0.174 | 0.522 | 0.704 |
| Contrastive Qwen2-VL, Plan-6 | 0.111 | 0.402 | 0.646 |

Interpretation:

- Caption-based retrieval is still better at top-10 ranking.
- Marqo FashionCLIP is the primary baseline for top-ranking quality.
- Qwen3-Embedding-8B is the strongest R@50 candidate-recall baseline.
- Contrastive retrieval learns useful composed embeddings.
- Contrastive R@50 is close to the caption baselines, especially compared with its R@10 gap.
- The contrastive model is not useless; it retrieves the right neighborhood but lacks top-10 precision.
- More epochs and simple LR scheduling are unlikely to close the gap.

Likely next research levers:

1. hard-negative mining
2. reranking top-50 results
3. stronger supervision, e.g. image + caption multi-positive targets
4. better pooling or query representation ablations

---

## 16. Demo Work

I started Plan-8 as the first user-facing artifact.

Goal:

- one UI
- same query
- compare pipelines side by side

Pipelines:

| Pipeline | Status | Description |
|---|---|---|
| P1 caption-based | cached in v0.1 | speechQwen2VL caption + Marqo FashionCLIP retrieval |
| P2 contrastive | cached in v0.1, live later | Plan-6 Qwen2-VL contrastive checkpoint |
| P3 native audio | placeholder | future model that skips ASR |

The demo has 8 curated presets:

- 5 examples where contrastive P2 beats caption P1
- 2 examples where caption P1 wins
- 1 deliberate P2 failure

This lets the demo tell an honest story: contrastive retrieval is useful and sometimes better, but not globally stronger than the caption baseline yet.

---

## 17. Code Review Map

If the mentor wants to inspect code, I should guide the review in this order:

| Topic | File / symbol | What it proves |
|---|---|---|
| Base dataset schema | `src/data/facap_dataset.py`, `FacapDataset.__getitem__` | exact fields returned: candidate path, modification text, target caption, target ID |
| Image loading | `FacapDataset.load_image()` | images are loaded lazily from local cache |
| Caption DB source | `src/baseline/build_caption_db.py`, `build_db_full()` | database rows come from FACap target captions |
| Caption embeddings | `src/baseline/text_encoder.py`, `TextEncoder.encode()` | text embeddings are L2-normalized and reused for query and DB |
| Caption retrieval | `src/baseline/retrieve.py`, `top_k()` / `rank_of()` | cosine search is dot product over normalized text embeddings |
| Baseline orchestration | `src/baseline/run_baseline.py`, `run()` | full query path from VLM caption to retrieval metrics |
| Contrastive split | `src/data/contrastive_dataset.py`, `FacapContrastiveDataset` | headline/dev/train split and leakage filtering |
| Image target cache | `src/training/target_cache.py`, `build_target_cache()` / `load_target_cache()` | contrastive gallery embeddings come from FashionCLIP image tower |
| Query encoder | `src/training/contrastive_model.py`, `ContrastiveQwen2VL.forward()` | candidate image plus text becomes a 512-d query vector |
| Loss | `src/training/loss.py`, `SymmetricInfoNCE.forward()` | multi-positive mask handles duplicate target IDs |
| Training loop | `src/training/train_plan5.py` | how batches, optimizer, eval, W&B, and checkpoints connect |
| Online eval | `src/training/online_eval.py` | dev/headline R@K and modification-sensitivity probes |

This ordering starts from data, then baseline retrieval, then contrastive training. It avoids jumping into the training loop before the reviewer understands what each embedding database contains.

---

## 18. Suggested Meeting Narrative

### Opening

"I started by validating the data and building a caption-based retrieval baseline. Then I found that the weak MiniLM baseline was mostly encoder-bound, so I established a strong Marqo FashionCLIP baseline. After that I implemented direct contrastive training for Qwen2-VL as a composed retriever. It improved substantially after fixing pooling and false-negative issues, reaching R@10 0.402, but still trails the caption baseline at R@10 0.533. I now have enough results to motivate either a demo-focused next step or a hard-negative/reranking research next step."

### What I would emphasize

1. **Engineering foundation is solid.** Dataset, DB, retrieval, eval, tests, distributed training, checkpointing, and demo cache all exist.
2. **Encoder ablation changed the baseline.** MiniLM was not enough; Marqo FashionCLIP is the real bar.
3. **Contrastive training works but has a ceiling.** It gets close at R@50 but not R@10.
4. **The remaining gap is about ranking precision.** It is not solved by more epochs or cosine LR.
5. **The demo can now show a nuanced story.** P1 is stronger overall; P2 is meaningful and wins on selected examples.

---

## 19. Questions for Mentor

1. Should the next research step be hard-negative mining / reranking, or should I prioritize demo and final presentation polish?
2. Is R@50 competitiveness enough to justify the contrastive route as a useful intermediate result?
3. For the final story, should I frame caption-based retrieval as the main method and contrastive as a research extension, or present them as two competing pipelines?
4. Should native audio still be attempted, or should audio be handled through Whisper ASR for the final demo?
5. For writeup, is FACap dress-only acceptable, or should I add FashionIQ cross-dataset evaluation?

---

## 20. Source Documents

- `Documentation/Progress_1_20260420.md` - dataset exploration
- `Documentation/Progress_2_20260420.md` - local baseline scaffolding
- `Documentation/Progress_3_20260430.md` - real VLM baseline + encoder ablation
- `Documentation/Progress_4_20260501.md` - caption analysis notebook
- `Documentation/Progress_5_20260502.md` - first contrastive implementation and debugging
- `Documentation/Progress_6_20260503.md` - clean contrastive run
- `Documentation/Progress_7_20260503.md` - dev loss + cosine LR results
- `Documentation/Progress_8_20260503.md` - demo app v0.1
- `Documentation/encoder_swap_table.md` - encoder ablation table
