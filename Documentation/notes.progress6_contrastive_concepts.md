# Notes: Plan-6 contrastive concepts

Personal study notes for Plan-6 concepts. Companion file:
`notes.progress5_contrastive_learning.md` (Plan-5 study notes).

---

## Q0: Why did we design the `FacapDataset` item schema this way?

**Context:** The dataset item has more fields than any single pipeline needs:

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

This was deliberate. The point was to keep both kinds of FACap supervision
available in one canonical dataset object:

- **text supervision:** `target_caption`
- **image retrieval supervision:** `target_id` / `target_image_path`
- **query input:** candidate image + `modification_text`

That let the same base dataset support the caption baseline, contrastive
training, evaluation, debugging, and demo examples without reparsing FACap in
each pipeline.

### Field-by-field usage

| Key | Why it exists | How it is used later |
|---|---|---|
| `candidate_image_path` | FACap-relative path of the reference image | Provenance/debug. The actual image load path is resolved through `candidate_id` and the local image cache. |
| `modification_text` | User/query-side modification instruction | Caption baseline sends it to the VLM with the candidate image; contrastive training sends it to Qwen2-VL as text input; eval probes strip/shuffle it. |
| `target_image_path` | FACap-relative path of the correct target image | Provenance/debug and image lookup. It records which image is the ground-truth answer. |
| `target_caption` | FACap's long caption for the target image | Caption-based baseline database source: all target captions are embedded into `embeddings.npy`. |
| `target_id` | Filename-stem ID of the correct target | Evaluation uses it to compute true rank; contrastive training uses it to look up the frozen FashionCLIP target embedding; multi-positive InfoNCE uses it to identify duplicate targets. |
| `candidate_id` | Filename-stem ID of the reference image | Candidate image cache lookup; train/eval leakage filtering; qualitative/debug/demo display. |

### Pipeline-specific view

Caption-based baseline:

```text
Offline DB:
target_caption
  -> text encoder
  -> target-caption text embedding database

Online query:
candidate image + modification_text
  -> VLM-generated target caption
  -> same text encoder
  -> search target-caption text embedding database
  -> evaluate by target_id
```

Contrastive training:

```text
Query side:
candidate image + modification_text
  -> Qwen2-VL query encoder
  -> query embedding

Target side:
target_id
  -> lookup frozen FashionCLIP image embedding from cache
  -> InfoNCE target vector
```

Leakage filtering:

```text
candidate_id and target_id
  -> exclude headline/dev image IDs from training
```

So the schema is not just an arbitrary list of fields. It keeps the text
target, image target, and query input aligned under one item.

Code:

- `src/data/facap_dataset.py`: `FacapDataset.__getitem__`
- `src/baseline/build_caption_db.py`: uses `target_caption`
- `src/training/contrastive_dataset.py`: uses candidate image, `mod_text`,
  `target_id`, `candidate_id`

---

## Q1: What is EOS pooling, and what was the bug?

**Context:** Plan-5 used `sum-1` to find the EOS position. Plan-6 switched
to `flip+argmax`. The same checkpoint went from dev R@10=0.212 → 0.310
purely from this change.

**Answer:**

After the VLM processes a query (candidate image + modification text), the
final hidden layer produces a tensor of shape `(B, seq_len, 3584)` — one
3584-dim vector per token. We only want the vector at the last *real* token
(the EOS token). Everything downstream (projection head, cosine similarity)
depends on reading this correctly.

Qwen2-VL pads on the **left** side. So in a batch padded to length 6:

```
Position:   0      1      2      3      4      5
Seq A:   [tok1] [tok2] [tok3] [tok4] [tok5] [EOS]   (no pad)
Seq B:   [PAD]  [PAD]  [tok1] [tok2] [tok3] [EOS]   (2 pads on left)
Seq C:   [PAD]  [PAD]  [PAD]  [PAD]  [tok1] [EOS]   (4 pads on left)
```

Attention masks (1 = real token, 0 = PAD):

```
A: [1, 1, 1, 1, 1, 1]
B: [0, 0, 1, 1, 1, 1]
C: [0, 0, 0, 0, 1, 1]
```

**Bug — `sum-1`:** `sum(attention_mask) - 1` counts real tokens and
subtracts 1. This works for right-padding (no pads → index = last position),
but fails for left-padding:

| Seq | sum(mask) - 1 | Position read | Actually is |
|-----|--------------|---------------|-------------|
| A   | 6 - 1 = 5   | 5             | EOS ✓       |
| B   | 4 - 1 = 3   | 3             | tok2 ✗      |
| C   | 2 - 1 = 1   | 1             | PAD ✗       |

B lands on a middle token; C lands on a PAD. The model was learning, but
its output embedding was read from the wrong position throughout all of Plan-5.

**Fix — `flip+argmax`:** Reverse the mask row, find the first `1`
(which is the last `1` in the original), then map back to the original index:

```python
seq_ends = (seq_len - 1
            - inputs["attention_mask"].flip(dims=[1]).long().argmax(dim=1))
```

Worked out:

```
flip(A): [1, 1, 1, 1, 1, 1] → argmax=0 → 5-0 = 5 ✓
flip(B): [1, 1, 1, 1, 0, 0] → argmax=0 → 5-0 = 5 ✓
flip(C): [1, 1, 0, 0, 0, 0] → argmax=0 → 5-0 = 5 ✓
```

All three sequences now correctly point to position 5 (EOS).

**Code breakdown — what each operation does:**

```python
seq_ends = (seq_len - 1
            - inputs["attention_mask"].flip(dims=[1]).long().argmax(dim=1))
```

Starting from the attention_mask `(3, 6)`:

```
A: [1, 1, 1, 1, 1, 1]
B: [0, 0, 1, 1, 1, 1]
C: [0, 0, 0, 0, 1, 1]
```

**`.flip(dims=[1])`** — reverse each row left-to-right. The last real token
in the original becomes the *first* real token after flipping:

```
A: [1, 1, 1, 1, 1, 1]
B: [1, 1, 1, 1, 0, 0]
C: [1, 1, 0, 0, 0, 0]
```

**`.long()`** — cast to int64. No value changes; ensures argmax gets
integers (attention_mask can be stored as bool or int8).

**`.argmax(dim=1)`** — for each row, return the index of the first maximum
value. Since all values are 0 or 1, this returns the index of the first `1`
in each flipped row:

```
Flipped A: [1, 1, 1, 1, 1, 1] → first 1 at index 0
Flipped B: [1, 1, 1, 1, 0, 0] → first 1 at index 0
Flipped C: [1, 1, 0, 0, 0, 0] → first 1 at index 0
→ result: [0, 0, 0]
```

This is the *flipped-space* index of the last real token. Convert back to
the original position with the identity `p = seq_len - 1 - flipped_index`:

```
A: 6 - 1 - 0 = 5 ✓
B: 6 - 1 - 0 = 5 ✓
C: 6 - 1 - 0 = 5 ✓
```

`seq_ends = [5, 5, 5]` — EOS position for every sequence.

Why argmax works: after flipping, "rightmost real token in original" becomes
"leftmost real token in flipped." `argmax` on a 0/1 tensor always picks the
first `1`, so it reliably finds the leftmost real token in the flipped mask =
the rightmost real token in the original.

Code: `src/training/contrastive_model.py` lines 174–175.

---

## Q1.5: Does the Qwen forward pass still call `lm_head`, and do we use it?

**Context:** Plan-6 uses Qwen2-VL as a query encoder, not as a text generator.
This created confusion because the model still has an `lm_head`, but the
retrieval embedding is computed from hidden states.

**Short answer:** `lm_head` is still called inside the HuggingFace VLM
forward, but the original LM-head weights are replaced by a cheap dummy head.
The dummy logits exist in `outputs.logits`, but the retrieval path discards
them and uses `outputs.hidden_states[-1]` instead.

### Original Qwen2-VL path

For a normal causal language model forward:

```text
image + text
  -> transformer backbone
  -> hidden_states: (B, seq_len, 3584)
  -> original lm_head: Linear(3584 -> vocab_size)
  -> logits: (B, seq_len, vocab_size)
```

`lm_head` is the next-token prediction head. It maps each token hidden state
to vocabulary logits for text generation.

### What HuggingFace still does internally

The local `transformers` implementation of
`Qwen2VLForConditionalGeneration.forward()` computes logits like this:

```python
outputs = self.model(...)
hidden_states = outputs[0]
logits = self.lm_head(hidden_states)
```

So when Plan-6 calls:

```python
outputs = self.vlm(
    **inputs,
    output_hidden_states=True,
    use_cache=False,
)
```

the HuggingFace wrapper still calls `self.lm_head(hidden_states)`.

### What Plan-6 changes

Plan-6 replaces the original vocab-sized LM head at model construction time:

```python
_lm_head = vlm.base_model.model.lm_head
vlm.base_model.model.lm_head = nn.Linear(
    QWEN2VL_HIDDEN_DIM, 1, bias=False, dtype=torch.bfloat16
).to(next(_lm_head.parameters()).device)
del _lm_head
```

After this replacement:

```text
original lm_head: Linear(3584 -> vocab_size)
dummy lm_head:   Linear(3584 -> 1)
```

The `lm_head` attribute still exists in the model object, but it is no longer
the original vocabulary projection. The original LM-head weights are not used
in forward or training.

### What the retrieval path actually uses

Plan-6 ignores `outputs.logits` and reads hidden states:

```python
last_hs = outputs.hidden_states[-1]  # (B, seq_len, 3584)
seq_len = inputs["attention_mask"].shape[1]
seq_ends = (
    seq_len - 1
    - inputs["attention_mask"].flip(dims=[1]).long().argmax(dim=1)
)
pooled = last_hs[torch.arange(B, device=device), seq_ends, :]  # (B, 3584)
emb = self.proj(pooled.float())                                # (B, 512)
emb = F.normalize(emb, dim=-1)
```

The architecture is therefore best drawn as two branches after the transformer
hidden states:

```text
image + text
  -> transformer backbone
  -> hidden_states: (B, seq_len, 3584)
      ├── dummy lm_head: Linear(3584 -> 1)
      │     -> dummy logits: (B, seq_len, 1)     # discarded
      └── retrieval path:
            select rightmost non-pad token       # (B, 3584)
            -> projection MLP 3584 -> 1024 -> 512
            -> L2-normalized query embedding     # (B, 512)
```

The projection MLP is **not** attached after `lm_head`. It is attached after
the selected final hidden state.

### Why keep any `lm_head` at all?

Because Plan-6 uses the standard HuggingFace causal-LM wrapper, whose forward
implementation computes `logits = self.lm_head(hidden_states)`. Removing the
module entirely would break that wrapper unless we rewrote the forward call to
go deeper into the base transformer.

The practical engineering compromise:

- keep the HuggingFace forward path intact
- replace the expensive vocab head with a cheap dummy head
- discard dummy logits
- use hidden states for retrieval

This avoids allocating the original `(B, seq_len, vocab_size)` logits tensor,
which was a major memory cost and was never needed for retrieval.

Code:

- `src/training/contrastive_model.py` lines 90–97: replace original `lm_head`
- `src/training/contrastive_model.py` lines 163–181: call `self.vlm`, ignore
  `outputs.logits`, pool `outputs.hidden_states[-1]`, and apply projection MLP

---

## Q1.6: What is the current contrastive model architecture?

**Context:** It is easy to confuse the Qwen2-VL language-model wrapper,
the dummy `lm_head`, and the retrieval projection head. They are not the
same component.

### Architecture summary

```text
ContrastiveQwen2VL
├── vlm: Qwen2-VL-7B-Speech
│   ├── Stage-2 LoRA: loaded and merged into base weights
│   ├── fresh Plan LoRA: q_proj/k_proj/v_proj/o_proj, r=32, alpha=64
│   └── lm_head: original vocab head replaced by dummy Linear(3584 -> 1)
├── proj: projection MLP
│   └── 3584 -> 1024 -> 512
└── output: L2-normalized 512-d query embedding
```

Target side is not inside the trainable model:

```text
FashionCLIP image tower
  -> precompute all gallery image embeddings
  -> save target_emb_cache_marqo-fashionclip.npy
  -> training/eval only look up frozen vectors
```

### Forward path

```text
candidate image + modification text
  -> Qwen2-VL processor / chat template
  -> Qwen2-VL transformer hidden states: (B, seq_len, 3584)
  -> choose rightmost non-pad token: (B, 3584)
  -> projection head: (B, 512)
  -> L2 normalize
  -> cosine retrieval against FashionCLIP image cache
```

The fresh LoRA adapters modify the Qwen2-VL decoder attention projections.
The projection head maps the selected Qwen hidden state into the same
512-d coordinate space as the frozen FashionCLIP image embeddings.

### Safe architecture summary command

This command prints the architecture from constants in the code. It does
**not** instantiate the 7B model, so it is safe to run locally:

```bash
python - <<'PY'
from src.training.contrastive_model import (
    BASE_REPO,
    LORA_REPO,
    LORA_RANK,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    QWEN2VL_HIDDEN_DIM,
)

D_TARGET = 512
print("ContrastiveQwen2VL architecture summary")
print("=========================================")
print(f"Base VLM: {BASE_REPO}")
print(f"Stage-2 LoRA loaded+merged from: {LORA_REPO}")
print("Fresh trainable LoRA:")
print(f"  rank: {LORA_RANK}")
print(f"  alpha: {LORA_ALPHA}")
print("  dropout: 0.0")
print(f"  target modules: {LORA_TARGET_MODULES}")
print("Frozen components:")
print("  Qwen2-VL vision tower")
print("  merged Qwen2-VL base weights")
print("  FashionCLIP image tower (not inside train forward; precomputed cache)")
print("Trainable components:")
print("  LoRA adapters on Qwen2-VL decoder attention projections")
print(f"  projection head: {QWEN2VL_HIDDEN_DIM} -> 1024 -> {D_TARGET}")
print("  SymmetricInfoNCE.logit_scale")
print("Runtime lm_head replacement:")
print(f"  original lm_head: {QWEN2VL_HIDDEN_DIM} -> vocab_size")
print(f"  replaced with dummy lm_head: {QWEN2VL_HIDDEN_DIM} -> 1")
print("Retrieval forward path:")
print("  images + modification text")
print("  -> Qwen2-VL processor/chat template")
print("  -> Qwen2-VL transformer hidden states: (B, seq_len, 3584)")
print("  -> choose rightmost non-pad token: (B, 3584)")
print("  -> projection head: (B, 512)")
print("  -> L2 normalize")
print("  -> cosine retrieval against FashionCLIP image cache")
PY
```

Expected output:

```text
ContrastiveQwen2VL architecture summary
=========================================
Base VLM: DanJZY/Qwen2-VL-7B-Speech
Stage-2 LoRA loaded+merged from: DanJZY/Qwen2-VL-7B-Speech-LoRA
Fresh trainable LoRA:
  rank: 32
  alpha: 64
  dropout: 0.0
  target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
Frozen components:
  Qwen2-VL vision tower
  merged Qwen2-VL base weights
  FashionCLIP image tower (not inside train forward; precomputed cache)
Trainable components:
  LoRA adapters on Qwen2-VL decoder attention projections
  projection head: 3584 -> 1024 -> 512
  SymmetricInfoNCE.logit_scale
Runtime lm_head replacement:
  original lm_head: 3584 -> vocab_size
  replaced with dummy lm_head: 3584 -> 1
Retrieval forward path:
  images + modification text
  -> Qwen2-VL processor/chat template
  -> Qwen2-VL transformer hidden states: (B, seq_len, 3584)
  -> choose rightmost non-pad token: (B, 3584)
  -> projection head: (B, 512)
  -> L2 normalize
  -> cosine retrieval against FashionCLIP image cache
```

### Full module inspection command

Only run this on a GPU server with enough memory; it instantiates the 7B model:

```bash
python - <<'PY'
from src.training.contrastive_model import ContrastiveQwen2VL

model = ContrastiveQwen2VL(d_target=512, device_map="cuda:0")

print("Projection head:")
print(model.proj)

print("\nCurrent lm_head:")
print(model.vlm.base_model.model.lm_head)

print("\nTrainable parameters:")
model.vlm.print_trainable_parameters()

print("\nFull wrapper:")
print(model)
PY
```

For a meeting/code review, the most useful outputs are usually:

- `model.proj`
- `model.vlm.base_model.model.lm_head`
- `model.vlm.print_trainable_parameters()`

`print(model)` is very long because it expands the whole Qwen2-VL module.

Code:

- `src/training/contrastive_model.py` lines 36–49: projection head
- `src/training/contrastive_model.py` lines 71–81: fresh LoRA config
- `src/training/contrastive_model.py` lines 90–97: dummy `lm_head`
- `src/training/contrastive_model.py` lines 163–181: forward path

---

## Q2: What is multi-positive InfoNCE, and why did we need it?

**Context:** Plan-5 used standard (diagonal-only) InfoNCE. Plan-6 replaced
it with multi-positive InfoNCE. The FACap training split has 53,686 triplets
but only 28,252 unique target images — ~47% collision rate.

**Answer:**

Standard InfoNCE with a batch of N samples:

```
logits = exp(logit_scale) * (q @ t.T)   # (N, N) similarity matrix
labels = [0, 1, 2, ..., N-1]            # diagonal is the positive
loss = cross_entropy(logits, labels)
```

Every off-diagonal pair `(i, j)` is treated as a negative. But with 47%
target collision, in almost every batch multiple queries share the same
target image. Standard InfoNCE then *penalizes* the model for giving similar
embeddings to queries that should genuinely be similar — this directly
contradicts the learning objective.

**Fix — multi-positive mask:**

Build a boolean mask: `pos_mask[i, j] = True` iff `target_id[i] == target_id[j]`.
Then for each row `i`:

```
loss[i] = logsumexp(logits[i, all j])         # denominator: compete against everyone
         − logsumexp(logits[i, j where pos_mask[i,j]])  # numerator: sum over all co-positives
```

If there is only one positive (no collision), this reduces to standard
cross-entropy. With multiple positives it says: "pull *all* co-positive
embeddings toward query i, not just the diagonal one."

The mask is built by broadcasting `target_ids` against itself:

```python
pos_mask = target_ids.unsqueeze(1) == target_ids.unsqueeze(0)  # (N, N)
```

`target_ids` is gathered across all 8 GPUs *before* building the mask, so
co-positives from other GPUs' batches are included.

Code: `src/training/loss.py` — `_multi_positive_nce`, `SymmetricInfoNCE.forward`.

---

## Q3: What is hard negative mining, and why is it on the Plan-7 list?

**Context:** Plan-6 headline R@10 plateaued at ~0.40 after epoch 10.
Hard negative mining is item 2 on the Plan-7 recommendation list.

**Answer:**

In InfoNCE the "negatives" are all other samples in the batch. Random batch
negatives are mostly **easy negatives** — items clearly dissimilar to the
query. The model quickly assigns them low similarity; their gradient
contribution approaches zero. The model stops learning.

A **hard negative** is a gallery item that is very similar to the query but
is *not* the correct target. Example:

- Query: red dress + "make it dark red"
- Correct target: dark red dress A
- Hard negative: dark red dress B (almost identical style, different product)

**Hard negative mining** explicitly selects these difficult samples:

1. Compute embeddings for all gallery items with the current model.
2. For each query, retrieve the top-K most similar gallery items.
3. Remove the true target; the rest are hard negatives.
4. Use those hard negatives as the negative set in the InfoNCE loss.

**Why Plan-6 hit its ceiling:** with random in-batch negatives the model
already separates them confidently by epoch 10 — gradients are near zero,
no signal to improve further. Hard negatives would force finer distinctions
between near-identical items, which is exactly what is needed to push past
the current ~0.40 ceiling toward the Phase-A bar of 0.533.
