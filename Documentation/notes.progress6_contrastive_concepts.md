# Notes: Plan-6 contrastive concepts

Personal study notes for Plan-6 concepts. Companion file:
`notes.progress5_contrastive_learning.md` (Plan-5 study notes).

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
