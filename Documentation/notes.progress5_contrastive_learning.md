# Notes: understanding the Plan_5 contrastive design from scratch

Personal study notes from the 2026-05-01 design session. Each entry is
a question I had while reading `Documentation/Plan_5_20260501.md`,
plus the context and answer so I can re-read this and recover the
reasoning later.

These notes accompany Plan 5 (single-GPU contrastive fine-tune,
Phase B). Companion file: `notes_baseline.md` (Phase A study notes).

---

## Q1: Wait, Phase A also didn't *train* anything, right?

**Context:** I assumed Phase A involved training because the docs talk
about a "weak baseline." Re-reading made me realize Phase A only ran
inference on a stack of pretrained pieces.

**Answer:** Correct — Phase A trained nothing. It was pure
inference + benchmarking:

- Pretrained `speechQwen2VL`, frozen → generate a caption for
  `(ref image, mod text)`
- Pretrained Marqo FashionCLIP **text** tower, frozen → encode that
  caption into a query vector
- Same frozen text tower → encode all 59 k FACap-shipped gallery
  captions into target vectors (one-time pre-compute)
- Cosine similarity → top-K

So Phase A is best phrased as: *"How well does this stack of
pretrained pieces retrieve, with zero fine-tuning?"* Answer with the
right encoder: R@1 = 0.258 (Marqo FashionCLIP).

**Plan 5 is the first time anything in this project actually gets
trained.** Specifically: the LoRA adapters on Qwen2-VL's LLM layers +
a small projection head we attach to the end. Everything else (vision
tower, FashionCLIP image tower) stays frozen.

---

## Q2: Why single-GPU first instead of multi-GPU from day one?

**Answer:** Three reasons, in order of weight.

1. **Smaller debug surface.** Multi-GPU adds DDP, gradient sync,
   distributed sampler, NCCL config. When something breaks (it will),
   single-GPU debugs *one* failure mode (model / loss / data);
   multi-GPU debugs that *plus* the distributed plumbing. Hard to
   tell which one bit you.
2. **Faster iteration on early failures.** Single-GPU run starts in
   seconds. Multi-GPU has spawn overhead and crash recovery is
   messier. Most early failures are silly (wrong tensor shape, NaN
   at step 5) — tight feedback loop matters.
3. **You don't always need it.** If Plan 5's "alive bar"
   (R@10 ≥ 0.35) hits in 30 min on one GPU, we're done with v1.
   Skipping straight to multi-GPU would be premature optimization.

The general ML pattern: *get the loss right first, then scale.*
Plan 7 ramps to multi-GPU after Plan 5 proves the recipe.

---

## Q3: How can the same Qwen2-VL produce both a caption AND an embedding? What's "the text generation head" and why is it "unused at inference" in Plan 5?

**Context:** I was confused because I thought of LLMs as "text in,
text out" — so how can the same model output a *vector* instead?
This is the single most important thing to understand for Plan 5.

### What Qwen2-VL actually computes

A transformer is a function: `(sequence of tokens) → (sequence of
vectors)`. For an input of length L, the model outputs an
`L × hidden_size` matrix at every layer. Qwen2-VL's
`hidden_size = 3584`.

The "last layer hidden states" = the matrix from the deepest layer.
Shape `(L, 3584)`. Each row is one token's "summary," computed by
attending to every earlier token.

That is the model's **body**. By itself, it produces vectors, not
text.

### How captions get produced (Phase A)

Captions come from an extra layer on top of the body called the
**language modeling head** (`lm_head`). It's a single linear layer:

```
lm_head: (3584,) → (vocab_size,)   # ~152 k for Qwen2-VL
```

Apply softmax to its output → probability distribution over the
vocabulary → sample the most likely next token → append to the input
→ run forward again → repeat until EOS.

Phase A's caption was produced by:

```python
hidden_states = qwen2vl.body(image + prompt)        # (L, 3584)
last = hidden_states[-1]                             # (3584,)
logits = qwen2vl.lm_head(last)                       # (152_000,)
next_token = argmax(logits)                          # one token
# repeat 256 times for a 256-token caption
```

The `lm_head` is what turns the model from "sequence-of-vectors-
machine" into "text generator."

### How embeddings get produced (Plan 5)

We **skip the lm_head entirely**. Run forward exactly once, take the
last hidden state, project it down:

```python
hidden_states = qwen2vl.body(image + prompt)        # (L, 3584)
last = hidden_states[-1]                             # (3584,)
embedding = projection_head(last)                    # (512,)
embedding = normalize(embedding)                     # L2-normalized
# done — this is the retrieval embedding
```

No iterative sampling. No vocab projection. Just `body → last vector
→ projection → done`.

**"Text generation head unused at inference"** = we never call
`lm_head` in Plan 5's inference path. Its weights still live inside
the model (we don't delete them — they were trained on causal LM and
they keep the body's representations sensible), but we don't read its
output.

### The mental model shift

Old (LLM-as-text-machine): "model takes text, outputs text."

New (the right one): "model takes a multimodal sequence and produces
a vector at every position. The `lm_head` is one *optional consumer*
of those vectors, specifically tuned to output 'next-token
probability.' We can build other consumers — like a retrieval head —
that read the same vectors and output different things."

**The same model can produce different outputs because the body is
general; only the *head* is specialized.** Phase A used the lm_head.
Plan 5 attaches and trains a retrieval head.

### What does the last hidden state "look like"?

Concretely, just `(3584,)` floats. Not human-readable. It's a learned
representation that happens to be close (in cosine sim) to
representations of similar content. Whether "similar content" means
"next-token-similar" or "fashion-retrieval-similar" depends on what
the model was trained to produce.

That's why we *fine-tune* in Plan 5: the body's current
representations are tuned for next-token prediction. We need them to
instead encode "this dress, after applying this modification." LoRA
+ a contrastive loss reshapes the representations toward retrieval.

---

## Q4: What does "target" mean in retrieval / contrastive training?

**Answer:** Two roles in retrieval:

- **Query** = what the user shows up with. In our task: `(reference
  image, modification text)`.
- **Target** = the correct answer item. In our task: an image of the
  modified dress.

In contrastive training:

- **Query embedding**: produced by the *trainable* model from the
  query.
- **Target embedding**: produced by *some encoder* from the target.
- **Loss**: pull (query_emb, positive_target_emb) together; push
  (query_emb, *other*-target_emb) apart.

**"FashionCLIP image tower as target"** = "use FashionCLIP's image
encoder to produce the embedding of the target dress image. That
embedding is the fixed point Qwen2-VL is trying to match."

---

## Q5: Doesn't freezing FashionCLIP cap us at FashionCLIP's level?

**Context:** This was my biggest worry — if we lock the target side
to a pretrained encoder, our retrieval can never beat that encoder.
The Phase A best was R@1 = 0.258. Aren't we stuck near 0.258?

**Answer:** Yes, frozen does cap us — but not where you might think.

The Phase A ceiling was R@1 = 0.258 using FashionCLIP-**text** on
encoded *captions*. That's a caption-to-caption retrieval ceiling.

The Plan 5 ceiling is different: **how well can FACap dress targets
be retrieved purely image-to-image with FashionCLIP-image?** That
ceiling is *separate* from the caption ceiling.

To measure it, the verification section has a sanity check (item 4):
encode every FACap dress image with FashionCLIP-image, ignore the
VLM entirely, do image-to-image retrieval — same target dresses
retrieved by their own image embeddings. The R@K we get is the
ceiling for Plan 5.

- If that ceiling is, say, R@10 = 0.7 → plenty of headroom over Phase
  A's 0.533.
- If it's R@10 = 0.55 → Plan 5 is fundamentally limited; we'd want to
  either (a) train both sides jointly (Plan 6), or (b) swap target
  tower (DINOv2, SigLIP).

**Honest summary:** frozen target caps us at FashionCLIP-image's
image-to-image ceiling. We don't yet know what that ceiling is. The
first thing Plan 5 does is measure it. Frozen is for v1 stability;
if the ceiling is too low, Plan 6 unfreezes.

---

## Q6: At application time we don't have target captions. How does that affect training?

**Context:** The user's app at deploy time has only `(ref image,
modification text or audio)`. No target caption. Does this constrain
the training setup?

**Answer:** It actually *simplifies* training.

### Phase A at application time

- User: `(ref image, mod text)`
- VLM generates a target-oriented caption (text out)
- Encode that caption with FashionCLIP-text → query embedding
- Compare against pre-computed embeddings of all gallery captions
- Return top-K

Notice: Phase A *requires gallery captions*. We pre-computed
captions for all 59 k dresses (FACap ships these). No human writes a
caption at query time, but the pipeline relies on captions
everywhere.

### Plan 5 at application time

- User: `(ref image, mod text)`
- Trained Qwen2-VL outputs an embedding directly
- Compare against pre-computed embeddings of all gallery target
  *images*
- Return top-K

Notice: Plan 5 *requires gallery target image embeddings*. We
pre-compute these with FashionCLIP-image once, cache to disk, load at
startup. **Captions never enter the picture.**

### Implications for training

For Plan 5, training data is `(ref image, mod text, target image)`
triplets. FACap ships exactly this. **No caption needed at training
time, no caption needed at inference time.**

This is a feature: **Plan 5's training setup matches the application
setup more closely than Phase A does.** Phase A trained nothing, but
its inference path (caption-then-retrieve) was a slightly artificial
construction — it relied on captions the application doesn't
naturally have. Plan 5 trains the model directly on what the
application actually does: take a multimodal query and find a
matching image.

---

## Q7: So Plan 5 doesn't use FACap's target_caption field at all?

**Answer:** Correct. Plan 5 ignores `target_caption` entirely. The
field is still on disk in FACap's annotations; we just don't read it.

The target embedding comes purely from feeding the target *image*
into FashionCLIP's image encoder:

```
target image (e.g., FACap_dress_train_28473.jpg)
    ↓
frozen Marqo FashionCLIP image tower (open_clip)
    ↓
(512,) L2-normalized vector
```

No caption involved on the target side. The FACap shipped captions
served Phase A; Plan 5 lives without them.

Plan 6 may re-introduce captions as an ablation — e.g., multi-positive
contrastive: match BOTH the FashionCLIP-image embedding AND the
FashionCLIP-text caption embedding, summed or averaged. That's a
Plan 6 question; v1 stays single-target (image only).

---

---

## Q8: We have target captions — can we use them to improve performance?

**Context:** I asked whether ignoring `target_caption` is leaving free
information on the table. Yes, there are several ways to use it.

Important distinction: **caption as supervision** (what the loss
compares against) vs **caption as input** (what the model reads).
Different mechanisms.

### As supervision (loss-side)

These keep the user-facing pipeline unchanged (no caption needed at
app time) but use captions during training to lift the learned
embedding's quality.

**(a) Multi-positive contrastive — easy, low risk.** Pull the query
embedding toward TWO target representations: the FashionCLIP-image
embedding AND the FashionCLIP-text embedding of the target caption.
Loss = `0.5 * InfoNCE(query, image_target) + 0.5 * InfoNCE(query,
text_target)`. Image grounds visual semantics; caption grounds
linguistic semantics. Cheap — extra encoder forward + second loss
term. Could be a Plan 5 ablation rather than waiting for Plan 6.

**(b) Auxiliary captioning loss.** Keep Qwen2-VL's `lm_head` active.
Add LM cross-entropy: given `(ref image, mod text)`, predict the
target caption. Mix with contrastive. BLIP-2-style multi-task. Risk:
LM and contrastive objectives fight; needs careful loss weighting.

**(c) Distillation from caption-encoded target.** Like (a) but as
MSE/KL distillation rather than contrastive — query target IS the
FashionCLIP-text caption embedding. Cleaner gradients but locks the
ceiling to FashionCLIP-text (the Phase A R@1 = 0.258 ceiling we want
to escape). Probably worse than (a).

### As input (model-side)

**(d) Reference-image caption as additional input feature.**
Pre-compute a caption for every gallery image, prepend at query time:
`(ref image, ref caption, mod text) → embedding`. Adds free info; at
app time we'd need to caption the user's ref image (cheap via the
same VLM). Probably small lift since the VLM can already see the
image.

**(e) Target caption as optional input with dropout.** At training,
sometimes give the model the target caption directly ("hint mode");
always test without. Teaches the model to use captions when
available. Limited value for our app — we *never* have target
captions at inference.

**Best fit for our setup:** (a) multi-positive supervision. Probably
the single biggest "free lunch" from this brainstorm.

---

## Q9: Can Qwen2-VL supervise itself? Why a separate target encoder?

**Context:** I wanted to know if we could just use Qwen2-VL on both
the query AND target sides — same model, different prompts — without
any external encoder. "Why do we need FashionCLIP at all?"

**Answer:** Yes, it's possible — but it's the trickiest direction and
not v1 territory. Three architecture flavors:

| Flavor | Query encoder | Target encoder | Risk | Potential ceiling |
|---|---|---|---|---|
| **Two-tower frozen** (current Plan 5) | trainable Qwen2-VL | frozen FashionCLIP-image | Low — debug-friendly | Capped at FashionCLIP-image |
| Two-tower joint (Plan 6) | trainable Qwen2-VL | trainable FashionCLIP-image | Medium — moving targets | Higher, harder to debug |
| **Siamese — Qwen2-VL on both sides** | trainable Qwen2-VL | trainable Qwen2-VL (same weights, different prompt) | High — representation collapse | Highest in principle |
| Momentum / BYOL — Qwen2-VL teaches Qwen2-VL | trainable Qwen2-VL | EMA copy of itself | Medium-high | High, very stable |

For the siamese version: same weights encode both sides. Query prompt:
`(ref image, mod text)`. Target prompt: `(target image alone,
"describe this dress")`. Same Qwen2-VL → embedding on both sides.
Loss pulls them together.

**Why this is risky in v1:** without a fixed anchor, contrastive can
collapse — the model maps everything to the same vector and loss → 0
trivially. The whole BYOL / SimSiam / MoCo / DINO line of papers
exists specifically to prevent this. Tricks like stop-gradient,
momentum encoders, asymmetric prediction heads. None hard, but each
is one more thing that has to work.

**Why this might be the right long-term direction:** at the limit,
the model is supervising itself end-to-end. Nothing external caps the
ceiling. Image grounding is preserved (it sees images on both sides).
The application could use the same forward pass for both query and
gallery encoding — one model, one inference path.

**Decision for Plan 5: don't start here.** For Plan 6: definitely
worth trying once Plan 5 proves the contrastive plumbing works.

---

## Q10: Won't the embeddings be in different latent spaces?

**Context:** Big-picture worry — if Qwen2-VL produces query embeddings
and FashionCLIP produces gallery embeddings, aren't they in
completely different latent spaces? Wouldn't cosine similarity be
nonsense?

**Pre-training: yes, completely different spaces.** Off-the-shelf
Qwen2-VL hidden states and off-the-shelf FashionCLIP image embeddings
have nothing to do with each other. Naive comparison would be
random.

**Post-training: same space, by construction.** This is the entire
point of contrastive training. InfoNCE explicitly pushes
`cosine(query_vec, positive_target_vec)` UP and
`cosine(query_vec, other_target_vec)` DOWN. The gradient flows back
through the trainable projection head (and LoRA), reshaping query
vectors so they land *near their correct target vectors* — which live
in FashionCLIP-image's space. After training, `query_vec` lives in
that same space because that's where the loss put it.

**The projection head's literal job: "take a 3584-dim Qwen2-VL hidden
state and map it into FashionCLIP-image's 512-dim space."** That
mapping is the core thing being learned in Plan 5.

### Useful intuition

Think of FashionCLIP-image's space as fixed coordinates on a map.
Pre-training, Qwen2-VL produces vectors in its own coordinate system.
Plan 5's training fits a **learned coordinate-transform** that
converts Qwen2-VL's natural output into FashionCLIP's coordinates.

### Concrete analogy: original CLIP itself

OpenAI's CLIP works because the text and image towers were trained
*together* with contrastive loss. Before training: completely
different spaces. After training: "a photo of a golden retriever"
(text) lands near a photo of a golden retriever (image). Plan 5 does
the same alignment, except:
- Only one side moves (Qwen2-VL); FashionCLIP-image stays put as the
  anchor
- The "text" side is a multimodal `(ref image, mod text)` query

**Contrastive learning is literally a recipe for aligning two
different encoders into a shared space.** That's the whole magic.

### When the worry IS valid

The worry would be real if any of these held:
- Run retrieval *without* training — yes, would fail
- Training fails to converge (collapse, NaN, etc.) — but then no
  embedding works anyway
- Projection head has insufficient capacity to map into FashionCLIP
  space — unlikely; 3584→1024→512 has plenty of capacity

After successful training: not a real concern.

---

## Q11: Should we measure the image-to-image retrieval ceiling first?

**Answer: yes, absolutely, before any training.** Plan 5 verification
step #4 is exactly this: encode every FACap dress image with
FashionCLIP-image, ignore the VLM, do image-to-image retrieval. The
R@K we get tells us the ceiling Plan 5 can hit *if Qwen2-VL learns
the perfect mapping*.

- High ceiling (R@10 ≥ 0.7): plenty of headroom over Phase A's
  R@10 = 0.533. Frozen-target is fine.
- Mid ceiling (R@10 ≈ 0.6): some headroom but limited. Frozen still
  reasonable for v1; Plan 6 should unfreeze.
- Low ceiling (R@10 ≤ 0.55): frozen-target was a bad bet. Need to
  reconsider before committing to a long training run — try a
  different target encoder (DINOv2, SigLIP), or jump to joint
  training in Plan 5 itself.

Cheap to run — a few minutes. **Should be the first thing we do when
implementation starts.**

---

## Q12: What do existing CIR systems do? Single-model or two-tower?

**Both clusters exist; Plan 5's design has direct precedent.**

| System | Year | Architecture | Target encoder |
|---|---|---|---|
| **CLIP4CIR** (Baldrati et al.) | 2022 | CLIP image + text → "Combiner" network → query embedding | **Frozen CLIP-image** (same as Plan 5) |
| **Pic2Word** | 2023 | Map ref image to "pseudo-word," prepend to mod text, run frozen CLIP | Frozen CLIP |
| **BLIP4CIR** | 2023 | Fine-tuned BLIP-2 + Q-Former | Frozen image encoder |
| **CompoDiff** | 2023 | Diffusion-based latent generation on CLIP space | Frozen CLIP-image |
| **SEARLE / LDRE** | 2023-24 | Pseudo-token zero-shot CIR | Frozen CLIP |
| **CIReVL** | 2024 | Zero-shot CIR — LLM generates target caption from ref+mod, then CLIP-text retrieval | Frozen CLIP-text |
| **LinCIR** | 2024 | Linear projection + language-only training | Frozen CLIP |
| **MagicLens** (Google) | 2024 | Single dual-encoder model trained end-to-end | **Same model** for query and target (siamese) |
| **E5-V** | 2024 | MLLM-as-retriever — pool last-token hidden state of LLaVA-style VLM, contrastive fine-tune | Same MLLM |
| **GME** | 2024 | Generalist multimodal embedder — Qwen-VL-style backbone + contrastive | Same MLLM |
| **InstructIR variants** | 2024 | Instruction-tuned retrieval — natural-language instructions steer the embedding | Same MLLM |

Two design clusters:

1. **Frozen-target / two-tower** — CLIP4CIR, BLIP4CIR, CompoDiff,
   CIReVL, LinCIR, Pic2Word, SEARLE. Easier to train; defensible v1.
   **Plan 5 sits here.**
2. **Single-model / siamese** — MagicLens, E5-V, GME, InstructIR.
   Higher ceiling but trickier (collapse risks, more compute).

Most CIR papers report on FashionIQ (~6 k gallery), where SOTA R@10
sits ~50–65 % with end-to-end fine-tuning. Our 59 k FACap gallery is
10× harder, so absolute numbers look lower, but relative gains
should carry over.

### More up-to-date — honest disclaimer

My training-data cutoff is January 2026. I have most 2024 work and
*some* 2025 work, but not the full 2025 picture and almost nothing
past early 2026. Treat what follows as "what I have confidence in,"
not "current SOTA."

**2024 trends I'm confident about:**

- **MLLM-as-retriever** (E5-V, GME, InstructIR): pool the last-token
  hidden state of a multimodal LLM, fine-tune contrastively. **This
  is exactly the Plan 5 recipe.**
- **Decoder-LLM-as-text-embedder** (NV-Embed, Qwen3-Embedding,
  GritLM): same idea on text-only. Plan 3's encoder ablation tested
  several of these.
- **Zero-shot CIR via pseudo-tokens** (SEARLE, LinCIR, CIReVL,
  Pic2Word): no retrieval-specific training data, surprisingly
  competitive.
- **VLM-as-reranker**: dual-encoder fetches top-50, then a VLM
  cross-encoder rescores. Common production pattern.

**2025+ (lower confidence, broad trends only):**

- Continued scaling of MLLM retrievers (7 B → 13 B → 70 B).
- LoRA / PEFT methods on MLLM retrievers became standard.
- Instruction-tuned retrieval — natural-language instructions
  controlling what the model retrieves.
- Diffusion-based generative retrieval (CompoDiff-flavored) as a
  serious alternative.

**Where to look for current SOTA (more reliable than asking me):**

- [Papers With Code — "Composed Image Retrieval"](https://paperswithcode.com/task/composed-image-retrieval) leaderboard.
- arxiv.org with filter `cat:cs.CV cat:cs.IR "composed image retrieval"`, sort by date.
- CVPR 2025 / ICCV 2025 / NeurIPS 2025 proceedings.
- Lab blogs: Marqo, Snowflake AI, Google Research, Salesforce.
- Search query that has worked: `"composed image retrieval" "MLLM" 2025` on arxiv.

If I need a fresh sweep past my training cutoff, I can use WebSearch
to pull the most recent 5–10 papers on demand.

---

---

## Q13: What is a CIR system, and what's two-tower vs single-model?

**Context:** Q12 surveys existing CIR systems and uses architecture
terms (two-tower, single-model, dual-encoder, siamese, cross-encoder)
without defining them. This entry pins down the vocabulary first so
Q12's table actually makes sense.

### Q13a — What is a CIR system?

**CIR = Composed Image Retrieval.** The retrieval task where the
query has TWO parts that must be **composed** together:

- A **reference image** ("here's the dress I'm starting from")
- A **modification text** ("...but make it shorter and remove the
  sleeves")

The system retrieves an image that matches the reference *modified
by* the text.

| Task | Query | Output |
|---|---|---|
| Image-to-image retrieval | image | similar image |
| Text-to-image retrieval | text | matching image |
| **CIR** | **image + text** | **modified-image match** |

The "composed" part = multiple input modalities composed together to
specify *one* query. FACap, FashionIQ, and CIRR are the standard CIR
benchmarks. Our project is a CIR system on the FACap dress slice.

### Q13b — Two-tower vs single-model: three architectures, often confused

The terminology gets sloppy in papers. The clean taxonomy:

#### 1. Dual-encoder / two-tower (separate weights)

- Two networks: one for query, one for target. They never see each
  other's input.
- Query → `query_vec` independently. Target → `target_vec`
  independently. Score = `cosine(query_vec, target_vec)`.
- **Plan 5 is this.** Trainable Qwen2-VL on the query side; frozen
  FashionCLIP-image on the target side. Different weights, different
  modalities.
- *Pros:* target embeddings precompute once, cache for the whole
  gallery; query at inference = one forward + cheap dot product.
  Scales to millions of items.
- *Cons:* query and target only "talk" through the embedding
  bottleneck — no fine-grained interaction.

#### 2. Siamese / shared-weight two-tower

- Same network used twice — once for query, once for target. ONE
  set of weights.
- Different prompts let the model know which side it's encoding.
  E.g., query side: `(ref_image, mod_text)`; target side:
  `(target_image, "describe this")`.
- Still produces independent embeddings → still scales like a
  dual-encoder.
- **MagicLens / E5-V / GME are this flavor.**
- *Pros:* one model to deploy; no separate target encoder; higher
  ceiling potentially.
- *Cons:* representation collapse risk (no fixed anchor); trickier
  training.

#### 3. Cross-encoder

- Single network takes BOTH query AND target as one concatenated
  input, outputs a similarity score directly. No separate embeddings.
- E.g., feed `[query_image] + [mod_text] + [SEP] + [target_image]`
  → model outputs a scalar score.
- *Pros:* query–target interaction at every layer. Highest accuracy.
- *Cons:* doesn't scale — for each query, run the model once *per
  target* in the gallery. Useless for 59 k galleries except as a
  reranker on the top-K.

### Why this matters for our scale

For 59 k gallery + 1000 queries + near-real-time retrieval, only the
dual-encoder family is feasible at inference. Cross-encoders sometimes
show up as a **reranker** stacked on top of dual-encoder retrieval
(top-50 from dual-encoder → cross-encoder rescores → top-1).

---

## Q14: What does "training X against Y" / "frozen image-tower target" actually mean?

**Context:** I kept getting tripped up by the sentence *"Train the
speechQwen2VL query embedding against a frozen Marqo FashionCLIP
image-tower target."* Several phrases doing work in one line. This
entry unpacks each.

### "Image-tower"

Marqo FashionCLIP has TWO halves: a **text tower** (text → vector)
and an **image tower** (image → vector). Both halves were CLIP-style
co-trained, so they live in the same vector space.

In Plan 5 we use **only the image tower** — the half that takes a
dress photo and returns a vector. We feed it the target image; out
comes a 512-dim embedding.

### "Against ... target"

In ML, **"training X against Y"** means: adjust X's weights so that
X's output matches Y. **Y is the target value** the loss compares
against — the ground-truth-like reference signal. Same idea as
supervised learning where you train a classifier "against" the label.

In Plan 5, on every training step:

```python
# Trainable side (what we're adjusting)
query_vec  = Qwen2VL_with_LoRA(ref_image, mod_text)    # 512-dim
            └── this is what changes during training

# Frozen side (fixed reference value)
target_vec = FashionCLIP_image_tower(target_image)     # 512-dim
            └── this never changes; FashionCLIP weights are frozen

# Loss says: make these two vectors similar
loss = -cosine_similarity(query_vec, target_vec)       # plus negatives
```

After many steps, Qwen2-VL learns: "when I see `(red dress, 'make it
blue')`, produce a vector that lands near where FashionCLIP places
the blue dress."

### "Target" is doing double duty (sorry, the field is sloppy)

Two meanings collapse into one in retrieval:
- "Target" as in **target dress** = the right-answer item we want to
  retrieve from the gallery (the blue long dress).
- "Target" as in **training target** = the reference value the loss
  compares the prediction against.

In contrastive retrieval training they're the SAME thing: the
*embedding of the target dress* IS the *training target*. So
"FashionCLIP image-tower target" reads as both:
- "the target [dress] embedding produced by FashionCLIP's image tower"
- "the target [training-loss reference value] produced by
  FashionCLIP's image tower"

Both readings are simultaneously correct.

### Plain-English rewrite of the original sentence

> "Adjust Qwen2-VL's weights so that, given a query
> `(ref_image, mod_text)`, it produces a 512-dim vector that lands
> close to the 512-dim vector that frozen FashionCLIP's image-half
> produces for the correct target dress image. After training, those
> two vectors live in the same space — that's what makes
> nearest-neighbor retrieval work."

That's the whole Plan 5 training objective in one sentence.

---

## Q15: Did Marqo FashionCLIP train on Fashion200K? Are our encoder-ablation results contaminated?

**Context:** Plan 3's encoder ablation showed Marqo FashionCLIP at
R@1 = 0.258, top of the table. FACap is built on top of Fashion200K
images. If Marqo trained on Fashion200K, the result would be partly
"the encoder has seen these images before" rather than "the encoder
is genuinely strong on this task."

### Verdict: No contamination — Marqo trained on a separate corpus

Confirmed via WebSearch + cross-check with another agent's
investigation of Marqo's model card, blog post, and dataset README.
Findings:

- Marqo-FashionCLIP and Marqo-FashionSigLIP were trained on **over 1M
  fashion products** from Marqo's own curated corpus.
- The Marqo model card and blog explicitly state: *"this training
  dataset was not a part of any of the evaluation datasets."*
- Fashion200K is listed as one of seven **evaluation datasets** Marqo
  used (alongside DeepFashion In-shop, DeepFashion Multimodal, KAGL,
  Atlas, Polyvore, iMaterialist).
- The HuggingFace `Marqo/fashion200k` dataset card describes it as
  "used to evaluate Marqo-FashionCLIP and Marqo-FashionSigLIP."

So the Plan 3 / Progress_3 encoder-ablation result for
`marqo-fashionclip` is **not a "trained on test data" cheat**.

### Two caveats to keep on file

1. **Don't confuse with OpenFashionCLIP.** OpenFashionCLIP (Cartella
   et al. 2023) is a *different* fashion-CLIP model. Its paper
   explicitly lists FashionIQ, Fashion-Gen, **Fashion200K**, and
   iMaterialist as part of its training data. If we ever swap to
   OpenFashionCLIP, Fashion200K contamination becomes real.
2. **Web-scale CLIP pretraining is fuzzy.** Marqo-FashionCLIP's
   backbone is LAION-pretrained CLIP. We can't strictly prove the
   web-scale pretraining set didn't include images that overlap with
   Fashion200K's product photos. This is a softer "may have seen
   similar images" exposure — distinct from "directly fine-tuned on
   the test set."

### Implication for Plan 5

- Frozen FashionCLIP image-tower as target is fine — the encoder
  isn't memorizing FACap targets.
- The Plan 3 vs. Plan 5 head-to-head comparison is internally fair
  (both pipelines use FashionCLIP image semantics, so any pretrain
  bias cancels).
- For the eventual writeup: disclose "frozen target encoder is Marqo
  FashionCLIP, trained on a separate 1M-product corpus disjoint from
  Fashion200K evaluation; web-scale pretraining overlap is untestable
  but considered minor." Standard limitations paragraph.

**Sources:** Marqo FashionCLIP [GitHub](https://github.com/marqo-ai/marqo-FashionCLIP),
[Marqo blog](https://www.marqo.ai/blog/search-model-for-fashion),
[HuggingFace dataset card](https://huggingface.co/datasets/Marqo/fashion200k).

---

## Mental-model summary in one diagram

```
Phase A (no training, benchmark):
  query  → frozen VLM     → caption_text → frozen text-encoder → query_vec
  target → FACap caption                 → frozen text-encoder → target_vec
  cosine sim → top-K

Plan 5 (first training, direct embedding):
  query  → trainable VLM + new head           → query_vec
  target → frozen FashionCLIP image tower     → target_vec
  cosine sim → top-K
  Loss: pull (query_vec, correct_target_vec) together via InfoNCE.
```

The thing that changed: **we replaced the
"VLM → caption → text-encoder → vec" detour with a direct VLM-to-vec
path, and we train the VLM-to-vec mapping with a contrastive
objective.**

---

## Q16: Training 是不是加了太多 considerations？

担忧来自 cross-review 跑完后 Plan 5 显得很厚 —— 各种 probe、threshold、
exclusion set。看起来工程量很大。下面是分类，说明 design 是不是过度了。

### 训练循环本身就 5 步

```
for batch in dataloader:
    # 1. Encode query: Qwen2-VL((ref_image, mod_text)) → projection → (B, 512)
    # 2. Lookup target: 从 pre-computed cache 拿 (B, 512) target_emb
    # 3. Compute symmetric InfoNCE loss
    # 4. Backward + AdamW step
    # 5. 每 500 步：dev slice 上跑 retrieval，log R@K
```

核心训练逻辑跟 CLIP 训练几乎一样。没有任何花哨的东西。

### Considerations 分三类

**A. 必须做的事**（不写下来训练就跑不起来）
- Stage-2 LoRA merge → 加新 LoRA：必须有，不写就不知道怎么 init
- LoRA config (rank 32, q/k/v/o_proj)：PEFT 要这些参数
- L2 train filter + dev slice 切分：不然训练数据和 eval 数据重叠
- VRAM profile 选 micro-batch：不然要么 OOM 要么浪费显存
- Logit-scale 参数化：不然数值不稳定

**B. Sanity check / debug hook**（额外开销很小，翻车时能定位问题）
- 三路 sensitivity probe（normal / mod-stripped / mod-shuffled）：
  每 500 步多跑两次额外 retrieval，cached gallery 上每次 ~1 分钟
- Image-grounding probe（已并入 sensitivity probe）
- 重现性 check（同 seed → 同结果）
- Eval-harness sanity（同图 in/out → R@1=1.0）

翻车时能告诉你"翻在哪" —— 模型不学（loss 不降）、collapse 到视觉 NN
（normal ≈ mod-stripped）、还是 encoder 选错了。

**C. Threshold / 决策规则**（"什么时候停下来"）
- Alive bar: dev R@10 ≥ 0.35 by 0.25 epoch
- Promising bar: headline R@10 ≥ 0.55 at end of epoch 1
- Headline win: R@1 ≥ 0.28 at convergence
- Sensitivity gap > 0 by 0.25 epoch

只是把"什么时候该停"写成规则。不写也得靠看数字 ad-hoc 决定，写下来反而省时间。

### 是不是过度了？老实说没有，但显得多

判断标准：**有没有写哪个 consideration，删掉之后训练会变更好或更省事**？

| Consideration | 删掉之后会怎样 |
|---|---|
| L2 train filter | 训练数据 leak 到 eval，headline 数字不可信 |
| Dev/headline 分开 | Headline 变成 fitted quantity，跟 Phase A 比就没意义 |
| 三路 sensitivity probe | Collapse 到视觉 NN 抓不到，最后做出"看起来 work 但其实没用 mod"的废模型 |
| Logit-scale clamp | Temperature 飘走，loss 数值爆掉 |
| Profile-first true batch | 要么 OOM 要么 batch 选小了 |
| LoRA merge_and_unload | PEFT 双重 wrap 出 bug，训练根本跑不起来 |
| Step 0 | **可以删** —— 已经删了 |

每条都有真实代价对应。**唯一被删掉的恰好是 Step 0**。

### 实际编码工作量估算

| 工作 | 估算 |
|---|---|
| `_load_qwen2vl_base()` 抽出 + Stage-2 merge 逻辑 | 0.5 天 |
| `target_cache.py`（FashionCLIP-image cache） | 0.5 天 |
| `dataset.py`（L2 + dev slice + sensitivity probe 数据） | 1 天 |
| `loss.py`（symmetric InfoNCE + logit_scale） | 0.5 天 |
| `online_eval.py`（三路 probe 包装） | 0.5 天 |
| `train_plan5.py` 主循环 | 1.5 天 |
| VRAM profile + 调通 smoke test | 1 天 |
| 实际跑训练 + 看结果 | 1–2 天（取决于 epoch 时长） |

**总共 5–7 天编码 + 1–2 天训练。**

### 觉得多想砍的话，最容易砍的

- Mod-shuffled probe（保留 normal vs mod-stripped 就够）：省 0.5 天
- Reproducibility check（自己心里有数就行）：省 0.5 天
- Online dev slice（直接每 500 步跑 headline，接受 overfitting risk）：
  省 0.5 天但损失大

---

## Q17: Sensitivity probe 是什么？跟大厂 pretraining 那种是一回事吗？

不是同一回事。

### 大厂的"模态丢失也能跑"是什么

你描述的那个叫 **modality dropout / modality robustness training**，
是一种**训练手段**：训练时随机扔掉一个输入模态（image / text / audio），
强迫模型学会"缺哪个都能输出合理 embedding"。

**目的**：让模型在 inference 时 robust 到缺失输入 —— 用户只给图、只给
文本、只给音频，模型都能产生合理的输出。

代表论文：FLAVA（image-text dropout）、VATT（video-audio-text）、
ImageBind 等。

### 我们的 sensitivity probe 是什么

完全不同 —— 是**评估时的诊断工具**，不是训练手段。

每次 dev eval 跑三次 retrieval，看模型对不同输入版本的反应：

| 输入 | 模型应该的反应 | 如果 collapse 到视觉 NN |
|---|---|---|
| `(红色长裙, "make it shorter")` | 找红色短裙 | 找别的红色长裙（忽略 "shorter"） |
| `(红色长裙, "")` | 不知道用户想要啥 → 乱猜 → R@10 ↓ | 还是找红色长裙 → R@10 跟上面一样 |
| `(红色长裙, "make it longer")` (shuffle) | 找红色更长的裙 | 还是找红色长裙 → R@10 跟前两个一样 |

**关键信号是"三个数字差多少"，不是某个数字本身。**
- 三个数字差不多 → 模型废了（不在用 mod_text）
- normal >> mod-stripped >> 0 → 模型在做正确的事

### 为什么我们 *不* 想要 modality dropout

Plan 5 要的恰恰相反 —— **我们想让模型强烈依赖 mod_text**。如果训练时
做 modality dropout（随机扔 mod_text），模型反而会学会"没 mod_text 也
能跑"，这就是我们要避免的 visual-NN collapse。

Plan 8 加 audio 的时候才会考虑 modality dropout —— 让 text-mod 和
audio-mod 可互换。但那是后话。

---

## Q18: 决策 threshold 是怎么定的？会不会太严？

Threshold 是**有 reference 的 educated guess**，不是凭空拍的，也不是
跑出来的。每个 bar 的来源：

### Alive bar: dev R@10 ≥ 0.20 by 0.25 epoch

参考点（同一个 ~58k gallery，同一个 1000-query 评估方法）：
- **完全随机猜**：R@10 = 10/58082 ≈ **0.00017**（地板）
- **Phase A MiniLM-L6**（一个通用的非 fashion 文本 encoder + 整个 caption-
  then-retrieve pipeline）：**R@10 = 0.24**
- **Phase A Marqo FashionCLIP**（Phase A 最强）：**R@10 = 0.533**
- **Plan 5 的 headline 目标**：要 beat 0.533

把这些点排在数轴上：
```
0.00017 ─────────────── 0.20 ─── 0.24 ──────── 0.533
random              alive bar  MiniLM      Marqo (= goal)
```

为什么选 0.20：
1. **比 random 高 ~1000 倍** → 模型不是在乱猜
2. **接近 Phase A MiniLM 但略低** → 0.25 epoch 时（模型才看了 ~14k 样本，
   新 LoRA 还在 warmup）能达到一个"通用文本 encoder + 整个 Phase A
   pipeline"水平的 80%，是合理的健康度
3. **margin from both ends**：低于 0.15 容易跟"还没收敛"混淆；高于 0.25
   会误杀慢 warmup 的 run

这个 bar 答的不是"模型够好了吗"，而是 **"模型在学吗"**。0.25 epoch 时 0.20
不算了不起的成绩，但能淘汰掉"完全没学到"的 run（loss 不降、梯度爆炸、
data loader 错位等）。

### Promising bar: headline R@10 ≥ 0.55 at end of epoch 1

参考点：Phase A best = 0.533。0.55 比它略高，意思是 **"训了 1 个完整 epoch
之后已经超过 Phase A baseline"** —— 这是整个 Phase B 存在的理由。如果 1
个 epoch 都不能 beat Phase A，那 Plan 5 的"训练比 freeze + retrieve 强"
这个论点就站不住。

### Headline win: R@1 ≥ 0.28 at convergence

参考点：Phase A R@1 = 0.258。0.28 比它略高 —— **就是项目的成功定义**：
我们把 R@1 推到比 Phase A 高，证明 contrastive training 有意义。

### Sensitivity gap > 0

不是数值阈值，是**方向阈值**。要求 R@10(normal) > R@10(mod-stripped)，
不管差多少。意思是 "模型至少有一点点在用 mod_text"。差距越大越好，
但任何正值都比 0 好。0 或负数 → 模型完全没在用 mod_text → debug。

### 为什么是 educated guess 而不是测出来的

要"测"出 threshold，需要先跑过类似的训练实验、知道学习曲线长啥样。
我们这是**第一次**做 contrastive on Qwen2-VL，没有先验数据。所以只能：
- 用 Phase A 的数字当 anchor
- 跟 random / 已知 baseline 比远近
- 留 margin 防误杀

**这些数字不是"圣经" —— 跑出第一次结果后，在 Progress_5 里 calibrate
一下，下个 run 就有真实数据 backing 了。**

### 决策：alive bar 改成 warning，不硬停

经过讨论后改了：
- **alive 阈值从 0.35 降到 0.20**
- **action 从 "stop and debug" 改成 "log warning, 人来看"**

理由：
1. 0.25 epoch 时模型才看了 ~14k 样本，新 LoRA 没 warmup 完，0.35 可能太严
2. 看 trajectory 比看绝对数字重要：
   - R@10 = 0.18 但**在涨**（0.05 → 0.12 → 0.18）→ 继续训
   - R@10 = 0.18 且**flat**（0.17 → 0.18 → 0.18）→ debug
3. 硬停止只留给真正的 numerical failure（NaN / Inf / OOM）

---

## Q19: Dev slice / headline slice / "headline" 是什么？

### 名词解释

- **"Headline" 数字**：我们最后**对外报告**的数字。例如："Plan 5 R@10 = 0.55, beat Phase A 的 0.533"。这是会写进 paper / portfolio / progress doc 的那个数。
- **"Headline slice"**：用来计算 headline 数字的 eval set。我们用 Plan 3 的 1000-query slice，因为这样能跟 Phase A 做 apples-to-apples 比较。
- **"Dev slice"**：另一个 ~500-query 的 eval set，从 train 切出来的，**专门用来在训练过程中频繁看**（每 500 步），用来做训练决策。

### 为什么要分开

如果**同一个 eval 集**既用来做训练决策，又用来报 headline 数字，问题是
**你实际上在 fit 那个 eval 集**。

具体场景：
1. 训练 20 次（不同 seed / lr / batch size）
2. 每次每 500 步 eval slice X，挑 R@10 最高的 checkpoint
3. 报告 "我们 R@10 = 0.58"
4. 但如果你拿一个**全新的 eval slice Y** 测同一个 checkpoint，可能只有 0.50
5. 那个 0.58 是 fitted 出来的，不是模型真实水平

这就是 **"headline 变成 fitted quantity"** 的意思。

### 我们的做法

| Slice | 大小 | 用途 | 频率 |
|---|---|---|---|
| **Dev slice** | ~500 queries（从 train 切出来，L2 排除后再分） | Online R@K，决定何时停 / 选 checkpoint / 调 hyperparam | 每 500 步 |
| **Headline slice** | 1000 queries（Plan 3 用的同一个 slice） | 对外报告数字，跟 Phase A 比 | 仅在 epoch 结束 + 收敛时摸一次 |

**Headline slice 不参与任何训练决策**。这样最后 "我们 beat Phase A 的
0.533" 这个 claim 才**可信** —— 因为 headline slice 没被 fit 过。

### 实际工程实现

`dataset.py` 启动时：
1. 从 train split 取最后 1000 triplets → headline slice
2. 在剩下的 train 里随机取 500 triplets → dev slice
3. 把 headline + dev 的所有 `target_id` 和 `candidate_id` 加进 exclusion set
4. Train loader 过滤掉所有命中 exclusion 的 triplet（L2 filtering）
5. Assert: train loader 里没有任何 ID 跟 dev / headline 重叠

---

## Q20: Logit-scale clamp 究竟是什么？

### 先看 InfoNCE 怎么算

我们在算 query 跟 target 的相似度后，要做 softmax 来得到概率分布。具体：

```
sim = query_emb @ target_emb.T          # 范围 [-1, 1]（L2-normalized 后）
logits = sim / τ                         # τ 是温度
loss = cross_entropy(softmax(logits), correct_index)
```

τ（温度）控制 softmax 的"尖锐度"。

### 用具体数字看 τ 大小的影响

假设 batch 里有 4 个候选 target，正确答案的 cosine sim 是 0.8，
其他三个错的是 0.3, 0.2, 0.5。

**情况 A: τ = 0.07（合理初值）→ 1/τ ≈ 14.3**
```
logits = [11.4, 4.3, 2.9, 7.1]
softmax = [0.98, 0.001, 0.0003, 0.025]
```
softmax 把 98% 概率给了正确答案，模型有信号但不极端。

**情况 B: τ = 0.001（飘走了）→ 1/τ = 1000**
```
logits = [800, 300, 200, 500]
softmax = [1.0, 0, 0, 0]
```
softmax 已经完全 saturate 成 one-hot 了。**问题**：cross-entropy 对一个
完美 one-hot 的 softmax，梯度几乎为 0。模型再也学不动了。

**情况 C: τ = 1（飘大了）→ 1/τ = 1**
```
logits = [0.8, 0.3, 0.2, 0.5]
softmax = [0.32, 0.20, 0.18, 0.30]
```
softmax 太平了，正确答案才 32%。模型不容易学，但不会爆。

### Clamp 是什么

τ 是**可学习参数**（每 step 跟着梯度走）。如果训练让 τ 飘到接近 0，会
出现情况 B —— **训练 collapse**，再也学不动。

CLIP 的标准做法：
1. 不直接学 τ，学 `logit_scale = log(1/τ)`（用对数让数值更稳）
2. 每个 optimizer step 后做 `logit_scale.data.clamp_(max=log(100))`，
   等价于 `1/τ ≤ 100`，等价于 **τ ≥ 0.01**

这样 τ 不会变得比 0.01 还小，logits 不会爆。

### 为什么之前是 bug

第一版我写的是 "clamp τ ≤ 100"，这是**夹错方向了**：
- τ ≤ 100 防的是 τ 变大 → 但 τ 变大不危险（情况 C 而已，模型只是学得慢）
- 真正危险的是 τ 变**小** → softmax saturate → 梯度消失 → 死局

正确的 clamp 是夹**下限**（τ ≥ 0.01），或等价地夹 `1/τ` 的**上限**
（≤ 100）。Codex round 1 抓到这个错。

---

## Q21: Adapter merge 的时机问题

### speechQwen2VL 的 release 格式

我去查了 `src/baseline/vlm_caption.py` line 118-126：

```python
self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    "DanJZY/Qwen2-VL-7B-Speech",        # ← BASE
)
self.model = PeftModel.from_pretrained(
    self.model, "DanJZY/Qwen2-VL-7B-Speech-LoRA"  # ← Stage-2 ADAPTER
)
```

所以 speechQwen2VL **不是一个 merge 完的 model**，而是**两个 HF repo**：
- `DanJZY/Qwen2-VL-7B-Speech` = base
- `DanJZY/Qwen2-VL-7B-Speech-LoRA` = Stage-2 audio adapter

每次用都要先 load base → load adapter。

### 你的两个问题

**Q1：fine-tuning 时是不是把 merged model 完全 frozen 了？**

是的。Plan 5 训练时：
- merged base（含 Stage-2 已 fold 进去）= **完全 frozen**，0 个可训练参数
- vision tower（Qwen2-VL 的 image encoder 部分）= **完全 frozen**
- FashionCLIP image tower = **完全 frozen**
- **Plan 5 的 LoRA adapter（rank 32, 加在 LLM decoder 的 q/k/v/o_proj）= 唯一可训练 ~30M 参数**
- **Projection head（3584 → 1024 → 512 MLP）= 可训练 ~4.2M 参数**
- Logit-scale = 1 个可训练参数（CLIP convention）

总共 ~34M 可训练 / Qwen2-VL 7B 总参数。99.5% frozen。

**Q2：Adapter 加在哪里？**

加在 LLM decoder 的注意力层 q/k/v/o_proj 上 —— 跟 Stage-2 一样的位置，
但 Plan 5 不动 gate/up/down_proj（Stage-2 动了那些，做的是 audio
adaptation；retrieval 任务不需要 MLP 层的额外容量）。

### 你真正问的：Stage-2 该什么时候 merge？

你问的是：**应该把 Stage-2 一次性 merge 成新 base 存下来**，还是
**每次训练都重新做 merge**？

两个选项：

| 选项 | 流程 | Pro | Con |
|---|---|---|---|
| **A. 每次训练都 merge** | 每次启动训练: load base → load Stage-2 → `merge_and_unload()` → 加 Plan 5 adapter → 训 | 不用多管一个 checkpoint；训练脚本自包含 | 每次启动多花 ~30 秒 merge；多个实验时累计开销 |
| **B. 一次性 merge 存本地** | 跑一次 prep 脚本: load base → load Stage-2 → `merge_and_unload()` → `save_pretrained("local/stage2_merged_base")`。之后每次训练: load `stage2_merged_base` → 加 Plan 5 adapter → 训 | 启动快；多次训练共享同一份 merged base | 占 ~14GB 本地磁盘；多管一个 checkpoint |

**数学上完全等价 —— 训出来的模型一模一样。**

### 我的建议：选项 B（一次性 merge）

理由：
1. Plan 5 之后还会有 Plan 6（loss / embedding-source / target-tower
   ablations）—— 多次训练同一个 merged base，省 merge 时间
2. 14GB 不是问题（你 A6000 server 肯定有空间）
3. 多管一个 checkpoint 是小代价

**不上传 HuggingFace** —— 这是本地中间产物，没必要公开。

### 你说的"上 HF 减少出错"

你可能在想：上 HF 后用 `from_pretrained` 加载就标准化了，少出 bug。
这个想法对，**但跟 Plan 5 训练成功不成功无关** —— 只跟 load 流程标准化
有关。

如果之后 Plan 5 训出一个能 beat Phase A 的 model，**那时候**再考虑：
- 把 Plan 5 adapter merge 进 base + Stage-2 → 一个完整的 fashion-CIR model
- push 到 HF（让 Plan 10 的 demo / 别人复现你工作时方便）

现在没必要急。**先把训练跑通，有结果再说**。

### 决定（2026-05-01）

Plan 5 保持**选项 A（每次训练都 merge Stage-2）** —— 不 over-complicate。
选项 B（一次性 merge 存本地 checkpoint）记下来作为可能的优化，等真的觉得
每次启动 ~30 秒 merge 太慢、跑实验次数多到累计成本明显时再切换。两者
**数学等价**，切换是纯工程改动。

（决定的正式记录在 `Plan_5_20260501.md` §3。）

---

## Q22: Dev slice / Headline slice 跟标准 ML 的 train/val/test 是什么关系？

完全对应：

| 标准 ML | 我们的命名 | 用途 |
|---|---|---|
| Training set | Train (L2-filtered) | 训模型 |
| Validation set / Dev set | **Dev slice** (~500) | 训练中频繁看，调超参 / 选 checkpoint / 早停 |
| Test set | **Headline slice** (1000) | 最后报告用，从不参与决策 |

### 为啥用 "dev" 不用 "validation"

习惯问题。
- NLP 圈 (BERT、GLUE、SuperGLUE) 习惯叫 **dev**
- CV / RL 圈习惯叫 **validation**
- 功能完全一样

### 为啥要分两个 eval set

如果用 eval set 来调 hyperparameter，又用同一个 eval set 来报告数字，
那肯定只会选到 eval set 表现最好的时候 —— 数字被 fit 出来了，不可信。
所以分两个：dev 用来调，headline 用来报告。

---

## Q23: τ (temperature) 是什么？是不是跟 LLM sampling 里的 temperature 一回事？

**对，完全是同一个 temperature 概念。**

### 直觉

τ 控制 softmax 分布的"尖锐度"：
- **τ 大** → softmax 分布**平**（每个候选概率接近 uniform）→ 模型不"果断"
- **τ 小** → softmax 分布**尖**（最大那个吃掉大部分概率）→ 模型很"果断"

跟你在 GPT 里调 `temperature=0.7 vs 1.5` 是一回事 —— 高温 → 输出更随机，
低温 → 输出更 deterministic。

### 数值例子

假设 4 个候选 target 的 logits = `[0.8, 0.3, 0.2, 0.5]`（cosine sim）：

| τ | 1/τ | softmax(logits / τ) | 解读 |
|---|---|---|---|
| τ = 2 (高温) | 0.5 | `[0.30, 0.23, 0.22, 0.26]` | 几乎 uniform，模型没什么 confidence |
| τ = 1 (中) | 1 | `[0.32, 0.20, 0.18, 0.30]` | 略有偏好，但很接近 |
| τ = 0.07 (CLIP 默认) | ~14 | `[0.98, 0.001, 0.0003, 0.025]` | 已经很尖，正确答案吃掉 98% |
| τ = 0.001 (灾难) | 1000 | `[1.0, 0, 0, 0]` | 完全 one-hot，**梯度消失** |

### InfoNCE 想要 τ 多大？

InfoNCE 想要 τ **比较小**（softmax 尖一点）—— 这样"对错差距"被放大，
模型 signal 强，学得快。但**不能太小**：太小了 saturate 到 one-hot，
梯度消失，训练 collapse。

CLIP 的初值 0.07 是经验最优区。Plan 5 把它做成**可学习参数**，让模型在
训练中自己调。但要 clamp 下限防它飘到 collapse 区（见 Q24）。

---

## Q24: "学 logit_scale = log(1/τ)" 这个参数化具体是啥意思？为啥不直接学 τ？

### 错误做法（直接学 τ）

```python
self.tau = nn.Parameter(torch.tensor(0.07))  # 可学习参数

# Forward
logits = (q @ t.T) / self.tau    # 用除法
loss = cross_entropy(logits, labels)
```

**问题：**
1. AdamW 不知道 τ 必须 > 0。如果某次 gradient 很大，把 τ 推到 0 或负数：
   - τ = 0 → **除以零，NaN，训练崩**
   - τ < 0 → logits 符号翻转，等于把"对的答案"训成"错的答案"，**反向训练**
2. τ 跨不同 scale 时优化器很难调步长。从 0.07 → 0.007（变 10 倍尖锐）
   只动了 0.063；从 0.07 → 0.7（变 10 倍平）动了 0.63。**同样大的"语义
   变化"在 τ 空间步长差 10 倍**，AdamW 调不好。

### 正确做法（学 log(1/τ)）

```python
import math
import torch.nn as nn

# 不存 τ，存 log(1/τ)
# 初始化：要让 τ 起步在 0.07，所以 log(1/0.07) ≈ 2.659
self.logit_scale = nn.Parameter(
    torch.tensor(math.log(1.0 / 0.07))
)

# Forward
scale = self.logit_scale.exp()    # 把它转回 1/τ
                                   # exp() 永远 > 0，τ = 1/scale 也永远 > 0
logits = scale * (q @ t.T)         # 用乘法（等价于 (q @ t.T) / τ）
loss = cross_entropy(logits, labels)

# 训练循环里，每个 step 后 hard clamp
with torch.no_grad():
    self.logit_scale.clamp_(max=math.log(100.0))   # ≈ 4.605
```

### 为什么这样做就 OK 了

1. **`exp()` 永远 > 0** → τ = 1/exp(logit_scale) 永远 > 0 → 数值安全。
   AdamW 怎么折腾 `logit_scale` 都不会破坏 τ > 0 的约束
2. **log 空间步长均匀** → 从 τ=0.07 → 0.007 是 logit_scale + log10≈2.30；
   从 τ=0.07 → 0.7 是 logit_scale - 2.30。**对称的步长** → AdamW 调得好
3. **Hard clamp 防止飘走** → 每个 step 后强制 `logit_scale ≤ log(100)`，
   等价于 `1/τ ≤ 100`，等价于 **τ ≥ 0.01**。τ 不会变得比 0.01 还小，
   softmax 不会 saturate

### 跟 CLIP 论文一致

这是 **OpenAI CLIP 原始实现**的做法，OpenCLIP / SigLIP / 几乎所有现代
contrastive 项目都沿用。原始代码（CLIP repo `model.py`）：

```python
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
# ...
logit_scale = self.logit_scale.exp()
logits_per_image = logit_scale * image_features @ text_features.t()
```

Plan 5 直接照抄。**就这么写就对了**。

---

## Q25: speechQwen2VL 为啥 release 成 base + adapter 而不是 merged？

直觉问题："merge 完上传不更省事吗？" 但 **PEFT 研究项目分开 release 是常规
做法**，理由：

1. **adapter 文件小**（~150MB vs full model ~14GB）—— 下载快、不占 HF 空间
2. **可以 disable / 替换** —— 用户能选"用 base 不用这个 adapter"或"换成自己的 adapter"
3. **可以继续训** —— 用户可以从这个 adapter 继续 fine-tune
4. **Composability** —— 多个 adapter 可以叠加（虽然实际很少这么用）
5. **License 隔离** —— base 跟 adapter 可以挂不同 license

### `DanJZY/Qwen2-VL-7B-Speech` 这个 base 是啥

**不是纯 vanilla Qwen2-VL** —— 它应该是把 Whisper audio encoder + audio
projector 的**架构**焊接进 Qwen2-VL 之后的 base（架构改了，但 audio path
的权重还没训）。

Stage-2 LoRA (`DanJZY/Qwen2-VL-7B-Speech-LoRA`) 才是真正的 audio
fine-tuning 权重。

### Release 链条

```
Qwen2-VL-7B (vanilla, 来自 Alibaba)
    ↓ 加 Whisper 架构 + audio projector (架构改造，权重 random init)
DanJZY/Qwen2-VL-7B-Speech (这个 release 是上面这个东西)
    ↓ 用 audio 数据训 LoRA
DanJZY/Qwen2-VL-7B-Speech-LoRA (Stage-2 audio adapter，单独 release)
```

### 研究 vs 生产

研究项目分开放是标准做法。**生产 release**（直接给最终用户用的）才会
merge 成单文件让 load 更简单。

---

## Q26: "LLM 的 decoder" 是什么？LLM 哪来的 decoder？

历史命名遗留问题。

### Transformer 三种结构

| 类型 | 例子 | 用途 |
|---|---|---|
| **Encoder-decoder** | 原版 Transformer (2017)、T5、BART | 翻译、seq2seq 类任务 |
| **Encoder-only** | BERT、RoBERTa | 分类、NER、理解类任务 |
| **Decoder-only** | GPT / LLaMA / Qwen / Claude / ChatGPT | 生成类任务 |

**现代 LLM 几乎都是 decoder-only 架构**。但是哪怕没有 encoder，里面的
transformer block 还是**沿用了"decoder layer"这个名字**（历史包袱）。

所以 "Qwen2-VL 的 LLM decoder" 意思是：**Qwen2-VL 里那个 Qwen2 大语言模型
部分的所有 transformer block**。HF 代码里
`model.model.layers` 访问到的就是这些。

### 一个 decoder layer 内部长啥样

```
input x (B, seq_len, hidden_dim)
  ↓
self-attention（4 个线性投影）：
  q = x @ W_q   ←─┐
  k = x @ W_k   ←─┼── 这 4 个就是 "q_proj / k_proj / v_proj / o_proj"
  v = x @ W_v   ←─┤
  attn_out = softmax(q @ k.T / sqrt(d)) @ v
  x = x + attn_out @ W_o ←┘
  ↓
RMSNorm
  ↓
MLP (3 个线性投影)：
  gate = x @ W_gate   ←─┐
  up = x @ W_up       ←─┼── 这 3 个是 "gate_proj / up_proj / down_proj"
  out = (silu(gate) * up) @ W_down ←┘
  x = x + out
  ↓
output (B, seq_len, hidden_dim)
```

### "LoRA on q/k/v/o_proj of LLM decoder" 啥意思

意思是：**在每个 decoder layer 的 self-attention 那 4 个线性投影
(W_q, W_k, W_v, W_o) 上，加一个 low-rank 的 delta**。
gate/up/down_proj（MLP 部分）不动。

具体 LoRA 做的事：把 `W_q ∈ R^(d×d)` 替换为 `W_q + B @ A`，其中
`A ∈ R^(r×d)`，`B ∈ R^(d×r)`，`r` 是 rank（我们选 32）。
原本 d×d ≈ 16M 参数，LoRA 只训 r×d×2 = 64×3584 = 229k 参数 / 矩阵。

Qwen2-VL-7B 大约 28 层 decoder × 4 个 proj × 2 个 LoRA 矩阵 ≈ 224 个小矩阵
→ 总共 ~30M 可训练参数。

### 那 vision 部分呢

Qwen2-VL 还有一个**单独的 vision encoder（ViT）**处理图片，跟 LLM decoder
是分开的两块：

```
Image  → Vision Encoder (ViT, ~600M 参数, frozen) → vision tokens
                                                          │
                                                          ↓
Text   → Text tokens                                       (concat)
                                                          ↓
                                                LLM Decoder (Qwen2, ~6.5B 参数, 加 LoRA 训)
                                                          ↓
                                                last hidden state
```

Plan 5 **不动 vision encoder**（它已经能 extract 不错的视觉特征了），
只动 LLM decoder（让它学会把"视觉特征 + mod text"映射到 retrieval 友好
的 embedding）。

---

## Q27: "Tower" 是什么？为什么不是塔？

英文 "tower" 在 NN 里是个术语，意思是 **"一条独立的、垂直堆叠的 encoder
路径"**。跟塔确实有点关系 —— 大家画 architecture diagram 时把每个 encoder
画成**垂直往上堆的一摞 layers**，长得像塔。

### Two-tower / dual-encoder 架构

CLIP 这种结构有**两条独立的 encoder 路径**：

```
图片 ──→ image encoder (ViT, 一摞 transformer layers) ──→ image embedding
                ↑ 这是 "image tower"

文本 ──→ text encoder (一摞 transformer layers)         ──→ text embedding
                ↑ 这是 "text tower"

最后：cosine_similarity(image_embedding, text_embedding) → 训练 / retrieval
```

两个 tower **分别**把不同模态的输入压成同一个空间的 embedding，然后比对。

**关键性质**：两个 tower 训完后**可以独立用** —— image tower 单独提图特征
就能去 retrieve；text tower 单独提文本特征就能去 retrieve。这就叫
**two-tower / dual-encoder** 架构。

### 对比：Single-model 架构

不是 two-tower 的代表是 **cross-encoder**（比如 BERT 用来做 pair
classification）：

```
[CLS] image_tokens [SEP] text_tokens [SEP] → BERT → score
                                                    ↑
                                       一个数（不是两个 embedding）
```

Cross-encoder 准但慢 —— 每来一个 query 就要跟 gallery 里所有 item 重新跑
一次 BERT。Two-tower 可以**预先计算 gallery 那一边的 embedding 缓存起来**，
query 来了只跑 query 那边一次 + cosine 查表。**所以 retrieval 几乎都用
two-tower 架构**。

### 我们 Plan 5 system 里的 tower 清单

| 角色 | 是谁 | 训不训 |
|---|---|---|
| Qwen2-VL 的 vision encoder | Qwen2-VL 内部的 ViT，把图变成 vision tokens | **frozen** |
| Qwen2-VL 的 LLM decoder | 处理 vision + text tokens，输出 query embedding | **加 LoRA 训** |
| Projection head | 3584 → 1024 → 512 MLP，把 LLM 输出投影到目标空间 | **train (random init)** |
| FashionCLIP 的 image tower | FashionCLIP 的 ViT，把 target 图变成 512-d embedding | **frozen** |
| FashionCLIP 的 text tower | FashionCLIP 的 text encoder | **不用，整个 Plan 5 不碰** |

整个 Plan 5 system 是 **Qwen2-VL（query 边）+ FashionCLIP image tower
（target 边）**的 dual-encoder 结构 —— 类似 CLIP，但 query 边换成了
multimodal Qwen2-VL，target 边只用 FashionCLIP 的图 tower。

### 文献里 "tower" 还会出现的地方

- "two-tower retrieval" / "dual-encoder retrieval" —— 检索系统主流架构
- "siamese network" —— 两个 tower **共享参数**的特例（Q13 里讨论过）
- "asymmetric two-tower" —— 两个 tower 用不同架构（我们就是这种：query 边
  Qwen2-VL，target 边 FashionCLIP，不同模型）

---

## Q28: FashionCLIP 到底在 training 里干啥？是 architecture 的一部分吗？Loss 会 backprop 经过它吗？

非常关键的问题，搞清楚这个就理解了 Plan 5 一半的设计。

### TL;DR

- **FashionCLIP 不在我们的可训练 architecture 里。**
- **Training 时 FashionCLIP 完全不跑** —— 我们只查它**之前**算好的 embedding cache。
- **Loss 不会 backprop 经过 FashionCLIP** —— gradient 只流回 query 边的 Qwen2-VL，target 边的 embedding 是常数（PyTorch 视角下没有 `requires_grad`）。
- **FashionCLIP 唯一的作用**：定义 target embedding 应该长什么样（提供"靶子"），然后**一次性把所有 target 算成 embedding 存到磁盘**。training 时只是查表。

### 完整流程图

#### 阶段 1: One-time prep（跑一次，~10 分钟）

```
全部 ~58k 张 FACap dress target 图
        │
        ↓  (run frozen FashionCLIP image tower 一次)
        │
~58k × 512 floats (~120MB)
        │
        ↓
存到磁盘：runs/plan5/target_emb_cache.npy
```

这一步之后 **FashionCLIP 就可以被 evict 出 GPU 了**（cache 已经存好）。

#### 阶段 2: Training（每个 step 反复跑）

```
─── Query 边（每个 step 全跑一遍） ─────────────────────────────

  (ref_image, mod_text)
        │
        ↓  Qwen2-VL vision encoder (frozen)
  vision_tokens
        │
        ↓  concat with text tokens
        │
  (vision_tokens + text_tokens)
        │
        ↓  Qwen2-VL LLM decoder (LoRA 训) ← gradient 流过
        │
  last_hidden_state[EOS]  (3584-d)
        │
        ↓  Projection head (MLP, train) ← gradient 流过
        │
  query_emb (B, 512)  ← L2-normalized
        │
        ↓
        │
─── Target 边（每个 step 只查表，0 计算） ──────────────────────

  batch 里的 target_id 列表 (B 个 ID)
        │
        ↓  np.lookup
        │
  target_emb (B, 512)  ← 直接从 numpy cache 拿，requires_grad=False
        │
        ↓
        │
─── Loss ─────────────────────────────────────────────────────

  loss = SymmetricInfoNCE(query_emb, target_emb)
        │
        ↓ loss.backward()
        │
  gradient 流向：
    ✅ query_emb → projection head → LLM decoder LoRA → 更新参数
    ❌ target_emb 是常数张量，gradient 走不进去
    ❌ FashionCLIP 根本没参与这次 forward，gradient 跟它没关系
```

### 类比

把 FashionCLIP 想象成**一个出题老师**：
1. 出题阶段（one-time prep）：老师把 ~58k 张图都看了一遍，给每张图打了一个 "正确答案的 embedding"（512-d 向量），写在答案本上。
2. 考试阶段（training）：学生（Qwen2-VL）每次看到 query，要给出自己的 embedding。我们对照**答案本**（cache）打分，告诉学生"差多少"，学生改进。**老师本人不在考场，也不参与改卷**。

老师只是**定义了"答案应该长啥样"的标准**。学生学的是**怎么向这个标准靠拢**，学生本身不会变成老师，但学完之后产出的 embedding 跟老师的 embedding 在同一个 512-d 空间里，可以比对。

### 那为啥 Plan 5 §3 的 VRAM budget 提到 "+ ~600 MB FashionCLIP-image"？

那个数字是**保守估计**，假设 FashionCLIP 在 GPU 里 co-resident。但实际上：
- 严格来说，prep 阶段算完 cache 后就可以 `del fashion_clip; torch.cuda.empty_cache()`
- Training 时只用 cache 的 numpy 数组（CPU 内存里 120MB，按 batch 拷到 GPU 也才几 MB）
- 真正占 GPU 的就只有 Qwen2-VL + LoRA + projection head + activations

所以那 600MB 是过度保守，**实际上 training 时 FashionCLIP 可以完全不在 GPU 里**。

### 关键概念区分

| | "在 architecture 里" | "在 loss 里" | "在 training compute 里" | "影响最终 embedding 空间" |
|---|---|---|---|---|
| Qwen2-VL vision encoder | ✅ | ✅ | ✅ frozen forward | ✅ |
| Qwen2-VL LLM decoder | ✅ | ✅ | ✅ trainable forward + backward | ✅ |
| Projection head | ✅ | ✅ | ✅ trainable forward + backward | ✅ |
| **FashionCLIP image tower** | ❌ | ❌ (只贡献常数) | ❌ (cache lookup, 不跑模型) | ✅ (它定义了 target 空间) |

### 为啥设计成 "FashionCLIP 不参与 training"

1. **省算力** —— FashionCLIP 是 frozen 的，每个 step 重复算它的 forward 是浪费
2. **省 VRAM** —— 不用 co-resident 一个 ~600MB 的模型
3. **训练更稳定** —— target embedding 是确定的常数，不会因为某种 bug 突然变化
4. **概念清晰** —— "学生学怎么靠近一个固定靶子" 比 "学生跟老师同时变化" 要好理解、好 debug

如果哪天 Plan 6 想做 **joint training**（让 FashionCLIP image tower 也跟着训），那时候就必须每个 step 都跑 FashionCLIP forward + backward 了 —— **那是 Plan 6 的 ablation 之一**，但 Plan 5 不做。

### 一句话总结

**FashionCLIP 在 Plan 5 里是"靶子提供方"，不是"学生"也不是"老师在场"。
我们提前用 FashionCLIP 把所有 target 图算成靶子坐标存下来，training 时
就是让 Qwen2-VL 学怎么把 query 投射到正确的靶子坐标上。FashionCLIP 本身
全程不动，loss 也不会 backprop 经过它。**

## Q29: Dataset class 的设计——`__getitem__` return paths 还是 PIL？跟 Phase A 的 `FacapDataset` 不一样有没有问题？放哪个文件夹？

### 背景

Phase A 的 `src/data/facap_dataset.py` 里 `FacapDataset.__getitem__` 返回的是
**只有 string 字段的 dict**（`candidate_image_path`、`modification_text`、
`target_caption`、`target_id` 等），PIL 通过单独的 `load_image(item, side)`
方法才打开。

Plan 5 contrastive 训练我最初的草案里写的是 `(cand_image, mod_text, target_id)`
——直接 return PIL。这是两种不同的设计模式。

### 两种模式各自的适用场景

**Phase A 模式（return paths + 单独的 `load_image()`）**
- ✅ 适合 caption-then-retrieve：99% 的访问只需要 `target_caption` + `target_id`，
  根本不需要 PIL。如果 `__getitem__` 强制 decode jpeg，每访问一条 triplet 就读
  一张图，**纯属浪费**——因为 caption-based pipeline 永远不会真正用到那张图。
- ✅ 内存非常 cheap：只存 strings，59k 条 triplets 全部塞进 RAM 没压力。
- ❌ Caller 要记得调 `load_image()`，多一步——但因为 Phase A 几乎不需要 PIL，
  这"一步"几乎不发生。

**Plan 5 模式（return PIL）**
- 训练每个 step 一定要 `cand_image` PIL（VLM 要吃 `(ref_image, mod_text)` 进去），
  **不可能跳过 decode**。
- Sensitivity probe 要 mod-stripped / mod-shuffled 三种变体，但 `cand_image`
  在三种变体里都一样——不会重复 decode。
- 这种 "每个 item 都需要 PIL" 的场景，**直接在 `__getitem__` return PIL 是
  PyTorch 的标准 pattern**：DataLoader 的 `num_workers > 0` 会让 worker 进程
  并行 decode，跟训练的 GPU forward 在 pipeline 上重叠，相当于免费拿到并行 I/O。

**关键区别不在于"哪个更好"，而在于"哪个匹配你的 access pattern"**：
- Phase A 的 access pattern 是 sparse（绝大多数访问不需要 PIL）→ 返回 paths
- Plan 5 的 access pattern 是 dense（每次访问都需要 PIL）→ 返回 PIL

两个都是合理设计，**不是矛盾，是两种 pattern 各自最优**。

### 修订后的设计：wrap 而不是重写

不要把 `FacapDataset` 的 triplet/caption 加载逻辑在 Plan 5 里再写一遍。新类
（暂叫 `FacapContrastiveDataset`）应该 **compose / wrap 现有的 `FacapDataset`**：

- L2 filtering、dev slice 切分、sensitivity probe 的 mod-shuffled 变体——
  这些 Plan 5-specific 的逻辑都加在新类里。
- 实际读 metadata 还是走 `FacapDataset`（single source of truth）。
- `__getitem__` 末尾调 `base.load_image(item, "candidate")` 拿 PIL。
- 这样 Phase A 的代码完全不受影响；如果以后 FACap 数据格式变了，只有
  `FacapDataset` 需要改。

伪代码：

```python
class FacapContrastiveDataset(Dataset):
    def __init__(self, base: FacapDataset, exclusion_ids: set[str], ...):
        self.base = base
        # 在 base 的全部 indices 里筛掉 L2 排除的 triplets
        self.indices = [i for i in range(len(base))
                        if base[i]["target_id"] not in exclusion_ids
                        and base[i]["candidate_id"] not in exclusion_ids]

    def __getitem__(self, idx):
        item = self.base[self.indices[idx]]                # metadata
        cand_image = self.base.load_image(item, "candidate")  # PIL
        return {
            "cand_image": cand_image,
            "mod_text": item["modification_text"],
            "target_id": item["target_id"],
        }
```

### Folder 放哪

**`src/data/contrastive_dataset.py`**，跟 `facap_dataset.py` 平级——所有
dataset 类放一起，更连贯。

判断标准：
- `src/data/` —— 跟数据集本身相关的（FACap 的 schema、L2 filtering、image
  loading），换数据集的话整个文件夹都要重写。
- `src/training/` —— 纯训练 infra（loss、model wrapper、loop、target cache、
  online eval），换数据集这些代码完全不动。

`FacapContrastiveDataset` 的逻辑（L2 排除、dev slice 切分、sensitivity
probe 数据组织）都跟 FACap 的 schema 紧耦合，所以放 `src/data/` 更合适。

### 一句话总结

**Phase A 返回 paths 是因为 99% 的访问不需要 PIL；Plan 5 返回 PIL 是因为
100% 的访问都需要。两个设计都对，匹配各自的 access pattern。新类 wrap 老类
而不是重写，dataset 都放 `src/data/` 一个文件夹里。**

## Q30: 我们训练 Qwen2-VL 的哪些 layers？Prompt 怎么给？对结果有什么影响？

### 训练哪些 layers

**训练的：**
- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`（LLM decoder 每一层的注意力权重，rank=32, alpha=64）— ~28M 参数
- Projection head（3584 → 1024 → 512，GELU + LayerNorm）— ~4.2M 参数
- `logit_scale`（InfoNCE temperature，单个标量）

**冻结的：**
- Vision tower（ViT 图像编码部分）
- LLM 所有 MLP 层（`gate_proj`, `up_proj`, `down_proj`）
- Token embedding 层
- LLM base weights（只有 LoRA delta 参与梯度）

### 为什么只训 q/k/v/o，不训 MLP？

注意力机制（q/k/v/o）控制"哪些 token 和哪些 token 交互"——这是把图像 token 和文字 token 融合在一起的关键。训 LoRA 在这里 = 教模型如何把图像信息和修改文字信息 aggregate 到 EOS position。

MLP（gate/up/down_proj）是每个 token 自身的特征变换，不控制跨 token 信息流。它们的参数量是 attention 的约 3 倍（Qwen2-VL-7B: intermediate_size=18944, hidden=3584）。不训它们的理由：
1. VRAM 太大（每层 MLP LoRA 参数量是 attention LoRA 的 ~3×）
2. MLP 存储"语言知识"，改了更容易 catastrophic forgetting
3. 我们需要的跨模态信息融合是 attention 的职责

Token embedding 不训：改了会破坏所有文字 token 的表示，几乎所有 fine-tuning 都不动它。

### Stage-2 ASR LoRA 和当前 task 的关系

Stage-2 LoRA（ASR 训练）已通过 `merge_and_unload()` bake 进 base model weights，不再是独立 adapter。不存在"某个 prompt 激活 ASR，另一个 prompt 激活 retrieval"——prompt 不切换 capability，weights 决定 capability。

ASR 能力通过 speech token（音频输入）触发，我们的 input 里没有音频，ASR pathway 完全不走。换 instruction prompt 不会和 ASR 干扰。Stage-2 merge 的好处：更强的 multimodal 理解和 instruction-following，对 retrieval 有利，与 prompt 格式无关。

### Prompt 设计与对结果的影响

**Phase-A baseline（Plan-3）用的 prompt（生成任务）：**
```
"Given the reference fashion image and the modification instruction,
write a concise caption describing the target fashion item after
applying the modification."

Modification: {mod_text}
```
那是让模型**生成文字 caption**，与 Plan-5 的用途不同。

**Plan-5 embedding 任务的 prompt：**

我们不 generate——只取 EOS position 的 last hidden state 作为 embedding。所以 prompt 的作用是"告诉模型 EOS token 应该 represent 什么"。

**v1（最初的简单版本）：**
```python
content = [{"type": "image", "image": img}]
content.append({"type": "text", "text": f"Modification: {txt}"})
```
问题：没有告诉模型任务是什么。EOS 可能只是"图像 + 文字的语义摘要"，而不是"应该找到的目标商品的表示"。

**v2（当前版本，更明确的 instruction）：**
```python
content = [{"type": "image", "image": img}]
content.append({"type": "text", "text": (
    f"Given this product image, find the item that looks like the image "
    f"but with the following modification: {txt}"
    if txt else
    "Describe the product shown in this image."
)})
```
改进：明确告诉模型这是一个 retrieval 任务（"find the item"），EOS representation 更可能朝"目标商品是什么"的方向走。模型是 instruction-tuned 的，给了明确任务描述会更好地激活 instruction-following 能力。

### 图像 grounding 风险

图像 token 在序列前半部分，modification text 在后面。Causal attention 下 EOS 可以 attend 所有 token，但 LLM 倾向于偏重最近 context（text）。训练过程中模型可能越来越忽略图像。监控指标：`dev/r10_stripped`（无文字版 R@10）。如果它随训练趋近 0，说明模型已经不用图像了。

### System prompt

**现在没有显式 system prompt**。`messages_list` 只有 `role: user`，Qwen2-VL 的 chat template 遇到没有 system message 时会自动插入默认的：
```
You are a helpful assistant.
```

**为什么最初没有考虑改 system prompt？** System prompt 在 embedding 任务里是一个更进阶的优化，优先级低于 user instruction 本身。初版先把 user instruction 说清楚（v1 → v2 的改动），system prompt 是下一步可以考虑的地方。

**System prompt 有没有用？** 有可能。把默认的 "You are a helpful assistant." 换成更明确的任务描述，例如：
```
You are a fashion product retrieval assistant. Given a product image and a modification description, your goal is to represent what the modified target item should look like.
```
这样模型的 EOS representation 从一开始就被 prime 成"retrieval 任务"而不是"general assistant 任务"。但这还没有验证过，是一个值得试的改动，改之前需要先跑完当前的 run 有个 baseline 数字做对比。

## Q31: 为什么 eval 只在 GPU 0 上跑？一起跑不是更快吗？

两个阶段会出现"GPU 0 独自工作"的现象：

1. **模型加载时**（前 2 分钟）：rank 0 先加载模型再 broadcast，其他 GPU 等着。
2. **每次 eval 时**：代码里 `if _should_eval() and accelerator.is_main_process` — eval 只在 rank 0 上跑。其他 7 个 GPU 跳过 eval block，立刻开始下一个 batch 的 forward+backward，到 AllReduce 时卡住等 rank 0 跑完 eval。所以 eval 期间 GPU 1–7 瞬间跑一步就变成 0% util，GPU 0 一直在跑 eval inference。

正常 training step 期间（没有 eval）所有 8 个 GPU 应该都在工作。

理论上完全可以分布式跑，而且确实会更快。现在只在 GPU 0 跑是因为代码里用了 `if accelerator.is_main_process` 这个最简单的写法，省掉了跨 GPU 通信。

Eval 分两步：
1. **推理**：把 1500 个 query 过一遍 7B VLM，得到 embedding
2. **检索**：把 1500 个 embedding 和 59k gallery 做 cosine similarity，算 R@K

步骤 2 必须在一个地方做（需要所有 embedding 在一起）。步骤 1 是可以并行的——1500 queries 分给 8 个 GPU，每个 GPU 只跑 188 个，快 8 倍，然后用 `all_gather` 把结果汇总到 rank 0 再算 metrics。

现在的代价：
- 每次 eval 约 6–8 分钟（1500 queries 在单 GPU 上串行推理）
- 3 epochs 共 6 次 eval = ~40 分钟纯 eval 时间
- 这期间 7 个 GPU 完全在空转

分布式 eval 之后：
- 每次 eval 约 1 分钟
- 6 次 eval = ~6 分钟
- 节省约 35 分钟

值得在下个 run 改，代码改动约 30 行。

## Q32: 为什么没有 validation loss？能不能在 validation set 上也算一个 loss？

理论上可以，但对 retrieval 任务来说 R@K 比 validation loss 更有用，所以我们直接用 R@K 作为 validation 信号。

### 为什么不用 validation loss

**能算吗？** 可以。对 dev_slice 里的 500 个 triplet 算一遍 InfoNCE loss，就是 validation loss。

**有用吗？** 有限。InfoNCE loss 衡量的是"在 ~255 个 in-batch negatives 里，model 能不能把 positive 排第一"。它告诉你模型有没有严重过拟合（val loss >> train loss），仅此而已。

**R@K 更有用的原因：**
1. R@K 衡量的是在完整的 59k gallery 里检索——这才是实际任务
2. InfoNCE loss 低不代表 gallery retrieval 好，模型可能记住了 in-batch negative 的分布模式
3. R@K 直接可读：R@10=0.154 = 15.4% 的 query 能在前10名里找到答案
4. `sensitivity_gap`（normal - stripped R@10）能检测图像 grounding 是否崩溃，loss 完全看不出来

### 结论

我们每 0.5 epoch 跑一次的 dev eval（500 queries × 59k gallery）就是 validation——只是用 R@K 而不是 loss 来表达。这是 retrieval/contrastive learning 的标准做法（CLIP、BLIP、FashionCLIP 都这样）。

没有"validation loss"是刻意的设计，不是遗漏。

---

## Q33: lm_head 是什么？为什么它会占 2.31 GB 显存，即使我们根本不用它的输出？

### 背景

commit message 里写了：*"replaces lm_head with a 1-output stub to eliminate the 2.31 GB (B, seq_len, 152064) bf16 tensor that was causing OOM on RTX 3090 at bs=16"*

### lm_head 是什么

Qwen2-VL 本质是个语言模型，正常用途是生成文字。它的最后一层叫 `lm_head`，作用是把每个 token 的 hidden state（维度 3584）映射到整个词表（152064 个词），这样才能预测下一个词是什么：

```
hidden_states: (B, seq_len, 3584)
      ↓ lm_head = Linear(3584, 152064)
logits:        (B, seq_len, 152064)   ← 每个位置对 152064 个词的打分
```

### 为什么 OOM 发生在 fix 之前

Plan-5 里我们不生成文字，只取最后一层的 hidden state 的 EOS 位置来算 embedding。所以 `logits` 那个大 tensor 我们完全不用。

**但 PyTorch 的 forward pass 是顺序执行的——不管你用不用输出，只要执行到那一行，tensor 就被分配了：**

```python
# Qwen2-VL 的 forward() 内部（简化）：
hidden_states = self.transformer(input_ids, ...)  # 跑完所有 transformer 层
logits = self.lm_head(hidden_states)              # ← 这行执行时，(B, seq, 152064) 就在显存里分配了
                                                  #   哪怕你之后根本不用 logits
return CausalLMOutput(logits=logits, hidden_states=hidden_states)

# 我们的代码只取这个：
outputs = self.vlm(**inputs, output_hidden_states=True)
pooled = outputs.hidden_states[-1][...]           # 只用 EOS 位置的 hidden state
# 但 logits 已经占着 2.31 GB 了，要等 backward 结束才能释放
```

在 bs=16、序列长度约 509 token 时：

```
16 × 509 × 152064 × 2 bytes (bf16) ≈ 2.31 GB
```

RTX 3090 只有 24 GB，forward + backward 的其余部分已经用了约 21 GB，这 2.31 GB 直接把显存撑爆。OOM 发生在 backward pass（因为 forward 分配之后，backward 还需要保留中间激活来计算梯度）。

### Fix

把 `lm_head` 替换成只输出 1 个数的假层（stub）：

```python
# contrastive_model.py
_lm_head = vlm.base_model.model.lm_head
vlm.base_model.model.lm_head = nn.Linear(
    QWEN2VL_HIDDEN_DIM, 1, bias=False, dtype=torch.bfloat16
).to(next(_lm_head.parameters()).device)
del _lm_head
```

现在那行变成：
```python
logits = self.lm_head(hidden_states)  # shape: (B, seq, 1)，几乎不占显存
```

`lm_head` 不在 LoRA 的 target modules 里，所以没有 trainable 参数受影响。节省 **≈2.3 GB**，bs=16 在 RTX 3090 上就能跑了。

### 补充："分配"和"backward 才释放"是什么意思

**"分配"** = 在 GPU 显存里划出一块物理空间，把数字存进去。GPU 显存就像 RAM，是有限的物理空间，存了东西就被占用。

```python
logits = self.lm_head(hidden_states)
```

这行执行完，GPU 显存里就多了 2.31 GB 的数字。不管后面用不用、return 不 return，**数字已经在那里了**。`return` 只是 Python 层面传引用，跟 GPU 显存里的物理数据无关：

```python
# 这两种写法，显存占用完全一样：
logits = self.lm_head(hidden_states)
return logits          # 写法 A：return 出去

logits = self.lm_head(hidden_states)
return hidden_states   # 写法 B：不 return logits — logits 还是在显存里！
```

**为什么要等 backward 结束才释放：** PyTorch autograd 在 forward 时记了一张计算图，backward 计算梯度时需要把 forward 的中间 tensor 再拿出来用。所以 forward 跑完后，那些中间 tensor 不能释放，要等 backward 把梯度全算完才统一清理。OOM 发生在 backward 正是因为那时显存压力最大：forward 留下的所有中间 tensor 都还在，backward 自己还要额外分配梯度 tensor。

### 实际的代码改动

```python
# contrastive_model.py — 模型加载完之后立即替换 lm_head

_lm_head = vlm.base_model.model.lm_head          # 存起来只是为了拿它的 device
vlm.base_model.model.lm_head = nn.Linear(
    QWEN2VL_HIDDEN_DIM, 1,                        # 输出从 152064 改成 1
    bias=False, dtype=torch.bfloat16
).to(next(_lm_head.parameters()).device)          # 放到跟原来同一张 GPU
del _lm_head                                      # 原来那个大层彻底删掉，释放权重显存
```

之后 forward 里那行变成：
```python
logits = self.lm_head(hidden_states)
# shape: (B, seq_len, 1) 而不是 (B, seq_len, 152064)
# 16 × 509 × 1 × 2 bytes ≈ 0.016 MB，可以忽略不计
```

我们仍然取 `outputs.hidden_states[-1]`，lm_head 的输出继续被无视——但现在那个"被无视的 tensor"只有 16 KB 而不是 2.31 GB。

---

## Q34: `PeftModel.from_pretrained` 的 `device_map` vs `torch_device` bug

### 背景

commit message 里写了：*"Fixed PeftModel.from_pretrained to pass torch_device= instead of device_map= so LoRA adapter weights load to the correct per-rank GPU under DDP (device_map triggers HF dispatch_model which always placed weights on GPU 0)"*

### DDP 里每个进程应该做什么

8 卡 DDP 训练时，8 个进程（rank 0–7）各自跑一份完整的模型，每个进程应该把模型加载到自己对应的 GPU（rank 0 → GPU 0，rank 1 → GPU 1……）。

### bug 的完整路径

先看 PEFT 源码的关键片段：

```python
# peft/utils/other.py
def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"    # 永远返回 "cuda"，即 GPU 0
    ...
```

```python
# peft/peft_model.py — load_adapter() 方法
def load_adapter(self, ..., torch_device=None, ...):
    if torch_device is None:
        torch_device = infer_device()   # ← 如果没有显式传 torch_device，就用 GPU 0

    adapters_weights = load_peft_weights(
        model_id, device=torch_device, ...   # adapter 权重放到 torch_device 指定的位置
    )
```

`PeftModel.from_pretrained()` 最终会把 `**kwargs` 原封不动传给 `load_adapter()`。问题在于：

```
你调：PeftModel.from_pretrained(model, lora_repo, device_map="cuda:1")
                                                   ↑
                                          进了 **kwargs，不是 torch_device=

→ from_pretrained 把 **kwargs 传给 load_adapter(**kwargs)

→ load_adapter 签名是 load_adapter(self, ..., torch_device=None, ...)
  你传的是 device_map=，不是 torch_device=
  所以 torch_device 依然是 None

→ torch_device = infer_device()  →  "cuda"  →  GPU 0

→ 8 个 DDP 进程全部把 LoRA 权重放到 GPU 0 → OOM
```

### Fix

```python
# vlm_caption.py — _load_qwen2vl_base()

# 错误写法：
model = PeftModel.from_pretrained(model, lora_repo, device_map=device_map)

# 正确写法：
torch_device = device_map if isinstance(device_map, str) else None
model = PeftModel.from_pretrained(model, lora_repo, torch_device=torch_device)
#                                                    ↑
#                          load_adapter 里 torch_device 不为 None
#                          跳过 infer_device()，直接用 "cuda:1"（或对应的 rank GPU）
```

`torch_device` 的语义是"把这些 tensor 直接放到这张卡上"，不触发任何模型并行逻辑。

---

## Q35: `encode_image` 怎么实现的？open_clip 又是什么？

### open_clip 是什么

CLIP（OpenAI 提出）是用大量图文对做对比学习的模型，目标是让图片和文字的 embedding 落在同一个向量空间里（所以可以用文字搜图）。`open_clip` 是 LAION 社区的开源复现版，我们用的 **FashionCLIP** 就是在时装数据上 fine-tune 的 open_clip 模型。

```python
import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")
```

返回三样东西：
- `model`：有 `.encode_image()` 和 `.encode_text()` 两个方法
- `preprocess_train`：训练时的图片预处理（含随机裁剪等数据增强）
- `preprocess_val`：推理时的图片预处理（确定性变换：resize → center crop → ToTensor → Normalize）

`preprocess_val` 本质是一个 `torchvision.transforms.Compose`，必须和训练时一致，否则图片分布对不上，embedding 会偏移。

### encode_image 的实现

原来的 `_OpenClipWrapper` 只包装了文字编码（`.encode(texts)`），没有图片编码接口。我们在 target_cache.py 里需要把 59k 张图片全部编码成向量，所以加了 `.encode_image(images)`：

```python
class _OpenClipWrapper:
    def __init__(self, model, tokenizer, preprocess_val, device, max_seq_length):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.preprocess_val = preprocess_val   # ← 新加：推理预处理函数
        self.device = device

    def encode_image(self, images, batch_size=32):
        """PIL images → L2-normalized float32 ndarray (N, D)"""
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                tensors = torch.stack(
                    [self.preprocess_val(img) for img in chunk]  # PIL → tensor（resize/normalize）
                ).to(self.device)
                feats = self.model.encode_image(tensors)         # open_clip 的图片编码
                feats = feats / feats.norm(dim=-1, keepdim=True) # L2 normalize
                out.append(feats.cpu().float().numpy())
        return np.concatenate(out, axis=0)   # shape: (N, D), float32
```

`preprocess_val` 原来在 `_load_open_clip` 里被丢掉了：

```python
# 原来（broken）：
model, _, _ = open_clip.create_model_and_transforms(...)
#              ↑ preprocess_val 是第三个返回值，用 _ 丢掉了

# 修复后：
model, _, preprocess_val = open_clip.create_model_and_transforms(...)
return _OpenClipWrapper(model, tokenizer, preprocess_val, ...)   # 存进 wrapper
```

### 为什么需要 encode_image

Plan-5 的训练目标是：让 Qwen2-VL 的 query embedding 和 **FashionCLIP 图片 tower 算出来的 target embedding** 对齐。Target embedding 是固定的（FashionCLIP 冻结），所以训练前把 59k 张图全部编码一次存起来（`target_cache.py`），训练时直接按 `target_id` 查表，不需要每个 step 重新跑 FashionCLIP。`encode_image` 就是用来做这个离线编码的。

---

## Q36: Contrastive learning 的 dataset 是怎么构造的？negative 是什么？为什么 batch 越大越好？

### Dataset 的结构

我们的任务：给定一张候选图片 + 一段修改文字，找到目标图片。

Dataset 里每条数据是一个 **triplet（三元组）**：

```
(候选图片, 修改文字, 目标图片)

例如：
  候选图片: 一件红色连衣裙
  修改文字: "make it blue and shorter"
  目标图片: 一件蓝色短裙
```

`FacapContrastiveDataset.__getitem__` 返回的就是：

```python
return (cand_image_PIL, mod_text_str, target_id)
#       候选图片         修改文字        目标图片的 ID（用来查 embedding 缓存）
```

### Negative 是什么

模型的目标：**query embedding（候选图 + 修改文字）应该离正确的 target embedding 近，离其他所有 target embedding 远。**

"其他所有 target embedding"就是 **negatives（负样本）**：

```
query: "红裙子，改成蓝色短裙"

✅ positive:  蓝色短裙的 embedding   ← 应该靠近
❌ negative:  绿色长裙的 embedding   ← 应该推远
❌ negative:  白色上衣的 embedding   ← 应该推远
❌ negative:  黑色裤子的 embedding   ← 应该推远
...（gallery 里其他所有图片）
```

### In-batch Negatives

理想情况下每个 query 对着 59k 个 negatives 训练，但计算量太大。**In-batch negatives** 的思路：**同一个 batch 里其他样本的 target，就当作这个 query 的 negative**，不需要额外采样：

```
batch 里有 4 个样本：
  query_1 → target_1  (红裙→蓝裙)
  query_2 → target_2  (牛仔裤→黑裤)
  query_3 → target_3  (白T恤→条纹T)
  query_4 → target_4  (运动鞋→皮鞋)

对 query_1：
  ✅ positive: target_1
  ❌ negative: target_2, target_3, target_4  ← 借用同 batch 里其他人的 target
```

### InfoNCE Loss 怎么算

把 batch 里所有 query 和所有 target 的相似度算成一个矩阵，Loss 让对角线最大、非对角线最小：

```
              target_1  target_2  target_3  target_4
query_1  →  [  高,       低,       低,       低   ]  ← 希望第 1 个最高
query_2  →  [  低,       高,       低,       低   ]
query_3  →  [  低,       低,       高,       低   ]
query_4  →  [  低,       低,       低,       高   ]
```

本质是 N 分类的交叉熵：对 query_1，在 N 个候选里做 softmax，希望 target_1 的概率趋近于 1。

### 为什么 batch 越大越好

**negatives 越多，任务越难，loss 给的梯度信号越有区分度。**

```
batch=4:   在 4 选 1，随机猜有 25% 概率对  → loss 饱和快，模型学不到什么
batch=64:  在 64 选 1，有类似颜色的干扰项  → 模型必须真正理解修改文字
batch=512: 在 512 选 1，大量外观相近的混淆项 → 学得更精细
```

batch 大了之后，同一个 batch 里自然会出现外观相似但不是正确答案的图（**hard negatives**），这些才是真正有价值的训练信号。

数学上：InfoNCE loss 的理论下界是 `-log(1/N)`，N 越大，loss 的动态范围越大，梯度越有区分度。

实验也验证了这一点：bs=64（512 effective）的 v3 run 比 bs=8（64 effective）收敛更快，每步学到的东西更多。

### 我们用 all_gather 进一步扩大 effective batch

8 张 GPU，每张 bs=64。如果每张卡只用自己的数据算 loss，effective batch = 64。用 `--gather` 后，每张卡做完 forward，把所有卡的 embeddings 汇总再算 loss：

```
GPU 0 ~ GPU 7，每张各有 64 个 query + 64 个 target embedding

all_gather → 把 8 × 64 = 512 个 target embeddings 汇总

每个 query 现在对着 512 个 negatives 训练
```

server 10 的 effective batch = 64 × 8 = **512**，是 bs=8 run 的 8 倍，理论上应该能突破 bs=8 的 R@10=0.226 瓶颈——但实验还在进行中，见下表。

### 各 batch size 的 epoch 对比（截至 2026-05-03）

Step 单位没法跨 batch size 比较（bs=64 每步用 512 个样本，bs=8 只用 64 个），统一用 epoch：

| Epoch | bs=8 R@10 | bs=64 R@10 | bs=16 v4 R@10 |
|-------|-----------|------------|---------------|
| 0.5   | 0.166     | 0.008      | 0.168         |
| 1.0   | 0.182     | 0.054      | —             |
| 1.5   | —         | 0.082      | **0.196**     |
| ~1.8  | **0.226** | —          | —             |
| 2.0   | 0.216     | 0.122      | —             |
| 2.5   | 0.220     | 0.154      | —             |
| 3.0   | —         | 0.114 †    | —             |
| 3.5   | —         | 0.188      | —             |
| 4.0   | —         | 0.198      | —             |
| 4.5   | —         | 0.212      | —             |
| 5.0   | —         | **0.222**  | —             |

† temperature reset on resume，人为因素导致下降

**从这张表看，bs=8 目前数值最高（0.226）——但结论要小心：**

- bs=8 在 epoch ~1.8 达到峰值后就 **plateau** 了，epoch 2.0 之后没有继续涨
- bs=64 在 epoch 5.0 还在上升（每半 epoch 涨约 0.01），距离 0.226 只差 0.004
- bs=16 v4 在 epoch 1.5 只有 0.196，trajectory 很陡，还在快速上升，没有 plateau 迹象

**结论（当前）：** bs=8 收敛最快，但也 plateau 最早。bs=16/bs=64 收敛慢，但上限可能更高。理论上更多 negatives 应该给更高的上限（更难的对比任务 → 更好的 representation），实验还需要再跑几个 epoch 才能确认。
