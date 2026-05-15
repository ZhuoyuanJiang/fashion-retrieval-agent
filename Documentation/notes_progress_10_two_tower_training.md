# Notes — Plan-10 Two-Tower Training Concepts

Personal study / reference notes that came up during Plan-10 V1 design (see `Documentation/Plan_10_20260510.md`). Each entry follows the structure: **question → context → answer (verbatim from the original explanation)**.

---

## Q1: When backward runs on the loss, does the backbone get changed twice using the same loss? Is that what we want?

### Context

This came up during the 2026-05-10 design discussion of Plan-10 V1 (see `Documentation/Plan_10_20260510.md` §4.3, "Weight sharing — Q1, TBD"). The plan considers two architectures:

- **Option A — Shared backbone + separate LoRA adapters + separate heads.** One frozen `Qwen2-VL-7B-Instruct` backbone in GPU memory. Two LoRA adapters attached via PEFT: `lora_query` and `lora_target`. Two projection heads: `head_query` and `head_target`. The same backbone forward is invoked twice per training step, with `model.set_adapter("query")` before the query forward and `model.set_adapter("target")` before the target forward.
- **Option B — Two completely separate Qwen2VL instances.** Two independent `ContrastiveQwen2VL` objects. Each has its own backbone weights, own LoRA, own head. No PyTorch graph sharing between them.

The confusion was specifically about Option A: if there's only one backbone but it's invoked twice per step (once with `lora_query` active, once with `lora_target` active), and both forward passes feed into the same scalar loss, does that mean the backbone gets "updated twice" by backward? And is that the intended behavior?

The answer also clarifies how Option A relates to Option B from an autograd perspective — they end up doing the same thing to the trainable parameters; the only difference is whether the frozen substrate is one copy or two.

### Answer

Great question — and the short answer is **no, the backbone is not updated at all** (it's frozen). Let me be more precise about what actually changes.

#### What's frozen vs trainable in Option A

```
backbone (Qwen2-VL-7B)            ← FROZEN, requires_grad=False everywhere
   ├─ lora_query (A_q, B_q)       ← trainable
   └─ lora_target (A_t, B_t)      ← trainable
head_query                        ← trainable
head_target                       ← trainable
```

The backbone receives **zero** gradient. Its weights are static for the entire training run.

What gets updated per step:
- `lora_query`: gradient computed from `q_emb`'s subgraph → **updated once**
- `lora_target`: gradient computed from `t_emb`'s subgraph → **updated once**
- `head_query`: gradient from `q_emb`'s subgraph → **updated once**
- `head_target`: gradient from `t_emb`'s subgraph → **updated once**

Each trainable parameter is updated **exactly once per `optimizer.step()`**, with gradient contributed by **exactly one** of the two forward subgraphs.

#### "But the gradient flows through the backbone twice — doesn't that matter?"

This is probably what's bothering you. Yes, gradient *passes through* the frozen backbone activations twice during backward — once on the path from `q_emb` back to `lora_query`, once on the path from `t_emb` back to `lora_target`. But "passing through" ≠ "updating". The backbone is just a frozen tensor highway that gradient travels along to reach the trainable LoRA leaves.

Concretely, inside each layer, the forward is:

```python
y = base_layer(x)                    # frozen weights, no grad accumulated
    + active_lora.B @ active_lora.A @ x   # LoRA weights, grad accumulates here
```

During backward, the chain rule routes gradient through both terms, but since `base_layer.weight.requires_grad = False`, PyTorch just doesn't bother accumulating the gradient for it. The LoRA branch is where gradient actually lands.

#### "Is this what we want?"

Yes. Here's the mental model:

- `lora_query` learns: "how to perturb the frozen backbone so it produces good *query* embeddings."
- `lora_target` learns: "how to perturb the same frozen backbone so it produces good *target* embeddings."

Each adapter is specializing the *same* base model in a *different* direction. Two specializations of the same shared substrate. This is exactly the point of Option A — share the expensive 15 GB of backbone weights, let each adapter (~150 MB) carve out its own specialization for its tower's role.

It's not "training the backbone twice with the same loss". It's "training two small adapters that each modify the frozen backbone's behavior differently". The backbone itself is just the shared foundation both towers stand on; it doesn't move.

#### How this differs from Option B

| | Option A | Option B |
|---|---|---|
| Backbone weights | One frozen copy, shared | Two frozen copies, identical |
| Trainable for query tower | `lora_query` + `head_query` | `lora_query` + `head_query` |
| Trainable for target tower | `lora_target` + `head_target` | `lora_target` + `head_target` |
| Gradient updates per step (trainable) | 4 param groups, each updated once | 4 param groups, each updated once |
| Backbone gradient | Zero (frozen) | Zero (frozen, ×2 copies) |

**From the trainable-parameter perspective, A and B are doing the exact same thing.** The only difference is whether the frozen substrate is one copy or two. Option A is strictly cheaper in VRAM and otherwise equivalent in optimization behavior.

The only real concern with Option A is: can the *same* base activations support both query-style and target-style specialization simultaneously? Empirically, yes — this is exactly how multi-adapter PEFT is used in production (DPR with shared encoder, multi-task LoRA fine-tuning, etc.). The adapters have enough capacity (~150 MB each, rank 32) to carve out their own behaviors on top of the shared substrate.

#### TLDR

**Nothing gets updated twice; the backbone doesn't get updated at all; each adapter is updated once from its own forward pass, and they specialize the same shared backbone in different ways.**

---

## Q2: Why can one loss backprop to two models? Is that allowed?

### Context

This came up earlier in the same 2026-05-10 design discussion, before drilling into Option A's adapter-switching mechanics. The dual-encoder pattern feels unusual to anyone coming from single-model training: a query tower produces `q_emb` from one set of inputs, a target tower produces `t_emb` from another set of inputs, then they get combined in a single contrastive loss and `loss.backward()` is called. Does PyTorch actually handle this correctly, or do we need to define two separate losses and run two backward passes?

The question applies to **both Option A (shared backbone, two adapters) and Option B (two separate Qwen2VL instances)** — the graph-level answer is the same in both.

### Answer

From PyTorch's perspective there's no concept of "model" — there's only:
- A **computation graph** (DAG of operations from inputs to loss)
- **Leaf parameters** with `requires_grad=True`
- One **scalar loss** at the end

`loss.backward()` walks the graph backward from the loss and accumulates gradients in every reachable leaf. It doesn't care whether the leaves live in "one Python object" or "two Python objects" — it just walks the graph topology.

In Plan-10:
```
batch ──→ query forward  ──→ q_emb ─┐
                                     ├─→ loss ──→ backward()
batch ──→ target forward ──→ t_emb ─┘
            (different leaves)
```

The query forward creates a subgraph with `lora_query + head_query` as leaves. The target forward creates a subgraph with `lora_target + head_target` as leaves. `loss` depends on both `q_emb` and `t_emb`, so backward walks both subgraphs in parallel and updates all four sets of trainable params in one `optimizer.step()`. **This is exactly how CLIP / DPR / SBERT work — it's the canonical dual-encoder training pattern.**

The "two models" framing is just Python-side organization. PyTorch sees one big graph.

---

## Q3: With two trainable towers, how do we know if there's overfitting? What if one loss goes up and one goes down?

### Context

In Plan-6, only the query tower was trainable; the target tower was a frozen FashionCLIP image encoder. Dev loss was the primary overfitting signal — and Plan-7's data confirmed it works diagnostically (server-6 dev/loss bottomed at epoch 10 then drifted up while train loss kept decreasing, a textbook overfit pattern).

Plan-10 V1 makes BOTH towers trainable. The natural worry: do we now need separate overfit signals per tower? What does it mean if "one loss goes up and one goes down"?

### Answer

The misconception worth busting: **there is only ONE loss.** Not "query loss + target loss". The contrastive loss is a single scalar that depends on both `q_emb` and `t_emb` together. So "one going up, one going down" is not a thing here.

Overfitting signals (unchanged from Plan-6/7):

| Signal | What it tells us |
|---|---|
| `train/loss` | Should decrease smoothly. Logged every step. |
| `dev/loss` | **Primary signal**. Computed on 500 dev items with on-the-fly target forwards. If train/loss falls while dev/loss starts to rise → overfit. Same dynamics Plan-7 already proved diagnostic. |
| `dev/R@K`, `headline/R@K` | Noisier; should plateau as overfit begins. |
| `tau_inv` | Watches for **embedding collapse** (both towers learning constant outputs). If `tau_inv` rises aggressively while loss is stuck near `log(N)`, suspect collapse. |

You don't need a separate signal per tower. If anything overfits, it's the **joint system**, and `dev/loss` measures exactly that.

(Advanced diagnostic you can add for free if you want: log per-param-group gradient norms — `‖∇lora_query‖` vs `‖∇lora_target‖` — to see if one side dominates. But not required for V1.)

---

## Q4: What is `gallery_emb_epoch0.npy`? Why does the run dir have one of these per epoch?

### Context

When you look at a Plan-10 / Plan-12 run dir, alongside the `ckpt_epoch*` checkpoint directories you also see 19 files named `gallery_emb_epoch{0..18}.npy`, each ~116 MB. They're not checkpoints. What are they, and why one per epoch?

### Answer

`gallery_emb_epoch0.npy` is a `(59048, 512)` float32 numpy array. Each row is the **encoded vector for one of 59,048 candidate fashion images** in the FACap dress slice. It's the precomputed "fingerprint" of the entire retrieval search space.

**Why we need it.** The task is composed-image retrieval (CIR): given a query of `(reference_image, "but in blue")`, find which of 59,048 candidate images is the right answer. The model produces a 512-dim "query embedding"; we want to rank the 59,048 candidates by similarity to it (cosine, since everything is L2-normalized).

But re-running the target tower over all 59,048 candidates **every time you score a query** would be brutal: 59,048 × ~50 ms = ~50 min per query. Instead we **precompute** the gallery embeddings once and store them. At eval time, we just compute the query embedding (one forward pass, ~50 ms) and dot-product it against the cached 59,048-row matrix (a single matmul, ~10 ms). That's the speedup that makes retrieval feasible at 1500 queries per eval.

**Why the `epoch0` suffix.** In Plan-6 the target tower was frozen FashionCLIP, so its embeddings never changed and one file was enough. In Plan-10 V1 the **target tower is trainable** (this is what Nima asked for over Plan-6, per `Documentation/meeting_memo_20260503.md`). So the gallery embeddings change as training progresses — a candidate that scored 0.70 at epoch 1 might score 0.85 at epoch 5 because the target tower has learned better representations.

So we re-encode the entire gallery at the **start** of training (`gallery_emb_epoch0.npy` = random target tower init, embeddings are essentially random) and again at the **end of each training epoch** (`gallery_emb_epoch1.npy`, `gallery_emb_epoch2.npy`, … `gallery_emb_epoch18.npy`). That's why your Plan-12 run dir has 19 of these files — startup + 18 epoch-end refreshes.

In `src/training/train_plan10.py` the calls are at `:388` (`epoch_tag=0` startup encode) and `:561` (`epoch_tag=epoch+1` end-of-epoch refresh).

The "first informative R@K" comes from `gallery_emb_epoch1.npy` (after one full epoch, when the target tower has learned something nontrivial). That's why the Plan-12 sanity check insists on `epoch1.npy` not `epoch0.npy` — `epoch0` just verifies the encoding pipeline ran; `epoch1` verifies the refresh hook + actual training are both working.

**One sentence to remember:** the gallery is the search space; the gallery embedding is its precomputed fingerprint; we refresh it each epoch because the target tower keeps learning.

---

## Q5: What's actually using all that GPU memory? Are we secretly loading two models?

### Context

Looking at `nvidia-smi` during a Plan-10 V1 / Plan-12 run, each of the 8 GPUs shows **~33 GB / 48 GB used** even *before* training has properly started (during the initial gallery encoding phase, when activations should be tiny). The number sounds suspicious — almost as if two copies of the 7B model are loaded. Are we doing model-parallel? Are the two PEFT adapters secretly doubling the base?

### Answer

**No, we are not loading two models.** With `--num_processes 8` accelerate uses **DDP**, not model parallelism — each of the 8 GPUs has its own *one* copy of the base model. The "two adapters" trick (`get_peft_model` + `add_adapter`) attaches two **small** correction-matrix sets on top of the *same* base weights, not two copies of the 7B.

Per-GPU memory breakdown (DDP style — each GPU has its own copy of everything below):

| Component | Approximate size | What it is |
|---|---|---|
| **Base Qwen2-VL-7B** in bf16 (frozen) | ~14-16 GB | The 7B-parameter foundation model. Loaded once. |
| **+ Stage-2 LoRA merged in** | (already counted above) | Stage-2 weights got folded into the base via `merge_and_unload()` at load time. They're not a separate model. |
| **Audio encoder** (it's `Qwen2-VL-7B-Speech`!) | ~1-2 GB | Even though we don't use audio in Plan-12, the speech-extended checkpoint has an audio encoder. Loaded but inactive. |
| Vision encoder | (already counted in the 7B) | Part of the model. |
| **LoRA adapter "query"** (bf16) | ~60 MB | Rank-32 LoRA on q/k/v/o, ~32M params × 2 bytes |
| **LoRA adapter "target"** (bf16) | ~60 MB | Same dimensions; second named adapter on the **same** base weights |
| Projection head "query" (fp32) | ~14 MB | 3584 → 1024 → 512 MLP |
| Projection head "target" (fp32) | ~14 MB | Same shape |
| **AdamW optimizer state** (fp32) | ~1 GB | Trainable params only (~64M), AdamW keeps two fp32 moments per param: 64M × 4 bytes × 2 = ~0.5 GB |
| Gradients (bf16) | ~0.13 GB | Trainable params again |
| PyTorch CUDA allocator cache / fragmentation | a few GB | The CUDA caching allocator reserves memory in blocks; what `nvidia-smi` shows is *reserved*, not strictly active |
| NCCL communicators, DDP buckets | ~1 GB | Distributed comms buffers |
| Gallery encoding activations (when active) | a few GB | The 59,048-image encoding pass runs forward passes through the target tower in chunks |

**Subtotal at steady state**: ~18-22 GB *active*, with the caching allocator reserving ~10 GB more in cached blocks. Reported total: ~30-33 GB.

### Why "two adapters" doesn't mean "two models"

This is the part worth burning into the mental model. A PEFT LoRA adapter is **not** a second copy of the base model. It's a tiny pair of low-rank matrices that get **added on top of** specific layers of the base model at forward time.

Concretely, for each attention projection (e.g., `q_proj`, a `4096 × 4096` weight matrix in the base):

```
Base:    W   (4096 × 4096) = 16M params
LoRA A:  A   (4096 ×   32) = 131K params   ← rank-32
LoRA B:  B   (  32 × 4096) = 131K params   ← rank-32
Effective forward: x → W·x + B·(A·x)
```

The LoRA adapter is two skinny matrices (`A` and `B`) whose product `B·A` is a **low-rank correction** to `W`. Rank 32 means the correction lives in a 32-dimensional subspace. The total LoRA addition per attention projection is `131K + 131K = 262K` params vs the base's `16M` — about **1.6%** of the base size.

So "two PEFT adapters" means: each attention projection has the base `W` (one copy, frozen) plus **two** sets of rank-32 corrections (`A_query, B_query` and `A_target, B_target`). At forward time:
- When `active_adapter == "query"`: forward is `x → W·x + B_query·(A_query·x)`
- When `active_adapter == "target"`: forward is `x → W·x + B_target·(A_target·x)`

Same base `W`, different correction matrix. The "shared backbone" name in Plan-10 V1 Option A / Plan-12 means *exactly this*: one copy of the base ~14 GB of weights, two small correction sets we toggle between.

For comparison, Option B (separate backbones) literally **does** load two base models. That's why Option B's per-GPU memory at bs=8 was similar (~30-33 GB) — Option B paid 2× the model size but didn't use gradient checkpointing (so activations were larger); Plan-12 saves the second-model memory but burns it back on bigger activations + CUDA cache. The bottom line ends up similar.

### Self-check: how to verify you're not loading two backbones

Look at the model-loading log. Plan-12 prints exactly one of:

```
Loading speechQwen2VL base + merging Stage-2 LoRA (Option A)...
TwoTowerSharedBackbone ready: d_target=512, trainable params ≈ 64.5M
```

Per rank, you'll see this **once**, not twice. If we were loading two backbones (Option B), the equivalent line `Loading speechQwen2VL base + merging Stage-2 LoRA (...)` would print **twice** per rank, and the wall-clock load time would be ~2× longer (~60 s vs ~30 s).

**One sentence to remember:** ~14 GB is the base model, ~120 MB total is both LoRA adapters, the rest of the ~33 GB is optimizer state + gradients + CUDA allocator cache + activations during the forward pass.

---

## Q6: Why doesn't `nvidia-smi memory.used` scale linearly with batch size?

### Context

During the Plan-13 batch-size scan, we observed something that looked wrong. We ran the same training script (shared backbone, gradient checkpointing on) at three batch sizes and watched per-GPU memory via `nvidia-smi memory.used`:

| bs | effective negatives (× 8 GPUs) | observed max memory | step on the curve |
|---|---|---|---|
| 8 (Plan-12) | 64 | **~33 GB** | reference |
| 16 | 128 | **~33 GB** ← unchanged! | bs doubled, memory didn't move |
| 24 | 192 | **~43.5 GB** ← big jump | bs went up 50%, memory jumped 32% |

Doubling batch from 8→16 cost zero memory; going 16→24 cost ~10 GB. That can't be right — activation memory should scale roughly linearly with batch size. So what is `nvidia-smi memory.used` actually showing us?

### Answer

`nvidia-smi memory.used` is **not** what most people think it is. It reports **memory reserved by the CUDA driver for your process** — the pool the GPU has handed over to PyTorch. It does **not** show the bytes your live tensors are actively occupying.

The relevant distinction inside PyTorch:

| Number | What it means | Behavior |
|---|---|---|
| `torch.cuda.memory_allocated()` | Bytes currently held by live tensors | Goes up and down as tensors are created/freed |
| `torch.cuda.memory_reserved()` | Bytes the caching allocator has grabbed from the driver | **Monotonically grows** (almost never returns memory to the driver) |
| `nvidia-smi memory.used` | ≈ `memory_reserved` + a few hundred MB for CUDA context | Same monotonic-growth behavior |

So when we see "33 GB used at bs=8" and "33 GB used at bs=16", that doesn't necessarily mean both runs have 33 GB of *active* tensor memory. It means **the caching allocator's reservation high-water mark was 33 GB at both batch sizes**. The actively-used memory could be quite different between the two.

#### Why the caching allocator works this way

When PyTorch needs memory:
1. It first checks its **cache** for a free block of the right size.
2. If found, reuse it. (Cheap, no driver call.)
3. If not found, ask CUDA for a new block. (Expensive — `cudaMalloc` is slow.)
4. When a tensor is freed, the block goes back to the **cache**, not the driver. The allocator hangs onto it for future allocations.

This is necessary because `cudaMalloc` / `cudaFree` are very slow (~ms per call), and a training loop does thousands of allocations per step. Without caching, training would be unusably slow. So the design optimizes for speed at the cost of memory looking "wasted" from `nvidia-smi`'s perspective.

Block sizes are **rounded up to allocator-friendly chunks** (often 2 MB / 4 MB / 8 MB granularity for medium blocks, larger for big blocks). A 5.3 MB activation tensor and a 6.1 MB tensor might both get an 8 MB block — they look identical from the allocator's perspective.

#### Why bs=8 and bs=16 looked identical (both at 33 GB)

At bs=8, the run allocates activation tensors of sizes like `(8, 512, 3584)` for hidden states and `(8, 28, 512, 512)` for attention. The caching allocator rounds these up to its block sizes. Total reservation: ~33 GB.

At bs=16, the same tensor shapes have a bigger first dimension: `(16, 512, 3584)`. Activation memory has genuinely roughly doubled. **But**:

- **Gradient checkpointing** only keeps activations for one block at a time during recompute, so the in-flight activation memory is smaller than naive `bs × layers × hidden`.
- The new bigger tensors fit inside the same block-size buckets the allocator pre-reserved at bs=8.
- Most of the 33 GB is **batch-size-independent**: model weights (~14 GB) + AdamW state (~1 GB) + gradients (~0.1 GB) + NCCL buffers + CUDA context.

So `memory_allocated` went up at bs=16, but stayed inside the 33 GB the allocator had already reserved. No new `cudaMalloc` calls, no new reservation, same `nvidia-smi` number.

#### Why bs=24 finally crossed a threshold (jumped to 43.5 GB)

At bs=24, activations are big enough that **at least one** needs a block size that wasn't pre-reserved at bs=8. So:

1. Allocator looks for a free block of size X. Not found.
2. Allocator calls `cudaMalloc` for a new block (possibly rounded up to 1 GB or 2 GB chunk).
3. `nvidia-smi memory.used` jumps by that chunk size.

Once the chunk is reserved, it stays reserved forever (until process exits). That's why bs=24 jumped to 43.5 GB and **stayed there** — the allocator grabbed a few more chunks and is holding them.

### Practical takeaway

1. `nvidia-smi` shows an **upper bound on what you've ever needed**, not steady-state current usage.
2. The number is noisy across ranks (see Q7) because each rank sees different data and may trip different allocator thresholds.
3. **OOM risk depends on the highest-water-mark of any rank during the worst-case batch in the entire run**, not the typical case. That's why we leave several GB headroom even when bs=N looks comfortable.
4. If you want the real allocation curve, log `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` per step from inside the training loop. Drop-in patch (suggested for future debugging):

```python
if global_step % 20 == 0:
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_resv  = torch.cuda.memory_reserved()  / 1e9
    print(f"  mem alloc={mem_alloc:.1f}G reserved={mem_resv:.1f}G")
```

`alloc` will go up roughly linearly with batch size (tracks live tensors), while `reserved` jumps in chunks (tracks the caching allocator pool). The discrepancy is exactly what made our bs scan look weird from `nvidia-smi`.

**One sentence to remember:** the actual memory cost of going bs=8 → bs=16 → bs=24 is roughly linear; the *reservation pool* just expanded in two jumps instead of three because the allocator is a coarse step function on top of the smooth underlying curve.

---

## Q7: Why is per-rank GPU memory different across the 8 ranks?

### Context

During the Plan-13 bs=24 run on server 11, `nvidia-smi` showed a notable spread of memory across the 8 RTX 6000 Ada GPUs, with the same training script and the same model:

```
GPU  memory.used   delta vs rank-min
─────────────────────────────────────
 0   37463 MiB     +1322 MiB
 1   40411 MiB     +4270 MiB
 2   36141 MiB         min
 3   38883 MiB     +2742 MiB
 4   39051 MiB     +2910 MiB
 5   40357 MiB     +4216 MiB
 6   43487 MiB     +7346 MiB   ← outlier
 7   40273 MiB     +4132 MiB
```

The spread is **7.3 GB between rank 2 (lowest, 36.1 GB) and rank 6 (highest, 43.5 GB)**. All ranks ran identical training code, on identical hardware, with the same model weights. Why don't they have identical memory footprints?

### Answer

The dominant factor is **per-rank data variance combined with Qwen2-VL's variable-shape inputs**. Smaller contributors include per-rank CUDA context and NCCL buffers (those add ~1 GB of spread total, but the bulk of the 7 GB is the data story).

#### DDP's data sharding means each rank sees different data

With `--num_processes 8`, each step has 8 ranks processing **non-overlapping shards** of the same global batch:

```
global batch (192 samples at bs=24 × 8 GPUs)
  ├── rank 0: samples 0-23
  ├── rank 1: samples 24-47
  ├── rank 2: samples 48-71
  ├── ...
  └── rank 7: samples 168-191
```

`DistributedSampler` shuffles deterministically per epoch and gives each rank a different subset. Crucially, **none of these shards are coordinated to have the same activation memory footprint** — they're just whatever index ranges of the shuffled dataset.

#### Why per-rank data variance matters specifically for Qwen2-VL

Most pure-text LMs have a fixed input shape — pad text to N tokens and every batch has the same activation tensor sizes. Qwen2-VL doesn't:

1. **Variable image resolution.** Qwen2-VL uses *dynamic image tokenization* — it preserves aspect ratio and produces between ~256 and ~1280 vision tokens per image depending on pixel dimensions. A 224×224 thumbnail might become 256 vision tokens; a 1280×960 product photo might become 1200+.
2. **Variable text length.** The modification text varies from "make it red" (~5 tokens) to longer descriptive edits (~50 tokens). After padding to the longest sample in the batch, the text portion can be 5-100 tokens.
3. **Padding is per-batch, not global.** The image processor and tokenizer pad to the longest sample **in that rank's batch**, not globally. So if rank 6 happens to have a high-resolution image in its shard, *its* tensors are bigger than rank 2's.

The activation tensors have shape proportional to `(batch × seq_len × hidden)`. Attention is the worst offender — its activation is `O(seq_len² × hidden_dim)`, so a 30% bigger sequence becomes ~70% bigger attention activation.

#### The stickiness — why rank 6 stays high forever

Connects to Q6's caching-allocator behavior. The flow on rank 6 was likely:

1. **Early in training** (epoch 0, first ~50 batches), rank 6 got at least one batch with a few unusually large images.
2. Activation tensors for that batch needed bigger memory blocks than the allocator had cached.
3. Allocator called `cudaMalloc` to grab new chunks → reserved memory on rank 6 jumped to ~43.5 GB.
4. **The pool never shrinks.** Even when subsequent batches on rank 6 are smaller, the reservation stays at the 43.5 GB high-water mark.

Rank 2, meanwhile, never happened to draw a batch quite that big. Its allocator never had to expand past ~36 GB. So rank 2 sits at the low end for the rest of the run.

The pattern is deterministic per random seed — same shuffle would produce the same per-rank spread. It can't really change mid-training because once a high-water mark is set, it stays.

#### Smaller contributors (not the main story)

| Source | Typical spread | Why |
|---|---|---|
| CUDA context (per process) | ~300-600 MB | First `cudaMalloc` allocates the context; per-process |
| NCCL communicator buffers | ~100 MB | Slightly different per rank position |
| cuBLAS / cuDNN handles | ~50 MB | Loaded lazily, per process |

Total: ~1 GB of cross-rank variance from these. The other ~6 GB is the image-size story.

#### What you could do about it (if you cared to)

| Approach | Effect |
|---|---|
| **Resize all images to a fixed resolution** before training | Eliminates ~90% of the variance. Loses Qwen2-VL's variable-resolution advantage (might slightly hurt model accuracy). |
| **Bucketing sampler** — sort batches by image size so each batch has uniformly sized images | Reduces within-batch variance, but rank-to-rank shards still differ unless you also coordinate across ranks. |
| **Pre-warm the allocator** with a synthetic max-size batch at startup | Forces every rank to its peak reservation immediately, so OOM (if any) happens at startup, not mid-run. Easy to add. |
| **Live with it** (what Plan-12/13 do) | Use the max-rank memory as the bs ceiling. Annoying but tractable. |

For Plan-12/13 the right call has been "live with it" — we already need headroom for the high-water rank, and pre-warming or bucketing would add complexity without buying much.

### The deeper principle

When you're training on **structured data with uniform shape** (e.g., fixed-size MNIST images, fixed-length sequences), DDP gives you uniform memory across ranks. When you're training on **variable-shape data** (images of different sizes, sequences of different lengths, point clouds with different numbers of points), DDP exposes that variability via per-rank memory differences. The model isn't doing anything wrong — it's just that different ranks did slightly different amounts of work on slightly different inputs.

It's one of those things that's invisible in a fixed-size workload but shows up loudly with vision models and especially multimodal models like Qwen2-VL.

**One sentence to remember:** the 7 GB rank spread happens because each rank sees different data, image sizes vary across the dataset, the caching allocator never gives back reserved memory, so whichever rank drew the biggest images first stays at the highest reservation for the rest of training.
