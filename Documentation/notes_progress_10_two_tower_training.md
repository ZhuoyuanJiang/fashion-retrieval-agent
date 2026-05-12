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
