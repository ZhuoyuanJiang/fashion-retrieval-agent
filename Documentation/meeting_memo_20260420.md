下面是**只保留 TODO / execution plan** 的版本。按照 Nima 的反馈，当前最重要的不是继续讨论 idea，而是先做出一个 **text-based baseline**，然后再进入 contrastive training。Nima 明确建议：**先不要把 modification text 转成 audio；先让 text 版本跑起来，看 metrics，再决定下一步。** 

---

# Core TODO Summary

## Priority 0 — 先锁定方向

### TODO 0.1 — 暂时不要做 audio conversion

现在不要先把 dataset 里的 modification text 转成 audio。

Nima 的理由是：这个 task 里 audio 不是天然不可替代的输入。别人可能会问：为什么不直接把用户语音 transcribe 成 text，然后用 text + image 做 retrieval？所以现在更合理的路线是：

```text
先做 text modification version
→ 让 retrieval task work
→ 看 baseline metrics
→ 再加 audio as another layer of complexity
```

所以当前版本应该是：

```text
reference image + modification text → retrieval
```

不是：

```text
reference image + audio modification → retrieval
```

---

## Priority 1 — 先做 baseline，而不是马上 train contrastive model

### TODO 1.1 — 做 caption-based retrieval baseline

这是 Nima 最明确建议先做的事情。

Baseline 逻辑是：

```text
reference image + modification text
→ prompt VLM generate target caption
→ encode generated caption
→ compare with target captions in database
→ retrieve target item
```

你们现在应该先实现这个 pipeline，然后看 retrieval metrics。Nima 说这个 baseline 可能已经很强，也可能很差；不管是哪种情况，都需要先知道结果，才能判断后面 contrastive learning 有没有必要、能不能 beat baseline。

---

### TODO 1.2 — 用 FACap 的 target caption 建 retrieval database

因为 FACap 已经有 target caption，所以 baseline 需要先把所有 target captions 编码成 embeddings。

具体要做：

```text
For every target item:
    load target caption
    encode target caption with text encoder
    save caption embedding
    save mapping: embedding → target image / target id
```

这个 database 是之后 retrieval 用的。

---

### TODO 1.3 — 给 VLM 写 prompt，让它生成 target caption

对每个 sample，输入：

```text
reference image
modification text
```

让 VLM 输出：

```text
caption of the target image
```

Prompt 可以类似：

```text
Given the reference fashion image and the modification instruction,
write a concise caption describing the target fashion item after applying the modification.
```

重点是：不要让 VLM 描述 reference image 本身，而是要描述 **modified target item**。

---

### TODO 1.4 — 做 text-to-text retrieval

把 VLM 生成的 target caption 编码成 embedding，然后和 database 里的 target caption embeddings 做 similarity search。

流程：

```text
generated target caption
→ text encoder
→ generated caption embedding
→ compare with all target caption embeddings
→ rank target items
→ check whether ground-truth target is retrieved
```

可以先用 cosine similarity。

---

### TODO 1.5 — 先看 metrics

Nima 明确说：**Let’s look at the metrics first. That would be our baseline.** 

建议你们至少记录：

```text
Recall@1
Recall@5
Recall@10
Recall@50
Mean Rank / Median Rank
```

这些不是 transcript 里逐个列出的，但对于 retrieval task 是最自然的 evaluation metrics。

这个 baseline report 应该成为第一个 deliverable。

---

# Priority 2 — Dataset / dataloader 相关 TODO

## TODO 2.1 — 把 FACap dataset 正确 load 起来

Nima 说下一步首先要：

```text
get the dataset loaded properly
```

你们的 dataloader 每个 sample 至少应该返回：

```python
{
    "candidate_image": reference image,
    "modification_text": modification instruction,
    "target_image": target image,
    "target_caption": target caption,
    "target_id": target identifier
}
```

这里的重点不是重新 explore dataset，而是把 dataset 变成可以直接喂给 baseline / training loop 的形式。

---

## TODO 2.2 — 检查 batch 是否正确

在进入训练之前，要先确认 batch 组织是对的。

对于 baseline，batch 可以包含：

```python
candidate_images
modification_texts
target_captions
target_ids
```

对于后面的 contrastive learning，batch 会更重要，因为 batch 内其他 samples 会变成 negatives。

---

# Priority 3 — Baseline 完成后，再写 custom training loop

Nima 明确说：这个 project 不能直接依赖 Hugging Face 的默认 training setup。你们可以继续用 Hugging Face model，但是训练流程本身要自己写。

---

## TODO 3.1 — 写自己的 training loop

Nima 说 training loop 应该是一个 simple for-loop，大概包括：

```python
for batch in dataloader:
    outputs = model(...)
    loss = compute_loss(...)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

但这里的重点是，你们要自己控制：

```text
how to sample data
how to construct batch
how to do forward pass
how to extract embeddings
how to calculate loss
how to update model weights
```

这一步是 contrastive training 的基础。

---

## TODO 3.2 — 不要直接用 Hugging Face Trainer 当黑盒

Nima 的意思不是不能用 Hugging Face model，而是不能只用默认 Trainer，因为你们的 loss 不是普通 language modeling loss。

你们需要自己处理：

```text
triplet data
query embedding
target embedding
contrastive similarity matrix
contrastive loss
```

所以代码结构应该是：

```text
Hugging Face model: 可以用
Default Hugging Face training loop: 不建议直接依赖
Custom PyTorch training loop: 需要自己写
```

---

# Priority 4 — Contrastive Learning 版本的 TODO

等 baseline 做完、metrics 出来之后，再进入这一部分。

---

## TODO 4.1 — 定义 query embedding

Contrastive version 里，query 是：

```text
reference image + modification text
```

模型要把它变成一个 embedding：

```text
query_embedding = model(reference_image, modification_text)
```

这个 embedding 应该接近正确 target，远离错误 targets。

---

## TODO 4.2 — 定义 target embedding

你们需要决定 target side 用什么表示。

可能有两种：

```text
Option A: target image embedding
Option B: target caption embedding
```

因为 FACap 有 target caption，所以一开始用 target caption embedding 可能更容易。但如果你们想做真正 image retrieval，也可以考虑 target image embedding。

这需要你们根据 baseline 结果和 model setup 决定。

---

## TODO 4.3 — 实现 contrastive loss

假设 batch size 是 `N`：

```text
query_embeddings: N x D
target_embeddings: N x D
```

计算 similarity matrix：

```python
logits = query_embeddings @ target_embeddings.T
```

那么：

```text
logits[i][i] = correct pair
logits[i][j] = incorrect pair when i != j
```

loss 的目标是：

```text
make diagonal scores high
make off-diagonal scores low
```

也就是：

```text
query_1 close to target_1
query_1 far from target_2, target_3, ...
```

Nima 特别强调：**loss calculation is very important.** 

---

# Priority 5 — 需要思考的 model design choice

## TODO 5.1 — 决定用 last-token embedding 还是 projection layer

Nima 提到一个需要你们之后思考的 technical choice：

```text
Should we add one additional projection layer on top of the VLM
to generate the embedding before contrastive learning?

Or should we just use the embedding of the last token in the sequence?
```

也就是两种方案：

### Option A — Last token embedding

```text
VLM hidden states
→ take last token hidden state
→ use as retrieval embedding
```

优点：

```text
simple
easy to implement
good first experiment
```

缺点：

```text
not necessarily optimized for retrieval
```

---

### Option B — Projection layer

```text
VLM hidden representation
→ projection layer
→ retrieval embedding
```

优点：

```text
more retrieval-specific
more standard for contrastive learning
```

缺点：

```text
more parameters
more complexity
may need more training stability
```

建议执行顺序：

```text
先试 last-token embedding
如果效果不好，再加 projection layer
```

---

# Priority 6 — Batch size / multi-GPU 相关 TODO

Nima 对 contrastive learning 最大的担心是：**convergence**。他说 contrastive learning 不容易，尤其是因为它通常需要 large batch。

---

## TODO 6.1 — 先用 one GPU 跑通，不要一开始就 multi-GPU

Nima 明确说：

```text
do it for one GPU only first
make sure the code is not buggy
once you have enough confidence, then use more than one GPU
```

所以执行顺序应该是：

```text
1 GPU sanity check
→ make sure dataloader works
→ make sure forward pass works
→ make sure loss works
→ make sure training loss decreases
→ then think about multi-GPU
```

---

## TODO 6.2 — 不要以为 gradient accumulation 一定能解决 contrastive batch size 问题

Nima 提到 small batch 是问题，并且 gradient accumulation 不一定是 right fit。

原因是 contrastive learning 依赖 **same batch 里的 negatives**。

比如 batch size 是 8：

```text
each query only sees 7 in-batch negatives
```

就算你 gradient accumulation 4 次，如果每次 loss 都只在 8 个 samples 内计算，那么每个 query 还是只看到了 7 个 negatives，而不是 31 个 negatives。

所以对于 contrastive learning，真正有用的是：

```text
larger actual contrastive batch
```

或者：

```text
global batch across GPUs
```

---

## TODO 6.3 — 之后研究 data-parallel training + global contrastive loss

Nima 说之后可能需要：

```text
data-parallel training
global contrastive loss across GPUs
```

意思是，如果有多张 GPU：

```text
GPU 1: local batch
GPU 2: local batch
GPU 3: local batch
GPU 4: local batch
```

你们需要把所有 GPU 上的 embeddings gather 起来，然后一起算 contrastive loss。

例如：

```text
local batch size = 8
number of GPUs = 4
global batch size = 32
```

contrastive loss 应该在 32 个 samples 上算，而不是每张 GPU 各自只在 8 个 samples 上算。

---

## TODO 6.4 — 之后看 Hugging Face Accelerate 文档

Nima 说 Hugging Face Accelerate 应该支持这种多 GPU training，你们之后需要自己看 documentation，然后 figure it out。

但这不是现在第一步。

现在第一步是：

```text
one-GPU version works
```

之后才是：

```text
Accelerate
multi-GPU
global contrastive loss
larger batch
better accuracy
```

---

# Suggested Execution Order

你们现在可以按这个顺序做。

## Step 1 — Prepare FACap dataloader

Deliverable:

```text
A working dataloader returning:
reference image
modification text
target image
target caption
target id
```

---

## Step 2 — Build target caption embedding database

Deliverable:

```text
All target captions encoded into embeddings
A mapping from embedding index to target image / target id
```

---

## Step 3 — Run VLM caption-generation baseline

Deliverable:

```text
For each query:
reference image + modification text
→ generated target caption
```

---

## Step 4 — Run text-to-text retrieval

Deliverable:

```text
generated caption embedding
→ nearest target caption embeddings
→ top-k retrieved target ids
```

---

## Step 5 — Report baseline metrics

Deliverable:

```text
Recall@1
Recall@5
Recall@10
Recall@50
qualitative success/failure examples
```

This is the first serious checkpoint.

---

## Step 6 — Decide whether to train contrastive model

Based on baseline:

```text
If baseline is strong:
    need a stronger contribution to beat it

If baseline is weak:
    contrastive learning has clear motivation
```

---

## Step 7 — Implement one-GPU contrastive training loop

Deliverable:

```text
custom PyTorch training loop
query embeddings
target embeddings
contrastive loss
training loss curve
validation retrieval metrics
```

---

## Step 8 — Try embedding design choices

Start simple:

```text
last-token embedding
```

Then try:

```text
projection layer on top of VLM
```

Compare retrieval performance.

---

## Step 9 — Scale to multi-GPU only after one-GPU works

Deliverable:

```text
Hugging Face Accelerate setup
global embedding gathering
global contrastive loss
larger effective contrastive batch
```

---

## Step 10 — Add audio later

Only after text version works:

```text
replace modification text with audio input
or compare direct audio input vs ASR-transcribed text
```

This becomes the “extra layer of complexity,” not the starting point.

---

# What You Should Do First This Week

The most practical immediate TODO list is:

1. **Write FACap Dataset class.**
2. **Write DataLoader.**
3. **Build target-caption embedding database.**
4. **Prompt VLM with reference image + modification text to generate target caption.**
5. **Encode generated caption.**
6. **Retrieve nearest target captions.**
7. **Compute Recall@K.**
8. **Save qualitative examples: good retrievals and bad retrievals.**
9. **Only after that, start custom contrastive training loop.**

The key mentor feedback is: **baseline first, metrics first, text first, custom training loop later, multi-GPU even later.**
