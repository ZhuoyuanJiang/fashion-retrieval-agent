# Fashion Shopping Agent — Project Proposal

---

## Part 1: Motivation

当我在商店里逛街，看到喜欢的衣服时，我可能会先拍下来，然后再回头打字问模型有没有更便宜的类似款，或者有没有其他颜色。

但这种交互其实很慢，因为我需要不断停下来输入文字。

如果我能一边看衣服、一边直接说：

- "这件有没有更便宜的？"
- "有没有黑色？"
- "帮我找个更正式一点的类似款"

那整个流程会自然很多，也更适合 real-time shopping exploration。

---

When I shop for clothes in real life, I often see an item I like, take a quick photo of it, and later type questions like: *Is there a cheaper version of this?* or *Does it come in another color?* But that workflow is slow and awkward. Every follow-up requires me to stop browsing, type another query, and manually describe the item again.

A more natural shopping experience would let me keep looking at clothes and simply speak as I browse:

- "Is there a cheaper version of this?"
- "Do they have it in black?"
- "Can you find something similar but more formal?"

This kind of interaction is faster, more intuitive, and better suited to real-time shopping exploration. Instead of forcing users to translate what they see into typed search keywords, the system should understand both the garment in view and the spoken shopping request directly.

A better interface would support shopping in the way people naturally think and speak. Instead of pausing to type, the user could simply point the camera at an item and ask questions in real time, such as "Is there a cheaper version of this?" or "Can you find something similar in black?" This would make the interaction more fluid, reduce friction during browsing, and enable a more natural form of real-time shopping exploration.

### In other words

When people shop for clothes in stores, they may notice a garment they like, compare it with nearby items, and immediately want to ask follow-up questions such as whether it comes in another color, whether a cheaper alternative exists online, or whether there is a similar but more formal version. In practice, however, this workflow is usually broken by the interface: the user has to stop browsing, type a query, manually describe the garment, wait for response, and repeat the process for every refinement.

A better interface would support shopping in the way people naturally think and speak. Instead of pausing to type, the user could simply point the camera at an item and ask questions in real time, such as "Is there a cheaper version of this?" or "Can you find something similar in black?" This would make the interaction more fluid, reduce friction during browsing, and enable a more natural form of real-time shopping exploration.

---

## Part 2: Why speech + vision are both necessary

*(justifying why our solution was to add audio modality to a VLM)*

### Why speech is necessary

Speech is necessary because, in a real-time shopping setting, it is faster and more convenient than typed search. When a user is actively browsing, holding items, moving around a store, or quickly comparing multiple garments, stopping to type every question creates unnecessary friction. Voice makes it much easier to ask short, iterative follow-up questions in the moment.

This matters because shopping queries are often not fully specified upfront. Users naturally refine their intent step by step:

- "Is there a cheaper version of this?"
- "What about in black?"
- "Actually, something similar but more formal."
- "Not this one — the jacket on the left."

These are easier to say than to type, especially during active browsing. In this setting, speech is not just an extra convenience feature. It is the more natural interface for fast, multi-turn shopping exploration.

### Why vision is necessary

Vision is necessary because the user's request is grounded in a specific garment they are currently looking at. The system must understand:

- which item the user is referring to
- what visual properties it has
- how it differs from nearby items
- what visual similarity means for retrieval

Without vision, the system cannot resolve references like "this one," "that jacket," or "the dress on the left." It also cannot extract appearance cues such as color, silhouette, style, or texture. In that case, the problem collapses into a generic shopping chatbot rather than a visual shopping assistant.

### Our solution: add audio modality to a VLM

The key reason to add audio as a native modality to a VLM is so that the model can jointly interpret a visual scene and a spoken shopping query. This application requires the model to jointly interpret:

- a visual scene containing one or more garments
- a spoken query referring to one of them
- a shopping-related intent expressed conversationally

This is not just ASR plus image understanding in isolation. The value comes from linking **spoken referring expressions** to **visual grounding** in a real shopping workflow. This makes the system suitable for a real shopping workflow, where spoken referring expressions must be linked directly to visual grounding.

(I added audio modality so the model can directly localize the garment referred to in speech, instead of requiring the user to type or relying on a separate text-first interaction.)

---

## Part 3: Core Engineering Question

The core engineering question of this project should be framed at **two levels**. These two levels are closely connected, but they should not be collapsed into one. The first level defines the **overall end-to-end application goal**. The second level defines the **specific role of the fine-tuned audio-VLM within that larger system**.

This distinction is important because the end-to-end goal of the product does **not** necessarily mean that the fine-tuned model itself must solve the entire pipeline. Instead, the end-to-end goal should first define what kind of system we want to build, and then guide us in deciding what part of that system is most appropriate to assign to the current audio-extended VLM. That decision will directly shape the system design, and in turn determine the model's input, output, and fine-tuning target.

### Part 3.1: Overall picture (system-level scope) — what system do we ultimately want to build?

At the application level, the question is:

**What should a spoken visual shopping system ultimately do for the user?**

For this project, the overall end-to-end goal is to build a system that takes a user's spoken query together with the image or video they are currently viewing, and returns useful shopping results such as:

- cheaper alternatives
- other colors
- similar products
- more formal or more casual variants

**In short, the system-level scope is to return a final garment (or a ranked set of garments) that matches what the user actually wants** — not just intermediate signals like a transcription, a bounding box, or an intent label. This defines the **overall picture** of the project: not just speech transcription or garment recognition, but a spoken visual shopping workflow that supports real-time product exploration, ending in an actual retrieved product.

### Part 3.2: Model scope — what should the fine-tuned model actually do?

Once the overall system goal is clear, the next question is:

**Given that end-to-end goal, what is the right role for the fine-tuned audio-VLM?**

This is where we move from product thinking to system design and model task formulation.

The key point is that the end-to-end system does **not** imply that the fine-tuned model itself must perform the full shopping pipeline. For example, the final system may need to return cheaper alternatives, other colors, or similar products, but that does not mean the audio-VLM itself must directly generate product search results or complete recommendation outputs. **Those final results may come from downstream retrieval, search, or recommendation modules.** The fine-tuned model's job is therefore not to produce the final garment itself, but to produce **intermediate, retrieval-ready signals that downstream modules can act on** — for example, a target-oriented caption (Option 2) or a query embedding (Option 3).

So the model-scope question becomes:

**Within the overall spoken shopping system, what part of the problem should the audio-extended VLM be fine-tuned to solve?**

The same end-to-end application can be supported by very different model-level formulations. In the next section, we make this concrete by laying out three candidate task formulations for the audio-VLM, comparing what each one assigns to the model itself versus what it delegates to downstream modules.

---

## Part 4: Task Definition and Method Options

This section first defines the system-level task that the entire project is trying to solve, and then presents three candidate methods for what the fine-tuned audio-VLM should actually do inside that system.

A useful way to read this section is to keep three layers separate, because they are easy to conflate:

1. **System-level end goal** (does not change across options): retrieve the target fashion item from the catalog.
2. **Model-level training target** (this is what changes between Option 2 and Option 3): what the fine-tuned audio-VLM is trained to produce — a caption, or directly an embedding.
3. **Retrieval step** (always present): how the catalog database is actually searched.

> **完整论述见 Appendix F：三层分离 — system goal / model training target / retrieval step。**

### Part 4.1: System-level task definition

At the system level, the task is **audio-conditioned composed fashion retrieval**.

**System input:**

- a **reference garment image** (the item the user is currently looking at)
- a **spoken query** that refers to this item and expresses a shopping-related modification or request (e.g., "make it black," "find something similar but more formal," "longer sleeves")

**System output:**

- a ranked list of **top-k retrieved fashion items** from a catalog
- ideally, the most relevant item is the **target garment** that best matches the visual content of the reference image after applying the user's spoken modification

**Demo flow:**

- the user selects (or uploads) a reference image
- the user speaks a short modification query, for example:
  - "make it black"
  - "more formal"
  - "longer sleeves"
- the system returns the top-k retrieved items
- the demo UI also includes a sidebar with example reference images and example audio / text queries, so reviewers can quickly try the system without having to construct their own inputs

This system-level definition is **fixed across all three method options below**. What varies between the options is **only the responsibility assigned to the fine-tuned audio-VLM** — i.e., what part of this end-to-end pipeline the model is trained to solve, and what is delegated to other modules.

### Part 4.2: Three method options for the fine-tuned audio-VLM

We consider three candidate formulations for the role of the fine-tuned model:

1. **Option 1**: Spoken garment grounding + downstream tool / API call *(ruled out)*
2. **Option 2**: Caption-generation retrieval *(main method)*
3. **Option 3**: Direct contrastive embedding retrieval *(research extension)*

---

#### Option 1: Spoken garment grounding + downstream tool / API call *(ruled out)*

**Task formulation.** In this option, the audio-VLM is responsible only for resolving what the user is referring to in the image, and (optionally) what kind of shopping action they want. The actual retrieval, price comparison, or product search is delegated to downstream modules — for example, an external shopping API, a tool-calling agent, or a separate retrieval engine.

**Pipeline:**

1. The audio-VLM takes a reference image + spoken query and outputs a structured grounding result (e.g., a bounding box, possibly together with an intent label or a normalized query).
2. A downstream module (shopping API, search engine, tool-calling LLM) takes that structured output and performs the actual product search.
3. The system returns the search results to the user.

**Variants of model output.** Within this framing, the model output can take several forms, ranging from minimal to richer:

- bounding box only (pure spoken garment grounding)
- bounding box + coarse intent label (e.g., `find_cheaper`, `find_other_color`, `find_similar`)
- bounding box + normalized textual query (e.g., "Find cheaper black alternatives for this blazer.")
- bounding box + structured shopping constraints (color, price direction, style direction, fit, length, etc.)

> **完整 task specification、output schema、JSON 例子和每个变体的详细分析见 Appendix A：Detailed Task Variants for Option 1。**

**Why we rule this option out.** After discussion with my mentor, this entire route is ruled out as the main method, for the following reasons:

- The hardest and most novel part of this pipeline is **outsourced to downstream modules** (the shopping API or tool-calling agent). The audio-VLM's role collapses into a grounding component, which underuses the joint audio + vision modeling capability the project is supposed to demonstrate.
- The overall system becomes more of a **software engineering integration** ("detect the clothing item, then call an external API") than a research project. It does not add much new value beyond existing items on my resume.
- It also does not naturally connect to the kind of fine-grained fashion retrieval research that recent CIR (composed image retrieval) work — including the FACap paper — has been pushing forward.

For these reasons, Option 1 is preserved in the appendix as background and as a design space we explored, but it is **not** the proposed method of this project.

---

#### Option 2: Caption-generation retrieval *(main method)*

This is the **main method** of the proposal. It is the safest and most realistic version that still puts the audio-VLM at the center of the system.

##### 2.1 What the model is trained to produce

For Option 2, applying the three-layer view from the start of Part 4:

- The end goal is still to **retrieve the target fashion item**.
- The model's training target is to **generate a target-oriented caption** given (reference image + audio query).
- The retrieval step is performed by a **separate, frozen text embedding model** that encodes both the catalog captions and the model-generated query caption into the same text embedding space, and then runs nearest-neighbor search.

A natural question here is: *if the end goal is retrieval, why is the model trained to output a caption rather than directly outputting a caption embedding?* The short answer is that the audio-VLM is fundamentally a generative model — it is naturally good at consuming image + audio and producing text, not at producing dense vectors. So we let it do what it is best at (caption generation), and let a separate text embedding model handle the embedding step.

> **完整论述见 Appendix D：方法 A 中，模型输出的为什么是 caption 而不是 embedding？**

##### 2.2 The training input and output

Given:

- a reference garment image $I_r$
- a spoken modification query $a$

the model is trained to produce a **target-oriented caption** $c_t$ — a short textual description of the item the user actually wants to retrieve.

Crucially, this caption is **not** a transcription of the audio, and **not** a description of the reference image. It is a fusion of both:

- The reference image provides the **base** garment identity, style, silhouette, pattern, etc.
- The spoken query provides the **delta** — what the user wants to change relative to the reference.

**Example.**

- Reference image: a white long-sleeve lace dress
- Audio query: *"make it black and a little shorter"*
- Target-oriented caption (model output): *"a black version of the dress with shorter length and lace sleeves"*

The supervision target for this caption can come from the dataset's target-image caption, or from a normalized catalog description derived from the target item.

> **关于 caption 应该长什么样、反例和正例的完整讨论见 Appendix G：Caption 应该长什么样？反例和正例。**

##### 2.3 How retrieval is performed

**Step A: offline catalog indexing.** For each candidate fashion item in the catalog:

- pair the item image with a caption (either from the dataset's existing product descriptions, or generated and normalized from metadata)
- encode the caption into a vector using a **frozen** text embedding model (e.g., Sentence-BERT, CLIP text encoder, or an off-the-shelf text embedding API)
- store `(item_image, item_caption, item_caption_embedding)` in a vector index

**Step B: online inference.** For each user query:

1. The audio-VLM takes the reference image + audio query and generates a target-oriented caption.
2. The same frozen text embedding model encodes the generated caption into a **query caption embedding**.
3. We run nearest-neighbor search in the vector index using cosine similarity, and return the top-k items.

##### 2.4 Full pipeline

```
reference image + audio query
  → audio-VLM generates target-oriented caption
  → frozen text embedding model encodes caption
  → nearest-neighbor search in catalog caption-embedding index
  → retrieved target fashion item(s)
```

This caption-recomposition style pipeline is consistent with prior CIR (composed image retrieval) literature — most directly with CIReVL, which captions the reference image, recomposes the caption with the modification text, and retrieves with a CLIP-style text retriever.

> **更多关于 CIReVL / FACap / FashionBLIP-2 在 fashion CIR 文献里的定位见 Appendix I：文献定位。**

##### 2.5 Optional reranking stage

To improve precision, we can add a reranking module on top of the top-k retrieved candidates. For each candidate image, we feed `(reference image, spoken query, candidate image)` back into the VLM and ask it to score whether the candidate satisfies the requested modification. This second-pass validation is much cheaper than full end-to-end retrieval training, but it gives the system a way to recover from noisy first-stage text retrieval.

##### 2.6 Why this option is the safer first milestone

A natural question is: *given that Option 3 is more aligned with the retrieval objective (see 3.x below), why start with Option 2 at all?* The short answer is that Option 2 is **more stable and easier to get to a working demo**, for three reasons:

- **Aligned with the model's natural capability.** The audio-VLM's most natural mode is image + audio in, text out. Caption generation goes with that grain.
- **Easier to debug.** When retrieval fails, the failure surface is layer-localizable (caption quality / embedding model / wording / catalog caption). Option 3, by contrast, has a much wider and less inspectable failure surface.
- **Faster path to a working demo.** A proposal needs a credible, demonstrable first version more than it needs immediate SOTA.

> **完整论述见 Appendix C：方法 A 为什么是更稳的第一步？**

##### 2.7 Limitations

- **Extra intermediate step.** Inference goes through caption generation + caption encoding before retrieval, which adds latency and introduces an extra failure mode.
- **Misaligned training objective.** The model is trained to generate good captions, not to retrieve the correct item. A caption that is linguistically reasonable is not necessarily the most retrieval-friendly representation.
- **Language bottleneck.** Some fine-grained visual differences (subtle silhouette changes, fabric cues, fit nuance) are hard to fully express in language. Going through a textual intermediate may lose retrieval-relevant information.

These three limitations are exactly what motivates Option 3 below.

---

#### Option 3: Direct contrastive embedding retrieval *(research extension)*

This is the **research extension** of the proposal. It addresses the limitations of Option 2 by removing the language intermediate step and training the model directly for retrieval.

##### 3.1 What the model is trained to produce

For Option 3, applying the same three-layer view:

- The end goal is still to **retrieve the target fashion item**.
- The model's training target changes: instead of generating a caption, the model is trained to **directly produce a query embedding** for the multimodal query, in a shared space with target image embeddings.
- The retrieval step becomes nearest-neighbor search in this learned shared embedding space.

The system has two branches:

**Query branch.**

- Input: reference image $I_r$ + spoken modification $a$
- Output: a dense **query embedding** $q$ representing "the target garment the user is asking for"

**Target branch.**

- Input: target image $I_t$
- Output: a dense **target embedding** $t$ representing the candidate garment

Both branches map into the same embedding space.

##### 3.2 Training objective

We train the two branches jointly with a **contrastive loss** (e.g., InfoNCE with in-batch negatives), so that:

- matching `(query, target)` pairs have **high** cosine similarity
- non-matching pairs have **low** cosine similarity

This is the CIR-style training paradigm used by recent fashion CIR models such as FashionBLIP-2 in the FACap paper.

##### 3.3 How the embeddings are extracted

In the meeting, my mentor sketched something like "grab the last token / logits" as the retrieval representation. In the actual implementation, I would clean this up: instead of using raw vocabulary logits, I would use a **pooled hidden representation** or a small **learned projection head** on top of the model's final hidden states.

Concretely:

- For the query branch: pass `(reference image + audio)` through the audio-VLM, take the final hidden states, pool them (or run them through a small projection head) to obtain $q$.
- For the target branch: pass the target image through the vision encoder of the VLM (or through the full VLM with a "describe this image" style prompt) and pool / project the final hidden states to obtain $t$.

This preserves the spirit of the mentor's idea while making the retrieval formulation technically cleaner.

##### 3.4 How retrieval is performed

**Step A: offline catalog indexing.** Encode every candidate target image with the target branch and store the resulting embeddings in a vector index.

**Step B: online inference.**

1. The audio-VLM (query branch) takes the reference image + audio query and outputs a query embedding.
2. We run nearest-neighbor search against the target-image embedding index using cosine similarity.
3. Return the top-k items.

##### 3.5 Full pipeline

```
reference image + audio query
  → query branch (audio-VLM + pooling/projection)
  → query embedding

target image database
  → target branch (vision encoder + pooling/projection)
  → target image embeddings

→ nearest-neighbor search in shared embedding space
→ retrieved target fashion item(s)
```

##### 3.6 What "more end-to-end" actually means here

A common misconception is that "end-to-end" means the entire product is one model that does everything. That is **not** what is meant here.

In this context, "more end-to-end" means: **the model directly learns the mapping from a multimodal query to retrieval space**, rather than going through a textual intermediate. The training objective and the deployment objective are the same — similarity in the retrieval space — so the optimization is directly aligned with the task.

> **完整论述见 Appendix E：方法 B 里的 "end-to-end" 到底是什么意思？**

##### 3.7 Why this option is worth pursuing (and not just a technical flex)

Option 3 looks more elaborate than Option 2, so a fair question is: *is this just a technical flex, or is it actually likely to be better?* The honest answer is that it is likely to be better, for three concrete reasons:

- **Training objective directly aligned with the task.** Option 2 optimizes "generate a good caption" and *hopes* that good captions translate into good retrieval. Two captions can be linguistically equally reasonable and yet differ in retrieval quality. Option 3 optimizes retrieval directly.
- **Lower potential latency.** Option 2 inference is three steps (generate caption → encode caption → retrieve). Option 3 inference is two steps (produce query embedding → retrieve). The pipeline is structurally shorter.
- **Avoids the language bottleneck.** Some visual differences are hard to express in language (e.g. "保留这个 pattern 但换成 darker tone"). Going through a textual intermediate can lose retrieval-relevant signal that an embedding could otherwise preserve.

> **完整论述、所有具体例子（包括 "a more formal black version" vs. "an elegant black dress" 的对比、所有 language bottleneck 的具体例子）见 Appendix B：方法 B 为什么不是炫技，而是值得做的方法？**

##### 3.8 Limitations

- **Requires training-loop changes.** The current model is set up for next-token prediction loss. Implementing contrastive retrieval requires going into the Hugging Face training code and replacing the default loss with a contrastive objective on pooled representations.
- **Sensitive to data and batch design.** Contrastive learning depends on good positives, hard negatives, and large enough batch sizes. Noisy triplets and small batches can easily destabilize training.
- **Harder to debug.** When retrieval fails, the failure mode is much less inspectable than in Option 2 — possible causes include fusion issues, audio alignment, contrastive loss design, embedding collapse, weak negatives, or pooling problems.

For these reasons, Option 3 is positioned as a **second-stage extension**, to be attempted after Option 2 produces a working baseline.

---

### Part 4.3: Recommended plan

Putting the three options together, the recommended plan is:

- **Option 1** is preserved in the appendix as background, but is **not** the proposed method.
- **Option 2 (caption-generation retrieval)** is the **main method**. It will be the first milestone and is what the demo will be built around.
- **Option 3 (direct contrastive embedding retrieval)** is the **research extension**. It will be attempted once Option 2 produces a working pipeline, and is where the more research-heavy contribution would come from.

The two methods can be summarized in a single line each — see **Appendix H：一句话区分方法 A 和方法 B**.

This staged plan matches my mentor's overall intent from the meeting: build a real audio-conditioned fashion retrieval assistant rather than a "grounding + API call" wrapper, start from a stable text-based retrieval baseline, and extend toward direct contrastive retrieval if time and data permit.

---

## Part 5: Data Synthesis Design

*(to be filled in)*

---

# Appendices

The appendices preserve the full thinking process and detailed rationale for every key decision in Part 4. Each appendix is named after the question it answers, so readers can jump directly to whichever question they want a complete answer to.

| Appendix | Question |
|---|---|
| A | Option 1 完整 task variants（被排除路线的细节） |
| B | 方法 B 为什么不是炫技，而是值得做的方法？ |
| C | 方法 A 为什么是更稳的第一步？ |
| D | 方法 A 中，模型输出的为什么是 caption 而不是 embedding？ |
| E | 方法 B 里的 "end-to-end" 到底是什么意思？ |
| F | 三层分离 — system goal / model training target / retrieval step |
| G | Caption 应该长什么样？反例和正例 |
| H | 一句话区分方法 A 和方法 B |
| I | 文献定位 — CIReVL / FACap / FashionBLIP-2 |
| J | 看 dataset 时应该关注什么？ |

---

## Appendix A: Detailed Task Variants for Option 1 (Ruled Out)

This appendix preserves the full design space we considered for Option 1 — spoken garment grounding with downstream tool / API calls. Although this route is ruled out as the main method, the detailed task variants are kept here for completeness, since they document the design space we explored before settling on the retrieval-based formulation.

The common input setting for all four variants is:

- an image or video frame containing one or more garments
- a spoken user query referring to one garment or expressing a shopping-related request about it

Example spoken queries:

- "Is there a cheaper version of this?"
- "Do they have this in black?"
- "Can you find something similar to the jacket on the left?"
- "Not this one — the white dress behind it."

What changes between the variants is the model's output format.

---

### Version A: Spoken garment grounding

**Task.** The model identifies which garment the user is referring to in the image, based on the spoken query.

**Input.**

- image or video frame
- spoken query

**Output.**

- target bounding box

**Example.**

- Input: an image with several garments + audio "Can you find a cheaper version of this blazer?"
- Output: a bounding box around the referred blazer

**What this version means.** The model's responsibility is limited to resolving the spoken reference visually. It does not need to infer the full shopping action or generate any search query. Its role is to determine **what item the user means**.

**Why this version is attractive.**

- It is the cleanest and most realistic first milestone.
- It directly tests the value of adding audio modality to a VLM.
- It is easier to define, supervise, and evaluate than richer output formats.
- It keeps the model focused on the most clearly multimodal part of the problem: linking spoken reference to visual grounding.

**Limitation.** This version does not tell the rest of the system what the user wants to do with the garment. Downstream modules would still need to infer whether the user wants a cheaper option, another color, a similar item, and so on.

---

### Version B: Spoken garment grounding + coarse intent classification

**Task.** The model identifies the referred garment and predicts a coarse shopping intent.

**Input.**

- image or video frame
- spoken query

**Output.**

- target bounding box
- intent label

**Example intent labels.**

- `find_cheaper`
- `find_other_color`
- `find_similar`
- `find_more_formal`
- `find_more_casual`

**Example output.**

```json
{
  "target_box": [x1, y1, x2, y2],
  "intent": "find_cheaper"
}
```

**What this version means.** This formulation extends grounding with a lightweight semantic interpretation of the spoken request. The model must answer both:

- which garment the user is talking about
- what kind of shopping action the user wants

**Why this version is attractive.**

- It is still relatively simple and structured.
- It makes the model more tightly aligned with the shopping workflow.
- It reduces the burden on downstream modules.
- It gives the system a more explicit bridge between spoken interaction and retrieval behavior.

**Limitation.** The intent space is still coarse. It may not capture compound requests such as:

- "Find something similar but cheaper"
- "Show me this in black and in a shorter cut"

So while it is stronger than Version A, it still abstracts away a lot of nuance.

---

### Version C: Spoken garment grounding + normalized query generation

**Task.** The model identifies the referred garment and generates a normalized shopping query that can be passed downstream.

**Input.**

- image or video frame
- spoken query

**Output.**

- target bounding box
- normalized textual query

**Example output.**

```json
{
  "target_box": [x1, y1, x2, y2],
  "query": "Find cheaper black alternatives for this blazer."
}
```

**What this version means.** Instead of predicting a fixed intent label, the model rewrites the user's spoken request into a cleaner, more retrieval-friendly form. This can serve as an interface between the audio-VLM and a downstream search or recommendation engine.

**Why this version is attractive.**

- It is more flexible than a small intent label set.
- It may connect more naturally to text-based retrieval pipelines.
- It allows the model to capture more nuanced requests without requiring a complex structured schema.

**Limitation.** This version is harder to supervise and evaluate consistently. Because the output is free-form text, there may be many acceptable reformulations for the same spoken request. That makes the task less clean than box-only or box-plus-intent formulations.

---

### Version D: Spoken garment grounding + structured shopping constraints

**Task.** The model identifies the referred garment and outputs a structured representation of the shopping request.

**Input.**

- image or video frame
- spoken query

**Output.**

- target bounding box
- intent
- constraint slots

**Possible slots.**

- color
- price direction
- style direction
- fit preference
- length preference

**Example output.**

```json
{
  "target_box": [x1, y1, x2, y2],
  "intent": "search_similar_items",
  "constraints": {
    "price": "lower",
    "color": "black",
    "style": "more_formal"
  }
}
```

**What this version means.** This is the most structured formulation. It asks the model not only to ground the visual reference, but also to convert the spoken shopping request into a machine-friendly representation for downstream retrieval.

**Why this version is attractive.**

- It is the closest to a full shopping-agent interface.
- It supports cleaner downstream tool use.
- It makes the model output highly interpretable and easy to connect to APIs.

**Limitation.** This is also the most demanding formulation. It requires a carefully designed schema, more complex supervision, and higher confidence that the current model can stably learn multiple structured outputs at once.

---

### Comparison across Option 1 task variants

The four versions can be understood as a progression from minimal multimodal grounding to richer shopping-aware parsing.

- **Version A** focuses only on the most essential multimodal question: *what garment is the user referring to?*
- **Version B** adds a lightweight answer to the next question: *what broad shopping action do they want?*
- **Version C** keeps the grounding task but makes the semantic output more flexible by producing a normalized query.
- **Version D** pushes toward a more fully structured shopping-agent output.

All four formulations share the same end-to-end application surface. What changes is not the user-facing goal, but the specific responsibility assigned to the fine-tuned model — and, as discussed in Part 4.2 (Option 1, "Why we rule this option out"), all four ultimately delegate the actual retrieval to a downstream module, which is why the project has moved to Options 2 and 3 instead.

---

## Appendix B: 方法 B 为什么不是炫技，而是值得做的方法？

方法 B 看起来比方法 A 更复杂，所以一个合理的问题是：它真的不只是炫技吗？答案是：理论上是可能更好的，不只是炫技。而且它好，不只是"更酷"，而是有几个很实际的潜在优势。

### 原因 1：训练目标和最终任务更一致

方法 A 的训练目标是：

> 生成一个好的 caption

但真正的任务不是 captioning，真正的任务是：

> retrieve the correct target item

这两件事相关，但**不完全一样**。

有时候一个 caption 看起来挺合理，但它未必是最适合 retrieval 的表述。比如：

- "a more formal black version of the dress"
- "an elegant black dress with similar silhouette"

这两句话语义很接近，但 retrieval 时，哪句话更好，不一定。

所以方法 A 有点像绕路：

> 先优化"说得对不对"，
> 再希望"说得对"能帮助"找得准"。

而方法 B 是直接优化：

> 这个 query embedding 能不能把正确 target 拉近、错误 target 拉远。

所以方法 B 的训练目标和最终 retrieval 目标是**直接对齐**的。这就是它更 research、也更"end-to-end"的本质。

### 原因 2：延迟可能更低

**方法 A 推理时要做：**

1. VLM 生成 caption
2. 再调用 embedding model 把 caption 编码成向量
3. 再检索

**方法 B 推理时要做：**

1. 直接输出 query embedding
2. 直接检索

所以方法 B 少了一步 text generation / text encoding 的链路，理论上确实可能：

- latency 更低
- pipeline 更短
- engineering 更干净

当然，实际是不是更快，还取决于具体怎么实现。但从系统结构上讲，它确实更直接。

### 原因 3：少一个"language bottleneck"

方法 A 的问题是，它必须先把需求"翻译成文字"。但有些视觉差异其实很细：

- 更像这件的版型
- 袖子短一点但不要太紧
- 材质类似但更正式
- 保留这个 pattern 但换成 darker tone

这些东西，有时候语言能表达，但不一定表达得最完整。所以方法 A 有时候会受限于：

> **language bottleneck**

也就是：先转成 caption 的过程中，信息可能会损失。

方法 B 则有机会直接在 embedding space 里保留更多"难以语言化但对 retrieval 有用"的信息。这也是为什么 retrieval 研究里，direct embedding learning 往往是有意义的，不只是花活。

---

## Appendix C: 方法 A 为什么是更稳的第一步？

既然方法 B 在三个方面都有潜在优势，为什么 proposal 里还要把方法 A 作为主方法？因为方法 A **更稳**。这是 proposal 里非常重要的一点。

### 原因 1：更符合现有模型形态

我们现在的模型本质上还是一个 audio-extended VLM。它天然最擅长的是：

- 看图
- 听音频
- 输出文本

所以方法 A 是顺着它现在最自然的能力走的。

### 原因 2：更容易 debug

如果 retrieval 不好，方法 A 可以**逐层 debug**：

- caption 生成得对不对
- embedding model 好不好
- caption wording 有没有问题
- database caption 有没有问题

但方法 B 一旦 retrieval 不好，就很难一下子看出来：

- 是 fusion 没学好
- 是 audio 没对齐
- 是 contrastive loss 有问题
- 是 embedding collapse
- 是 negatives 不够难
- 是 pooling 不对

所以方法 B 更强，但也更难调。

### 原因 3：更容易先做出 demo

Proposal 最重要的不是一上来就 SOTA，而是先有一个**可信、能做出来、能演示**的版本。方法 A 很适合这个目的。

---

## Appendix D: 方法 A 中，模型输出的为什么是 caption 而不是 embedding？

在方法 A 里，模型直接输出的是 caption，caption embedding 是后处理（用一个 frozen 的 text embedding model 算出来的），不是模型直接 supervised 的输出。这一点和方法 B 完全相反，所以值得单独讲清楚。

### 为什么是 caption 而不是 caption embedding？

因为在这个 baseline 里，我们训练的是一个**生成式模型**，也就是 audio-extended VLM。它最自然的事情是：

- 输入图像和语音
- 生成文本

所以它最适合做的是：

> **image + audio → generated caption**

而不是直接训练成：

> **image + audio → embedding vector**

后者更像方法 B 的 research extension。所以在方法 A 里：

- **模型直接输出的是 caption**
- **caption embedding 是后处理出来的，不是模型直接 supervised output**

这个区分很重要，否则容易把方法 A 和方法 B 混在一起。

### Caption embedding 怎么得到？

不是 VLM 直接输出的，而是：

1. VLM 先输出一段 caption 文本
2. 再把这段 caption 丢给一个 **frozen 的 text embedding model**（比如 Sentence-BERT、CLIP text encoder、OpenAI embedding model）
3. 这个 text embedding model 把 caption 转成一个向量
4. 再拿这个向量去数据库里检索

所以 caption embedding 不是凭空来的，也不是手写 feature。它是通过一个**单独的 text embedding model**得到的。

### 这意味着方法 A 要训两个模型吗？

**不一定。** 更准确地说，方法 A 里通常有两个模块：

- **模块 1：audio-VLM** —— 这是要 fine-tune 的主模型，负责 `reference image + audio → target-oriented caption`
- **模块 2：text embedding model** —— 通常**不需要训练**，可以直接用 pretrained 的现成模型

所以方法 A 更常见的做法是：**只 fine-tune VLM，embedding model 直接用现成的**。只有当发现通用 text embedding 对 fashion retrieval 不够好时，才会考虑微调 embedding model，但 proposal 第一版完全没必要把这件事写成必须做。

---

## Appendix E: 方法 B 里的 "end-to-end" 到底是什么意思？

提到方法 B 的时候经常会用 "more end-to-end" 这个说法，但这里有一个常见的误解需要先澄清。

### 常见的误解

"end-to-end" 不是说**整个产品只有一个模型就全干完**。整个系统在两种方法里都还会有：

- offline 的 catalog indexing
- online 的 nearest-neighbor 检索
- 可能的 reranking 模块

这些都不是"一个模型搞定"的事。

### 真正的意思

这里的 "end-to-end" 更准确的意思是：

> **模型直接学会：从 multimodal query 到 retrieval space 的映射**

而不是：

- 先生成文本
- 再用另一个 text retriever 去检索

所以更 end-to-end 的点在于：

- **query representation 是直接学出来的**
- **retrieval objective 直接进入训练**
- **最终优化目标和检索任务本身更一致**

换句话说，方法 B 的 "end-to-end" 指的是 **训练目标和最终任务的对齐程度更高**，而不是 **系统模块的数量更少**。

---

## Appendix F: 三层分离 — system goal / model training target / retrieval step

读 Part 4 的 Option 2 和 Option 3 时，最容易混淆的地方是把"系统目标"、"模型训练目标"和"数据库检索"这三件事混在一起。这里把它们彻底分开。

### 第 1 层：最终产品目标（system-level end goal）

用户给：

- reference image
- audio query

系统最后要返回：

- top-k retrieved fashion items
- 最理想的那个就是 target fashion item

这层是**整个 project 的最终目标**。这一层在两种方法里**都不变**。

### 第 2 层：模型训练目标（model training target）

这里才是 method 的关键。**模型不一定直接输出最终商品**，它可以只负责生成一个**适合检索的中间表示**。这个中间表示有两种主流思路：

- **方法 A**：输出一个 **target-oriented caption**
- **方法 B**：输出一个 **retrieval embedding**

这两个都能服务同一个 end goal：最后检索出 target item。所以 Option 2 和 Option 3 真正不同的地方就在这一层。

### 第 3 层：数据库检索怎么发生（retrieval step）

不管上面输出 caption 还是 embedding，最后都还会有一个 retrieval step：

- 要么用 caption embed 后去检索（方法 A）
- 要么直接拿 learned query embedding 去检索（方法 B）

所以：

- **最终 retrieve 出 target item** 是系统目标（第 1 层）
- **caption / embedding** 是模型为了实现检索而产生的中间结果（第 2 层）
- **数据库检索** 是把中间结果用起来的最后一步（第 3 层）

把这三层分开看，Part 4 的两个 option 就不会混。

---

## Appendix G: Caption 应该长什么样？反例和正例

在方法 A 里，模型输出的是 caption。但这个 caption 应该是什么样的？这一点经常会被误解，所以单独写清楚。

### 反例：不是这两种

**它不是单纯转录 audio。**

- 输入 audio: "make it black and a little shorter"
- ❌ 错误的输出: "make it black and a little shorter"（只是把语音抄下来）

**它不是单纯描述 reference image。**

- 输入 reference image: 一件白色长袖蕾丝裙
- ❌ 错误的输出: "a white long-sleeve lace dress"（只描述了原图，忽略了用户的修改请求）

### 正例：fusion of both

它应该是一个 **target-oriented caption**，把两者融合起来，输出"目标衣服"的描述。

**例子：**

- Reference image: 一件白色长袖蕾丝裙
- Audio query: "make it black and a little shorter"
- ✅ 正确的输出 caption: "a black version of the dress with shorter length and lace sleeves"

这个 caption 的作用，是把：

- 原图里的 base garment identity / style（白色蕾丝裙的版型、蕾丝元素）
- 用户语音里的 modification delta（变黑色、变短一点）

组合到一起。

### 为什么这个区分重要

这正是 mentor 在 meeting 里讲的核心：**audio 说的是 delta，image 提供的是 base style，最后模型要把两者合成新的描述**。

如果 caption 只转录 audio，那就丢了 base 信息；如果 caption 只描述 reference image，那就丢了 modification 信息。两种情况下检索都会失败。所以 supervision 必须设计成 fusion-style caption，而不是简单的 transcription 或 image captioning。

---

## Appendix H: 一句话区分方法 A 和方法 B

为了快速理解 Part 4 里 Option 2 和 Option 3 的核心区别，可以用一句话压缩：

### 方法 A：主方法 / baseline

> **train the model to generate a target-oriented caption, then retrieve via caption embeddings**

更稳、更容易落地、更适合作为 proposal main method。它的模型输出是 **caption**，不是 embedding。embedding 是后处理。这个 caption 最终服务于 retrieval。

### 方法 B：extension / research version

> **train the model to directly produce a retrieval embedding for the multimodal query**

更 research、更接近 retrieval 本身、更 end-to-end。它的模型输出核心是 **embedding**，不是 caption。然后直接在 embedding 空间里找 target image。

### 两个流程图

**图 1：方法 A（baseline）**

```
ref image + audio
  → VLM generates target caption
  → embed caption (frozen text embedding model)
  → retrieve from database of caption embeddings
  → target fashion item
```

**图 2：方法 B（research extension）**

```
ref image + audio
  → query embedding

target image database
  → target image embeddings

→ nearest neighbor search in shared embedding space
→ target fashion item
```

---

## Appendix I: 文献定位 — CIReVL / FACap / FashionBLIP-2

这个 project 的两条方法路线都不是凭空想的，分别对应 composed image retrieval (CIR) 文献里两种主流做法。这里把对应关系说清楚。

### 方法 A 对应：CIReVL 类的 caption-recomposition 路线

CIReVL 的核心思路是：

- 给 reference image 配 caption
- 用语言把 modification 重新组合进 caption
- 再用一个 CLIP 风格的 text retriever 去检索

这正是方法 A 在做的事情，只不过 modification 从 text 变成了 spoken audio。所以可以把方法 A 看成 **CIReVL 的 audio-conditioned 版本**。

### 方法 B 对应：FACap / FashionBLIP-2 的 contrastive composed retrieval 路线

FACap 是 mentor 在 meeting 里发的那篇 paper，它做的是 fine-grained fashion composed image retrieval。FACap 配套的 FashionBLIP-2 模型走的是：

- 直接学一个 shared embedding space
- query 是 (reference image + modification text)
- target 是 target image
- 用 contrastive loss 训练

这正是方法 B 在做的事情，只不过 modification 从 text 变成了 spoken audio。所以可以把方法 B 看成 **FashionBLIP-2 的 audio-conditioned 版本**。

### 为什么两条路线都要写在 proposal 里

因为这正是 mentor 在 meeting 里的逻辑：先有一个稳的、CIReVL 风格的 caption-based baseline 能跑通，再往 FACap / FashionBLIP-2 风格的 contrastive retrieval 升级。两条路线都有公开文献做支撑，proposal 同时写出来，既能体现思路完整，又能给 mentor 一个明确的 staged plan。

### FACap 数据集的特殊价值

FACap 的卖点是它是为 **fine-grained fashion CIR** 构造的，强调更细粒度的 modification 描述（而不只是 "change color" 这种粗粒度）。这对方法 B 尤其重要，因为 contrastive retrieval 学到的 embedding 质量很大程度依赖于 modification 是不是足够细。

---

## Appendix J: 看 dataset 时应该关注什么？

接下来真正去看 FACap / FashionIQ 这些数据集的时候，应该盯住哪些字段？这里按方法 A 和方法 B 各列一份关注点清单，最后再加一份"打开数据先问的 6 个问题"。

### J.1 方法 A（caption-generation retrieval）的 dataset 关注点

如果走 caption-based baseline，去看 dataset 的时候最该盯的是这些字段：

**A. 有没有 triplet**

最想看到的是：

- reference image
- modification text
- target image

因为这定义了 composed retrieval 任务本身。FACap 就是这种格式。

**B. target image 有没有对应 caption**

因为 baseline 训练时最好有一个 target-oriented textual supervision。如果数据里没有现成 target caption，就得想办法：

- 用已有 product description
- 自己 caption target image
- 或者从 annotation 里构造 normalized target description

**C. modification text 的质量高不高**

因为之后要把它变成 speech。如果 modification text 太短、太粗糙，audio supervision 也会很弱。FACap 的卖点就是它是为 fine-grained fashion CIR 构造的，强调更细粒度的修改描述。

**D. candidate pool 是怎么定义的**

因为 retrieval evaluation 最后要在 candidate database 里找 target。需要弄清楚：

- target 是不是唯一正样本
- evaluation 是全库检索还是子集检索
- metric 是 Recall@K 还是别的

**E. 有没有类别和属性信息**

比如 color, sleeve, length, style 这些字段。这些会帮助：

- caption normalization
- error analysis
- demo filtering

### J.2 方法 B（direct contrastive embedding retrieval）的 dataset 关注点

如果以后做更 research 的方法 B，看 dataset 时就要重点看这些：

**A. triplet supervision 是否清楚**

必须要有：

- reference image
- modification
- target image

因为要明确知道谁和谁是正样本。

**B. candidate pool 是否够大**

contrastive retrieval 更依赖：

- 足够多 negative examples
- 合理的 retrieval setting
- batch 里能不能形成有效难负样本

这也是 mentor 提 large batch / positives / negatives 的原因。

**C. modification 是否足够细粒度**

因为如果 modification 太粗，比如只是 "change color"，那 embedding 学到的东西可能比较浅；而 fashion retrieval 更难的地方恰恰在于：

- sleeve
- length
- formality
- cut
- fabric-like cues
- style nuance

FACap 就是专门强调 fine-grained fashion CIR 的数据。

**D. 是否有高质量 target candidates**

因为 retrieval 学得好不好，很大程度取决于：

- target 跟 ref 的差异是不是恰到好处
- negatives 是否 visually confusing
- 同类款式是不是足够丰富

**E. 训练规模和 data cleanliness**

contrastive retrieval 对脏数据更敏感。如果 triplet 对应关系乱，模型就很容易学坏。

### J.3 打开数据先问的 6 个问题

不论走方法 A 还是方法 B，第一次打开数据的时候优先问这 6 个问题：

1. **它是不是 composed retrieval 格式？** 有没有 `reference image + modification + target image`？
2. **modification 是不是足够细？** 能不能支持 fashion 里的细粒度变化？
3. **target 有没有 caption / product description？** 如果没有，baseline 的 supervision 怎么构造？
4. **candidate pool 怎么组织？** 最后 retrieval 是在什么集合里检索？
5. **有没有 metadata 可用于 demo？** 比如 category、color、style、price、brand。
6. **text 转 speech 后会不会自然？** 如果 modification text 太书面，做成 spoken query 可能会很假。

这 6 个问题答清楚，proposal 的 Part 5（Data Synthesis Design）就有了起点。