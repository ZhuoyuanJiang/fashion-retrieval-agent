# Notes: understanding the M1–M3 baseline scaffolding

Personal study notes from the 2026-04-27 review session. Each entry is
a question I had while reading the code, plus the context and answer
so I can re-read this and recover the reasoning later.

---

## Q1: What does each of the 5 files in M3 do?

**Context:** Plan_2 Milestone 3 added five Python files to `src/baseline/`
on top of M2's two files (`text_encoder.py`, `build_caption_db.py`):

```
src/baseline/
├── vlm_caption.py        # M3
├── prepare_images.py     # M3
├── retrieve.py           # M3
├── eval.py               # M3
└── run_baseline.py       # M3
```

I want a one-line answer for each, plus the data-flow picture so I can
see how they fit together.

### How they fit together

The five files together implement the **eval pipeline** for Option 2
(caption-generation retrieval). One eval query flows through them
like this:

```
┌─ FacapDataset[i] gives you (ref_image_path, modification_text, true_target_id)
│
▼
[1] vlm_caption.py     ─→ generated caption ("a red knee-length dress with...")
                          ↑↑↑ this is the expensive step that needs a GPU
[2] (text_encoder.py from M2 encodes that caption to a 384-dim vector)
                          ↓
[3] retrieve.py        ─→ top-K matching target_ids from the caption DB
[4] eval.py            ─→ where did the TRUE target rank? recall@K? median rank?
[5] run_baseline.py     = the orchestrator that wires 1→2→3→4 in a loop
```

`prepare_images.py` is a separate utility, not part of the per-query
flow: it pre-fetches reference images from HuggingFace **before** the
eval loop starts so the GPU run isn't blocked on network calls
mid-eval.

### File-by-file

| File | One-line purpose |
|---|---|
| `vlm_caption.py` | Defines the captioner interface + 4 backends (Q2) |
| `prepare_images.py` | Pre-downloads reference images so the GPU run is offline-safe |
| `retrieve.py` | Given a query embedding, returns the top-K closest captions in the DB |
| `eval.py` | Computes Recall@1/5/10/50, median/mean rank; dumps qualitative samples |
| `run_baseline.py` | The CLI orchestrator: parses args, loads DB, runs the eval loop, prints metrics |

**What the "caption DB" actually is on disk.**

When we say "caption database" we don't mean SQL, Postgres, or any
real database server. The "DB" is just three plain files in
`runs/<run_name>/caption_db/`:

```
embeddings.npy   ← (N, 384) numpy array of float32 vectors
metadata.jsonl   ← N JSON lines, one per row, each carrying the
                   target_id / image_path / source for that row
config.json      ← provenance: encoder name, FACap commit SHA,
                   build args, seed
```

Retrieval is just `db_embeddings @ query_vector` (one matmul) plus
sorting — milliseconds for 1000 rows, fast enough for tens of
thousands.

Why not a "real" vector DB (FAISS, Pinecone, Weaviate, Chroma)?

- For ≤ ~60k vectors at 384-dim, plain numpy is fast enough and
  simpler. Adding a vector DB would be premature optimization.
- No external service to set up, no schema migrations, no network
  hops. Everything is reproducible from disk.
- If we ever scale to millions of vectors, we'd swap the numpy
  backend for FAISS in *one* place (`src/baseline/retrieve.py`).
  The rest of the code wouldn't notice.

So "caption DB" is honest as a name (it's a database in the loose
sense — a structured collection of records you can query), but
pedantic readers might prefer "caption embedding store" or
"caption index".

**The mental model behind those three files.**

If you start from the goal and work backwards, you arrive at exactly
these three files:

1. **Goal:** given a query embedding, find the most similar caption
   embeddings in our catalog.
2. **What we need to compute that:** a precomputed matrix of all
   the catalog's caption embeddings to compare against. → `embeddings.npy`.
3. **What the matrix lacks:** there's no way to map "row 47" back
   to "which item in the catalog is that?". The matrix is just
   numbers. → we need a parallel record telling us which target_id
   / image_path / source each row corresponds to. → `metadata.jsonl`.
4. **What both files don't capture:** how were the embeddings
   produced? With which encoder? On which dataset version? Without
   that, you might unwittingly compare embeddings from different
   encoders (different vector spaces, totally meaningless results).
   → we need provenance. → `config.json`.

So the three files come from three distinct needs:

- `embeddings.npy` answers *"what are the things to search?"*
- `metadata.jsonl` answers *"what does row N refer to?"*
- `config.json` answers *"are these embeddings still valid for this query?"*

Each file is the smallest, simplest representation of its concern.
None overlap. None could be dropped without losing real
information. This kind of "minimum sufficient representation" is a
good test for any data structure: take it apart and see if you can
describe what each piece does without referring to the others.

**More on the vector-DB alternatives, plus what "schema migrations" and "network hops" mean.**

The four alternatives mentioned above (FAISS / Pinecone / Weaviate
/ Chroma) each occupy a different point in the design space:

| Tool | What it is | Best for | Cost |
|---|---|---|---|
| **FAISS** | A C++/Python *library* (not a service) for nearest-neighbor search. Supports approximate methods (IVF, HNSW, PQ) for sub-linear search at scale | 1M–1B vectors on a single machine; offline retrieval | Free, open-source |
| **Pinecone** | A *managed cloud vector DB*. Serverless, scales automatically; you talk to it via HTTPS API | Production apps where you don't want to run your own DB | Pay-per-use; free tier limited |
| **Weaviate** | An *open-source self-hosted* vector DB with hybrid search (vector + keyword + filter); GraphQL API | Production apps that need filtering and hybrid search | Free if self-hosted; commercial cloud also available |
| **Chroma** | An *embedded* vector DB (SQLite under the hood); Pythonic API; designed for RAG / "AI app" workflows | Prototypes, local-first AI apps | Free, open-source |

What we have (numpy + JSONL + JSON files) compared to these:

- ✅ Zero dependencies, fully reproducible from disk, easy to
  inspect with `cat` / `np.load`.
- ✅ Works fine for our scale (≤ 60k vectors).
- ❌ O(N) per query (no approximate-NN); fine until ~1M vectors.
- ❌ No metadata filtering (e.g. *"find dresses under $50"*); we'd
  have to write that ourselves.
- ❌ No concurrent writes; we rebuild the whole DB from scratch each
  time `build_caption_db.py` runs.

**What "schema migrations" means.** A *schema* is the structure of
your data (column names, types, vector dimension, etc.). When you
change the structure (add a column, change the embedding dimension
from 384 to 768, switch from cosine to dot-product), you have to
*migrate* — move existing data from the old schema to the new. For
a real DB, this is annoying:

- Existing rows must be transformed (or re-embedded with the new
  encoder).
- App code referencing the old structure breaks.
- Rollback is complex if the migration fails halfway.

For our setup, "schema" doesn't really exist. If we change the
encoder, we delete `runs/.../caption_db/` and rerun
`build_caption_db.py`. The stale-DB gate at
`run_baseline.py:78–88` already does this check for us. No
migration logic — just rebuild from scratch.

**What "network hops" means.** A *hop* is one network round-trip
between your code and a remote service. Each hop adds latency
(typically 1–100ms depending on the network) and a small chance of
failure. Pinecone-style: every retrieval is a hop to the Pinecone
cloud. FAISS / Chroma local: zero hops. Our numpy: also zero hops
— just RAM accesses, microseconds. For a 1000-query smoke test,
the difference is *"instant"* vs *"several seconds of network
round-trips, plus retry logic if any fail"*.

Network hops also bring:

- **Cost** — paid services charge per request.
- **Reliability concerns** — networks fail sometimes; retries
  needed.
- **Privacy** — your data leaves your machine, possibly your
  jurisdiction.

So our local-only choice means: fast iteration, no cloud bill, no
privacy concerns, no extra reliability surface. Once we hit a scale
where these things matter (millions of items, multi-user production
app), we'd revisit.

**Lesson:** The expensive piece (the VLM in `vlm_caption.py`) is
quarantined behind a single interface so the rest of the pipeline
doesn't care which backend produced the caption. That's what makes
the same harness work locally with mocks AND on the server with a real
model — see Q2.

---

## Q2: What are the "4 backends" in `vlm_caption.py`? What is a "pluggable backend"?

**Context:** `vlm_caption.py` is the only M3 file that touches the VLM,
which is the GPU-hungry piece (~14 GB VRAM for Qwen2-VL-7B at bf16).
We don't always want or need the real one. The file defines an
abstract base class with a single method:

```python
class VLMCaptioner(ABC):
    @abstractmethod
    def caption(self, item: dict) -> str: ...
```

…and four concrete implementations of it.

### The four backends

| Backend | What `caption(item)` returns | When to use | Where it runs |
|---|---|---|---|
| `MockVLMCaptioner` | `"A fashion item that is {modification_text}"` (just templates the modification text) | Testing the *pipeline* on CPU without any model loaded | local |
| `OracleCaptioner` | `item["target_caption"]` — the ground-truth caption (cheating!) | Verifying retrieve/eval are bug-free (Q4) | local |
| `Qwen2VLCaptioner` | Vanilla `Qwen/Qwen2-VL-7B-Instruct` actually looks at `item["candidate_image_path"]` and generates a caption from `(image, modification_text)` | Reference number — what a generic VLM produces | **server** |
| `SpeechQwen2VLCaptioner` | Stage-1 base + Stage-2 LoRA from speechQwen2VL, in text-only mode (audio path ignored) | Headline baseline — preserves Stage 1→Stage 2 narrative | **server** |

All four expose the same method `caption(item) → str`. The orchestrator
picks one via the `--vlm {mock,oracle,qwen2vl,speechqwen2vl}` CLI
flag. **Everything downstream of the captioner doesn't know or care
which one ran.**

### Walk-through: what Mock and Oracle actually do

Suppose I'm processing one eval triplet:

```python
>>> item = ds[59042]                  # one eval triplet
>>> item["modification_text"]
"the dress is shorter and red instead of black"

>>> item["target_caption"]
"A red knee-length cocktail dress with a fitted bodice and flared skirt"

>>> item["candidate_image_path"]
"f200k_images/dresses/casual_and_day_dresses/.../candidate.jpeg"
```

**`MockVLMCaptioner`** — never opens the image, never loads any
model. Just templates the modification text into a caption-shaped
string:

```python
>>> mock = MockVLMCaptioner()
>>> mock.caption(item)
"A fashion item that is the dress is shorter and red instead of black"
```

Microseconds per call. Output is then passed to the encoder,
retrieve, eval — same as any other backend.

**Unpacking "templates the modification text into a caption-shaped string".**

Mock has a fixed format string with a placeholder; `caption()` does
nothing more than Python string substitution:

```python
class MockVLMCaptioner(VLMCaptioner):
    TEMPLATE = "A fashion item that is {modification}"
    def caption(self, item):
        return self.TEMPLATE.format(modification=item["modification_text"])
```

So `mock.caption(item)` reads `item["modification_text"]` and slots
it into the `{modification}` placeholder. That's all — no model,
no learning, no image. The result *looks* like a caption ("A
fashion item that is...") but is just the modification text wrapped
in fixed boilerplate. The point: downstream code (encoder →
retrieve → eval) processes any string the captioner returns, so
Mock lets us verify the downstream code runs end-to-end without
touching a GPU.

**`OracleCaptioner`** — also never opens the image, never loads any
model. Just looks up the ground-truth caption that the dataset
already has:

```python
>>> oracle = OracleCaptioner()
>>> oracle.caption(item)
"A red knee-length cocktail dress with a fitted bodice and flared skirt"
```

That's the *exact* string we encoded into the DB earlier (because
`build_caption_db.py` forced every eval-target's caption into the DB
as an `eval_target` row — Q6). So when Oracle's output is re-encoded
and used as a query, it should match exactly one DB row at cosine
similarity 1.0 → rank 1 every time.

**`Qwen2VLCaptioner`** (server-only) would actually open the
candidate image, send it plus the modification text to
Qwen2-VL-7B, and generate something like:

```python
>>> qwen.caption(item)
"A red, knee-length sleeveless dress with a flared silhouette"
```

Notice this is *similar* to the target caption but not identical.
That's the realistic case — and the whole baseline question is:
does "similar but not identical" still retrieve the right item?

### Why we have both Mock AND Oracle

It's reasonable to ask: if Oracle is stricter, why keep Mock?

| Backend | The question it answers |
|---|---|
| Mock | "Does the pipeline run end-to-end without crashing? Are there code errors in eval/encode/retrieve/metrics?" |
| Oracle | "Given a perfect caption, does retrieval find the right item? Is encoder → DB → retrieval → rank wired correctly?" (Q4) |

Two complementary pre-flight checks:

- **Mock is the cheapest test.** It exercises every line of the
  pipeline (parse args → load DB → generate caption → encode →
  retrieve → compute Recall@K → write artifacts) without needing
  any model. If Mock crashes, the bug is in `run_baseline.py`
  itself, not in any model. Mock numbers will be poor — that's
  expected.
- **Oracle is the strictest plumbing test.** Its perfect score
  rules out a specific class of subtle bugs (encoder
  non-determinism, off-by-one in rank computation, sort-direction
  errors, DB row alignment issues). Mock can pass while Oracle
  fails — they catch different things.

You could in principle ship with only Oracle. We keep Mock because
it's near-zero cost, gives a "baseline floor" to compare the real
VLM against (if the real VLM doesn't clearly beat Mock, something is
wrong with the model), and is faster to iterate on if you're
debugging the orchestrator itself.

### Why is it called "Oracle"? What are the alternatives?

The term comes from theoretical computer science: an **oracle
machine** is a hypothetical computer that can answer one specific
class of question instantly, with no work. In software testing,
the *oracle pattern* means "for testing purposes, give the system
the perfect answer for free, then check that the rest of the system
handles that perfect input correctly".

Alternatives / related patterns:

| Pattern | Idea | Common usage |
|---|---|---|
| **Oracle** (ours) | Replace expensive component with perfect-answer stub | Retrieval pipelines, search engines, ML inference |
| **Mock** (also ours) | Replace with fast/cheap stand-in (no perfect answer) | Almost everywhere |
| **Golden / snapshot test** | Compare current output against a saved "expected" output file | UI tests, parser tests, regression tests |
| **Differential test** | Run two implementations, compare outputs | Compilers, optimizers |
| **Property-based test** | Specify invariants the output must satisfy; generate random inputs | Algorithms, parsers |

The Oracle pattern is the natural fit here because the FACap dataset
*has* the ground-truth answer (`target_caption`) baked in — we can
just hand it to the captioner. If we didn't have ground truth (e.g.
testing a system where the right answer isn't known), we'd need a
different strategy.

This is a standard, well-regarded design choice. Mock + Oracle
pairs as smoke tests for ML pipelines is industry-common; the extra
~50 lines in `vlm_caption.py` are well worth the debugging value.

### What "pluggable backend" means

The pattern (sometimes called *strategy pattern* in textbooks) is:

1. Define an abstract interface (`VLMCaptioner.caption`).
2. Provide multiple concrete implementations, each satisfying the
   same interface contract.
3. Let the caller pick at runtime via a flag, config, or factory
   function.

In `vlm_caption.py:174–184` this lives in `make_captioner(name)`,
which maps the CLI string to the right class.

The value is concrete and cheap to demonstrate: same eval harness
runs locally with `--vlm mock` (no GPU) AND on the server with
`--vlm speechqwen2vl` (real model) with **zero code changes**. That's
the test of whether the abstraction is real — would swapping break
anything? No.

**Lesson:** When one component is much more expensive or constrained
than the others (GPU vs CPU, paid API vs local mock, network call vs
in-memory), put it behind an interface and provide cheap stand-ins.
You'll iterate 10x faster on the cheap path and pay the expensive
cost only at the end.

---

## Q3: Why are Mock and Oracle local-only, and Qwen2VL/SpeechQwen2VL server-only?

**Context:** `vlm_caption.py:13–35` has a hard guard
`_check_can_host_qwen2vl_7b()` that raises
`RuntimeError("server-only: ...")` on machines with <14 GB VRAM,
*before* any model-loading code runs. This means
`Qwen2VLCaptioner()` and `SpeechQwen2VLCaptioner()` cannot be
instantiated on my 8 GB laptop — the constructor itself fails.

Why is this even a thing?

### Per-backend resource needs

| Backend | Loads any model? | Touches the image? | Resource ceiling |
|---|---|---|---|
| Mock | No | No (ignores it entirely) | A few MB of RAM |
| Oracle | No | No (just dict lookup of `target_caption`) | A few MB of RAM |
| Qwen2VL | Yes — Qwen2-VL-7B at bf16 | Yes — opens via PIL | ~14–15 GB VRAM |
| SpeechQwen2VL | Yes — Qwen2-VL-7B base + LoRA adapter | Yes — same path | ~14–15 GB VRAM |

A 7B-parameter model at bfloat16 is 7 × 10⁹ × 2 bytes = 14 GB just
for the weights. Add KV cache, activations, image tokens, and you
need a 16 GB GPU minimum. My RTX 4070 laptop has 8 GB → cannot fit.

### Why fail in `__init__` (eagerly) instead of in `caption()` (lazily)?

If we let `Qwen2VLCaptioner()` succeed on a too-small GPU, the
failure would happen later — possibly mid-eval-loop, possibly
silently with a CUDA OOM that's harder to interpret. The eager
guard means:

- Clear error message naming the actual constraint (`14.0 GB needed,
  8.0 GB available`).
- Fails *before* you've already spent 30s downloading model weights.
- Gives a clear remediation hint ("Run on the GPU server").

**Where the guard lives in code.** The check is at
`src/baseline/vlm_caption.py:14–35`:

```python
MIN_VRAM_GB_FOR_QWEN2VL_7B = 14.0

def _check_can_host_qwen2vl_7b(model_label: str) -> None:
    """Raise RuntimeError if this machine can't hold Qwen2-VL-7B at bf16."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"server-only: {model_label} needs CUDA; this machine has no GPU"
        )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb < MIN_VRAM_GB_FOR_QWEN2VL_7B:
        raise RuntimeError(
            f"server-only: {model_label} needs >= {MIN_VRAM_GB_FOR_QWEN2VL_7B} GB VRAM "
            f"at bf16; this machine has {total_gb:.1f} GB. "
            f"Run on the GPU server."
        )
```

It's invoked as the **first line** of `_Qwen2VLLikeCaptioner.__init__`
(`vlm_caption.py:103–107`), before any model-loading code:

```python
class _Qwen2VLLikeCaptioner(VLMCaptioner):
    def __init__(self, image_cache_root=None) -> None:
        _check_can_host_qwen2vl_7b(self.MODEL_LABEL)   # ← eager guard
        self.image_cache_root = Path(image_cache_root) if image_cache_root else None
        self._load_model()                              # only reached if check passed
```

So `Qwen2VLCaptioner()` on this 8 GB laptop fails immediately with a
clean message. `_load_model()` (which would download ~15 GB of
weights and try to put them on the GPU) is never reached.

**What would happen *without* this guard.** The model-loading line
inside `_load_model()`:

```python
self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    self.BASE_REPO,            # "Qwen/Qwen2-VL-7B-Instruct" — 14 GB at bf16
    torch_dtype=torch.bfloat16,
    device_map="cuda",          # tries to put it on the GPU
)
```

would attempt to allocate ~14 GB on an 8 GB GPU. After spending time
downloading weights, you'd eventually see something like:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 75.81 MiB is free.
```

That error has three problems:

- **Misleading.** It says "tried to allocate 56 MiB", but the real
  problem is the 14 GB total — 56 MiB was just the straw that broke
  the camel's back after partial loading already filled the GPU.
- **Late.** It surfaces after you've already paid ~30s+ to download
  the model weights from HuggingFace.
- **Unhelpful for remediation.** Gives no hint that "Run on the GPU
  server" is the answer.

The eager guard turns all three into the bullets above — clear,
fast, actionable.

**Lesson:** When a class has hard hardware preconditions (VRAM,
disk, network), check them in `__init__` so failures land at the
earliest possible point, with the actionable error message close to
the call site that triggered them.

---

## Q4: Why does Oracle's perfect score prove that retrieve/eval is "bug-free"?

**Context:** Plan_2's exit criterion for M3 is `R@1 ≥ 0.95` under the
oracle backend. The first time I ran it I got `R@1 = 1.0000` (50/50).
The codex reviewer pushed back on calling this "bug-free" — let me
write down exactly what it does and doesn't prove.

### What `OracleCaptioner` actually does

```python
class OracleCaptioner(VLMCaptioner):
    def caption(self, item: dict) -> str:
        return item["target_caption"]
```

Just one line: return the dataset's ground-truth caption verbatim.
No model, no image, no template.

### Why R@1 should be 100% if everything is wired correctly

The smoke DB is built so that every eval-target's caption is
already a row in the DB (`build_caption_db.py:88–106`, the 50+950
trick — see Q5). So:

```
encode(item["target_caption"])  →  vector A
                                       │
                                       ▼ (cosine sim against all 1000 DB vectors)
                                       │
DB has a row whose embedding came from encode(item["target_caption"])
                                       │
                                       ▼
                          That row's vector is A (identical text → identical vector)
                                       │
                                       ▼
                          cosine(A, A) = 1.0, beats all others
                                       │
                                       ▼
                          rank of true target = 1 → R@1 contributes 1
```

Repeat for all 50 queries → R@1 = 50/50 = 1.0.

### What this proves

If R@1 < 100%, then *one of* the following is broken:

- Encoder is non-deterministic (same input → different outputs).
- Index has duplicates that confuse the rank.
- Retrieval picks the wrong row given the right scores.
- Eval mis-counts rank (off-by-one, sort direction, etc.).
- DB construction silently dropped/duplicated the eval-target row.

A perfect score rules out all of those *for the identity-text path*.
That's what we mean by "plumbing is correct" — encode → retrieve →
score → rank works the way the eval code assumes.

**What "index" means, and what "duplicates" would do.**

"Index" in retrieval is just a name for the **searchable data
structure** — the thing you query. In our setup, the index *is*
the caption DB at `runs/<run_name>/caption_db/`:

```
embeddings.npy   ← (1000, 384) matrix; row i is the i-th caption's vector
metadata.jsonl   ← row i tells you which target_id row i corresponds to
```

So "index has duplicates" means **two different rows of the index
that point to the same target_id (or carry the exact same caption
text)**. Concrete example:

```
Healthy DB (no duplicates):           DB with a duplicate bug:
row 0  target=91400899_0  score=0.87  row 0  target=91400899_0  score=0.87  ← true target
row 1  target=91312345_0  score=0.85  row 1  target=91400899_0  score=0.87  ← duplicate of row 0!
row 2  target=89001234_0  score=0.81  row 2  target=91312345_0  score=0.85
                                       row 3  target=89001234_0  score=0.81
```

Why this would confuse the rank:

- `rank_of()` walks the sorted list and returns the position of the
  first row whose target matches. If duplicates exist, the
  *second* occurrence's "rank" is wrong (it's later than the
  same target's first appearance).
- For Recall@K, K positions in the top-K may be filled by the
  same target, leaving fewer distinct candidates than expected. A
  borderline R@K=1 can flip to R@K=0 in edge cases.
- If de-duplication logic runs at retrieve time, ranks shift and
  the metrics no longer match what the eval code computed.

In our smoke DB, no duplicates exist by construction:
`build_caption_db.py:88–94` uses a `seen` set to dedupe eval-target
paths before sampling, and `random.sample()` (line 106) draws
distractors **without replacement**. Oracle's perfect score is
therefore also implicit confirmation that DB de-duplication worked
at build time.

### What this does NOT prove

- The encoder produces *good* embeddings for non-identical text.
- Retrieval handles real-world VLM-generated captions (which differ
  from the ground truth in subtle ways).
- The full pipeline does what we want when the captions are
  paraphrased, partial, or wrong.

The codex reviewer caught this overclaim in round 1 and I rewrote
`Progress_2`'s wording to match: "**identity-path** sanity check",
not "bug-free". Treat oracle as a *floor* for plumbing correctness,
not a ceiling.

**Lesson:** A test that always passes when its narrow precondition
is met (here: identical text in query and DB) only verifies the
narrow path. Honest progress reports name what was actually
exercised, not the broader thing the test gestures at.

---

## Q5: What does "where did the true target rank?" mean? Walk me through one query.

**Context:** Recall@K and median rank are the headline metrics, but
they aggregate per-query *ranks*. To trust the metrics I need to
trace one query end-to-end and see exactly what "rank" means.

**Term-by-term glossary** (since the rest of this answer uses these
freely):

- **Rank** — the position of the true target in the sorted list of
  all DB candidates. Lower is better. If the true target's caption
  is the closest match in the DB, rank = 1.
- **K** — a cutoff number. We compute Recall@K for K ∈ {1, 5, 10,
  50}. Different K's tell different stories: R@1 = "did we get the
  right answer first try?"; R@50 = "is the right answer at least
  somewhere in the top 50?".
- **Recall@K** — the fraction of queries whose true target appeared
  in the top-K. Always between 0 and 1.
- **Median rank** — the middle value of the rank distribution. If
  you sort all 50 per-query ranks, the median is the 25th-26th
  value. Robust to outliers — one really bad query doesn't move it
  much.
- **Mean rank** — the arithmetic mean of all 50 ranks. Sensitive to
  outliers — a few "rank 800" queries pull the mean way up while
  the median barely moves.
- **Aggregate** — combine many per-query numbers into one summary
  number. So "aggregate over 50 queries" = compute one R@K (or one
  median rank) by combining the 50 individual ranks. The
  aggregation function for R@K is "fraction with rank ≤ K"; for
  median rank it's `numpy.median`; for mean rank it's `numpy.mean`.
- **Headline metrics** — the small set of summary numbers you'd
  put in a paper or progress report. For us: R@1, R@5, R@10, R@50,
  median rank, mean rank — six numbers. Anything else (per-query
  details, score distributions) is supporting evidence, not
  headline.

Why both median and mean? They tell different stories:

- Median ≈ "what does a typical query look like?"
- Mean ≈ "are there catastrophic failures pulling things down?"
- Big gap (e.g., median 3 / mean 80) → most queries work fine but
  a few are catastrophically bad. Worth investigating.

### Setup

- Eval triplet `i = 59042` (one of the last 50 in `dress_train_triplets.json`)
- `item["candidate_image_path"]` = candidate dress photo
- `item["modification_text"]` = *"the dress is shorter and red instead of black"*
- `item["target_id"]` = `91400899_0` ← TRUE target

### Inside the loop, step by step

**Step 1 — VLM generates a caption.** Say we're using Mock for this
example (so it's reproducible without a GPU):

```
Mock returns: "A fashion item that is the dress is shorter and red instead of black"
```

**Step 2 — Text encoder embeds it** (`text_encoder.py:30`):

```
encoder.encode([generated]) → numpy array of shape (1, 384), L2-normalized
                                                    └─ call this q
```

**Step 3 — Retrieve top-K** (`retrieve.py:39`). Compute cosine
similarity of `q` against every one of the 1000 caption embeddings
in the DB, sort descending. Top 10 might look like:

| rank | target_id      | similarity | what it actually is               |
|------|----------------|------------|-----------------------------------|
| 1    | 89001234_0     | 0.81       | red dress, knee-length (close)   |
| 2    | 90123456_2     | 0.79       | red dress, V-neck                 |
| 3    | **91400899_0** | **0.77**   | **← TRUE TARGET**                 |
| 4    | 87654321_1     | 0.74       | wrong                             |
| …    | …              | …          | …                                 |
| 1000 | 12345678_0     | 0.05       | totally unrelated                 |

**Step 4 — Score the rank.** `rank_of(item["target_id"], q, db)`
in `retrieve.py:55–67` walks the sorted list and returns the 1-based
position of the true target's row.

```
rank_of("91400899_0", q, db) → 3
```

For this single query:

```
Recall@1  = 0  (true target wasn't first)
Recall@5  = 1  (true target appeared in top 5)
Recall@10 = 1
Recall@50 = 1
rank      = 3
```

**Step 5 — Aggregate over all 50 queries** (`eval.py:36–62`):

```
R@K          = (number of queries with rank ≤ K) / 50
median rank  = numpy.median([rank for each query])
mean rank    = numpy.mean([rank for each query])
```

That's all the metrics are. Lower ranks (closer to 1) are better;
high ranks indicate retrieval struggled to find the true item.

### Why oracle hits R@1 = 1.0 in this picture

Under oracle, the generated caption IS the true target's caption
verbatim, so `q` matches exactly one DB row (cosine 1.0, beats all
others), placing the true target at rank 1 every time.

### How fast is retrieval at scale?

You might have noticed: if the DB has N items, then **each query
computes N similarity scores** (one per DB row). Yes — 100,000 items
means 100,000 dot products per query. With M queries, total work is
O(N × M).

Why this is fast in practice: the N dot products aren't a Python
for-loop. They're one **matrix multiplication**:

```python
scores = db_embeddings @ query_vector
#        (N, 384)        (384,)         → (N,)
```

NumPy executes this in optimized C code (BLAS); CUDA does it on GPU
even faster. Rough numbers:

| N (DB size) | Time per query (CPU) | Time per query (GPU) |
|---|---|---|
| 1,000 (smoke DB) | ~50 μs | ~5 μs |
| 60,000 (full FACap dress) | ~3 ms | ~50 μs |
| 1,000,000 | ~50 ms | ~500 μs |
| 100,000,000 | ~5 s | ~50 ms |

So for our 1000-row smoke DB and even the full ~59k server DB,
brute-force scan is comfortable — **linear in N, but the constant
is tiny** (a few hundred nanoseconds per row on CPU).

When does brute force stop being good enough? Around 1M+ vectors,
where each query starts taking hundreds of ms. At that scale,
**approximate nearest-neighbor (ANN) indexes** like FAISS, ScaNN,
or HNSW become valuable. They precompute a tree or graph that lets
you find approximately the right answer in O(log N) per query,
trading exactness for speed.

We don't need ANN for this project — the FACap dress catalog is
~59k. We'd revisit if retrieval ever ran over the full Fashion200k
or DeepFashion-MultiModal catalogs (a few hundred thousand items).

**Lesson:** "Recall@K" is just bookkeeping over per-query ranks.
If you can describe one query's rank by hand, you can predict what
the metrics will say.

---

## Q6: What does "the answer wasn't in the index" mean?

**Context:** I keep seeing the comment "guarantee every eval query's
true target is in the index" and the 50+950 DB construction
(`build_caption_db.py:88–106`). Why is this needed? What goes wrong
without it?

**Why use a "trick" at all — what about the obvious alternatives?**

The natural instinct is *"just put 1000 captions in the DB and
test retrieval — why complicate it?"* The reason: there are several
obvious ways to build a 1000-row smoke DB, and most of them are
broken in different ways. Going through them:

| Alternative DB | What goes wrong |
|---|---|
| Use ALL ~59,082 captions, no subsetting | Encoding 59k captions on CPU takes ~15 min. Too slow for a smoke loop where we want sub-minute iteration. (Fine on the server with GPU — that's exactly what Plan_3 will do.) |
| 1000 random captions, no special inclusion | The 50 eval-targets are unlikely to be among 1000 random picks (each has ~1.7% chance → on average <1 of the 50 lands in the DB). For most queries the true target isn't in the DB. R@K ≈ 0 because the answer was never a candidate, not because retrieval is broken. The metric is meaningless. |
| Only the 50 eval-target captions, no extras | Each query has 1 right answer and 49 unrelated dresses. Retrieval becomes trivial — R@1 ≈ 1.0 even with garbage queries, because the right caption is dramatically more similar to itself than to any other dress. Doesn't actually test ranking ability. |
| **50 eval-targets + 950 randomly-sampled other captions** (our choice) | ✅ Coverage 100% by construction — we **force-include** the 50 eval-targets by name. ✅ The other 950 give realistic ranking difficulty. ✅ Builds in ~15s. |

So "the trick" is the **only design that's both fast (small DB) and
meaningful (eval coverage + realistic difficulty)**. It isn't a
hack papering over a bug — it's the right way to build a small
smoke DB.

The word "trick" here just refers to the **deliberate asymmetry**:
50 captions are *force-included* by name, 950 are sampled randomly
from the rest. It is **NOT** synthetic / fake / random text we
generate to confuse the model — that misreading was the original
source of confusion. Both groups are real FACap captions; "trick"
is just the way they were chosen, not what they are.

**Where the 950 actually come from.**

Spelling it out concretely, since "950 random ___" can sound like
we're inventing fake text. The 950 are real FACap dress captions,
sampled at random from the part of the catalog that isn't an
eval-target:

```python
# src/baseline/build_caption_db.py:97–106
captions = ds.captions                          # all ~59,082 real FACap dress captions
distractor_pool = [p for p in captions          # variable name uses "distractor",
                     if p not in seen]          # the standard retrieval-eval term for
                                                # "real catalog item, not the right answer".
                                                # 59,082 − 50 = 59,032 candidates here.
distractor_paths = rng.sample(distractor_pool,
                              950)              # pick 950 real captions from the pool
```

So the 1000-row smoke DB is:

- 50 captions of real FACap dresses that happen to be the answers
  to our 50 eval queries (the "eval-targets").
- 950 captions of OTHER real FACap dresses, sampled at random from
  the remaining ~59,032. Not the right answer for any of our 50
  specific queries — but still real items in the catalog.

**Every row in the smoke DB is a real FACap caption.** The
asymmetry is in *how they were chosen*, not in what they are.

### The failure mode it prevents

If the DB is built from 1000 *random* captions, there's no
guarantee that any particular eval query's true target is one of
those 1000. Suppose:

- Eval query has true target `91400899_0`.
- Our random sampling didn't pick `91400899_0` for the DB.
- Then `91400899_0` is **not in the index at all**.

Now no matter how good retrieval is, `91400899_0` cannot appear in
the top K — it's not a candidate. The query's rank is **undefined**
(or `None`, in our code: `retrieve.py:64`). Recall@K = 0 for all K.

If many queries are in this state, you'd see low Recall@K and
think *retrieval is broken*, when really the answer simply wasn't an
option.

### How the 50+950 trick fixes it

`build_caption_db.py` builds the smoke DB as:

```
50 captions  ← the eval-target captions, FORCED into the DB
950 captions ← random distractors (sampled from the rest)
─────
1000 captions, 100% eval coverage by construction
```

So every eval query's true target is guaranteed to be a candidate.
Any miss is a real retrieval bug, not "the answer wasn't in the
index".

### Why we don't need this trick on the server

Plan_3's server run will encode the **full** ~59,082 dress target
captions into the DB. Every triplet's target is automatically in
there because they all come from the same source. Coverage is 100%
without any tricks.

**Lesson:** When you debug a retrieval system and it looks broken,
first check coverage — is the answer even a possible option? "Wrong
result" and "right result was never an option" need different fixes.

---

## Q7: How should we export the conda environment for the project?

**Context:** I built a `fashion_retrieval` conda env with Python
3.10.18 + pip-installed torch / sentence-transformers / etc. To make
the project reproducible on the server (and Colab), I need to ship
env files. The reference project speechQwen2VL ships both an
`environment.yml` (conda layer) and a `requirements.txt` (pip
layer). What's the professional default?

### Three options

| Option | Tooling | Captures | Downside |
|---|---|---|---|
| **(i) Export-only** | `conda env export --no-builds > environment.yml` + `pip freeze > requirements.txt` | Everything actually installed, including transitive deps | environment.yml may be a bit long; `--no-builds` flag matters |
| **(ii) Hand-curated** | I write `requirements.txt` listing only deps I deliberately installed | Just what's intentional → readable | Forgetting a transitive dep is invisible until a fresh install fails |
| **(iii) Both** | Hand-curated `requirements.txt` + auto `requirements.lock` (from `pip freeze`) | Both intent AND exact reproduction | More files to keep in sync |

### Why `--no-builds` matters for `environment.yml`

`conda env export` by default writes platform-specific build
hashes for each package, e.g.:

```yaml
- numpy=2.1.2=py310h0c4d23_1   # ← =py310h0c4d23_1 is the build hash
```

Those hashes are tied to the OS/CPU/build chain. If I copy that file
to a Linux server with a slightly different setup, conda may fail to
solve the env. With `--no-builds`:

```yaml
- numpy=2.1.2                  # ← portable, conda can pick a build
```

### What speechQwen2VL does

speechQwen2VL ships a hand-curated `requirements.txt` (option ii) with
explicit version pins, plus a separate `environment.yml` for the conda
layer. So in their case (ii) — but they accept the maintenance burden
of keeping it in sync.

### Recommendation

For this project, **option (i) — just export both files** — is the
professional default and the safest. It captures the actual installed
state, including transitive deps that we might forget about, and
`--no-builds` keeps it portable.

```bash
conda activate fashion_retrieval
conda env export --no-builds > environment.yml
pip freeze > requirements.txt
```

These two files commit at repo root.

**Lesson:** "Hand-curated env files" sound clean but rot fast as the
project grows. Auto-export is the safer default for any non-trivial
project; lean on tooling, don't trust your memory.

---

## Q8: Why does `FacapDataset` return paths instead of loaded `PIL.Image` objects? What is "lazy I/O"? What does "pickle" mean?

**Context:** `src/data/facap_dataset.py:73–82` returns each item as a dict
where the image fields are **strings** (paths), not loaded PIL images:

```python
return {
    "candidate_image_path": cand_path,   # just a string
    "modification_text": ...,
    "target_image_path": ...,             # just a string
    "target_caption": ...,
    "target_id": ...,
    "candidate_id": ...,
}
```

A separate `load_image(item, side)` helper opens an image only when
the caller asks. Why this design rather than pre-loading the images?

### Reason 1 — Memory

59,082 dress images at ~50 KB each = ~3 GB just to keep them all in
RAM. Multiply by other categories (shirt, toptee, …) and the cost
grows fast. With paths, the dataset object is essentially just JSON
in memory — a few MB.

### Reason 2 — Lazy I/O

"**Lazy**" = do the work *only when the result is actually needed*,
not upfront. "**I/O**" = input/output, here specifically disk reads.
So **lazy I/O = don't read the image file until something asks for
the pixels.**

Concrete example — this loop never needs an image:

```python
for item in ds:
    print(item["modification_text"], item["target_caption"])
```

If `__getitem__` eagerly returned loaded `PIL.Image` objects, every
iteration would read a JPEG from disk — 59,082 disk reads, all
wasted, because we only wanted the text fields. With paths the loop
touches zero disk. The read only happens when someone calls
`ds.load_image(item, "candidate")`.

This matters concretely for our pipeline because **Mock and Oracle
never look at the image** (Q2). Eager loading would cost 59k disk
reads on every smoke test that uses those backends.

### Reason 3 — Pickling and `DataLoader`

**The constraint.** `DataLoader(ds, num_workers=4)` spawns 4 worker
processes. Each worker is a **separate operating-system process**
with its own private memory.

This isn't a networking concept — it's about OS-level process
isolation. A `dict` in Python lives at some memory address, say
`0x7f8a3c0b1234`. That address is meaningful **only inside the
process that created it**. If worker process A handed the raw address
to the main process, the main process's attempt to read it would fail
(or read random garbage from its own memory) — the OS forbids one
process from looking into another's memory. This is a hard security
guarantee; without it, any program could spy on any other.

So even though the worker and main process run on the same machine,
they can't share Python objects directly. They can only share **bytes
through a channel both can access**. That channel is called a
**pipe** — an OS-managed buffer in kernel memory that one process
writes to and the other reads from. (Local, not over the network.)
The pipe only carries bytes, not Python objects.

The full chain for one item:

```
Worker process A                              Main process
─────────────────                            ─────────────
{"path": "...", ...}                          (empty)
       │
       │  pickle.dumps(...)
       ▼
b'\x80\x04\x95...'  ──── pipe (bytes) ────▶  b'\x80\x04\x95...'
                                                     │
                                                     │  pickle.loads(...)
                                                     ▼
                                            {"path": "...", ...}
                                            (an equivalent dict, in main's memory)
```

**Why we specifically need pickling.** The pipe carries bytes; a
`dict` is a Python object. The "object → bytes" conversion is called
**serialization**. Pickle is Python's default serialization format —
it handles arbitrary Python objects (dicts, lists, numpy arrays,
custom classes) by turning each into a deterministic byte string.
Alternatives exist (JSON, msgpack, protobuf), but pickle is the
simplest and works for any Python object the user puts in their
Dataset, so PyTorch defaults to it.

The same idea applies to saving an object to disk or sending it
over a network — both also need "object → bytes" first. Pickle
works in all three cases.

**Pickle examples:**

```python
import pickle

# pickling: object → bytes
data = {"path": "image1.jpeg", "id": "91400899_0"}
serialized = pickle.dumps(data)        # b'\x80\x04\x95...' (bytes)

# unpickling: bytes → object
restored = pickle.loads(serialized)    # back to the original dict
```

**How fast/slow is pickling for different return types?**

| Return type | Pickle cost per item | Bytes shipped per batch of 16 |
|---|---|---|
| Path string `"images/91400899_0.jpeg"` | microseconds | ~hundreds of bytes |
| Loaded `PIL.Image` object (decoded) | milliseconds | ~10s of MB (raw pixels) |
| Open file handle / socket | **error** — not picklable | n/a |

So returning paths instead of `PIL.Image` objects gets us:

- **1000× smaller pipes** (a few hundred bytes vs tens of MB per batch)
- **No risk of file-handle / lazy-decode pickling issues** (some PIL objects can't be cleanly pickled if they hold open file handles)
- **Workers stay cheap** (less CPU spent serializing, less memory used per worker)

Combined with the no-disk-reads-when-not-needed point in Reason 2,
that's why PyTorch `Dataset` classes that work with files commonly
return paths or IDs, not loaded payloads.

### What a "typical" PyTorch `Dataset` returns

Three common patterns; the right choice depends on whether all
consumers of `__getitem__` need the bytes:

| Pattern | When to use | Example |
|---|---|---|
| **Tensors ready for the model** | Uniform consumer — every batch needs the image (e.g. training a vision classifier) | CIFAR-10's `(image_tensor, label)` |
| **Paths**, caller transforms/loads | Non-uniform consumer — some paths through the code don't need the bytes | Our `FacapDataset` |
| **Raw bytes / numpy arrays** | When opening is fast and you want to defer *decoding* | Some HF image datasets |

Decision rule: **do all consumers of `__getitem__` need the loaded
payload?**

- Uniform yes → return tensors (pay the load cost up front, get
  maximum efficiency in the training loop).
- Non-uniform → return paths/IDs (let each consumer decide).

Our pipeline has Mock + Oracle (no image), Qwen2VL + SpeechQwen2VL
(image needed), `prepare_images.py` (only the candidate side),
`dataset_inspection.ipynb` (occasional spot checks). Non-uniform →
paths is the right call.

**Lesson:** When designing a `Dataset`, ask "do all consumers need
the loaded payload?" If yes, load it. If no, return a cheap
reference (path or ID) and provide a helper to load on demand.

---

## Q8.5: Why return both `candidate_image_path` and `candidate_id`? Why not just use the path as the key?

**Context:** `FacapDataset.__getitem__` returns both raw FACap paths and
derived IDs:

```python
{
    "candidate_image_path": "f200k_images/dresses/12345678_0.jpeg",
    "target_image_path": "f200k_images/dresses/87654321_0.jpeg",
    "candidate_id": "12345678_0",
    "target_id": "87654321_0",
}
```

The IDs are derived from the paths. They are not extra labels and not extra
supervision.

### The design question from scratch

Start with what FACap gives us:

```text
candidate_image_path
target_image_path
modification_text
target_caption
```

If the only goal were to inspect one row, this would be enough. But the
retrieval pipeline also needs to answer operational questions:

- Which reference image produced this generated caption?
- Which target image is the ground-truth answer?
- Did the ranked retrieval list contain the true target?
- Which image does row `i` of the caption DB or image cache refer to?
- Did train and eval share the same image?

Those questions are easier to answer with a normalized image key.

### Path vs. ID

The raw path answers provenance:

```text
Where did FACap say this image came from?
```

The ID answers identity:

```text
Which image is this, independent of how this local copy is stored?
```

So:

```text
candidate_image_path = provenance / original FACap metadata
candidate_id         = operational image key
```

The path could be used as the key if the whole system committed to raw path
strings everywhere. I chose not to make path strings the central key because
paths mix two concepts:

```text
identity:       12345678_0
storage/layout: f200k_images/dresses/...jpeg
```

If the storage layout, directory prefix, or extension changes, the path string
can change while the image identity remains the same.

### Concrete qualitative example

Retrieval/evaluation naturally wants records like:

```json
{
  "query_id": "12345678_0",
  "true_target": "87654321_0",
  "top10_predicted": ["11111111_0", "87654321_0", "22222222_0"],
  "rank": 2
}
```

This says:

```text
reference image = 12345678_0
correct target  = 87654321_0
retrieved rank  = 2
```

The same record could be written with full paths, but it is longer and ties
the result artifact to one path convention:

```json
{
  "query_path": "f200k_images/dresses/12345678_0.jpeg",
  "true_target_path": "f200k_images/dresses/87654321_0.jpeg"
}
```

IDs make logs, cache metadata, retrieval outputs, and split filters speak the
same compact naming language.

### Why `target_id` is obvious

`target_id` is the answer key for retrieval:

```python
rank = rank_of(item["target_id"], q_emb, db)
```

The caption DB and image target cache both need row metadata:

```text
embedding row i -> target_id
```

So evaluation can compare:

```text
retrieved target_id == true target_id
```

### Why `candidate_id` is also useful

`candidate_id` identifies the **query/reference image**, not the answer.
It is useful for:

1. **Qualitative debugging**

   If a query fails, we need to know which reference image produced that
   generated caption and retrieval result. The qualitative dump records it as
   `query_id`.

2. **Image loading / cache lookup**

   The local image cache can be organized however we want. In the current
   setup it is keyed by filename stem, so loading the candidate image is:

   ```python
   image_id = item["candidate_id"]
   path = image_cache_root / f"{image_id}.jpeg"
   ```

   The point is not that this is the only possible cache layout. The point is
   that downstream code can use a normalized key instead of depending on raw
   FACap path strings.

3. **Train/eval image-level filtering**

   For contrastive training, it is not enough to exclude eval target images.
   If an eval reference image appears elsewhere in training as a target, the
   model has still seen that eval image. Clean filtering uses both:

   ```text
   eval candidate IDs + eval target IDs
   ```

### Final rationale

`candidate_id` and `target_id` are convenience fields derived from the raw
paths. They are not required by the math, but they make the system cleaner:

- path fields preserve dataset provenance
- ID fields provide compact operational keys
- downstream code does not repeatedly parse path strings
- retrieval DB rows, qualitative logs, image cache lookup, and split filtering
  use the same image identity convention

### Plain-language version from the discussion

我们存 ID 的核心原因是：不要把“图片是谁”绑定到“图片在某个文件系统 / 数据集
manifest 里怎么写”。

从零开始设计可以这样推：

1. FACap 给我们的原始信息是：

   ```text
   candidate_image_path = "f200k_images/dresses/12345678_0.jpeg"
   target_image_path    = "f200k_images/dresses/87654321_0.jpeg"
   modification_text    = "make it floral and sleeveless"
   target_caption       = "A sleeveless floral dress ..."
   ```

   如果只是读数据，这四个字段够了。

2. 但 retrieval 系统需要判断“哪张图是哪张图”：

   ```text
   eval: retrieved image == true target?
   cache lookup: this embedding row belongs to which image?
   qualitative dump: this query/reference image is which image?
   split filtering: train and eval share the same image?
   ```

   这些操作需要一个 image key。

3. 可以直接用 path 当 key，但 path 混合了 identity 和 location：

   ```text
   f200k_images/dresses/12345678_0.jpeg
   ```

   里面有两种信息：

   ```text
   identity: 12345678_0
   dataset/layout: f200k_images/dresses/...jpeg
   ```

   如果以后图片位置、目录结构、extension、cache layout 变了，path string 会变，
   但图片 identity 没变。所以 path 不是不能当 key，而是它把 identity 和
   storage layout 绑在一起了。

4. 所以我们 normalize 出 image ID：

   ```text
   candidate_id = "12345678_0"
   target_id    = "87654321_0"
   ```

   这两个 ID 是从 path 里 derived 出来的，不是额外 label，不是新的 supervision。

5. 为什么 candidate 也要 ID，不只是 target 要 ID：

   `target_id` 很直观：判断 retrieve 对没对。

   `candidate_id` 是 query/reference image 的 identity。它用于：

   ```text
   1. 记录这条 query 是哪张 reference image
   2. 从 image cache 找 reference image
   3. qualitative dump/debug/demo 里显示 query_id
   4. train/eval split filtering：避免 eval candidate image 在 train 里出现
   ```

   尤其是第 4 点：如果 eval 里的 candidate image 在 train 里作为 target 出现过，
   也算 image-level leakage。所以 clean split 需要同时看：

   ```text
   candidate_id
   target_id
   ```

Meeting wording:

> We keep both the raw FACap path and a normalized image ID. The path preserves
> provenance; the ID is the operational key. Retrieval/evaluation is naturally
> phrased as "reference image ID plus modification retrieves a ranked list of
> target image IDs." The ID is derived from the path, not extra supervision.

---

## Q9: What's the actual query Qwen2VL gets, and what query strategies were considered?

**Context:** The real VLM backends (`Qwen2VLCaptioner`,
`SpeechQwen2VLCaptioner`) are the only ones that actually look at
the candidate image and generate a real caption. What's the prompt
they receive? Were other prompt designs considered?

### The actual query

The prompt template is a class constant at
`src/baseline/vlm_caption.py:78–82`:

```python
PROMPT_TEMPLATE = (
    "Given the reference fashion image and the modification instruction, "
    "write a concise caption describing the target fashion item after "
    "applying the modification."
)
```

In `caption()`, this gets combined with the modification text and
the candidate image into a chat-style message:

```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": str(path)},
        {"type": "text", "text": f"{self.PROMPT_TEMPLATE}\n\nModification: {modification}"},
    ]},
]
```

So a real query for the example item from Q5 looks like:

```
[image: candidate dress photo]
Given the reference fashion image and the modification instruction,
write a concise caption describing the target fashion item after
applying the modification.

Modification: the dress is shorter and red instead of black
```

The model is asked to generate a caption *describing the target
after the modification has been applied* — not a literal
description of the reference image, and not a paraphrase of the
modification. The expected output is a fashion-style caption like
*"A red, knee-length sleeveless dress with a flared silhouette."*

Generation kwargs (`vlm_caption.py:84–88`):

```python
GENERATION_KWARGS = {
    "max_new_tokens": 128,    # cap on caption length
    "num_beams": 1,           # greedy decoding (no beam search)
    "do_sample": False,       # deterministic, no sampling
}
```

Greedy + deterministic so the same input always produces the same
caption — important for reproducibility of the baseline numbers.

> **Update (2026-04-30).** After the empirical caption-length
> measurement below, `max_new_tokens` was bumped from `128` to
> `256` to remove a potential silent-truncation risk if a future
> prompt elicits FACap-length captions (~170 tokens at the max).
> Under greedy decoding the bump is free on outputs that finish
> early under the current `"concise"` prompt — it only matters
> when the model wants to keep going. The remainder of this Q9
> still references `128` because the analysis predates the bump
> and the *prompt* (not the cap) is the binding length constraint.

### Strategies considered (and not chosen)

There are many ways to construct this prompt. Trade-offs:

| Strategy | What it does | Trade-off |
|---|---|---|
| **Zero-shot (current)** | Just describe the task in the prompt; no examples | Simple, fastest, no example-bias |
| **Few-shot** | Show 2–3 example (image, modification, ideal caption) triplets in the prompt before the real query | Output style matches the examples better; but the "ideal" examples bias the model toward a specific phrasing, which may or may not match FACap's caption style |
| **Chain-of-thought** | Tell the model to first reason about the change, then output the caption | May produce more accurate captions; doubles token cost; reasoning text needs to be parsed out of the final answer |
| **Two-step** | First call: describe the reference image. Second call: apply the modification to that description | More controllable, more inspectable; but two model calls per query → 2× cost and latency |
| **Style-matched prompt** | Add *"in the style of catalog descriptions, ~80 words, mention color / silhouette / material"* to match FACap's caption style | Should improve retrieval directly (the generated captions look more like the DB captions); but couples our prompt to a specific dataset's style and may transfer poorly to other catalogs |

### How query strategy affects retrieval

Retrieval works by encoding the generated caption AND the DB
captions with the *same* SBERT encoder, then ranking by cosine
similarity. So whatever style the VLM generates determines how
close the embeddings land to the DB caption embeddings.

Concretely:

- If the VLM produces a 3-word caption (*"red short dress"*) and
  FACap captions are 80-word descriptions (*"This is a sleeveless,
  knee-length, vibrant red mini dress with..."*), the embeddings
  will differ in topic structure — retrieval will be weaker even if
  both refer to the same item.
- If the VLM produces FACap-style 80-word captions, embeddings will
  be closer in style → retrieval will be stronger, *for the right
  reasons*.
- If the VLM hallucinates content (mentions colors not implied by
  the modification, invents materials), retrieval may sometimes
  succeed by accident — the embeddings still cluster around the
  right item, but the interpretation is fragile.

The current zero-shot prompt doesn't optimize for any of this — it
asks for "a concise caption", which may be much shorter than FACap
targets. **Plan_3's first run will tell us how big this style
mismatch is.** If it's a problem, we'd revisit the prompt
(style-matched or few-shot).

### Should we match the DB style? (Yes, mostly.)

Our case is **symmetric retrieval**: both the query (generated
caption) and the DB items (FACap captions) are *captions of fashion
items*. Same content type, same domain. For symmetric retrieval,
matching style is genuinely helpful — if both sides encode the same
kinds of attributes in the same way, embeddings cluster correctly.

**What we should match:**

- ✅ **Length / scope** — roughly comparable number of attributes
  mentioned.
- ✅ **Vocabulary register** — fashion-specific words (silhouette,
  neckline, hemline) instead of generic ("a thing that is red").
- ✅ **Order of attributes** — less critical for SBERT (mean-pooling
  is order-insensitive), but still helps marginally.

**What we shouldn't match:**

- ❌ **Boilerplate** — phrases like *"This is a..."* or *"The image
  shows..."* are noise. They appear in every DB caption *and* every
  generated caption, so they cancel out (no signal added) but they
  cost tokens. Better to skip.
- ❌ **Hallucinated content** — if the DB mentions a material the
  modification doesn't imply, *don't* invent material to match.
  Hallucinations land embeddings randomly; the cure is worse than the
  disease.

### How professionals do this

The IR / RAG community has converged on a few standard moves:

| Move | Idea | When to use |
|---|---|---|
| **Symmetric encoder + style-matched prompt** | Use few-shot or explicit style instructions to nudge VLM output toward DB style; encode both with the same SBERT-style model | Our case: DB and queries are same content type |
| **Asymmetric encoder** | Use a model trained with separate "query:" / "passage:" prefixes (e.g., BGE, E5, msmarco-MiniLM) so the encoder *learns* to bridge query↔doc style differences | When queries are short / questions but docs are long / passages — not our case |
| **Two-stage retrieval (rerank)** | Dense first-stage retrieves top-50; a cross-encoder reranks them by joint scoring | When first-stage Recall@50 is good but Recall@1 is poor |
| **Hybrid retrieval (BM25 + dense)** | Sparse (BM25, keyword) + dense (SBERT) scores combined. BM25 is robust to length and rewards exact term overlap | When key attributes are rare/specific words (brand names, exact colors) |
| **Few-shot prompting** | Show 2–3 (image, modification, ideal_caption) triplets in the prompt | The standard cheap fix to nudge output style |

**The professional default for our exact problem** (composed image
retrieval via captions) is roughly:

1. Use a symmetric SBERT-family encoder for both sides (✅ we do this).
2. Prompt the VLM with explicit style guidance — e.g. *"in the style
   of fashion catalog descriptions, ~80 words, mention color,
   silhouette, fabric, length, neckline"*. Much cheaper than
   retraining anything.
3. Include 1–2 few-shot examples if zero-shot isn't enough.
4. If retrieval is still bad, try a cross-encoder reranker as a
   second stage.
5. Only swap the encoder (e.g., to BGE-large or fashion-CLIP) once
   1–4 are exhausted.

### Empirical: how long are FACap captions actually?

Measured from the `smoke_oracle` DB (n=1000 dress targets):

| Stat | chars | ~words |
|---|---|---|
| min | 201 | 40 |
| p10 | 427 | 85 |
| median | 566 | 113 |
| mean | 538 | 108 |
| p90 | 615 | 123 |
| max | 651 | 130 |

So FACap captions are **~110 words median**, range mostly 85–125
words — longer than the "80-word" rule-of-thumb estimate. That makes
the zero-shot prompt's *"concise caption"* instruction a likely
mismatch: Qwen2-VL with `max_new_tokens=128` (~95 words ceiling) plus
an explicit *"concise"* instruction will probably land at 30–60
words. That's a real gap.

### Decision (2026-04-30): zero-shot first, evidence-driven iteration

The above analysis suggests style-matched prompting should help — but
hardcoding the style (e.g., explicit *"~80 words, mention color,
silhouette, fabric, length, neckline"*) feels too prescriptive
*before we have a number*. The plan:

1. **Plan_3 runs zero-shot** (current `PROMPT_TEMPLATE`) and produces
   the first metrics + qualitative samples.
2. **Read ~5 generated captions** from `qualitative/results.jsonl`.
   If they're 30-word generic vs. FACap's 110-word fashion-rich, the
   style mismatch is real and visible in the data.
3. **Iterate the prompt** with style guidance only if the numbers
   warrant it.

The principle: don't optimize a prompt against an *imagined* problem.
Run the experiment, look at the failures, then prescribe.

**Lesson:** Prompt design is a *real* knob, not just boilerplate.
For caption-generation retrieval, the prompt determines where
generated captions land in embedding space relative to the catalog
— which is the entire mechanism the metrics measure. Worth
returning to once Plan_3 produces the first numbers.
