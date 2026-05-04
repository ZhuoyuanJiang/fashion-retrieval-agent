# Notes: Plan-8 demo-app concepts

Personal study / reference notes for Plan-8 (the demo MVP). Companion to:
`Documentation/Plan_8_20260503.md`.

---

## Q1: What do "cached", "live", and "mocked" mean in this demo?

These three terms come up throughout Plan-8 and the v0.1/v0.2/v0.3 stage progression. They describe **how a piece of the pipeline produces its output at request time**, not what the output is.

### Cached

> We ran the pipeline OFFLINE on a fixed preset, saved the answer to JSON. At demo time the app reads JSON. No model loaded, no GPU needed at demo time. Real result (we computed it), just precomputed.

- The **inputs are fixed** (the 8 preset triplets).
- The **model output is real** — we ran the actual trained pipeline once and stored its output verbatim.
- The **latency stored in JSON is a real recorded measurement** from the offline run, not a synthesised number. (Caveat: it should be labelled as "recorded" in the UI, not as live inference time.)
- The cost is paid once, ahead of time.

When to use: any pipeline whose output is deterministic for a given input (Pipeline 1's caption is deterministic at temperature 0; cosine similarity is deterministic). The retrieval results for the preset set are an obvious candidate.

### Live

> The model is loaded into GPU memory at startup; when the user clicks Run, the app actually runs inference on whatever they uploaded. Real-time, input-dependent, costs GPU.

- The **inputs are arbitrary** (whatever the user uploads / records / types).
- The model has to be **resident in GPU memory** for the whole demo session.
- The **latency is the actual wall-clock time** of the forward pass at request time.
- The cost is paid every request.

When to use: any feature where the user's specific input matters and we want them to see the model's actual behaviour on it. Pipeline 2 in v0.2+ is live because it's the headline result and reviewers must see it run.

### Mocked

> Hard-coded answer regardless of input. Used in v0.1 only for ASR (whatever audio comes in, return the preset's predefined text).

- The **input is ignored**.
- The **output is hard-coded** ahead of time.
- This is a **Wizard-of-Oz substitute**: it makes the UX flow feel real (user records audio, sees a transcript appear, results follow) without doing the actual computation.
- The cost is zero.

When to use: only for steps where running the real thing isn't yet possible (the native-audio model isn't trained) OR where running the real thing is a distraction from the thing we actually want to demo (in v0.1 we want the UX flow, not Whisper).

### Why cached is preferred over mocked for retrieval results

Cached and mocked both produce the answer instantly at demo time. The difference is *what* the answer is.

- **Cached** answers are real. We ran the model. The top-K thumbnails the user sees are what the trained pipeline actually produced. If a reviewer asks "what did Pipeline 2 retrieve for this query?", the cached answer is the truth.
- **Mocked** answers are made up. If we mocked retrieval results, we'd have to either hand-pick "good-looking" matches (misleading — those aren't what the model produced) or just run the model to find good ones, at which point we should save them as cache anyway.

So: **mocking only saves work for components where there is no real model yet** (P3 column = placeholder; ASR in v0.1 = mocked because we want to skip the Whisper plumbing for the first build). For everything else, cached ≥ mocked.

### Summary table

| | Inputs | Output | Latency at demo | GPU at demo |
|---|---|---|---|---|
| **Cached** | Fixed (presets) | Real (precomputed) | Real (recorded) | None |
| **Live** | Arbitrary | Real (computed now) | Real (actual) | Yes |
| **Mocked** | Ignored | Hard-coded | Synthetic / zero | None |

### Stage-by-stage usage

| Stage | Cached | Live | Mocked |
|-------|--------|------|--------|
| **v0.1** | P1 results, P2 results (both pipelines on the 8 presets) | — | ASR (returns preset's `mock_transcript`) |
| **v0.2** | P1 results (presets only) | P2 inference, Whisper ASR | — |
| **v0.3** | — | P1 + P2 inference, Whisper ASR | — |

Pipeline 3 is a *placeholder column* across all stages until the native-audio model is trained — it's neither cached nor live nor mocked, just an architectural promise rendered as text in the UI.

---

## Q2: What is Gradio, and why are we using it?

Gradio is a Python library that turns a Python function into a web UI with no HTML / CSS / JavaScript. We picked it as the demo's framework in Plan_8 §5.

### What it is

You write a function — `def retrieve(image, mod_text) -> list[image_paths]` — and Gradio generates the entire frontend (image upload widget, text box, results gallery) automatically. The whole thing runs as one Python process that hosts an HTTP server on `localhost:7860`.

Minimal example:

```python
import gradio as gr

def retrieve(image, mod_text):
    # ... call our pipelines ...
    return [path1, path2, path3]  # list of image paths

demo = gr.Interface(
    fn=retrieve,
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=gr.Gallery(),
)
demo.launch()                     # opens http://localhost:7860
demo.launch(share=True)           # also gets a public *.gradio.live URL valid 72h
```

That's a working web app. The same flow in Flask + custom HTML/JS would be 200+ lines across multiple files.

### Why we need it for this demo

Without Gradio, the demo would require:

- **Frontend** — HTML + CSS + JavaScript: image upload widget, microphone capture, top-K thumbnail gallery, layout, examples row.
- **Backend** — FastAPI / Flask: HTTP handlers, multipart file upload parsing, MIME type handling, websocket connections.
- **Glue** — temporary file storage, audio file format handling, error rendering, etc.

Easily 500+ lines split across two or three languages. Gradio collapses all of that into ~100 lines of Python in one file.

### Why specifically Gradio (not alternatives)

| Option | Verdict | Reason |
|---|---|---|
| **Gradio** | ✅ chosen | Native widgets for every component we need (image upload, mic, gallery, examples). One Python file. `share=True` gives a public URL with no deployment work. De facto standard in the ML/HuggingFace ecosystem — mentors recognise it. |
| Streamlit | ❌ | Reruns the whole script on every interaction. With 28 GB models loaded, every click would reload the model. Dealbreaker. |
| FastAPI + custom HTML/JS | ❌ | 5–10× the code. Two languages, more bug surface. Wrong tool for an MVP. |
| Jupyter notebook | ❌ | Not a shareable webapp; can't be opened by a non-technical mentor in their browser. |

### Native widgets we use

- `gr.Image(type="pil")` — drag-and-drop image upload, returns a `PIL.Image`.
- `gr.Audio(source="microphone")` — one-click record, returns a 16 kHz waveform.
- `gr.Textbox()` — text input / output, editable.
- `gr.Gallery()` — grid of thumbnails with hover scores. Used for top-K results in each pipeline column.
- `gr.Examples()` — a clickable row of preset thumbnails that auto-fill inputs.
- `gr.Blocks()` — a layout container with rows/columns/tabs. We use this for the three-column results layout.

### What Gradio is *not* good for

- Custom branding / pixel-perfect design — fine for an MVP, not a product.
- Heavy concurrency / multi-tenant scaling — fine for a single-user demo, not a production app.
- Highly custom interactions outside the supplied widget set — covers everything we need here.

If the demo ever graduates to a proper product (Plan-10 territory), we'd swap Gradio for a React + FastAPI stack. For now Gradio is the right tool.

### How it relates to v0.1 / v0.2 / v0.3

Gradio is the UI in all three stages — what changes between stages is which functions get called when the user clicks Run:

- **v0.1**: Run handler reads the cached JSON; mocked ASR returns the preset's predefined transcript.
- **v0.2**: Run handler calls the real `ContrastivePipeline` for P2 and the real Whisper for ASR; P1 still reads JSON.
- **v0.3**: Run handler calls the real `CaptionPipeline` for P1 too.

The Gradio Blocks layout doesn't change between stages.
