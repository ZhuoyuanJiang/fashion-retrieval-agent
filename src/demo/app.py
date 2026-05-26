"""Plan-8 demo app — scripted pipeline comparison + live audio row.

Layout (vertical sections, full-width galleries):
  1. Header (title + stage badge)
  2. Preset row (clickable thumbnails)
  3. Active query (candidate image + mod text + preset TTS clip / mic)
  4. K slider + Run button
  5. P1 — caption-based retrieval        (cached)
  6. P2 — direct contrastive embedding   (cached)
  7. P3 — text two-tower (Plan-13)       (cached)
  8. P4 — audio two-tower (Plan-15)      (cached)
  9. Live audio row — record your own speech, retrieve live (LIVE_AUDIO=1)
 10. About expander

P1–P4 are read from runs/demo/preset_cache.json (real model outputs, computed
offline once). The live row runs the Plan-15 audio two-tower in-process on a
GPU; it is only built when LIVE_AUDIO=1 so the cached demo still runs CPU-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from . import config, gallery
from .pipelines.base import PipelineResult


# ---------------------------------------------------------------------------
# Live audio two-tower — loaded once at startup when LIVE_AUDIO=1
# ---------------------------------------------------------------------------
_AUDIO_MODEL = None              # TwoTowerSharedBackbone (audio query modality)
_AUDIO_GALLERY = None            # (emb tensor, ids) — target-tower gallery


def load_audio_tower() -> None:
    """Load the Plan-15 audio two-tower + its gallery embeddings into memory.

    Called once from main() when LIVE_AUDIO=1. Constructing the model pulls in
    the ~9B speechQwen2VL base, so this takes ~10s and needs a GPU.
    """
    global _AUDIO_MODEL, _AUDIO_GALLERY
    if _AUDIO_MODEL is not None:
        return
    from .pipelines.two_tower import load_gallery, load_two_tower
    print("[live-audio] loading audio two-tower (Plan-15)...", flush=True)
    _AUDIO_GALLERY = load_gallery(config.AUDIO_2T_GALLERY)
    _AUDIO_MODEL = load_two_tower(
        config.AUDIO_2T_CKPT, "audio", config.LIVE_AUDIO_DEVICE,
    )
    print("[live-audio] ready", flush=True)


def _load_preset_cache() -> dict:
    if not config.PRESET_CACHE_JSON.exists():
        raise FileNotFoundError(
            f"preset cache not found at {config.PRESET_CACHE_JSON}; "
            "run src/demo/precompute_presets.py first (or copy the placeholder)."
        )
    with config.PRESET_CACHE_JSON.open() as f:
        return json.load(f)


CACHE = _load_preset_cache()
PRESETS = {p["preset_id"]: p for p in CACHE["presets"]}
PRESET_ORDER = [p["preset_id"] for p in CACHE["presets"]]


# Hand-written narrative for each curated preset. Keep short — shown in the
# button label and in the "About these 8 examples" expander.
PRESET_NARRATIVES: dict[str, dict[str, str]] = {
    "preset_01": {
        "emoji": "🏆",
        "title": "Red V-neck dress, belt + tie",
        "story": "**P2 huge win.** Long, multi-attribute modification (color + neckline + cinched waist + tie belt). P1's caption captures the gist but the cosine search wanders — true target buried at #40. P2 picks up the full visual intent and ranks it #1.",
    },
    "preset_02": {
        "emoji": "🏆",
        "title": "Gray high-neck gathered waist",
        "story": "**P2 narrow win.** Both pipelines find the true target high in the list. P1 ranks it #5 (caption is solid), P2 promotes it to #1 — small win for end-to-end embedding.",
    },
    "preset_03": {
        "emoji": "🏆",
        "title": "Pastel pink satin, modest cut",
        "story": "**P2 dramatic win.** The caption is decent, but cosine search against caption DB misses the right item entirely (true rank #109 — outside top-50). P2's embedding nails it at #1. Best example of why an end-to-end embedding can beat the caption bottleneck.",
    },
    "preset_04": {
        "emoji": "🏆",
        "title": "Royal blue sunburst pattern",
        "story": "**P2 win.** Distinctive sunburst design. P1 #7, P2 #2 — visual pattern signal is captured more directly by the embedding model.",
    },
    "preset_05": {
        "emoji": "🏆",
        "title": "Short dark denim, side pockets",
        "story": "**P2 win.** Specific construction detail (\"side pockets\") gets blurred by the caption-encoder bottleneck. P1 #17, P2 #4.",
    },
    "preset_06": {
        "emoji": "🤝",
        "title": "White lace-up shift",
        "story": "**P1 narrow win.** Caption is precise and the visual is unambiguous, so caption-encoder retrieval wins by a hair: P1 #1 vs P2 #2. Honest demonstration that the older pipeline isn't dead.",
    },
    "preset_07": {
        "emoji": "🤝",
        "title": "High neck, color-block black/navy",
        "story": "**P1 big win.** Unusual feature combo (high neck + color block) is well-captured by an explicit caption; P2's embedding misses the precise color-pairing semantics. P1 #2, P2 #45.",
    },
    "preset_08": {
        "emoji": "❌",
        "title": "Gray tie-split (P2 fail)",
        "story": "**Honest failure.** The modification text relies on a subtle distinction (\"tie split\" vs \"tie strap\"). P1's caption captures it precisely (#1). P2 misses entirely (true target outside top-50). Reminder of P2's current ceiling — R@10 = 0.40 leaves real failures.",
    },
}


def _result_from_cached(entry: dict, true_target_id: str | None) -> PipelineResult:
    """Build a PipelineResult from a cached pipeline JSON entry."""
    target_ids = entry["target_ids"]
    paths = [gallery.image_path(tid) for tid in target_ids]
    return PipelineResult(
        target_ids=target_ids,
        scores=entry["scores"],
        image_paths=paths,
        latency=dict(entry.get("latency", {})),
        intermediate=dict(entry.get("intermediate", {})),
        true_target_id=true_target_id,
        true_target_rank=entry.get("true_target_rank"),
    )


def _format_latency(latency: dict[str, float]) -> str:
    if not latency:
        return "_(no latency recorded)_"
    parts = [f"{k.replace('_s', '')}={v:.2f}s" for k, v in latency.items()]
    total = sum(latency.values())
    return f"**Latency** (recorded): {' + '.join(parts)} = **{total:.2f}s**"


def _format_true_rank(rank: int | None, k_displayed: int) -> str:
    if rank is None:
        return "True target: **not in top-50** ❌"
    if rank <= k_displayed:
        return f"True target ranked **#{rank}** ✓ (within displayed top-{k_displayed})"
    return f"True target ranked **#{rank}** (outside displayed top-{k_displayed})"


def _gallery_items(result: PipelineResult, k: int) -> list[tuple[str, str]]:
    """Return list of (image_path_str, caption) for gr.Gallery."""
    items = []
    for i, (tid, score, path) in enumerate(
        zip(result.target_ids[:k], result.scores[:k], result.image_paths[:k]), start=1
    ):
        marker = " ★" if tid == result.true_target_id else ""
        caption = f"#{i}  {tid}{marker}  ({score:.2f})"
        items.append((str(path), caption))
    return items


# ---------------------------------------------------------------------------
# Click handlers
# ---------------------------------------------------------------------------

def on_preset_click(preset_id: str):
    """Load preset image, modification text, TTS clip, ground truth into the query area."""
    p = PRESETS[preset_id]
    cand_path = str(gallery.image_path(p["candidate_image_id"]))
    gt_path = str(gallery.image_path(p["true_target_id"])) if p.get("true_target_id") else None
    audio_path = config.PRESET_AUDIO_DIR / f"{preset_id}.wav"
    audio_path = str(audio_path) if audio_path.exists() else None
    narrative = PRESET_NARRATIVES.get(preset_id, {})
    emoji = narrative.get("emoji", "")
    title = narrative.get("title", "")
    story = narrative.get("story", p.get("notes", ""))
    label = (
        f"### {emoji} Active preset: `{preset_id}` — {title}\n\n"
        f"{story}"
    )
    return (
        preset_id,                          # State
        cand_path,                          # candidate image
        p["modification_text"],             # mod text
        audio_path,                         # preset TTS clip
        p["mock_transcript"],               # mock transcript
        gt_path,                            # ground truth image
        label,
    )


def on_run_p1(active_preset_id: str | None, k: int):
    """Render Pipeline 1 results for the active preset."""
    if not active_preset_id:
        return [], "", "_no preset selected — pick one above_"
    preset = PRESETS[active_preset_id]
    p1 = _result_from_cached(preset["p1"], preset.get("true_target_id"))
    return (
        _gallery_items(p1, k),
        p1.intermediate.get("caption", ""),
        "\n\n".join([
            _format_latency(p1.latency),
            _format_true_rank(p1.true_target_rank, k),
        ]),
    )


def on_run_p2(active_preset_id: str | None, k: int):
    """Render Pipeline 2 results for the active preset."""
    if not active_preset_id:
        return [], "_no preset selected — pick one above_"
    preset = PRESETS[active_preset_id]
    p2 = _result_from_cached(preset["p2"], preset.get("true_target_id"))
    return (
        _gallery_items(p2, k),
        "\n\n".join([
            _format_latency(p2.latency),
            _format_true_rank(p2.true_target_rank, k),
        ]),
    )


def on_run_text2t(active_preset_id: str | None, k: int):
    """Render the text two-tower (Plan-13) results for the active preset."""
    if not active_preset_id:
        return [], "_no preset selected — pick one above_"
    preset = PRESETS[active_preset_id]
    r = _result_from_cached(preset["text2t"], preset.get("true_target_id"))
    return (
        _gallery_items(r, k),
        "\n\n".join([
            _format_latency(r.latency),
            _format_true_rank(r.true_target_rank, k),
        ]),
    )


def on_run_audio2t(active_preset_id: str | None, k: int):
    """Render the audio two-tower (Plan-15) results for the active preset."""
    if not active_preset_id:
        return [], "_no preset selected — pick one above_"
    preset = PRESETS[active_preset_id]
    r = _result_from_cached(preset["audio2t"], preset.get("true_target_id"))
    return (
        _gallery_items(r, k),
        "\n\n".join([
            _format_latency(r.latency),
            _format_true_rank(r.true_target_rank, k),
        ]),
    )


def on_run_all(active_preset_id: str | None, k: int):
    """Render all four cached pipelines at once for the master Run-all button."""
    p1_gallery, p1_caption, p1_meta = on_run_p1(active_preset_id, k)
    p2_gallery, p2_meta = on_run_p2(active_preset_id, k)
    t2t_gallery, t2t_meta = on_run_text2t(active_preset_id, k)
    a2t_gallery, a2t_meta = on_run_audio2t(active_preset_id, k)
    return (
        p1_gallery, p1_caption, p1_meta,
        p2_gallery, p2_meta,
        t2t_gallery, t2t_meta,
        a2t_gallery, a2t_meta,
    )


def on_run_live(active_preset_id: str | None, audio_path: str | None, k: int):
    """Encode (active preset's candidate image + the user's recorded speech)
    through the Plan-15 audio two-tower and retrieve live against the gallery.

    Unlike P1–P4 this is not cached — the audio is whatever the user just
    recorded. The candidate image is still the active preset's, so the user
    is asking "given this garment, find what my spoken modification describes."
    """
    if not active_preset_id:
        return [], "_no preset selected — pick an example above first_"
    if not audio_path:
        return [], "_no audio recorded — use the microphone above_"
    if _AUDIO_MODEL is None:
        return [], "_audio tower not loaded — relaunch with `LIVE_AUDIO=1`_"

    from .pipelines.two_tower import run_two_tower_inference
    from .precompute_presets import find_rank, load_candidate_image

    preset = PRESETS[active_preset_id]
    image = load_candidate_image(preset["candidate_image_id"])
    g_emb, g_ids = _AUDIO_GALLERY
    ids, scores, lat = run_two_tower_inference(
        _AUDIO_MODEL, config.LIVE_AUDIO_DEVICE, g_emb, g_ids,
        image, audio_path, k=config.K_MAX,
    )
    true_tid = preset.get("true_target_id")
    result = PipelineResult(
        target_ids=ids,
        scores=scores,
        image_paths=[gallery.image_path(t) for t in ids],
        latency=lat,
        intermediate={},
        true_target_id=true_tid,
        true_target_rank=find_rank(true_tid, ids),
    )
    return (
        _gallery_items(result, k),
        "\n\n".join([
            _format_latency(result.latency),
            _format_true_rank(result.true_target_rank, k),
        ]),
    )


# ---------------------------------------------------------------------------
# UI text blocks
# ---------------------------------------------------------------------------

ABOUT_MD = """\
### About this demo

This is the **v0.1 scripted demo** of a fashion-retrieval research project. The user provides a
candidate fashion image plus a modification (text or speech), and three retrieval pipelines compete
to find the matching gallery item from ~59,000 dress images (FACap dataset).

In v0.1, results are read from a precomputed JSON (real model outputs, computed offline once on the
preset set). ASR is mocked: whatever audio is recorded, we return the preset's predefined transcript.
Latencies shown are **recorded**, not real-time.

When v0.2 ships, Pipeline 2 + Whisper run live on any user input; v0.3 makes Pipeline 1 live too.
"""

P1_DESCRIPTION = """\
## 🅿️1 — Caption-based Retrieval (Phase A baseline)

**How it works:** A vision-language model (Qwen2-VL-7B) looks at the candidate image *and* reads the
modification text, then writes a single sentence describing what the *target* item should look like.
A separate text encoder (Marqo FashionCLIP) embeds that sentence, and we cosine-rank against a
pre-built database of caption embeddings for all 59 k gallery items.

**Strength:** captions are interpretable — you can see exactly what the model "thinks" the target looks like.
**Weakness:** information bottleneck — the entire query must be compressed into one sentence.

**Headline accuracy:** R@10 = 0.533 on the FACap dress eval slice.
"""

P2_DESCRIPTION = """\
## 🅿️2 — Direct Contrastive Embedding (Plan-5/6, our trained model)

**How it works:** A fine-tuned Qwen2-VL-7B takes the same `(candidate image, modification text)` pair
and emits a single 512-dimensional query vector — no caption intermediate. The model was trained with
multi-positive symmetric InfoNCE on FACap triplets to align this query vector with the target image's
FashionCLIP embedding. Top-K is cosine-ranked over the same 59 k gallery, against the frozen
FashionCLIP image embeddings.

**Strength:** end-to-end optimisation — the model learns whatever query representation works best for retrieval.
**Weakness:** opaque — no human-readable intermediate; harder to debug a specific failure.

**Headline accuracy:** R@10 = 0.402 (Plan-6 best checkpoint, step 1664 / epoch 16).
"""

TEXT2T_DESCRIPTION = """\
## 🅿️3 — Text Two-Tower (Plan-13, shared backbone)

**How it works:** A shared Qwen2-VL-7B backbone carries two LoRA adapters — one
for the query side `(candidate image, modification text)`, one for the target
image — trained together with multi-positive symmetric InfoNCE so the query
embedding and the target image embedding land in the same 512-d space. Unlike
P2, the target tower is *also* trained (not a frozen FashionCLIP encoder).

**Headline accuracy:** R@10 = 0.654 on the FACap dress eval slice — the
project's best text result.
"""

AUDIO2T_DESCRIPTION = """\
## 🅿️4 — Audio Two-Tower (Plan-15, native speech)

**How it works:** The same shared two-tower as P3, but the query-side
modification enters as **spoken audio** — straight through the model's Whisper
encoder, with no ASR step. The query is `(candidate image, raw speech)`. Trained
fresh from scratch on TTS-synthesized speech.

**Headline accuracy:** R@10 = 0.624 (dev-selected peak) / 0.643 (best epoch) —
within ~0.01–0.03 of the text two-tower above. Swapping typed text for spoken
audio costs almost nothing in retrieval quality.
"""

LIVE_DESCRIPTION = """\
## 🎙 Live — Record Your Own Modification (Audio Two-Tower)

**Not cached.** Pick an example above to set the candidate garment, then record
your own voice describing what should change. The clip goes straight into the
Plan-15 audio two-tower's Whisper encoder — no ASR — and the model retrieves
live against the full ~59 k gallery on this GPU host.

Speak naturally, ~5–12 seconds. The candidate image stays the active preset's;
you are supplying a fresh spoken modification for it.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Fashion Retrieval — Pipeline Comparison",
    ) as demo:
        gr.Markdown(
            f"# 👗 Fashion Retrieval — Pipeline Comparison\n"
            f"**Stage:** `{config.STAGE}` &nbsp;·&nbsp; "
            f"Same query (image + modification), three retrieval backends side-by-side."
        )

        # State: which preset is currently active
        active_preset = gr.State(value=None)

        # ----- Preset row -----
        gr.Markdown(
            "### 1. Pick an example\n"
            "🏆 = Pipeline 2 (contrastive) wins &nbsp;·&nbsp; "
            "🤝 = Pipeline 1 (caption) wins &nbsp;·&nbsp; "
            "❌ = both fail (honest)"
        )
        with gr.Row():
            preset_btns = []
            for pid in PRESET_ORDER:
                p = PRESETS[pid]
                narrative = PRESET_NARRATIVES.get(pid, {})
                emoji = narrative.get("emoji", "")
                title = narrative.get("title", p["modification_text"][:60] + "…")
                btn = gr.Button(value=f"{emoji} {pid}\n{title}")
                preset_btns.append((pid, btn))

        with gr.Accordion("📋 About these 8 examples — what each one shows", open=False):
            rows = []
            for pid in PRESET_ORDER:
                p = PRESETS[pid]
                n = PRESET_NARRATIVES.get(pid, {})
                p1_rank = p["p1"]["true_target_rank"]
                p2_rank = p["p2"]["true_target_rank"]
                p1s = f"#{p1_rank}" if p1_rank is not None else "—"
                p2s = f"#{p2_rank}" if p2_rank is not None else "outside top-50"
                rows.append(
                    f"| {n.get('emoji', '')} `{pid}` | **{n.get('title', '')}** | {p1s} | {p2s} | {n.get('story', '')} |"
                )
            gr.Markdown(
                "These 8 presets were curated from the 1000-query FACap headline split to "
                "tell a balanced story (5 P2 wins, 2 P1 wins, 1 P2 fail). Selection criterion "
                "documented in `src/demo/precompute_presets.py`.\n\n"
                "| | Preset | P1 rank | P2 rank | Story |\n"
                "|---|---|---|---|---|\n" + "\n".join(rows)
            )

        # ----- Active query -----
        gr.Markdown("### 2. Your query")
        active_label = gr.Markdown("_no preset selected — click one above_")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**🖼 Candidate image**\n_The reference garment._")
                candidate_img = gr.Image(label="", type="filepath", interactive=False, height=320)
            with gr.Column(scale=2):
                mod_text = gr.Textbox(
                    label="Modification (typed) — what should change about the garment?",
                    lines=3,
                    interactive=True,
                )
                preset_audio_player = gr.Audio(
                    label="🔊 Spoken modification (TTS) — the exact clip the "
                          "audio two-tower (P4) hears",
                    type="filepath",
                    interactive=False,
                )
                gr.Markdown("**Or speak the modification:**")
                mic = gr.Audio(sources=["microphone"], type="numpy", label="🎙 Record audio")
                transcript = gr.Textbox(
                    label="Transcript (mocked in v0.1 — returns preset's predefined text)",
                    lines=2,
                    interactive=True,
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "**✅ Ground truth**\n"
                    "_The correct match. Visible in preset mode only — to "
                    "show how close each pipeline gets to the right answer._"
                )
                ground_truth_img = gr.Image(label="", type="filepath", interactive=False, height=320)

        with gr.Row():
            k_slider = gr.Slider(
                minimum=config.K_MIN,
                maximum=config.K_MAX,
                value=config.K_DEFAULT,
                step=1,
                label=f"Top-K to display (max {config.K_MAX}) — applies to all pipelines",
                scale=3,
            )
            run_all_btn = gr.Button(
                "🔍 Run ALL pipelines", variant="primary", size="lg", scale=1,
            )

        gr.Markdown("---")
        gr.Markdown(
            "### 3. Results\n"
            "_Click **Run ALL** above to fire every pipeline at once, "
            "or use each section's own Run button to call them independently._"
        )

        # ----- Pipeline 1: full-width section -----
        gr.Markdown(P1_DESCRIPTION)
        p1_run_btn = gr.Button("🔍 Run Pipeline 1", variant="primary", size="lg")
        p1_caption_out = gr.Textbox(
            label="📝 Generated caption (P1's text intermediate)",
            interactive=False,
            lines=2,
        )
        p1_gallery_out = gr.Gallery(
            label="P1 — Top-K retrieved (left = highest cosine similarity)",
            columns=10,
            rows=5,
            height=600,
            object_fit="contain",
            allow_preview=True,
        )
        p1_meta_out = gr.Markdown()

        gr.Markdown("---")

        # ----- Pipeline 2: full-width section -----
        gr.Markdown(P2_DESCRIPTION)
        p2_run_btn = gr.Button("🔍 Run Pipeline 2", variant="primary", size="lg")
        gr.Markdown("_No human-readable intermediate — the query is a 512-d vector._")
        p2_gallery_out = gr.Gallery(
            label="P2 — Top-K retrieved (left = highest cosine similarity)",
            columns=10,
            rows=5,
            height=600,
            object_fit="contain",
            allow_preview=True,
        )
        p2_meta_out = gr.Markdown()

        gr.Markdown("---")

        # ----- P3: text two-tower (cached) -----
        gr.Markdown(TEXT2T_DESCRIPTION)
        text2t_run_btn = gr.Button("🔍 Run Text Two-Tower", variant="primary", size="lg")
        text2t_gallery_out = gr.Gallery(
            label="P3 — Top-K retrieved (left = highest cosine similarity)",
            columns=10, rows=5, height=600, object_fit="contain", allow_preview=True,
        )
        text2t_meta_out = gr.Markdown()

        gr.Markdown("---")

        # ----- P4: audio two-tower (cached) -----
        gr.Markdown(AUDIO2T_DESCRIPTION)
        audio2t_run_btn = gr.Button("🔍 Run Audio Two-Tower", variant="primary", size="lg")
        audio2t_gallery_out = gr.Gallery(
            label="P4 — Top-K retrieved (left = highest cosine similarity)",
            columns=10, rows=5, height=600, object_fit="contain", allow_preview=True,
        )
        audio2t_meta_out = gr.Markdown()

        # ----- Live audio row (only when LIVE_AUDIO=1) -----
        if config.LIVE_AUDIO:
            gr.Markdown("---")
            gr.Markdown(LIVE_DESCRIPTION)
            live_mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙 Record your modification for the active preset's garment "
                      "— press ■ to stop, then Run",
            )
            with gr.Row():
                live_run_btn = gr.Button(
                    "🔍 Run Live Audio Retrieval", variant="primary",
                    size="lg", scale=3,
                )
                live_clear_btn = gr.Button(
                    "🗑 Clear / Re-record", size="lg", scale=1,
                )
            live_gallery_out = gr.Gallery(
                label="Live — Top-K retrieved (left = highest cosine similarity)",
                columns=10, rows=5, height=600, object_fit="contain",
                allow_preview=True,
            )
            live_meta_out = gr.Markdown()

        with gr.Accordion("ℹ️ About this demo", open=False):
            gr.Markdown(ABOUT_MD)

        # ----- Wiring -----
        for pid, btn in preset_btns:
            btn.click(
                fn=lambda pid=pid: on_preset_click(pid),
                inputs=None,
                outputs=[active_preset, candidate_img, mod_text, preset_audio_player,
                         transcript, ground_truth_img, active_label],
            )

        p1_run_btn.click(
            fn=on_run_p1,
            inputs=[active_preset, k_slider],
            outputs=[p1_gallery_out, p1_caption_out, p1_meta_out],
        )
        p2_run_btn.click(
            fn=on_run_p2,
            inputs=[active_preset, k_slider],
            outputs=[p2_gallery_out, p2_meta_out],
        )
        text2t_run_btn.click(
            fn=on_run_text2t,
            inputs=[active_preset, k_slider],
            outputs=[text2t_gallery_out, text2t_meta_out],
        )
        audio2t_run_btn.click(
            fn=on_run_audio2t,
            inputs=[active_preset, k_slider],
            outputs=[audio2t_gallery_out, audio2t_meta_out],
        )
        run_all_btn.click(
            fn=on_run_all,
            inputs=[active_preset, k_slider],
            outputs=[
                p1_gallery_out, p1_caption_out, p1_meta_out,
                p2_gallery_out, p2_meta_out,
                text2t_gallery_out, text2t_meta_out,
                audio2t_gallery_out, audio2t_meta_out,
            ],
        )
        if config.LIVE_AUDIO:
            live_run_btn.click(
                fn=on_run_live,
                inputs=[active_preset, live_mic, k_slider],
                outputs=[live_gallery_out, live_meta_out],
            )
            # Reset the mic + results so the user can record a fresh take.
            live_clear_btn.click(
                fn=lambda: (None, [], "_cleared — record a new clip above_"),
                inputs=None,
                outputs=[live_mic, live_gallery_out, live_meta_out],
            )

    return demo


# ---------------------------------------------------------------------------
# v2 demo — "A + B = C": reference image ➕ modification 🟰 retrieved target
# ---------------------------------------------------------------------------
# Reads the same cached preset_cache.json as v1, so v2 needs no GPU. The unit is
# a *preset* (= candidate image + modification text + spoken clip + cached
# results for all 4 pipelines + ground truth). The whole query→answer reads as
# one left-to-right equation on a single row, so nothing needs scrolling: the
# reference garment, the spoken/typed modification, and the top-1 target sit
# side by side; the rest of the Top-K hangs directly underneath.

_V2_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"
_V2_MODES = {                       # radio label -> preset_cache result key
    "P1 — Caption baseline": "p1",
    "P2 — Direct embedding": "p2",
    "P3 — Text two-tower": "text2t",
    "P4 — Audio two-tower": "audio2t",
}
_V2_DEMO = "🎬 Demo — compare all four"
_V2_MODE_CHOICES = [*_V2_MODES.keys(), _V2_DEMO]
_V2_SHORT = {                       # compact name for Demo-strip captions
    "P1 — Caption baseline": "P1 caption",
    "P2 — Direct embedding": "P2 direct",
    "P3 — Text two-tower": "P3 text two-tower",
    "P4 — Audio two-tower": "P4 audio two-tower",
}

_V2_PIPELINE_EXPLAINER = """\
Every pipeline ends the same way: **pre-compute an embedding index over the
~59k-image gallery (offline), then encode the query and do a cosine
nearest-neighbour search against that index (online).** They differ only in how
the query vector is produced.

### 🅿️1 — Caption baseline
1. **Offline (once):** caption every gallery image, embed each caption with
   Marqo-FashionCLIP, store a **caption-embedding index** (~59k × d).
2. **Online:** Qwen2-VL-7B reads (reference image + text modification) → writes a
   one-sentence target caption.
3. Embed that caption with Marqo-FashionCLIP → **one query vector**.
4. **Cosine similarity search** against the index → Top-K.
&nbsp;·&nbsp; R@10 = 0.533

### 🅿️2 — Direct embedding
1. **Offline:** encode every gallery image with the FashionCLIP image encoder →
   store an **image-embedding index**.
2. **Online:** a fine-tuned Qwen2-VL-7B encodes (reference image + text) → a 512-d
   query vector **directly** (no caption step).
3. **Cosine similarity search** against the index → Top-K.
&nbsp;·&nbsp; R@10 = 0.402

### 🅿️3 — Text two-tower
1. **Offline:** the **target tower** encodes every gallery image → store a
   **target-embedding index** (this tower is trained, not a frozen encoder).
2. **Online:** the **query tower** encodes (reference image + text modification)
   → 512-d query vector, in the same space as the index.
3. **Cosine similarity search** → Top-K.
&nbsp;·&nbsp; R@10 = 0.654 — best text result

### 🅿️4 — Audio two-tower
1. **Offline:** the same target tower encodes every gallery image → target index.
2. **Online:** the query tower takes (reference image + **raw speech**); the audio
   passes through the model's Whisper encoder **with no speech-to-text** → 512-d
   query vector.
3. **Cosine similarity search** → Top-K.
&nbsp;·&nbsp; R@10 = 0.624 / 0.643

---
**🎬 Demo** runs all four query encoders on the **same** reference + modification
and searches each one's index, so you can see which pipeline's Top-1 is the true
target (✓).
"""

_V2_CSS = """
.v2-glyph{font-size:42px;font-weight:800;color:#9aa0a6;display:flex;
  align-items:center;justify-content:center;min-height:260px;}
.v2-cue{font-size:1.05rem;padding:6px 4px;}
.v2-hero .label-wrap{font-weight:700;}
.v2-howto p{font-size:1.15rem;line-height:1.55;}
.v2-explain .label-wrap, .v2-explain .label-wrap span{font-size:1.2rem;font-weight:700;}
/* Reference Image and Modification headings — same prominent style on both
   columns so the content below them lines up at the same vertical position. */
.v2-ref-title h3, .v2-mod-title h3{font-size:1.35rem; margin:2px 0 6px 0;}
/* Modification text box: match the reference image box height so the two
   columns are visually balanced. */
#v2-mod-textbox textarea{min-height:220px !important;}
/* Real, always-clickable upload button (dashed border so it reads as "drop zone"). */
.v2-upload-btn{font-size:1rem !important; padding:12px !important;
  border:2px dashed #4493f8 !important; margin-top:4px !important;}
.v2-upload-btn:hover{background:#eff6fc !important;}
/* Make the empty image area read as a clickable drop zone (gradio's native
   click-to-upload works, but the empty state looks inert without a hover cue). */
#v2-ref-img .empty, #v2-ref-img [class*="empty"]{cursor:pointer;}
#v2-ref-img .empty:hover, #v2-ref-img [class*="empty"]:hover{background:#f6f8fa;}
/* The gr.Group wrappers around the text/audio modification widgets shouldn't
   render as visible gray-bordered boxes — they're only logical containers. */
.v2-naked > div, .v2-naked > .form{border:none !important; padding:0 !important;
  background:transparent !important; box-shadow:none !important;}
"""


def _v2_tag(pid: str) -> str:
    """Circled-number tag for a preset (e.g. '③'), used as the pairing cue."""
    i = PRESET_ORDER.index(pid)
    return _V2_CIRCLED[i] if i < len(_V2_CIRCLED) else f"#{i + 1}"


def _v2_inputs(sel):
    """(reference image path, modification text, audio path) for one example."""
    if not sel or sel not in PRESETS:
        return None, "", None
    p = PRESETS[sel]
    img = str(gallery.image_path(p["candidate_image_id"]))
    w = config.PRESET_AUDIO_DIR / f"{sel}.wav"
    aud = str(w) if w.exists() else None
    return img, p["modification_text"], aud


def _v2_run(sel, mode, k):
    """Compute results for the loaded example, in order (11 values):
      top1_image, top1_badge, topk_items,
      demo_strip_items (the 4-thumb top-1 comparison shown in the equation row),
      demo_p1_items, demo_p2_items, demo_p3_items, demo_p4_items (full Top-K rows),
      gt_image, gt_label, status
    """
    if not sel or sel not in PRESETS:
        return (None, "", [], [], [], [], [], [], None, "_—_",
                "👉 Click an example on the right, then press **▶ Run**.")

    preset = PRESETS[sel]
    tt = preset.get("true_target_id")
    k = int(k)

    def rank_str(rank):
        return f"#{rank}" if rank else "not in top-50"

    # Single-mode top-1 + full Top-K for the selected pipeline.
    r = _result_from_cached(preset[_V2_MODES.get(mode, "audio2t")], tt)
    top1_id = r.target_ids[0] if r.target_ids else None
    top1_img = str(gallery.image_path(top1_id)) if top1_id else None
    hit = top1_id == tt
    top1_badge = (
        f"## {'✅ Correct!' if hit else '❌ Top-1 is not the target'}\n"
        f"true target ranked **{rank_str(r.true_target_rank)}**"
        if top1_id else "_no result_"
    )
    topk_items = _gallery_items(r, k)

    # Demo strip (quick at-a-glance): each pipeline's top-1 in one comparison.
    demo_strip = []
    demo_lists = []
    for label, key in _V2_MODES.items():
        rr = _result_from_cached(preset[key], tt)
        t1 = rr.target_ids[0] if rr.target_ids else None
        mark = "✓ correct" if t1 == tt else "✗"
        demo_strip.append((str(gallery.image_path(t1)),
                           f"{_V2_SHORT[label]}  {mark}  (true {rank_str(rr.true_target_rank)})"))
        # Full Top-K for that pipeline.
        items = []
        for i, t in enumerate(rr.target_ids[:k]):
            star = " ★" if t == tt else ""
            items.append((str(gallery.image_path(t)), f"#{i + 1}  {t}{star}"))
        demo_lists.append(items)

    gt = str(gallery.image_path(tt)) if tt else None
    gt_label = f"✅ **Correct answer:** `{tt}`"
    title = PRESET_NARRATIVES.get(sel, {}).get("title", "")
    if mode == _V2_DEMO:
        status = f"Showing **{title}** — all four models' Top-K compared below."
    else:
        status = f"Showing **{title}** — model **{mode.split(' — ')[0]}**."
    return (top1_img, top1_badge, topk_items, demo_strip,
            demo_lists[0], demo_lists[1], demo_lists[2], demo_lists[3],
            gt, gt_label, status)


def _v2_run_live(sel, audio_path, k):
    """Run the Plan-15 audio two-tower LIVE on the user's recorded/loaded clip
    (P4 mode). Reference garment = the selected example's candidate image. Returns
    the same 10-tuple shape as _v2_run (4 demo rows empty for single-mode view)."""
    from .pipelines.two_tower import run_two_tower_inference
    from .precompute_presets import find_rank, load_candidate_image

    preset = PRESETS[sel]
    tt = preset.get("true_target_id")
    image = load_candidate_image(preset["candidate_image_id"])
    g_emb, g_ids = _AUDIO_GALLERY
    ids, scores, lat = run_two_tower_inference(
        _AUDIO_MODEL, config.LIVE_AUDIO_DEVICE, g_emb, g_ids,
        image, audio_path, k=config.K_MAX,
    )
    k = int(k)
    rank = find_rank(tt, ids)
    rstr = f"#{rank}" if rank else "not in top-50"
    top1_id = ids[0] if ids else None
    top1_img = str(gallery.image_path(top1_id)) if top1_id else None
    hit = top1_id == tt
    badge = (f"## {'✅ Correct!' if hit else '❌ Top-1 is not the target'}\n"
             f"🎙 **LIVE P4 on your audio** · true target ranked **{rstr}** · "
             f"embed {lat['embed_s']:.2f}s")
    topk_items = []
    for i, t in enumerate(ids[:k]):
        mark = " ★" if t == tt else ""
        topk_items.append((str(gallery.image_path(t)),
                           f"#{i + 1}  {t}{mark}  ({scores[i]:.2f})"))
    gt = str(gallery.image_path(tt)) if tt else None
    gt_label = f"✅ **Correct answer:** `{tt}`"
    title = PRESET_NARRATIVES.get(sel, {}).get("title", "")
    status = (f"🎙 Ran **P4 audio two-tower LIVE** on your clip "
              f"(reference garment: {title}).")
    return (top1_img, badge, topk_items, [], [], [], [], [], gt, gt_label, status)


def _v2_visibility(mode):
    """Visibility for the chosen mode, in order:
    (v2_mod_text, v2_aud, single_group, demo_group, topk_group,
     text_prompts_group, voice_clips_group, demo_topk_group).

    Modality-matched: text models (P1/P2/P3) show the typed text + text-prompt
    buttons; the audio model (P4) shows the spoken clip (recordable, run live) +
    voice-clip buttons; 🎬 Demo shows both. Single modes get the single Top-K
    list; Demo gets the four-pipeline Top-K rows. Visibility is toggled on the
    components directly so no empty gray container is left behind.
    """
    is_demo = mode == _V2_DEMO
    is_audio = mode == "P4 — Audio two-tower"
    show_text = (not is_audio) or is_demo
    show_audio = is_audio or is_demo
    single = not is_demo
    return (gr.update(visible=show_text),    # v2_mod_text (left)
            gr.update(visible=show_audio),   # v2_aud (left)
            gr.update(visible=single),       # single_group (result)
            gr.update(visible=is_demo),      # demo_group (top-1 strip in eq row)
            gr.update(visible=single),       # topk_group (single's full Top-K)
            gr.update(visible=show_text),    # text_prompts_group (right)
            gr.update(visible=show_audio),   # voice_clips_group (right)
            gr.update(visible=is_demo))      # demo_topk_group (4 ranked rows)


def build_ui_v2() -> gr.Blocks:
    """v2 demo — the A+B=C equation view, simplified.

    One click on an example (right panel) loads its reference image + modification
    (text & spoken) AND runs the selected model, so the answer appears next to the
    query with no scrolling and no separate "match the number" step. A ▶ Run
    button re-runs; Clear resets. Preset-only (cached, no GPU); uploading your own
    image is accepted but its live retrieval is not wired yet.

    A separate Blocks so v1 (`build_ui`) is never touched; the two are combined
    into switchable tabs by `main()` via gr.TabbedInterface.
    """
    with gr.Blocks(title="Fashion Retrieval — v2", css=_V2_CSS) as demo:
        gr.HTML(f"<style>{_V2_CSS}</style>")  # robust inject under TabbedInterface
        gr.Markdown(
            "# 👗 Find the matching garment\n"
            "**How to use:** click an **example** in the panel on the right — it "
            "instantly loads a **reference image ➕ a modification** (typed for the "
            "text models, spoken for the audio model) and shows the model's "
            "**🟰 matching garment**. Switch models below, or **🎬 Demo** to compare "
            "all four at once. _(In **P4** mode you can **🎙 record your own voice** "
            "and Run it live; uploading your own image is coming soon.)_",
            elem_classes=["v2-howto"],
        )

        sel = gr.State(None)        # the loaded example's preset id

        v2_mode = gr.Radio(
            _V2_MODE_CHOICES, value=_V2_DEMO,
            label="Model (pick one) — or 🎬 Demo to compare all four",
        )

        with gr.Accordion("❓ What does each pipeline (P1–P4) mean?  ▸ click to expand",
                          open=False, elem_classes=["v2-explain"]):
            gr.Markdown(_V2_PIPELINE_EXPLAINER)

        with gr.Row():
            # ===================== LEFT: the equation =====================
            with gr.Column(scale=3):
                # ---- query: REFERENCE + MODIFICATION ----
                with gr.Row(elem_classes=["v2-hero"], equal_height=True):
                    with gr.Column(scale=6):
                        gr.Markdown("### 🖼 Reference Image",
                                    elem_classes=["v2-ref-title"])
                        v2_img = gr.Image(
                            label="", show_label=False,
                            sources=["upload"], type="filepath", height=220,
                            elem_id="v2-ref-img",
                        )
                        v2_upload_btn = gr.UploadButton(
                            "➕  Click to upload your own image",
                            file_types=["image"], type="filepath",
                            elem_classes=["v2-upload-btn"],
                        )
                        with gr.Accordion(
                            "⚠ Preset-only demo warning  ▸ click to expand",
                            open=False,
                        ):
                            gr.Markdown(
                                "**No inference server is deployed for the image-text "
                                "models** (P1 / P2 / P3, plus the image side of P4) — "
                                "so **uploading a custom image** (or typing custom "
                                "text in the modification box) **won't return "
                                "retrieval results here**. The upload path is "
                                "intentionally left in: clone the repo and deploy "
                                "the two-tower / caption pipeline yourself to enable "
                                "it."
                            )
                    with gr.Column(scale=1, min_width=40):
                        gr.HTML("<div class='v2-glyph'>+</div>")
                    with gr.Column(scale=6):
                        gr.Markdown("### 📝 Modification — what should change",
                                    elem_classes=["v2-mod-title"])
                        v2_mod_text = gr.Textbox(
                            label="📝 as text — read by P1 / P2 / P3  "
                                  "(type your own, or pick a prompt on the right)",
                            lines=9, interactive=True, visible=True,
                            elem_id="v2-mod-textbox",
                            placeholder="Type how the garment should change… "
                                        "(custom text needs a deployed inference "
                                        "server — see ⚠ warning under the image)",
                        )
                        v2_aud = gr.Audio(
                            label="🔊 as speech — heard by P4 · ▶ to play, or "
                                  "🎙 record your own (P4 ▶ Run = live retrieval)",
                            sources=["microphone", "upload"], type="filepath",
                            visible=True,
                        )

                # ---- the Run / Clear actions ----
                with gr.Row():
                    v2_run_btn = gr.Button("▶ Run", variant="primary",
                                           size="lg", scale=3)
                    v2_clear_btn = gr.Button("✖ Clear", size="lg", scale=1)

                gr.HTML("<div class='v2-glyph'>=</div>")

                # ---- answer: model result + ground truth, side by side ----
                with gr.Row(equal_height=True):
                    with gr.Column(scale=6):
                        with gr.Group(visible=False) as single_group:
                            v2_top1 = gr.Image(label="🎯 MODEL'S TOP-1", type="filepath",
                                               interactive=False, height=210)
                            v2_top1_badge = gr.Markdown()
                        with gr.Group(visible=True) as demo_group:
                            v2_demo_gallery = gr.Gallery(
                                label="🎯 ALL FOUR models' top-1 (✓ = found the target)",
                                columns=2, rows=2, height=230, object_fit="contain",
                                allow_preview=True)
                    with gr.Column(scale=6):
                        v2_gt_img = gr.Image(label="✅ GROUND TRUTH (correct answer)",
                                             type="filepath", interactive=False,
                                             height=210)
                        v2_gt_label = gr.Markdown("_run to reveal_")

                v2_status = gr.Markdown(
                    "👉 Click a **garment** or a **🔊 voice clip** on the right — it loads and runs instantly.",
                    elem_classes=["v2-cue"],
                )

                # ---- K slider (always visible — used by single Top-K + Demo rows) ----
                v2_k = gr.Slider(config.K_MIN, config.K_MAX,
                                 value=config.K_DEFAULT, step=1,
                                 label=f"Top-K (max {config.K_MAX})")

                # ---- Top-K for the chosen single model ----
                with gr.Group(visible=False) as topk_group:
                    v2_topk = gr.Gallery(label="Full ranked list (left = best match)",
                                         columns=10, rows=2, height=300,
                                         object_fit="contain", allow_preview=True)

                # ---- Demo mode: Top-K stacked per pipeline ----
                with gr.Group(visible=True) as demo_topk_group:
                    gr.Markdown("### 📊 Top-K per pipeline (same query, four rankings)")
                    gr.Markdown("**P1 — Caption baseline**")
                    v2_demo_p1 = gr.Gallery(label="", columns=10, rows=1,
                                            height=120, object_fit="contain",
                                            allow_preview=True)
                    gr.Markdown("**P2 — Direct embedding**")
                    v2_demo_p2 = gr.Gallery(label="", columns=10, rows=1,
                                            height=120, object_fit="contain",
                                            allow_preview=True)
                    gr.Markdown("**P3 — Text two-tower**")
                    v2_demo_p3 = gr.Gallery(label="", columns=10, rows=1,
                                            height=120, object_fit="contain",
                                            allow_preview=True)
                    gr.Markdown("**P4 — Audio two-tower**")
                    v2_demo_p4 = gr.Gallery(label="", columns=10, rows=1,
                                            height=120, object_fit="contain",
                                            allow_preview=True)

            # ================ RIGHT: the examples panel ================
            with gr.Column(scale=1, min_width=240):
                gr.Markdown(
                    "### 📋 Examples\nPick by **garment** (click a photo) — or by "
                    "its **text prompt** / **voice clip** below (the picker matches "
                    "the model you selected). Any click loads the matching example "
                    "and runs instantly."
                )
                v2_examples = gr.Gallery(
                    value=[(str(gallery.image_path(PRESETS[pid]["candidate_image_id"])),
                            f"{i + 1}. {PRESET_NARRATIVES.get(pid, {}).get('title', '')}")
                           for i, pid in enumerate(PRESET_ORDER)],
                    label="🖼 Garments — click a photo", columns=2, rows=4,
                    height=340, allow_preview=False, object_fit="cover",
                )
                # Text-prompt picker (text models P1/P2/P3 + Demo).
                with gr.Group(visible=True) as text_prompts_group:
                    gr.Markdown("**📝 …or pick a text prompt:**")
                    v2_text_btns = [
                        (pid, gr.Button(
                            f"📝 {i + 1}. {PRESETS[pid]['modification_text'][:60]}"
                            + ("…" if len(PRESETS[pid]['modification_text']) > 60 else ""),
                            size="sm"))
                        for i, pid in enumerate(PRESET_ORDER)
                    ]
                # Voice-clip picker (audio model P4 + Demo).
                with gr.Group(visible=True) as voice_clips_group:
                    gr.Markdown("**🔊 …or pick a voice clip:**")
                    v2_voice_btns = [
                        (pid, gr.Button(
                            f"🔊 {i + 1}. {PRESET_NARRATIVES.get(pid, {}).get('title', '')}",
                            size="sm"))
                        for i, pid in enumerate(PRESET_ORDER)
                    ]

        # ----- wiring -----
        run_out = [v2_top1, v2_top1_badge, v2_topk, v2_demo_gallery,
                   v2_demo_p1, v2_demo_p2, v2_demo_p3, v2_demo_p4,
                   v2_gt_img, v2_gt_label, v2_status]
        vis_out = [v2_mod_text, v2_aud, single_group, demo_group,
                   topk_group, text_prompts_group, voice_clips_group,
                   demo_topk_group]

        # Selecting an example (by photo OR by voice clip) loads its reference
        # image + both modification forms AND runs — never a mismatch.
        select_out = [sel, v2_img, v2_mod_text, v2_aud, *run_out]

        def _select(pid, mode, k):
            img, text, aud = _v2_inputs(pid)
            return (pid, img, text, aud, *_v2_run(pid, mode, k))

        def on_example(mode, k, evt: gr.SelectData):
            pid = PRESET_ORDER[evt.index]
            if mode == _V2_DEMO:
                # Demo: full auto-load (image + text + audio) and run all four.
                return _select(pid, mode, k)
            # Single modes: load only the reference image — the user provides the
            # modification (type text, or record audio in P4, or click a prompt /
            # voice clip on the right). Results stay cleared until they Run.
            img, _text, _aud = _v2_inputs(pid)
            if mode == "P4 — Audio two-tower":
                hint = (f"🖼 Reference loaded. **🎙 Record your audio** (or click "
                        f"a 🔊 voice clip on the right), then press **▶ Run**.")
            else:
                hint = (f"🖼 Reference loaded. **Type your modification** in the 📝 "
                        f"box (or click a 📝 prompt on the right), then press "
                        f"**▶ Run**.")
            # 15 outputs = sel + v2_img + v2_mod_text + v2_aud + 11 run_out
            return (pid, img, "", None,
                    None, "", [], [], [], [], [], [],
                    None, "_run to reveal_", hint)

        v2_examples.select(fn=on_example, inputs=[v2_mode, v2_k], outputs=select_out)
        for pid, b in v2_voice_btns:
            b.click(fn=lambda mode, k, pid=pid: _select(pid, mode, k),
                    inputs=[v2_mode, v2_k], outputs=select_out)
        for pid, b in v2_text_btns:
            b.click(fn=lambda mode, k, pid=pid: _select(pid, mode, k),
                    inputs=[v2_mode, v2_k], outputs=select_out)

        # Typing custom text and pressing Enter: honest "needs deploy" message
        # (no inference server for the text pipelines in this preset-only build).
        def on_custom_text():
            msg = ("🛈 Custom text received — but no inference server is deployed "
                   "for the text pipelines (see **⚠ Preset-only demo warning** "
                   "under the image). Click a 📝 prompt on the right to see real "
                   "cached results.")
            return (None, None, None,
                    None, "", [], [], [], [], [], [],
                    None, "_run to reveal_", msg)

        v2_mod_text.submit(
            fn=on_custom_text, inputs=None,
            outputs=[sel, v2_img, v2_aud,
                     v2_top1, v2_top1_badge, v2_topk, v2_demo_gallery,
                     v2_demo_p1, v2_demo_p2, v2_demo_p3, v2_demo_p4,
                     v2_gt_img, v2_gt_label, v2_status],
        )

        # ▶ Run: P4 runs the audio two-tower LIVE on the current clip (your
        # recording or the loaded preset); everything else reads the cache.
        def on_run(cur_sel, mode, k, audio_path):
            if (mode == "P4 — Audio two-tower" and _AUDIO_MODEL is not None
                    and audio_path and cur_sel in PRESETS):
                return _v2_run_live(cur_sel, audio_path, k)
            return _v2_run(cur_sel, mode, k)

        v2_run_btn.click(
            fn=on_run, inputs=[sel, v2_mode, v2_k, v2_aud], outputs=run_out,
        )
        v2_k.release(
            fn=on_run, inputs=[sel, v2_mode, v2_k, v2_aud], outputs=run_out,
        )

        def on_mode(mode, cur_sel, k):
            return (*_v2_visibility(mode), *_v2_run(cur_sel, mode, k))

        v2_mode.change(
            fn=on_mode, inputs=[v2_mode, sel, v2_k],
            outputs=[*vis_out, *run_out],
        )

        def on_clear():
            return (None, None, "", None,
                    None, "", [], [], [], [], [], [],
                    None, "_run to reveal_",
                    "👉 Click a **garment** or a **🔊 voice clip** on the right — it loads and runs instantly.")

        v2_clear_btn.click(
            fn=on_clear, inputs=None,
            outputs=[sel, v2_img, v2_mod_text, v2_aud,
                     v2_top1, v2_top1_badge, v2_topk, v2_demo_gallery,
                     v2_demo_p1, v2_demo_p2, v2_demo_p3, v2_demo_p4,
                     v2_gt_img, v2_gt_label, v2_status],
        )

        # Custom image upload: honest "not wired yet" (preset-only build).
        def on_custom_image():
            msg = ("🛈 Custom image received — but no inference server is deployed "
                   "(see **⚠ Preset-only demo warning** under the image). Click an "
                   "example on the right to see real cached results.")
            # Clear the old preset's modification too, so the custom image isn't
            # shown next to a stale text/audio modification.
            return (None, msg, "", None,
                    None, "", [], [], [], [], [], [],
                    None, "_run to reveal_")

        v2_img.upload(
            fn=on_custom_image, inputs=None,
            outputs=[sel, v2_status, v2_mod_text, v2_aud,
                     v2_top1, v2_top1_badge, v2_topk, v2_demo_gallery,
                     v2_demo_p1, v2_demo_p2, v2_demo_p3, v2_demo_p4,
                     v2_gt_img, v2_gt_label],
        )

        # Real click-to-upload entry point (the gr.Image's native empty area
        # isn't always reliably clickable; this is the guaranteed-clickable one).
        def on_upload(path):
            msg = ("🛈 Custom image received — but no inference server is deployed "
                   "(see **⚠ Preset-only demo warning** under the image). Click "
                   "an example on the right to see real cached results.")
            return (None, path, msg, "", None,
                    None, "", [], [], [], [], [], [],
                    None, "_run to reveal_")

        v2_upload_btn.upload(
            fn=on_upload, inputs=v2_upload_btn,
            outputs=[sel, v2_img, v2_status, v2_mod_text, v2_aud,
                     v2_top1, v2_top1_badge, v2_topk, v2_demo_gallery,
                     v2_demo_p1, v2_demo_p2, v2_demo_p3, v2_demo_p4,
                     v2_gt_img, v2_gt_label],
        )

        # Make the entire empty Reference Image box click-to-upload (gradio's
        # native empty area only treats the small inner icon as a click target,
        # which is why clicking the box itself appeared to do nothing). This JS
        # routes any click on the empty box through the UploadButton's hidden
        # file input, so hovering + clicking the box now opens the file dialog
        # exactly like the dashed button beneath it.
        demo.load(
            fn=lambda: None, inputs=None, outputs=None,
            js="""() => {
  // Document-level capture-phase delegation so gradio's own click handlers
  // can't swallow the event before we intercept. When the user clicks the empty
  // Reference Image box, we fire the UploadButton's file input (which is the
  // SAME path as the visible '➕ Click to upload' button — guaranteed to work).
  if (window.__v2_img_click_wired) return;
  window.__v2_img_click_wired = true;
  document.addEventListener('click', function(e) {
    const imgBox = e.target.closest('#v2-ref-img');
    if (!imgBox) return;
    if (imgBox.querySelector('img')) return;  // image already loaded — leave it
    // Find the UploadButton's hidden file input.
    let fi = null;
    const ub = document.querySelector('.v2-upload-btn');
    if (ub) {
      fi = ub.querySelector('input[type=file]');
      if (!fi && ub.parentElement) fi = ub.parentElement.querySelector('input[type=file]');
    }
    if (!fi) {
      // last-ditch: any file input on the page
      const all = document.querySelectorAll('input[type=file]');
      for (const inp of all) {
        if (inp.closest('.v2-upload-btn')) { fi = inp; break; }
      }
      if (!fi && all.length) fi = all[0];
    }
    if (fi) {
      e.preventDefault();
      e.stopPropagation();
      fi.click();
    }
  }, true);
}"""
        )

    return demo


def main() -> None:
    # Load the audio two-tower before launch so the first live click is fast
    # (and so a missing checkpoint fails loudly at startup, not mid-demo).
    if config.LIVE_AUDIO:
        load_audio_tower()

    # v1 and v2 are independent Blocks; TabbedInterface presents them as two
    # switchable pages. v1 is preserved verbatim — it remains the fallback until
    # v2 fully replaces it.
    # v2 is the default tab (shown first on load); v1 stays as the stable backup.
    app = gr.TabbedInterface(
        [build_ui_v2(), build_ui()],
        ["✨ v2 (new)", "📦 v1 (stable)"],
        title="👗 Fashion Retrieval",
    )

    # allowed_paths: gradio 6.x sandboxes file serving to cwd + /tmp by default.
    # We need to whitelist the gallery + preset thumbs + preset audio so the
    # Image / Gallery / Audio components can serve files from those paths.
    allowed = [
        str(config.GALLERY_DIR),
        str(config.PRESET_THUMBS_DIR),
        str(config.PRESET_CACHE_JSON.parent),
        str(config.PRESET_AUDIO_DIR),
    ]
    app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        allowed_paths=allowed,
    )


if __name__ == "__main__":
    main()
