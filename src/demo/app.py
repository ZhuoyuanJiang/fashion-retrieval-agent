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


def main() -> None:
    # Load the audio two-tower before launch so the first live click is fast
    # (and so a missing checkpoint fails loudly at startup, not mid-demo).
    if config.LIVE_AUDIO:
        load_audio_tower()

    demo = build_ui()
    # allowed_paths: gradio 6.x sandboxes file serving to cwd + /tmp by default.
    # We need to whitelist the gallery + preset thumbs + preset audio so the
    # Image / Gallery / Audio components can serve files from those paths.
    allowed = [
        str(config.GALLERY_DIR),
        str(config.PRESET_THUMBS_DIR),
        str(config.PRESET_CACHE_JSON.parent),
        str(config.PRESET_AUDIO_DIR),
    ]
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        allowed_paths=allowed,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
