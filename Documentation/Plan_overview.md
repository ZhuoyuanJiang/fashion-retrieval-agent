# Plan Overview — full research roadmap

A bird's-eye view tying `project_proposal_20260427.md` to the per-plan
execution files. Plan files are written *just-in-time* (one at a time,
revised as prior plans complete). This overview is the long view.

Last refreshed: 2026-05-01.

---

## The proposal in one minute

From `project_proposal_20260427.md`:

- **Goal**: an audio-conditioned composed fashion retrieval system —
  reference image + spoken modification → retrieve the matching item.
- **Three method options** considered:
  - **Option 1** (spoken garment grounding + tool/API call) — **ruled out**.
  - **Option 2** (caption-generation retrieval) — **main method**.
    The fine-tuned audio-VLM produces a *target-oriented caption*; a
    separate text encoder + retrieval module finds the matching item.
  - **Option 3** (direct contrastive embedding retrieval) — **research
    extension**. The audio-VLM produces a *retrieval embedding directly*,
    trained with contrastive loss against target embeddings.
- **Order**: Option 2 first (stable, easier to debug, gets a working
  demo). Option 3 attempted only after Option 2 has a working baseline.

So the entire project naturally splits into three phases:

| Phase | What it covers |
|---|---|
| **A. Baseline (Option 2)** | Text-modification retrieval baseline — get a number, decide if Option 3 is worth chasing |
| **B. Contrastive (Option 3)** | Direct contrastive training of the audio-VLM for retrieval |
| **C. Audio + polish** | Add the audio modality to whichever option won, plus demo polish |

Each phase is broken into multiple plan files. **Phase A's main run is
done (Plan_3 + encoder ablation, 2026-04-30); a caption-quality
follow-up (Plan_4) and the first contrastive design (Plan_5) are in
parallel.**

---

## Plan-by-plan map

| # | Plan (working title) | Phase | Scope (one line) | Status |
|---|---|---|---|---|
| 1 | `Plan_1_20260420 — Dataset exploration` | A | FashionIQ + FACap viability check; inspection notebook | ✅ done |
| 2 | `Plan_2_20260427 — Local baseline scaffolding` | A | Build the text-modification baseline pipeline locally with mock+oracle VLM (FACap dataset class, text encoder, caption DB, retrieve, eval, orchestrator). Real VLM stub ready to drop in on server. | ✅ done |
| 3 | `Plan_3_20260430 — Server baseline run + encoder ablation` | A | `git pull` on GPU box, swap mock for real `speechQwen2VL` inference, full FACap dress run, encoder ablation (11 encoders) — **first real numbers** | ✅ done — R@1 anchor 0.084 (MiniLM); 0.258 best (Marqo FashionCLIP) |
| 4 | `Plan_4_<TBD> — Caption-quality ablation (VLM swap)` | A→B | Hold encoder fixed (Marqo FashionCLIP), swap the VLM (vanilla Qwen2-VL, GPT-4V, etc.) — isolate caption-quality vs encoder-quality contributions to the remaining gap | 🔄 in progress (forked chat, 2026-05-01) |
| 5 | **`Plan_5_20260501 — Contrastive training v1 (single GPU)`** | B | Custom PyTorch loop, symmetric InfoNCE on Qwen2-VL query vs frozen Marqo-FashionCLIP-image target, single-GPU sanity check | 🔄 design (this chat); execution pending discussion |
| 6 | `Plan_6 — Contrastive training v2` | B | Loss / embedding-source / target-tower ablations (distillation alternative, mean-pool, alt target encoders, hard-negative mining) | future |
| 7 | `Plan_7 — Multi-GPU scaling` | B | HuggingFace Accelerate, global contrastive loss with `all_gather`, larger effective batch | future |
| 8 | `Plan_8 — Audio extension` | C | TTS pipeline for FACap modifications, wire `speechQwen2VL` audio path, compare text/ASR/native-audio retrieval | future, lowest priority per mentor |
| 9 | `Plan_9 — Cross-dataset eval + real speech` | C | Add FashionIQ + enhFashionIQ to dataloader, cross-dataset retrieval, small recorded-speech test set | future |
| 10 | `Plan_10 — Demo + final polish` | C | Catalog UI, live retrieval demo, polished README + portfolio writeup | future, optional |

**Confidence:**

- Plans 1–3 are firm and complete.
- Plan 4 (caption-quality ablation) was inserted on 2026-05-01 after
  the Plan_3 encoder ablation showed the original "weak baseline"
  framing was partly encoder-bound. Cheap to run, decisive — informs
  whether contrastive (Plan_5+) needs to chase caption quality too.
- Plans 5–7 reshape based on Plan_5's headline number vs. the Phase-A
  best-encoder bar (R@1 = 0.258 / R@10 = 0.533).
- Plans 8–10 are stretch goals for the portfolio-facing build.
- Expect ~2–3 ad-hoc plans (1–2 milestones each) to slot in for
  unexpected detours like server setup gotchas or dataset bugs.

---

## What's been done — current status (2026-05-01)

### Phase A — baseline (Option 2)

- ✅ **Plan 1 — dataset exploration** complete. FashionIQ and FACap
  both confirmed usable. Notebook + small sample renders committed.
- ✅ **Plan 2 — local baseline scaffolding** complete (M1–M3 +
  M4 cleanup folded into Plan_3 commits).
- ✅ **Plan 3 — server baseline run + encoder ablation** complete.
  Headline numbers on the 1000-query / ~58 k-target FACap dress slice:
  - MiniLM-L6 anchor: R@1 = 0.084, R@10 = 0.240
  - Marqo FashionCLIP (best of 11 swapped encoders): **R@1 = 0.258,
    R@10 = 0.533** — the new "strong" reference for Phase B.
  - Full encoder table at `Documentation/encoder_swap_table.md`.
  - Read: the original baseline was partly encoder-bound, not just
    VLM-bound. Plan 5 must beat 0.258 / 0.533 to be worth the
    contrastive bet.
- 🔄 **Plan 4 — caption-quality ablation** in progress (forked chat,
  2026-05-01). Holds the encoder fixed at Marqo FashionCLIP and swaps
  the VLM. Cheap; informs Plan_5 design.

### Phase B — contrastive (Option 3)

- 🔄 **Plan 5 — contrastive v1 (single GPU)** in design (this chat,
  forked on 2026-05-01). See `Documentation/Plan_5_20260501.md`.
  Implementation paused for user discussion.

### Phase C — audio + polish

- Not started. Plan 8 explicitly the lowest priority per mentor's
  guidance (text first, then look at metrics, then decide).

---

## What "done" looks like at each phase boundary

- **End of Phase A** (Plan 3 + Plan 4): we have the meeting memo's
  "first serious checkpoint" — Recall@1/5/10/50 + median rank on the
  FACap dress eval slice, the encoder-ablation table, and the
  caption-quality-ablation results that decompose the remaining error.
  Plan_3 already wrote `Baseline_1_Report` (forthcoming after Plan_4
  finalizes); this is the go/no-go decision point for Phase B.
- **End of Phase B** (after Plan 7): Option 3 (contrastive) has been
  trained, scaled, and evaluated. We have head-to-head numbers vs the
  Phase A baseline.
- **End of Phase C** (after Plan 10): real audio path works, demo is
  shareable, portfolio writeup is done.

---

## Where the *meeting memo* steps land

The mentor's TODO list (`Documentation/meeting_memo_20260420.md`,
"What You Should Do First This Week" — steps 1–10) maps to plans like so:

| Memo step | Action | Lands in |
|---|---|---|
| 1 | Pin the project direction (Option 2 first, audio later) | done in proposal |
| 2 | Build target-caption embedding DB | Plan 2 (M2) ✅ |
| 3 | Run VLM caption-generation baseline | Plan 3 ✅ |
| 4 | Run text-to-text retrieval | Plan 2 (M3) ✅ (framework) / Plan 3 ✅ (real run) |
| 5 | Report baseline metrics ("first serious checkpoint") | Plan 3 ✅ — full encoder table; `Baseline_1_Report` finalizes after Plan 4 |
| 6 | Decide whether to train contrastive model | Plan 3 result (encoder-bound) + Plan 4 outcome → unblocks Plan 5 |
| 7 | Implement one-GPU contrastive training loop | Plan 5 |
| 8 | Try embedding design choices (last-token vs projection layer) | Plan 6 |
| 9 | Multi-GPU scaling | Plan 7 |
| 10 | Audio extension | Plan 8 |

Steps 1–7 of the memo are within Phase A/B-design scoping; the rest are in B/C.

---

## Open / yet-to-decide

- **Whether to also support FashionIQ + enhFashionIQ** in the dataset
  class before Phase B is fully done (currently scoped to FACap dress
  only). Probably defer to Plan 9.

- **Contrastive loss design open questions** (Plan 5 + Plan 6) — see
  `Documentation/Plan_5_20260501.md` "Open questions for the user"
  and "Open questions deferred to Plan_6". Includes target-tower
  freeze vs joint training, eval-slice reuse vs fresh holdout, logging
  backend.
