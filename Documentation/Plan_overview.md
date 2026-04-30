# Plan Overview — full research roadmap

A bird's-eye view tying `project_proposal_20260427.md` to the per-plan
execution files. Plan files are written *just-in-time* (one at a time,
revised as prior plans complete). This overview is the long view.

Last refreshed: 2026-04-27.

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

Each phase is broken into multiple plan files. **You are currently at the
end of Phase A's local scaffolding stage.**

---

## Plan-by-plan map

| # | Plan (working title) | Phase | Scope (one line) | Status |
|---|---|---|---|---|
| 1 | `Plan_1_20260420 — Dataset exploration` | A | FashionIQ + FACap viability check; inspection notebook | ✅ done |
| 2 | `Plan_2_20260427 — Local baseline scaffolding` | A | Build the text-modification baseline pipeline locally with mock+oracle VLM (FACap dataset class, text encoder, caption DB, retrieve, eval, orchestrator). Real VLM stub ready to drop in on server. | 🔄 M1–M3 done, M4 pending |
| 3 | `Plan_3 — Server baseline run + report` | A | `git pull` on GPU box, swap mock for real `speechQwen2VL` (and optionally vanilla `Qwen2-VL`) inference, run on the FACap dress evaluation slice, write `Baseline_1_Report` — **the first real number** | ⏳ next |
| 4 | `Plan_4 — Contrastive training v1 (single GPU)` | B | Custom PyTorch training loop, InfoNCE loss, query/target embedding extraction, sanity check on small data | future |
| 5 | `Plan_5 — Contrastive training v2` | B | Compare last-token vs projection-layer embedding; real training run; validation Recall@K | future |
| 6 | `Plan_6 — Multi-GPU scaling` | B | HuggingFace Accelerate, global contrastive loss with `all_gather`, larger effective batch | future |
| 7 | `Plan_7 — Audio extension` | C | TTS pipeline for FACap modifications, wire `speechQwen2VL` audio path, compare text/ASR/native-audio retrieval | future, lowest priority per mentor |
| 8 | `Plan_8 — Cross-dataset eval + real speech` | C | Add FashionIQ + enhFashionIQ to dataloader, cross-dataset retrieval, small recorded-speech test set | future |
| 9 | `Plan_9 — Demo + final polish` | C | Catalog UI, live retrieval demo, polished README + portfolio writeup | future, optional |

**Confidence:**

- Plans 1–3 are firm.
- Plans 4–6 reshape based on what Plan 3's baseline metrics tell us
  (strong baseline → contrastive less urgent; weak baseline →
  contrastive critical — see proposal Part 4.3).
- Plans 7–9 are stretch goals for the portfolio-facing build.
- Expect ~2–3 ad-hoc plans (1–2 milestones each) to slot in for
  unexpected detours like server setup gotchas or dataset bugs.

---

## What's been done — current status (2026-04-27)

### Phase A — baseline (Option 2)

- ✅ **Plan 1 — dataset exploration** complete. FashionIQ and FACap
  both confirmed usable. Notebook + small sample renders committed.
- 🔄 **Plan 2 — local baseline scaffolding** in progress. Status:
  - ✅ M1: workspace + FACap `Dataset` class
  - ✅ M2: text encoder + caption DB build
  - ✅ M3: VLM captioners (Mock + Oracle running locally; Qwen2VL +
    SpeechQwen2VL stubbed for server) + retrieve + eval + orchestrator
    + image prefetch
  - ⏸ M4: README narrative + `docs/server_handoff.md` + commits/push
  - ✅ Codex cross-review converged ("no findings", round 3)
- ⏳ **Plan 3 — server baseline run** not yet started. This is the
  step where we **actually get a baseline number** — the meeting memo's
  "Step 5: Report baseline metrics" / "first serious checkpoint".

### Phase B — contrastive (Option 3)

- Not started. Blocked on Plan 3's metrics.

### Phase C — audio + polish

- Not started. Plan 7 explicitly the lowest priority per mentor's
  guidance (text first, then look at metrics, then decide).

---

## What "done" looks like at each phase boundary

- **End of Phase A** (after Plan 3): we have the meeting memo's
  "first serious checkpoint" — Recall@1/5/10/50 + median rank + 5
  qualitative success and 5 failure cases on the FACap dress eval
  slice, written up as `Baseline_1_Report_<YYYYMMDD>.md`. This is the
  go/no-go decision point for Phase B.
- **End of Phase B** (after Plan 6): Option 3 (contrastive) has been
  trained and evaluated. We have head-to-head numbers vs the Phase A
  baseline.
- **End of Phase C** (after Plan 9): real audio path works, demo is
  shareable, portfolio writeup is done.

---

## Where the *meeting memo* steps land

The mentor's TODO list (`Documentation/meeting_memo_20260420.md`,
"What You Should Do First This Week" — steps 1–10) maps to plans like so:

| Memo step | Action | Lands in |
|---|---|---|
| 1 | Pin the project direction (Option 2 first, audio later) | done in proposal |
| 2 | Build target-caption embedding DB | Plan 2 (M2) ✅ |
| 3 | Run VLM caption-generation baseline | Plan 3 ⏳ |
| 4 | Run text-to-text retrieval | Plan 2 (M3) ✅ (framework) / Plan 3 (real run) ⏳ |
| 5 | Report baseline metrics ("first serious checkpoint") | Plan 3 ⏳ — `Baseline_1_Report` |
| 6 | Decide whether to train contrastive model | Plan 3 outcome → blocks Plan 4 |
| 7 | Implement one-GPU contrastive training loop | Plan 4 |
| 8 | Try embedding design choices (last-token vs projection layer) | Plan 5 |
| 9 | Multi-GPU scaling | Plan 6 |
| 10 | Audio extension | Plan 7 |

Steps 1–7 of the memo are within Phase A scoping; the rest are in B/C.

---

## Open / yet-to-decide

- The **GPU host** for Plan 3+. Two options the user has:
  - Local server (RTX A6000, 48 GB) — already set up, used for speechQwen2VL.
  - Google Colab (L4 / A100) with credits — useful as a quick smoke
    if the server is busy or for a portable workflow.

  Both can run `python -m src.baseline.run_baseline --vlm speechqwen2vl --n-eval 1000`.
  Plan 3 just needs to specify which one is the canonical host for the
  reported numbers.

- **Whether to also support FashionIQ + enhFashionIQ** in the dataset
  class before Plan 3 (currently scoped to FACap dress only). Probably
  defer to Plan 8.

- **Contrastive loss design** (Plans 4–6) — currently a black box;
  fill in once Plan 3 numbers exist.
