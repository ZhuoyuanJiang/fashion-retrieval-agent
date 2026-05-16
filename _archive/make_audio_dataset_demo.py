"""ARCHIVED, non-core — regenerator for notebooks/audio_dataset_demo.ipynb.

This is NOT part of the active pipeline. It is kept only so the audio-dataset
spot-check notebook can be regenerated if needed.

It builds a human-inspection DEMO of the **already-synthesized** audio dataset.
It does NOT synthesize audio — the audio dataset is produced by
`src/data/build_tts_audio.py` (see `Documentation/Progress_14_20260515.md`).

The committed notebook `notebooks/audio_dataset_demo.ipynb` is fully
self-contained (pure markdown cells with base64-embedded images/audio/text) and
does **not** reference this script — so this file is purely a regeneration
convenience, not a dependency of the notebook.

What it does: samples 15 triplets from the Plan-14 audio manifest, transcribes
each clip with faster-whisper, and writes the notebook.

Run (needs faster-whisper + jiwer, and the audio data on local SSD):
    <faster-whisper-env>/bin/python _archive/make_audio_dataset_demo.py
"""
import base64
import html
import json
import random
import re
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer

REPO = Path(__file__).resolve().parents[1]
PLAN14 = Path("/tmp3/zhuoyuan/plan14_audio")
TRIPLETS = (REPO / "data_exploration/datasets/facap-repo/data/facap/"
            "cir_triplets/dress_train_triplets.json")
IMG = Path("/ssd1/zhuoyuan/facap-images")
OUT = REPO / "notebooks" / "audio_dataset_demo.ipynb"

N_IN_DIST = 12   # distinct speakers, split across train/dev/headline
N_OOD = 3        # distinct held-out speakers

triplets = json.load(open(TRIPLETS))
manifest = json.load(open(PLAN14 / "manifest.json"))


def iid(p):
    return p.rsplit("/", 1)[-1].removesuffix(".jpeg")


def norm(t):
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


# ---- sample a varied set -------------------------------------------------
random.seed(15)
indist, ood = manifest["in_dist"], manifest["ood"]
in_keys = list(indist)
random.shuffle(in_keys)

picked = []          # (key, condition, record)
seen = set()
for want in ["dev", "dev", "headline", "headline"] + ["train"] * (N_IN_DIST - 4):
    for k in in_keys:
        r = indist[k]
        if r["split"] == want and r["speaker"] not in seen and k not in (p[0] for p in picked):
            picked.append((k, "in_dist", r))
            seen.add(r["speaker"])
            break

ood_keys = list(ood)
random.shuffle(ood_keys)
seen_ood = set()
for k in ood_keys:
    r = ood[k]
    if r["speaker"] not in seen_ood:
        picked.append((k, "ood", r))
        seen_ood.add(r["speaker"])
    if len(seen_ood) >= N_OOD:
        break

print(f"sampled {len(picked)} clips; transcribing with faster-whisper medium.en...",
      flush=True)
asr = WhisperModel("medium.en", device="cpu", compute_type="int8")

rows = []
for k, cond, rec in picked:
    idx = int(k)
    caption = triplets[idx]["captions"][0]
    segs, _ = asr.transcribe(rec["wav"], language="en", beam_size=5)
    hyp = " ".join(s.text for s in segs).strip()
    w = wer(norm(caption), norm(hyp))
    rows.append({
        "idx": idx, "cond": cond, "split": rec["split"],
        "speaker": rec["speaker"], "gender": rec["gender"], "accent": rec["accent"],
        "caption": caption, "hyp": hyp, "wer": w,
        "cand": triplets[idx]["candidate"], "tgt": triplets[idx]["target"],
        "wav": rec["wav"],
    })
    print(f"  idx {idx:6d} [{cond}] {rec['speaker']} WER={w*100:.1f}%", flush=True)

avg_wer = sum(r["wer"] for r in rows) / len(rows)
n = len(rows)
n_in = sum(r["cond"] == "in_dist" for r in rows)
n_ood = n - n_in


# ---- build the notebook --------------------------------------------------
def md_cell(cid, source):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}


cells = [md_cell("intro", f"""# Audio dataset — spot-check

Synthesized **modification audio** for the FACap dress-slice retrieval triplets
(Plan 14). Each training triplet is *(candidate image, modification, target
image)*; this dataset turns the text-only modification into **speech**, so the
retrieval model can take a spoken modification as its query input.

The full dataset is **56,686 clips** — 55,186 in-distribution (one per used
triplet, voiced by a 100-speaker training pool) + 1,500 OOD-voice (dev/headline
re-synthesized with 10 held-out speakers). Each clip is a different VCTK
speaker, zero-shot voice-cloned by Chatterbox; 16 kHz mono.

Below are **{n} sampled clips** ({n_in} in-distribution across distinct
training-pool speakers + {n_ood} OOD-voice). For each: the candidate image, the
spoken modification, the target image, the source caption, and **what
faster-whisper transcribes back from the audio**.

**Average WER over this sample: {avg_wer * 100:.1f}%** — transcript vs source
caption; low means the words survived synthesis intact.

> The embedded audio players below play in **Jupyter / nbviewer / when this
> notebook is opened locally**. GitHub's inline notebook viewer may not render
> the `<audio>` element — but the candidate/target images, the source caption,
> and the Whisper transcript render everywhere, so the verification (is the
> audio faithful to the caption?) is fully visible regardless of viewer.

---
*This notebook is a fixed snapshot for inspection — it does not need to be run.
It was generated by sampling the Plan-14 audio manifest and transcribing each
clip with faster-whisper. The audio dataset itself is produced by
`src/data/build_tts_audio.py` — see `Documentation/Progress_14_20260515.md` for
how it was built. Regeneration script: `_archive/make_audio_dataset_demo.py`
(archived, non-core).*""")]

tbl = ["## Sample summary", "",
       "| Triplet | Condition | Split | Speaker | Gender | Accent | WER |",
       "|---|---|---|---|---|---|---|"]
for r in rows:
    tbl.append(f"| {r['idx']} | {r['cond']} | {r['split']} | {r['speaker']} "
               f"| {r['gender']} | {r['accent']} | {r['wer'] * 100:.1f}% |")
cells.append(md_cell("summary", "\n".join(tbl)))
cells.append(md_cell("clips-header", "## The clips"))

for i, r in enumerate(rows):
    cand = b64(IMG / (iid(r["cand"]) + ".jpeg"), "image/jpeg")
    tgt = b64(IMG / (iid(r["tgt"]) + ".jpeg"), "image/jpeg")
    aud = b64(r["wav"], "audio/wav")
    cond = "OOD voice" if r["cond"] == "ood" else "in-distribution voice"
    card = (
        f'<h3>Triplet {r["idx"]} &mdash; {cond}</h3>\n'
        f'<p><b>Voice:</b> VCTK {r["speaker"]} ({r["gender"]}, '
        f'{html.escape(r["accent"])} accent) &nbsp;&middot;&nbsp; '
        f'<b>split:</b> {r["split"]} &nbsp;&middot;&nbsp; '
        f'<b>WER:</b> {r["wer"] * 100:.1f}%</p>\n'
        f'<table><tr>\n'
        f'<td align="center"><img src="{cand}" height="180"><br><sub>candidate</sub></td>\n'
        f'<td>&nbsp;&nbsp;+ spoken modification &rarr;&nbsp;&nbsp;</td>\n'
        f'<td align="center"><img src="{tgt}" height="180"><br><sub>target</sub></td>\n'
        f'</tr></table>\n'
        f'<p><b>Modification (source caption):</b> {html.escape(r["caption"])}</p>\n'
        f'<audio controls src="{aud}"></audio>\n'
        f'<p><b>Whisper transcribed from the audio:</b> '
        f'<i>{html.escape(r["hyp"])}</i></p>'
    )
    cells.append(md_cell(f"clip-{i}", card))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(cells)} cells)")
print(f"avg WER over {n} sampled clips: {avg_wer * 100:.1f}%")
