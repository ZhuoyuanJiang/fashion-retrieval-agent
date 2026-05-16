"""Plan 14 M2 — TTS synthesis: FACap dress modifications -> speech (Chatterbox).

Turns the text-only FACap dress-slice modification strings into spoken audio,
the training input for the audio-native query tower. Voice diversity comes from
a VCTK speaker reference bank (Chatterbox is a zero-shot voice clone — the
reference clip *is* the voice; see Progress_14 §7).

Three modes:

  python -m src.data.build_tts_audio bank
      One-off. Builds the VCTK reference bank (~110 speakers, gender-balanced,
      split into a ~100-speaker training pool + ~10 held-out OOD speakers) and
      the deterministic synthesis plan: every *used* dress-train triplet (train
      + dev + headline, mirroring FacapContrastiveDataset) is assigned a
      training-pool speaker and lightly-jittered Chatterbox knobs; dev+headline
      triplets additionally get a held-out speaker for the separate OOD eval.
      Writes ref_bank/ + bank.json + synth_plan.json. Run in an env with
      torchaudio + numpy (e.g. fashion_retrieval); needs VCTK extracted.

  python -m src.data.build_tts_audio synth --shard I --num-shards N
      Synthesizes shard I of the plan with Chatterbox. Resumable (skips wavs
      that already exist). Run N copies in parallel, one per GPU, in the
      Chatterbox conda env. Writes audio/<idx>.wav + audio_ood/<idx>.wav.

  python -m src.data.build_tts_audio manifest
      After all shards finish: verifies every planned wav exists and collates
      manifest.json (triplet index + condition -> wav, speaker, split).

Layout under $PLAN14_AUDIO (default /tmp3/zhuoyuan/plan14_audio):
  vctk/                 extracted VCTK-Corpus-0.92
  ref_bank/<spk>.wav    one 24 kHz mono reference clip per speaker
  bank.json             speaker -> {gender, accent, pool, ref_wav}
  synth_plan.json       list of synthesis jobs
  audio/<idx>.wav       in-distribution synthesis (16 kHz mono, one per triplet)
  audio_ood/<idx>.wav   OOD-voice synthesis (dev + headline only)
  manifest.json         final clip manifest for the dataset + both evals
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SEED = 14  # Plan 14
HEADLINE_SLICE = 1000
DEV_SLICE = 500
N_OOD_SPEAKERS = 10          # held out for the separate OOD-voice eval
REF_DUR_TARGET = 7.0         # seconds — pick each speaker's clip closest to this
REF_DUR_RANGE = (4.0, 12.0)
REF_SR = 24000               # reference-bank sample rate (Chatterbox-native)
OUT_SR = 16000               # synthesized-audio sample rate (Whisper-standard)

PLAN14_AUDIO = Path(os.environ.get("PLAN14_AUDIO", "/tmp3/zhuoyuan/plan14_audio"))
REPO_ROOT = Path(__file__).resolve().parents[2]
FACAP_TRIPLETS = Path(os.environ.get(
    "FACAP_TRIPLETS",
    REPO_ROOT / "data_exploration" / "datasets" / "facap-repo"
    / "data" / "facap" / "cir_triplets" / "dress_train_triplets.json",
))

REF_BANK = PLAN14_AUDIO / "ref_bank"
BANK_JSON = PLAN14_AUDIO / "bank.json"
PLAN_JSON = PLAN14_AUDIO / "synth_plan.json"
AUDIO_DIR = PLAN14_AUDIO / "audio"
AUDIO_OOD_DIR = PLAN14_AUDIO / "audio_ood"
MANIFEST_JSON = PLAN14_AUDIO / "manifest.json"


def _img_id(facap_path: str) -> str:
    """`f200k_images/.../51727804_0.jpeg` -> `51727804_0`."""
    return facap_path.rsplit("/", 1)[-1].removesuffix(".jpeg")


def _facap_split(triplets: list) -> dict:
    """Replicate FacapContrastiveDataset's deterministic split (seed 42).

    Returns {idx -> 'train'|'dev'|'headline'} for every *used* triplet —
    L2-filtered triplets are simply absent (they are never looked up).
    """
    import numpy as np

    N = len(triplets)
    tid = [_img_id(t["target"]) for t in triplets]
    cid = [_img_id(t["candidate"]) for t in triplets]

    headline_idx = list(range(N - HEADLINE_SLICE, N))
    headline_ids = set()
    for i in headline_idx:
        headline_ids.add(tid[i])
        headline_ids.add(cid[i])

    clean = [i for i in range(N - HEADLINE_SLICE)
             if tid[i] not in headline_ids and cid[i] not in headline_ids]
    perm = np.random.RandomState(42).permutation(len(clean))
    dev_pool_pos = set(perm[:DEV_SLICE].tolist())
    dev_idx = [clean[perm[k]] for k in range(DEV_SLICE)]
    dev_ids = set()
    for i in dev_idx:
        dev_ids.add(tid[i])
        dev_ids.add(cid[i])

    train_idx = [clean[p] for p in range(len(clean))
                 if p not in dev_pool_pos
                 and tid[clean[p]] not in dev_ids and cid[clean[p]] not in dev_ids]

    split = {}
    for i in train_idx:
        split[i] = "train"
    for i in dev_idx:
        split[i] = "dev"
    for i in headline_idx:
        split[i] = "headline"
    return split


# ---------------------------------------------------------------------------
# bank mode
# ---------------------------------------------------------------------------
def _find_vctk_root() -> Path:
    """Locate the extracted VCTK dir containing wav48_silence_trimmed/."""
    base = PLAN14_AUDIO / "vctk"
    for cand in [base, *sorted(base.glob("*"))]:
        if cand.is_dir() and (cand / "wav48_silence_trimmed").is_dir():
            return cand
    raise FileNotFoundError(
        f"VCTK not found under {base} (need wav48_silence_trimmed/). "
        f"Extract VCTK-Corpus-0.92.zip there first."
    )


def _parse_speaker_info(path: Path) -> dict:
    """speaker-info.txt -> {speaker_id: {'gender':..., 'accent':...}}."""
    info = {}
    for line in path.read_text().splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) < 4:
            continue
        sid, _age, gender, accent = parts[0], parts[1], parts[2], parts[3]
        info[sid] = {"gender": gender.upper(), "accent": accent}
    return info


def build_bank() -> None:
    import torchaudio
    import torchaudio.functional as AF

    vctk = _find_vctk_root()
    wav_root = vctk / "wav48_silence_trimmed"
    speakers = sorted(d.name for d in wav_root.iterdir() if d.is_dir())
    print(f"VCTK: {len(speakers)} speaker dirs under {wav_root}", flush=True)

    info_path = next(vctk.rglob("speaker-info.txt"))
    spk_info = _parse_speaker_info(info_path)

    REF_BANK.mkdir(parents=True, exist_ok=True)
    bank = {}
    for n, sid in enumerate(speakers):
        clips = sorted((wav_root / sid).glob(f"{sid}_*_mic1.flac"))
        if not clips:
            clips = sorted((wav_root / sid).glob(f"{sid}_*.flac"))
        if not clips:
            print(f"  WARN {sid}: no clips, skipped", flush=True)
            continue
        # pick the clip whose duration is closest to REF_DUR_TARGET, within range
        best, best_score = None, None
        for c in clips:
            meta = torchaudio.info(str(c))
            dur = meta.num_frames / meta.sample_rate
            if not (REF_DUR_RANGE[0] <= dur <= REF_DUR_RANGE[1]):
                continue
            score = abs(dur - REF_DUR_TARGET)
            if best_score is None or score < best_score:
                best, best_score = c, score
        if best is None:  # no clip in range — take the longest available
            best = max(clips, key=lambda c: torchaudio.info(str(c)).num_frames)

        wav, sr = torchaudio.load(str(best))
        wav = wav.mean(dim=0, keepdim=True)               # -> mono
        if sr != REF_SR:
            wav = AF.resample(wav, sr, REF_SR)
        out = REF_BANK / f"{sid}.wav"
        torchaudio.save(str(out), wav, REF_SR)
        meta = spk_info.get(sid, {"gender": "?", "accent": "?"})
        bank[sid] = {"gender": meta["gender"], "accent": meta["accent"],
                     "ref_wav": str(out), "src_clip": best.name}
        if (n + 1) % 20 == 0:
            print(f"  bank {n + 1}/{len(speakers)}", flush=True)

    # gender-balanced OOD split: shuffle, take 5 M + 5 F as held-out
    rng = random.Random(SEED)
    males = [s for s in bank if bank[s]["gender"] == "M"]
    females = [s for s in bank if bank[s]["gender"] == "F"]
    rng.shuffle(males)
    rng.shuffle(females)
    half = N_OOD_SPEAKERS // 2
    ood = set(males[:half] + females[:half])
    for s in bank:
        bank[s]["pool"] = "ood" if s in ood else "train"

    n_train = sum(1 for s in bank if bank[s]["pool"] == "train")
    print(f"bank: {len(bank)} speakers — {n_train} training-pool, "
          f"{len(ood)} held-out OOD ({sorted(ood)})", flush=True)
    BANK_JSON.write_text(json.dumps(bank, indent=1))
    print(f"wrote {BANK_JSON}", flush=True)

    _build_plan(bank)


def _build_plan(bank: dict) -> None:
    """Deterministic synthesis plan: used triplet -> speaker + jittered knobs."""
    triplets = json.load(open(FACAP_TRIPLETS))
    split = _facap_split(triplets)
    train_pool = sorted(s for s in bank if bank[s]["pool"] == "train")
    ood_pool = sorted(s for s in bank if bank[s]["pool"] == "ood")
    print(f"FACap: {len(triplets)} triplets, {len(split)} used "
          f"({sum(v == 'train' for v in split.values())} train / "
          f"{sum(v == 'dev' for v in split.values())} dev / "
          f"{sum(v == 'headline' for v in split.values())} headline)", flush=True)

    def knobs(r: random.Random) -> dict:
        # light per-utterance jitter around Chatterbox defaults (0.5/0.5/0.8)
        return {"exaggeration": round(r.uniform(0.40, 0.55), 3),
                "cfg_weight": round(r.uniform(0.45, 0.60), 3),
                "temperature": round(r.uniform(0.70, 0.90), 3)}

    rng = random.Random(SEED)
    jobs = []
    for idx in sorted(split):
        text = triplets[idx]["captions"][0]
        spk = rng.choice(train_pool)
        jobs.append({"idx": idx, "cond": "in_dist", "split": split[idx],
                     "speaker": spk, "ref": bank[spk]["ref_wav"],
                     "text": text, "out": str(AUDIO_DIR / f"{idx}.wav"),
                     **knobs(rng)})
    # OOD: re-synthesize dev + headline with held-out voices
    for idx in sorted(i for i, s in split.items() if s in ("dev", "headline")):
        text = triplets[idx]["captions"][0]
        spk = rng.choice(ood_pool)
        jobs.append({"idx": idx, "cond": "ood", "split": split[idx],
                     "speaker": spk, "ref": bank[spk]["ref_wav"],
                     "text": text, "out": str(AUDIO_OOD_DIR / f"{idx}.wav"),
                     **knobs(rng)})

    PLAN_JSON.write_text(json.dumps({"seed": SEED, "n_jobs": len(jobs),
                                     "jobs": jobs}))
    n_ood = sum(j["cond"] == "ood" for j in jobs)
    print(f"plan: {len(jobs)} jobs ({len(jobs) - n_ood} in-dist + {n_ood} OOD)",
          flush=True)
    print(f"wrote {PLAN_JSON}", flush=True)


# ---------------------------------------------------------------------------
# synth mode
# ---------------------------------------------------------------------------
def synth(shard: int, num_shards: int) -> None:
    import torch
    import torchaudio
    import torchaudio.functional as AF
    from chatterbox.tts import ChatterboxTTS

    plan = json.load(open(PLAN_JSON))
    jobs = plan["jobs"][shard::num_shards]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OOD_DIR.mkdir(parents=True, exist_ok=True)
    fail_log = PLAN14_AUDIO / f"synth_fail_shard{shard}.log"

    print(f"[shard {shard}/{num_shards}] {len(jobs)} jobs; loading Chatterbox...",
          flush=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    done = skipped = failed = 0
    t0 = time.time()
    for n, j in enumerate(jobs):
        out = Path(j["out"])
        if out.exists():
            skipped += 1
            continue
        try:
            wav = model.generate(
                j["text"], audio_prompt_path=j["ref"],
                exaggeration=j["exaggeration"], cfg_weight=j["cfg_weight"],
                temperature=j["temperature"])
            wav = wav.detach().cpu()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if model.sr != OUT_SR:
                wav = AF.resample(wav, model.sr, OUT_SR)
            torchaudio.save(str(out), wav, OUT_SR)
            done += 1
        except Exception as e:  # noqa: BLE001 — one bad utterance must not kill the shard
            failed += 1
            with open(fail_log, "a") as f:
                f.write(f"{j['idx']}\t{j['cond']}\t{type(e).__name__}: {e}\n")
        if (n + 1) % 200 == 0:
            rate = (done + skipped) / max(time.time() - t0, 1e-9)
            print(f"[shard {shard}] {n + 1}/{len(jobs)}  done={done} "
                  f"skip={skipped} fail={failed}  {rate:.2f} job/s", flush=True)
    print(f"[shard {shard}] DONE  done={done} skipped={skipped} failed={failed}"
          f"  ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# manifest mode
# ---------------------------------------------------------------------------
def build_manifest() -> None:
    plan = json.load(open(PLAN_JSON))
    bank = json.load(open(BANK_JSON))
    manifest = {"in_dist": {}, "ood": {}}
    missing = 0
    for j in plan["jobs"]:
        rec = {"wav": j["out"], "speaker": j["speaker"], "split": j["split"],
               "gender": bank[j["speaker"]]["gender"],
               "accent": bank[j["speaker"]]["accent"]}
        if not Path(j["out"]).exists():
            missing += 1
            continue
        manifest[j["cond"]][str(j["idx"])] = rec
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest['in_dist'])} in-dist + "
          f"{len(manifest['ood'])} OOD clips; {missing} planned wavs missing",
          flush=True)
    print(f"wrote {MANIFEST_JSON}", flush=True)
    if missing:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("bank")
    sp = sub.add_parser("synth")
    sp.add_argument("--shard", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sub.add_parser("manifest")
    args = ap.parse_args()

    if args.mode == "bank":
        build_bank()
    elif args.mode == "synth":
        synth(args.shard, args.num_shards)
    elif args.mode == "manifest":
        build_manifest()


if __name__ == "__main__":
    main()
