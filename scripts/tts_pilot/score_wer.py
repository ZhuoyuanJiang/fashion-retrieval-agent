"""WER scoring for the TTS pilot.

Transcribes every audio_out/<model>/<tag>.wav with faster-whisper and computes
WER vs the source caption. Low WER = the words survived = good for retrieval.
Scans whatever model folders exist, so it can be re-run as models are added.
"""
import json, re
from pathlib import Path
from faster_whisper import WhisperModel
from jiwer import wer

CAPS = json.load(open("/tmp3/zhuoyuan/tts_pilot/sample_captions.json"))
AUDIO = Path("/tmp3/zhuoyuan/tts_pilot/audio_out")


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


print("loading faster-whisper medium.en (cpu)...", flush=True)
asr = WhisperModel("medium.en", device="cpu", compute_type="int8")

results = {}
for mdir in sorted(p for p in AUDIO.iterdir() if p.is_dir()):
    per = {}
    for tag, ref in CAPS.items():
        wav = mdir / f"{tag}.wav"
        if not wav.exists():
            continue
        segs, _ = asr.transcribe(str(wav), language="en", beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        per[tag] = {"wer": wer(norm(ref), norm(hyp)), "hyp": hyp, "ref": ref}
    avg = sum(v["wer"] for v in per.values()) / len(per) if per else None
    results[mdir.name] = {"avg_wer": avg, "per_caption": per}
    print(f"{mdir.name:16s} avg WER = {avg*100:.1f}%" if avg is not None else mdir.name, flush=True)

json.dump(results, open("/tmp3/zhuoyuan/tts_pilot/wer_results.json", "w"), indent=1)
print("saved -> wer_results.json", flush=True)
