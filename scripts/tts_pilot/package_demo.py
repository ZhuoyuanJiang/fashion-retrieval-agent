"""Package the TTS pilot audio into ~/tts_pilot_demo/ for listening.

Copies every audio_out/<model>/<tag>.wav as <tag>__<model>.wav so the same
caption across models sorts together. Writes a README with captions + WER.
Only ~small wav files go to home (the deliverable); envs stay on /tmp3.
"""
import json, shutil, pathlib

OUT = pathlib.Path.home() / "tts_pilot_demo"
OUT.mkdir(exist_ok=True)
AUDIO = pathlib.Path("/tmp3/zhuoyuan/tts_pilot/audio_out")
caps = json.load(open("/tmp3/zhuoyuan/tts_pilot/sample_captions.json"))
try:
    wer = json.load(open("/tmp3/zhuoyuan/tts_pilot/wer_results.json"))
except FileNotFoundError:
    wer = {}

models = sorted(d.name for d in AUDIO.iterdir() if d.is_dir() and any(d.iterdir()))
n = 0
for m in models:
    for tag in caps:
        src = AUDIO / m / f"{tag}.wav"
        if src.exists():
            shutil.copy(src, OUT / f"{tag}__{m}.wav")
            n += 1

L = ["# TTS pilot — listening demo", "",
     "Same 5 FACap dress modification captions, each synthesized by every",
     "candidate TTS using one shared reference voice",
     "(`ref_en.wav`: \"Some call me nature, others call me mother nature.\").",
     "Files: `<caption-tag>__<model>.wav` — the same caption across models sorts together.",
     "",
     "## WER (faster-whisper medium.en transcription vs source text; lower = better)",
     "", "| Model | avg WER |", "|---|---|"]
for m in models:
    aw = wer.get(m, {}).get("avg_wer")
    L.append(f"| {m} | {aw*100:.1f}% |" if aw is not None else f"| {m} | (not scored) |")
L += ["", "## The 5 captions", ""]
for tag, txt in caps.items():
    L.append(f"- **{tag}** — {txt}")
L += ["", "## Status of other candidates", "",
      "- **IndexTTS-2** — not run: pins torch 2.8 / CUDA 12.8, this server's",
      "  driver is 12.4. Needs a newer-driver host.", ""]
(OUT / "README.md").write_text("\n".join(L) + "\n")
print(f"packaged {n} wavs + README -> {OUT}")
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
