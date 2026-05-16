"""Synthesize the 5 pilot captions with F5-TTS (zero-shot voice clone).

Usage: python synth_f5.py <reference_wav> <reference_text>
F5-TTS requires a reference clip + its transcript.
"""
import sys, json, time, pathlib
from f5_tts.api import F5TTS

OUT = pathlib.Path("/tmp3/zhuoyuan/tts_pilot/audio_out/f5tts")
OUT.mkdir(parents=True, exist_ok=True)
ref_wav, ref_text = sys.argv[1], sys.argv[2]
caps = json.load(open("/tmp3/zhuoyuan/tts_pilot/sample_captions.json"))

print(f"loading F5-TTS (ref={ref_wav})...", flush=True)
f5 = F5TTS()
for tag, text in caps.items():
    t0 = time.time()
    p = OUT / f"{tag}.wav"
    f5.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=text, file_wave=str(p))
    print(f"  {tag}: {time.time()-t0:.1f}s -> {p}", flush=True)
print("DONE f5tts", flush=True)
