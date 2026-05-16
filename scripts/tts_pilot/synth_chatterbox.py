"""Synthesize the 5 pilot captions with Chatterbox TTS.

Usage: python synth_chatterbox.py [reference_wav]
  no arg  -> Chatterbox default voice
  ref wav -> zero-shot clone of that voice
"""
import sys, json, time, pathlib
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

OUT = pathlib.Path("/tmp3/zhuoyuan/tts_pilot/audio_out/chatterbox")
OUT.mkdir(parents=True, exist_ok=True)
ref = sys.argv[1] if len(sys.argv) > 1 else None
caps = json.load(open("/tmp3/zhuoyuan/tts_pilot/sample_captions.json"))

print(f"loading ChatterboxTTS (ref={ref})...", flush=True)
model = ChatterboxTTS.from_pretrained(device="cuda")
for tag, text in caps.items():
    t0 = time.time()
    wav = model.generate(text, audio_prompt_path=ref) if ref else model.generate(text)
    p = OUT / f"{tag}.wav"
    ta.save(str(p), wav.detach().cpu(), model.sr)
    print(f"  {tag}: {time.time()-t0:.1f}s  ({wav.shape[-1]/model.sr:.1f}s audio) -> {p}", flush=True)
print(f"DONE chatterbox  sr={model.sr}", flush=True)
