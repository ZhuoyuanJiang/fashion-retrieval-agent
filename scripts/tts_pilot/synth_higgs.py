"""Synthesize the 5 pilot captions with Higgs Audio V2 (zero-shot voice clone).

Usage: python synth_higgs.py <reference_wav> <reference_text>
"""
import sys, json, time, base64, pathlib
import torch, torchaudio
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

OUT = pathlib.Path("/tmp3/zhuoyuan/tts_pilot/audio_out/higgs")
OUT.mkdir(parents=True, exist_ok=True)
ref_wav, ref_text = sys.argv[1], sys.argv[2]
caps = json.load(open("/tmp3/zhuoyuan/tts_pilot/sample_captions.json"))
ref_b64 = base64.b64encode(open(ref_wav, "rb").read()).decode("utf-8")

# Local checkpoint pinned to revision 7cfe9946a0 — the last old-format config
# (model_type "higgs_audio", nested). main HEAD migrated to native-transformers
# format which the standalone boson_multimodal repo cannot load.
MODEL = "/tmp3/zhuoyuan/tts_pilot/models/higgs_ckpt"
TOKENIZER = "/tmp3/zhuoyuan/tts_pilot/models/higgs_tokenizer"

print("loading HiggsAudioServeEngine...", flush=True)
engine = HiggsAudioServeEngine(MODEL, TOKENIZER, device="cuda")
for tag, text in caps.items():
    t0 = time.time()
    sample = ChatMLSample(messages=[
        Message(role="user", content=ref_text),
        Message(role="assistant", content=AudioContent(raw_audio=ref_b64, audio_url="placeholder")),
        Message(role="user", content=text),
    ])
    out = engine.generate(chat_ml_sample=sample, max_new_tokens=1024,
                          temperature=1.0, top_p=0.95, top_k=50,
                          stop_strings=["<|end_of_text|>", "<|eot_id|>"])
    p = OUT / f"{tag}.wav"
    torchaudio.save(str(p), torch.from_numpy(out.audio)[None, :], out.sampling_rate)
    print(f"  {tag}: {time.time()-t0:.1f}s -> {p}", flush=True)
print("DONE higgs", flush=True)
