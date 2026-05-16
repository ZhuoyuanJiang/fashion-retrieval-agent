"""Plan 14 M1 — image+audio multimodal-forward smoke test (Plan 14 §6).

The gate before any audio-tower work. speechQwen2VL (DanJZY/Qwen2-VL-7B-Speech)
was only ever fed *audio + text* (ASR); Qwen2-VL natively does *image + text*.
**image + audio in ONE message is untested.** This one-off check verifies:

  A. The model loads with `audio_encoder` + `audio_projector` present, no audio
     weights among the missing keys, audio weights finite and non-zero, and
     `config.audio_token_id` set.
  B. A single [image, audio, text] user message tokenizes with audio tokens
     actually placed, forwards to a finite last hidden state, and yields a
     finite pooled query embedding (last non-pad token, mirroring the towers).
  C. bs=2 with two different-length audios produces different per-sample
     audio-token counts (the audio duration is honored), pads, and forwards
     to finite hidden states + distinct pooled embeddings for both rows.

Run in the `fashion_retrieval` conda env. Exits non-zero if any check fails.
This is a one-off check, not a new test harness (Plan 14 §6).
"""
import glob
import sys

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

BASE_REPO = "DanJZY/Qwen2-VL-7B-Speech"
FACAP_DIR = "/ssd1/zhuoyuan/facap-images"
AUDIO_DIR = "/tmp3/zhuoyuan/tts_pilot/audio_out/chatterbox"
DEVICE = "cuda:0"

# Plan 14 §4: the query user turn is [image, audio, fixed instruction]. The
# instruction is constant task framing — it does NOT contain the modification
# (that lives entirely in the audio).
INSTR = (
    "Given this product image, find the item that looks like the image "
    "but with the modification described in the spoken audio."
)

_fails = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------------------
# Check A — weights load
# ---------------------------------------------------------------------------
print(f"=== loading {BASE_REPO} ===", flush=True)
model, loading_info = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_REPO,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    output_loading_info=True,
)
model.eval()
processor = Qwen2VLProcessor.from_pretrained(BASE_REPO)

# audio_encoder / audio_projector may sit at the top level or nested under
# the inner Qwen2VLModel depending on the transformers refactor — search the
# whole module tree rather than assuming an attribute path.
audio_mods = {n: m for n, m in model.named_modules()
              if n.split(".")[-1] in ("audio_encoder", "audio_projector")}
enc = next((m for n, m in audio_mods.items()
            if n.split(".")[-1] == "audio_encoder"), None)
proj = next((m for n, m in audio_mods.items()
             if n.split(".")[-1] == "audio_projector"), None)
check("audio_encoder + audio_projector modules present",
      enc is not None and proj is not None,
      f"found at {sorted(audio_mods)}" if audio_mods else "not found anywhere")

missing = loading_info.get("missing_keys", [])
audio_missing = [k for k in missing
                 if "audio_encoder" in k or "audio_projector" in k]
check("no audio weights among missing_keys", not audio_missing,
      f"{len(audio_missing)} audio keys missing"
      + (f" e.g. {audio_missing[:2]}" if audio_missing else ""))

if enc is not None and proj is not None:
    ae = torch.cat([p.flatten().float() for p in enc.parameters()])
    ap = torch.cat([p.flatten().float() for p in proj.parameters()])
    check("audio weights finite",
          bool(torch.isfinite(ae).all()) and bool(torch.isfinite(ap).all()))
    check("audio weights not all-zero",
          ae.abs().sum().item() > 0 and ap.abs().sum().item() > 0,
          f"encoder={ae.numel()/1e6:.1f}M params, projector={ap.numel()/1e6:.1f}M params")

audio_token_id = getattr(model.config, "audio_token_id", None)
check("config.audio_token_id is set", audio_token_id is not None,
      f"audio_token_id={audio_token_id}")


# ---------------------------------------------------------------------------
# Shared helper: build inputs for a batch of (image_path, audio_path|None)
# ---------------------------------------------------------------------------
def build_and_forward(items):
    """items: list of (image_path, audio_path or None). Returns the model
    outputs + the processor batch."""
    convs = []
    for img_p, aud_p in items:
        content = [{"type": "image", "image": Image.open(img_p).convert("RGB")}]
        if aud_p is not None:
            content.append({"type": "audio", "audio": aud_p})
        content.append({"type": "text", "text": INSTR})
        convs.append([{"role": "user", "content": content}])

    texts = [processor.apply_chat_template(c, tokenize=False,
                                           add_generation_prompt=True)
             for c in convs]
    image_inputs, video_inputs, audio_inputs = process_vision_info(convs)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        padding=True,
    )
    batch = {k: (v.to(DEVICE) if hasattr(v, "to") else v)
             for k, v in batch.items()}
    with torch.inference_mode():
        outputs = model(**batch, output_hidden_states=True, use_cache=False)
    return outputs, batch


def pooled_last_token(outputs, batch):
    """Last non-pad token hidden state — mirrors the tower pooling
    (Qwen2-VL is left-padded; find the rightmost attended token)."""
    last_hs = outputs.hidden_states[-1]            # (B, S, 3584)
    attn = batch["attention_mask"]
    B, S = attn.shape
    seq_ends = S - 1 - attn.flip(dims=[1]).long().argmax(dim=1)
    return last_hs[torch.arange(B, device=last_hs.device), seq_ends, :]


# ---------------------------------------------------------------------------
# Check B — single [image, audio, text] forward
# ---------------------------------------------------------------------------
imgs = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG")
              for p in glob.glob(f"{FACAP_DIR}/**/{ext}", recursive=True))
if len(imgs) < 2:
    print(f"FATAL: need >=2 images under {FACAP_DIR}, found {len(imgs)}")
    sys.exit(2)
wav_short = f"{AUDIO_DIR}/shortest.wav"
wav_long = f"{AUDIO_DIR}/longest.wav"

print("\n=== Check B — single image+audio+text forward ===", flush=True)
out1, batch1 = build_and_forward([(imgs[0], wav_short)])

n_audio_tok = int((batch1["input_ids"] == audio_token_id).sum())
check("audio tokens placed in input_ids", n_audio_tok > 0,
      f"{n_audio_tok} audio tokens")
check("audio_features present in processor batch", "audio_features" in batch1,
      f"shape={tuple(batch1['audio_features'].shape)}" if "audio_features" in batch1 else "absent")

last_hs1 = out1.hidden_states[-1]
check("last hidden state finite", bool(torch.isfinite(last_hs1).all()),
      f"shape={tuple(last_hs1.shape)}")
pooled1 = pooled_last_token(out1, batch1)
check("pooled query embedding finite", bool(torch.isfinite(pooled1).all()),
      f"shape={tuple(pooled1.shape)}, norm={pooled1.float().norm():.2f}")


# ---------------------------------------------------------------------------
# Check C — bs=2 padded, two different-length audios
# ---------------------------------------------------------------------------
print("\n=== Check C — bs=2 padded (different-length audios) ===", flush=True)
out2, batch2 = build_and_forward([(imgs[0], wav_short), (imgs[1], wav_long)])

ids = batch2["input_ids"]
per_sample_audio = [int((ids[i] == audio_token_id).sum()) for i in range(ids.shape[0])]
check("bs=2 batched and padded", ids.shape[0] == 2,
      f"batch shape={tuple(ids.shape)}, attention_mask={tuple(batch2['attention_mask'].shape)}")
check("per-sample audio-token counts differ (audio length honored)",
      per_sample_audio[0] != per_sample_audio[1],
      f"shortest->{per_sample_audio[0]} toks, longest->{per_sample_audio[1]} toks")

last_hs2 = out2.hidden_states[-1]
check("bs=2 last hidden state finite", bool(torch.isfinite(last_hs2).all()),
      f"shape={tuple(last_hs2.shape)}")
pooled2 = pooled_last_token(out2, batch2)
check("bs=2 pooled embeddings finite", bool(torch.isfinite(pooled2).all()))
cos = torch.nn.functional.cosine_similarity(
    pooled2[0].float(), pooled2[1].float(), dim=0).item()
check("bs=2 rows are distinct (not collapsed)", cos < 0.999,
      f"cos(row0,row1)={cos:.4f}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n=== M1 smoke verdict ===", flush=True)
if _fails:
    print(f"FAILED ({len(_fails)}): {_fails}")
    sys.exit(1)
print("ALL CHECKS PASSED — image+audio multimodal forward is wired correctly.")
print("Plan 14 §6 gate cleared; tower work may proceed.")
