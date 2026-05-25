"""Two-tower retrieval pipeline for the demo — shared-backbone
`TwoTowerSharedBackbone` (Plan-13 text query tower / Plan-15 audio query tower).

One implementation; `query_modality` selects text vs audio. Both checkpoints
share the layout written by `train_plan10.py`:

    ckpt_epochN/
      shared_backbone/   PEFT save_pretrained — both adapters ("query","target")
      head_query.pt      query projection-head state_dict
      head_target.pt     target projection-head state_dict
      logit_scale.pt     unused here (cosine ranking is scale-invariant)

The gallery is the target tower's pre-encoded embeddings from the training run
(`gallery_emb_epochN.npy` + sibling `.meta.json`), so the demo never re-encodes
the ~59k-image gallery at request time.

Used by:
  - `precompute_presets` — the two cached rows (text / audio two-tower)
  - `app.py` live row — audio two-tower, mic input encoded on the fly

`run_two_tower_inference` serves both: pass a modification-text string (text
modality) or a wav path (audio modality, incl. a freshly-recorded clip).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import config


def load_two_tower(ckpt_dir: Path, query_modality: str, device: str = "cuda:0"):
    """Construct `TwoTowerSharedBackbone` and load a checkpoint for inference.

    `query_modality` is "text" or "audio". Returns the eval-ready model.
    Constructing the model loads the ~9B speechQwen2VL base; load the text and
    audio models sequentially (free one before the next) on a <48 GB GPU.
    """
    sys.path.insert(0, str(config.REPO_ROOT))
    from peft import load_peft_weights, set_peft_model_state_dict

    from src.training.two_tower_model import TwoTowerSharedBackbone

    ckpt_dir = Path(ckpt_dir)
    tag = f"two-tower:{query_modality}"
    print(f"[{tag}] constructing model on {device}...", flush=True)
    t0 = time.perf_counter()
    model = TwoTowerSharedBackbone(
        d_target=512, device_map=device, query_modality=query_modality,
    )

    print(f"[{tag}] loading checkpoint {ckpt_dir}...", flush=True)
    # `save_pretrained` on the two-adapter PeftModel writes one subdir per
    # adapter (shared_backbone/query, shared_backbone/target). Load each into
    # its named slot — both adapters already exist on the constructed model.
    for adapter in ("query", "target"):
        weights = load_peft_weights(str(ckpt_dir / "shared_backbone" / adapter))
        set_peft_model_state_dict(model.vlm, weights, adapter_name=adapter)
    model.head_query.load_state_dict(
        torch.load(str(ckpt_dir / "head_query.pt"), map_location=device)
    )
    model.head_target.load_state_dict(
        torch.load(str(ckpt_dir / "head_target.pt"), map_location=device)
    )
    # The projection heads are created on CPU during training (Accelerator
    # places them per rank). For single-GPU inference, move them explicitly.
    model.head_query.to(device)
    model.head_target.to(device)
    model.eval()
    print(f"[{tag}] ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def load_gallery(gallery_emb_npy: Path) -> tuple[torch.Tensor, list[str]]:
    """Load target-tower gallery embeddings + ids from a run's
    `gallery_emb_epochN.npy` and its sibling `.meta.json`."""
    gallery_emb_npy = Path(gallery_emb_npy)
    emb = np.load(gallery_emb_npy)
    meta = json.loads(gallery_emb_npy.with_suffix(".meta.json").read_text())
    # The meta is either a bare list of ids or a dict carrying them.
    if isinstance(meta, list):
        gallery_ids = meta
    else:
        gallery_ids = (meta.get("gallery_ids") or meta.get("ids")
                       or meta.get("target_ids"))
    if gallery_ids is None or len(gallery_ids) != emb.shape[0]:
        raise ValueError(
            f"gallery meta mismatch: {len(gallery_ids) if gallery_ids else None}"
            f" ids vs {emb.shape[0]} embeddings ({gallery_emb_npy})"
        )
    return torch.from_numpy(emb).float(), list(gallery_ids)


@torch.inference_mode()
def run_two_tower_inference(
    model,
    device: str,
    gallery_emb: torch.Tensor,
    gallery_ids: list[str],
    image: Image.Image,
    mod,
    k: int = 50,
) -> tuple[list[str], list[float], dict[str, float]]:
    """Encode one query and cosine-rank against the gallery.

    `mod` is a modification-text string (text modality) or a wav path
    (audio modality — including a freshly-recorded live clip). `encode_query`
    returns an L2-normalized embedding, so a plain dot product is cosine
    similarity. Returns (top_ids, top_scores, latency).
    """
    t0 = time.perf_counter()
    q = model.encode_query([image], [mod])          # (1, 512), L2-normalized
    embed_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    sims = (gallery_emb.to(device) @ q.T.float()).squeeze(1)   # (N,)
    topk = torch.topk(sims, min(k, sims.numel()))
    search_s = time.perf_counter() - t1

    top_ids = [gallery_ids[i] for i in topk.indices.tolist()]
    top_scores = [round(float(s), 4) for s in topk.values.tolist()]
    return top_ids, top_scores, {
        "embed_s": round(embed_s, 4),
        "search_s": round(search_s, 4),
    }
