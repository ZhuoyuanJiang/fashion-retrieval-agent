"""Augment the demo preset cache with the two new two-tower rows.

Reads the existing `runs/demo/preset_cache.json` (P1 + P2 already present) and
adds, per preset, the retrieval results of:

  - `text2t`  : Plan-13 text two-tower  (modification = typed text)
  - `audio2t` : Plan-15 audio two-tower (modification = the preset's spoken clip)

Both are `TwoTowerSharedBackbone` (see `pipelines/two_tower.py`). The audio clip
for each preset is its Plan-14 in-distribution TTS wav, located by matching the
preset (candidate image id + modification text) to its FACAP dress triplet
index — no fresh synthesis needed.

Run on a GPU host that has the checkpoints, gallery embeddings and the Plan-14
audio wavs (vllab11). The two ~9B models are loaded sequentially (text freed
before audio) to fit a single GPU.

Usage (defaults are the vllab11 paths):
  python -m src.demo.precompute_two_tower
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch

from . import config
from .pipelines.two_tower import load_gallery, load_two_tower, run_two_tower_inference
from .precompute_presets import find_rank, load_candidate_image

# vllab11 defaults — the run hosts for Plan-13 (text) and Plan-15 (audio).
TEXT_RUN = "/ssd2/zhuoyuan/plan10_runs/plan13_shared_bs24_20260513_202739"
AUDIO_RUN = "/ssd2/zhuoyuan/plan15_runs/audio_query_bs32"
DEF_TEXT_CKPT = f"{TEXT_RUN}/ckpt_epoch14"          # Plan-13 peak
DEF_TEXT_GALLERY = f"{TEXT_RUN}/gallery_emb_epoch14.npy"
DEF_AUDIO_CKPT = f"{AUDIO_RUN}/ckpt_epoch17"        # Plan-15 dev-selected peak
DEF_AUDIO_GALLERY = f"{AUDIO_RUN}/gallery_emb_epoch17.npy"
DEF_AUDIO_WAV_DIR = "/ssd2/zhuoyuan/plan14_audio/audio"
DEF_TRIPLETS = str(config.REPO_ROOT / "data_exploration" / "datasets"
                   / "facap-repo" / "data" / "facap" / "cir_triplets"
                   / "dress_train_triplets.json")


def _img_id(facap_path: str) -> str:
    return facap_path.rsplit("/", 1)[-1].removesuffix(".jpeg")


def map_presets_to_wavs(presets: list, triplets_path: str,
                        wav_dir: str) -> dict[str, str]:
    """preset_id -> in-distribution audio wav, via the FACAP triplet index.

    Each preset is a FACAP dress triplet; the Plan-14 synthesis named its wav
    `audio/<triplet_idx>.wav`. Match on candidate image id + modification text.
    """
    trip = json.loads(Path(triplets_path).read_text())
    by_cand: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(trip):
        by_cand[_img_id(t["candidate"])].append(i)

    out: dict[str, str] = {}
    for p in presets:
        cid, txt = p["candidate_image_id"], p["modification_text"]
        match = [i for i in by_cand.get(cid, [])
                 if trip[i]["captions"][0] == txt]
        if not match:
            raise RuntimeError(
                f"{p['preset_id']}: no FACAP triplet matches "
                f"candidate {cid} + its modification text"
            )
        wav = Path(wav_dir) / f"{match[0]}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"{p['preset_id']}: wav missing — {wav}")
        out[p["preset_id"]] = str(wav)
    return out


def _result_dict(ids, scores, lat, true_tid, extra=None) -> dict:
    d = {
        "target_ids": ids,
        "scores": scores,
        "latency": lat,
        "intermediate": {},
        "true_target_rank": find_rank(true_tid, ids),
    }
    if extra:
        d.update(extra)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(config.PRESET_CACHE_JSON))
    ap.add_argument("--text-ckpt", default=DEF_TEXT_CKPT)
    ap.add_argument("--text-gallery", default=DEF_TEXT_GALLERY)
    ap.add_argument("--audio-ckpt", default=DEF_AUDIO_CKPT)
    ap.add_argument("--audio-gallery", default=DEF_AUDIO_GALLERY)
    ap.add_argument("--audio-wav-dir", default=DEF_AUDIO_WAV_DIR)
    ap.add_argument("--triplets", default=DEF_TRIPLETS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-k", type=int, default=50)
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    presets = cache["presets"]
    print(f"[init] {len(presets)} presets from {args.cache}", flush=True)

    wav_map = map_presets_to_wavs(presets, args.triplets, args.audio_wav_dir)
    images = {p["preset_id"]: load_candidate_image(p["candidate_image_id"])
              for p in presets}
    new_thumbs: set[str] = set()

    # ---- text two-tower (Plan-13) -----------------------------------------
    g_emb, g_ids = load_gallery(Path(args.text_gallery))
    model = load_two_tower(Path(args.text_ckpt), "text", args.device)
    for p in presets:
        ids, scores, lat = run_two_tower_inference(
            model, args.device, g_emb, g_ids,
            images[p["preset_id"]], p["modification_text"], k=args.k,
        )
        p["text2t"] = _result_dict(ids, scores, lat, p["true_target_id"])
        new_thumbs.update(ids)
        print(f"  [text2t]  {p['preset_id']}: "
              f"rank={p['text2t']['true_target_rank']}  ({lat['embed_s']:.2f}s)",
              flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- audio two-tower (Plan-15) ----------------------------------------
    g_emb, g_ids = load_gallery(Path(args.audio_gallery))
    model = load_two_tower(Path(args.audio_ckpt), "audio", args.device)
    for p in presets:
        wav = wav_map[p["preset_id"]]
        ids, scores, lat = run_two_tower_inference(
            model, args.device, g_emb, g_ids,
            images[p["preset_id"]], wav, k=args.k,
        )
        p["audio2t"] = _result_dict(ids, scores, lat, p["true_target_id"],
                                    extra={"audio_wav": wav})
        new_thumbs.update(ids)
        print(f"  [audio2t] {p['preset_id']}: "
              f"rank={p['audio2t']['true_target_rank']}  ({lat['embed_s']:.2f}s)",
              flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- write back --------------------------------------------------------
    cache.setdefault("metadata", {})["two_tower"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "text_ckpt": args.text_ckpt,
        "audio_ckpt": args.audio_ckpt,
        "note": "text2t = Plan-13 text two-tower (ckpt_epoch14); "
                "audio2t = Plan-15 audio two-tower (ckpt_epoch17, dev-selected).",
    }
    Path(args.cache).write_text(json.dumps(cache, indent=2))
    print(f"[done] wrote {args.cache}", flush=True)

    # ---- thumbnails (portability) -----------------------------------------
    thumbs = config.PRESET_THUMBS_DIR
    thumbs.mkdir(parents=True, exist_ok=True)
    n = 0
    for tid in new_thumbs:
        src = config.GALLERY_DIR / f"{tid}.jpeg"
        dst = thumbs / f"{tid}.jpeg"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            n += 1
    print(f"[done] copied {n} new thumbnails ({len(new_thumbs)} needed)",
          flush=True)


if __name__ == "__main__":
    main()
