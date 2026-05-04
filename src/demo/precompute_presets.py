"""Build the v0.1 demo preset cache from real model outputs.

P1 (caption-based, Marqo-FashionCLIP) results come from a precomputed dump:
  runs/baseline_v1_speechqwen2vl_marqo-fashionclip/qualitative/results.jsonl

P2 (contrastive) results are computed live by loading the Plan-6 checkpoint
and running inference for each preset's (candidate_image, modification_text):
  ckpt_epoch16 / vlm_lora + proj_head.pt → 512-d query
  cosine vs runs/plan5/target_emb_cache_marqo-fashionclip.npy → top-50

Two modes:
  --survey: run P2 on N candidate queries, print rank-comparison table.
            Use this to choose which 8 query_ids tell the best story.
  --finalize: write the full preset_cache.json + copy thumbnails to
              runs/demo/preset_thumbs/, given a list of chosen query_ids.

Usage:
  # 1. Survey — sample 30 across P1 rank tiers, see which ones look interesting
  python -m src.demo.precompute_presets survey --sample 30

  # 2. Finalize — given chosen query_ids, write the real preset cache
  python -m src.demo.precompute_presets finalize \\
      --query-ids 91306678_2 90834118_0 ...
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from .pipelines.base import PipelineResult


# -------- Constants --------
P1_DUMP = config.REPO_ROOT / "runs" / "baseline_v1_speechqwen2vl_marqo-fashionclip" / "qualitative" / "results.jsonl"
P1_CAPTION_DB_DIR = config.REPO_ROOT / "runs" / "caption_db" / "hf-hub:Marqo__marqo-fashionCLIP"
ENCODER_ID = "marqo-fashionclip"


# ============================================================================
# Loaders
# ============================================================================

def load_p1_dump() -> dict[str, dict]:
    """Read the marqo-fashionclip P1 results.jsonl into dict[query_id -> entry]."""
    if not P1_DUMP.exists():
        raise FileNotFoundError(f"P1 dump not found: {P1_DUMP}")
    out = {}
    with P1_DUMP.open() as f:
        for line in f:
            e = json.loads(line)
            out[e["query_id"]] = e
    return out


def load_target_cache() -> tuple[torch.Tensor, list[str]]:
    """Load the (59048, 512) gallery embedding cache + gallery_ids list."""
    sys.path.insert(0, str(config.REPO_ROOT))
    from src.training.target_cache import load_target_cache as _ltc
    embeddings, gallery_ids, _dim = _ltc(
        config.REPO_ROOT / "runs" / "plan5",
        ENCODER_ID,
    )
    return torch.from_numpy(embeddings), gallery_ids


def load_p1_encoder_and_db():
    """Load the Marqo-FashionCLIP text encoder + caption DB used for P1.

    Returns (encoder_with_encode_method, caption_db).
    """
    sys.path.insert(0, str(config.REPO_ROOT))
    from src.baseline.encoder_zoo import get as get_encoder_cfg
    from src.baseline.replay_with_encoder import _load_st_model
    from src.baseline.retrieve import CaptionDB

    print(f"[P1] loading marqo-fashionclip text encoder...", flush=True)
    cfg = get_encoder_cfg(ENCODER_ID)
    encoder = _load_st_model(cfg)

    print(f"[P1] loading caption DB from {P1_CAPTION_DB_DIR}...", flush=True)
    db = CaptionDB.load(P1_CAPTION_DB_DIR)
    print(f"[P1] DB has {db.embeddings.shape[0]} captions, dim={db.embeddings.shape[1]}", flush=True)
    return encoder, db


def run_p1_topk(encoder, db, caption: str, k: int = 50) -> tuple[list[str], list[float], dict[str, float]]:
    """Encode caption + cosine search over caption DB. Returns (top_ids, scores, latency)."""
    import sys
    sys.path.insert(0, str(config.REPO_ROOT))
    from src.baseline.retrieve import top_k as _top_k

    t0 = time.perf_counter()
    q = encoder.encode([caption], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
    encode_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    hits = _top_k(q, db, k=k)  # list of (target_id, score)
    search_s = time.perf_counter() - t1

    top_ids = [h[0] for h in hits]
    top_scores = [round(float(h[1]), 4) for h in hits]
    return top_ids, top_scores, {"encode_s": round(encode_s, 4), "search_s": round(search_s, 4)}


def load_p2_model(device: str = "cuda:0"):
    """Load Plan-6 contrastive checkpoint. Returns (model, device)."""
    sys.path.insert(0, str(config.REPO_ROOT))
    from src.training.contrastive_model import ContrastiveQwen2VL
    from peft import load_peft_weights, set_peft_model_state_dict

    print(f"[P2] loading ContrastiveQwen2VL on {device}...", flush=True)
    t0 = time.perf_counter()
    model = ContrastiveQwen2VL(d_target=512, device_map=device)

    ckpt = config.CKPT_DIR
    print(f"[P2] loading checkpoint weights from {ckpt}...", flush=True)
    peft_weights = load_peft_weights(str(ckpt / "vlm_lora"))
    set_peft_model_state_dict(model.vlm, peft_weights)
    model.proj.load_state_dict(
        torch.load(str(ckpt / "proj_head.pt"), map_location=device)
    )
    # Projection head is created on CPU (training relies on Accelerator to
    # place it on each rank's GPU). For single-GPU inference we move it manually.
    model.proj.to(device)
    model.eval()
    print(f"[P2] ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model, device


# ============================================================================
# Per-query inference
# ============================================================================

@torch.inference_mode()
def run_p2_inference(
    model,
    device: str,
    target_emb: torch.Tensor,
    gallery_ids: list[str],
    image: Image.Image,
    mod_text: str,
    k: int = 50,
) -> tuple[list[str], list[float], dict[str, float]]:
    """Run P2 forward + cosine search. Returns (top_ids, top_scores, latency)."""
    t0 = time.perf_counter()
    q = model([image], [mod_text])  # (1, 512), L2-normalized
    embed_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    sims = (target_emb.to(device) @ q.T.float()).squeeze(1)  # (N,)
    topk = torch.topk(sims, k)
    search_s = time.perf_counter() - t1

    top_ids = [gallery_ids[i] for i in topk.indices.tolist()]
    top_scores = [round(s, 4) for s in topk.values.tolist()]
    return top_ids, top_scores, {"embed_s": round(embed_s, 4), "search_s": round(search_s, 4)}


def find_rank(true_target: str, ranked_ids: list[str]) -> int | None:
    """1-based rank if true_target is in ranked_ids, else None."""
    try:
        return ranked_ids.index(true_target) + 1
    except ValueError:
        return None


def load_candidate_image(query_id: str) -> Image.Image:
    """Open the candidate image for a query from /ssd1/zhuoyuan/facap-images/."""
    path = config.GALLERY_DIR / f"{query_id}.jpeg"
    if not path.exists():
        raise FileNotFoundError(f"candidate image not found: {path}")
    return Image.open(path).convert("RGB")


# ============================================================================
# Modes
# ============================================================================

def cmd_survey(args, p1_dump: dict[str, dict]) -> None:
    """Run P2 on N candidate queries; print rank-comparison table."""
    rng = random.Random(args.seed)

    # Pick candidate query_ids: stratified sample across P1 rank tiers.
    by_tier = {"top1": [], "top5": [], "top10": [], "top50": [], "outside_top50": []}
    for qid, e in p1_dump.items():
        r = e["rank"]
        if r <= 1: by_tier["top1"].append(qid)
        elif r <= 5: by_tier["top5"].append(qid)
        elif r <= 10: by_tier["top10"].append(qid)
        elif r <= 50: by_tier["top50"].append(qid)
        else: by_tier["outside_top50"].append(qid)

    if args.query_ids:
        candidate_ids = args.query_ids
    else:
        per_tier = max(1, args.sample // 5)
        candidate_ids = []
        for tier_ids in by_tier.values():
            rng.shuffle(tier_ids)
            candidate_ids.extend(tier_ids[:per_tier])
        rng.shuffle(candidate_ids)
        candidate_ids = candidate_ids[:args.sample]

    print(f"\n[survey] running P2 on {len(candidate_ids)} candidate queries\n")

    target_emb, gallery_ids = load_target_cache()
    model, device = load_p2_model(args.device)
    target_emb = target_emb.to(device).float()

    rows = []
    for i, qid in enumerate(candidate_ids, 1):
        if qid not in p1_dump:
            print(f"  [{i:>3}/{len(candidate_ids)}] {qid}: NOT IN P1 DUMP — skip")
            continue
        e = p1_dump[qid]
        try:
            img = load_candidate_image(qid)
        except FileNotFoundError as ex:
            print(f"  [{i:>3}/{len(candidate_ids)}] {qid}: image missing — skip ({ex})")
            continue
        top_ids, _, lat = run_p2_inference(
            model, device, target_emb, gallery_ids,
            img, e["modification_text"], k=50,
        )
        p2_rank = find_rank(e["true_target"], top_ids)
        p1_rank = e["rank"] if e["rank"] <= 50 else None

        rows.append({
            "query_id": qid,
            "true_target": e["true_target"],
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "mod_text": e["modification_text"][:80],
            "p2_embed_s": lat["embed_s"],
        })
        p1s = f"{p1_rank}" if p1_rank else ">50"
        p2s = f"{p2_rank}" if p2_rank else ">50"
        print(f"  [{i:>3}/{len(candidate_ids)}] {qid}  P1={p1s:>4}  P2={p2s:>4}  ({lat['embed_s']:.2f}s)")

    # Save full table
    out_csv = config.REPO_ROOT / "runs" / "demo" / "survey.jsonl"
    with out_csv.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[survey] wrote {len(rows)} rows to {out_csv}")

    # Pretty-print sortable summaries
    def category(p1, p2):
        if p2 is None: return "P2_fail"
        if p1 is None: return "P1_fail_P2_ok"
        if p2 < p1: return "P2_better"
        if p2 > p1: return "P1_better"
        return "tie"

    print("\n=== sorted by P2 rank (best P2 first) ===")
    for r in sorted([x for x in rows if x["p2_rank"]], key=lambda x: x["p2_rank"])[:15]:
        print(f"  P1={str(r['p1_rank']):>4}  P2={r['p2_rank']:>3}  {r['query_id']}  ({category(r['p1_rank'], r['p2_rank']):>14})  {r['mod_text']}")
    print("\n=== P2 fails (true target outside P2 top-50) ===")
    fails = [x for x in rows if x["p2_rank"] is None]
    for r in fails[:10]:
        print(f"  P1={str(r['p1_rank']):>4}  {r['query_id']}  {r['mod_text']}")
    print("\n=== P1 better than P2 (caption wins) ===")
    p1_wins = [x for x in rows if x["p1_rank"] and x["p2_rank"] and x["p1_rank"] < x["p2_rank"]]
    for r in sorted(p1_wins, key=lambda x: x["p1_rank"])[:10]:
        print(f"  P1={r['p1_rank']:>3}  P2={r['p2_rank']:>3}  {r['query_id']}  {r['mod_text']}")


def cmd_finalize(args, p1_dump: dict[str, dict]) -> None:
    """Build full preset_cache.json + copy thumbnails for the chosen query_ids."""
    chosen = args.query_ids
    if not chosen:
        raise SystemExit("--query-ids required for finalize mode")
    print(f"[finalize] building cache for {len(chosen)} preset(s)")

    p1_encoder, p1_db = load_p1_encoder_and_db()
    target_emb, gallery_ids = load_target_cache()
    model, device = load_p2_model(args.device)
    target_emb = target_emb.to(device).float()

    presets = []
    needed_thumbs: set[str] = set()
    for i, qid in enumerate(chosen, 1):
        if qid not in p1_dump:
            print(f"  [{i}/{len(chosen)}] {qid}: NOT IN P1 DUMP — skip")
            continue
        e = p1_dump[qid]
        img = load_candidate_image(qid)
        true_tid = e["true_target"]

        # P2 top-50
        p2_ids, p2_scores, p2_lat = run_p2_inference(
            model, device, target_emb, gallery_ids,
            img, e["modification_text"], k=50,
        )
        p2_rank = find_rank(true_tid, p2_ids)

        # P1 top-50: re-rank using the cached caption + Marqo-FashionCLIP encoder + caption DB.
        # The original baseline run only saved top-10; we now re-do the cosine search to get top-50.
        p1_ids, p1_scores, p1_lat = run_p1_topk(
            p1_encoder, p1_db, e["generated_caption"], k=50,
        )
        # The caption-DB rank (out of N captions, typically 836 in the dress-eval slice).
        p1_rank_in_topk = find_rank(true_tid, p1_ids)
        # Use the eval's full-gallery rank as the canonical rank reported in the UI;
        # the per-K thumbnail strip is the new top-50.
        p1_rank = e["rank"] if p1_rank_in_topk is None else p1_rank_in_topk

        # Story tag
        if p2_rank is None:
            cat = "P2_fail"
        elif p1_rank is None:
            cat = "P1_fail_P2_ok"
        elif p2_rank < p1_rank:
            cat = "P2_win"
        elif p2_rank > p1_rank:
            cat = "P1_win"
        else:
            cat = "tie"

        presets.append({
            "preset_id": f"preset_{i:02d}",
            "candidate_image_id": qid,
            "modification_text": e["modification_text"],
            "mock_transcript": e["modification_text"],
            "true_target_id": true_tid,
            "category": cat,
            "notes": f"P1 rank={p1_rank}, P2 rank={p2_rank}",
            "p1": {
                "target_ids": p1_ids,
                "scores": p1_scores,
                "latency": p1_lat,
                "intermediate": {"caption": e["generated_caption"]},
                "true_target_rank": p1_rank,
            },
            "p2": {
                "target_ids": p2_ids,
                "scores": p2_scores,
                "latency": p2_lat,
                "intermediate": {},
                "true_target_rank": p2_rank,
            },
        })

        # Track all thumbnails needed for portability
        needed_thumbs.add(qid)
        if true_tid:
            needed_thumbs.add(true_tid)
        needed_thumbs.update(p1_ids)
        needed_thumbs.update(p2_ids)

        print(f"  [{i:>2}/{len(chosen)}] {qid}: P1={p1_rank} P2={p2_rank} ({cat}) — P2 took {p2_lat['embed_s']:.2f}s")

    # Write JSON
    out = {
        "metadata": {
            "schema_version": "v0.1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stage": "v0.1-real",
            "ckpt_dir": str(config.CKPT_DIR.relative_to(config.REPO_ROOT)),
            "encoder": ENCODER_ID,
            "k_p1": 50,
            "k_p2": 50,
            "note": "Real precompute. P1 = Marqo-FashionCLIP encode(cached_caption) + cosine over caption DB (top-50). P2 = Plan-6 ckpt_epoch16 inference (top-50).",
        },
        "presets": presets,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[finalize] wrote {len(presets)} presets to {out_path}")

    # Copy thumbnails
    thumbs_dir = config.PRESET_THUMBS_DIR
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for tid in needed_thumbs:
        src = config.GALLERY_DIR / f"{tid}.jpeg"
        dst = thumbs_dir / f"{tid}.jpeg"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    print(f"[finalize] copied {n_copied} thumbnails to {thumbs_dir} ({len(needed_thumbs)} total needed)")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sur = sub.add_parser("survey", help="Run P2 on N candidates; print P1 vs P2 comparison table")
    p_sur.add_argument("--sample", type=int, default=30)
    p_sur.add_argument("--seed", type=int, default=42)
    p_sur.add_argument("--query-ids", nargs="+", default=None,
                       help="explicit list of query_ids (overrides --sample)")
    p_sur.add_argument("--device", default="cuda:0")

    p_fin = sub.add_parser("finalize", help="Write preset_cache.json + copy thumbs for chosen query_ids")
    p_fin.add_argument("--query-ids", nargs="+", required=True, help="query_ids to use as the demo presets")
    p_fin.add_argument("--output", default=str(config.PRESET_CACHE_JSON))
    p_fin.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    print("[init] loading P1 dump...", flush=True)
    p1_dump = load_p1_dump()
    print(f"[init] {len(p1_dump)} P1 entries loaded", flush=True)

    if args.cmd == "survey":
        cmd_survey(args, p1_dump)
    elif args.cmd == "finalize":
        cmd_finalize(args, p1_dump)
    else:
        raise ValueError(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
