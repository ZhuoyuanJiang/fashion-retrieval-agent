"""Replay an existing baseline run with a new text encoder.

Skips the expensive VLM step (~25 min on a 3090) and just:
  1. Loads the saved generated_caption strings from a previous run's
     qualitative/results.jsonl
  2. Builds (or reuses) a caption DB at runs/caption_db/<encoder_slug>/
     using the new encoder
  3. Re-encodes each generated caption as a query, retrieves top-K, computes
     metrics + qualitative dump
  4. Writes results to runs/<new_run_name>/

Usage:
    python -m src.baseline.replay_with_encoder \
        --source-run runs/baseline_v1_speechqwen2vl \
        --encoder-slug bge-large \
        --run-name baseline_v1_speechqwen2vl_bge-large
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.baseline.build_caption_db import (
    _facap_commit_sha,
    _write_db,
    build_signature_full,
    encoder_slug as encoder_slug_fn,
)
from src.baseline.encoder_zoo import EncoderConfig, get as get_encoder_config
from src.baseline.eval import (
    compute_metrics,
    format_metrics_table,
    write_metrics,
    write_qualitative,
)
from src.baseline.retrieve import CaptionDB, rank_of, top_k
from src.data.facap_dataset import (
    DEFAULT_FACAP_ROOT,
    FacapDataset,
    REPO_ROOT,
    _path_to_image_id,
)

TOP_K_QUALITATIVE = 10


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_st_model(cfg: EncoderConfig):
    """Load via sentence-transformers OR open_clip depending on cfg.loader."""
    if cfg.loader == "open_clip":
        return _load_open_clip(cfg)

    kwargs = {"device": _device()}
    if cfg.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if cfg.bf16:
        # low_cpu_mem_usage avoids the fp32->bf16 .to() spike that OOMs the
        # 7-8B models on a 24GB 3090 — weights stream directly into bf16
        # via accelerate's init_empty_weights path.
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
    model = SentenceTransformer(cfg.hf_model_id, **kwargs)
    if cfg.max_seq_length is not None:
        try:
            model.max_seq_length = cfg.max_seq_length
        except Exception:
            pass
        try:
            model[0].max_seq_length = cfg.max_seq_length
        except Exception:
            pass
    return model


class _OpenClipWrapper:
    """Minimal duck-typed shim so callers can treat open_clip encoders
    as if they were a sentence-transformers model.

    Supports `.encode(texts, ...)` for text and `.encode_image(images, ...)`
    for PIL images, both returning L2-normalized np.ndarray.
    Used for Marqo Fashion CLIP/SigLIP, which choke on the sentence-transformers
    -> transformers loader path (meta-tensor errors).
    """
    def __init__(self, model, tokenizer, preprocess_val, device: str, max_seq_length: int | None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.preprocess_val = preprocess_val
        self.device = device
        self.max_seq_length = max_seq_length

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                # open_clip tokenizer truncates to model's context_length
                tokens = self.tokenizer(chunk).to(self.device)
                feats = self.model.encode_text(tokens)
                if normalize_embeddings:
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def encode_image(self, images, batch_size=32):
        """Encode PIL images to L2-normalized float32 np.ndarray (N, D)."""
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                tensors = torch.stack(
                    [self.preprocess_val(img) for img in chunk]
                ).to(self.device)
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().float().numpy())
        return np.concatenate(out, axis=0)


def _load_open_clip(cfg: EncoderConfig) -> _OpenClipWrapper:
    import open_clip
    model, _, preprocess_val = open_clip.create_model_and_transforms(cfg.hf_model_id)
    tokenizer = open_clip.get_tokenizer(cfg.hf_model_id)
    return _OpenClipWrapper(model, tokenizer, preprocess_val, _device(), cfg.max_seq_length)


def _pre_truncate(model, texts: list[str], max_len: int) -> list[str]:
    """Guard against CLIP-family models whose forward errors out on tokens >
    max_position_embeddings regardless of the sentence-transformers
    max_seq_length attr.

    For CLIP/SigLIP (max_len <= 77) we just hard-truncate by character count.
    BPE on English averages ~4 chars/token, so 3*max_len chars gives a wide
    safety margin even before the model's own tokenizer truncation kicks in.
    Cheap and tokenizer-agnostic. We lose tail content past ~250 chars, but
    CLIP couldn't see it anyway.
    """
    if isinstance(model, _OpenClipWrapper):
        return texts  # open_clip tokenizer truncates correctly
    char_budget = max_len * 3  # ~3 chars/token, conservative for English
    return [t[:char_budget] if len(t) > char_budget else t for t in texts]


def _encode(
    model,
    texts: list[str],
    batch_size: int = 32,
    *,
    max_seq_length: int | None = None,
) -> np.ndarray:
    if max_seq_length is not None and max_seq_length <= 128:
        texts = _pre_truncate(model, texts, max_seq_length)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb).astype(np.float32, copy=False)


def build_full_db_with_encoder(
    out_dir: Path,
    cfg: EncoderConfig,
    model: SentenceTransformer,
    category: str = "dress",
    split: str = "train",
) -> None:
    """Encode all FACap target captions with this encoder, write the DB."""
    ds = FacapDataset(category=category, split=split)
    captions: dict[str, str] = ds.captions
    rows: list[dict] = [
        {
            "image_path": path,
            "target_id": _path_to_image_id(path),
            "caption": caption,
            "caption_length_chars": len(caption),
        }
        for path, caption in captions.items()
    ]
    # Apply passage_prefix only at encoding time; rows store the raw caption.
    texts_to_encode = [cfg.passage_prefix + r["caption"] for r in rows]

    print(f"  encoding {len(rows)} target captions with {cfg.hf_model_id} ...")
    t0 = time.time()
    embeddings = _encode(model, texts_to_encode, batch_size=32, max_seq_length=cfg.max_seq_length)
    print(f"  done in {time.time() - t0:.1f}s. embeddings shape={embeddings.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    config = {
        "encoder_name": cfg.hf_model_id,
        "embedding_dim": int(embeddings.shape[1]),
        "category": category,
        "split": split,
        "n_total": len(rows),
        "facap_commit_sha": _facap_commit_sha(DEFAULT_FACAP_ROOT),
        "build_args": build_signature_full(
            category=category, split=split, encoder_name=cfg.hf_model_id,
        ),
        "passage_prefix": cfg.passage_prefix,
        "query_prefix": cfg.query_prefix,
        "encoder_notes": cfg.notes,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def ensure_db(
    db_dir: Path, cfg: EncoderConfig, model: SentenceTransformer,
) -> CaptionDB:
    if not (db_dir / "embeddings.npy").exists():
        print(f"[caption_db] missing — building at {db_dir}")
        build_full_db_with_encoder(db_dir, cfg, model)
    else:
        print(f"[caption_db] reusing {db_dir}")
    return CaptionDB.load(db_dir)


def replay(
    source_run: Path,
    encoder_slug: str,
    run_name: str,
    out_root: Path,
) -> None:
    cfg = get_encoder_config(encoder_slug)
    print(f"=== replay: encoder = {encoder_slug} ({cfg.hf_model_id}) ===")
    print(f"  notes: {cfg.notes}")
    print(f"  query_prefix={cfg.query_prefix!r}  passage_prefix={cfg.passage_prefix!r}")

    # 1. Load the source run's queries (generated_caption + true_target).
    qual_path = source_run / "qualitative" / "results.jsonl"
    if not qual_path.exists():
        raise FileNotFoundError(f"source run has no qualitative dump at {qual_path}")
    src_rows: list[dict] = [json.loads(l) for l in open(qual_path)]
    print(f"  loaded {len(src_rows)} queries from {qual_path}")

    # 2. Load encoder once, share across DB build + query encoding.
    print(f"  loading encoder on {_device()}...")
    t0 = time.time()
    model = _load_st_model(cfg)
    print(f"  encoder loaded in {time.time() - t0:.1f}s")

    # 3. Build / load the DB.
    db_dir = out_root / "caption_db" / encoder_slug_fn(cfg.hf_model_id)
    db = ensure_db(db_dir, cfg, model)
    print(f"  DB ready: {len(db.target_ids)} rows, dim={db.embeddings.shape[1]}")

    # 4. Encode all 1000 generated captions in one batch.
    queries = [cfg.query_prefix + r["generated_caption"] for r in src_rows]
    print(f"  encoding {len(queries)} VLM-generated captions ...")
    t0 = time.time()
    q_embs = _encode(model, queries, batch_size=32, max_seq_length=cfg.max_seq_length)
    print(f"  done in {time.time() - t0:.1f}s")

    # 5. Retrieve + score.
    qualitative_rows: list[dict] = []
    ranks: list[int | None] = []
    for src, q_emb in tqdm(zip(src_rows, q_embs), total=len(src_rows), desc=f"replay [{encoder_slug}]"):
        topk = top_k(q_emb, db, k=TOP_K_QUALITATIVE)
        r = rank_of(src["true_target"], q_emb, db)
        ranks.append(r)
        qualitative_rows.append({
            "query_idx": src.get("query_idx"),
            "query_id": src["query_id"],
            "true_target": src["true_target"],
            "modification_text": src["modification_text"],
            "generated_caption": src["generated_caption"],
            "top10_predicted": [t for t, _ in topk],
            "top10_scores": [round(s, 4) for _, s in topk],
            "rank": r,
            "failure_category": "",
        })

    # 6. Metrics + outputs.
    result = compute_metrics(ranks)
    print(f"\n[{encoder_slug}] metrics on {len(ranks)} queries:")
    print(format_metrics_table(result))

    run_dir = out_root / run_name
    qual_out = write_qualitative(qualitative_rows, run_dir)
    metrics_out = write_metrics(result, run_dir, extra={
        "vlm": "speechqwen2vl",
        "n_eval": len(ranks),
        "category": "dress",
        "split": "train",
        "encoder_slug": encoder_slug,
        "encoder_name": cfg.hf_model_id,
        "embedding_dim": int(db.embeddings.shape[1]),
        "query_prefix": cfg.query_prefix,
        "passage_prefix": cfg.passage_prefix,
        "db_path": str(db_dir),
        "db_mode": "full",
        "source_run": str(source_run),
        "facap_commit_sha": db.config.get("facap_commit_sha"),
    })
    print(f"\nwrote qualitative -> {qual_out}")
    print(f"wrote metrics    -> {metrics_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a baseline run with a new encoder")
    parser.add_argument("--source-run", required=True,
                        help="path to existing run dir (must have qualitative/results.jsonl)")
    parser.add_argument("--encoder-slug", required=True,
                        help=f"slug from src.baseline.encoder_zoo")
    parser.add_argument("--run-name", required=True,
                        help="output run dir name under --out-root")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "runs"))
    args = parser.parse_args()
    replay(
        source_run=Path(args.source_run),
        encoder_slug=args.encoder_slug,
        run_name=args.run_name,
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()
