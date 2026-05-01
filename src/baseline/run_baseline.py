"""Top-level driver for the text-modification caption-generation baseline.

Workflow per query (one FACap triplet):
    1. VLM backend takes (reference_image, modification_text) -> generated caption.
    2. TextEncoder embeds the generated caption -> query embedding.
    3. Retrieve top-K target_ids by cosine similarity against the caption DB.
    4. Score against the true target_id; aggregate Recall@K + ranks.
    5. Save qualitative dump (failure_category to be filled in by hand later).

Caption DB layout (two modes):

  full mode (default, production):
    runs/caption_db/<encoder_slug>/
    Built once per encoder. Reused across all eval runs that use the same
    encoder, regardless of n_eval / VLM / run_name. Encodes ALL targets.

  subset mode (opt-in via --db-size, debug):
    runs/caption_db_subset/eval{n}_db{m}_seed{s}/<encoder_slug>/
    Smaller DB for fast iteration: guaranteed eval targets + sampled
    distractors. Each (eval_n, db_size, seed) triple gets its own dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.baseline.build_caption_db import (
    _facap_commit_sha,
    build_db_full,
    build_db_subset,
    build_signature_full,
    build_signature_subset,
    encoder_slug,
)
from src.data.facap_dataset import DEFAULT_FACAP_ROOT
from src.baseline.eval import (
    compute_metrics,
    format_metrics_table,
    write_metrics,
    write_qualitative,
)
from src.baseline.retrieve import CaptionDB, rank_of, top_k
from src.baseline.text_encoder import DEFAULT_MODEL, TextEncoder
from src.baseline.vlm_caption import make_captioner
from src.data.facap_dataset import DEFAULT_IMAGE_CACHE, FacapDataset, REPO_ROOT

TOP_K_QUALITATIVE = 10  # how many predictions to dump per query


def caption_db_dir(
    out_root: Path,
    encoder_name: str,
    *,
    db_size: int | None,
    eval_n: int | None,
    seed: int | None,
) -> Path:
    """Resolve the on-disk DB directory for the given mode.

    Full mode (db_size is None):
        out_root / caption_db / <encoder_slug>
    Subset mode:
        out_root / caption_db_subset / eval{n}_db{m}_seed{s} / <encoder_slug>
    """
    enc = encoder_slug(encoder_name)
    if db_size is None:
        return out_root / "caption_db" / enc
    return (
        out_root
        / "caption_db_subset"
        / f"eval{eval_n}_db{db_size}_seed{seed}"
        / enc
    )


def ensure_caption_db(
    db_dir: Path,
    category: str,
    split: str,
    encoder_name: str,
    *,
    db_size: int | None,
    eval_n: int | None,
    seed: int | None,
) -> CaptionDB:
    """Load the caption DB at `db_dir`, building it if missing.

    Mode is selected by `db_size`:
      None          -> full mode (encode all captions)
      int (>0)      -> subset mode (eval_n guaranteed + sampled distractors)
    """
    if db_size is None:
        expected = build_signature_full(
            category=category, split=split, encoder_name=encoder_name,
        )
        if not (db_dir / "embeddings.npy").exists():
            print(f"[caption_db] full mode: building at {db_dir}...")
            build_db_full(
                out_dir=db_dir,
                category=category,
                split=split,
                encoder_name=encoder_name,
            )
    else:
        if eval_n is None or seed is None:
            raise ValueError(
                "subset mode (db_size set) requires both eval_n and seed"
            )
        expected = build_signature_subset(
            category=category, split=split, encoder_name=encoder_name,
            eval_n=eval_n, db_size=db_size, seed=seed,
        )
        if not (db_dir / "embeddings.npy").exists():
            print(
                f"[caption_db] subset mode (eval_n={eval_n}, db_size={db_size}, "
                f"seed={seed}): building at {db_dir}..."
            )
            build_db_subset(
                out_dir=db_dir,
                category=category,
                split=split,
                encoder_name=encoder_name,
                eval_n=eval_n,
                db_size=db_size,
                seed=seed,
            )

    db = CaptionDB.load(db_dir)
    existing = db.config.get("build_args")
    if existing is None:
        raise RuntimeError(
            f"caption DB at {db_dir} has no `build_args` in config.json. "
            f"Delete it (rm -rf {db_dir}) and rerun."
        )
    if existing != expected:
        mismatches = {
            k: {"existing": existing.get(k), "requested": expected[k]}
            for k in expected if existing.get(k) != expected[k]
        }
        raise RuntimeError(
            f"caption DB at {db_dir} was built with different args than "
            f"this run.\nmismatches: {mismatches}\n"
            f"Delete it (rm -rf {db_dir}) and rerun."
        )
    existing_sha = db.config.get("facap_commit_sha")
    current_sha = _facap_commit_sha(DEFAULT_FACAP_ROOT)
    if existing_sha and existing_sha != current_sha:
        raise RuntimeError(
            f"caption DB at {db_dir} was built against FACap commit "
            f"{existing_sha[:8]} but local checkout is now at "
            f"{current_sha[:8]}. Delete it (rm -rf {db_dir}) and rerun."
        )
    return db


def run(
    vlm: str,
    n_eval: int,
    run_name: str,
    category: str,
    split: str,
    encoder_name: str,
    out_root: Path,
    image_cache: Path,
    *,
    db_size: int | None = None,
    seed: int = 42,
) -> None:
    # 1. Dataset + eval slice (last n_eval triplets)
    ds = FacapDataset(category=category, split=split)
    n_total = len(ds)
    if n_eval > n_total:
        raise ValueError(f"n_eval {n_eval} > total triplets {n_total}")
    eval_indices = list(range(n_total - n_eval, n_total))

    # 2. Caption DB — full mode by default; subset mode if --db-size given.
    db_dir = caption_db_dir(
        out_root, encoder_name,
        db_size=db_size,
        eval_n=n_eval if db_size is not None else None,
        seed=seed if db_size is not None else None,
    )
    db = ensure_caption_db(
        db_dir=db_dir,
        category=category,
        split=split,
        encoder_name=encoder_name,
        db_size=db_size,
        eval_n=n_eval if db_size is not None else None,
        seed=seed if db_size is not None else None,
    )
    db_id_set = set(db.target_ids)
    eval_targets_in_db = sum(1 for i in eval_indices if ds[i]["target_id"] in db_id_set)
    print(f"[caption_db] {len(db.target_ids)} rows; "
          f"{eval_targets_in_db}/{n_eval} eval targets present in DB")

    # 3. VLM captioner
    captioner = make_captioner(vlm, image_cache_root=image_cache)

    # 4. Text encoder for the query side
    encoder = TextEncoder(model_name=encoder_name)

    # 5. Eval loop
    qualitative_rows: list[dict] = []
    ranks: list[int | None] = []
    for idx in tqdm(eval_indices, desc=f"eval ({vlm})"):
        item = ds[idx]
        generated = captioner.caption(item)
        q_emb = encoder.encode([generated])[0]
        topk = top_k(q_emb, db, k=TOP_K_QUALITATIVE)
        r = rank_of(item["target_id"], q_emb, db)
        ranks.append(r)
        qualitative_rows.append({
            "query_idx": idx,
            "query_id": item["candidate_id"],
            "true_target": item["target_id"],
            "modification_text": item["modification_text"],
            "generated_caption": generated,
            "top10_predicted": [t for t, _ in topk],
            "top10_scores": [round(s, 4) for _, s in topk],
            "rank": r,
            "failure_category": "",
        })

    # 6. Metrics + outputs
    result = compute_metrics(ranks)
    print(f"\n[{vlm}] metrics on {n_eval} queries:")
    print(format_metrics_table(result))

    run_dir = out_root / run_name
    qual_path = write_qualitative(qualitative_rows, run_dir)
    metrics_path = write_metrics(result, run_dir, extra={
        "vlm": vlm,
        "n_eval": n_eval,
        "category": category,
        "split": split,
        "encoder": encoder_name,
        "db_path": str(db_dir),
        "db_mode": "full" if db_size is None else "subset",
        "facap_commit_sha": db.config.get("facap_commit_sha"),
    })
    print(f"\nwrote qualitative -> {qual_path}")
    print(f"wrote metrics    -> {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the caption-generation retrieval baseline")
    parser.add_argument("--vlm", default="speechqwen2vl",
                        choices=["mock", "oracle", "qwen2vl", "speechqwen2vl"])
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--category", default="dress")
    parser.add_argument("--split", default="train")
    parser.add_argument("--encoder", default=DEFAULT_MODEL)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--image-cache", default=str(DEFAULT_IMAGE_CACHE))
    parser.add_argument("--db-size", type=int, default=None,
                        help="if set, use subset DB of this size for fast debugging "
                             "(default: full DB encoding all captions)")
    parser.add_argument("--seed", type=int, default=42,
                        help="subset mode only: seed for distractor sampling")
    args = parser.parse_args()
    run(
        vlm=args.vlm,
        n_eval=args.n_eval,
        run_name=args.run_name,
        category=args.category,
        split=args.split,
        encoder_name=args.encoder,
        out_root=Path(args.out_root),
        image_cache=Path(args.image_cache),
        db_size=args.db_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
