"""Top-level driver for the text-modification caption-generation baseline.

Workflow per query (one FACap triplet):
    1. VLM backend takes (reference_image, modification_text) -> generated caption.
    2. TextEncoder embeds the generated caption -> query embedding.
    3. Retrieve top-K target_ids by cosine similarity against the caption DB.
    4. Score against the true target_id; aggregate Recall@K + ranks.
    5. Save qualitative dump (failure_category to be filled in by hand later).

The smoke runs use the last `--n-eval` triplets (FACap has no formal val
split — see Plan_2 "FACap evaluation slice" note).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.baseline.build_caption_db import _facap_commit_sha, build_db, build_signature
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


def ensure_caption_db(
    db_dir: Path,
    run_name: str,
    eval_n: int,
    db_size: int,
    category: str,
    split: str,
    encoder_name: str,
    seed: int,
    out_root: Path,
) -> CaptionDB:
    """Load the caption DB at `db_dir`, building it if missing.

    If a DB exists but was built with different args (especially a
    different encoder, which would put query and DB embeddings in
    different spaces), refuse to load it — silent reuse here would
    produce metrics that look fine but are meaningless.
    """
    expected = build_signature(
        eval_n=eval_n, db_size=db_size, category=category,
        split=split, encoder_name=encoder_name, seed=seed,
    )
    if not (db_dir / "embeddings.npy").exists():
        print(f"[caption_db] not found at {db_dir}, building it...")
        build_db(
            run_name=run_name,
            eval_n=eval_n,
            db_size=db_size,
            category=category,
            split=split,
            encoder_name=encoder_name,
            seed=seed,
            out_root=out_root,
        )
    db = CaptionDB.load(db_dir)
    existing = db.config.get("build_args")
    if existing is None:
        raise RuntimeError(
            f"caption DB at {db_dir} was built by an older version of "
            f"build_caption_db.py and has no `build_args` in config.json. "
            f"Delete it (rm -rf {db_dir}) and rerun."
        )
    if existing != expected:
        mismatches = {
            k: {"existing": existing.get(k), "requested": expected[k]}
            for k in expected if existing.get(k) != expected[k]
        }
        raise RuntimeError(
            f"caption DB at {db_dir} was built with different args than "
            f"this run; silent reuse would produce meaningless metrics.\n"
            f"mismatches: {mismatches}\n"
            f"either delete it (rm -rf {db_dir}) or use a fresh --run-name."
        )
    # FACap revision check: if the local FACap clone changed since the DB was
    # built, the cached captions/triplets may differ from what's on disk now.
    existing_sha = db.config.get("facap_commit_sha")
    current_sha = _facap_commit_sha(DEFAULT_FACAP_ROOT)
    if existing_sha and existing_sha != current_sha:
        raise RuntimeError(
            f"caption DB at {db_dir} was built against FACap commit "
            f"{existing_sha[:8]} but the local FACap checkout is now at "
            f"{current_sha[:8]}; silent reuse risks running queries against "
            f"a different dataset revision than the DB was built from.\n"
            f"either delete the DB (rm -rf {db_dir}) or use a fresh --run-name."
        )
    return db


def run(
    vlm: str,
    n_eval: int,
    run_name: str,
    category: str,
    split: str,
    db_size: int,
    encoder_name: str,
    seed: int,
    out_root: Path,
    image_cache: Path,
) -> None:
    # 1. Dataset + eval slice
    ds = FacapDataset(category=category, split=split)
    n_total = len(ds)
    if n_eval > n_total:
        raise ValueError(f"n_eval {n_eval} > total triplets {n_total}")
    eval_indices = list(range(n_total - n_eval, n_total))

    # 2. Caption DB (auto-build at runs/<run_name>/caption_db if missing)
    run_dir = out_root / run_name
    db_dir = run_dir / "caption_db"
    db = ensure_caption_db(
        db_dir=db_dir,
        run_name=run_name,
        eval_n=n_eval,
        db_size=db_size,
        category=category,
        split=split,
        encoder_name=encoder_name,
        seed=seed,
        out_root=out_root,
    )
    # Sanity: every eval target should be in the DB (smoke build guarantees it).
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
            "failure_category": "",  # filled in by hand later: caption_wrong /
                                     # embedding_mismatch / dataset_ambiguity /
                                     # visual_nuance_lost / null
        })

    # 6. Metrics + outputs
    result = compute_metrics(ranks)
    print(f"\n[{vlm}] metrics on {n_eval} queries:")
    print(format_metrics_table(result))

    qual_path = write_qualitative(qualitative_rows, run_dir)
    metrics_path = write_metrics(result, run_dir, extra={
        "vlm": vlm,
        "n_eval": n_eval,
        "category": category,
        "split": split,
        "encoder": encoder_name,
        "seed": seed,
        "db_subset": db.config.get("subset_description"),
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
    parser.add_argument("--db-size", type=int, default=1000,
                        help="size of the caption DB if it has to be built (smoke default)")
    parser.add_argument("--encoder", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--image-cache", default=str(DEFAULT_IMAGE_CACHE))
    args = parser.parse_args()
    run(
        vlm=args.vlm,
        n_eval=args.n_eval,
        run_name=args.run_name,
        category=args.category,
        split=args.split,
        db_size=args.db_size,
        encoder_name=args.encoder,
        seed=args.seed,
        out_root=Path(args.out_root),
        image_cache=Path(args.image_cache),
    )


if __name__ == "__main__":
    main()
