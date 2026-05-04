"""Online eval for Plan-5 contrastive training.

Provides:
  run_retrieval_eval  — evaluate a list of items against the image gallery
  run_dev_probe       — 3-way sensitivity probe (normal / stripped / shuffled)
  harness_sanity      — Verification #4: same image as query and gallery → R@1 ≈ 1.0
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from src.baseline.eval import EvalResult, compute_metrics
from src.baseline.retrieve import CaptionDB, rank_of
from src.data.facap_dataset import FacapDataset


def make_gallery_db(embeddings: np.ndarray, gallery_ids: list[str]) -> CaptionDB:
    """Wrap image-tower cache as a CaptionDB for reuse of Phase-A retrieval code."""
    return CaptionDB(
        embeddings=embeddings,
        target_ids=gallery_ids,
        image_paths=[""] * len(gallery_ids),
        config={"source": "image_tower_cache"},
    )


@torch.inference_mode()
def run_retrieval_eval(
    model: torch.nn.Module,
    items: list[dict],
    gallery_db: CaptionDB,
    base_ds: FacapDataset,
    mod_texts: list[str] | None = None,
    batch_size: int = 16,
) -> EvalResult:
    """Evaluate items against gallery_db.

    mod_texts: if None, uses item["modification_text"] (normal mode).
               Pass [""] * N for mod-stripped, or a pre-shuffled list for mod-shuffled.
    """
    was_training = model.training
    model.eval()

    ranks: list[int | None] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        images = [base_ds.load_image(it, "candidate") for it in chunk]
        if mod_texts is not None:
            texts = mod_texts[i:i + batch_size]
        else:
            texts = [it["modification_text"] for it in chunk]

        q_embs = model(images, texts).cpu().float().numpy()  # (B, D)
        for j, it in enumerate(chunk):
            ranks.append(rank_of(it["target_id"], q_embs[j], gallery_db))

    if was_training:
        model.train()
    return compute_metrics(ranks)


def run_dev_probe(
    model: torch.nn.Module,
    dev_items: list[dict],
    gallery_db: CaptionDB,
    base_ds: FacapDataset,
    train_mod_texts: list[str],
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, EvalResult]:
    """3-way sensitivity probe on the dev slice.

    Returns dict with keys 'normal', 'mod_stripped', 'mod_shuffled'.
    The gap R@10(normal) − R@10(mod_stripped) must be > 0 by 0.25 epoch
    (warning if violated — see Plan-5 §7).
    """
    rng = np.random.RandomState(seed)
    # Draw unique shuffled mods from the training pool
    pool = np.array(train_mod_texts)
    idx = rng.choice(len(pool), size=len(dev_items), replace=len(pool) < len(dev_items))
    shuffled_mods = pool[idx].tolist()
    stripped_mods = [""] * len(dev_items)

    return {
        "normal":       run_retrieval_eval(model, dev_items, gallery_db, base_ds,
                                           None, batch_size),
        "mod_stripped": run_retrieval_eval(model, dev_items, gallery_db, base_ds,
                                           stripped_mods, batch_size),
        "mod_shuffled": run_retrieval_eval(model, dev_items, gallery_db, base_ds,
                                           shuffled_mods, batch_size),
    }


@torch.inference_mode()
def run_dev_loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dev_items: list[dict],
    gallery_lookup: dict[str, np.ndarray],
    tid_to_idx: dict[str, int],
    base_ds: "FacapDataset",
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Multi-positive InfoNCE on the full dev set. Primary overfitting signal.

    Encodes all dev queries in batches, then computes loss over the full set
    in one shot (no cross-GPU gather — single-process eval on a fixed held-out set).
    Returns scalar loss value.
    """
    was_training = model.training
    model.eval()

    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []
    all_tids: list[torch.Tensor] = []

    for i in range(0, len(dev_items), batch_size):
        chunk = dev_items[i:i + batch_size]
        images = [base_ds.load_image(it, "candidate") for it in chunk]
        texts = [it["modification_text"] for it in chunk]
        q_embs = model(images, texts).cpu().float()
        t_embs = torch.stack(
            [torch.from_numpy(gallery_lookup[it["target_id"]]) for it in chunk]
        ).float()
        tids = torch.tensor(
            [tid_to_idx[it["target_id"]] for it in chunk], dtype=torch.int64
        )
        all_q.append(q_embs)
        all_t.append(t_embs)
        all_tids.append(tids)

    q = torch.cat(all_q, dim=0).to(device)
    t = torch.cat(all_t, dim=0).to(device)
    tids_tensor = torch.cat(all_tids, dim=0).to(device)

    loss = loss_fn(q, t, gather=False, target_ids=tids_tensor)

    if was_training:
        model.train()
    return loss.item()


def harness_sanity(
    gallery_db: CaptionDB,
    n_samples: int = 20,
    expected_r1: float = 0.99,
) -> bool:
    """Verification #4 (Plan-5 §Verification): prove top_k / rank_of plumbing is correct.

    Uses the first n_samples gallery embeddings as their own queries against a
    mini-gallery. Each embedding should retrieve itself at rank 1 (dot-product of
    an L2-normalized vector with itself is the maximum possible score).
    This test is independent of the VLM — a failure here means the numpy matmul /
    argpartition / rank_of code is broken, not the model.
    """
    n = min(n_samples, len(gallery_db.target_ids))
    mini_db = CaptionDB(
        embeddings=gallery_db.embeddings[:n],
        target_ids=gallery_db.target_ids[:n],
        image_paths=gallery_db.image_paths[:n],
        config=gallery_db.config,
    )

    ranks: list[int | None] = []
    for i in range(n):
        emb = mini_db.embeddings[i]          # (D,) already L2-normalized
        r = rank_of(mini_db.target_ids[i], emb, mini_db)
        ranks.append(r)

    result = compute_metrics(ranks)
    r1 = result.recall.get(1, 0.0)
    passed = r1 >= expected_r1
    print(
        f"[harness_sanity] n={n}  R@1={r1:.3f}  "
        f"{'PASS' if passed else 'FAIL (check top_k / rank_of plumbing)'}"
    )
    return passed
