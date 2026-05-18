"""Online eval for Plan-5 / Plan-10 contrastive training.

Provides:
  run_retrieval_eval         — evaluate items against the image gallery
  run_dev_probe              — 3-way sensitivity probe (normal/stripped/shuffled)
  run_dev_loss               — multi-positive InfoNCE on dev (Plan-5 path: uses gallery_lookup)
  run_dev_loss_two_tower     — same, but encodes targets on the fly (Plan-10 path)
  encode_gallery_with_tower  — Plan-10 distributed gallery encoder; pad/gather/sort/truncate
  harness_sanity             — Verification: same image as query and gallery → R@1 ≈ 1.0
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
    query_modality: str = "text",
    mod_audios: list | None = None,
) -> EvalResult:
    """Evaluate items against gallery_db.

    Text mode (query_modality=="text"): `mod_texts` — if None, uses
    item["modification_text"] (normal); pass [""] * N for mod-stripped, or a
    pre-shuffled list for mod-shuffled.
    Audio mode (query_modality=="audio"): `mod_audios` — a per-item list of
    wav paths (a None entry = image-only / stripped). Required when audio.
    """
    was_training = model.training
    model.eval()

    ranks: list[int | None] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        images = [base_ds.load_image(it, "candidate") for it in chunk]
        if query_modality == "audio":
            assert mod_audios is not None, "audio eval requires mod_audios"
            mods = mod_audios[i:i + batch_size]
        elif mod_texts is not None:
            mods = mod_texts[i:i + batch_size]
        else:
            mods = [it["modification_text"] for it in chunk]

        # encode_query routes to the query tower; it takes text strings or
        # (audio mode) wav paths per the model's query_modality.
        q_embs = model.encode_query(images, mods).cpu().float().numpy()  # (B, D)
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
    query_modality: str = "text",
    dev_audios: list[str] | None = None,
    train_mod_audios: list[str] | None = None,
) -> dict[str, EvalResult]:
    """3-way sensitivity probe on the dev slice.

    Returns dict with keys 'normal', 'mod_stripped', 'mod_shuffled'.
    The gap R@10(normal) − R@10(mod_stripped) must be > 0 by 0.25 epoch
    (warning if violated — see Plan-5 §7).

    Audio mode (Plan 15 §3.4): normal = each dev item's real clip
    (`dev_audios`); stripped = image-only (no audio); shuffled = another
    item's clip drawn from `train_mod_audios`.
    """
    rng = np.random.RandomState(seed)

    if query_modality == "audio":
        assert dev_audios is not None and train_mod_audios is not None, \
            "audio dev probe requires dev_audios + train_mod_audios"
        pool = np.array(train_mod_audios)
        idx = rng.choice(len(pool), size=len(dev_items),
                         replace=len(pool) < len(dev_items))
        shuffled = pool[idx].tolist()
        stripped = [None] * len(dev_items)
        return {
            "normal":       run_retrieval_eval(
                model, dev_items, gallery_db, base_ds, batch_size=batch_size,
                query_modality="audio", mod_audios=dev_audios),
            "mod_stripped": run_retrieval_eval(
                model, dev_items, gallery_db, base_ds, batch_size=batch_size,
                query_modality="audio", mod_audios=stripped),
            "mod_shuffled": run_retrieval_eval(
                model, dev_items, gallery_db, base_ds, batch_size=batch_size,
                query_modality="audio", mod_audios=shuffled),
        }

    # Text mode — draw unique shuffled mods from the training pool
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


# =====================================================================
# Plan-10 V1: two-tower helpers (target tower is trainable, no cache)
# =====================================================================

@torch.inference_mode()
def run_dev_loss_two_tower(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dev_items: list[dict],
    tid_to_idx: dict[str, int],
    base_ds: FacapDataset,
    device: torch.device,
    batch_size: int = 32,
    query_modality: str = "text",
    dev_audios: list[str] | None = None,
) -> float:
    """Plan-10 multi-positive InfoNCE on the full dev set.

    Mirrors `run_dev_loss` but encodes target embeddings on the fly via
    the model's trainable `encode_target(images)` helper (instead of
    looking them up in a precomputed gallery_lookup dict — Plan-5/6/7
    pattern that's no longer correct when the target tower is being
    trained).

    `model` must be the UNWRAPPED two-tower module
    (`accelerator.unwrap_model(prepared_model)`); it must expose
    `encode_query(images, texts)` and `encode_target(images)`.

    Single-process eval (no cross-GPU gather). Returns scalar loss.
    """
    was_training = model.training
    model.eval()

    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []
    all_tids: list[torch.Tensor] = []

    for i in range(0, len(dev_items), batch_size):
        chunk = dev_items[i:i + batch_size]
        cand_images = [base_ds.load_image(it, "candidate") for it in chunk]
        tgt_images = [base_ds.load_image(it, "target") for it in chunk]
        if query_modality == "audio":
            assert dev_audios is not None, "audio dev loss requires dev_audios"
            mods = dev_audios[i:i + batch_size]
        else:
            mods = [it["modification_text"] for it in chunk]

        q_embs = model.encode_query(cand_images, mods).cpu().float()
        t_embs = model.encode_target(tgt_images).cpu().float()
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


def _ids_hash_local(gallery_ids: list[str]) -> str:
    """Mirror of target_cache._ids_hash for self-consistency checks."""
    h = hashlib.md5()
    for gid in gallery_ids:
        h.update(gid.encode())
    return h.hexdigest()[:12]


@torch.inference_mode()
def encode_gallery_with_tower(
    model: torch.nn.Module,
    gallery_ids: list[str],
    gallery_paths: list[Path],
    batch_size: int,
    accelerator,
    out_dir: Path | None = None,
    epoch_tag: str | int = 0,
) -> np.ndarray:
    """Plan-10 distributed gallery encoder.

    Encodes the full gallery through the target tower across all ranks.
    Pads to ceil(N/world) so `accelerator.gather` works (N=59,082 is
    not divisible by 8). Gathers paired (embedding, original_index)
    tensors, re-sorts by index, truncates the padding rows.

    Must be called from ALL ranks (do NOT gate on `is_main_process`).

    Args:
      model: UNWRAPPED two-tower module, must expose
        `encode_target(images) -> (B, D)`.
      gallery_ids, gallery_paths: lists of length N in canonical row
        order. Use `src/training/target_cache._gallery_ids_and_paths`
        to build them.
      batch_size: per-rank chunk size for target_tower forward.
      accelerator: the Accelerator object (for gather + sync).
      out_dir: if given, rank 0 writes
        `<out_dir>/gallery_emb_epoch<epoch_tag>.{npy,meta.json}`.
      epoch_tag: tag stamped into the cache filename.

    Returns:
      full: (N, D) float32 L2-normalized — same on every rank.
    """
    was_training = model.training
    model.eval()

    rank = accelerator.process_index
    world = accelerator.num_processes
    N = len(gallery_ids)
    assert len(gallery_paths) == N, (
        f"gallery_ids ({N}) / gallery_paths ({len(gallery_paths)}) mismatch"
    )

    # Pad gallery to ceil(N/world)*world so every rank's shard has the
    # SAME length — required by accelerator.gather. Padding entries
    # duplicate the last item; they're truncated after the gather.
    pad_to = math.ceil(N / world) * world
    pad = pad_to - N
    padded_ids = gallery_ids + ([gallery_ids[-1]] * pad)
    padded_paths = gallery_paths + ([gallery_paths[-1]] * pad)

    my_idx = list(range(rank, pad_to, world))            # equal length per rank
    my_paths = [padded_paths[i] for i in my_idx]
    my_orig_idx = torch.tensor(my_idx, dtype=torch.long, device=accelerator.device)

    my_embs: list[torch.Tensor] = []
    for c in range(0, len(my_paths), batch_size):
        chunk_paths = my_paths[c:c + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk_paths]
        embs = model.encode_target(imgs)                  # (b, D), L2-normalized fp32
        my_embs.append(embs.to(accelerator.device).float())
    my_embs_t = torch.cat(my_embs, dim=0)                  # (pad_to/world, D)

    # Gather paired (embeddings, original index) across ranks.
    all_embs = accelerator.gather(my_embs_t)               # (pad_to, D)
    all_idx = accelerator.gather(my_orig_idx)              # (pad_to,)

    # Re-sort to canonical order; padding indices were valid positions
    # but pointed at duplicated rows, so after argsort we just take the
    # first N rows (positions [0, N)). Padding rows occupy positions
    # [N, pad_to) but their ids are valid (last id reused) — argsort is
    # not unique on duplicate ids; defensively truncate by index value
    # instead.
    perm = torch.argsort(all_idx)
    full_padded = all_embs[perm]                            # (pad_to, D)
    full = full_padded[:N].contiguous().cpu().numpy().astype(np.float32)

    accelerator.wait_for_everyone()

    if out_dir is not None and accelerator.is_main_process:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"gallery_emb_epoch{epoch_tag}.npy"
        meta_path = out_dir / f"gallery_emb_epoch{epoch_tag}.meta.json"
        np.save(npy_path, full)
        with open(meta_path, "w") as f:
            json.dump({
                "encoder_id": "plan10_two_tower_target",
                "embedding_dim": full.shape[1],
                "n_images": N,
                "gallery_ids": gallery_ids,
                "image_hash": _ids_hash_local(gallery_ids),
                "epoch_tag": str(epoch_tag),
            }, f)
        print(
            f"[encode_gallery_with_tower] wrote {npy_path} "
            f"({full.shape}, {full.nbytes / 1e6:.1f} MB)"
        )

    if was_training:
        model.train()
    return full

