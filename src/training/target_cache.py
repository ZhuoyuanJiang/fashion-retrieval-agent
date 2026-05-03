"""Plan-5 image-tower embedding cache for the retrieval gallery.

Builds and loads a frozen FashionCLIP-image (or any open_clip) embedding
cache for the full ~58k FACap gallery. Pre-computed once; reused at every
training startup and for eval.

Cache files (in out_dir):
  target_emb_cache_<encoder_id>.npy       float32, (N, D), L2-normalized
  target_emb_cache_<encoder_id>.meta.json {encoder_id, embedding_dim,
                                            gallery_ids, n_images, image_hash}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.facap_dataset import FacapDataset, _path_to_image_id


def _gallery_ids_and_paths(ds: FacapDataset) -> list[tuple[str, Path]]:
    """All unique (image_id, local_path) pairs from the dataset's caption dict.

    The caption dict covers all images in the split (candidates and targets),
    making this the full ~58k gallery used for retrieval eval.
    """
    image_cache = ds.image_cache
    seen: dict[str, Path] = {}
    for facap_path in ds.captions.keys():
        image_id = _path_to_image_id(facap_path)
        if image_id not in seen:
            seen[image_id] = image_cache / f"{image_id}.jpeg"
    return list(seen.items())


def _ids_hash(gallery_ids: list[str]) -> str:
    h = hashlib.md5()
    for gid in gallery_ids:
        h.update(gid.encode())
    return h.hexdigest()[:12]


def build_target_cache(
    ds: FacapDataset,
    encoder_id: str,
    out_dir: Path | str,
    batch_size: int = 64,
    device: str = "cuda",
) -> Path:
    """Encode all gallery images and write cache files.

    Returns path to the .npy file.
    Only needs to run once per encoder; subsequent startups just call
    load_target_cache().
    """
    from src.baseline.encoder_zoo import ENCODER_ZOO as ENCODERS
    from src.baseline.replay_with_encoder import _load_open_clip

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ENCODERS[encoder_id]
    wrapper = _load_open_clip(cfg)
    wrapper.model = wrapper.model.to(device)

    pairs = _gallery_ids_and_paths(ds)
    gallery_ids = [gid for gid, _ in pairs]
    paths = [p for _, p in pairs]

    # Verify images are accessible (spot-check first few)
    missing = [p for p in paths[:10] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Gallery images not in cache: {missing[:3]}... "
            f"Run src/baseline/prepare_images.py first."
        )

    print(f"Building target cache for '{encoder_id}' over {len(gallery_ids)} images...")
    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="encoding gallery"):
        chunk_paths = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk_paths]
        embs = wrapper.encode_image(imgs, batch_size=len(imgs))
        all_embs.append(embs)

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)  # (N, D)
    embedding_dim = embeddings.shape[1]

    npy_path = out_dir / f"target_emb_cache_{encoder_id}.npy"
    meta_path = out_dir / f"target_emb_cache_{encoder_id}.meta.json"

    np.save(npy_path, embeddings)
    meta = {
        "encoder_id": encoder_id,
        "embedding_dim": embedding_dim,
        "n_images": len(gallery_ids),
        "gallery_ids": gallery_ids,
        "image_hash": _ids_hash(gallery_ids),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"Saved: {npy_path}  ({embeddings.shape}, {embeddings.nbytes / 1e6:.1f} MB)")
    return npy_path


def load_target_cache(
    out_dir: Path | str,
    encoder_id: str,
) -> tuple[np.ndarray, list[str], int]:
    """Load a pre-built cache.

    Returns:
        embeddings: (N, D) float32 L2-normalized
        gallery_ids: list of N image IDs in row order
        embedding_dim: D
    """
    out_dir = Path(out_dir)
    npy_path = out_dir / f"target_emb_cache_{encoder_id}.npy"
    meta_path = out_dir / f"target_emb_cache_{encoder_id}.meta.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"Target cache not found: {npy_path}\n"
            f"Run: python -m src.training.target_cache --ds-split train "
            f"--encoder-id {encoder_id} --out-dir {out_dir}"
        )

    embeddings = np.load(npy_path)
    with open(meta_path) as f:
        meta = json.load(f)

    gallery_ids: list[str] = meta["gallery_ids"]
    embedding_dim: int = meta["embedding_dim"]

    assert embeddings.shape == (len(gallery_ids), embedding_dim), \
        f"Cache shape mismatch: {embeddings.shape} vs ({len(gallery_ids)}, {embedding_dim})"

    return embeddings, gallery_ids, embedding_dim


def make_gallery_lookup(
    embeddings: np.ndarray,
    gallery_ids: list[str],
) -> dict[str, np.ndarray]:
    """Build a dict[target_id -> embedding (D,)] for fast per-sample lookup during training."""
    return {gid: embeddings[i] for i, gid in enumerate(gallery_ids)}


if __name__ == "__main__":
    import argparse
    from src.data.facap_dataset import FacapDataset

    parser = argparse.ArgumentParser(description="Build target embedding cache")
    parser.add_argument("--ds-split", default="train")
    parser.add_argument("--encoder-id", default="marqo-fashionclip")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/plan5"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ds = FacapDataset(split=args.ds_split)
    build_target_cache(ds, args.encoder_id, args.out_dir, args.batch_size)
