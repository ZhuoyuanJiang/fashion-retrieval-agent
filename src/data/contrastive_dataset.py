"""Plan-5 contrastive training dataset.

Wraps FacapDataset with:
- L2 train filtering (excludes triplets whose target/candidate IDs appear
  in the headline eval slice or dev slice)
- Deterministic dev slice carving (~500 queries for online R@K every 500 steps)
- PIL image loading in __getitem__ (one image per item, always needed in training)

Usage:
    base = FacapDataset(category="dress", split="train")
    ds = FacapContrastiveDataset(base, dev_seed=42)
    # ds wraps ~55k training triplets; ds.dev_items / ds.headline_items for eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.facap_dataset import FacapDataset

HEADLINE_SLICE_SIZE = 1000  # last N triplets of train split (Plan-3's eval slice)
DEV_SLICE_SIZE = 500        # carved from post-L2 pool for online dev eval


class FacapContrastiveDataset(Dataset):
    """Training triplets for Plan-5 / Plan-10 contrastive fine-tuning.

    __getitem__ returns a dict:
        cand_image:  PIL.Image   candidate (reference) image
        mod_text:    str         modification instruction
        target_id:   str         ID of the correct target image
        tgt_image:   PIL.Image   target image  (Plan-10 only; loaded when
                                  `load_target=True`. Plan-5/6/7 ignore.)

    Plan-5/6/7 callers leave `load_target=False` (default) — they look up
    target embeddings from a precomputed FashionCLIP cache by `target_id`.
    Plan-10 sets `load_target=True` so the target tower can encode the
    target image on the fly each batch.

    Attributes:
        dev_items:      list[dict]  FacapDataset item dicts for the dev slice
        headline_items: list[dict]  FacapDataset item dicts for the headline slice
        exclusion_ids:  set[str]    all IDs excluded from training (headline ∪ dev)
        train_mod_texts: list[str]  modification texts for training items (for shuffled probe)
    """

    def __init__(
        self,
        base: FacapDataset,
        dev_seed: int = 42,
        dev_slice_json: Path | str | None = None,
        load_target: bool = False,
        query_modality: str = "text",
        audio_manifest: Path | str | None = None,
    ) -> None:
        self.base = base
        self.dev_seed = dev_seed
        self.load_target = load_target
        if query_modality not in ("text", "audio"):
            raise ValueError(
                f"query_modality must be 'text' or 'audio', got {query_modality!r}"
            )
        self.query_modality = query_modality

        N = len(base)
        if N <= HEADLINE_SLICE_SIZE + DEV_SLICE_SIZE:
            raise ValueError(
                f"Dataset too small ({N}) for headline ({HEADLINE_SLICE_SIZE}) "
                f"+ dev ({DEV_SLICE_SIZE}) slices."
            )

        # Step 1: headline slice = last 1000 triplets
        headline_base_indices = list(range(N - HEADLINE_SLICE_SIZE, N))
        self.headline_items: list[dict[str, Any]] = [base[i] for i in headline_base_indices]
        # Plan 15 §3.1: every eval item carries its FACap triplet index so the
        # audio path can look the clip up in the manifest. Pure additive — the
        # text path ignores the key.
        for k, bi in enumerate(headline_base_indices):
            self.headline_items[k]["triplet_index"] = int(bi)
        headline_ids: set[str] = set()
        for it in self.headline_items:
            headline_ids.add(it["target_id"])
            headline_ids.add(it["candidate_id"])

        # Step 2: L2 filter — drop any non-headline triplet sharing an ID with headline
        clean_indices: list[int] = [
            i for i in range(N - HEADLINE_SLICE_SIZE)
            if base[i]["target_id"] not in headline_ids
            and base[i]["candidate_id"] not in headline_ids
        ]

        # Step 3: deterministic dev slice from clean pool
        rng = np.random.RandomState(dev_seed)
        perm = rng.permutation(len(clean_indices))
        dev_pool_positions: set[int] = set(perm[:DEV_SLICE_SIZE].tolist())
        self.dev_items: list[dict[str, Any]] = [
            base[clean_indices[p]] for p in perm[:DEV_SLICE_SIZE]
        ]
        for k, p in enumerate(perm[:DEV_SLICE_SIZE]):
            self.dev_items[k]["triplet_index"] = int(clean_indices[p])
        dev_ids: set[str] = set()
        for it in self.dev_items:
            dev_ids.add(it["target_id"])
            dev_ids.add(it["candidate_id"])

        # Step 4: training indices — clean pool minus dev positions and dev IDs
        self._train_indices: list[int] = [
            clean_indices[p]
            for p in range(len(clean_indices))
            if p not in dev_pool_positions
            and base[clean_indices[p]]["target_id"] not in dev_ids
            and base[clean_indices[p]]["candidate_id"] not in dev_ids
        ]

        self.exclusion_ids: set[str] = headline_ids | dev_ids

        # Verify no leakage
        for i in self._train_indices:
            it = base[i]
            assert it["target_id"] not in self.exclusion_ids, \
                f"leakage: {it['target_id']} in training and exclusion set"
            assert it["candidate_id"] not in self.exclusion_ids, \
                f"leakage: {it['candidate_id']} in training and exclusion set"

        # Mod texts for shuffled sensitivity probe
        self.train_mod_texts: list[str] = [
            base[i]["modification_text"] for i in self._train_indices
        ]

        # Audio-query manifest (Plan 15 §3.1). When query_modality=="audio",
        # every modification is a synthesized wav looked up by triplet index.
        self.audio_manifest: dict | None = None
        self.train_mod_audios: list[str] | None = None
        if self.query_modality == "audio":
            if audio_manifest is None:
                raise ValueError(
                    "query_modality='audio' requires audio_manifest"
                )
            with open(audio_manifest) as f:
                self.audio_manifest = json.load(f)
            self._assert_manifest_complete()
            in_dist = self.audio_manifest["in_dist"]
            self.train_mod_audios = [
                in_dist[str(i)]["wav"] for i in self._train_indices
            ]

        # Optionally dump dev slice for reproducibility logging
        if dev_slice_json is not None:
            dev_slice_json = Path(dev_slice_json)
            dev_slice_json.parent.mkdir(parents=True, exist_ok=True)
            with open(dev_slice_json, "w") as f:
                json.dump(
                    {
                        "dev_seed": dev_seed,
                        "dev_size": len(self.dev_items),
                        "train_size": len(self._train_indices),
                        "headline_size": len(self.headline_items),
                        "dev_target_ids": [it["target_id"] for it in self.dev_items],
                    },
                    f,
                    indent=2,
                )

    def _assert_manifest_complete(self) -> None:
        """Plan 15 §3.1 — fail fast at construction if the audio manifest does
        not cover every train/dev/headline triplet with matching split labels.

        The `--query-modality text` control never exercises the audio lookup,
        so this assertion (not the text control) is what guards the indexing.
        """
        m = self.audio_manifest
        in_dist, ood = m["in_dist"], m["ood"]
        for i in self._train_indices:
            rec = in_dist.get(str(i))
            assert rec is not None, \
                f"audio manifest in_dist missing train triplet {i}"
            assert rec["split"] == "train", (
                f"manifest split mismatch for triplet {i}: "
                f"manifest={rec['split']!r}, dataset='train'"
            )
        for split_name, items in (("dev", self.dev_items),
                                  ("headline", self.headline_items)):
            for it in items:
                i = it["triplet_index"]
                for cond_name, cond in (("in_dist", in_dist), ("ood", ood)):
                    rec = cond.get(str(i))
                    assert rec is not None, (
                        f"audio manifest {cond_name} missing "
                        f"{split_name} triplet {i}"
                    )
                    assert rec["split"] == split_name, (
                        f"manifest split mismatch for triplet {i}: "
                        f"manifest={rec['split']!r}, dataset={split_name!r}"
                    )

    def __len__(self) -> int:
        return len(self._train_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self._train_indices[idx]
        item = self.base[base_idx]
        cand_image = self.base.load_image(item, "candidate")
        out: dict[str, Any] = {
            "cand_image": cand_image,
            "mod_text": item["modification_text"],
            "target_id": item["target_id"],
            "triplet_index": int(base_idx),
        }
        if self.query_modality == "audio":
            out["mod_audio"] = self.audio_manifest["in_dist"][str(base_idx)]["wav"]
        if self.load_target:
            out["tgt_image"] = self.base.load_image(item, "target")
        return out

    def summary(self) -> str:
        n_excl = len(self.exclusion_ids)
        return (
            f"FacapContrastiveDataset: {len(self)} train | "
            f"{len(self.dev_items)} dev | "
            f"{len(self.headline_items)} headline | "
            f"{n_excl} excluded IDs"
        )


def contrastive_collate(batch: list[dict]) -> dict:
    """DataLoader collate that keeps PIL images as a list (not a stacked tensor).

    The VLM processor handles batching of PIL images — don't pre-stack them.
    If items have `tgt_image` (Plan-10 mode), it's collated into `tgt_images`;
    otherwise the key is absent and Plan-5/6/7 callers won't trip on it.
    """
    out = {
        "cand_images": [item["cand_image"] for item in batch],
        "mod_texts": [item["mod_text"] for item in batch],
        "target_ids": [item["target_id"] for item in batch],
    }
    if "triplet_index" in batch[0]:
        out["triplet_indices"] = [item["triplet_index"] for item in batch]
    if "mod_audio" in batch[0]:
        out["mod_audios"] = [item["mod_audio"] for item in batch]
    if "tgt_image" in batch[0]:
        out["tgt_images"] = [item["tgt_image"] for item in batch]
    return out
