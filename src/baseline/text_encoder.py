"""Sentence-BERT text encoder used by both the caption DB and the
generated-caption query side. Single source of truth: same model, same
normalization, on both sides of cosine similarity.

See `Documentation/Plan_2_20260427.md` -> "Encoder choice" for why
all-MiniLM-L6-v2 is the baseline pick and when to swap.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"  # text encoder must run on CPU; VRAM is reserved for the VLM
EMBED_DIM = 384


class TextEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE) -> None:
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Return float32 array of shape (N, EMBED_DIM), L2-normalized.

        Normalizing on both sides means cosine similarity == dot product,
        so retrieval can use a plain matmul.
        """
        texts = list(texts)
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)
