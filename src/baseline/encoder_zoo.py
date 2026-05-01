"""Registry of text encoders we ablate, with the per-model quirks each one needs.

Most modern retrievers expect a tiny prefix on queries vs passages (BGE, E5),
or longer instruction prompts (Qwen3-Embedding, NV-Embed). Loading them all
through one interface (`sentence-transformers`) only works if we apply the
right prompt for each.

Each entry says how to load + encode the model. Anything not in the registry
falls back to the plain sentence-transformers default with no prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncoderConfig:
    """How to load + encode a specific text encoder.

    Fields:
        hf_model_id      HuggingFace model ID
        query_prefix     prepended to the VLM-generated caption (the "query")
        passage_prefix   prepended to FACap target captions (the "passage")
        max_seq_length   override max input length (None = use model default)
        trust_remote_code  pass to from_pretrained
        bf16             load weights at bfloat16 (needed for 4B+ on a 24GB 3090)
        loader           "sentence_transformers" (default) or "open_clip" for
                         Marqo Fashion CLIP/SigLIP (sentence-transformers'
                         loader hits meta-tensor errors on those)
        notes            free-form description shown in logs / docs
    """
    hf_model_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int | None = None
    trust_remote_code: bool = False
    bf16: bool = False
    loader: str = "sentence_transformers"
    notes: str = ""


# 12-encoder ablation lineup, ordered by era + size.
# Slug (key) is a short human-friendly name used in logs and run_name.
ENCODER_ZOO: dict[str, EncoderConfig] = {
    # ----- existing anchor (already evaluated; just for completeness) -----
    "minilm-l6": EncoderConfig(
        hf_model_id="sentence-transformers/all-MiniLM-L6-v2",
        notes="2022 SBERT classic, 22M params, 384d. Anchor (already in baseline_v1_speechqwen2vl).",
    ),

    # ----- classic SBERT mid-tier -----
    "mpnet-base": EncoderConfig(
        hf_model_id="sentence-transformers/all-mpnet-base-v2",
        notes="2021 SBERT, 109M, 768d. Mid-size SBERT classic.",
    ),

    # ----- 2023 BERT retrievers -----
    "bge-large": EncoderConfig(
        hf_model_id="BAAI/bge-large-en-v1.5",
        # BGE was trained with a specific query prefix; passage side is plain.
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="2023 SOTA BERT retriever, 335M, 1024d.",
    ),
    "e5-large-v2": EncoderConfig(
        hf_model_id="intfloat/e5-large-v2",
        # E5 was trained with explicit query: / passage: tags on both sides.
        query_prefix="query: ",
        passage_prefix="passage: ",
        notes="2023 BERT retriever, 335M, 1024d. Different training recipe vs BGE.",
    ),

    # ----- 2025 architecture upgrade -----
    "gte-modernbert-base": EncoderConfig(
        hf_model_id="Alibaba-NLP/gte-modernbert-base",
        trust_remote_code=True,
        notes="2025 ModernBERT-based retriever, 149M, 768d. New architecture, fast.",
    ),

    # ----- 2025 LLM-based encoders (Qwen3-Embedding family) -----
    # All three sizes, see scaling.
    "qwen3-emb-0.6b": EncoderConfig(
        hf_model_id="Qwen/Qwen3-Embedding-0.6B",
        trust_remote_code=True,
        notes="2025 LLM-based encoder, 0.6B. Smallest of Qwen3-Embedding family.",
    ),
    "qwen3-emb-4b": EncoderConfig(
        hf_model_id="Qwen/Qwen3-Embedding-4B",
        trust_remote_code=True,
        bf16=True,  # 4B*4B=16GB at fp32 + activations OOMs on 24GB; bf16 fits
        notes="2025 LLM-based encoder, 4B.",
    ),
    "qwen3-emb-8b": EncoderConfig(
        hf_model_id="Qwen/Qwen3-Embedding-8B",
        trust_remote_code=True,
        bf16=True,  # 8B*4B=32GB at fp32 doesn't fit; bf16 ~16GB does
        notes="2025 SOTA LLM-based encoder, 8B. Top of MTEB at release.",
    ),

    # ----- 2024 Mistral-based SOTA (peer to Qwen3-8B) -----
    "nv-embed-v2": EncoderConfig(
        hf_model_id="nvidia/NV-Embed-v2",
        trust_remote_code=True,
        bf16=True,  # 7B; same OOM concern as Qwen3-8B
        notes="2024 LLM-based encoder, 7B. Mistral backbone. Compares with Qwen3-8B.",
    ),

    # ----- CLIP / SigLIP architectural reference -----
    "clip-vit-l-14": EncoderConfig(
        hf_model_id="sentence-transformers/clip-ViT-L-14",
        max_seq_length=77,
        notes="2021 CLIP text tower, 123M. 77-token hard limit — FACap captions are truncated.",
    ),
    "marqo-fashionclip": EncoderConfig(
        hf_model_id="hf-hub:Marqo/marqo-fashionCLIP",
        max_seq_length=77,
        loader="open_clip",
        notes="2024 fashion-domain CLIP, ~150M. 77-token hard limit. Loaded via open_clip.",
    ),
    "marqo-fashionsiglip": EncoderConfig(
        hf_model_id="hf-hub:Marqo/marqo-fashionSigLIP",
        max_seq_length=64,
        loader="open_clip",
        notes="2024 fashion-domain SigLIP, ~150M. 64-token hard limit. Loaded via open_clip.",
    ),
}


def get(slug: str) -> EncoderConfig:
    if slug not in ENCODER_ZOO:
        raise KeyError(f"unknown encoder slug {slug!r}; known: {sorted(ENCODER_ZOO)}")
    return ENCODER_ZOO[slug]


def all_slugs_to_run() -> list[str]:
    """All slugs except the anchor (which was already evaluated)."""
    return [s for s in ENCODER_ZOO if s != "minilm-l6"]
