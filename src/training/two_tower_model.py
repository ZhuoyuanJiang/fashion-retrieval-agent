"""Plan-10 V1 (Option B): two-tower contrastive retrieval model.

Architecture: two independent ContrastiveQwen2VL instances.

  - query_tower: reuses ContrastiveQwen2VL.forward verbatim — wraps
    `mod_text` in Plan-6's modification-instruction template
    ("Given this product image, find the item that looks like the
    image but with the following modification: {txt}").

  - target_tower: a ContrastiveQwen2VL instance whose `.forward` is
    rebound (via `__get__`) to use the locked Plan-10 target prompt
    ("Describe this image in detail.") regardless of input text.
    This avoids passing the target prompt through Plan-6's wrap.

Parent `forward(cand_images, mod_texts, tgt_images) -> (q_emb, t_emb)`
is the DDP-safe entry point used inside the training loop after
`accelerator.prepare(model)`. Eval helpers `encode_query` /
`encode_target` are `@torch.inference_mode` and should be called on
the **unwrapped** model (`accelerator.unwrap_model(...)`).

See `Documentation/Plan_10_20260510.md` (Appendix B Option B) for the
design discussion and trade-offs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.training.contrastive_model import ContrastiveQwen2VL


# Locked V1 target prompt (per Plan-10 §4.2). Alternates are V3.
TARGET_PROMPT = "Describe this image in detail."


def _target_forward(
    self,
    images: list[Image.Image],
    _ignored_texts: list[str] | None = None,
    max_mod_len: int = 512,
) -> torch.Tensor:
    """Bound replacement for ContrastiveQwen2VL.forward on the target tower.

    Same pooling / projection / L2-normalize as the parent class
    (see src/training/contrastive_model.py:115-181). The only
    difference: every sample is wrapped with TARGET_PROMPT instead of
    Plan-6's "Given this product image, find the item..." template.

    Bound via `target_tower.forward = _target_forward.__get__(target_tower, ...)`
    inside `TwoTowerSeparateBackbones.__init__`.
    """
    B = len(images)
    messages_list = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": TARGET_PROMPT},
        ],
    }] for img in images]

    text_inputs = [
        self.processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages_list
    ]

    device = next(self.vlm.parameters()).device
    inputs = self.processor(
        text=text_inputs,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    outputs = self.vlm(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    last_hs = outputs.hidden_states[-1]  # (B, seq_len, 3584), bf16
    seq_len = inputs["attention_mask"].shape[1]
    seq_ends = (
        seq_len - 1
        - inputs["attention_mask"].flip(dims=[1]).long().argmax(dim=1)
    )  # (B,)
    pooled = last_hs[torch.arange(B, device=device), seq_ends, :]

    emb = self.proj(pooled.float())
    emb = F.normalize(emb, dim=-1)
    return emb


class TwoTowerSeparateBackbones(nn.Module):
    """Plan-10 V1 Option B model wrapper.

    Two independent ContrastiveQwen2VL instances. No PEFT
    adapter-switching, no requires_grad footgun — each tower's LoRA
    lives in a separate PeftModel and stays trainable throughout.

    DDP-safe public API:
      - `forward(cand_images, mod_texts, tgt_images) -> (q_emb, t_emb)`
        is the training entry. Call as `model(cand, mod, tgt)` after
        `accelerator.prepare(model)`.
      - `encode_query(images, texts)` and `encode_target(images)` are
        `@torch.inference_mode` eval helpers — call on the unwrapped
        model only.
    """

    def __init__(self, d_target: int, device_map: str = "cuda:0") -> None:
        super().__init__()
        self.d_target = d_target

        self.query_tower = ContrastiveQwen2VL(
            d_target=d_target, device_map=device_map
        )
        self.target_tower = ContrastiveQwen2VL(
            d_target=d_target, device_map=device_map
        )

        # Override target tower's forward to use the locked target prompt.
        # See `_target_forward` above for the replacement implementation.
        self.target_tower.forward = _target_forward.__get__(
            self.target_tower, type(self.target_tower)
        )

    def forward(
        self,
        cand_images: list[Image.Image],
        mod_texts: list[str],
        tgt_images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDP-safe training entry point.

        Returns (q_emb, t_emb), both (B, D) fp32 L2-normalized.
        """
        q_emb = self.query_tower(cand_images, mod_texts)
        t_emb = self.target_tower(tgt_images)
        return q_emb, t_emb

    @torch.inference_mode()
    def encode_query(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> torch.Tensor:
        """Eval-only query encoder. Call on the unwrapped model."""
        return self.query_tower(images, texts)

    @torch.inference_mode()
    def encode_target(self, images: list[Image.Image]) -> torch.Tensor:
        """Eval-only target encoder. Uses locked TARGET_PROMPT internally.
        Call on the unwrapped model."""
        return self.target_tower(images)

    def trainable_parameters(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter],
               list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Returns (q_lora, q_proj, t_lora, t_proj) for optimizer construction.

        Used by train_plan10.py to build 4 LoRA/proj param groups
        (+ logit_scale = 5 groups total).
        """
        q_lora, q_proj = self.query_tower.trainable_parameters()
        t_lora, t_proj = self.target_tower.trainable_parameters()
        return q_lora, q_proj, t_lora, t_proj
