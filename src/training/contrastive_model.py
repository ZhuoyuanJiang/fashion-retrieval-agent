"""Plan-5 query encoder: speechQwen2VL + LoRA + projection head.

Architecture:
  1. Load DanJZY/Qwen2-VL-7B-Speech base + Stage-2 LoRA
  2. merge_and_unload() to fold Stage-2 weights into base (~30s)
  3. Attach a fresh Plan-5 LoRA (rank 32, alpha 64, q/k/v/o_proj)
  4. 2-layer MLP projection head: 3584 → 1024 → D_target (fp32, GELU + LayerNorm)
  5. encode_query: EOS-position last hidden state → projection → L2-normalize

D_target is read from the target embedding cache metadata at construction
time (512 for marqo-fashionclip, etc.) — not hard-coded.

forward(images, texts) → (B, D_target) fp32 L2-normalized.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.baseline.vlm_caption import _load_qwen2vl_base

QWEN2VL_HIDDEN_DIM = 3584

BASE_REPO = "DanJZY/Qwen2-VL-7B-Speech"
LORA_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _ProjectionHead(nn.Module):
    """3584 → 1024 → D_target, GELU + LayerNorm, fp32."""

    def __init__(self, d_target: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(QWEN2VL_HIDDEN_DIM, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, d_target),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveQwen2VL(nn.Module):
    """Query encoder for Plan-5 contrastive training.

    Call forward(images, texts) or encode_query(images, texts) interchangeably.
    DDP wraps forward(), so always use model(images, texts) inside the train loop.
    """

    def __init__(self, d_target: int, device_map: str = "cuda:0") -> None:
        super().__init__()
        self.d_target = d_target

        print(f"Loading speechQwen2VL base + merging Stage-2 LoRA (~30s)...")
        vlm, processor = _load_qwen2vl_base(
            base_repo=BASE_REPO,
            lora_repo=LORA_REPO,
            merge_stage2=True,
            device_map=device_map,
        )

        # Attach Plan-5 LoRA on the merged base
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        vlm = get_peft_model(vlm, lora_cfg)
        vlm.enable_input_require_grads()   # required for grad-checkpoint + PEFT
        # use_reentrant=False avoids DDP bucket mismatch when some LoRA params
        # don't receive grads in the reentrant checkpointing graph.
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Disable caching (incompatible with gradient checkpointing)
        vlm.config.use_cache = False

        # We only use hidden_states[-1], never logits — replace lm_head with a 1-output
        # stub to avoid allocating (B, S, 152064) bf16 tensors (~2 GB at bs=16, seq=512).
        # lm_head is not in LoRA target modules so no trainable params are affected.
        _lm_head = vlm.base_model.model.lm_head
        vlm.base_model.model.lm_head = nn.Linear(
            QWEN2VL_HIDDEN_DIM, 1, bias=False, dtype=torch.bfloat16
        ).to(next(_lm_head.parameters()).device)
        del _lm_head

        self.vlm = vlm
        self.processor = processor

        # Projection head in fp32
        self.proj = _ProjectionHead(d_target).float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"ContrastiveQwen2VL ready: "
            f"d_target={d_target}, trainable params ≈ {n_trainable / 1e6:.1f}M"
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def forward(
        self,
        images: list[Image.Image],
        texts: list[str],
        max_mod_len: int = 512,
    ) -> torch.Tensor:
        """
        images: list of B PIL candidate images
        texts:  list of B modification strings (empty string = mod-stripped)
        Returns: (B, D_target) float32, L2-normalized.
        """
        B = len(images)
        assert len(texts) == B

        # Build per-sample messages; truncate mod text (rough char-level guard)
        messages_list = []
        for img, txt in zip(images, texts):
            txt = txt[:max_mod_len]
            content = [{"type": "image", "image": img}]
            if txt:
                instruction = (
                    f"Given this product image, find the item that looks like "
                    f"the image but with the following modification: {txt}"
                )
            else:
                instruction = "Describe the product shown in this image."
            content.append({"type": "text", "text": instruction})
            messages_list.append([{"role": "user", "content": content}])

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

        # Index of the last real (non-pad) token — works for both left and right
        # padding by finding the rightmost 1 in each attention_mask row.
        # (Qwen2-VL uses left padding; sum-1 would point into the pad region.)
        last_hs = outputs.hidden_states[-1]  # (B, seq_len, 3584), bf16
        seq_len = inputs["attention_mask"].shape[1]
        seq_ends = (seq_len - 1
                    - inputs["attention_mask"].flip(dims=[1]).long().argmax(dim=1))  # (B,)
        pooled = last_hs[torch.arange(B, device=device), seq_ends, :]  # (B, 3584)

        # Projection + L2 normalize (in fp32)
        emb = self.proj(pooled.float())          # (B, D_target)
        emb = F.normalize(emb, dim=-1)
        return emb

    def encode_query(
        self,
        images: list[Image.Image],
        texts: list[str],
        **kwargs,
    ) -> torch.Tensor:
        return self(images, texts, **kwargs)

    def trainable_parameters(self):
        """Yield only trainable parameter groups with names, for optimizer construction."""
        lora_params = [p for n, p in self.vlm.named_parameters() if p.requires_grad]
        proj_params = list(self.proj.parameters())
        return lora_params, proj_params
