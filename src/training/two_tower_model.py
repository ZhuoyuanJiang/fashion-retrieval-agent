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

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.training.contrastive_model import ContrastiveQwen2VL


# Locked V1 target prompt (per Plan-10 §4.2). Alternates are V3.
TARGET_PROMPT = "Describe this image in detail."


def make_adapter_restoring_context_fn(peft_model):
    """PEFT-aware context_fn for torch.utils.checkpoint.checkpoint.

    Captures `peft_model.active_adapter` at forward time, restores it
    during the backward-pass recompute, and snapshots/restores
    `requires_grad` on every LoRA param around `set_adapter` to defeat
    PEFT's trainability side effect (set_adapter flips inactive adapter
    requires_grad to False; see peft/tuners/tuners_utils.py:1009-1018).

    Without this fix, HF gradient checkpointing's recompute pass would
    silently use whichever adapter is active when backward runs (i.e.
    the other branch's adapter), corrupting gradients.

    See Plan_12 §4 for the full design + safety argument.
    """
    # Cache the LoRA param list once at factory-construction time.
    # Recompute fires many times per backward (once per checkpointed
    # block); avoid walking named_parameters() inside it.
    lora_params = [
        (n, p) for n, p in peft_model.named_parameters() if "lora_" in n
    ]

    def context_fn():
        # Called at forward time — capture the active adapter via closure.
        adapter_at_forward = peft_model.active_adapter
        fwd_ctx = contextlib.nullcontext()

        @contextlib.contextmanager
        def recompute_ctx_mgr():
            # Snapshot state at recompute entry.
            prev_adapter = peft_model.active_adapter
            prev_grad = [(p, p.requires_grad) for _, p in lora_params]

            # Switch to forward-time adapter. set_adapter flips
            # requires_grad on all LoRA layers; immediately reassert
            # True so autograd can accumulate gradients into both
            # adapters during this recompute.
            peft_model.set_adapter(adapter_at_forward)
            for _, p in lora_params:
                p.requires_grad = True
            try:
                yield
            finally:
                # Restore active_adapter to what backward saw on entry
                # (set_adapter again applies the trainability side
                # effect, then we restore the snapshot below).
                peft_model.set_adapter(prev_adapter)
                for p, was_trainable in prev_grad:
                    p.requires_grad = was_trainable

        return fwd_ctx, recompute_ctx_mgr()

    return context_fn


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


# ---------------------------------------------------------------------------
# Option A — Shared backbone + two PEFT LoRA adapters + two projection heads.
#
# Single Qwen2VL backbone (frozen base, Stage-2 LoRA merged); two PEFT LoRA
# adapters ("query" / "target") attached on top; two projection heads.
#
# Gradient checkpointing is INTENTIONALLY DISABLED here. See Progress_11
# §"Summary" and Appendix C: PEFT `set_adapter` mutates `model.active_adapter`,
# and non-reentrant gradient checkpointing recomputes forward at backward
# time reading the *current* `active_adapter`. With checkpointing on, the
# query branch's backward recompute would read "target" (the last value
# set), apply lora_target weights to query inputs, and produce silently
# wrong gradients on lora_query. V1.1 may revisit this via PyTorch's
# `checkpoint(context_fn=...)` hook once the architecture is proven.
#
# The PEFT `requires_grad` flip footgun (Plan_10 §4.3) still applies and is
# mitigated by an explicit `requires_grad=True` reset at the end of
# `forward()` before returning.
# ---------------------------------------------------------------------------

# Reuse the same query template that ContrastiveQwen2VL uses, so the shared-
# backbone query forward matches Plan-6 / Option B query prompts byte-for-byte.
_QUERY_INSTR_WITH_MOD = (
    "Given this product image, find the item that looks like "
    "the image but with the following modification: {txt}"
)
_QUERY_INSTR_NO_MOD = "Describe the product shown in this image."


class TwoTowerSharedBackbone(nn.Module):
    """Plan-10 V1 Option A: shared Qwen2VL backbone + two PEFT LoRA adapters.

    Architecture:
        - one speechQwen2VL backbone (frozen base + Stage-2 LoRA merged in)
        - PEFT LoRA adapter "query" (rank 32, alpha 64, q/k/v/o_proj)
        - PEFT LoRA adapter "target" (same config, attached via add_adapter)
        - two independent projection heads (head_query, head_target), fp32

    Forward calls `self.vlm.set_adapter("query")` then forwards the query
    side, then `self.vlm.set_adapter("target")` then forwards the target
    side with the locked TARGET_PROMPT. Before returning, both adapters'
    `lora_*` params have `requires_grad` reset to True to undo PEFT's
    set_adapter-flips-inactive-adapter-False footgun.

    DDP-safe public API mirrors TwoTowerSeparateBackbones:
        - forward(cand_images, mod_texts, tgt_images) -> (q_emb, t_emb)
        - encode_query(images, texts), encode_target(images) on UNWRAPPED model
        - trainable_parameters() -> (q_lora, q_proj, t_lora, t_proj)

    Gradient checkpointing is ON (Plan-12), wrapped with a PEFT-aware
    context_fn that captures the forward-time adapter and restores it
    during backward recompute. See `make_adapter_restoring_context_fn`
    above for the safety argument.
    """

    def __init__(self, d_target: int, device_map: str = "cuda:0") -> None:
        super().__init__()
        self.d_target = d_target

        # Use the same backbone loader as ContrastiveQwen2VL so Stage-2 LoRA
        # is merged into the base BEFORE our adapters get attached.
        from src.baseline.vlm_caption import _load_qwen2vl_base
        from src.training.contrastive_model import (
            BASE_REPO, LORA_REPO, LORA_RANK, LORA_ALPHA,
            LORA_TARGET_MODULES, QWEN2VL_HIDDEN_DIM, _ProjectionHead,
        )

        print("Loading speechQwen2VL base + merging Stage-2 LoRA (Option A)...")
        vlm, processor = _load_qwen2vl_base(
            base_repo=BASE_REPO,
            lora_repo=LORA_REPO,
            merge_stage2=True,
            device_map=device_map,
        )

        # Attach the first PEFT adapter as "query"; this also wraps vlm in
        # a PeftModel and sets active_adapter="query" by default.
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        vlm = get_peft_model(vlm, lora_cfg, adapter_name="query")
        # Add second adapter; will live alongside "query" inside the same
        # PeftModel object. After add_adapter, the new adapter starts with
        # requires_grad=False on its lora_* params.
        vlm.add_adapter("target", lora_cfg)

        # Both adapters must be trainable simultaneously for our two-branch
        # forward to receive gradient on both. PEFT's set_adapter API only
        # tracks ONE active adapter; we manage requires_grad ourselves.
        for n, p in vlm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        vlm.enable_input_require_grads()
        # Plan-12: enable gradient checkpointing with a PEFT-aware
        # context_fn. The context_fn captures the active adapter at
        # forward time so the backward recompute uses the correct
        # adapter, not whichever one happens to be active when backward
        # runs. Without this fix, recompute would silently use the
        # other branch's adapter and corrupt gradients. See
        # Plan_12_20260512.md §4 for the full design + safety argument.
        vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": make_adapter_restoring_context_fn(vlm),
            }
        )
        vlm.config.use_cache = False

        # Replace lm_head with a 1-output stub — we never use logits, only
        # hidden_states[-1]. Same trick as ContrastiveQwen2VL.
        _lm_head = vlm.base_model.model.lm_head
        vlm.base_model.model.lm_head = nn.Linear(
            QWEN2VL_HIDDEN_DIM, 1, bias=False, dtype=torch.bfloat16
        ).to(next(_lm_head.parameters()).device)
        del _lm_head

        self.vlm = vlm
        self.processor = processor

        # Two independent projection heads (fp32)
        self.head_query = _ProjectionHead(d_target).float()
        self.head_target = _ProjectionHead(d_target).float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"TwoTowerSharedBackbone ready: d_target={d_target}, "
            f"trainable params ≈ {n_trainable / 1e6:.1f}M "
            f"(both LoRA adapters + both heads)"
        )

    # ------------------------------------------------------------------
    # Encoding helpers (shared by training forward and eval helpers)
    # ------------------------------------------------------------------

    def _forward_one_side(
        self,
        images: list[Image.Image],
        texts: list[str],
        head: nn.Module,
        side: str,                # "query" or "target"
        max_mod_len: int = 512,
    ) -> torch.Tensor:
        """Build prompt, run vlm, pool, project, L2-normalize. Caller is
        responsible for calling `self.vlm.set_adapter(...)` before this.
        """
        B = len(images)
        messages_list = []
        for img, txt in zip(images, texts):
            if side == "target":
                instr = TARGET_PROMPT
            else:
                txt = txt[:max_mod_len]
                instr = (
                    _QUERY_INSTR_WITH_MOD.format(txt=txt) if txt
                    else _QUERY_INSTR_NO_MOD
                )
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instr},
                ],
            }])

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
        )
        pooled = last_hs[torch.arange(B, device=device), seq_ends, :]

        emb = head(pooled.float())
        emb = F.normalize(emb, dim=-1)
        return emb

    # ------------------------------------------------------------------
    # Public API: DDP-safe parent forward + eval helpers + param grouping
    # ------------------------------------------------------------------

    def forward(
        self,
        cand_images: list[Image.Image],
        mod_texts: list[str],
        tgt_images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDP-safe training entry point.

        Calls vlm twice with the per-side LoRA adapter active. The PEFT
        set_adapter call mutates `model.active_adapter` AND flips
        `requires_grad=False` on the inactive adapter — we undo the
        requires_grad flip via an explicit reset before returning so
        autograd's leaf-grad accumulation will fire on BOTH adapters
        during backward.

        Returns (q_emb, t_emb), both (B, D) fp32 L2-normalized.
        """
        self.vlm.set_adapter("query")
        q_emb = self._forward_one_side(
            cand_images, mod_texts, self.head_query, side="query"
        )

        self.vlm.set_adapter("target")
        t_emb = self._forward_one_side(
            tgt_images, [""] * len(tgt_images), self.head_target, side="target"
        )

        # PEFT set_adapter footgun mitigation (Plan_10 §4.3): set_adapter
        # flipped requires_grad=False on the inactive adapter. Reset both
        # to True so autograd accumulates gradients on both during backward.
        for n, p in self.vlm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        return q_emb, t_emb

    @torch.inference_mode()
    def encode_query(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> torch.Tensor:
        """Eval-only. Call on the UNWRAPPED model."""
        self.vlm.set_adapter("query")
        return self._forward_one_side(
            images, texts, self.head_query, side="query"
        )

    @torch.inference_mode()
    def encode_target(self, images: list[Image.Image]) -> torch.Tensor:
        """Eval-only. Uses the locked TARGET_PROMPT internally.
        Call on the UNWRAPPED model."""
        self.vlm.set_adapter("target")
        return self._forward_one_side(
            images, [""] * len(images), self.head_target, side="target"
        )

    def trainable_parameters(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter],
               list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """Returns (q_lora, q_proj, t_lora, t_proj) for optimizer construction.

        Filters by parameter name only (not requires_grad), because
        set_adapter would mid-step flip requires_grad on whichever adapter
        is inactive. The optimizer holds references to all four groups;
        actual gradient flow per step is controlled by forward()'s reset.
        """
        q_lora = [
            p for n, p in self.vlm.named_parameters()
            if "lora_" in n and ".query." in n
        ]
        t_lora = [
            p for n, p in self.vlm.named_parameters()
            if "lora_" in n and ".target." in n
        ]
        q_proj = list(self.head_query.parameters())
        t_proj = list(self.head_target.parameters())
        return q_lora, q_proj, t_lora, t_proj
