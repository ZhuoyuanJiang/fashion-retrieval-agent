"""Symmetric InfoNCE loss for Plan-5 contrastive training."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    """All-gather with gradient flowing back only to the originating GPU."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        return grads[dist.get_rank()]


class SymmetricInfoNCE(nn.Module):
    """CLIP-style symmetric in-batch InfoNCE.

    Parameterization: logit_scale = log(1/τ), init log(1/0.07) ≈ 2.659.
    Logits = exp(logit_scale) * (q @ t.T).
    Call clamp_logit_scale() after every optimizer.step() to enforce τ ≥ 0.01.

    gather=True all-gathers (q, t) across GPUs before the softmax so each
    anchor sees negatives from all GPUs' batches. No-op when gather=False or
    when distributed is not initialized (single-GPU run).
    """

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def clamp_logit_scale(self) -> None:
        """Enforce exp(logit_scale) ≤ 100 (τ ≥ 0.01). Call after optimizer.step()."""
        self.logit_scale.data.clamp_(max=math.log(100.0))

    def forward(
        self,
        q: torch.Tensor,
        t: torch.Tensor,
        gather: bool = False,
    ) -> torch.Tensor:
        """
        q: (B, D) query embeddings, L2-normalized.
        t: (B, D) target embeddings, L2-normalized.
        Returns scalar cross-entropy loss (average of q→t and t→q directions).
        """
        if gather and dist.is_initialized():
            q = torch.cat(GatherLayer.apply(q), dim=0)
            t = torch.cat(GatherLayer.apply(t), dim=0)

        scale = self.logit_scale.exp()
        logits_q2t = scale * (q @ t.T)  # (B, B)
        logits_t2q = logits_q2t.T

        labels = torch.arange(logits_q2t.shape[0], device=q.device)
        loss = (
            F.cross_entropy(logits_q2t, labels)
            + F.cross_entropy(logits_t2q, labels)
        ) / 2.0
        return loss
