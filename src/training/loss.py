"""Symmetric InfoNCE loss for Plan-5/6 contrastive training."""
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


def _gather_tensor(x: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor; no-op if distributed is not initialized.
    Float tensors: uses GatherLayer so gradients flow back to originating rank.
    Integer tensors: uses plain all_gather (no gradient needed).
    """
    if not dist.is_initialized():
        return x
    if x.is_floating_point():
        return torch.cat(GatherLayer.apply(x), dim=0)
    # Plain all_gather for non-float (e.g. int64 target_ids)
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def _multi_positive_nce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive InfoNCE for one direction.

    logits : (N, N) scaled similarity matrix.
    pos_mask : (N, N) bool — True where (i, j) share the same target.

    For each row i:
        loss[i] = logsumexp(logits[i, all]) - logsumexp(logits[i, positives])
    Returns the mean over rows.
    """
    # Mask out non-positives with -inf for the numerator logsumexp
    NEG_INF = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~pos_mask, NEG_INF)
    log_pos = torch.logsumexp(pos_logits, dim=1)   # (N,)
    log_all = torch.logsumexp(logits, dim=1)        # (N,)
    return (log_all - log_pos).mean()


class SymmetricInfoNCE(nn.Module):
    """CLIP-style symmetric in-batch InfoNCE with multi-positive masking.

    Parameterization: logit_scale = log(1/τ), init log(1/0.07) ≈ 2.659.
    Logits = exp(logit_scale) * (q @ t.T).
    Call clamp_logit_scale() after every optimizer.step() to enforce τ ≥ 0.01.

    gather=True all-gathers (q, t, target_ids) across GPUs before the softmax
    so each anchor sees negatives from all GPUs' batches. No-op when gather=False
    or when distributed is not initialized (single-GPU run).

    target_ids: (B,) int64 — gallery index for each sample. When provided,
    all (i, j) pairs with the same target_id are treated as co-positives
    (multi-positive InfoNCE). When None, falls back to diagonal-only labels.
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
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q: (B, D) query embeddings, L2-normalized.
        t: (B, D) target embeddings, L2-normalized.
        target_ids: (B,) int64 gallery indices, optional.
        Returns scalar loss (average of q→t and t→q directions).
        """
        if gather and dist.is_initialized():
            q = _gather_tensor(q)
            t = _gather_tensor(t)
            if target_ids is not None:
                target_ids = _gather_tensor(target_ids)

        N = q.shape[0]
        scale = self.logit_scale.exp()
        logits_q2t = scale * (q @ t.T)  # (N, N)

        if target_ids is not None:
            # Multi-positive mask: (N, N), True where target_ids match
            pos_mask = target_ids.unsqueeze(1) == target_ids.unsqueeze(0)  # (N, N)
            loss = (
                _multi_positive_nce(logits_q2t, pos_mask)
                + _multi_positive_nce(logits_q2t.T, pos_mask.T)
            ) / 2.0
        else:
            # Diagonal-only fallback (Plan-5 behavior)
            labels = torch.arange(N, device=q.device)
            loss = (
                F.cross_entropy(logits_q2t, labels)
                + F.cross_entropy(logits_q2t.T, labels)
            ) / 2.0
        return loss
