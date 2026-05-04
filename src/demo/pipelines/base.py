"""Common pipeline contract.

Every pipeline (P1 caption-based, P2 contrastive, P3 native-audio future) returns
a PipelineResult so the UI consumes a single shape regardless of backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineResult:
    target_ids: list[str]                       # top-K gallery image ids, ordered best-first
    scores: list[float]                         # cosine similarity for each target_id
    image_paths: list[Path]                     # absolute paths to the top-K images
    latency: dict[str, float] = field(default_factory=dict)   # per-stage seconds
    intermediate: dict[str, str] = field(default_factory=dict)  # caption text (P1); empty for P2
    true_target_id: str | None = None           # ground truth, if known (presets only)
    true_target_rank: int | None = None         # 1-based rank of true target, or None if outside top-K
    note: str = ""                              # free-form note shown in the UI

    @property
    def total_latency_s(self) -> float:
        return sum(self.latency.values())
