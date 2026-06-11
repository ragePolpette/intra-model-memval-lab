"""Tunable hyperparameters for the consolidation cycle, with the v0.1 contract defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConsolidationConfig(BaseModel):
    """Defaults follow the design contract table, ordered by expected impact."""

    replay_ratio: float = Field(default=1.0, gt=0.0)  # anchors per new example (1:1)
    variations_per_lesson: int = Field(default=30, gt=0)
    negative_share: float = Field(default=0.2, ge=0.0, le=1.0)
    anchor_min_share: float = Field(default=0.3, ge=0.0, le=1.0)
    verifier_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_reject_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    lora_rank: int = Field(default=8, gt=0)
    seed: int = 42
