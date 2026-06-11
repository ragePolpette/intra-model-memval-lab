"""Replay buffer assembly: mix new variations with anchors at a configurable ratio.

Anchors are whatever guards against forgetting: generic-corpus samples,
self-distillation outputs snapshotted before training, or variations of lessons
consolidated in previous cycles. This module only does the deterministic mixing;
sourcing the anchors is the caller's job.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ReplayBuffer:
    items: list[Any]
    new_count: int
    anchor_count: int
    requested_anchor_count: int
    anchor_shortfall: int

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "new_count": self.new_count,
            "anchor_count": self.anchor_count,
            "requested_anchor_count": self.requested_anchor_count,
            "anchor_shortfall": self.anchor_shortfall,
            "effective_ratio": (self.anchor_count / self.new_count) if self.new_count else 0.0,
        }


def build_replay_buffer(
    new_examples: Sequence[Any],
    anchors: Sequence[Any],
    *,
    ratio: float = 1.0,
    seed: int = 42,
) -> ReplayBuffer:
    """Interleave new examples with sampled anchors at new:anchor ratio 1:`ratio`.

    Sampling and shuffling are seeded for reproducible buffers. If the anchor pool
    is too small the buffer is still built, but the shortfall is reported so the
    run manifest records that the forgetting guard was weaker than configured.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    rng = random.Random(seed)
    requested = round(len(new_examples) * ratio)
    pool = list(anchors)
    if requested >= len(pool):
        sampled = pool
    else:
        sampled = rng.sample(pool, requested)
    items = list(new_examples) + sampled
    rng.shuffle(items)
    return ReplayBuffer(
        items=items,
        new_count=len(new_examples),
        anchor_count=len(sampled),
        requested_anchor_count=requested,
        anchor_shortfall=max(0, requested - len(sampled)),
    )
