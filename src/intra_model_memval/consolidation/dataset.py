"""Verifier gating, composition checks and JSONL serialization for synthetic datasets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..domain import ArtifactRef, SyntheticExample, SyntheticExampleKind
from ..persistence import ExperimentStore
from .config import ConsolidationConfig

DATASET_MEDIA_TYPE = "application/jsonl"


@dataclass
class VerifierGateResult:
    accepted: list[SyntheticExample]
    rejected: list[SyntheticExample]
    reject_rate: float
    lesson_needs_reformulation: bool


def apply_verifier_gate(
    examples: Sequence[SyntheticExample],
    *,
    config: ConsolidationConfig | None = None,
) -> VerifierGateResult:
    """Discard examples below the verifier threshold; unscored examples count as rejected.

    A reject rate above the configured maximum means the lesson is badly formulated
    and must go back to extraction (step 2), not be patched at dataset level.
    """
    cfg = config or ConsolidationConfig()
    accepted: list[SyntheticExample] = []
    rejected: list[SyntheticExample] = []
    for example in examples:
        if example.verifier_score is not None and example.verifier_score >= cfg.verifier_threshold:
            accepted.append(example)
        else:
            rejected.append(example)
    total = len(accepted) + len(rejected)
    reject_rate = (len(rejected) / total) if total else 0.0
    return VerifierGateResult(
        accepted=accepted,
        rejected=rejected,
        reject_rate=reject_rate,
        lesson_needs_reformulation=reject_rate > cfg.max_reject_rate,
    )


@dataclass
class DatasetCompositionReport:
    total: int
    positive_count: int
    negative_count: int
    anchor_count: int
    issues: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        return not self.issues


def check_dataset_composition(
    examples: Sequence[SyntheticExample],
    *,
    config: ConsolidationConfig | None = None,
) -> DatasetCompositionReport:
    """Validate the dataset mix against the contract: kind=anchor share and negative quota.

    `anchor_count` counts SyntheticExampleKind.ANCHOR variations, not replay anchors.
    """
    cfg = config or ConsolidationConfig()
    positive = sum(1 for item in examples if item.kind == SyntheticExampleKind.POSITIVE)
    negative = sum(1 for item in examples if item.kind == SyntheticExampleKind.NEGATIVE)
    anchor = sum(1 for item in examples if item.kind == SyntheticExampleKind.ANCHOR)
    total = len(examples)

    issues: list[str] = []
    if total == 0:
        issues.append("dataset is empty")
        return DatasetCompositionReport(0, 0, 0, 0, issues)
    if anchor / total < cfg.anchor_min_share:
        issues.append(
            f"kind=anchor example share {anchor / total:.2f} below minimum {cfg.anchor_min_share:.2f}"
        )
    if negative / total < cfg.negative_share:
        issues.append(
            f"negative share {negative / total:.2f} below quota {cfg.negative_share:.2f}"
        )
    return DatasetCompositionReport(total, positive, negative, anchor, issues)


def dataset_to_jsonl_bytes(examples: Sequence[SyntheticExample]) -> bytes:
    """Serialize with the contract line format: lesson_id, kind, input, target, verifier_score.

    The optional evaluation `rubric` is written only when the example defines one.
    """
    lines = []
    for example in examples:
        row: dict[str, object] = {
            "lesson_id": example.lesson_id,
            "kind": example.kind.value,
            "input": example.input,
            "target": example.target,
            "verifier_score": example.verifier_score,
        }
        if example.rubric is not None:
            row["rubric"] = example.rubric.model_dump(mode="json")
        lines.append(json.dumps(row, ensure_ascii=True, sort_keys=True))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def examples_from_jsonl_bytes(data: bytes) -> list[SyntheticExample]:
    examples: list[SyntheticExample] = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            examples.append(SyntheticExample.model_validate(json.loads(stripped)))
    return examples


def register_dataset(
    store: ExperimentStore,
    examples: Sequence[SyntheticExample],
    *,
    role: str = "synthetic-dataset",
) -> ArtifactRef:
    """Persist the verified JSONL in the artifact registry and return its reference."""
    ref = store.register_artifact_bytes(
        dataset_to_jsonl_bytes(examples),
        media_type=DATASET_MEDIA_TYPE,
        suffix=".jsonl",
    )
    return ref.model_copy(update={"role": role})


@dataclass
class HoldoutSplit:
    train: list[SyntheticExample]
    holdout: list[SyntheticExample]

    def counts_by_kind(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for split_name, items in (("train", self.train), ("holdout", self.holdout)):
            for item in items:
                bucket = counts.setdefault(item.kind.value, {"train": 0, "holdout": 0})
                bucket[split_name] += 1
        return counts


def split_holdout(
    examples: Sequence[SyntheticExample],
    *,
    fraction: float,
    seed: int = 42,
    train_only_kinds: Iterable[SyntheticExampleKind] = (SyntheticExampleKind.ANCHOR,),
) -> HoldoutSplit:
    """Seeded held-out split stratified by example kind.

    Each eligible kind contributes round(n * fraction) examples to the held-out set,
    at least one when it has two or more examples and never all of them, so every
    kind stays represented in training. Kinds in `train_only_kinds` (by default the
    anchor-kind variations built from real episodes) always stay in training.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    excluded = set(train_only_kinds)
    rng = random.Random(seed)
    train: list[SyntheticExample] = []
    holdout: list[SyntheticExample] = []
    for kind in SyntheticExampleKind:
        items = [item for item in examples if item.kind == kind]
        if kind in excluded or fraction == 0.0 or len(items) < 2:
            train.extend(items)
            continue
        rng.shuffle(items)
        count = min(len(items) - 1, max(1, round(len(items) * fraction)))
        holdout.extend(items[:count])
        train.extend(items[count:])
    return HoldoutSplit(train=train, holdout=holdout)
