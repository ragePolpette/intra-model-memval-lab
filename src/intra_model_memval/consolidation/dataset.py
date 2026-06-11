"""Verifier gating, composition checks and JSONL serialization for synthetic datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

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
    """Validate the dataset mix against the contract: anchor share and negative quota."""
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
            f"anchor share {anchor / total:.2f} below minimum {cfg.anchor_min_share:.2f}"
        )
    if negative / total < cfg.negative_share:
        issues.append(
            f"negative share {negative / total:.2f} below quota {cfg.negative_share:.2f}"
        )
    return DatasetCompositionReport(total, positive, negative, anchor, issues)


def dataset_to_jsonl_bytes(examples: Sequence[SyntheticExample]) -> bytes:
    """Serialize with the contract line format: lesson_id, kind, input, target, verifier_score."""
    lines = []
    for example in examples:
        lines.append(
            json.dumps(
                {
                    "lesson_id": example.lesson_id,
                    "kind": example.kind.value,
                    "input": example.input,
                    "target": example.target,
                    "verifier_score": example.verifier_score,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
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
