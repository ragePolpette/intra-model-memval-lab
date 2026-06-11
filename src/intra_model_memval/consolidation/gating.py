"""Promote / rollback decision from pre/post evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..domain import ConsolidationCandidateStatus


class GateThresholds(BaseModel):
    learning_min: float = Field(default=0.7, ge=0.0, le=1.0)
    regression_tolerance: float = Field(default=0.05, ge=0.0)
    drift_max: float | None = Field(default=None, ge=0.0)  # v2: SAE feature distance


@dataclass
class PromotionDecision:
    status: ConsolidationCandidateStatus
    checks: dict[str, bool]
    reasons: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.status == ConsolidationCandidateStatus.PROMOTED


def decide_promotion(
    *,
    learning_score: float,
    regression_pre: float,
    regression_post: float,
    drift_distance: float | None = None,
    thresholds: GateThresholds | None = None,
) -> PromotionDecision:
    """All checks above threshold -> PROMOTED; any failure -> ROLLED_BACK with reasons.

    - learning: held-out variation application rate (0-1)
    - regression: drop on the fixed protected-capability benchmarks (pre - post)
    - drift (v2): SAE feature distance on fixed probe prompts; skipped when either
      the measurement or the threshold is absent
    """
    gates = thresholds or GateThresholds()
    regression_drop = regression_pre - regression_post
    checks = {
        "learning": learning_score >= gates.learning_min,
        "regression": regression_drop <= gates.regression_tolerance,
        "drift": drift_distance is None or gates.drift_max is None or drift_distance <= gates.drift_max,
    }
    reasons: list[str] = []
    if not checks["learning"]:
        reasons.append(
            f"learning score {learning_score:.3f} below minimum {gates.learning_min:.3f}"
        )
    if not checks["regression"]:
        reasons.append(
            f"regression drop {regression_drop:.3f} above tolerance {gates.regression_tolerance:.3f}"
        )
    if not checks["drift"]:
        reasons.append(f"drift distance {drift_distance:.3f} above maximum {gates.drift_max:.3f}")
    status = (
        ConsolidationCandidateStatus.PROMOTED
        if all(checks.values())
        else ConsolidationCandidateStatus.ROLLED_BACK
    )
    return PromotionDecision(status=status, checks=checks, reasons=reasons)
