"""Promote / rollback decision from pre/post evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..domain import ConsolidationCandidateStatus


class GateThresholds(BaseModel):
    """Promotion thresholds.

    `regression_tolerance` is a drop in match rate. On a small benchmark one case is
    worth 1/N, so any tolerance below that means zero tolerance; set
    `regression_tolerance_cases` to express the tolerance as a number of lost cases
    instead. `min_regression_cases` fails the gate when the regression benchmark is
    smaller than that, or when its size is not reported.
    """

    learning_min: float = Field(default=0.7, ge=0.0, le=1.0)
    regression_tolerance: float = Field(default=0.05, ge=0.0)
    regression_tolerance_cases: int | None = Field(default=None, ge=0)
    min_regression_cases: int = Field(default=0, ge=0)
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
    regression_case_count: int | None = None,
    thresholds: GateThresholds | None = None,
) -> PromotionDecision:
    """All checks above threshold -> PROMOTED; any failure -> ROLLED_BACK with reasons.

    - learning: held-out variation application rate (0-1)
    - regression: drop on the fixed protected-capability benchmarks (pre - post), as a
      rate or, with `regression_tolerance_cases`, as a number of lost cases
    - regression_coverage: the benchmark has at least `min_regression_cases` cases
    - drift (v2): SAE feature distance on fixed probe prompts; skipped when either
      the measurement or the threshold is absent
    """
    gates = thresholds or GateThresholds()
    regression_drop = regression_pre - regression_post
    reasons: list[str] = []

    learning_ok = learning_score >= gates.learning_min
    if not learning_ok:
        reasons.append(f"learning score {learning_score:.3f} below minimum {gates.learning_min:.3f}")

    if gates.regression_tolerance_cases is not None:
        if regression_case_count is None:
            regression_ok = False
            reasons.append("regression tolerance is set in cases but the case count is unknown")
        else:
            dropped_cases = round(regression_drop * regression_case_count)
            regression_ok = dropped_cases <= gates.regression_tolerance_cases
            if not regression_ok:
                reasons.append(
                    f"regression lost {dropped_cases} cases, above tolerance "
                    f"{gates.regression_tolerance_cases}"
                )
    else:
        regression_ok = regression_drop <= gates.regression_tolerance
        if not regression_ok:
            reasons.append(
                f"regression drop {regression_drop:.3f} above tolerance {gates.regression_tolerance:.3f}"
            )

    coverage_ok = gates.min_regression_cases == 0 or (
        regression_case_count is not None and regression_case_count >= gates.min_regression_cases
    )
    if not coverage_ok:
        count_label = "an unknown number of" if regression_case_count is None else str(regression_case_count)
        reasons.append(
            f"regression benchmark has {count_label} cases, below minimum {gates.min_regression_cases}"
        )

    checks = {
        "learning": learning_ok,
        "regression": regression_ok,
        "regression_coverage": coverage_ok,
        "drift": drift_distance is None or gates.drift_max is None or drift_distance <= gates.drift_max,
    }
    if not checks["drift"]:
        reasons.append(f"drift distance {drift_distance:.3f} above maximum {gates.drift_max:.3f}")
    status = (
        ConsolidationCandidateStatus.PROMOTED
        if all(checks.values())
        else ConsolidationCandidateStatus.ROLLED_BACK
    )
    return PromotionDecision(status=status, checks=checks, reasons=reasons)
