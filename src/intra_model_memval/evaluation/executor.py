"""Evaluation executor that runs real cases against a text generator and records metrics.

The generator is a narrow protocol so the executor works the same with the
HF-backed implementation (train extra) and with fakes in tests. Scoring is
deliberately simple for the POC: normalized containment match per case.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..domain.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationSpec,
    ExperimentRun,
    ExperimentStatus,
    RunKind,
)
from ..utils.ids import new_id, utc_now_iso


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def score_case(expected: str, generated: str) -> float:
    """1.0 when the normalized expected response is contained in the generation."""
    expected_norm = _normalize(expected)
    if not expected_norm:
        return 0.0
    return 1.0 if expected_norm in _normalize(generated) else 0.0


class EvaluationExecutor:
    """Execute an EvaluationSpec case by case, populating metrics and observations."""

    def execute(
        self,
        spec: EvaluationSpec,
        generator: TextGenerator,
        *,
        subject_type: str,
        subject_id: str,
        run_id: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> tuple[ExperimentRun, EvaluationRun]:
        now = utc_now_iso()
        actual_run_id = run_id or new_id("run")
        actual_evaluation_run_id = evaluation_run_id or new_id("evaluation-run")

        families: dict[str, list[EvaluationCase]] = {
            "target": spec.target_facts,
            "related": spec.related_facts,
            "unrelated": spec.unrelated_facts,
        }
        metrics: dict[str, float | int] = {}
        observations: list[dict[str, object]] = []
        for family, cases in families.items():
            scores: list[float] = []
            for case in cases:
                generated = generator.generate(case.prompt)
                score = score_case(case.expected_response, generated)
                scores.append(score)
                observations.append(
                    {
                        "family": family,
                        "case_id": case.case_id,
                        "prompt": case.prompt,
                        "expected_response": case.expected_response,
                        "generated": generated,
                        "score": score,
                    }
                )
            metrics[f"{family}_case_count"] = len(cases)
            metrics[f"{family}_match_rate"] = (sum(scores) / len(scores)) if scores else 0.0

        manifest = {
            "evaluation_spec_id": spec.evaluation_spec_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "case_counts": {family: len(cases) for family, cases in families.items()},
            "mode": "executor-containment-match",
        }
        experiment_run = ExperimentRun(
            run_id=actual_run_id,
            run_kind=RunKind.EVALUATION,
            status=ExperimentStatus.COMPLETED,
            title=f"evaluation:{spec.name}",
            manifest=manifest,
            created_at_utc=now,
            updated_at_utc=now,
        )
        evaluation_run = EvaluationRun(
            evaluation_run_id=actual_evaluation_run_id,
            evaluation_spec_id=spec.evaluation_spec_id,
            run_id=actual_run_id,
            subject_type=subject_type,
            subject_id=subject_id,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            observations=observations,
            created_at_utc=now,
            updated_at_utc=now,
        )
        return experiment_run, evaluation_run
