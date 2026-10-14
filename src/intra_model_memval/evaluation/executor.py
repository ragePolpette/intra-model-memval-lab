"""Evaluation executor that runs real cases against a text generator and records metrics.

The generator is a narrow protocol so the executor works the same with the
HF-backed implementation (train extra) and with fakes in tests. Each case is
scored with its keyword rubric when it defines one, otherwise with a normalized
containment match on the expected response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationSpec,
    ExperimentRun,
    ExperimentStatus,
    Rubric,
    RunKind,
)
from ..utils.ids import new_id, utc_now_iso


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _contains_term(text: str, term: str) -> bool:
    prefix = r"(?<!\w)" if re.match(r"\w", term[0]) else ""
    suffix = r"(?!\w)" if re.match(r"\w", term[-1]) else ""
    return re.search(prefix + re.escape(term) + suffix, text) is not None


@dataclass
class RubricResult:
    score: float
    missing_groups: list[list[str]] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"missing_groups": self.missing_groups, "forbidden_hits": self.forbidden_hits}


def score_rubric(rubric: Rubric, generated: str) -> RubricResult:
    """1.0 when every required group matches and no forbidden term appears, else 0.0.

    Terms match on word boundaries of the normalized generation, so "12" does not
    match "112". This is keyword evidence, not semantics: negations such as
    "not Rome" still match "rome" unless the rubric forbids them explicitly.
    """
    text = _normalize(generated)
    missing = [
        list(group)
        for group in rubric.required
        if not any(_contains_term(text, term) for term in group)
    ]
    forbidden = [term for term in rubric.forbidden if _contains_term(text, term)]
    score = 1.0 if not missing and not forbidden else 0.0
    return RubricResult(score=score, missing_groups=missing, forbidden_hits=forbidden)


def score_case(expected: str, generated: str, rubric: Rubric | None = None) -> float:
    """Rubric score when a rubric is given, else 1.0 on normalized containment of `expected`."""
    if rubric is not None:
        return score_rubric(rubric, generated).score
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
                observation: dict[str, object] = {
                    "family": family,
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "expected_response": case.expected_response,
                    "generated": generated,
                }
                if case.rubric is not None:
                    result = score_rubric(case.rubric, generated)
                    score = result.score
                    observation["scorer"] = "rubric"
                    observation["rubric_detail"] = result.as_dict()
                else:
                    score = score_case(case.expected_response, generated)
                    observation["scorer"] = "containment"
                observation["score"] = score
                scores.append(score)
                observations.append(observation)
            metrics[f"{family}_case_count"] = len(cases)
            metrics[f"{family}_match_rate"] = (sum(scores) / len(scores)) if scores else 0.0

        manifest = {
            "evaluation_spec_id": spec.evaluation_spec_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "case_counts": {family: len(cases) for family, cases in families.items()},
            "mode": "executor-rubric-or-containment",
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
