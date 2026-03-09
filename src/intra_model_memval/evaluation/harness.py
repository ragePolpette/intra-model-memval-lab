"""Baseline evaluation harness for real adapter inference."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters import BaseModelAdapter
from ..domain.models import EvaluationRun, EvaluationSpec, ExperimentRun, ExperimentStatus, RunKind
from ..utils.ids import new_id, utc_now_iso
from .metrics import aggregate_observations, score_case_prediction


@dataclass(slots=True)
class EvaluationExecutionResult:
    experiment_run: ExperimentRun
    evaluation_run: EvaluationRun


class EvaluationHarness:
    """Execute a baseline pre-update evaluation against an adapter."""

    def prepare_run(
        self,
        spec: EvaluationSpec,
        *,
        subject_type: str,
        subject_id: str,
        run_id: str | None = None,
        evaluation_run_id: str | None = None,
    ) -> tuple[ExperimentRun, EvaluationRun]:
        now = utc_now_iso()
        actual_run_id = run_id or new_id("run")
        actual_evaluation_run_id = evaluation_run_id or new_id("evaluation-run")

        manifest = {
            "evaluation_spec_id": spec.evaluation_spec_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "case_counts": {
                "target": len(spec.target_facts),
                "related": len(spec.related_facts),
                "unrelated": len(spec.unrelated_facts),
            },
            "mode": "baseline-next-token",
        }
        experiment_run = ExperimentRun(
            run_id=actual_run_id,
            run_kind=RunKind.EVALUATION,
            status=ExperimentStatus.PLANNED,
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
            status=ExperimentStatus.PLANNED,
            metrics={
                "target_case_count": len(spec.target_facts),
                "related_case_count": len(spec.related_facts),
                "unrelated_case_count": len(spec.unrelated_facts),
                "mode": "baseline-next-token",
            },
            observations=[],
            created_at_utc=now,
            updated_at_utc=now,
        )
        return experiment_run, evaluation_run

    def run_baseline(
        self,
        spec: EvaluationSpec,
        *,
        adapter: BaseModelAdapter,
        run_id: str | None = None,
        evaluation_run_id: str | None = None,
        subject_type: str = "model_adapter",
        subject_id: str | None = None,
    ) -> EvaluationExecutionResult:
        actual_subject_id = subject_id or adapter.model_id
        run, evaluation_run = self.prepare_run(
            spec,
            subject_type=subject_type,
            subject_id=actual_subject_id,
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
        )

        observations: list[dict[str, object]] = []
        for group_name, cases in (
            ("target", spec.target_facts),
            ("related", spec.related_facts),
            ("unrelated", spec.unrelated_facts),
        ):
            for case in cases:
                forward_result = adapter.forward(case.prompt, output_hidden_states=False, output_attentions=False)
                case_metrics = score_case_prediction(adapter, forward_result, case.expected_response)
                observations.append(
                    {
                        "case_id": case.case_id,
                        "group": group_name,
                        "prompt": case.prompt,
                        "expected_response": case.expected_response,
                        "metadata": case.metadata,
                        "metrics": case_metrics,
                    }
                )

        aggregate_metrics = aggregate_observations(observations)
        now = utc_now_iso()
        completed_run = run.model_copy(
            update={
                "status": ExperimentStatus.COMPLETED,
                "manifest": {
                    **run.manifest,
                    "adapter_id": adapter.adapter_id,
                    "model_id": adapter.model_id,
                    "adapter_metadata": adapter.describe().model_dump(mode="json"),
                },
                "updated_at_utc": now,
            }
        )
        completed_evaluation_run = evaluation_run.model_copy(
            update={
                "status": ExperimentStatus.COMPLETED,
                "metrics": {**evaluation_run.metrics, **aggregate_metrics},
                "observations": observations,
                "updated_at_utc": now,
            }
        )
        return EvaluationExecutionResult(
            experiment_run=completed_run,
            evaluation_run=completed_evaluation_run,
        )
