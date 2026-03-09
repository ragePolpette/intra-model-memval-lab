"""Minimal evaluation harness that prepares auditable evaluation runs without faking model execution."""

from __future__ import annotations

from ..domain.models import EvaluationRun, EvaluationSpec, ExperimentRun, ExperimentStatus, RunKind
from ..utils.ids import new_id, utc_now_iso


class EvaluationHarness:
    """Create run manifests and placeholder evaluation runs."""

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
            "mode": "placeholder-harness",
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
            },
            observations=[],
            created_at_utc=now,
            updated_at_utc=now,
        )
        return experiment_run, evaluation_run
