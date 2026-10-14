"""Planner that records auditable consolidation candidates without faking training.

Mirrors EvaluationHarness: it produces run manifests and status transitions
(candidate: PREPARED -> TRAINED -> PROMOTED / ROLLED_BACK; run: PLANNED -> RUNNING ->
COMPLETED / FAILED); the actual LoRA training and evaluation execution live behind
the adapter interfaces.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..domain import (
    ArtifactRef,
    ConsolidationCandidate,
    ConsolidationCandidateStatus,
    ExperimentRun,
    ExperimentStatus,
    Lesson,
    LessonStatus,
    RunKind,
)
from ..utils.ids import new_id, utc_now_iso
from .config import ConsolidationConfig
from .gating import PromotionDecision

PLANNER_MODE = "planner-no-training"
TRAINING_MODE = "lora-training"

_FINAL_RUN_STATUSES = frozenset(
    {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED}
)

class ConsolidationPlanner:
    def __init__(self, config: ConsolidationConfig | None = None):
        self.config = config or ConsolidationConfig()

    def prepare_candidate(
        self,
        lessons: Sequence[Lesson],
        dataset_ref: ArtifactRef,
        *,
        base_model_ref: str,
        replay_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
        run_id: str | None = None,
        consolidation_candidate_id: str | None = None,
    ) -> tuple[ExperimentRun, ConsolidationCandidate]:
        """Create the run manifest and a PREPARED candidate for the given lessons + dataset."""
        now = utc_now_iso()
        actual_run_id = run_id or new_id("run")
        actual_candidate_id = consolidation_candidate_id or new_id("consolidation-candidate")
        actual_replay_config = replay_config or {"ratio": self.config.replay_ratio, "seed": self.config.seed}
        actual_train_config = train_config or {
            "adapter": "lora",
            "rank": self.config.lora_rank,
            "seed": self.config.seed,
        }

        manifest = {
            "consolidation_candidate_id": actual_candidate_id,
            "lesson_ids": [lesson.lesson_id for lesson in lessons],
            "dataset_artifact_id": dataset_ref.artifact_id,
            "dataset_artifact_hash": dataset_ref.artifact_hash,
            "base_model_ref": base_model_ref,
            "replay_config": actual_replay_config,
            "train_config": actual_train_config,
            "mode": PLANNER_MODE,
        }
        experiment_run = ExperimentRun(
            run_id=actual_run_id,
            run_kind=RunKind.CONSOLIDATION,
            status=ExperimentStatus.PLANNED,
            title=f"consolidation:{','.join(lesson.lesson_id for lesson in lessons)}",
            manifest=manifest,
            created_at_utc=now,
            updated_at_utc=now,
        )
        candidate = ConsolidationCandidate(
            consolidation_candidate_id=actual_candidate_id,
            lesson_ids=[lesson.lesson_id for lesson in lessons],
            dataset_ref=dataset_ref,
            replay_config=actual_replay_config,
            train_config=actual_train_config,
            base_model_ref=base_model_ref,
            status=ConsolidationCandidateStatus.PREPARED,
            run_id=actual_run_id,
            created_at_utc=now,
            updated_at_utc=now,
        )
        return experiment_run, candidate

    def update_run(
        self,
        run: ExperimentRun,
        *,
        status: ExperimentStatus,
        mode: str | None = None,
        manifest_updates: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """Return the consolidation run with a new status, execution mode and manifest fields.

        PLANNED/RUNNING runs may move forward; COMPLETED, FAILED and CANCELLED are final.
        """
        if run.run_kind != RunKind.CONSOLIDATION:
            raise ValueError(f"expected a consolidation run, got {run.run_kind.value}")
        if run.status in _FINAL_RUN_STATUSES:
            raise ValueError(f"cannot update a run in final status {run.status.value}")
        if status == ExperimentStatus.PLANNED and run.status != ExperimentStatus.PLANNED:
            raise ValueError(f"cannot move a run back to planned from {run.status.value}")
        manifest = dict(run.manifest)
        if manifest_updates:
            manifest.update(manifest_updates)
        if mode is not None:
            manifest["mode"] = mode
        return run.model_copy(
            update={"status": status, "manifest": manifest, "updated_at_utc": utc_now_iso()}
        )

    def mark_trained(
        self,
        candidate: ConsolidationCandidate,
        *,
        adapter_ref: ArtifactRef,
        training_summary: dict[str, Any] | None = None,
    ) -> ConsolidationCandidate:
        """Record that a real adapter was produced for a PREPARED candidate."""
        if candidate.status != ConsolidationCandidateStatus.PREPARED:
            raise ValueError(f"cannot mark trained from status {candidate.status.value}")
        result_summary = dict(candidate.result_summary)
        result_summary["adapter_ref"] = adapter_ref.model_dump(mode="json")
        if training_summary:
            result_summary["training"] = training_summary
        return candidate.model_copy(
            update={
                "status": ConsolidationCandidateStatus.TRAINED,
                "result_summary": result_summary,
                "updated_at_utc": utc_now_iso(),
            }
        )

    def apply_decision(
        self,
        candidate: ConsolidationCandidate,
        lessons: Sequence[Lesson],
        decision: PromotionDecision,
        *,
        evaluation_run_ids: Sequence[str] = (),
    ) -> tuple[ConsolidationCandidate, list[Lesson]]:
        """Apply the promote/rollback verdict to a TRAINED candidate and its lessons.

        On promotion lessons become CONSOLIDATED; on rollback they stay CANDIDATE and
        the diagnostic reasons are attached to the candidate for the next attempt.
        """
        if candidate.status != ConsolidationCandidateStatus.TRAINED:
            raise ValueError(f"cannot apply decision from status {candidate.status.value}")
        if decision.status not in (
            ConsolidationCandidateStatus.PROMOTED,
            ConsolidationCandidateStatus.ROLLED_BACK,
        ):
            raise ValueError(f"decision status must be terminal, got {decision.status.value}")

        now = utc_now_iso()
        result_summary = dict(candidate.result_summary)
        result_summary["decision"] = {
            "status": decision.status.value,
            "checks": decision.checks,
            "reasons": decision.reasons,
        }
        updated_candidate = candidate.model_copy(
            update={
                "status": decision.status,
                "evaluation_run_ids": list(candidate.evaluation_run_ids) + list(evaluation_run_ids),
                "result_summary": result_summary,
                "updated_at_utc": now,
            }
        )

        updated_lessons: list[Lesson] = []
        for lesson in lessons:
            if decision.promoted:
                updated_lessons.append(
                    lesson.model_copy(
                        update={
                            "status": LessonStatus.CONSOLIDATED,
                            "consolidation_run_id": candidate.run_id,
                            "updated_at_utc": now,
                        }
                    )
                )
            else:
                updated_lessons.append(
                    lesson.model_copy(
                        update={"status": LessonStatus.CANDIDATE, "updated_at_utc": now}
                    )
                )
        return updated_candidate, updated_lessons
