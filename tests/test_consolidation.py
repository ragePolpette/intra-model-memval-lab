from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intra_model_memval.consolidation import (
    ConsolidationConfig,
    ConsolidationPlanner,
    GateThresholds,
    apply_verifier_gate,
    build_replay_buffer,
    check_dataset_composition,
    decide_promotion,
    examples_from_jsonl_bytes,
    register_dataset,
)
from intra_model_memval.domain import (
    ConsolidationCandidateStatus,
    ExperimentStatus,
    Lesson,
    LessonStatus,
    RunKind,
    SyntheticExample,
    SyntheticExampleKind,
)
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.utils.ids import new_id, utc_now_iso


def _lesson(**overrides) -> Lesson:
    now = utc_now_iso()
    payload = {
        "lesson_id": new_id("lesson"),
        "statement": "Always confirm the billing period before quoting totals.",
        "scope_tags": ["billing"],
        "boundary": "Does not apply to one-off purchases without a billing period.",
        "source_cluster_ids": ["cluster-1"],
        "source_episode_ids": ["episode-1"],
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    payload.update(overrides)
    return Lesson(**payload)


def _example(kind: SyntheticExampleKind, score: float | None, lesson_id: str = "lesson-1") -> SyntheticExample:
    return SyntheticExample(
        lesson_id=lesson_id,
        kind=kind,
        input="A customer asks for the renewal total.",
        target="Confirm the billing period first, then quote.",
        verifier_score=score,
    )


def test_lesson_requires_boundary():
    with pytest.raises(ValidationError):
        _lesson(boundary="   ")


def test_verifier_gate_filters_and_flags_reformulation():
    examples = [
        _example(SyntheticExampleKind.POSITIVE, 0.9),
        _example(SyntheticExampleKind.NEGATIVE, 0.71),
        _example(SyntheticExampleKind.ANCHOR, 0.4),
        _example(SyntheticExampleKind.POSITIVE, None),
    ]
    result = apply_verifier_gate(examples)
    assert len(result.accepted) == 2
    assert len(result.rejected) == 2
    assert result.reject_rate == 0.5
    assert not result.lesson_needs_reformulation

    bad = apply_verifier_gate([_example(SyntheticExampleKind.POSITIVE, 0.1)] * 3)
    assert bad.lesson_needs_reformulation


def test_dataset_composition_checks_anchor_and_negative_quotas():
    balanced = (
        [_example(SyntheticExampleKind.POSITIVE, 0.9)] * 5
        + [_example(SyntheticExampleKind.NEGATIVE, 0.9)] * 2
        + [_example(SyntheticExampleKind.ANCHOR, 0.9)] * 3
    )
    report = check_dataset_composition(balanced)
    assert report.is_acceptable
    assert report.anchor_count == 3

    skewed = [_example(SyntheticExampleKind.POSITIVE, 0.9)] * 10
    report = check_dataset_composition(skewed)
    assert not report.is_acceptable
    assert len(report.issues) == 2


def test_dataset_jsonl_round_trip_via_artifact_registry(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    examples = [
        _example(SyntheticExampleKind.POSITIVE, 0.87),
        _example(SyntheticExampleKind.NEGATIVE, 0.72),
    ]
    ref = register_dataset(store, examples)
    assert ref.role == "synthetic-dataset"
    assert ref.media_type == "application/jsonl"

    loaded = examples_from_jsonl_bytes(Path(ref.path).read_bytes())
    assert [item.kind for item in loaded] == [item.kind for item in examples]
    assert loaded[0].verifier_score == 0.87


def test_replay_buffer_is_seeded_and_reports_shortfall():
    new_examples = [f"new-{i}" for i in range(10)]
    anchors = [f"anchor-{i}" for i in range(50)]
    buffer = build_replay_buffer(new_examples, anchors, ratio=1.0, seed=7)
    assert buffer.new_count == 10
    assert buffer.anchor_count == 10
    assert buffer.anchor_shortfall == 0
    assert sorted(item for item in buffer.items if item.startswith("new-")) == sorted(new_examples)

    again = build_replay_buffer(new_examples, anchors, ratio=1.0, seed=7)
    assert again.items == buffer.items

    short = build_replay_buffer(new_examples, ["anchor-0"], ratio=2.0, seed=7)
    assert short.anchor_count == 1
    assert short.anchor_shortfall == 19
    assert short.summary()["effective_ratio"] == 0.1


def test_promotion_decision_checks_all_gates():
    promoted = decide_promotion(learning_score=0.8, regression_pre=0.9, regression_post=0.88)
    assert promoted.promoted
    assert promoted.reasons == []

    rolled_back = decide_promotion(
        learning_score=0.8,
        regression_pre=0.9,
        regression_post=0.7,
        drift_distance=0.5,
        thresholds=GateThresholds(drift_max=0.2),
    )
    assert rolled_back.status == ConsolidationCandidateStatus.ROLLED_BACK
    assert not rolled_back.checks["regression"]
    assert not rolled_back.checks["drift"]
    assert len(rolled_back.reasons) == 2


def test_planner_cycle_promotes_lessons(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    lesson = store.save_lesson(_lesson())
    dataset_ref = register_dataset(store, [_example(SyntheticExampleKind.POSITIVE, 0.9, lesson.lesson_id)])

    planner = ConsolidationPlanner(ConsolidationConfig())
    run, candidate = planner.prepare_candidate(
        [lesson], dataset_ref, base_model_ref="gemma-3-270m-it+lora-stack-0"
    )
    assert run.run_kind == RunKind.CONSOLIDATION
    assert run.status == ExperimentStatus.PLANNED
    assert candidate.status == ConsolidationCandidateStatus.PREPARED
    assert candidate.train_config["rank"] == 8
    store.register_run(run)
    store.save_consolidation_candidate(candidate)

    adapter_ref = store.register_artifact_bytes(b"adapter-weights", suffix=".bin")
    trained = planner.mark_trained(candidate, adapter_ref=adapter_ref)
    assert trained.status == ConsolidationCandidateStatus.TRAINED

    decision = decide_promotion(learning_score=0.9, regression_pre=0.8, regression_post=0.79)
    final_candidate, final_lessons = planner.apply_decision(
        trained, [lesson], decision, evaluation_run_ids=["evaluation-run-1"]
    )
    store.save_consolidation_candidate(final_candidate)
    for item in final_lessons:
        store.save_lesson(item)

    loaded_candidate = store.load_consolidation_candidate(candidate.consolidation_candidate_id)
    assert loaded_candidate is not None
    assert loaded_candidate.status == ConsolidationCandidateStatus.PROMOTED
    assert loaded_candidate.evaluation_run_ids == ["evaluation-run-1"]
    assert loaded_candidate.result_summary["decision"]["status"] == "promoted"

    consolidated = store.list_lessons(status="consolidated")
    assert [item.lesson_id for item in consolidated] == [lesson.lesson_id]
    assert consolidated[0].consolidation_run_id == candidate.run_id


def test_planner_rollback_keeps_lessons_candidate(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    lesson = _lesson()
    dataset_ref = register_dataset(store, [_example(SyntheticExampleKind.POSITIVE, 0.9, lesson.lesson_id)])

    planner = ConsolidationPlanner()
    run, candidate = planner.prepare_candidate([lesson], dataset_ref, base_model_ref="gemma-3-270m-it")
    store.register_run(run)

    with pytest.raises(ValueError):
        planner.apply_decision(candidate, [lesson], decide_promotion(
            learning_score=0.0, regression_pre=1.0, regression_post=0.0
        ))

    adapter_ref = store.register_artifact_bytes(b"adapter", suffix=".bin")
    trained = planner.mark_trained(candidate, adapter_ref=adapter_ref)
    decision = decide_promotion(learning_score=0.2, regression_pre=0.9, regression_post=0.9)
    final_candidate, final_lessons = planner.apply_decision(trained, [lesson], decision)
    assert final_candidate.status == ConsolidationCandidateStatus.ROLLED_BACK
    assert final_lessons[0].status == LessonStatus.CANDIDATE
    assert final_candidate.result_summary["decision"]["reasons"]
