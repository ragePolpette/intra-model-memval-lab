from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intra_model_memval.consolidation import (
    PLANNER_MODE,
    TRAINING_MODE,
    ConsolidationConfig,
    ConsolidationPlanner,
    GateThresholds,
    apply_verifier_gate,
    build_replay_buffer,
    check_dataset_composition,
    decide_promotion,
    examples_from_jsonl_bytes,
    register_dataset,
    split_holdout,
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
    assert report.issues[0].startswith("kind=anchor example share")


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
    assert buffer.summary() == {
        "total": 20,
        "new_count": 10,
        "replay_anchor_count": 10,
        "requested_replay_anchor_count": 10,
        "replay_anchor_shortfall": 0,
        "effective_ratio": 1.0,
    }
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


def test_planner_updates_consolidation_run_status_and_mode(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    lesson = _lesson()
    dataset_ref = register_dataset(store, [_example(SyntheticExampleKind.POSITIVE, 0.9, lesson.lesson_id)])
    planner = ConsolidationPlanner()
    run, _ = planner.prepare_candidate([lesson], dataset_ref, base_model_ref="gemma-3-270m-it")
    assert run.manifest["mode"] == PLANNER_MODE
    store.register_run(run)

    running = planner.update_run(run, status=ExperimentStatus.RUNNING, mode=TRAINING_MODE)
    completed = planner.update_run(
        running,
        status=ExperimentStatus.COMPLETED,
        manifest_updates={"adapter_artifact_id": "artifact-1"},
    )
    store.register_run(completed)
    loaded = store.load_run(run.run_id)
    assert loaded.status == ExperimentStatus.COMPLETED
    assert loaded.manifest["mode"] == TRAINING_MODE
    assert loaded.manifest["adapter_artifact_id"] == "artifact-1"
    assert loaded.manifest["dataset_artifact_id"] == dataset_ref.artifact_id

    with pytest.raises(ValueError):
        planner.update_run(completed, status=ExperimentStatus.FAILED)
    with pytest.raises(ValueError):
        planner.update_run(running, status=ExperimentStatus.PLANNED)
    failed = planner.update_run(running, status=ExperimentStatus.FAILED, manifest_updates={"error": "boom"})
    assert failed.status == ExperimentStatus.FAILED
    assert failed.manifest["error"] == "boom"


def _kinds(examples):
    return sorted(item.kind.value for item in examples)


def _distinct_examples(kind: SyntheticExampleKind, count: int) -> list[SyntheticExample]:
    return [
        SyntheticExample(
            lesson_id="lesson-1", kind=kind, input=f"{kind.value} input {i}", target="t", verifier_score=0.9
        )
        for i in range(count)
    ]


def test_split_holdout_is_stratified_and_keeps_anchor_kind_in_training():
    examples = (
        _distinct_examples(SyntheticExampleKind.POSITIVE, 8)
        + _distinct_examples(SyntheticExampleKind.NEGATIVE, 4)
        + _distinct_examples(SyntheticExampleKind.ANCHOR, 6)
    )
    for seed in range(20):
        split = split_holdout(examples, fraction=0.2, seed=seed)
        assert _kinds(split.holdout) == ["negative", "positive", "positive"]
        assert all(item.kind != SyntheticExampleKind.ANCHOR for item in split.holdout)
        assert len(split.train) + len(split.holdout) == len(examples)
        assert not {item.input for item in split.train} & {item.input for item in split.holdout}

    assert split_holdout(examples, fraction=0.2, seed=3).holdout == split_holdout(
        examples, fraction=0.2, seed=3
    ).holdout
    counts = split_holdout(examples, fraction=0.2, seed=3).counts_by_kind()
    assert counts["anchor"] == {"train": 6, "holdout": 0}
    assert counts["negative"] == {"train": 3, "holdout": 1}


def test_split_holdout_never_empties_a_kind():
    examples = _distinct_examples(SyntheticExampleKind.POSITIVE, 2) + _distinct_examples(
        SyntheticExampleKind.NEGATIVE, 1
    )
    split = split_holdout(examples, fraction=0.9, seed=1)
    assert _kinds(split.holdout) == ["positive"]
    assert _kinds(split.train) == ["negative", "positive"]
    with pytest.raises(ValueError):
        split_holdout(examples, fraction=1.0)


def test_regression_gate_supports_case_tolerance_and_minimum_cases():
    one_lost_of_twelve = {"learning_score": 0.9, "regression_pre": 1.0, "regression_post": 11 / 12}
    strict = decide_promotion(**one_lost_of_twelve, regression_case_count=12)
    assert not strict.checks["regression"]

    tolerant = decide_promotion(
        **one_lost_of_twelve,
        regression_case_count=12,
        thresholds=GateThresholds(regression_tolerance_cases=1, min_regression_cases=10),
    )
    assert tolerant.promoted
    assert tolerant.checks["regression_coverage"]

    two_lost = decide_promotion(
        learning_score=0.9,
        regression_pre=1.0,
        regression_post=10 / 12,
        regression_case_count=12,
        thresholds=GateThresholds(regression_tolerance_cases=1),
    )
    assert not two_lost.promoted
    assert "lost 2 cases" in two_lost.reasons[0]

    too_small = decide_promotion(
        learning_score=0.9,
        regression_pre=1.0,
        regression_post=1.0,
        regression_case_count=6,
        thresholds=GateThresholds(min_regression_cases=10),
    )
    assert not too_small.promoted
    assert not too_small.checks["regression_coverage"]

    unknown_count = decide_promotion(
        learning_score=0.9,
        regression_pre=1.0,
        regression_post=1.0,
        thresholds=GateThresholds(min_regression_cases=10, regression_tolerance_cases=0),
    )
    assert not unknown_count.checks["regression_coverage"]
    assert not unknown_count.checks["regression"]
