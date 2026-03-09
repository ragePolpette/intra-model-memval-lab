from __future__ import annotations

import json
from pathlib import Path

from intra_model_memval.evaluation import EvaluationHarness, load_synthetic_dataset
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.tracing import TraceRunner

from tests.fixtures.fake_model_adapter import FakeExperimentAdapter


def test_trace_runner_materializes_trace_artifacts(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    runner = TraceRunner(store=store, adapter=FakeExperimentAdapter())

    result = runner.trace_episode(
        {"content_text": "What is the capital of France?"},
        prompt_template="Question: {content_text}\nAnswer:",
        trace_type="gpt2-forward-pass",
    )

    stored_trace = store.load_trace_artifact(result.trace_artifact.trace_id)
    assert stored_trace is not None
    assert stored_trace.status == "materialized"
    assert [item.role for item in stored_trace.artifact_refs] == ["trace-summary", "trace-tensors"]
    assert all(Path(item.path).exists() for item in stored_trace.artifact_refs)
    summary = json.loads(Path(stored_trace.artifact_refs[0].path).read_text(encoding="utf-8"))
    assert summary["input_text"] == "What is the capital of France?"


def test_evaluation_harness_runs_baseline_and_persists_results(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    dataset = load_synthetic_dataset()
    spec = dataset.to_evaluation_spec(name="fake-adapter-baseline")
    store.save_evaluation_spec(spec)

    harness = EvaluationHarness()
    result = harness.run_baseline(spec, adapter=FakeExperimentAdapter(), subject_id="fake-gpt2")
    store.register_run(result.experiment_run)
    store.save_evaluation_run(result.evaluation_run)

    loaded = store.load_evaluation_run(result.evaluation_run.evaluation_run_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.metrics["case_groups"]["target"]["top1_accuracy"] == 1.0
    assert len(loaded.observations) == 10
