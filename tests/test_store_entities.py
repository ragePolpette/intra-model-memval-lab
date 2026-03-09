from __future__ import annotations

from pathlib import Path

from intra_model_memval.domain import (
    EvaluationCase,
    EvaluationSpec,
    ExperimentStatus,
    ExperimentRun,
    RunKind,
    TraceArtifact,
    UpdateCandidate,
)
from intra_model_memval.evaluation import EvaluationHarness
from intra_model_memval.ingestion import EpisodeIngestionService
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.utils.ids import new_id, utc_now_iso


def test_store_persists_and_searches_episodes(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    service = EpisodeIngestionService()
    episode = service.ingest_episode(
        {
            "content_text": "Observed a contradiction in billing policy.",
            "topic_tags": ["billing", "policy"],
            "notes": "needs review",
            "provenance": {"source_type": "tool", "source_label": "audit"},
        }
    )
    store.save_episode(episode)

    loaded = store.load_episode(episode.episode_id)
    assert loaded is not None
    assert loaded.content_text == episode.content_text

    search = store.search_episodes("billing")
    assert [item.episode_id for item in search] == [episode.episode_id]

    listed = store.list_episodes(topic_tag="policy")
    assert [item.episode_id for item in listed] == [episode.episode_id]


def test_store_persists_trace_artifacts_and_registered_files(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    service = EpisodeIngestionService()
    episode = store.save_episode(
        service.ingest_episode(
            {
                "content_text": "The model confused the launch year.",
                "provenance": {"source_type": "dataset", "source_label": "manual-review"},
            }
        )
    )

    run_id = new_id("run")
    now = utc_now_iso()
    store.register_run(
        ExperimentRun(
            run_id=run_id,
            run_kind=RunKind.TRACE,
            status=ExperimentStatus.PLANNED,
            title="trace placeholder",
            manifest={"episode_id": episode.episode_id},
            created_at_utc=now,
            updated_at_utc=now,
        )
    )
    sample_file = tmp_path / "trace.txt"
    sample_file.write_text("trace summary", encoding="utf-8")
    artifact_ref = store.register_artifact_file(sample_file, role="trace-summary", media_type="text/plain")

    trace = TraceArtifact(
        trace_id=new_id("trace"),
        episode_id=episode.episode_id,
        run_id=run_id,
        adapter_id="adapter-placeholder",
        trace_input_spec={"prompt_template": "probe"},
        trace_type="activation-placeholder-contract",
        artifact_refs=[artifact_ref],
        summary_metrics={"artifact_count": 1},
        status="registered",
        reproducibility_metadata={"seed": 11},
        created_at_utc=now,
        updated_at_utc=now,
    )
    store.save_trace_artifact(trace)

    traces = store.list_trace_artifacts(episode_id=episode.episode_id)
    assert len(traces) == 1
    assert traces[0].artifact_refs[0].role == "trace-summary"
    assert Path(traces[0].artifact_refs[0].path).exists()


def test_store_persists_update_candidates(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    service = EpisodeIngestionService()
    episode = store.save_episode(
        service.ingest_episode(
            {
                "content_text": "The subject was born in 1984.",
                "provenance": {"source_type": "user", "source_label": "session"},
            }
        )
    )
    run_id = new_id("run")
    now = utc_now_iso()
    store.register_run(
        ExperimentRun(
            run_id=run_id,
            run_kind=RunKind.UPDATE_CANDIDATE,
            status=ExperimentStatus.PLANNED,
            title="candidate proposal",
            manifest={"episode_id": episode.episode_id},
            created_at_utc=now,
            updated_at_utc=now,
        )
    )

    candidate = UpdateCandidate(
        update_candidate_id=new_id("update-candidate"),
        episode_id=episode.episode_id,
        run_id=run_id,
        target_fact_spec={"claim": "born in 1984"},
        candidate_localization={"layers": ["future"]},
        update_budget={"max_parameters_ratio": 0.01},
        hypothesis="Localized correction may live in a narrow factual pathway.",
        created_at_utc=now,
        updated_at_utc=now,
    )
    store.save_update_candidate(candidate)

    loaded = store.load_update_candidate(candidate.update_candidate_id)
    assert loaded is not None
    assert loaded.update_budget["max_parameters_ratio"] == 0.01


def test_evaluation_harness_creates_planned_run(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    spec = EvaluationSpec(
        evaluation_spec_id=new_id("evaluation-spec"),
        name="fact-check",
        target_facts=[EvaluationCase(case_id="t-1", prompt="Q1", expected_response="A1")],
        related_facts=[EvaluationCase(case_id="r-1", prompt="Q2", expected_response="A2")],
        unrelated_facts=[EvaluationCase(case_id="u-1", prompt="Q3", expected_response="A3")],
        regression_rules=["target must improve without unrelated drift"],
        created_at_utc=utc_now_iso(),
    )
    store.save_evaluation_spec(spec)

    harness = EvaluationHarness()
    run, evaluation_run = harness.prepare_run(spec, subject_type="update_candidate", subject_id="uc-1")
    store.register_run(run)
    store.save_evaluation_run(evaluation_run)

    loaded = store.load_evaluation_run(evaluation_run.evaluation_run_id)
    assert loaded is not None
    assert loaded.metrics["target_case_count"] == 1
    assert loaded.status == ExperimentStatus.PLANNED
