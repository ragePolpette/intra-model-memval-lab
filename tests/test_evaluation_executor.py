from __future__ import annotations

from pathlib import Path

from intra_model_memval.domain import EvaluationCase, EvaluationSpec, ExperimentStatus
from intra_model_memval.evaluation import EvaluationExecutor, score_case
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.utils.ids import new_id, utc_now_iso


class FakeGenerator:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.get(prompt, "I do not know.")


def _spec() -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_spec_id=new_id("evaluation-spec"),
        name="executor-check",
        target_facts=[
            EvaluationCase(case_id="t-1", prompt="Q-target-1", expected_response="Paris"),
            EvaluationCase(case_id="t-2", prompt="Q-target-2", expected_response="Rome"),
        ],
        related_facts=[
            EvaluationCase(case_id="r-1", prompt="Q-related-1", expected_response="Jupiter"),
        ],
        created_at_utc=utc_now_iso(),
    )


def test_score_case_uses_normalized_containment():
    assert score_case("Paris", "The capital is  PARIS, of course.") == 1.0
    assert score_case("Paris", "The capital is Lyon.") == 0.0
    assert score_case("two words", "it has TWO   WORDS inside") == 1.0


def test_executor_populates_metrics_and_observations(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    spec = _spec()
    store.save_evaluation_spec(spec)
    generator = FakeGenerator(
        {
            "Q-target-1": "It is Paris.",
            "Q-target-2": "Maybe Milan?",
            "Q-related-1": "Jupiter is the largest.",
        }
    )

    run, evaluation_run = EvaluationExecutor().execute(
        spec, generator, subject_type="base_model", subject_id="fake-model"
    )
    store.register_run(run)
    store.save_evaluation_run(evaluation_run)

    assert run.status == ExperimentStatus.COMPLETED
    assert evaluation_run.metrics["target_match_rate"] == 0.5
    assert evaluation_run.metrics["related_match_rate"] == 1.0
    assert evaluation_run.metrics["unrelated_case_count"] == 0
    assert len(evaluation_run.observations) == 3
    assert generator.prompts == ["Q-target-1", "Q-target-2", "Q-related-1"]

    loaded = store.load_evaluation_run(evaluation_run.evaluation_run_id)
    assert loaded is not None
    assert loaded.observations[1]["score"] == 0.0
    assert loaded.observations[1]["generated"] == "Maybe Milan?"
