from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from intra_model_memval.consolidation import examples_from_jsonl_bytes
from intra_model_memval.domain import EvaluationCase, EvaluationSpec, ExperimentStatus, Rubric
from intra_model_memval.evaluation import EvaluationExecutor, score_case, score_rubric
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.utils.ids import new_id, utc_now_iso


class FakeGenerator:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.get(prompt, "I do not know.")


EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "consolidation_poc"

BILLING_RUBRIC = Rubric(
    required=[
        ["billing period", "billed monthly", "monthly or annually", "monthly or annual"],
        ["confirm", "can you", "could you", "whether"],
    ],
    forbidden=["$", "dollars", "directly"],
)


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


def test_rubric_scores_paraphrases_that_containment_misses():
    expected = "Before I quote the total, could you confirm whether your plan is billed monthly or annually?"
    paraphrase = "Can you tell me if you pay monthly or annually? Then I will work out the renewal."
    assert score_case(expected, paraphrase) == 0.0
    assert score_case(expected, paraphrase, BILLING_RUBRIC) == 1.0
    assert score_rubric(BILLING_RUBRIC, "Please CONFIRM your  Billing   Period first.").score == 1.0


def test_rubric_rejects_missing_groups_and_forbidden_terms():
    quoted = score_rubric(BILLING_RUBRIC, "Your renewal is $49 on the monthly or annual plan, can you pay?")
    assert quoted.score == 0.0
    assert quoted.forbidden_hits == ["$"]

    no_question = score_rubric(BILLING_RUBRIC, "Renewals depend on the billing period.")
    assert no_question.score == 0.0
    assert no_question.missing_groups == [["confirm", "can you", "could you", "whether"]]
    assert score_rubric(BILLING_RUBRIC, "I do not know.").score == 0.0


def test_rubric_terms_match_on_word_boundaries():
    rubric = Rubric(required=[["12", "twelve"]])
    assert score_rubric(rubric, "7 plus 5 is 12.").score == 1.0
    assert score_rubric(rubric, "Twelve.").score == 1.0
    assert score_rubric(rubric, "The answer is 112").score == 0.0
    assert Rubric(required=["H2O"]).required == [["h2o"]]


def test_rubric_requires_at_least_one_required_group():
    with pytest.raises(ValidationError):
        Rubric(required=[], forbidden=["dollars"])
    with pytest.raises(ValidationError):
        Rubric(required=[["  "]])


def test_executor_uses_case_rubric_and_records_scorer(tmp_path: Path):
    spec = EvaluationSpec(
        evaluation_spec_id=new_id("evaluation-spec"),
        name="rubric-check",
        target_facts=[
            EvaluationCase(
                case_id="t-1",
                prompt="Q-renewal",
                expected_response="Could you confirm your billing period?",
                rubric=BILLING_RUBRIC,
            )
        ],
        related_facts=[EvaluationCase(case_id="r-1", prompt="Q-capital", expected_response="Rome")],
        created_at_utc=utc_now_iso(),
    )
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    store.save_evaluation_spec(spec)
    generator = FakeGenerator(
        {"Q-renewal": "Can you say whether you are billed monthly?", "Q-capital": "Rome."}
    )

    run, evaluation_run = EvaluationExecutor().execute(
        spec, generator, subject_type="base_model", subject_id="fake-model"
    )
    assert evaluation_run.metrics["target_match_rate"] == 1.0
    assert evaluation_run.observations[0]["scorer"] == "rubric"
    assert evaluation_run.observations[0]["rubric_detail"] == {"missing_groups": [], "forbidden_hits": []}
    assert evaluation_run.observations[1]["scorer"] == "containment"
    assert run.manifest["mode"] == "executor-rubric-or-containment"

    loaded = store.load_evaluation_spec(spec.evaluation_spec_id)
    assert loaded.target_facts[0].rubric == BILLING_RUBRIC
    assert loaded.related_facts[0].rubric is None


def test_poc_fixture_targets_satisfy_their_own_rubrics():
    variations = examples_from_jsonl_bytes((EXAMPLES / "variations.jsonl").read_bytes())
    accepted = [item for item in variations if (item.verifier_score or 0.0) >= 0.7]
    assert accepted and all(item.rubric is not None for item in variations)
    for item in accepted:
        assert score_rubric(item.rubric, item.target).score == 1.0, item.target
    for item in variations:
        if item not in accepted:
            assert score_rubric(item.rubric, item.target).score == 0.0, item.target

    rows = [json.loads(line) for line in (EXAMPLES / "regression.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) >= 10
    for row in rows:
        case = EvaluationCase(case_id="r", prompt=row["input"], expected_response=row["target"], rubric=row["rubric"])
        assert score_case(case.expected_response, row["target"], case.rubric) == 1.0
