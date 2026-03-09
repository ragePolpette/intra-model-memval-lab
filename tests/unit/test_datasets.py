from __future__ import annotations

from intra_model_memval.evaluation import load_synthetic_dataset


def test_load_synthetic_dataset_from_experiment_assets():
    dataset = load_synthetic_dataset()
    spec = dataset.to_evaluation_spec(name="unit-test-spec")

    assert dataset.dataset_id == "gpt2-small-capitals-v1"
    assert len(dataset.target_cases) == 4
    assert len(dataset.related_cases) == 3
    assert len(dataset.unrelated_cases) == 3
    assert "{country}" in dataset.prompt_template
    assert spec.target_facts[0].expected_response == "Paris"
    assert spec.metadata["dataset_id"] == dataset.dataset_id
