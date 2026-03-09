from __future__ import annotations

import json
import sys
from pathlib import Path

from intra_model_memval import cli

from tests.fixtures.fake_model_adapter import FakeExperimentAdapter


class FakeCLIAdapter(FakeExperimentAdapter):
    pass


def test_cli_runs_gpt2_trace_and_baseline_eval(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "exp.db"
    artifact_dir = tmp_path / "artifacts"
    dataset_path = Path("experiments/gpt2_small/datasets/synthetic_capitals.json").resolve()
    prompt_path = Path("experiments/gpt2_small/prompts/capital_prompt.txt").resolve()

    monkeypatch.setattr(cli, "GPT2SmallAdapter", FakeCLIAdapter)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intra-model-exp",
            "--db-path",
            str(db_path),
            "--artifact-dir",
            str(artifact_dir),
            "run-gpt2-trace",
            "--content-text",
            "What is the capital of France?",
            "--prompt-template",
            "Question: {content_text}\nAnswer:",
        ],
    )
    assert cli.main() == 0
    trace_payload = json.loads(capsys.readouterr().out)
    assert trace_payload["trace"]["status"] == "materialized"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intra-model-exp",
            "--db-path",
            str(db_path),
            "--artifact-dir",
            str(artifact_dir),
            "run-gpt2-baseline-eval",
            "--dataset-path",
            str(dataset_path),
            "--prompt-template-file",
            str(prompt_path),
        ],
    )
    assert cli.main() == 0
    eval_payload = json.loads(capsys.readouterr().out)
    assert eval_payload["evaluation_run"]["status"] == "completed"
    assert eval_payload["evaluation_run"]["metrics"]["case_groups"]["overall"]["top1_accuracy"] == 1.0
