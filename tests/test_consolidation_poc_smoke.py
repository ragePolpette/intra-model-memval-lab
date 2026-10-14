from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from intra_model_memval import cli
from intra_model_memval.consolidation import release_accelerator_memory
from intra_model_memval.consolidation.manifest import describe_base_model, git_state, hash_model_dir
from intra_model_memval.domain import (
    ConsolidationCandidateStatus,
    ExperimentStatus,
    RunKind,
)
from intra_model_memval.persistence import ExperimentStore

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "consolidation_poc"


def _load_poc_module():
    spec = importlib.util.spec_from_file_location(
        "run_consolidation_poc_smoke", ROOT / "scripts" / "run_consolidation_poc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeGenerator:
    """Base model ignores the lesson; base+adapter applies it with paraphrased wording."""

    def __init__(self, calls: list[dict], **kwargs):
        calls.append(kwargs)
        self.trained = kwargs.get("adapter_dir") is not None
        self.device = "fake-device"
        self.kinds = {row["input"]: row["kind"] for row in _rows(EXAMPLES / "variations.jsonl")}
        self.regression = {row["input"]: row["target"] for row in _rows(EXAMPLES / "regression.jsonl")}

    def generate(self, prompt: str) -> str:
        if prompt in self.regression:
            return f"The answer is {self.regression[prompt]}."
        if not self.trained:
            return "Your total is $49."
        if self.kinds[prompt] == "negative":
            return "No period to check here, so it can be quoted directly."
        return "Can you tell me if you pay monthly or annually? Then I will work out the amount."


class FakeTrainer:
    def __init__(self, store: ExperimentStore, *, output_dir: Path, fail: bool = False, **kwargs):
        self.store = store
        self.output_dir = Path(output_dir)
        self.fail = fail
        self.buffers: list[list] = []

    def train(self, candidate, buffer):
        self.buffers.append(list(buffer))
        if self.fail:
            raise RuntimeError("out of memory")
        adapter_dir = self.output_dir / candidate.consolidation_candidate_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.bin").write_bytes(b"fake-weights")
        ref = self.store.register_artifact_bytes(b"fake-adapter-zip", media_type="application/zip", suffix=".zip")
        return ref.model_copy(
            update={
                "role": "lora-adapter",
                "metadata": {
                    "adapter_dir": str(adapter_dir),
                    "train_loss": 0.1234,
                    "device": "fake-device",
                    "base_model_revision": "fake-revision",
                    "train_config": {"rank": 8},
                },
            }
        )


def _argv(workdir: Path, regression: Path | None = None, *extra: str) -> list[str]:
    return [
        "--workdir", str(workdir),
        "--lesson-json", str(EXAMPLES / "lesson.json"),
        "--variations-jsonl", str(EXAMPLES / "variations.jsonl"),
        "--anchors-jsonl", str(EXAMPLES / "anchors.jsonl"),
        "--regression-jsonl", str(regression or EXAMPLES / "regression.jsonl"),
        "--max-new-tokens", "32",
        *extra,
    ]


def _run_poc(tmp_path: Path, *extra: str, regression: Path | None = None, fail: bool = False):
    module = _load_poc_module()
    calls: list[dict] = []
    trainers: list[FakeTrainer] = []

    def trainer_factory(store, **kwargs):
        trainer = FakeTrainer(store, fail=fail, **kwargs)
        trainers.append(trainer)
        return trainer

    workdir = tmp_path / "run"
    code = module.main(
        _argv(workdir, regression, *extra),
        generator_factory=lambda base_model, **kwargs: FakeGenerator(calls, base_model=base_model, **kwargs),
        trainer_factory=trainer_factory,
    )
    store = ExperimentStore(db_path=workdir / "experiment.db", artifact_dir=workdir / "artifacts")
    return code, workdir, store, calls, trainers


def _consolidation_run(store: ExperimentStore):
    runs = [item for item in store.list_runs() if item.run_kind == RunKind.CONSOLIDATION]
    assert len(runs) == 1
    return runs[0]


def test_poc_main_runs_end_to_end_with_fakes(tmp_path: Path, capsys):
    code, workdir, store, calls, trainers = _run_poc(tmp_path)
    assert code == 0
    assert [call["max_new_tokens"] for call in calls] == [32, 32]
    assert "adapter_dir" not in calls[0]
    assert calls[1]["adapter_dir"]

    verdict = json.loads((workdir / "verdict.json").read_text())
    assert verdict["status"] == "promoted"
    assert verdict["checks"]["regression_coverage"]

    run = _consolidation_run(store)
    assert verdict["run_id"] == run.run_id
    assert run.status == ExperimentStatus.COMPLETED
    assert run.manifest["mode"] == "lora-training"
    assert run.manifest["decision_status"] == "promoted"
    assert run.manifest["pre_evaluation_run_id"] and run.manifest["post_evaluation_run_id"]

    candidate = store.list_consolidation_candidates(run_id=run.run_id)[0]
    assert candidate.status == ConsolidationCandidateStatus.PROMOTED
    assert store.list_lessons(status="consolidated")[0].consolidation_run_id == run.run_id

    spec = store.list_evaluation_specs()[0]
    holdout_prompts = {case.prompt for case in spec.target_facts}
    assert {case.metadata["kind"] for case in spec.target_facts} == {"positive", "negative"}
    assert all(case.rubric is not None for case in spec.target_facts + spec.related_facts)
    trained_inputs = {getattr(item, "input", None) or item["input"] for item in trainers[0].buffers[0]}
    assert not holdout_prompts & trained_inputs

    manifest = json.loads((workdir / "experiment_manifest.json").read_text())
    assert manifest["run_id"] == run.run_id
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 42
    assert manifest["versions"]["python"]
    assert "torch" in manifest["versions"]
    assert "commit" in manifest["git"]
    assert manifest["base_model"]["id"] == "google/gemma-3-270m-it"
    assert manifest["base_model"]["resolved_revision"] == "fake-revision"
    assert manifest["devices"] == {"requested": None, "evaluation": "fake-device", "training": "fake-device"}
    assert manifest["config"]["generation"]["max_new_tokens"] == 32
    assert manifest["config"]["gate_thresholds"]["min_regression_cases"] == 10
    assert manifest["metrics"]["pre"]["target_match_rate"] == 0.0
    assert manifest["metrics"]["post"]["target_match_rate"] == 1.0
    assert manifest["adapter"]["adapter_dir"].endswith(candidate.consolidation_candidate_id)
    assert manifest["decision"]["status"] == "promoted"
    stored_manifest = run.manifest["experiment_manifest_artifact_id"]
    assert stored_manifest.startswith("artifact-")

    capsys.readouterr()
    base = ["--db-path", str(workdir / "experiment.db"), "--artifact-dir", str(workdir / "artifacts")]
    assert cli.main([*base, "report", "--run-id", run.run_id, "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    families = {item["family"]: item for item in report["families"]}
    assert families["target"]["pre"] == 0.0 and families["target"]["post"] == 1.0
    assert families["related"]["case_count"] == 12
    assert families["related"]["delta"] == 0.0
    assert len(report["cases"]) == len(spec.target_facts) + 12
    assert all(case["scorer"] == "rubric" for case in report["cases"])

    assert cli.main([*base, "report", "--run-id", run.run_id]) == 0
    text = capsys.readouterr().out
    assert "per family" in text and "per case" in text and "holdout-0" in text

    assert cli.main([*base, "show-verdict", "--consolidation-candidate-id", candidate.consolidation_candidate_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["decision"]["status"] == "promoted"

    assert cli.main([*base, "list-lessons", "--status", "consolidated"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["lesson_id"] == "lesson-poc-billing-period"
    assert cli.main([*base, "show-lesson", "--lesson-id", "lesson-poc-billing-period"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "consolidated"
    assert cli.main([*base, "list-consolidation-candidates", "--run-id", run.run_id]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert cli.main(
        [*base, "show-consolidation-candidate", "--consolidation-candidate-id", candidate.consolidation_candidate_id]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "promoted"

    with pytest.raises(SystemExit):
        cli.main([*base, "report", "--run-id", "missing"])
    with pytest.raises(SystemExit):
        cli.main([*base, "report", "--run-id", run.manifest["pre_evaluation_run_id"]])
    with pytest.raises(SystemExit):
        cli.main([*base, "show-verdict", "--consolidation-candidate-id", "missing"])


def test_poc_main_marks_run_failed_when_training_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="out of memory"):
        _run_poc(tmp_path, fail=True)
    workdir = tmp_path / "run"
    store = ExperimentStore(db_path=workdir / "experiment.db", artifact_dir=workdir / "artifacts")
    run = _consolidation_run(store)
    assert run.status == ExperimentStatus.FAILED
    assert run.manifest["mode"] == "lora-training"
    assert run.manifest["error"] == "RuntimeError: out of memory"
    candidate = store.list_consolidation_candidates(run_id=run.run_id)[0]
    assert candidate.status == ConsolidationCandidateStatus.PREPARED
    manifest = json.loads((workdir / "experiment_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["pre"] is not None
    assert manifest["adapter"] is None
    assert not (workdir / "verdict.json").exists()


def test_poc_main_rolls_back_when_regression_benchmark_is_too_small(tmp_path: Path):
    small = tmp_path / "regression-small.jsonl"
    small.write_text("".join(json.dumps(row) + "\n" for row in _rows(EXAMPLES / "regression.jsonl")[:6]))
    code, workdir, store, _, _ = _run_poc(tmp_path, regression=small)
    assert code == 0
    verdict = json.loads((workdir / "verdict.json").read_text())
    assert verdict["status"] == "rolled_back"
    assert not verdict["checks"]["regression_coverage"]
    assert verdict["checks"]["regression"]
    assert _consolidation_run(store).status == ExperimentStatus.COMPLETED

    code, workdir, _, _, _ = _run_poc(tmp_path / "again", "--min-regression-cases", "6", regression=small)
    assert json.loads((workdir / "verdict.json").read_text())["status"] == "promoted"


def test_poc_main_rejects_non_positive_max_new_tokens(tmp_path: Path):
    module = _load_poc_module()
    assert module.main(_argv(tmp_path / "run", None, "--max-new-tokens", "0")) == 2


def test_release_accelerator_memory_empties_cuda_cache(monkeypatch):
    calls: list[str] = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append("empty"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    release_accelerator_memory()
    assert calls == ["empty"]

    monkeypatch.setitem(sys.modules, "torch", None)
    release_accelerator_memory()
    assert calls == ["empty"]


def test_manifest_helpers_identify_local_models_and_git_state(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    first = hash_model_dir(model_dir)
    assert hash_model_dir(model_dir) == first
    (model_dir / "config.json").write_text('{"a": 1}')
    assert hash_model_dir(model_dir) != first

    described = describe_base_model(str(model_dir), requested_revision="main")
    assert described["local_sha256"] == hash_model_dir(model_dir)
    assert describe_base_model("google/gemma-3-270m-it")["local_sha256"] is None

    assert git_state(tmp_path) == {"commit": None, "dirty": None}
