from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from intra_model_memval.consolidation import buffer_item_to_messages, resolve_train_config
from intra_model_memval.consolidation.training import TRAIN_DEFAULTS, zip_directory_bytes
from intra_model_memval.domain import SyntheticExample, SyntheticExampleKind

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "consolidation_poc"


def test_buffer_item_to_messages_accepts_examples_and_anchor_dicts():
    example = SyntheticExample(
        lesson_id="lesson-1",
        kind=SyntheticExampleKind.POSITIVE,
        input="A question",
        target="An answer",
        verifier_score=0.9,
    )
    assert buffer_item_to_messages(example) == [
        {"role": "user", "content": "A question"},
        {"role": "assistant", "content": "An answer"},
    ]
    assert buffer_item_to_messages({"input": "Q", "target": "A"})[1]["content"] == "A"

    with pytest.raises(TypeError):
        buffer_item_to_messages("a bare string")
    with pytest.raises(ValueError):
        buffer_item_to_messages({"input": "  ", "target": "A"})


def test_resolve_train_config_merges_defaults():
    resolved = resolve_train_config({"rank": 16, "epochs": None, "adapter": "lora"})
    assert resolved["rank"] == 16
    assert resolved["epochs"] == TRAIN_DEFAULTS["epochs"]
    assert resolved["adapter"] == "lora"
    assert resolved["learning_rate"] == TRAIN_DEFAULTS["learning_rate"]


def test_zip_directory_bytes_round_trips(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sub" / "weights.bin").write_bytes(b"\x00\x01")
    data = zip_directory_bytes(tmp_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert sorted(archive.namelist()) == ["adapter_config.json", "sub/weights.bin"]


def test_training_module_imports_without_ml_dependencies():
    # The core install has no torch/transformers/peft; the module must still import.
    import intra_model_memval.consolidation.training  # noqa: F401


def test_poc_fixtures_are_consistent_with_the_gate():
    from intra_model_memval.consolidation import (
        apply_verifier_gate,
        check_dataset_composition,
        examples_from_jsonl_bytes,
    )
    from intra_model_memval.domain import Lesson
    from intra_model_memval.utils.ids import utc_now_iso

    lesson_payload = json.loads((EXAMPLES / "lesson.json").read_text(encoding="utf-8"))
    now = utc_now_iso()
    lesson = Lesson(**lesson_payload, created_at_utc=now, updated_at_utc=now)

    variations = examples_from_jsonl_bytes((EXAMPLES / "variations.jsonl").read_bytes())
    assert all(item.lesson_id == lesson.lesson_id for item in variations)

    gate = apply_verifier_gate(variations)
    assert len(gate.rejected) == 2
    assert not gate.lesson_needs_reformulation
    assert check_dataset_composition(gate.accepted).is_acceptable

    anchors = [
        json.loads(line)
        for line in (EXAMPLES / "anchors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for anchor in anchors:
        buffer_item_to_messages(anchor)


def test_poc_script_parses_without_ml_dependencies():
    spec = importlib.util.spec_from_file_location(
        "run_consolidation_poc", ROOT / "scripts" / "run_consolidation_poc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "--workdir", "runs/x",
            "--lesson-json", str(EXAMPLES / "lesson.json"),
            "--variations-jsonl", str(EXAMPLES / "variations.jsonl"),
            "--anchors-jsonl", str(EXAMPLES / "anchors.jsonl"),
            "--regression-jsonl", str(EXAMPLES / "regression.jsonl"),
        ]
    )
    assert args.holdout_fraction == 0.2
    lesson = module.load_lesson(EXAMPLES / "lesson.json")
    assert lesson.lesson_id == "lesson-poc-billing-period"
