from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from intra_model_memval.schemas import MemoryRecord
from intra_model_memval.storage import MemoryPersistence


def _record(
    entry_id: str,
    *,
    importance: int,
    novelty: float,
    is_external: bool,
    writer_model: str,
    conversation_id: str,
) -> MemoryRecord:
    payload = bytes([max(1, min(255, importance % 255))])
    return MemoryRecord(
        entry_id=entry_id,
        category="fact",
        raw_numeric={
            "dtype": "float32",
            "shape": [1],
            "encoding": "base64",
            "payload_b64": base64.b64encode(payload).decode("ascii"),
        },
        text_view=f"text-{entry_id}",
        modality_primary="numeric",
        importance_score=importance,
        novelty_score=novelty,
        is_external=is_external,
        provenance_level="verified_tool" if is_external else "declared_only",
        context_hash=f"ctx-{entry_id}",
        writer_model=writer_model,
        writer_agent_id="agent-script",
        created_at_utc="2026-03-05T00:00:00+00:00",
        metadata={"conversation_id": conversation_id},
    )


def test_build_dataset_script_exports_selected_rows(tmp_path: Path):
    db_path = tmp_path / "memval.db"
    blob_dir = tmp_path / "blobs"
    persistence = MemoryPersistence(db_path=db_path, blob_dir=blob_dir)

    rows: list[MemoryRecord] = []
    for idx in range(8):
        rows.append(
            _record(
                f"top-{idx}",
                importance=95 - idx,
                novelty=0.9,
                is_external=idx < 2,
                writer_model=f"wm-top-{idx % 3}",
                conversation_id=f"conv-top-{idx}",
            )
        )
    for idx in range(5):
        rows.append(
            _record(
                f"mid-{idx}",
                importance=55 - idx,
                novelty=0.85,
                is_external=idx == 0,
                writer_model=f"wm-mid-{idx % 2}",
                conversation_id=f"conv-mid-{idx}",
            )
        )
    for idx in range(4):
        rows.append(
            _record(
                f"low-{idx}",
                importance=20 - idx,
                novelty=0.8 if idx < 3 else 0.1,
                is_external=True,
                writer_model=f"wm-low-{idx}",
                conversation_id=f"conv-low-{idx}",
            )
        )
    persistence.save_many(rows)

    output_numeric = tmp_path / "out" / "numeric.jsonl"
    output_text = tmp_path / "out" / "text.jsonl"

    cmd = [
        sys.executable,
        "scripts/build_dataset.py",
        "--db-path",
        str(db_path),
        "--blob-dir",
        str(blob_dir),
        "--output-numeric",
        str(output_numeric),
        "--output-text",
        str(output_text),
        "--sample-size",
        "10",
        "--external-min-ratio",
        "0.30",
        "--max-per-writer",
        "1.0",
        "--max-per-conversation",
        "1.0",
        "--seed",
        "11",
    ]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout.strip())

    assert summary["input_rows"] == 17
    assert summary["after_novelty_filter"] == 16
    assert summary["selected_rows"] == 10
    assert summary["numeric_rows"] == 10
    assert summary["text_rows"] == 10
    assert summary["external_target_met"] is True

    numeric_lines = output_numeric.read_text(encoding="utf-8").splitlines()
    text_lines = output_text.read_text(encoding="utf-8").splitlines()
    assert len(numeric_lines) == 10
    assert len(text_lines) == 10

    numeric_rows = [json.loads(line) for line in numeric_lines]
    text_rows = [json.loads(line) for line in text_lines]
    assert all("raw_numeric" in row for row in numeric_rows)
    assert all("text_view" in row for row in text_rows)
