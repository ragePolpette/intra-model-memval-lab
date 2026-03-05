from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest

from intra_model_memval.schemas import MemoryRecord
from intra_model_memval.storage import MemoryPersistence
from intra_model_memval.self_eval import SelfEvalValidationError


def _record(entry_id: str, payload: bytes, *, importance_score: int = 50) -> MemoryRecord:
    return MemoryRecord(
        entry_id=entry_id,
        category="fact",
        raw_numeric={
            "dtype": "float32",
            "shape": [len(payload)],
            "encoding": "base64",
            "payload_b64": base64.b64encode(payload).decode("ascii"),
        },
        text_view=f"text-{entry_id}",
        modality_primary="numeric",
        importance_score=importance_score,
        novelty_score=0.9,
        is_external=True,
        provenance_level="verified_tool",
        context_hash="abcd1234efef9999",
        writer_model="gpt-5",
        writer_agent_id="agent-1",
        created_at_utc="2026-03-05T00:00:00+00:00",
    )


@pytest.fixture
def persistence(tmp_path: Path) -> MemoryPersistence:
    return MemoryPersistence(
        db_path=tmp_path / "memval.db",
        blob_dir=tmp_path / "blobs",
    )


def test_save_and_load_single_record(persistence: MemoryPersistence):
    record = _record("e-1", b"\x01\x02\x03")
    saved = persistence.save_memory_record(record)

    assert saved.raw_numeric.blob_hash
    assert saved.raw_numeric.blob_path
    assert Path(saved.raw_numeric.blob_path).exists()

    loaded = persistence.load_memory_record("e-1")
    assert loaded is not None
    assert loaded.entry_id == "e-1"
    assert loaded.text_view == "text-e-1"
    assert loaded.raw_numeric.blob_hash == saved.raw_numeric.blob_hash


def test_idempotent_upsert_keeps_single_record(persistence: MemoryPersistence):
    first = _record("same-id", b"\x11\x22", importance_score=20)
    second = _record("same-id", b"\x11\x22\x33", importance_score=90)

    persistence.save_memory_record(first)
    persistence.save_memory_record(second)

    assert persistence.count_records() == 1
    loaded = persistence.load_memory_record("same-id")
    assert loaded is not None
    assert loaded.importance_score == 90


def test_save_many_batch(persistence: MemoryPersistence):
    records = [_record(f"e-{idx}", bytes([idx, idx + 1, idx + 2])) for idx in range(10)]
    saved = persistence.save_many(records)

    assert len(saved) == 10
    assert persistence.count_records() == 10
    assert persistence.count_blobs() == 10


def test_rollback_logical_on_upsert_failure(persistence: MemoryPersistence, monkeypatch: pytest.MonkeyPatch):
    record = _record("e-rollback", b"\xAA\xBB\xCC")

    def _raise(*args, **kwargs):
        raise RuntimeError("forced-upsert-failure")

    monkeypatch.setattr(persistence, "_upsert_record", _raise)

    with pytest.raises(RuntimeError):
        persistence.save_memory_record(record)

    assert persistence.count_records() == 0
    assert persistence.count_blobs() == 0
    assert not list(persistence.blob_dir.glob("*.bin"))


def test_batch_smoke_performance(persistence: MemoryPersistence):
    records = [_record(f"p-{idx}", bytes([idx % 255] * 16)) for idx in range(100)]
    start = time.perf_counter()
    persistence.save_many(records)
    elapsed = time.perf_counter() - start

    assert persistence.count_records() == 100
    assert elapsed < 5.0


def test_self_eval_enforced_rejects_missing_fields(tmp_path: Path):
    persistence = MemoryPersistence(
        db_path=tmp_path / "enf.db",
        blob_dir=tmp_path / "blobs",
        self_eval_enforced=True,
    )
    record = _record("e-enf-miss", b"\x01\x02")

    with pytest.raises(SelfEvalValidationError):
        persistence.save_memory_record(record)


def test_self_eval_enforced_computes_scores_and_context_hash(tmp_path: Path):
    persistence = MemoryPersistence(
        db_path=tmp_path / "enf-ok.db",
        blob_dir=tmp_path / "blobs",
        self_eval_enforced=True,
    )
    record = _record("e-enf-ok", b"\xAA\xBB\xCC")
    record.metadata = {
        "project_id": "prj-a",
        "scope": "shared",
        "top_similarities": [0.5, 0.2],
        "context_fingerprint": {
            "conversation_id": "conv-1",
            "task_id": "task-1",
            "retrieved_ids": ["d2", "d1", "d1"],
            "tool_trace_fingerprint": {"tool": "db_read", "rows": 1},
            "prompt_fingerprint": "pf-1",
        },
        "importance": {
            "confidence": 0.2,
            "tool_steps": 10,
            "correction_count": 2,
            "inference_level": 4,
            "negative_impact": 0.2,
            "is_external": True,
        },
    }

    saved = persistence.save_memory_record(record)
    loaded = persistence.load_memory_record("e-enf-ok")
    assert loaded is not None

    assert saved.importance_score == 75
    assert saved.novelty_score == 0.5
    assert len(saved.context_hash) == 16
    assert saved.metadata["surprise_source"] == "confidence"
    assert saved.metadata["novelty_computed"] is True
    assert loaded.importance_score == 75
    assert loaded.context_hash == saved.context_hash
