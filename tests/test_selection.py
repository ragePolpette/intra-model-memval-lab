from __future__ import annotations

import base64
from pathlib import Path

import pytest

from intra_model_memval.schemas import MemoryRecord
from intra_model_memval.selection import SelectionPolicy, select_training_records
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
        text_view=f"row-{entry_id}",
        modality_primary="numeric",
        importance_score=importance,
        novelty_score=novelty,
        is_external=is_external,
        provenance_level="verified_tool" if is_external else "declared_only",
        context_hash=f"ctx-{entry_id[-8:]}",
        writer_model=writer_model,
        writer_agent_id="agent-test",
        created_at_utc=f"2026-03-05T00:00:{int(entry_id.split('-')[-1]) % 60:02d}+00:00",
        metadata={"conversation_id": conversation_id},
    )


@pytest.fixture
def persistence(tmp_path: Path) -> MemoryPersistence:
    return MemoryPersistence(
        db_path=tmp_path / "selection.db",
        blob_dir=tmp_path / "blobs",
    )


def test_selection_applies_novelty_filter_and_bucket_targets(persistence: MemoryPersistence):
    records: list[MemoryRecord] = []
    counter = 0
    for idx in range(12):
        records.append(
            _record(
                f"top-{counter}",
                importance=95 - idx,
                novelty=0.95,
                is_external=idx % 4 == 0,
                writer_model=f"wm-{idx % 4}",
                conversation_id=f"conv-top-{idx}",
            )
        )
        counter += 1
    for idx in range(5):
        records.append(
            _record(
                f"mid-{counter}",
                importance=60 - idx,
                novelty=0.9,
                is_external=False,
                writer_model=f"wm-mid-{idx}",
                conversation_id=f"conv-mid-{idx}",
            )
        )
        counter += 1
    for idx in range(3):
        records.append(
            _record(
                f"low-{counter}",
                importance=20 - idx,
                novelty=0.85,
                is_external=True,
                writer_model=f"wm-low-{idx}",
                conversation_id=f"conv-low-{idx}",
            )
        )
        counter += 1
    for idx in range(4):
        records.append(
            _record(
                f"filtered-{counter}",
                importance=99 - idx,
                novelty=0.1,
                is_external=True,
                writer_model=f"wm-filter-{idx}",
                conversation_id=f"conv-filter-{idx}",
            )
        )
        counter += 1

    persistence.save_many(records)
    result = select_training_records(
        persistence,
        SelectionPolicy(
            sample_size=20,
            external_min_ratio=0.0,
            max_per_writer=1.0,
            max_per_conversation=1.0,
            seed=42,
        ),
    )

    assert result.stats.after_novelty_filter == 20
    assert result.stats.selected == 20
    assert result.stats.top_count == 12
    assert result.stats.mid_count == 5
    assert result.stats.low_count == 3


def test_selection_enforces_external_quota_with_swap(persistence: MemoryPersistence):
    records: list[MemoryRecord] = []
    counter = 0
    for idx in range(10):
        records.append(
            _record(
                f"int-top-{counter}",
                importance=90 - idx,
                novelty=0.95,
                is_external=False,
                writer_model=f"wm-top-{idx % 3}",
                conversation_id=f"conv-top-{idx}",
            )
        )
        counter += 1
    for idx in range(2):
        records.append(
            _record(
                f"int-mid-{counter}",
                importance=55 - idx,
                novelty=0.9,
                is_external=False,
                writer_model=f"wm-mid-{idx}",
                conversation_id=f"conv-mid-{idx}",
            )
        )
        counter += 1
    for idx in range(8):
        records.append(
            _record(
                f"ext-low-{counter}",
                importance=20 - idx,
                novelty=0.85,
                is_external=True,
                writer_model=f"wm-low-{idx % 4}",
                conversation_id=f"conv-low-{idx}",
            )
        )
        counter += 1

    persistence.save_many(records)
    result = select_training_records(
        persistence,
        SelectionPolicy(
            sample_size=10,
            external_min_ratio=0.4,
            max_per_writer=1.0,
            max_per_conversation=1.0,
            seed=7,
        ),
    )

    external_count = sum(1 for item in result.records if item.is_external)
    assert result.stats.selected == 10
    assert result.stats.required_external == 4
    assert result.stats.external_target_met is True
    assert external_count >= 4


def test_selection_applies_writer_and_conversation_caps(persistence: MemoryPersistence):
    records: list[MemoryRecord] = []
    counter = 0

    for idx in range(12):
        records.append(
            _record(
                f"dom-{counter}",
                importance=95 - idx,
                novelty=0.92,
                is_external=idx % 2 == 0,
                writer_model="dominant-writer",
                conversation_id="dominant-conversation",
            )
        )
        counter += 1

    for idx in range(18):
        records.append(
            _record(
                f"mix-{counter}",
                importance=80 - (idx % 20),
                novelty=0.9,
                is_external=idx % 3 == 0,
                writer_model=f"wm-{idx % 9}",
                conversation_id=f"conv-{idx % 12}",
            )
        )
        counter += 1

    persistence.save_many(records)
    result = select_training_records(
        persistence,
        SelectionPolicy(
            sample_size=10,
            external_min_ratio=0.0,
            max_per_writer=0.3,
            max_per_conversation=0.2,
            seed=13,
        ),
    )

    writer_counts: dict[str, int] = {}
    conversation_counts: dict[str, int] = {}
    for item in result.records:
        writer_counts[item.writer_model] = writer_counts.get(item.writer_model, 0) + 1
        conversation_id = str(item.metadata.get("conversation_id") or "")
        if conversation_id:
            conversation_counts[conversation_id] = conversation_counts.get(conversation_id, 0) + 1

    assert result.stats.selected == 10
    assert max(writer_counts.values()) <= 3
    assert max(conversation_counts.values()) <= 2
    assert writer_counts.get("dominant-writer", 0) <= 3
    assert conversation_counts.get("dominant-conversation", 0) <= 2


def test_selection_rejects_invalid_ratios(persistence: MemoryPersistence):
    with pytest.raises(ValueError):
        select_training_records(
            persistence,
            SelectionPolicy(top_ratio=0.5, mid_ratio=0.3, low_ratio=0.3),
        )
