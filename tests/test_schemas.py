from __future__ import annotations

import pytest

from intra_model_memval.schemas import MemoryRecord, NumericPayload


def test_numeric_payload_requires_valid_shape():
    payload = NumericPayload(
        dtype="float32",
        shape=[384],
        encoding="base64",
        payload_b64="AAAA",
    )
    assert payload.shape == [384]

    with pytest.raises(ValueError):
        NumericPayload(
            dtype="float32",
            shape=[0],
            encoding="base64",
            payload_b64="AAAA",
        )


def test_memory_record_numeric_first_constraints():
    record = MemoryRecord(
        entry_id="e-1",
        category="fact",
        raw_numeric={
            "dtype": "float32",
            "shape": [128],
            "encoding": "base64",
            "payload_b64": "AAAA",
        },
        text_view="debug text",
        modality_primary="numeric",
        importance_score=105,
        novelty_score=1.5,
        context_hash="abcd1234efef9999",
        created_at_utc="2026-03-05T00:00:00+00:00",
    )
    assert record.importance_score == 100
    assert record.novelty_score == 1.0
    assert record.train_ready is True

    with pytest.raises(ValueError):
        MemoryRecord(
            entry_id="e-2",
            category="fact",
            raw_numeric={
                "dtype": "float32",
                "shape": [128],
                "encoding": "base64",
                "payload_b64": "AAAA",
            },
            modality_primary="text",
            context_hash="abcd1234efef9999",
            created_at_utc="2026-03-05T00:00:00+00:00",
        )
