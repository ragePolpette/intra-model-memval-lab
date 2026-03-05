from __future__ import annotations

from intra_model_memval.schemas import MemoryRecord
from intra_model_memval.self_eval import enrich_self_evaluation


def test_self_eval_context_hash_determinism():
    base = MemoryRecord(
        entry_id="e-det",
        category="fact",
        raw_numeric={
            "dtype": "float32",
            "shape": [3],
            "encoding": "base64",
            "payload_b64": "AAAA",
        },
        writer_model="gpt-5",
        writer_agent_id="agent-a",
        context_hash="placeholder",
        created_at_utc="2026-03-05T00:00:00+00:00",
        metadata={
            "project_id": "prj-1",
            "scope": "shared",
            "context_fingerprint": {
                "conversation_id": "c1",
                "task_id": "t1",
                "retrieved_ids": ["b", "a", "a"],
                "tool_trace_fingerprint": {"x": 1, "y": 2},
                "prompt_fingerprint": "p1",
            },
            "importance": {
                "self_rating": 0.6,
                "inference_level": 2,
            },
        },
    )

    variant = base.model_copy(deep=True)
    variant.metadata["context_fingerprint"]["retrieved_ids"] = ["a", "b"]
    variant.metadata["context_fingerprint"]["tool_trace_fingerprint"] = {"y": 2, "x": 1}

    a = enrich_self_evaluation(base, enforce=False)
    b = enrich_self_evaluation(variant, enforce=False)
    assert a.context_hash == b.context_hash
    assert len(a.context_hash) == 16
