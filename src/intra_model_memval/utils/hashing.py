"""Stable hashing helpers used across experiment entities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def stable_digest(value: Any, *, length: int = 16) -> str:
    normalized = _normalize(value)
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def compute_episode_context_hash(
    *,
    content_text: str,
    topic_tags: list[str],
    trigger_tags: list[str],
    conversation_id: str | None,
    session_id: str | None,
    source_type: str,
    source_label: str | None,
) -> str:
    return stable_digest(
        {
            "content_text": str(content_text).strip(),
            "topic_tags": sorted(str(item).strip().lower() for item in topic_tags if str(item).strip()),
            "trigger_tags": sorted(str(item).strip().lower() for item in trigger_tags if str(item).strip()),
            "conversation_id": str(conversation_id or "").strip(),
            "session_id": str(session_id or "").strip(),
            "source_type": str(source_type).strip().lower(),
            "source_label": str(source_label or "").strip(),
        }
    )
