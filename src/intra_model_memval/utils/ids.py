"""ID and timestamp helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    normalized = str(prefix).strip().lower().replace(" ", "-")
    return f"{normalized}-{uuid4().hex[:12]}"
