"""Contracts for future model adapters without implementing tracing or updates yet."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelAdapterSpec(BaseModel):
    adapter_id: str
    family: str
    capabilities: list[str] = Field(default_factory=list)
    notes: str | None = None
