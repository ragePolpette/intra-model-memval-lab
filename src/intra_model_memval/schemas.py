"""Core schemas for numeric-first intra-model memory records."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    DECISION = "decision"
    FACT = "fact"
    RULE = "rule"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"
    INVALIDATED = "invalidated"


class NumericEncoding(str, Enum):
    BASE64 = "base64"
    NPZ_REF = "npz_ref"
    ARROW_REF = "arrow_ref"


class NumericPayload(BaseModel):
    dtype: str
    shape: list[int]
    encoding: NumericEncoding
    payload_b64: Optional[str] = None
    blob_path: Optional[str] = None
    blob_hash: Optional[str] = None

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("shape must not be empty")
        if any(int(dim) <= 0 for dim in value):
            raise ValueError("shape dimensions must be > 0")
        return [int(dim) for dim in value]


class MemoryRecord(BaseModel):
    entry_id: str
    category: Category
    raw_numeric: NumericPayload
    text_view: Optional[str] = None
    modality_primary: str = Field(default="numeric")
    importance_score: int = 0
    novelty_score: float = 1.0
    is_external: bool = False
    provenance_level: str = "declared_only"
    context_hash: str
    writer_model: str = "unknown-model"
    writer_agent_id: str = "unknown-agent"
    created_at_utc: str
    metadata: dict = Field(default_factory=dict)
    schema_version: str = "v1"
    model_fingerprint: str = "unknown"

    @field_validator("importance_score")
    @classmethod
    def validate_importance_score(cls, value: int) -> int:
        return max(0, min(100, int(value)))

    @field_validator("novelty_score")
    @classmethod
    def validate_novelty_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("modality_primary")
    @classmethod
    def validate_modality_primary(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized != "numeric":
            raise ValueError("modality_primary must be 'numeric'")
        return normalized

    @property
    def train_ready(self) -> bool:
        return self.raw_numeric is not None

