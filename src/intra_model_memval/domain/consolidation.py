"""Domain contracts for the sleep-cycle consolidation pipeline (design contract v0.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .models import ArtifactRef, Rubric


class LessonStatus(str, Enum):
    CANDIDATE = "candidate"
    CONSOLIDATED = "consolidated"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"


class Lesson(BaseModel):
    lesson_id: str
    statement: str
    scope_tags: list[str] = Field(default_factory=list)
    # boundary is mandatory and non-empty: a lesson without explicit limits produces
    # variations that overgeneralize, the main failure mode of synthetic data.
    boundary: str
    source_cluster_ids: list[str] = Field(default_factory=list)
    source_episode_ids: list[str] = Field(default_factory=list)
    status: LessonStatus = LessonStatus.CANDIDATE
    consolidation_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    updated_at_utc: str
    schema_version: str = "lesson-v1"

    @field_validator("statement", "boundary")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized

    @field_validator("scope_tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        tags = [str(item).strip().lower() for item in value if str(item).strip()]
        return sorted(set(tags))


class SyntheticExampleKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANCHOR = "anchor"


class SyntheticExample(BaseModel):
    lesson_id: str
    kind: SyntheticExampleKind
    input: str
    target: str
    verifier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rubric: Rubric | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "synthetic-example-v1"

    @field_validator("lesson_id", "input", "target")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized


class ConsolidationCandidateStatus(str, Enum):
    PREPARED = "prepared"
    TRAINED = "trained"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class ConsolidationCandidate(BaseModel):
    consolidation_candidate_id: str
    lesson_ids: list[str]
    dataset_ref: ArtifactRef
    replay_config: dict[str, Any] = Field(default_factory=dict)
    train_config: dict[str, Any] = Field(default_factory=dict)
    base_model_ref: str
    status: ConsolidationCandidateStatus = ConsolidationCandidateStatus.PREPARED
    evaluation_run_ids: list[str] = Field(default_factory=list)
    run_id: str
    result_summary: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    updated_at_utc: str
    schema_version: str = "consolidation-candidate-v1"

    @field_validator("lesson_ids")
    @classmethod
    def validate_lesson_ids(cls, value: list[str]) -> list[str]:
        ids = [str(item).strip() for item in value if str(item).strip()]
        if not ids:
            raise ValueError("lesson_ids must not be empty")
        return ids

    @field_validator("base_model_ref")
    @classmethod
    def validate_base_model_ref(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("base_model_ref must not be empty")
        return normalized
