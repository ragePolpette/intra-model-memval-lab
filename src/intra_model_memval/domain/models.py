"""Core domain contracts for experiment episodes, traces, updates and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceProvenance(BaseModel):
    source_type: str = "unknown"
    source_label: str | None = None
    source_uri: str | None = None
    collected_by: str | None = None
    is_external: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("source_type must not be empty")
        return normalized


class EpisodeRecord(BaseModel):
    episode_id: str
    content_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    topic_tags: list[str] = Field(default_factory=list)
    trigger_tags: list[str] = Field(default_factory=list)
    provenance: SourceProvenance = Field(default_factory=SourceProvenance)
    observed_at_utc: str
    created_at_utc: str
    context_hash: str
    conversation_id: str | None = None
    session_id: str | None = None
    notes: str | None = None
    schema_version: str = "episode-v1"

    @field_validator("content_text")
    @classmethod
    def validate_content_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("content_text must not be empty")
        return normalized

    @field_validator("topic_tags", "trigger_tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        tags = [str(item).strip().lower() for item in value if str(item).strip()]
        return sorted(set(tags))


class ArtifactRef(BaseModel):
    artifact_id: str | None = None
    role: str = "generic"
    path: str
    media_type: str | None = None
    size_bytes: int | None = None
    artifact_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("role must not be empty")
        return normalized

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("path must not be empty")
        return normalized


class TraceStatus(str, Enum):
    PLACEHOLDER = "placeholder"
    REGISTERED = "registered"
    MATERIALIZED = "materialized"
    FAILED = "failed"


class TraceArtifact(BaseModel):
    trace_id: str
    episode_id: str
    run_id: str
    adapter_id: str
    trace_input_spec: dict[str, Any] = Field(default_factory=dict)
    trace_type: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    status: TraceStatus = TraceStatus.PLACEHOLDER
    reproducibility_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    updated_at_utc: str
    notes: str | None = None
    schema_version: str = "trace-v1"

    @field_validator("adapter_id", "trace_type")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("field must not be empty")
        return normalized


class UpdateCandidateStatus(str, Enum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EVALUATED = "evaluated"


class UpdateCandidate(BaseModel):
    update_candidate_id: str
    episode_id: str
    trace_id: str | None = None
    run_id: str
    target_fact_spec: dict[str, Any]
    candidate_localization: dict[str, Any] = Field(default_factory=dict)
    update_budget: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str
    status: UpdateCandidateStatus = UpdateCandidateStatus.PROPOSED
    evaluation_spec_id: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    updated_at_utc: str
    schema_version: str = "update-candidate-v1"

    @field_validator("hypothesis")
    @classmethod
    def validate_hypothesis(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("hypothesis must not be empty")
        return normalized


class EvaluationCase(BaseModel):
    case_id: str
    prompt: str
    expected_response: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "expected_response")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("evaluation case fields must not be empty")
        return normalized


class EvaluationSpec(BaseModel):
    evaluation_spec_id: str
    name: str
    description: str | None = None
    target_facts: list[EvaluationCase] = Field(default_factory=list)
    related_facts: list[EvaluationCase] = Field(default_factory=list)
    unrelated_facts: list[EvaluationCase] = Field(default_factory=list)
    regression_rules: list[dict[str, Any] | str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    schema_version: str = "evaluation-spec-v1"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("name must not be empty")
        return normalized


class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunKind(str, Enum):
    INGESTION = "ingestion"
    TRACE = "trace"
    UPDATE_CANDIDATE = "update_candidate"
    EVALUATION = "evaluation"
    EXPORT = "export"


class ExperimentRun(BaseModel):
    run_id: str
    run_kind: RunKind
    status: ExperimentStatus = ExperimentStatus.PLANNED
    title: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str
    updated_at_utc: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("title must not be empty")
        return normalized


class EvaluationRun(BaseModel):
    evaluation_run_id: str
    evaluation_spec_id: str
    run_id: str
    subject_type: str
    subject_id: str
    status: ExperimentStatus = ExperimentStatus.PLANNED
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    created_at_utc: str
    updated_at_utc: str
    schema_version: str = "evaluation-run-v1"

    @field_validator("subject_type", "subject_id")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("subject fields must not be empty")
        return normalized
