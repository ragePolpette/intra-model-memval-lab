"""Experiment infrastructure for episode, trace, update and evaluation workflows."""

from .adapters import (
    BaseModelAdapter,
    ForwardPassResult,
    GPT2SmallAdapter,
    ModelAdapterDependencyError,
    ModelAdapterError,
    ModelAdapterSpec,
    TokenizedInput,
)
from .domain import (
    ArtifactRef,
    EpisodeRecord,
    EvaluationCase,
    EvaluationRun,
    EvaluationSpec,
    ExperimentRun,
    ExperimentStatus,
    RunKind,
    SourceProvenance,
    TraceArtifact,
    TraceStatus,
    UpdateCandidate,
    UpdateCandidateStatus,
)
from .evaluation import (
    EvaluationExecutionResult,
    EvaluationHarness,
    SyntheticDataset,
    SyntheticDatasetCase,
    default_gpt2_dataset_path,
    default_gpt2_prompt_template_path,
    load_synthetic_dataset,
)
from .ingestion import EpisodeIngestionService
from .persistence import ExperimentStore, StoredArtifact
from .selection import EpisodeSelectionPolicy, EpisodeSelectionResult, select_episodes
from .tracing import TraceExecutionResult, TraceRunner

__all__ = [
    "ArtifactRef",
    "BaseModelAdapter",
    "EpisodeIngestionService",
    "EpisodeRecord",
    "EpisodeSelectionPolicy",
    "EpisodeSelectionResult",
    "EvaluationCase",
    "EvaluationExecutionResult",
    "EvaluationHarness",
    "EvaluationRun",
    "EvaluationSpec",
    "ExperimentRun",
    "ExperimentStatus",
    "ExperimentStore",
    "ForwardPassResult",
    "GPT2SmallAdapter",
    "ModelAdapterDependencyError",
    "ModelAdapterError",
    "ModelAdapterSpec",
    "RunKind",
    "SourceProvenance",
    "StoredArtifact",
    "SyntheticDataset",
    "SyntheticDatasetCase",
    "TokenizedInput",
    "TraceArtifact",
    "TraceExecutionResult",
    "TraceRunner",
    "TraceStatus",
    "UpdateCandidate",
    "UpdateCandidateStatus",
    "default_gpt2_dataset_path",
    "default_gpt2_prompt_template_path",
    "load_synthetic_dataset",
    "select_episodes",
]
