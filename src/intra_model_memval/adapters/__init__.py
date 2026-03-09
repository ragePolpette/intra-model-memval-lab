"""Reusable model adapter contracts and concrete adapters."""

from .base import (
    BaseModelAdapter,
    ForwardPassResult,
    ModelAdapterDependencyError,
    ModelAdapterError,
    ModelAdapterSpec,
    TokenizedInput,
)
from .gpt2_adapter import GPT2SmallAdapter

__all__ = [
    "BaseModelAdapter",
    "ForwardPassResult",
    "GPT2SmallAdapter",
    "ModelAdapterDependencyError",
    "ModelAdapterError",
    "ModelAdapterSpec",
    "TokenizedInput",
]
