"""Base contracts for real model adapters used by tracing and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ModelAdapterError(RuntimeError):
    """Base error for adapter failures."""


class ModelAdapterDependencyError(ModelAdapterError):
    """Raised when optional runtime dependencies are not available."""


@dataclass(slots=True)
class TokenizedInput:
    text: str
    input_ids: list[int]
    attention_mask: list[int] | None = None
    tokens: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForwardPassResult:
    adapter_id: str
    model_id: str
    prompt_text: str
    token_ids: list[int]
    tokens: list[str]
    logits: Any
    hidden_states: list[Any]
    attentions: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapterSpec(BaseModel):
    adapter_id: str
    model_id: str
    family: str
    device: str = "cpu"
    dtype: str | None = None
    context_window: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class BaseModelAdapter(ABC):
    """Minimal contract for model adapters used by the experiment framework."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> "BaseModelAdapter":
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> TokenizedInput:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        text: str,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> ForwardPassResult:
        raise NotImplementedError

    @abstractmethod
    def decode_token_ids(self, token_ids: list[int]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> ModelAdapterSpec:
        raise NotImplementedError
