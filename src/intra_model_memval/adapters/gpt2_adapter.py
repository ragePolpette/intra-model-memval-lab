"""Concrete GPT-2 small adapter built on top of Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    BaseModelAdapter,
    ForwardPassResult,
    ModelAdapterDependencyError,
    ModelAdapterSpec,
    TokenizedInput,
)


@dataclass(slots=True)
class _LoadedResources:
    torch: Any
    tokenizer: Any
    model: Any
    transformers_version: str | None


class GPT2SmallAdapter(BaseModelAdapter):
    """Load GPT-2 small in eval mode and expose reproducible forward tracing."""

    def __init__(
        self,
        *,
        model_name: str = "gpt2",
        adapter_id: str = "gpt2-small",
        device: str = "cpu",
        dtype: str | None = None,
        seed: int = 0,
        revision: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._adapter_id = adapter_id
        self.device = device
        self.dtype = dtype or "float32"
        self.seed = int(seed)
        self.revision = revision
        self._resources: _LoadedResources | None = None

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def model_id(self) -> str:
        return self._model_name

    @staticmethod
    def _load_dependencies() -> tuple[Any, Any, Any, str | None]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise ModelAdapterDependencyError(
                "torch is required to use GPT2SmallAdapter"
            ) from exc

        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise ModelAdapterDependencyError(
                "transformers is required to use GPT2SmallAdapter"
            ) from exc

        return torch, AutoModelForCausalLM, AutoTokenizer, getattr(transformers, "__version__", None)

    def load(self) -> "GPT2SmallAdapter":
        if self._resources is not None:
            return self

        torch, auto_model_cls, auto_tokenizer_cls, transformers_version = self._load_dependencies()
        torch.manual_seed(self.seed)
        if hasattr(torch, "cuda") and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        tokenizer = auto_tokenizer_cls.from_pretrained(self._model_name, revision=self.revision)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = auto_model_cls.from_pretrained(self._model_name, revision=self.revision)
        model.eval()
        if hasattr(model, "to"):
            model = model.to(self.device)

        self._resources = _LoadedResources(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            transformers_version=transformers_version,
        )
        return self

    def _require_resources(self) -> _LoadedResources:
        if self._resources is None:
            self.load()
        assert self._resources is not None
        return self._resources

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> TokenizedInput:
        resources = self._require_resources()
        encoded = resources.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=True,
        )
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        attention_mask = [int(value) for value in encoded.get("attention_mask", [])] or None
        tokens = resources.tokenizer.convert_ids_to_tokens(input_ids)
        return TokenizedInput(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=[str(token) for token in tokens],
            metadata={"add_special_tokens": add_special_tokens},
        )

    def forward(
        self,
        text: str,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> ForwardPassResult:
        resources = self._require_resources()
        tokenized = self.tokenize(text, add_special_tokens=False)
        encoded = resources.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with resources.torch.inference_mode():
            outputs = resources.model(
                **encoded,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )

        hidden_states = list(outputs.hidden_states) if output_hidden_states and outputs.hidden_states is not None else []
        attentions = list(outputs.attentions) if output_attentions and outputs.attentions is not None else None

        return ForwardPassResult(
            adapter_id=self.adapter_id,
            model_id=self.model_id,
            prompt_text=text,
            token_ids=tokenized.input_ids,
            tokens=tokenized.tokens,
            logits=outputs.logits.detach().cpu(),
            hidden_states=[tensor.detach().cpu() for tensor in hidden_states],
            attentions=[tensor.detach().cpu() for tensor in attentions] if attentions is not None else None,
            metadata={
                "device": self.device,
                "dtype": self.dtype,
                "seed": self.seed,
                "transformers_version": resources.transformers_version,
            },
        )

    def decode_token_ids(self, token_ids: list[int]) -> list[str]:
        resources = self._require_resources()
        return [str(token) for token in resources.tokenizer.convert_ids_to_tokens(token_ids)]

    def describe(self) -> ModelAdapterSpec:
        resources = self._require_resources()
        config = getattr(resources.model, "config", None)
        metadata = {
            "seed": self.seed,
            "revision": self.revision,
            "n_layer": getattr(config, "n_layer", None),
            "n_head": getattr(config, "n_head", None),
            "n_embd": getattr(config, "n_embd", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "eos_token_id": getattr(resources.tokenizer, "eos_token_id", None),
            "transformers_version": resources.transformers_version,
            "torch_version": getattr(resources.torch, "__version__", None),
        }
        return ModelAdapterSpec(
            adapter_id=self.adapter_id,
            model_id=self.model_id,
            family="gpt2",
            device=self.device,
            dtype=self.dtype,
            context_window=getattr(config, "n_positions", None),
            capabilities=["forward", "logits", "hidden_states", "next_token_scoring"],
            metadata=metadata,
            notes="Baseline GPT-2 small adapter for tracing and pre-update evaluation.",
        )
