from __future__ import annotations

import re
from typing import Any

from intra_model_memval.adapters import BaseModelAdapter, ForwardPassResult, ModelAdapterSpec, TokenizedInput


class FakeExperimentAdapter(BaseModelAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._adapter_id = str(kwargs.get("adapter_id") or "fake-gpt2-small")
        self._model_id = str(kwargs.get("model_name") or "fake-gpt2")
        self._response_token_map = {
            " Paris": 0,
            " Rome": 1,
            " Madrid": 2,
            " Lisbon": 3,
            " Berlin": 4,
            " Vienna": 5,
            " Brussels": 6,
            " Tokyo": 7,
            " Ottawa": 8,
            " Canberra": 9,
        }
        self._decoded = {value: key for key, value in self._response_token_map.items()}
        self._prompt_token_map: dict[str, int] = {}
        self._country_to_response = {
            "France": " Paris",
            "Italy": " Rome",
            "Spain": " Madrid",
            "Portugal": " Lisbon",
            "Germany": " Berlin",
            "Austria": " Vienna",
            "Belgium": " Brussels",
            "Japan": " Tokyo",
            "Canada": " Ottawa",
            "Australia": " Canberra",
        }

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def load(self) -> "FakeExperimentAdapter":
        return self

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> TokenizedInput:
        if text in self._response_token_map:
            token_id = self._response_token_map[text]
            return TokenizedInput(
                text=text,
                input_ids=[token_id],
                attention_mask=[1],
                tokens=[text],
                metadata={"add_special_tokens": add_special_tokens},
            )

        parts = [part for part in re.split(r"\s+", text.strip()) if part]
        token_ids: list[int] = []
        for part in parts:
            if part not in self._prompt_token_map:
                self._prompt_token_map[part] = 100 + len(self._prompt_token_map)
                self._decoded[self._prompt_token_map[part]] = part
            token_ids.append(self._prompt_token_map[part])
        return TokenizedInput(
            text=text,
            input_ids=token_ids,
            attention_mask=[1] * len(token_ids),
            tokens=parts,
            metadata={"add_special_tokens": add_special_tokens},
        )

    def forward(
        self,
        text: str,
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> ForwardPassResult:
        tokenized = self.tokenize(text)
        logits = self._build_logits(text, len(tokenized.input_ids))
        hidden_states = self._build_hidden_states(len(tokenized.input_ids)) if output_hidden_states else []
        attentions = self._build_attentions(len(tokenized.input_ids)) if output_attentions else None
        return ForwardPassResult(
            adapter_id=self.adapter_id,
            model_id=self.model_id,
            prompt_text=text,
            token_ids=tokenized.input_ids,
            tokens=tokenized.tokens,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            metadata={"seed": 0, "device": "cpu"},
        )

    def decode_token_ids(self, token_ids: list[int]) -> list[str]:
        return [self._decoded.get(token_id, f"tok-{token_id}") for token_id in token_ids]

    def describe(self) -> ModelAdapterSpec:
        return ModelAdapterSpec(
            adapter_id=self.adapter_id,
            model_id=self.model_id,
            family="fake-gpt2",
            device="cpu",
            dtype="float32",
            context_window=64,
            capabilities=["forward", "logits", "hidden_states"],
            metadata={"seed": 0, "n_layer": 2, "vocab_size": 10},
            notes="Test adapter",
        )

    def _build_logits(self, text: str, seq_len: int) -> list[list[list[float]]]:
        scores = [-5.0] * len(self._response_token_map)
        country_match = re.search(r"capital of ([A-Za-z]+)", text)
        if country_match:
            country = country_match.group(1)
            expected = self._country_to_response.get(country)
            if expected is not None:
                scores[self._response_token_map[expected]] = 5.0
        else:
            scores[self._response_token_map[" Paris"]] = 5.0
        return [[list(scores) for _ in range(max(1, seq_len))]]

    @staticmethod
    def _build_hidden_states(seq_len: int) -> list[list[list[list[float]]]]:
        return [
            [[[float(layer_index + token_index)] * 4 for token_index in range(max(1, seq_len))]]
            for layer_index in range(3)
        ]

    @staticmethod
    def _build_attentions(seq_len: int) -> list[list[list[list[float]]]]:
        return [
            [[[1.0 if row == col else 0.0 for col in range(max(1, seq_len))] for row in range(max(1, seq_len))]]
        ]
