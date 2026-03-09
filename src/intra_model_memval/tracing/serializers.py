"""Serialization helpers for trace artifacts."""

from __future__ import annotations

import io
import json
from typing import Any

from ..adapters import BaseModelAdapter, ForwardPassResult


def _shape_of(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    if isinstance(value, (list, tuple)) and value:
        return [len(value), *_shape_of(value[0])]
    if isinstance(value, (list, tuple)):
        return [0]
    return []


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_list(item) for item in value]
    if isinstance(value, list):
        return [_to_list(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_list(item) for key, item in value.items()}
    return value


def _final_token_scores(logits: Any) -> list[float]:
    values = _to_list(logits)
    if not isinstance(values, list):
        raise ValueError("logits must be tensor-like or nested lists")
    current = values
    while isinstance(current, list) and current and isinstance(current[0], list):
        current = current[-1]
    return [float(score) for score in current]


def build_top_token_summary(
    adapter: BaseModelAdapter,
    result: ForwardPassResult,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    scores = _final_token_scores(result.logits)
    ranked_ids = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: max(1, top_k)]
    decoded = adapter.decode_token_ids(ranked_ids)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token_text": decoded[rank - 1],
            "logit": float(scores[token_id]),
        }
        for rank, token_id in enumerate(ranked_ids, start=1)
    ]


def serialize_trace_summary(
    adapter: BaseModelAdapter,
    result: ForwardPassResult,
    *,
    input_text: str,
    prompt_template: str,
) -> bytes:
    payload = {
        "adapter_id": result.adapter_id,
        "model_id": result.model_id,
        "input_text": input_text,
        "prompt_text": result.prompt_text,
        "prompt_template": prompt_template,
        "token_ids": result.token_ids,
        "tokens": result.tokens,
        "logits_shape": _shape_of(result.logits),
        "hidden_state_shapes": [_shape_of(item) for item in result.hidden_states],
        "attention_shapes": [_shape_of(item) for item in result.attentions or []],
        "top_next_tokens": build_top_token_summary(adapter, result, top_k=10),
        "metadata": result.metadata,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")


def serialize_trace_tensors(result: ForwardPassResult) -> tuple[bytes, str, str]:
    payload = {
        "adapter_id": result.adapter_id,
        "model_id": result.model_id,
        "prompt_text": result.prompt_text,
        "token_ids": result.token_ids,
        "tokens": result.tokens,
        "logits": result.logits,
        "hidden_states": result.hidden_states,
        "attentions": result.attentions,
        "metadata": result.metadata,
    }

    try:
        import torch
    except ImportError:  # pragma: no cover - depends on local runtime
        serialized = json.dumps(_to_list(payload), ensure_ascii=True, sort_keys=True).encode("utf-8")
        return serialized, ".json", "application/json"

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue(), ".pt", "application/x-pytorch"
