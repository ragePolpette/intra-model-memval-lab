"""Metrics for baseline next-token evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..adapters import BaseModelAdapter, ForwardPassResult


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_list(item) for item in value]
    if isinstance(value, list):
        return [_to_list(item) for item in value]
    return value


def _final_token_scores(logits: Any) -> list[float]:
    values = _to_list(logits)
    if not isinstance(values, list):
        raise ValueError("logits must be tensor-like or nested lists")
    current = values
    while isinstance(current, list) and current and isinstance(current[0], list):
        current = current[-1]
    return [float(score) for score in current]


def _rank_index(scores: list[float], target_index: int) -> int:
    target_score = scores[target_index]
    return 1 + sum(1 for score in scores if score > target_score)


def _softmax_logprob(scores: list[float], index: int) -> float:
    max_score = max(scores)
    exp_sum = sum(math.exp(score - max_score) for score in scores)
    return float(scores[index] - max_score - math.log(exp_sum))


def score_case_prediction(
    adapter: BaseModelAdapter,
    result: ForwardPassResult,
    expected_response: str,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    scoring_response = expected_response if expected_response[:1].isspace() else f" {expected_response}"
    expected_tokens = adapter.tokenize(scoring_response, add_special_tokens=False)
    if not expected_tokens.input_ids:
        raise ValueError("expected response must produce at least one token")

    scores = _final_token_scores(result.logits)
    expected_token_id = int(expected_tokens.input_ids[0])
    predicted_token_id = max(range(len(scores)), key=lambda index: scores[index])
    predicted_token_text = adapter.decode_token_ids([predicted_token_id])[0]
    expected_token_text = adapter.decode_token_ids([expected_token_id])[0]
    rank = _rank_index(scores, expected_token_id)
    ranked_ids = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[: max(1, top_k)]

    return {
        "expected_first_token_id": expected_token_id,
        "expected_first_token_text": expected_token_text,
        "expected_response_token_count": len(expected_tokens.input_ids),
        "scoring_response": scoring_response,
        "predicted_first_token_id": predicted_token_id,
        "predicted_first_token_text": predicted_token_text,
        "expected_first_token_rank": rank,
        "expected_first_token_logprob": _softmax_logprob(scores, expected_token_id),
        "top1_match": predicted_token_id == expected_token_id,
        "topk_match": expected_token_id in ranked_ids,
        "scored_suffix": "first_token_only",
    }


def aggregate_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["group"])].append(observation)

    def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "count": 0,
                "top1_accuracy": 0.0,
                "topk_accuracy": 0.0,
                "mean_expected_rank": None,
                "mean_expected_logprob": None,
            }
        ranks = [int(item["metrics"]["expected_first_token_rank"]) for item in items]
        logprobs = [float(item["metrics"]["expected_first_token_logprob"]) for item in items]
        return {
            "count": len(items),
            "top1_accuracy": sum(1 for item in items if item["metrics"]["top1_match"]) / len(items),
            "topk_accuracy": sum(1 for item in items if item["metrics"]["topk_match"]) / len(items),
            "mean_expected_rank": sum(ranks) / len(ranks),
            "mean_expected_logprob": sum(logprobs) / len(logprobs),
        }

    by_group = {group_name: build_summary(items) for group_name, items in grouped.items()}
    by_group["overall"] = build_summary(observations)
    return {"case_groups": by_group}
