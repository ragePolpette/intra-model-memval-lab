"""Self-evaluation scoring policy for numeric-first memory persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .schemas import MemoryRecord


class SelfEvalValidationError(ValueError):
    """Raised when self-evaluation mandatory inputs are missing."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _norm_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _stable_fingerprint(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _normalized_retrieved_ids(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    values = [_norm_str(item) for item in raw_ids]
    values = [item for item in values if item]
    return sorted(set(values))


def _has_surprise_signal(importance: dict[str, Any]) -> bool:
    return any(
        importance.get(key) is not None
        for key in (
            "confidence",
            "predictive_confidence",
            "predictive_confidence_before",
            "proxy_disagreement",
            "disagreement_score",
            "self_rating",
            "surprise_self_rating",
        )
    )


def _has_inference_signal(importance: dict[str, Any]) -> bool:
    return any(
        importance.get(key) is not None
        for key in ("tool_steps", "correction_count", "inference_level", "inference_steps")
    )


def _context_hash(
    *,
    project_id: str,
    scope: str,
    writer_model: str,
    context_fingerprint: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    canonical_context = {
        "project_id": _norm_str(project_id) or "default",
        "scope": _norm_str(scope) or "shared",
        "conversation_id": _norm_str(context_fingerprint.get("conversation_id")),
        "task_id": _norm_str(context_fingerprint.get("task_id")),
        "retrieved_ids": _normalized_retrieved_ids(context_fingerprint.get("retrieved_ids")),
        "tool_trace_fingerprint": _stable_fingerprint(context_fingerprint.get("tool_trace_fingerprint")),
        "prompt_fingerprint": _stable_fingerprint(context_fingerprint.get("prompt_fingerprint")),
        "writer_model": _norm_str(writer_model) or "unknown-model",
    }
    serialized = json.dumps(canonical_context, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return digest, canonical_context


def _compute_surprise(importance: dict[str, Any]) -> tuple[float, str, str]:
    confidence = importance.get("confidence")
    if confidence is None:
        confidence = importance.get("predictive_confidence")
    if confidence is None:
        confidence = importance.get("predictive_confidence_before")
    if confidence is not None:
        return 1.0 - _clamp01(confidence, default=0.5), "confidence", "high"

    disagreement = importance.get("proxy_disagreement")
    if disagreement is None:
        disagreement = importance.get("disagreement_score")
    if disagreement is not None:
        return _clamp01(disagreement, default=0.5), "disagreement", "medium"

    self_rating = importance.get("self_rating")
    if self_rating is None:
        self_rating = importance.get("surprise_self_rating")
    if self_rating is not None:
        return _clamp01(self_rating, default=0.5), "self", "low"

    return 0.5, "self", "low"


def _compute_inference(importance: dict[str, Any]) -> tuple[float, int]:
    tool_steps = _clamp_int(importance.get("tool_steps"), minimum=0, maximum=10, default=0)
    correction_count = _clamp_int(importance.get("correction_count"), minimum=0, maximum=5, default=0)
    inference_level_raw = importance.get("inference_level")
    if inference_level_raw is None:
        inference_level_raw = importance.get("inference_steps")
    inference_level = _clamp_int(inference_level_raw, minimum=0, maximum=5, default=0)

    normalized = (tool_steps + correction_count + inference_level) / 20.0
    return _clamp01(normalized, default=0.0), inference_level


def _compute_novelty(metadata: dict[str, Any]) -> tuple[float, bool]:
    top_similarities = metadata.get("top_similarities")
    if not isinstance(top_similarities, list):
        return 1.0, False
    if not top_similarities:
        return 1.0, True
    max_similarity = max(_clamp01(value, default=0.0) for value in top_similarities)
    return _clamp01(1.0 - max_similarity, default=1.0), True


def enrich_self_evaluation(record: MemoryRecord, *, enforce: bool) -> MemoryRecord:
    metadata = dict(record.metadata)
    context_fingerprint = metadata.get("context_fingerprint")
    importance = metadata.get("importance")
    has_eval_inputs = (
        isinstance(context_fingerprint, dict)
        or isinstance(importance, dict)
        or isinstance(metadata.get("top_similarities"), list)
    )

    if not enforce and not has_eval_inputs:
        return record

    if not isinstance(context_fingerprint, dict):
        context_fingerprint = {}
    if not isinstance(importance, dict):
        importance = {}

    writer_model = _norm_str(record.writer_model) or "unknown-model"
    if enforce:
        missing: list[str] = []
        if not isinstance(metadata.get("context_fingerprint"), dict):
            missing.append("metadata.context_fingerprint")
        if not isinstance(metadata.get("importance"), dict):
            missing.append("metadata.importance")
        if writer_model == "unknown-model":
            missing.append("writer_model")
        if missing:
            raise SelfEvalValidationError(f"MISSING_REQUIRED_FIELDS: {', '.join(missing)}")
        if not _has_surprise_signal(importance):
            raise SelfEvalValidationError("MISSING_SURPRISE_SIGNAL")
        if not _has_inference_signal(importance):
            raise SelfEvalValidationError("MISSING_INFERENCE_SIGNAL")

    surprise_score, surprise_source, signal_quality = _compute_surprise(importance)
    novelty_score, novelty_computed = _compute_novelty(metadata)
    inference_score, inference_level = _compute_inference(importance)
    negative_impact = _clamp01(importance.get("negative_impact"), default=0.0)

    if surprise_source == "confidence":
        base = 0.45 * surprise_score + 0.35 * novelty_score + 0.20 * inference_score
    else:
        base = 0.20 * surprise_score + 0.45 * novelty_score + 0.35 * inference_score

    score_with_neg = base + 0.25 * negative_impact
    importance_score = int(max(0, min(100, round(score_with_neg * 100))))
    if importance_score >= 70:
        importance_class = "high"
    elif importance_score >= 40:
        importance_class = "medium"
    else:
        importance_class = "low"

    scope = _norm_str(metadata.get("scope")) or "shared"
    project_id = _norm_str(metadata.get("project_id")) or "default"
    context_hash, canonical_context = _context_hash(
        project_id=project_id,
        scope=scope,
        writer_model=writer_model,
        context_fingerprint=context_fingerprint,
    )
    event_ts_utc = _norm_str(metadata.get("event_ts_utc")) or _utc_now_iso()
    is_external = bool(importance.get("is_external", record.is_external))

    metadata.update(
        {
            "event_ts_utc": event_ts_utc,
            "context_hash": context_hash,
            "context_fingerprint": canonical_context,
            "writer_model": writer_model,
            "writer_agent_id": record.writer_agent_id,
            "scope": scope,
            "surprise_source": surprise_source,
            "signal_quality": signal_quality,
            "surprise_score": round(surprise_score, 6),
            "novelty_score": round(novelty_score, 6),
            "inference_score": round(inference_score, 6),
            "inference_level": inference_level,
            "negative_impact": round(negative_impact, 6),
            "importance_score": importance_score,
            "importance_class": importance_class,
            "is_external": is_external,
            "novelty_computed": novelty_computed,
            "self_eval_policy_mode": "experimental-v1",
        }
    )

    return record.model_copy(
        update={
            "importance_score": importance_score,
            "novelty_score": novelty_score,
            "is_external": is_external,
            "context_hash": context_hash,
            "metadata": metadata,
        }
    )

