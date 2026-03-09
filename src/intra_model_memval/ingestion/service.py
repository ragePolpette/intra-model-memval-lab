"""Services for acquiring and normalizing episode records."""

from __future__ import annotations

from typing import Any

from ..domain.models import EpisodeRecord, SourceProvenance
from ..utils.hashing import compute_episode_context_hash
from ..utils.ids import new_id, utc_now_iso


class EpisodeIngestionService:
    """Prepare and validate episodes before persistence."""

    def prepare_episode(self, payload: EpisodeRecord | dict[str, Any]) -> EpisodeRecord:
        if isinstance(payload, EpisodeRecord):
            return payload

        draft = dict(payload)
        provenance_model = SourceProvenance.model_validate(draft.get("provenance") or {})
        topic_tags = draft.get("topic_tags") or []
        trigger_tags = draft.get("trigger_tags") or []
        observed_at_utc = draft.get("observed_at_utc") or utc_now_iso()
        created_at_utc = draft.get("created_at_utc") or observed_at_utc
        episode_id = draft.get("episode_id") or new_id("episode")
        context_hash = draft.get("context_hash") or compute_episode_context_hash(
            content_text=str(draft.get("content_text") or ""),
            topic_tags=list(topic_tags),
            trigger_tags=list(trigger_tags),
            conversation_id=draft.get("conversation_id"),
            session_id=draft.get("session_id"),
            source_type=provenance_model.source_type,
            source_label=provenance_model.source_label,
        )

        return EpisodeRecord(
            episode_id=episode_id,
            content_text=draft.get("content_text") or "",
            metadata=draft.get("metadata") or {},
            topic_tags=topic_tags,
            trigger_tags=trigger_tags,
            provenance=provenance_model,
            observed_at_utc=observed_at_utc,
            created_at_utc=created_at_utc,
            context_hash=context_hash,
            conversation_id=draft.get("conversation_id"),
            session_id=draft.get("session_id"),
            notes=draft.get("notes"),
            schema_version=draft.get("schema_version") or "episode-v1",
        )

    def ingest_episode(self, payload: EpisodeRecord | dict[str, Any]) -> EpisodeRecord:
        return self.prepare_episode(payload)
