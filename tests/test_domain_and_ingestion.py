from __future__ import annotations

import importlib

import pytest

import intra_model_memval
from intra_model_memval.ingestion import EpisodeIngestionService


def test_episode_ingestion_normalizes_tags_and_context_hash():
    service = EpisodeIngestionService()
    episode = service.ingest_episode(
        {
            "content_text": "The user prefers sparse technical reports.",
            "topic_tags": ["Reports", "reports", "Memory"],
            "trigger_tags": [" preference ", "Preference"],
            "provenance": {
                "source_type": "User",
                "source_label": "interview",
            },
            "conversation_id": "conv-1",
            "session_id": "sess-1",
        }
    )

    assert episode.episode_id.startswith("episode-")
    assert episode.topic_tags == ["memory", "reports"]
    assert episode.trigger_tags == ["preference"]
    assert len(episode.context_hash) == 16


def test_episode_requires_non_empty_content():
    service = EpisodeIngestionService()
    with pytest.raises(ValueError):
        service.ingest_episode({"content_text": "   "})


def test_package_surface_does_not_expose_removed_legacy_components():
    assert not hasattr(intra_model_memval, "MemoryRecord")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intra_model_memval.legacy")
