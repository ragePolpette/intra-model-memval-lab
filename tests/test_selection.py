from __future__ import annotations

from pathlib import Path

from intra_model_memval.ingestion import EpisodeIngestionService
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.selection import EpisodeSelectionPolicy, select_episodes


def test_episode_selection_filters_by_tags_and_caps(tmp_path: Path):
    store = ExperimentStore(db_path=tmp_path / "exp.db", artifact_dir=tmp_path / "artifacts")
    service = EpisodeIngestionService()

    for idx in range(8):
        store.save_episode(
            service.ingest_episode(
                {
                    "content_text": f"Episode {idx}",
                    "topic_tags": ["target"] if idx < 6 else ["other"],
                    "provenance": {"source_type": "dataset", "source_label": f"src-{idx % 2}"},
                    "session_id": f"sess-{idx % 2}",
                }
            )
        )

    result = select_episodes(
        store,
        EpisodeSelectionPolicy(
            include_topic_tags=["target"],
            sample_size=4,
            max_per_source_label=2,
            max_per_session=2,
            seed=7,
        ),
    )

    assert result.total_candidates == 6
    assert result.selected_count == 4
    source_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    for episode in result.episodes:
        source_label = str(episode.provenance.source_label)
        source_counts[source_label] = source_counts.get(source_label, 0) + 1
        session_counts[episode.session_id or ""] = session_counts.get(episode.session_id or "", 0) + 1
    assert max(source_counts.values()) <= 2
    assert max(session_counts.values()) <= 2
