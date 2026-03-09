"""Optional episode curation utilities. This is sampling infrastructure, not memory logic."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .domain import EpisodeRecord
from .persistence import ExperimentStore


@dataclass
class EpisodeSelectionPolicy:
    include_topic_tags: list[str] | None = None
    exclude_topic_tags: list[str] | None = None
    source_types: list[str] | None = None
    sample_size: int | None = None
    max_per_source_label: int | None = None
    max_per_session: int | None = None
    seed: int = 42


@dataclass
class EpisodeSelectionResult:
    episodes: list[EpisodeRecord]
    total_candidates: int
    selected_count: int


def _matches(policy: EpisodeSelectionPolicy, episode: EpisodeRecord) -> bool:
    include_tags = {item.strip().lower() for item in policy.include_topic_tags or [] if item.strip()}
    exclude_tags = {item.strip().lower() for item in policy.exclude_topic_tags or [] if item.strip()}
    source_types = {item.strip().lower() for item in policy.source_types or [] if item.strip()}

    topic_tags = set(episode.topic_tags)
    if include_tags and not include_tags.intersection(topic_tags):
        return False
    if exclude_tags and exclude_tags.intersection(topic_tags):
        return False
    if source_types and episode.provenance.source_type not in source_types:
        return False
    return True


def select_episodes(
    store: ExperimentStore,
    policy: EpisodeSelectionPolicy,
    *,
    query: str | None = None,
) -> EpisodeSelectionResult:
    base = store.search_episodes(query, limit=5000, offset=0) if query else store.list_episodes(limit=5000, offset=0)
    filtered = [episode for episode in base if _matches(policy, episode)]
    rng = random.Random(policy.seed)
    ordered = list(filtered)
    rng.shuffle(ordered)

    selected: list[EpisodeRecord] = []
    per_source: dict[str, int] = {}
    per_session: dict[str, int] = {}
    target = policy.sample_size or len(ordered)

    for episode in ordered:
        if len(selected) >= target:
            break
        source_label = str(episode.provenance.source_label or "unknown")
        session_id = str(episode.session_id or "")
        if policy.max_per_source_label is not None and per_source.get(source_label, 0) >= policy.max_per_source_label:
            continue
        if session_id and policy.max_per_session is not None and per_session.get(session_id, 0) >= policy.max_per_session:
            continue
        selected.append(episode)
        per_source[source_label] = per_source.get(source_label, 0) + 1
        if session_id:
            per_session[session_id] = per_session.get(session_id, 0) + 1

    selected.sort(key=lambda item: (item.created_at_utc, item.episode_id))
    return EpisodeSelectionResult(
        episodes=selected,
        total_candidates=len(filtered),
        selected_count=len(selected),
    )
