"""Contracts for the model-dependent consolidation steps. No implementation is faked here.

Lesson extraction, variation generation, verification and conflict judging are
LLM-backed (ideally different models for generator and verifier); LoRA training
targets Gemma 3 270M IT. This repo only defines the seams those adapters plug into.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..domain import ArtifactRef, ConsolidationCandidate, EpisodeRecord, Lesson, SyntheticExample


class LessonExtractor(Protocol):
    """Step 2: turn a harvested cluster of raw episodes into a Lesson record."""

    def extract(self, cluster_id: str, episodes: Sequence[EpisodeRecord]) -> Lesson: ...


class ConflictChecker(Protocol):
    """Step 3 (v2): compare a candidate lesson against the consolidated ones.

    Returns one of: "none", "merge", "supersede", "conflict".
    """

    def check(self, candidate: Lesson, consolidated: Sequence[Lesson]) -> str: ...


class VariationGenerator(Protocol):
    """Step 4a: generate reasoning trajectories that apply the lesson in varied contexts.

    Must include negative examples inside the lesson boundary and anchor at least
    a share of variations to paraphrased real source episodes.
    """

    def generate(
        self,
        lesson: Lesson,
        source_episodes: Sequence[EpisodeRecord],
        *,
        count: int,
    ) -> list[SyntheticExample]: ...


class VariationVerifier(Protocol):
    """Step 4b: score whether a trajectory applies the lesson or merely recites it (0-1)."""

    def score(self, lesson: Lesson, example: SyntheticExample) -> float: ...


class LoraTrainerAdapter(Protocol):
    """Step 6: train a LoRA adapter from a prepared candidate and replay buffer.

    Returns the artifact reference of the trained adapter weights.
    """

    def train(self, candidate: ConsolidationCandidate, buffer: Sequence[Any]) -> ArtifactRef: ...
