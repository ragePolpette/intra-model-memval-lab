"""Dataset loaders for the GPT-2 small experiment vertical."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import EvaluationCase, EvaluationSpec
from ..utils.ids import new_id, utc_now_iso


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_gpt2_dataset_path() -> Path:
    return repository_root() / "experiments" / "gpt2_small" / "datasets" / "synthetic_capitals.json"


def default_gpt2_prompt_template_path() -> Path:
    return repository_root() / "experiments" / "gpt2_small" / "prompts" / "capital_prompt.txt"


@dataclass(slots=True)
class SyntheticDatasetCase:
    case_id: str
    country: str
    capital: str
    relation: str
    expected_response: str
    metadata: dict[str, Any]

    def to_evaluation_case(self, prompt_template: str) -> EvaluationCase:
        return EvaluationCase(
            case_id=self.case_id,
            prompt=prompt_template.format(country=self.country, capital=self.capital),
            expected_response=self.expected_response,
            metadata={
                "country": self.country,
                "capital": self.capital,
                "relation": self.relation,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class SyntheticDataset:
    dataset_id: str
    description: str
    prompt_template: str
    target_cases: list[SyntheticDatasetCase]
    related_cases: list[SyntheticDatasetCase]
    unrelated_cases: list[SyntheticDatasetCase]
    metadata: dict[str, Any]

    def to_evaluation_spec(
        self,
        *,
        evaluation_spec_id: str | None = None,
        name: str = "gpt2-small-capitals-baseline",
    ) -> EvaluationSpec:
        return EvaluationSpec(
            evaluation_spec_id=evaluation_spec_id or new_id("evaluation-spec"),
            name=name,
            description=self.description,
            target_facts=[item.to_evaluation_case(self.prompt_template) for item in self.target_cases],
            related_facts=[item.to_evaluation_case(self.prompt_template) for item in self.related_cases],
            unrelated_facts=[item.to_evaluation_case(self.prompt_template) for item in self.unrelated_cases],
            regression_rules=[
                "This phase is baseline-only. Compare target, related and unrelated behavior before any weight update."
            ],
            metadata={
                "dataset_id": self.dataset_id,
                "prompt_template": self.prompt_template,
                **self.metadata,
            },
            created_at_utc=utc_now_iso(),
        )


def _load_cases(rows: list[dict[str, Any]], relation: str) -> list[SyntheticDatasetCase]:
    cases: list[SyntheticDatasetCase] = []
    for row in rows:
        cases.append(
            SyntheticDatasetCase(
                case_id=str(row["case_id"]),
                country=str(row["country"]),
                capital=str(row["capital"]),
                relation=relation,
                expected_response=str(row.get("expected_response") or row["capital"]),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return cases


def load_synthetic_dataset(
    dataset_path: Path | None = None,
    *,
    prompt_template_path: Path | None = None,
) -> SyntheticDataset:
    actual_dataset_path = Path(dataset_path or default_gpt2_dataset_path())
    payload = json.loads(actual_dataset_path.read_text(encoding="utf-8"))
    prompt_template = Path(prompt_template_path or default_gpt2_prompt_template_path()).read_text(encoding="utf-8").strip()
    return SyntheticDataset(
        dataset_id=str(payload["dataset_id"]),
        description=str(payload["description"]),
        prompt_template=prompt_template,
        target_cases=_load_cases(list(payload.get("target", [])), "target"),
        related_cases=_load_cases(list(payload.get("related", [])), "related"),
        unrelated_cases=_load_cases(list(payload.get("unrelated", [])), "unrelated"),
        metadata=dict(payload.get("metadata") or {}),
    )
