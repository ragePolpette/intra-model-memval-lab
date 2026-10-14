from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dev_extra_ships_ruff_and_ci_runs_it():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any(item.startswith("ruff") for item in extras["dev"])
    assert "ruff check" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_spike_dependencies_are_declared_and_notes_live_in_history_docs():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any(item.startswith("sae-lens") for item in extras["spike"])
    assert not any(item.startswith("sae-lens") for item in _pyproject()["project"]["dependencies"])

    spike = ROOT / "scripts" / "spikes" / "sae_drift_spike.py"
    docstring = ast.get_docstring(ast.parse(spike.read_text(encoding="utf-8")))
    assert "docs/history/SAE_DRIFT_SPIKE.md" in docstring
    assert "NOT added to pyproject.toml" not in docstring
    assert (ROOT / "docs" / "history" / "SAE_DRIFT_SPIKE.md").exists()
