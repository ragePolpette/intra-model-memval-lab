from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import run


def test_cli_saves_lists_and_exports_episodes(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    db_path = tmp_path / "exp.db"
    artifact_dir = tmp_path / "artifacts"
    export_path = tmp_path / "episodes.jsonl"

    save_cmd = [
        sys.executable,
        str(root / "scripts" / "export_experiment_snapshot.py"),
        "--db-path",
        str(db_path),
        "--artifact-dir",
        str(artifact_dir),
        "save-episode",
        "--content-text",
        "Episode from CLI",
        "--topic-tag",
        "cli",
        "--source-type",
        "user",
    ]
    save_result = run(save_cmd, cwd=root, capture_output=True, text=True, check=True)
    saved = json.loads(save_result.stdout.strip())
    assert saved["content_text"] == "Episode from CLI"

    list_cmd = [
        sys.executable,
        str(root / "scripts" / "export_experiment_snapshot.py"),
        "--db-path",
        str(db_path),
        "--artifact-dir",
        str(artifact_dir),
        "list-episodes",
        "--query",
        "CLI",
    ]
    list_result = run(list_cmd, cwd=root, capture_output=True, text=True, check=True)
    listed = json.loads(list_result.stdout.strip())
    assert listed[0]["episode_id"] == saved["episode_id"]

    export_cmd = [
        sys.executable,
        str(root / "scripts" / "export_experiment_snapshot.py"),
        "--db-path",
        str(db_path),
        "--artifact-dir",
        str(artifact_dir),
        "export-episodes",
        "--output",
        str(export_path),
        "--include-topic-tag",
        "cli",
    ]
    export_result = run(export_cmd, cwd=root, capture_output=True, text=True, check=True)
    exported_run = json.loads(export_result.stdout.strip())
    assert exported_run["run_kind"] == "export"
    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
