"""SQLite-backed experiment store for episodes, traces, update candidates and evaluation runs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.models import (
    ArtifactRef,
    EpisodeRecord,
    EvaluationRun,
    EvaluationSpec,
    ExperimentRun,
    TraceArtifact,
    UpdateCandidate,
)
from ..utils.ids import utc_now_iso


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _sha256_hex(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


@dataclass
class StoredArtifact:
    artifact_id: str
    artifact_hash: str
    artifact_path: Path
    media_type: str | None
    size_bytes: int


class ExperimentStore:
    """Transactional storage for experiment entities and artifact files."""

    def __init__(self, db_path: Path, artifact_dir: Path):
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    run_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_hash TEXT NOT NULL UNIQUE,
                    artifact_path TEXT NOT NULL,
                    media_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    content_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    topic_tags_json TEXT NOT NULL,
                    trigger_tags_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    conversation_id TEXT,
                    session_id TEXT,
                    notes TEXT,
                    schema_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_specs (
                    evaluation_spec_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    target_facts_json TEXT NOT NULL,
                    related_facts_json TEXT NOT NULL,
                    unrelated_facts_json TEXT NOT NULL,
                    regression_rules_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trace_artifacts (
                    trace_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    trace_input_spec_json TEXT NOT NULL,
                    trace_type TEXT NOT NULL,
                    artifact_refs_json TEXT NOT NULL,
                    summary_metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reproducibility_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    notes TEXT,
                    schema_version TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS update_candidates (
                    update_candidate_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    trace_id TEXT,
                    run_id TEXT NOT NULL,
                    target_fact_spec_json TEXT NOT NULL,
                    candidate_localization_json TEXT NOT NULL,
                    update_budget_json TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evaluation_spec_id TEXT,
                    result_summary_json TEXT NOT NULL,
                    lineage_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                    FOREIGN KEY(trace_id) REFERENCES trace_artifacts(trace_id),
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id),
                    FOREIGN KEY(evaluation_spec_id) REFERENCES evaluation_specs(evaluation_spec_id)
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    evaluation_run_id TEXT PRIMARY KEY,
                    evaluation_spec_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    observations_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    FOREIGN KEY(evaluation_spec_id) REFERENCES evaluation_specs(evaluation_spec_id),
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_episode_context_hash ON episodes(context_hash);
                CREATE INDEX IF NOT EXISTS idx_episode_created ON episodes(created_at_utc);
                CREATE INDEX IF NOT EXISTS idx_trace_episode ON trace_artifacts(episode_id);
                CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_artifacts(run_id);
                CREATE INDEX IF NOT EXISTS idx_update_episode ON update_candidates(episode_id);
                CREATE INDEX IF NOT EXISTS idx_update_run ON update_candidates(run_id);
                CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_runs(run_id);
                """
            )

    def register_run(self, run: ExperimentRun) -> ExperimentRun:
        payload = ExperimentRun.model_validate(run)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_id, run_kind, status, title, manifest_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_kind = excluded.run_kind,
                    status = excluded.status,
                    title = excluded.title,
                    manifest_json = excluded.manifest_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    payload.run_id,
                    payload.run_kind.value,
                    payload.status.value,
                    payload.title,
                    _json_dumps(payload.manifest),
                    payload.created_at_utc,
                    payload.updated_at_utc,
                ),
            )
        return payload

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[ExperimentRun]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM experiment_runs
                ORDER BY created_at_utc DESC, run_id ASC
                LIMIT ? OFFSET ?
                """,
                (max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [
            ExperimentRun(
                run_id=row["run_id"],
                run_kind=row["run_kind"],
                status=row["status"],
                title=row["title"],
                manifest=json.loads(row["manifest_json"]),
                created_at_utc=row["created_at_utc"],
                updated_at_utc=row["updated_at_utc"],
            )
            for row in rows
        ]

    def _write_artifact_bytes(self, data: bytes, *, suffix: str = ".bin") -> StoredArtifact:
        artifact_hash = _sha256_hex(data)
        artifact_id = f"artifact-{artifact_hash[:12]}"
        target = self.artifact_dir / f"{artifact_hash}{suffix}"
        if not target.exists():
            fd, tmp_name = tempfile.mkstemp(prefix="artifact_", suffix=".tmp", dir=str(self.artifact_dir))
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                tmp_path.write_bytes(data)
                os.replace(tmp_path, target)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        return StoredArtifact(
            artifact_id=artifact_id,
            artifact_hash=artifact_hash,
            artifact_path=target,
            media_type=None,
            size_bytes=len(data),
        )

    def register_artifact_bytes(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        suffix: str = ".bin",
    ) -> ArtifactRef:
        stored = self._write_artifact_bytes(data, suffix=suffix)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_hash, artifact_path, media_type, size_bytes, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    artifact_hash = excluded.artifact_hash,
                    artifact_path = excluded.artifact_path,
                    media_type = excluded.media_type,
                    size_bytes = excluded.size_bytes
                """,
                (
                    stored.artifact_id,
                    stored.artifact_hash,
                    str(stored.artifact_path),
                    media_type,
                    stored.size_bytes,
                    utc_now_iso(),
                ),
            )
        return ArtifactRef(
            artifact_id=stored.artifact_id,
            role="registered",
            path=str(stored.artifact_path),
            media_type=media_type,
            size_bytes=stored.size_bytes,
            artifact_hash=stored.artifact_hash,
        )

    def register_artifact_file(
        self,
        source_path: Path,
        *,
        role: str = "registered",
        media_type: str | None = None,
    ) -> ArtifactRef:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"artifact source not found: {source}")
        ref = self.register_artifact_bytes(
            source.read_bytes(),
            media_type=media_type,
            suffix=source.suffix or ".bin",
        )
        return ref.model_copy(update={"role": role})

    def save_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        payload = EpisodeRecord.model_validate(episode)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, content_text, metadata_json, topic_tags_json, trigger_tags_json,
                    provenance_json, observed_at_utc, created_at_utc, context_hash,
                    conversation_id, session_id, notes, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    content_text = excluded.content_text,
                    metadata_json = excluded.metadata_json,
                    topic_tags_json = excluded.topic_tags_json,
                    trigger_tags_json = excluded.trigger_tags_json,
                    provenance_json = excluded.provenance_json,
                    observed_at_utc = excluded.observed_at_utc,
                    created_at_utc = excluded.created_at_utc,
                    context_hash = excluded.context_hash,
                    conversation_id = excluded.conversation_id,
                    session_id = excluded.session_id,
                    notes = excluded.notes,
                    schema_version = excluded.schema_version
                """,
                (
                    payload.episode_id,
                    payload.content_text,
                    _json_dumps(payload.metadata),
                    _json_dumps(payload.topic_tags),
                    _json_dumps(payload.trigger_tags),
                    _json_dumps(payload.provenance.model_dump(mode="json")),
                    payload.observed_at_utc,
                    payload.created_at_utc,
                    payload.context_hash,
                    payload.conversation_id,
                    payload.session_id,
                    payload.notes,
                    payload.schema_version,
                ),
            )
        return payload

    def load_episode(self, episode_id: str) -> EpisodeRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        return None if row is None else self._episode_from_row(row)

    def list_episodes(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        topic_tag: str | None = None,
        source_type: str | None = None,
    ) -> list[EpisodeRecord]:
        sql = ["SELECT * FROM episodes WHERE 1=1"]
        params: list[Any] = []
        if topic_tag:
            sql.append("AND topic_tags_json LIKE ?")
            params.append(f'%"{str(topic_tag).strip().lower()}"%')
        if source_type:
            sql.append("AND provenance_json LIKE ?")
            params.append(f'%"source_type": "{str(source_type).strip().lower()}"%')
        sql.append("ORDER BY created_at_utc DESC, episode_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._episode_from_row(row) for row in rows]

    def search_episodes(self, query: str, *, limit: int = 50, offset: int = 0) -> list[EpisodeRecord]:
        pattern = f"%{query}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM episodes
                WHERE content_text LIKE ? OR notes LIKE ? OR metadata_json LIKE ? OR topic_tags_json LIKE ?
                ORDER BY created_at_utc DESC, episode_id ASC
                LIMIT ? OFFSET ?
                """,
                (pattern, pattern, pattern, pattern, max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._episode_from_row(row) for row in rows]

    def count_episodes(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM episodes").fetchone()
        return int(row["c"])

    @staticmethod
    def _episode_from_row(row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=row["episode_id"],
            content_text=row["content_text"],
            metadata=json.loads(row["metadata_json"]),
            topic_tags=json.loads(row["topic_tags_json"]),
            trigger_tags=json.loads(row["trigger_tags_json"]),
            provenance=json.loads(row["provenance_json"]),
            observed_at_utc=row["observed_at_utc"],
            created_at_utc=row["created_at_utc"],
            context_hash=row["context_hash"],
            conversation_id=row["conversation_id"],
            session_id=row["session_id"],
            notes=row["notes"],
            schema_version=row["schema_version"],
        )

    def save_trace_artifact(self, trace: TraceArtifact) -> TraceArtifact:
        payload = TraceArtifact.model_validate(trace)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trace_artifacts (
                    trace_id, episode_id, run_id, adapter_id, trace_input_spec_json, trace_type,
                    artifact_refs_json, summary_metrics_json, status, reproducibility_json,
                    created_at_utc, updated_at_utc, notes, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    run_id = excluded.run_id,
                    adapter_id = excluded.adapter_id,
                    trace_input_spec_json = excluded.trace_input_spec_json,
                    trace_type = excluded.trace_type,
                    artifact_refs_json = excluded.artifact_refs_json,
                    summary_metrics_json = excluded.summary_metrics_json,
                    status = excluded.status,
                    reproducibility_json = excluded.reproducibility_json,
                    updated_at_utc = excluded.updated_at_utc,
                    notes = excluded.notes,
                    schema_version = excluded.schema_version
                """,
                (
                    payload.trace_id,
                    payload.episode_id,
                    payload.run_id,
                    payload.adapter_id,
                    _json_dumps(payload.trace_input_spec),
                    payload.trace_type,
                    _json_dumps([item.model_dump(mode="json") for item in payload.artifact_refs]),
                    _json_dumps(payload.summary_metrics),
                    payload.status.value,
                    _json_dumps(payload.reproducibility_metadata),
                    payload.created_at_utc,
                    payload.updated_at_utc,
                    payload.notes,
                    payload.schema_version,
                ),
            )
        return payload

    def list_trace_artifacts(
        self,
        *,
        episode_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TraceArtifact]:
        sql = ["SELECT * FROM trace_artifacts WHERE 1=1"]
        params: list[Any] = []
        if episode_id:
            sql.append("AND episode_id = ?")
            params.append(episode_id)
        sql.append("ORDER BY created_at_utc DESC, trace_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._trace_from_row(row) for row in rows]

    def load_trace_artifact(self, trace_id: str) -> TraceArtifact | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM trace_artifacts WHERE trace_id = ?", (trace_id,)).fetchone()
        return None if row is None else self._trace_from_row(row)

    @staticmethod
    def _trace_from_row(row: sqlite3.Row) -> TraceArtifact:
        return TraceArtifact(
            trace_id=row["trace_id"],
            episode_id=row["episode_id"],
            run_id=row["run_id"],
            adapter_id=row["adapter_id"],
            trace_input_spec=json.loads(row["trace_input_spec_json"]),
            trace_type=row["trace_type"],
            artifact_refs=json.loads(row["artifact_refs_json"]),
            summary_metrics=json.loads(row["summary_metrics_json"]),
            status=row["status"],
            reproducibility_metadata=json.loads(row["reproducibility_json"]),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            notes=row["notes"],
            schema_version=row["schema_version"],
        )

    def save_update_candidate(self, candidate: UpdateCandidate) -> UpdateCandidate:
        payload = UpdateCandidate.model_validate(candidate)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO update_candidates (
                    update_candidate_id, episode_id, trace_id, run_id, target_fact_spec_json,
                    candidate_localization_json, update_budget_json, hypothesis, status,
                    evaluation_spec_id, result_summary_json, lineage_json, created_at_utc,
                    updated_at_utc, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(update_candidate_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    trace_id = excluded.trace_id,
                    run_id = excluded.run_id,
                    target_fact_spec_json = excluded.target_fact_spec_json,
                    candidate_localization_json = excluded.candidate_localization_json,
                    update_budget_json = excluded.update_budget_json,
                    hypothesis = excluded.hypothesis,
                    status = excluded.status,
                    evaluation_spec_id = excluded.evaluation_spec_id,
                    result_summary_json = excluded.result_summary_json,
                    lineage_json = excluded.lineage_json,
                    updated_at_utc = excluded.updated_at_utc,
                    schema_version = excluded.schema_version
                """,
                (
                    payload.update_candidate_id,
                    payload.episode_id,
                    payload.trace_id,
                    payload.run_id,
                    _json_dumps(payload.target_fact_spec),
                    _json_dumps(payload.candidate_localization),
                    _json_dumps(payload.update_budget),
                    payload.hypothesis,
                    payload.status.value,
                    payload.evaluation_spec_id,
                    _json_dumps(payload.result_summary),
                    _json_dumps(payload.lineage),
                    payload.created_at_utc,
                    payload.updated_at_utc,
                    payload.schema_version,
                ),
            )
        return payload

    def list_update_candidates(
        self,
        *,
        episode_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[UpdateCandidate]:
        sql = ["SELECT * FROM update_candidates WHERE 1=1"]
        params: list[Any] = []
        if episode_id:
            sql.append("AND episode_id = ?")
            params.append(episode_id)
        sql.append("ORDER BY created_at_utc DESC, update_candidate_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._update_from_row(row) for row in rows]

    def load_update_candidate(self, update_candidate_id: str) -> UpdateCandidate | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM update_candidates WHERE update_candidate_id = ?",
                (update_candidate_id,),
            ).fetchone()
        return None if row is None else self._update_from_row(row)

    @staticmethod
    def _update_from_row(row: sqlite3.Row) -> UpdateCandidate:
        return UpdateCandidate(
            update_candidate_id=row["update_candidate_id"],
            episode_id=row["episode_id"],
            trace_id=row["trace_id"],
            run_id=row["run_id"],
            target_fact_spec=json.loads(row["target_fact_spec_json"]),
            candidate_localization=json.loads(row["candidate_localization_json"]),
            update_budget=json.loads(row["update_budget_json"]),
            hypothesis=row["hypothesis"],
            status=row["status"],
            evaluation_spec_id=row["evaluation_spec_id"],
            result_summary=json.loads(row["result_summary_json"]),
            lineage=json.loads(row["lineage_json"]),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            schema_version=row["schema_version"],
        )

    def save_evaluation_spec(self, spec: EvaluationSpec) -> EvaluationSpec:
        payload = EvaluationSpec.model_validate(spec)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_specs (
                    evaluation_spec_id, name, description, target_facts_json, related_facts_json,
                    unrelated_facts_json, regression_rules_json, metadata_json,
                    created_at_utc, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_spec_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    target_facts_json = excluded.target_facts_json,
                    related_facts_json = excluded.related_facts_json,
                    unrelated_facts_json = excluded.unrelated_facts_json,
                    regression_rules_json = excluded.regression_rules_json,
                    metadata_json = excluded.metadata_json,
                    schema_version = excluded.schema_version
                """,
                (
                    payload.evaluation_spec_id,
                    payload.name,
                    payload.description,
                    _json_dumps([item.model_dump(mode="json") for item in payload.target_facts]),
                    _json_dumps([item.model_dump(mode="json") for item in payload.related_facts]),
                    _json_dumps([item.model_dump(mode="json") for item in payload.unrelated_facts]),
                    _json_dumps(payload.regression_rules),
                    _json_dumps(payload.metadata),
                    payload.created_at_utc,
                    payload.schema_version,
                ),
            )
        return payload

    def list_evaluation_specs(self, *, limit: int = 50, offset: int = 0) -> list[EvaluationSpec]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluation_specs
                ORDER BY created_at_utc DESC, evaluation_spec_id ASC
                LIMIT ? OFFSET ?
                """,
                (max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._spec_from_row(row) for row in rows]

    def load_evaluation_spec(self, evaluation_spec_id: str) -> EvaluationSpec | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_specs WHERE evaluation_spec_id = ?",
                (evaluation_spec_id,),
            ).fetchone()
        return None if row is None else self._spec_from_row(row)

    @staticmethod
    def _spec_from_row(row: sqlite3.Row) -> EvaluationSpec:
        return EvaluationSpec(
            evaluation_spec_id=row["evaluation_spec_id"],
            name=row["name"],
            description=row["description"],
            target_facts=json.loads(row["target_facts_json"]),
            related_facts=json.loads(row["related_facts_json"]),
            unrelated_facts=json.loads(row["unrelated_facts_json"]),
            regression_rules=json.loads(row["regression_rules_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at_utc=row["created_at_utc"],
            schema_version=row["schema_version"],
        )

    def save_evaluation_run(self, evaluation_run: EvaluationRun) -> EvaluationRun:
        payload = EvaluationRun.model_validate(evaluation_run)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    evaluation_run_id, evaluation_spec_id, run_id, subject_type, subject_id,
                    status, metrics_json, observations_json, created_at_utc, updated_at_utc, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_run_id) DO UPDATE SET
                    evaluation_spec_id = excluded.evaluation_spec_id,
                    run_id = excluded.run_id,
                    subject_type = excluded.subject_type,
                    subject_id = excluded.subject_id,
                    status = excluded.status,
                    metrics_json = excluded.metrics_json,
                    observations_json = excluded.observations_json,
                    updated_at_utc = excluded.updated_at_utc,
                    schema_version = excluded.schema_version
                """,
                (
                    payload.evaluation_run_id,
                    payload.evaluation_spec_id,
                    payload.run_id,
                    payload.subject_type,
                    payload.subject_id,
                    payload.status.value,
                    _json_dumps(payload.metrics),
                    _json_dumps(payload.observations),
                    payload.created_at_utc,
                    payload.updated_at_utc,
                    payload.schema_version,
                ),
            )
        return payload

    def list_evaluation_runs(
        self,
        *,
        evaluation_spec_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        sql = ["SELECT * FROM evaluation_runs WHERE 1=1"]
        params: list[Any] = []
        if evaluation_spec_id:
            sql.append("AND evaluation_spec_id = ?")
            params.append(evaluation_spec_id)
        sql.append("ORDER BY created_at_utc DESC, evaluation_run_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._evaluation_run_from_row(row) for row in rows]

    def load_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE evaluation_run_id = ?",
                (evaluation_run_id,),
            ).fetchone()
        return None if row is None else self._evaluation_run_from_row(row)

    @staticmethod
    def _evaluation_run_from_row(row: sqlite3.Row) -> EvaluationRun:
        return EvaluationRun(
            evaluation_run_id=row["evaluation_run_id"],
            evaluation_spec_id=row["evaluation_spec_id"],
            run_id=row["run_id"],
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            status=row["status"],
            metrics=json.loads(row["metrics_json"]),
            observations=json.loads(row["observations_json"]),
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
            schema_version=row["schema_version"],
        )

    def export_jsonl(self, destination: Path, rows: list[dict[str, Any]]) -> None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_json_dumps(row))
                handle.write("\n")
