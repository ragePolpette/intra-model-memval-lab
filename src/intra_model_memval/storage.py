"""Core dual persistence service for numeric-first memory records."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .schemas import MemoryRecord, NumericEncoding
from .self_eval import enrich_self_evaluation


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


@dataclass
class BlobMaterialization:
    blob_hash: str
    blob_path: Path
    size_bytes: int
    created_new: bool


@dataclass
class PreparedRecord:
    record: MemoryRecord
    blob: BlobMaterialization


class MemoryPersistence:
    """Atomic dual persistence: blob files + SQLite index."""

    def __init__(self, db_path: Path, blob_dir: Path, *, self_eval_enforced: bool = False):
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir)
        self.self_eval_enforced = bool(self_eval_enforced)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS blobs (
                    blob_hash TEXT PRIMARY KEY,
                    blob_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS records (
                    entry_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    modality_primary TEXT NOT NULL,
                    importance_score INTEGER NOT NULL,
                    novelty_score REAL NOT NULL,
                    is_external INTEGER NOT NULL,
                    provenance_level TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    writer_model TEXT NOT NULL,
                    writer_agent_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    text_view TEXT,
                    metadata_json TEXT NOT NULL,
                    numeric_dtype TEXT NOT NULL,
                    numeric_shape_json TEXT NOT NULL,
                    numeric_encoding TEXT NOT NULL,
                    blob_hash TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    saved_at_utc TEXT NOT NULL,
                    FOREIGN KEY(blob_hash) REFERENCES blobs(blob_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_records_context_hash ON records(context_hash);
                CREATE INDEX IF NOT EXISTS idx_records_importance ON records(importance_score);
                CREATE INDEX IF NOT EXISTS idx_records_category ON records(category);
                CREATE INDEX IF NOT EXISTS idx_records_external ON records(is_external);
                CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at_utc);
                """
            )

    def _materialize_blob(self, record: MemoryRecord) -> BlobMaterialization:
        payload = record.raw_numeric
        data = self._extract_numeric_bytes(payload.encoding, payload.payload_b64, payload.blob_path)
        blob_hash = _sha256_hex(data)
        target_path = self.blob_dir / f"{blob_hash}.bin"
        existed = target_path.exists()

        if not existed:
            fd, tmp_name = tempfile.mkstemp(prefix="blob_", suffix=".tmp", dir=str(self.blob_dir))
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                tmp_path.write_bytes(data)
                os.replace(tmp_path, target_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        return BlobMaterialization(
            blob_hash=blob_hash,
            blob_path=target_path,
            size_bytes=len(data),
            created_new=not existed,
        )

    @staticmethod
    def _extract_numeric_bytes(
        encoding: NumericEncoding,
        payload_b64: str | None,
        blob_path: str | None,
    ) -> bytes:
        if encoding == NumericEncoding.BASE64:
            if not payload_b64:
                raise ValueError("BASE64 encoding requires payload_b64")
            return base64.b64decode(payload_b64.encode("ascii"), validate=True)

        if encoding in {NumericEncoding.NPZ_REF, NumericEncoding.ARROW_REF}:
            if not blob_path:
                raise ValueError(f"{encoding.value} encoding requires blob_path")
            source = Path(blob_path)
            if not source.exists():
                raise FileNotFoundError(f"Numeric blob source not found: {source}")
            return source.read_bytes()

        raise ValueError(f"Unsupported numeric encoding: {encoding}")

    def _upsert_blob(self, conn: sqlite3.Connection, blob: BlobMaterialization) -> None:
        conn.execute(
            """
            INSERT INTO blobs (blob_hash, blob_path, size_bytes, created_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(blob_hash) DO UPDATE SET
                blob_path = excluded.blob_path,
                size_bytes = excluded.size_bytes
            """,
            (
                blob.blob_hash,
                str(blob.blob_path),
                blob.size_bytes,
                utc_now_iso(),
            ),
        )

    def _upsert_record(
        self,
        conn: sqlite3.Connection,
        record: MemoryRecord,
        *,
        blob_hash: str,
        blob_path: Path,
    ) -> None:
        conn.execute(
            """
            INSERT INTO records (
                entry_id, category, modality_primary, importance_score, novelty_score, is_external,
                provenance_level, context_hash, writer_model, writer_agent_id, created_at_utc,
                schema_version, model_fingerprint, text_view, metadata_json, numeric_dtype,
                numeric_shape_json, numeric_encoding, blob_hash, blob_path, saved_at_utc
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(entry_id) DO UPDATE SET
                category = excluded.category,
                modality_primary = excluded.modality_primary,
                importance_score = excluded.importance_score,
                novelty_score = excluded.novelty_score,
                is_external = excluded.is_external,
                provenance_level = excluded.provenance_level,
                context_hash = excluded.context_hash,
                writer_model = excluded.writer_model,
                writer_agent_id = excluded.writer_agent_id,
                created_at_utc = excluded.created_at_utc,
                schema_version = excluded.schema_version,
                model_fingerprint = excluded.model_fingerprint,
                text_view = excluded.text_view,
                metadata_json = excluded.metadata_json,
                numeric_dtype = excluded.numeric_dtype,
                numeric_shape_json = excluded.numeric_shape_json,
                numeric_encoding = excluded.numeric_encoding,
                blob_hash = excluded.blob_hash,
                blob_path = excluded.blob_path,
                saved_at_utc = excluded.saved_at_utc
            """,
            (
                record.entry_id,
                record.category.value,
                record.modality_primary,
                record.importance_score,
                record.novelty_score,
                int(record.is_external),
                record.provenance_level,
                record.context_hash,
                record.writer_model,
                record.writer_agent_id,
                record.created_at_utc,
                record.schema_version,
                record.model_fingerprint,
                record.text_view,
                json.dumps(record.metadata, ensure_ascii=True, sort_keys=True),
                record.raw_numeric.dtype,
                json.dumps(record.raw_numeric.shape, ensure_ascii=True),
                record.raw_numeric.encoding.value,
                blob_hash,
                str(blob_path),
                utc_now_iso(),
            ),
        )

    def _prepare_record(self, payload: MemoryRecord | dict) -> PreparedRecord:
        record = MemoryRecord.model_validate(payload)
        record = enrich_self_evaluation(record, enforce=self.self_eval_enforced)
        blob = self._materialize_blob(record)
        return PreparedRecord(record=record, blob=blob)

    @staticmethod
    def _with_blob_metadata(record: MemoryRecord, blob: BlobMaterialization) -> MemoryRecord:
        updated_numeric = record.raw_numeric.model_copy(
            update={"blob_path": str(blob.blob_path), "blob_hash": blob.blob_hash}
        )
        return record.model_copy(update={"raw_numeric": updated_numeric})

    def save_memory_record(self, payload: MemoryRecord | dict) -> MemoryRecord:
        prepared = self._prepare_record(payload)
        record = prepared.record
        blob = prepared.blob

        try:
            with self._conn() as conn:
                self._upsert_blob(conn, blob)
                self._upsert_record(conn, record, blob_hash=blob.blob_hash, blob_path=blob.blob_path)
        except Exception:
            if blob.created_new and blob.blob_path.exists():
                blob.blob_path.unlink(missing_ok=True)
            raise

        return self._with_blob_metadata(record, blob)

    def save_many(self, payloads: Iterable[MemoryRecord | dict]) -> list[MemoryRecord]:
        prepared_items: list[PreparedRecord] = [self._prepare_record(item) for item in payloads]
        if not prepared_items:
            return []

        created_new_paths: dict[str, Path] = {}
        unique_blobs: dict[str, BlobMaterialization] = {}
        for prepared in prepared_items:
            blob = prepared.blob
            unique_blobs.setdefault(blob.blob_hash, blob)
            if blob.created_new:
                created_new_paths[blob.blob_hash] = blob.blob_path

        try:
            with self._conn() as conn:
                for blob in unique_blobs.values():
                    self._upsert_blob(conn, blob)
                for prepared in prepared_items:
                    self._upsert_record(
                        conn,
                        prepared.record,
                        blob_hash=prepared.blob.blob_hash,
                        blob_path=prepared.blob.blob_path,
                    )
        except Exception:
            for path in created_new_paths.values():
                if path.exists():
                    path.unlink(missing_ok=True)
            raise

        return [self._with_blob_metadata(item.record, item.blob) for item in prepared_items]

    def load_memory_record(self, entry_id: str) -> MemoryRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None

            return MemoryRecord(
                entry_id=row["entry_id"],
                category=row["category"],
                raw_numeric={
                    "dtype": row["numeric_dtype"],
                    "shape": json.loads(row["numeric_shape_json"]),
                    "encoding": row["numeric_encoding"],
                    "blob_path": row["blob_path"],
                    "blob_hash": row["blob_hash"],
                },
                text_view=row["text_view"],
                modality_primary=row["modality_primary"],
                importance_score=int(row["importance_score"]),
                novelty_score=float(row["novelty_score"]),
                is_external=bool(row["is_external"]),
                provenance_level=row["provenance_level"],
                context_hash=row["context_hash"],
                writer_model=row["writer_model"],
                writer_agent_id=row["writer_agent_id"],
                created_at_utc=row["created_at_utc"],
                metadata=json.loads(row["metadata_json"]),
                schema_version=row["schema_version"],
                model_fingerprint=row["model_fingerprint"],
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            entry_id=row["entry_id"],
            category=row["category"],
            raw_numeric={
                "dtype": row["numeric_dtype"],
                "shape": json.loads(row["numeric_shape_json"]),
                "encoding": row["numeric_encoding"],
                "blob_path": row["blob_path"],
                "blob_hash": row["blob_hash"],
            },
            text_view=row["text_view"],
            modality_primary=row["modality_primary"],
            importance_score=int(row["importance_score"]),
            novelty_score=float(row["novelty_score"]),
            is_external=bool(row["is_external"]),
            provenance_level=row["provenance_level"],
            context_hash=row["context_hash"],
            writer_model=row["writer_model"],
            writer_agent_id=row["writer_agent_id"],
            created_at_utc=row["created_at_utc"],
            metadata=json.loads(row["metadata_json"]),
            schema_version=row["schema_version"],
            model_fingerprint=row["model_fingerprint"],
        )

    def list_records(
        self,
        *,
        category: str | None = None,
        is_external: bool | None = None,
        context_hash: str | None = None,
        min_importance_score: int | None = None,
        max_importance_score: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        sql = ["SELECT * FROM records WHERE 1=1"]
        params: list[object] = []

        if category:
            sql.append("AND category = ?")
            params.append(category)
        if is_external is not None:
            sql.append("AND is_external = ?")
            params.append(int(is_external))
        if context_hash:
            sql.append("AND context_hash = ?")
            params.append(context_hash)
        if min_importance_score is not None:
            sql.append("AND importance_score >= ?")
            params.append(int(min_importance_score))
        if max_importance_score is not None:
            sql.append("AND importance_score <= ?")
            params.append(int(max_importance_score))

        sql.append("ORDER BY importance_score DESC, created_at_utc ASC, entry_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])

        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
            return [self._from_row(row) for row in rows]

    def search_records(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        is_external: bool | None = None,
        min_importance_score: int | None = None,
        max_importance_score: int | None = None,
    ) -> list[MemoryRecord]:
        sql = ["SELECT * FROM records WHERE (entry_id LIKE ? OR text_view LIKE ? OR metadata_json LIKE ?)"]
        pattern = f"%{query}%"
        params: list[object] = [pattern, pattern, pattern]

        if category:
            sql.append("AND category = ?")
            params.append(category)
        if is_external is not None:
            sql.append("AND is_external = ?")
            params.append(int(is_external))
        if min_importance_score is not None:
            sql.append("AND importance_score >= ?")
            params.append(int(min_importance_score))
        if max_importance_score is not None:
            sql.append("AND importance_score <= ?")
            params.append(int(max_importance_score))

        sql.append("ORDER BY importance_score DESC, created_at_utc ASC, entry_id ASC LIMIT ? OFFSET ?")
        params.extend([max(1, int(limit)), max(0, int(offset))])

        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
            return [self._from_row(row) for row in rows]

    def count_records(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM records").fetchone()
            return int(row["c"])

    def count_blobs(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM blobs").fetchone()
            return int(row["c"])
