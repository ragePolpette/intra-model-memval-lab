"""Dataset builder scaffold for intra-model-memval-lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intra_model_memval.schemas import MemoryRecord
from intra_model_memval.selection import SelectionPolicy, select_training_records
from intra_model_memval.storage import MemoryPersistence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build memval dataset from SQLite storage")
    parser.add_argument("--db-path", required=True, type=Path, help="SQLite DB path")
    parser.add_argument("--blob-dir", type=Path, help="Blob directory path (default: <db_dir>/blobs)")
    parser.add_argument("--output-numeric", required=True, type=Path, help="Output numeric JSONL path")
    parser.add_argument("--output-text", required=True, type=Path, help="Output text shadow JSONL path")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional target sample size")
    parser.add_argument("--novelty-min", type=float, default=0.2)
    parser.add_argument("--top-ratio", type=float, default=0.60)
    parser.add_argument("--mid-ratio", type=float, default=0.25)
    parser.add_argument("--low-ratio", type=float, default=0.15)
    parser.add_argument("--external-min-ratio", type=float, default=0.25)
    parser.add_argument("--max-per-writer", type=float, default=0.20)
    parser.add_argument("--max-per-conversation", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _numeric_row(record: MemoryRecord) -> dict:
    return {
        "entry_id": record.entry_id,
        "category": record.category.value,
        "importance_score": int(record.importance_score),
        "novelty_score": float(record.novelty_score),
        "is_external": bool(record.is_external),
        "context_hash": record.context_hash,
        "writer_model": record.writer_model,
        "writer_agent_id": record.writer_agent_id,
        "created_at_utc": record.created_at_utc,
        "raw_numeric": {
            "dtype": record.raw_numeric.dtype,
            "shape": list(record.raw_numeric.shape),
            "encoding": record.raw_numeric.encoding.value,
            "blob_path": record.raw_numeric.blob_path,
            "blob_hash": record.raw_numeric.blob_hash,
        },
        "meta": dict(record.metadata),
    }


def _text_row(record: MemoryRecord) -> dict:
    return {
        "entry_id": record.entry_id,
        "text_view": record.text_view or "",
        "category": record.category.value,
        "importance_score": int(record.importance_score),
        "novelty_score": float(record.novelty_score),
        "is_external": bool(record.is_external),
        "meta": {
            "context_hash": record.context_hash,
            "writer_model": record.writer_model,
            "writer_agent_id": record.writer_agent_id,
            "created_at_utc": record.created_at_utc,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"DB not found: {args.db_path}")

    blob_dir = args.blob_dir or (args.db_path.parent / "blobs")
    persistence = MemoryPersistence(db_path=args.db_path, blob_dir=blob_dir)
    policy = SelectionPolicy(
        novelty_min=float(args.novelty_min),
        top_ratio=float(args.top_ratio),
        mid_ratio=float(args.mid_ratio),
        low_ratio=float(args.low_ratio),
        external_min_ratio=float(args.external_min_ratio),
        max_per_writer=float(args.max_per_writer),
        max_per_conversation=float(args.max_per_conversation),
        sample_size=args.sample_size,
        seed=int(args.seed),
    )
    selected = select_training_records(persistence, policy)

    numeric_rows = [_numeric_row(record) for record in selected.records]
    text_rows = [_text_row(record) for record in selected.records]

    _write_jsonl(args.output_numeric, numeric_rows)
    _write_jsonl(args.output_text, text_rows)

    print(
        json.dumps(
            {
                "db_path": str(args.db_path),
                "input_rows": selected.stats.total_candidates,
                "after_novelty_filter": selected.stats.after_novelty_filter,
                "selected_rows": selected.stats.selected,
                "external_ratio": round(selected.stats.external_ratio, 6),
                "required_external": selected.stats.required_external,
                "external_target_met": selected.stats.external_target_met,
                "bucket_counts": {
                    "top": selected.stats.top_count,
                    "mid": selected.stats.mid_count,
                    "low": selected.stats.low_count,
                },
                "numeric_rows": len(numeric_rows),
                "text_rows": len(text_rows),
                "output_numeric": str(args.output_numeric),
                "output_text": str(args.output_text),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
