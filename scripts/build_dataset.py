"""Dataset builder scaffold for intra-model-memval-lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build memval dataset (scaffold)")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--output-numeric", required=True, type=Path, help="Output numeric JSONL path")
    parser.add_argument("--output-text", required=True, type=Path, help="Output text shadow JSONL path")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    rows = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))

    numeric_rows = []
    text_rows = []
    for row in rows:
        if row.get("raw_numeric"):
            numeric_rows.append(row)
        text_rows.append(
            {
                "entry_id": row.get("entry_id"),
                "text_view": row.get("text_view", row.get("content", "")),
                "category": row.get("category"),
                "importance_score": row.get("importance_score"),
                "meta": {
                    "context_hash": row.get("context_hash"),
                    "writer_model": row.get("writer_model"),
                    "writer_agent_id": row.get("writer_agent_id"),
                },
            }
        )

    args.output_numeric.parent.mkdir(parents=True, exist_ok=True)
    with args.output_numeric.open("w", encoding="utf-8") as out_numeric:
        for row in numeric_rows:
            out_numeric.write(json.dumps(row, ensure_ascii=True))
            out_numeric.write("\n")

    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    with args.output_text.open("w", encoding="utf-8") as out_text:
        for row in text_rows:
            out_text.write(json.dumps(row, ensure_ascii=True))
            out_text.write("\n")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "numeric_rows": len(numeric_rows),
                "text_rows": len(text_rows),
                "output_numeric": str(args.output_numeric),
                "output_text": str(args.output_text),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
