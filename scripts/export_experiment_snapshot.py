"""Export curated episodes from the experiment store to JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intra_model_memval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
