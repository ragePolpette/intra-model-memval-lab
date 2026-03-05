"""CLI scaffold for intra-model memory validation lab."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="intra-model-memval-lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser("describe-policy", help="Show default policy")
    describe.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "describe-policy":
        payload = {
            "modality": {
                "primary": "numeric",
                "require_numeric": True,
                "text_shadow_enabled": True,
            },
            "novelty_min": 0.2,
            "distribution": {"top": 0.60, "mid": 0.25, "low": 0.15},
            "external_min_ratio": 0.25,
            "hardening": {
                "enforce_provenance": True,
                "enforce_external_evidence": True,
                "self_eval_enforced": False,
                "max_per_writer": 0.20,
                "max_per_conversation": 0.05,
            },
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print("novelty>=0.2, buckets 60/25/15, external>=25%, provenance required")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
