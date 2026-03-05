from __future__ import annotations

from intra_model_memval.cli import build_parser


def test_cli_policy_command_parses():
    parser = build_parser()
    args = parser.parse_args(["describe-policy"])
    assert args.command == "describe-policy"
