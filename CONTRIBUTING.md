# Contributing

Thanks for your interest. This is an experimental lab; small, focused changes are easiest to review.

## Workflow

1. Fork the repository and create a branch from `main` (for example `fix/short-description`).
2. Make your change with tests where applicable.
3. Run the test suite (below) and make sure it passes.
4. Open a pull request against `main` using the pull request template. Describe what changed and why.

## Development setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

The `[train]` extra (`pip install -e ".[train]"`) is only needed for real LoRA training and generation. The test suite does not require it.

Lint with `ruff check .` (installed by the `[dev]` extra, configured in `pyproject.toml`, run in CI).

## Guidelines

- Keep the core package dependency-light; ML dependencies belong behind the `[train]` extra.
- Documentation is written in English, in a plain technical tone. State only what the code does.
- Keep README and `docs/` consistent with the source when behavior changes.

## Security rules

- Never commit secrets, credentials, `.env` files, databases (`*.db`, `*.sqlite*`), model checkpoints or adapters. These are covered by `.gitignore`; do not work around it.
- If code starts reading environment variables, document them in `.env.example` with placeholder values only.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
