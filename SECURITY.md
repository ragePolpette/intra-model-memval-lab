# Security Policy

## Scope

This repository is a local research lab: a SQLite-backed experiment store, a CLI (`intra-model-exp`), and a LoRA consolidation pipeline that can run a local Gemma 3 270M IT model. It is not a hosted service and has no network-facing component. The code reads no environment variables and ships no credentials.

## Threat model

The tool is intended to run on a trusted machine, on inputs the operator controls. Relevant considerations:

- **Local files**: the CLI and scripts read and write a SQLite database, an artifact directory, and a work directory (for example `runs/`) chosen by the operator. Paths are not sandboxed.
- **Input data**: episode text, lesson JSON and JSONL datasets are parsed as data. Treat files from untrusted sources with the same care as any other untrusted input.
- **Model downloads and training**: the optional `[train]` extra uses `torch`, `transformers` and `peft`, which may download model weights and execute third-party code paths. Their supply chain is outside this repository.
- **Secrets**: do not store credentials in episodes, metadata or artifacts. `.env` files are ignored by git; `.env.example` contains no real values.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

1. Use GitHub Security Advisories: open the repository's Security tab and choose "Report a vulnerability".
2. If that is unavailable, open an issue asking for a private contact channel, without including technical details.

Include the affected version or commit, a description of the problem, and steps to reproduce. This is a small project maintained on a best-effort basis; no response time is guaranteed.

## Not covered

- Vulnerabilities in dependencies (`pydantic`, `torch`, `transformers`, `peft`, `accelerate`); report those upstream.
- Behavior of models or adapters produced by the training pipeline, including output quality or safety.
- Issues that require an attacker who already has write access to the operator's machine, database or artifact directory.
- Experimental, throwaway code such as `scripts/spikes/`.
