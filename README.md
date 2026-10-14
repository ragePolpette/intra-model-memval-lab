# intra-model-memval-lab

[![CI](https://github.com/ragePolpette/intra-model-memval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ragePolpette/intra-model-memval-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](pyproject.toml)

Local, auditable infrastructure for running a full episode, lesson, LoRA consolidation, evaluation, promote/rollback cycle on a small local language model, plus a lighter-weight domain layer for tracking evaluation experiments in general.

## Overview

The repository contains a Pydantic domain model with a transactional SQLite store, a CLI for managing experiment records, and a consolidation pipeline that fine-tunes a LoRA adapter on Gemma 3 270M IT and decides whether to promote or roll back the result. Read "Features" and "Project status" before assuming any given capability is implemented.

## Features

Verified against the current source:

- **Domain + persistence layer**: `EpisodeRecord`, `TraceArtifact`, `UpdateCandidate`, `Lesson`, `ConsolidationCandidate`, `EvaluationSpec`, `EvaluationRun`, `ExperimentRun`, all Pydantic models backed by a transactional SQLite `ExperimentStore` with a content-addressed artifact registry (`src/intra_model_memval/domain/`, `persistence/`).
- **CLI** (`intra-model-exp`): create/list episodes, register trace-artifact placeholders, create/list update candidates, create/list evaluation specs and runs, export curated episode sets to JSONL. See [CLI commands](#cli-commands) for the full list.
- **Consolidation pipeline** (`src/intra_model_memval/consolidation/`): the "sleep cycle": verifier gating and composition checks on a synthetic dataset, a deterministic replay-buffer builder (new examples + anchors), a `ConsolidationPlanner` that records `PREPARED → TRAINED → PROMOTED/ROLLED_BACK` state transitions, and a promote/rollback gate (`decide_promotion`) driven by pre/post evaluation metrics.
- **Real LoRA training and evaluation**, gated behind the optional `[train]` extra so the core package stays dependency-light:
  - `GemmaLoraTrainer` fine-tunes a LoRA adapter on **Gemma 3 270M IT** via `transformers` + `peft`, on the replay buffer built by the pipeline above.
  - `HFTextGenerator` runs the base model, or the base model plus a trained adapter, for generation.
  - `EvaluationExecutor` runs an `EvaluationSpec` (target / related / unrelated cases) against a generator and scores each case with a normalized containment match. It is deliberately simple, meant to be replaced with a stronger scorer once the loop is validated end to end.
- **End-to-end script** (`scripts/run_consolidation_poc.py`): runs one full cycle: verifier gate → held-out split → replay buffer → pre-training eval on the base model → LoRA training → post-training eval on base+adapter → promote/rollback verdict. It writes the store, the trained adapter, and a `verdict.json` under a work directory. Toy fixtures for trying the loop live in `examples/consolidation_poc/`.
- **Episode selection utilities** (`selection.py`): optional sampling/curation over stored episodes (tag/source filters, per-source and per-session caps, seeded shuffling). This is dataset curation, not a memory mechanism.

## Quick start

Requires Python 3.11 or newer.

```bash
pip install -e .
intra-model-exp --help
```

Core install covers the domain, persistence, CLI and consolidation planning/gating logic, with no ML dependencies. For real LoRA training and generation (adds `torch`, `transformers`, `peft`, `accelerate`):

```bash
pip install -e ".[train]"
```

## Configuration

No environment variables are read. Configuration is passed through CLI flags (`intra-model-exp --help`, `python scripts/run_consolidation_poc.py --help`) or config objects in code (`ConsolidationConfig`, `GateThresholds`). `.env.example` is a placeholder.

## Usage

### CLI commands

`intra-model-exp --db-path <path> --artifact-dir <path> <command> ...` (or `python scripts/export_experiment_snapshot.py ...` from the repo root). Run `intra-model-exp --help` for full flag lists. Commands registered in `cli.py`:

```
save-episode              create-update-candidate       create-evaluation-run
list-episodes             list-update-candidates        list-evaluation-runs
register-trace            create-evaluation-spec        export-episodes
list-traces               list-evaluation-specs         list-runs
```

### Consolidation POC

Requires the `[train]` extra and a GPU or a patient CPU.

```bash
python scripts/run_consolidation_poc.py \
    --workdir runs/poc-001 \
    --lesson-json examples/consolidation_poc/lesson.json \
    --variations-jsonl examples/consolidation_poc/variations.jsonl \
    --anchors-jsonl examples/consolidation_poc/anchors.jsonl \
    --regression-jsonl examples/consolidation_poc/regression.jsonl
```

The fixtures in `examples/consolidation_poc/` are toy data for exercising the loop; a real cycle needs LLM-generated and verified lesson variations in their place. See `docs/CONSOLIDATION_DESIGN.md` for the pipeline design and its known simplifications.

## Consolidation cycle

The loop the consolidation pipeline implements, end to end:

`episode -> lesson -> verified synthetic dataset -> replay buffer + LoRA training -> pre/post evaluation -> promote/rollback`

1. **Episodes -> Lesson**: a cluster of `EpisodeRecord`s is turned into a `Lesson`. Extraction itself is an external, LLM-backed step (see "Project status").
2. **Lesson -> verified dataset**: candidate variations are gated by `apply_verifier_gate` and `check_dataset_composition` (`consolidation/dataset.py`), then registered as a JSONL artifact via `register_dataset`.
3. **Replay + LoRA**: `build_replay_buffer` (`consolidation/replay.py`) mixes new examples with anchors; `GemmaLoraTrainer` (`consolidation/training.py`, `[train]` extra) trains a LoRA adapter on Gemma 3 270M IT.
4. **Eval pre/post**: `EvaluationExecutor` (`evaluation/executor.py`) runs the same `EvaluationSpec` against the base model and against base+adapter.
5. **Promote/rollback**: `decide_promotion` (`consolidation/gating.py`) compares pre/post metrics against `GateThresholds`; `ConsolidationPlanner.apply_decision` (`consolidation/pipeline.py`) records the verdict and updates the candidate's and lessons' status.

`scripts/run_consolidation_poc.py` runs steps 2-5 end to end (step 1 arrives as an input file, see "Consolidation POC"). See `docs/CONSOLIDATION_DESIGN.md` for the full design contract and its per-step implementation-status mapping.

## Architecture

```text
src/intra_model_memval/
  domain/          # Pydantic contracts: episodes, traces, update candidates, lessons,
                    # consolidation candidates, evaluation specs/runs, run manifests
  persistence/      # ExperimentStore: SQLite + content-addressed artifact registry
  ingestion/        # Episode normalization (context hashing, provenance) before persistence
  evaluation/        # EvaluationHarness (placeholder run planning) and EvaluationExecutor
                     # (real scoring against a text generator)
  consolidation/    # The sleep-cycle pipeline: dataset gating, replay buffer, LoRA trainer,
                     # promote/rollback gate, ConsolidationPlanner, Protocol interfaces for
                     # the LLM-backed steps (lesson extraction, variation gen, verification)
  adapters/          # Placeholder contract for future model-adapter integrations
  selection.py       # Optional episode curation/sampling utilities
scripts/
  run_consolidation_poc.py        # End-to-end consolidation cycle runner
  export_experiment_snapshot.py   # Thin wrapper that exposes cli.py as a script
docs/
  ARCHITECTURE.md          # Entity-by-entity architecture reference
  CONSOLIDATION_DESIGN.md  # Design contract for the consolidation pipeline + implementation
                            # status mapping (contract step -> code -> status)
  history/                 # Superseded planning/migration notes, kept for context
examples/
  consolidation_poc/       # Toy fixtures for the consolidation POC script
tests/
```

For entity-level detail (fields, storage tables, module responsibilities) see `docs/ARCHITECTURE.md`. For the consolidation pipeline's design contract and its current implementation-status mapping, see `docs/CONSOLIDATION_DESIGN.md`.

## Project status

This is an active experimental lab, not a finished product. The following are not implemented:

- No activation tracing, pathway/circuit detection, or localized weight-update execution. `TraceArtifact` and `UpdateCandidate` are recorded contracts, not the result of real model introspection.
- No lesson extraction, variation generation, verification, or conflict-checking model. Those steps are defined as `Protocol` interfaces in `consolidation/interfaces.py` for an LLM-backed implementation to plug into; none ships in this repo.
- **No internal-feature / interpretability drift measurement.** The design anticipates a future SAE-based drift check (activation distance on fixed probe prompts, pre/post training) as part of the promote/rollback gate (`GateThresholds.drift_max` exists as a field for it), but it is not implemented or wired up. Promotion decisions today are based only on held-out learning rate and behavioral regression on containment-matched cases.
- No adapter stacking across consolidation cycles: the trainer produces one LoRA adapter per run.
- No completion-only loss masking in the POC trainer (it trains on the full chat-formatted sequence).


## Security

See [SECURITY.md](SECURITY.md) for scope, threat model and how to report vulnerabilities. Never commit secrets, databases or model artifacts.

## Development

```bash
pip install -e ".[dev]"
pytest
```

No linter is configured. See [CONTRIBUTING.md](CONTRIBUTING.md). `scripts/spikes/sae_drift_spike.py` is a throwaway experiment, not wired into the pipeline.

Built with AI-assisted workflows; architecture, tradeoffs, integration, review, and validation were directed by the author.

## License

MIT. See [LICENSE](LICENSE).
