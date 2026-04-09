# intra-model-memval-lab

> [!IMPORTANT]
> Work in progress.
> This repository is an active experimental lab for memory-evaluation infrastructure. It should be read as an evolving foundation, not as a finished runtime or research result.

Experimental infrastructure for auditable memory-evaluation workflows.

This repository explores a local, reproducible foundation for capturing episodes, registering trace artifacts, tracking localized update candidates, and running structured evaluations. It is not a live memory runtime and it is not a model training system. The current focus is infrastructure for experiments that can be inspected, replayed, and compared over time.

## What It Covers

- episode acquisition with contextual metadata
- trace artifact registration for future tracing workflows
- localized update candidate tracking
- evaluation spec and evaluation run management
- SQLite-backed persistence for experiments and manifests
- optional episode curation and selection utilities

## Current Status

This project is an active experimental lab, not a finished product.

What it is today:

- a coherent domain model for episode, trace, update, and evaluation workflows
- a minimal CLI for storing and inspecting experimental records
- a reproducible local persistence layer for small-scale experimentation

What it is not yet:

- a live intra-model memory system
- activation tracing over a real model
- pathway detection
- weight-update execution
- a trainer or fine-tuning pipeline
- a model-specific adapter for GPT-2 or other concrete runtimes

## Why This Exists

Most discussions around memory systems mix together storage, tracing, adaptation, and evaluation. This repository deliberately separates those concerns and starts from the infrastructure layer: contracts, persistence, and experiment bookkeeping.

The goal is to make future work on memory evaluation more inspectable and less hand-wavy. Before building a full runtime, it is useful to have a clean way to capture evidence, define candidate updates, and evaluate what changed.

## Architecture

Main areas:

```text
src/intra_model_memval/
  domain/        # core records and contracts
  persistence/   # transactional store and artifact registry
  ingestion/     # episode normalization and acquisition
  evaluation/    # evaluation spec/run primitives
  adapters/      # future model adapter extension point
  selection.py   # optional curation and sampling utilities
scripts/
  export_experiment_snapshot.py
docs/
  ARCHITECTURE.md
  MIGRATION_NOTES.md
  REALIGNMENT_REPORT.md
```

Key records:

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec`
- `EvaluationRun`
- `ExperimentStore`

## CLI

If installed as a package:

```bash
intra-model-exp --help
```

From the repository:

```bash
python scripts/export_experiment_snapshot.py --help
```

Main commands:

- `save-episode`
- `list-episodes`
- `register-trace`
- `list-traces`
- `create-update-candidate`
- `list-update-candidates`
- `create-evaluation-spec`
- `list-evaluation-specs`
- `create-evaluation-run`
- `list-evaluation-runs`
- `export-episodes`
- `list-runs`

## Documentation

Use these as the current source of truth:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/MIGRATION_NOTES.md`
- `REALIGNMENT_REPORT.md`

Historical planning notes remain in the repository for context, but they should not be treated as the current implementation contract.

## Development Process

Built with AI-assisted workflows, while architecture, tradeoffs, integration, review, and validation were directed by the author.
