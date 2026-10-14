# Historical documents

The documents in this directory are **superseded**. They record design intent, migration
decisions, or a status snapshot from an earlier point in the project's history. They are
kept for context - mainly to explain *why* the current architecture looks the way it does
- not as a description of what exists in the repository today.

For the current, verified state of the repository, use:

- [`README.md`](../../README.md) - what is implemented and how to run it
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) - current entity/module reference
- [`docs/CONSOLIDATION_DESIGN.md`](../CONSOLIDATION_DESIGN.md) - consolidation pipeline design and its implementation-status mapping

## Contents

- **`MIGRATION_NOTES.md`** - records the removal of an earlier "numeric memory" paradigm
  (`MemoryRecord`, `NumericPayload`, `MemoryPersistence`, `self_eval`) from the core package
  during an early rewrite. That paradigm no longer exists anywhere in the codebase; this note
  is kept only as a record of that decision.
