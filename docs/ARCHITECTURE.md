# Architecture

## 1. Scope

The repository is now an infrastructure base for auditable experiments on:

`episode -> trace artifact -> update candidate -> evaluation`

and for the consolidation cycle (see `CONSOLIDATION_DESIGN.md`):

`episodes -> lesson -> synthetic dataset -> consolidation candidate -> evaluation -> promote/rollback`

Not yet implemented:

- real activation tracing
- pathway/circuit detection
- localized weight updates
- LLM-backed lesson extraction / variation generation / verifier (only `Protocol` contracts)

Real LoRA training (`GemmaLoraTrainer`) and real eval execution
(`EvaluationExecutor` + `HFTextGenerator`) exist but live in the
optional `[train]` extra: the core stays installable without ML dependencies.
Eval scoring uses the per-case keyword rubric (`Rubric`: required term groups plus
forbidden terms) when a case defines one, and falls back to normalized containment
of the expected response otherwise.

## 2. Core entities

### EpisodeRecord

Represents an observed episode.
Used to describe the acquired fact and its context, not to encode learning.

Key fields:

- `episode_id`
- `content_text`
- `metadata`
- `topic_tags`
- `trigger_tags`
- `provenance`
- `observed_at_utc`
- `created_at_utc`
- `context_hash`
- `conversation_id`
- `session_id`
- `notes`

### TraceArtifact

Represents a tracing artifact or a registered placeholder.
The contract already exists to support future adapters, but the repo does not pretend to do real tracing.

Key fields:

- `trace_id`
- `episode_id`
- `run_id`
- `adapter_id`
- `trace_input_spec`
- `trace_type`
- `artifact_refs`
- `summary_metrics`
- `status`
- `reproducibility_metadata`

### UpdateCandidate

Represents a proposal for a local update.
It is not an executed update, does not modify weights, and does not declare success.

Key fields:

- `update_candidate_id`
- `episode_id`
- `trace_id`
- `run_id`
- `target_fact_spec`
- `candidate_localization`
- `update_budget`
- `hypothesis`
- `status`
- `evaluation_spec_id`
- `result_summary`
- `lineage`

### Lesson

Represents a lesson extracted from a cluster of episodes, a candidate for consolidation.
`boundary` is mandatory: a lesson without explicit boundaries produces synthetic
variations that overgeneralize.

Key fields:

- `lesson_id`
- `statement`
- `scope_tags`
- `boundary`
- `source_cluster_ids`
- `source_episode_ids`
- `status` (CANDIDATE / CONSOLIDATED / SUPERSEDED / CONFLICTED)
- `consolidation_run_id`

### SyntheticExample

A row of the verified synthetic dataset (JSONL registered as an artifact).

Key fields:

- `lesson_id`
- `kind` (positive / negative / anchor): `anchor` here means a variation built from a
  paraphrased real episode. It is not a replay anchor (the generic-corpus samples mixed in
  by `build_replay_buffer`, reported as `replay_anchor_*`).
- `input`
- `target`
- `verifier_score`
- `rubric` (optional): evaluation rubric used when the example lands in the held-out set

### ConsolidationCandidate

Represents a proposal for LoRA training with replay for a set of lessons.
It is not an executed training run: it records dataset, configurations and verdict.

Key fields:

- `consolidation_candidate_id`
- `lesson_ids`
- `dataset_ref`
- `replay_config`
- `train_config`
- `base_model_ref`
- `status` (PREPARED / TRAINED / PROMOTED / ROLLED_BACK)
- `evaluation_run_ids`
- `run_id`
- `result_summary`

### EvaluationSpec

Defines what should be evaluated.
Separates target, related and unrelated cases.

Key fields:

- `evaluation_spec_id`
- `name`
- `target_facts`
- `related_facts`
- `unrelated_facts`
- `regression_rules`
- `metadata`

### EvaluationRun

Represents a planned or registered evaluation run.
The repo today provides only the contract and a minimal planning harness.

Key fields:

- `evaluation_run_id`
- `evaluation_spec_id`
- `run_id`
- `subject_type`
- `subject_id`
- `status`
- `metrics`
- `observations`

### ExperimentRun

Basic run manifest for audit and reproducibility.
Used for trace, update candidate, evaluation and export.

## 3. Module layout

- `src/intra_model_memval/domain/`
  Data contracts.
- `src/intra_model_memval/persistence/`
  `ExperimentStore`, artifact registry, JSONL export.
- `src/intra_model_memval/ingestion/`
  `EpisodeIngestionService`, including `context_hash` normalization.
- `src/intra_model_memval/evaluation/`
  `EvaluationHarness` for placeholder but auditable runs;
  `EvaluationExecutor` for real runs against a `TextGenerator`.
- `src/intra_model_memval/consolidation/`
  Consolidation cycle: verifier gating, replay buffer, promote/rollback verdict,
  `ConsolidationPlanner` and `Protocol` contracts for the LLM/training-dependent steps.
- `src/intra_model_memval/adapters/`
  Future-facing contracts for model adapters.
- `src/intra_model_memval/selection.py`
  Optional curation/sampling utility. Not "memory".

## 4. Legacy removal

The core of the repository no longer contains:

- `MemoryRecord`
- `NumericPayload`
- `MemoryPersistence`
- `self_eval` as a compat layer
- internal bridges to the old numeric-memory paradigm

## 5. Storage model

`ExperimentStore` uses SQLite + an artifact directory.

Main tables:

- `episodes`
- `trace_artifacts`
- `update_candidates`
- `lessons`
- `consolidation_candidates`
- `evaluation_specs`
- `evaluation_runs`
- `experiment_runs`
- `artifacts`

Role:

- transactional persistence
- artifact file registry
- basic run manifests
- query and search for episodes
- initial audit trail

## 6. Design choices

- No heavy ML dependency at this stage.
- No fake behavior that simulates real tracing or update.
- Selection remains an optional utility.
- Scoring logic is no longer in the save path as if it were "memory".
- The old numeric-first model has been removed from the core, not just deprecated.

## 7. Future extension points

The planned but not yet implemented extensions are:

- model adapters that produce `TraceArtifact`
- pathway/localization analyzers that populate `candidate_localization`
- update planners/applicators that consume `UpdateCandidate`
- real evaluation executors that populate `EvaluationRun.metrics` and `observations`
- LLM-backed implementations of `LessonExtractor`, `VariationGenerator`,
  `VariationVerifier`, `ConflictChecker` (v2)
- real `LoraTrainerAdapter` on Gemma 3 270M IT that consumes `ConsolidationCandidate`
- SAE drift check (v2) registered as `TraceArtifact`
