# Architecture

## 1. Scope

Il repository e' ora una base di infrastruttura per esperimenti auditabili su:

`episode -> trace artifact -> update candidate -> evaluation`

Non implementa ancora:

- activation tracing reale
- pathway/circuit detection
- localized weight updates
- serious model-based evaluation

## 2. Core entities

### EpisodeRecord

Rappresenta un episodio osservato.
Serve a descrivere il fatto acquisito e il suo contesto, non a codificare apprendimento.

Campi chiave:

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

Rappresenta un artefatto di tracing o un placeholder registrato.
Il contratto esiste gia' per supportare futuri adapter, ma il repo non finge tracing reale.

Campi chiave:

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

Rappresenta una proposta di update locale.
Non e' un update eseguito, non modifica pesi e non dichiara successo.

Campi chiave:

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

### EvaluationSpec

Definisce cosa dovrebbe essere valutato.
Separa i casi target, related e unrelated.

Campi chiave:

- `evaluation_spec_id`
- `name`
- `target_facts`
- `related_facts`
- `unrelated_facts`
- `regression_rules`
- `metadata`

### EvaluationRun

Rappresenta un run di valutazione pianificato o registrato.
Il repo oggi fornisce solo il contratto e un harness minimale di pianificazione.

Campi chiave:

- `evaluation_run_id`
- `evaluation_spec_id`
- `run_id`
- `subject_type`
- `subject_id`
- `status`
- `metrics`
- `observations`

### ExperimentRun

Manifest basilare di run per audit e riproducibilita'.
Viene usato per trace, update candidate, evaluation ed export.

## 3. Module layout

- `src/intra_model_memval/domain/`
  Contratti dati.
- `src/intra_model_memval/persistence/`
  `ExperimentStore`, artifact registry, export JSONL.
- `src/intra_model_memval/ingestion/`
  `EpisodeIngestionService`, inclusa la normalizzazione del `context_hash`.
- `src/intra_model_memval/evaluation/`
  `EvaluationHarness` per run placeholder ma auditabili.
- `src/intra_model_memval/adapters/`
  Contratti future-facing per model adapters.
- `src/intra_model_memval/selection.py`
  Utility opzionale di curation/sampling. Non e' “memoria”.

## 4. Legacy removal

Il core del repository non contiene piu':

- `MemoryRecord`
- `NumericPayload`
- `MemoryPersistence`
- `self_eval` come compat layer
- bridge interni verso il vecchio paradigma numeric-memory

## 5. Storage model

`ExperimentStore` usa SQLite + directory artefatti.

Tabelle principali:

- `episodes`
- `trace_artifacts`
- `update_candidates`
- `evaluation_specs`
- `evaluation_runs`
- `experiment_runs`
- `artifacts`

Ruolo:

- persistenza transazionale
- registry di file artefatto
- run manifest basilari
- query e search per episodi
- audit trail iniziale

## 6. Design choices

- Nessuna dipendenza ML pesante in questa fase.
- Nessun comportamento fake che simuli tracing o update reale.
- La selection resta utility opzionale.
- La scoring logic non e' piu' nel save path come se fosse “memoria”.
- Il vecchio modello numeric-first e' stato rimosso dal core, non solo declassato.

## 7. Future extension points

Le estensioni previste, ma non ancora implementate, sono:

- model adapters che producono `TraceArtifact`
- pathway/localization analyzers che popolano `candidate_localization`
- update planners/applicators che consumano `UpdateCandidate`
- evaluation executors reali che popolano `EvaluationRun.metrics` e `observations`
