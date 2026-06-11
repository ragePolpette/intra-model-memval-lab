# Architecture

## 1. Scope

Il repository e' ora una base di infrastruttura per esperimenti auditabili su:

`episode -> trace artifact -> update candidate -> evaluation`

e per il ciclo di consolidamento (vedi `CONSOLIDATION_DESIGN.md`):

`episodi -> lesson -> dataset sintetico -> consolidation candidate -> evaluation -> promote/rollback`

Non implementa ancora:

- activation tracing reale
- pathway/circuit detection
- localized weight updates
- serious model-based evaluation
- lesson extraction / variation generation / verifier LLM-backed (solo contratti `Protocol`)
- training LoRA reale (solo contratto adapter + manifest auditabili)

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

### Lesson

Rappresenta una lezione estratta da un cluster di episodi, candidata al consolidamento.
`boundary` e' obbligatorio: una lezione senza confini espliciti produce variazioni
sintetiche che overgeneralizzano.

Campi chiave:

- `lesson_id`
- `statement`
- `scope_tags`
- `boundary`
- `source_cluster_ids`
- `source_episode_ids`
- `status` (CANDIDATE / CONSOLIDATED / SUPERSEDED / CONFLICTED)
- `consolidation_run_id`

### SyntheticExample

Una riga del dataset sintetico verificato (JSONL registrato come artifact).

Campi chiave:

- `lesson_id`
- `kind` (positive / negative / anchor)
- `input`
- `target`
- `verifier_score`

### ConsolidationCandidate

Rappresenta una proposta di training LoRA con replay per un insieme di lezioni.
Non e' un training eseguito: registra dataset, configurazioni e verdetto.

Campi chiave:

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
- `src/intra_model_memval/consolidation/`
  Ciclo di consolidamento: gating verifier, replay buffer, verdetto promote/rollback,
  `ConsolidationPlanner` e contratti `Protocol` per i passi LLM/training-dependent.
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
- `lessons`
- `consolidation_candidates`
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
- implementazioni LLM-backed di `LessonExtractor`, `VariationGenerator`,
  `VariationVerifier`, `ConflictChecker` (v2)
- `LoraTrainerAdapter` reale su Gemma 3 270M IT che consuma `ConsolidationCandidate`
- SAE drift check (v2) registrato come `TraceArtifact`
