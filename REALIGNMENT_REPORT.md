# 1. Summary of changes

Il repository e' stato riallineato da pipeline centrata su `numeric memory record`
a infrastruttura sperimentale per:

- acquisizione episodi
- registrazione trace artifact
- tracking di update candidate
- definizione di evaluation spec/run
- run manifest e artifact audit

Sono stati introdotti nuovi contratti di dominio, un nuovo store transazionale,
una CLI reale e una compatibilita' deprecata per il vecchio schema.

# 2. New architecture

Nuovi moduli principali:

- `src/intra_model_memval/domain/`
- `src/intra_model_memval/persistence/`
- `src/intra_model_memval/ingestion/`
- `src/intra_model_memval/evaluation/`
- `src/intra_model_memval/adapters/`

Nuove entita':

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec`
- `EvaluationRun`
- `ExperimentRun`

Persistenza:

- `ExperimentStore` su SQLite
- registry artefatti file-based
- manifest basilari di run

Entry point:

- nuova CLI con comandi per episode/trace/update/evaluation/export
- script `scripts/export_experiment_snapshot.py`

# 3. Deprecated concepts removed or downgraded

Sono stati rimossi o declassati:

- `MemoryRecord` come centro del dominio
- `raw_numeric` come fonte primaria generale
- `MemoryPersistence` come architettura principale
- `self_eval` nel save path
- `build_dataset.py`

Compatibilita' residua:

- `legacy.py` mantiene `MemoryRecord` e `NumericPayload` solo come ponte di migrazione
- `storage.py` e `self_eval.py` sono shim espliciti di deprecazione

# 4. Reused infrastructure

Parti riusate e riposizionate:

- SQLite transazionale e rollback
- artifact file registry
- hashing stabile del contesto
- export JSONL
- harness di test
- sampling/curation, ora come utility opzionale

# 5. Remaining gaps

Prima di arrivare a tracing/update/evaluation seri mancano ancora:

- model adapter reali
- produzione di `TraceArtifact` da osservabilita' interna reale
- pathway detection e localization logic
- update planner/applicator reale
- benchmark e executor di evaluation su modello
- safety gates pre/post update
- rollback di update reali

# 6. Test status

Test attuali:

- `tests/test_domain_and_ingestion.py`
  Copre validazione EpisodeRecord e migrazione legacy.
- `tests/test_store_entities.py`
  Copre persistenza di episode, trace, update candidate, evaluation spec/run.
- `tests/test_selection.py`
  Copre sampling/caps della selection utility.
- `tests/test_cli.py`
  Copre save/list/export da CLI.

Stato attuale:

- `9 passed`

# 7. Next recommended steps

1. Definire un contratto piu' preciso per `trace_input_spec`, `artifact_refs` e `candidate_localization`.
2. Introdurre un primo `ModelAdapterSpec` operativo senza ancora implementare update.
3. Aggiungere manifest piu' ricchi per run, export ed evaluation.
4. Costruire un evaluation executor separato dal persistence layer.
5. Solo dopo, introdurre tracing reale su un piccolo modello open-weight.
6. Solo dopo il tracing, progettare `UpdateCandidate` con budget e localization derivati da evidenza reale.
