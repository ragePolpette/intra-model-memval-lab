# intra-model-memval-lab

Questo progetto e' destinato a vivere come **repository standalone** dentro il workspace DEV `Yetzirah`.
Non ha runtime/deploy in `Binah`.

Repository per infrastruttura sperimentale su:

- `episode` acquisition
- `trace artifact` registration
- `update candidate` tracking
- `evaluation spec/run` management

Non e' ancora un sistema di memoria intra-modello.
Non fa activation tracing reale.
Non fa pathway detection.
Non fa micro-update dei pesi.
Non esegue valutazioni su un modello reale.

## Cosa e' il repo adesso

Questo progetto e' stato riallineato da un prototype storage-centered
a una base di infrastruttura per esperimenti auditabili e riproducibili.

Blocchi principali:

- `EpisodeRecord`: episodio/fatto osservato con contesto e provenienza.
- `TraceArtifact`: contratto e registry per artefatti di tracing futuri.
- `UpdateCandidate`: proposta di update localizzato, non ancora eseguito.
- `EvaluationSpec` / `EvaluationRun`: contratti per valutazione target/related/unrelated.
- `ExperimentStore`: persistenza SQLite + artifact registry + run manifests.
- `EpisodeSelectionPolicy`: utility opzionale di curation/sampling.

## Legacy removed

Il repository non contiene piu' bridge interni verso il vecchio paradigma numeric-memory.

Sono stati rimossi dal core:

- `legacy.py`
- shim di compatibilita' per `MemoryPersistence`
- shim di compatibilita' per `self_eval`
- export di `MemoryRecord` / `NumericPayload`

## Cosa non e' ancora

- non e' un runtime di memoria live
- non e' un trainer
- non e' un weight update engine
- non e' un adapter per GPT-2 small o altri modelli specifici

## Struttura

- `src/intra_model_memval/domain/`: contratti dati principali
- `src/intra_model_memval/persistence/`: store transazionale e artifact registry
- `src/intra_model_memval/ingestion/`: normalizzazione e acquisizione episodi
- `src/intra_model_memval/evaluation/`: harness minimale per evaluation run
- `src/intra_model_memval/adapters/`: punto di estensione futuro per model adapters
- `src/intra_model_memval/selection.py`: sampling/curation opzionale
- `docs/ARCHITECTURE.md`: architettura attuale
- `docs/MIGRATION_NOTES.md`: cosa e' stato deprecato e perche'
- `LEGACY_REMOVAL_REPORT.md`: report della rimozione definitiva del legacy
- `REALIGNMENT_REPORT.md`: report del refactor

## CLI

Se il package e' installato:

```bash
intra-model-exp --help
```

Da repository:

```bash
python scripts/export_experiment_snapshot.py --help
```

Comandi principali:

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

## Note sui documenti storici

I documenti storici in `docs/PROJECT_SPEC.md`, `docs/TARGET_OBJECTIVE.md`,
`docs/DEV_STATUS_ROADMAP.md`, `docs/correzioni_p0.md` e
`docs/MISSING_POINTS_SCHEMA_AND_PLAN.md` sono stati declassati a note di contesto.
Per lo stato reale del repository usare:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/MIGRATION_NOTES.md`
- `REALIGNMENT_REPORT.md`
