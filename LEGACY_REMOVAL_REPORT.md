# 1. Legacy components removed

Rimossi dal repository:

- `src/intra_model_memval/legacy.py`
- `src/intra_model_memval/storage.py` come shim di `MemoryPersistence`
- `src/intra_model_memval/self_eval.py` come shim compatibile
- export pubblici dal package root di `MemoryRecord`, `NumericPayload`, `NumericEncoding`
- test di migrazione legacy in `tests/test_domain_and_ingestion.py`

Concetti eliminati dal core:

- `MemoryRecord`
- `NumericPayload`
- `NumericEncoding`
- `MemoryPersistence`
- `self_eval` come compat layer
- bridge interni verso numeric-memory persistence

# 2. Reused neutral infrastructure

Mantenuto e riposizionato:

- `ExperimentStore` come persistenza neutra per episodi, trace, update, evaluation
- artifact registry file-based
- hashing stabile del contesto in `utils/hashing.py`
- ID e timestamp generation in `utils/ids.py`
- run manifests basilari tramite `ExperimentRun`
- export JSONL di snapshot sperimentali
- `selection.py` come utility opzionale di sampling/capture

# 3. Contract cleanup

Il core dei contratti dati ruota ora solo attorno a:

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec`
- `EvaluationRun`
- `ExperimentRun`

Pulizia effettuata:

- nessun campo legato a payload numerici nel dominio centrale
- nessun alias o wrapper che evochi `raw_numeric`
- nessuna proprieta' tipo `train_ready`
- nessun flag che faccia sembrare esistente apprendimento o update reale

# 4. CLI and docs cleanup

CLI:

- nessun comando legacy
- nessuna opzione che richiami numeric-memory persistence
- restano solo operazioni coerenti con episode/trace/update/evaluation/export

Documentazione:

- `README.md` aggiornato con sezione `Legacy removed`
- `docs/ARCHITECTURE.md` aggiornato con rimozione esplicita dei concetti legacy
- `docs/MIGRATION_NOTES.md` riscritto per descrivere rimozione, non compatibilita'
- documenti storici lasciati solo come contesto superato

# 5. Test impact

Rimossi:

- test che validavano la conversione dal vecchio schema legacy

Aggiornati:

- test di dominio/ingestion per verificare che il package non esporti piu' componenti legacy
- test di store, selection e CLI mantenuti sul nuovo modello

Stato finale:

- `9 passed`

# 6. Final architecture consistency check

Il core del repository non comunica piu':

`memoria = numeric record persistence`

Per ragioni precise:

- il package root non esporta piu' il vecchio modello
- i moduli legacy sono stati rimossi, non deprecati
- non esistono shim interni che tengano vivo il vecchio cuore architetturale
- i contratti dati centrali sono solo entita' sperimentali coerenti con la nuova direzione

# 7. Remaining non-legacy gaps

Gap ancora aperti, ma legittimi rispetto alla nuova architettura:

- activation tracing reale
- pathway detection
- localized update engine
- evaluation su modello reale

Questi gap non sono piu' confusi con residui del vecchio paradigma. Sono semplicemente il lavoro futuro ancora da implementare.
