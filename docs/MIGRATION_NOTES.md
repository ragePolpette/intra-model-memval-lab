# Migration Notes

## 1. Legacy removed from core

Il repository non mantiene piu' alcun ponte interno verso il vecchio paradigma.

Sono stati rimossi:

- `MemoryRecord`
- `NumericPayload`
- `NumericEncoding`
- `MemoryPersistence`
- `self_eval` come shim compatibile
- qualunque export pubblico del vecchio modello dal package root

## 2. Perche'

Il paradigma precedente trascinava un errore mentale strutturale:

- memoria = record numerico persistito
- conoscenza = payload esterno
- valutazione = scoring nel save path

Per un progetto ancora giovane era piu' corretto rompere esplicitamente questa continuita'
invece di mantenere compatibilita' interna superflua.

## 3. Cosa e' stato mantenuto

E' stata mantenuta solo l'infrastruttura neutra:

- `ExperimentStore`
- artifact registry
- hashing stabile del contesto
- ID generation
- run manifests basilari
- export JSONL
- selection come utility opzionale

## 4. Come leggere il repo ora

Il core del progetto ruota solo attorno a:

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec`
- `EvaluationRun`
- `ExperimentRun`

## 5. Cosa manca ancora

Restano gap legittimi della nuova architettura:

- activation tracing reale
- pathway detection
- localized update engine
- evaluation su modello reale

## 6. Riferimenti

Per lo stato reale del repository usare:

- `README.md`
- `docs/ARCHITECTURE.md`
- `REALIGNMENT_REPORT.md`
- `LEGACY_REMOVAL_REPORT.md`
