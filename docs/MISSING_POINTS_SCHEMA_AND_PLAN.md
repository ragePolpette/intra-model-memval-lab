# Intra-Model Memval - Schema e Piano Implementazione (Punti Mancanti)

Aggiornato al: 2026-03-06
Stato base: storage/scoring/selection/export offline presenti.
Focus: completare il sistema end-to-end con dual profile (`internal` + `MCP`) e loop cronico analyzer->update pesi.

## 1. Schema architetturale target

```mermaid
flowchart LR
    A[Main Model Runtime] -->|memories| B{Memory Save Layer}
    B -->|internal profile| C[(Internal Store: numeric+text)]
    B -->|MCP profile| D[(MCP Store: text primary)]

    C --> E[Dataset Builder]
    D --> E
    E --> F[Policy Engine\nnovelty/external quota/buckets]
    F --> G[(Curated Dataset + Manifest)]

    G --> H[Analyzer Model\n(clone base weights)]
    H --> I[Delta Proposal Engine\n(target 1.0%-1.5%)]
    I --> J[Safety Gates\nquality/regression/drift]
    J -->|pass| K[Apply Limited Update\non Main Model]
    J -->|fail| L[Reject + Report + Rollback]
```

## 2. Contratto unico (parita' semantica)

### 2.1 Campi comuni obbligatori
- `entry_id`, `category`, `importance_score`, `novelty_score`
- `is_external`, `provenance_level`, `context_hash`
- `writer_model`, `writer_agent_id`, `created_at_utc`, `metadata`
- metadati scoring/autovalutazione (surprise, inference, negative_impact, class)

### 2.2 Payload per profilo
- `internal`: `modality_primary=numeric`, `raw_numeric` richiesto, `text_view` opzionale.
- `MCP`: `modality_primary=text`, `text_view` richiesto, `raw_numeric` non richiesto e non sintetizzato.

### 2.3 Regole invariate tra profili
- stessa formula scoring/autovalutazione
- stesso filtro novelty e quota `is_external`
- stessa distribuzione bucket per dataset curation
- stesso algoritmo `context_hash` e stessa tracciabilita'

## 3. Punti mancanti (work packages)

### WP1 - Profile Contract & Validators
Output:
- schema v2 profile-aware (`internal`/`MCP`)
- validatori payload-specific
- test di parita' semantica cross-profile

### WP2 - MCP Text Writer
Output:
- adapter MCP per salvataggio text-native
- mapping metadata/scoring nel formato comune
- test integrazione MCP writer (no numeric conversion)

### WP3 - Manifest & Quality Gates
Output:
- `manifest.json` per ogni export dataset
- gate automatici: novelty, quota external, bucket tolerance, duplicate check
- report pass/fail machine-readable

### WP4 - Analyzer Orchestrator (cronico)
Output:
- scheduler/runner periodico
- read curated dataset + analysis job
- generazione proposta update pesi (senza apply diretto)

### WP5 - Limited Weight Update Engine
Output:
- applicazione update limitato `1.0%-1.5%` pesi target
- policy di selezione parametri aggiornabili
- audit log completo pre/post update

### WP6 - Safety & Rollback
Output:
- gate regressione funzionale/qualita'
- blocco automatico update non sicuri
- rollback rapido a checkpoint precedente

## 4. Piano implementazione (sequenza consigliata)

## Fase A (alta priorita', fondazioni cross-profile)
1. Definire `schema_version=v2` profile-aware.
2. Implementare validatori `internal` vs `MCP`.
3. Aggiornare export per includere `profile` + `manifest`.
4. Aggiungere test di compatibilita' e non-regressione.

Deliverable:
- schema v2 e validator
- manifest base
- test green su pipeline attuale

## Fase B (MCP path completo)
1. Implementare writer MCP text-native.
2. Integrare self-eval/scoring invariato.
3. Garantire parity su `importance_score`, `context_hash`, metadati obbligatori.

Deliverable:
- `mcp_writer` operativo
- test e2e MCP->dataset export

## Fase C (loop cronico analyzer)
1. Implementare orchestratore job periodici.
2. Ingerire dataset curato e produrre `delta proposal`.
3. Definire formato standard `delta_plan.json`.

Deliverable:
- job cronico ripetibile
- output proposal versionato

## Fase D (update limitato + safety)
1. Implementare motore apply con cap 1.0%-1.5%.
2. Aggiungere gate pre/post update.
3. Implementare rollback automatico su fail gate.

Deliverable:
- update engine production-like
- runbook sicurezza + rollback

## 5. Criteri di accettazione (Done per i mancanti)
- parita' semantica dimostrata tra `internal` e `MCP`.
- MCP salva solo testo come payload primario.
- manifest + gates bloccano export/update non conformi.
- analyzer cronico produce delta auditabile.
- update engine rispetta limite 1.0%-1.5% e rollback testato.

## 6. Rischi principali e mitigazioni
- Drift tra profili -> test di parity obbligatori su fixture comuni.
- Feedback loop residuo -> novelty hard filter + external quota + low-bucket guard.
- Update instabile -> apply in staging + gate regressione + rollback immediato.
- Overfitting locale -> sampling diversity caps + valutazioni out-of-sample.

## 7. Primo sprint operativo proposto (immediato)
1. WP1 completo (schema v2 + validator + test).
2. WP2 baseline (MCP text writer + test base).
3. WP3 minimo (manifest + duplicate/novelty/external checks).

Tempo stimato: 2-3 sessioni di sviluppo.
