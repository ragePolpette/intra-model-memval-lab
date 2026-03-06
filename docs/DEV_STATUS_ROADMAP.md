# intra-model-memval-lab - Stato sviluppo e roadmap

Aggiornato al: 2026-03-05
Branch attivo: `feature/llm-memory-autoval-v2`
Worktree: `C:\Users\Gianmarco\Urgewalt\Yetzirah-parallel`

## 1. Obiettivo progetto
Progetto standalone per gestione memorie intra-model categorizzate, con:
- salvataggio duale profile-aware:
  - `internal`: numeric-native (`raw_numeric` + text shadow)
  - `MCP`: text-native (`text_view` primary, niente conversione numerica artificiale)
- scoring/autovalutazione memoria
- selezione dataset con mitigazione feedback loop
- export dataset per training/eval offline

## 2. Stato attuale (implementato)

### Core persistence
- `MemoryPersistence` con persistenza atomica blob + SQLite.
- Upsert idempotente su `entry_id`.
- Batch `save_many` transazionale con rollback completo su errore.
- Query API:
  - `list_records(...)` con filtri + paginazione
  - `search_records(...)` con filtri
- Ordinamento stabile query: `importance desc`, `created_at asc`, `entry_id asc`.

### Self-eval scoring (regola autovalutazione)
- Integrazione scoring nel save path.
- Toggle enforcement disponibile (`self_eval_enforced`) e mantenuto attivabile/disattivabile.
- Metadati di scoring e context hash deterministico.

### Selezione dataset (mitigazione bias loop)
- Policy engine `selection.py` implementato:
  - novelty filter (`novelty_score >= soglia`)
  - bucket distribution (`top/mid/low` con target 60/25/15)
  - quota minima `is_external`
  - diversity caps per `writer_model` e `conversation_id`
  - backfill controllato quando un bucket non raggiunge target
- API: `select_training_records(...)` + dataclass di policy/result/stats.

### Export dataset
- Script `scripts/build_dataset.py` aggiornato da scaffold a pipeline reale su SQLite:
  - input da `--db-path` / `--blob-dir`
  - applica `SelectionPolicy`
  - output:
    - JSONL numerico
    - JSONL text shadow
  - summary JSON con metriche principali e stato quota esterna.

### Qualita'
- Suite test attuale: `20 passed`.
- Copertura funzionale presente su:
  - storage
  - self-eval
  - selection policy
  - dataset build script

## 3. Cosa manca (gap aperti)

### P0 (core per uso continuativo)
- Manifest di export versionato (`manifest.json`) con:
  - policy usata
  - statistiche output
  - gate pass/fail (novelty, quota external, distribuzione).
- Schema dataset trainer-ready esplicito e versionato (`schema_version` export).
- CLI dedicata per validazione post-export (consistenza record, duplicati, bucket drift).

### P1 (hardening tecnico)
- Evidence/provenance hardening (attestazioni forti oltre `declared_only`).
- Policy di dedup/consolidation estesa con similarita' semantica + `context_hash`.
- Guardrail su qualita' low bucket (esclusione errori grossolani in modo automatico).

### P2 (integrazione training/eval)
- Adapter per dataset target (es. GPT-2 small) con formato pronto trainer.
- Pipeline eval automatica (report quality + regressioni policy).
- Baseline comparativa pre/post filtro anti-feedback-loop.

## 4. Roadmap proposta

1. Milestone A (rapida, 1 sessione)
- Aggiungere `manifest.json` nel build dataset.
- Definire schema export stabile (numeric/text) + versione.
- Aggiungere test su manifest e quality gates.

2. Milestone B (2 sessioni)
- Implementare modulo `validators.py` per gate offline.
- Introdurre check distribuzione con tolleranza configurabile.
- Introdurre report sintetico errore/cause per export non conforme.

3. Milestone C (2-3 sessioni)
- Implementare dedup/consolidation operativa.
- Esporre flag policy in CLI per usare o no consolidazione.
- Aggiungere test su casi same-context vs cross-context.

4. Milestone D (successivo progetto training)
- Nuovo modulo adapter trainer.
- Eval automatica su modello test.
- Decisione su passaggio a profilo hardened intra-model completo.

## 5. File chiave da cui ripartire
- `src/intra_model_memval/storage.py`
- `src/intra_model_memval/self_eval.py`
- `src/intra_model_memval/selection.py`
- `scripts/build_dataset.py`
- `tests/test_storage.py`
- `tests/test_self_eval.py`
- `tests/test_selection.py`
- `tests/test_build_dataset_script.py`

## 6. Comandi rapidi ripresa lavoro

```powershell
cd C:\Users\Gianmarco\Urgewalt\Yetzirah-parallel\intra-model-memval-lab
pytest -q
python scripts/build_dataset.py --help
```

## 7. Ultimi commit rilevanti
- `8887bab` `feat(memval): integrate policy-driven sqlite dataset export`
- `9de5864` `feat(memval): add policy engine for dataset selection and bias controls`
- `85f35b8` `feat(memval): add filtered query APIs with stable pagination`
- `0d57a6b` `feat(memval): enforce and enrich self-eval scoring in save path`
- `e9f10b6` `feat(memval): add core dual persistence service with atomic idempotent save`
