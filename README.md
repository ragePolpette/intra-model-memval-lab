# intra-model-memval-lab

Progetto standalone per **memoria intra-model categorizzata + scoring + validation**.

Scopo:
- costruire una pipeline hardened per memorie selezionabili per training
- mitigare feedback loop con regole di qualità/provenance
- esportare dataset in formato adatto a modelli di test (es. GPT-2 small)
- supportare due profili coerenti:
  - `internal`: memoria numeric-native (`raw_numeric`) con text shadow opzionale
  - `MCP`: memoria text-native (`text_view`) senza conversione numerica artificiale

Questo progetto e' separato da `llm-memory` runtime MCP:
- `llm-memory` = memoria operativa live
- `intra-model-memval-lab` = pipeline offline di curation/validation/export

## Principio dati

- Profilo `internal`:
  - `raw_numeric` e' la source of truth per training intra-model.
  - `text_view` e' una vista secondaria per audit/debug/human review.
- Profilo `MCP`:
  - `text_view` e' la source of truth del payload memoria.
  - non viene forzata conversione a numerico pseudo-sintetico.

## Struttura

- `docs/PROJECT_SPEC.md`: specifiche proposte
- `src/intra_model_memval/`: package Python
- `scripts/`: script eseguibili
- `tests/`: test minimi pipeline

Core storage service:
- `src/intra_model_memval/storage.py` (`MemoryPersistence`)
- persistenza duale atomica (profilo internal): blob numerico + indice SQLite + text shadow
- self-eval rule nel save path con toggle `self_eval_enforced`

## Quickstart

```bash
cd intra-model-memval-lab
python -m src.intra_model_memval.cli --help
python scripts/build_dataset.py --help
```
