# claude.md — intra-model-memval-lab

## Project summary

Progetto Python **standalone** (non MCP runtime) per la pipeline offline di **curation, validation ed export di memorie intra-model** destinate al training. Separato da `llm-memory` (runtime operativo): questo è il laboratorio di selezione e scoring.

Principio dati: `raw_numeric` è la source of truth per il training (record senza `raw_numeric` non sono train-ready). `text_view` è vista secondaria per audit/debug.

Funzionalità principali:
- Persistenza duale atomica (blob numerico + indice SQLite + shadow testuale)
- Scoring/autovalutazione memorie (`self_eval.py`)
- Selezione filtrata con paginazione stabile (`selection.py`)
- CLI per operazioni da terminale
- Script `build_dataset.py` per export dataset training
- Mitigazione feedback loop tramite regole qualità/provenance

## Quickstart

```bash
cd intra-model-memval-lab

# Installa (dev)
pip install -e ".[dev]"

# CLI
python -m src.intra_model_memval.cli --help

# Build dataset
python scripts/build_dataset.py --help

# Test
pytest tests/ -v
```

## Architecture overview

```
CLI (cli.py)
    |
MemoryPersistence (storage.py)
    |-- blob numerico (raw_numeric)
    |-- indice SQLite (metadata + ricerca)
    |-- text shadow (text_view, opzionale)
    |-- self_eval rule nel save path (toggle self_eval_enforced)
    |
SelfEval (self_eval.py)   ← scoring deterministico autovalutazione
Selection (selection.py)  ← query filtrate con paginazione stabile
Schemas (schemas.py)      ← modelli Pydantic v2
```

## Key modules / folders

| Percorso | Ruolo |
|---|---|
| `src/intra_model_memval/storage.py` | Core: `MemoryPersistence`, persistenza duale atomica |
| `src/intra_model_memval/self_eval.py` | Scoring deterministico memorie |
| `src/intra_model_memval/selection.py` | Query filtrate con paginazione stabile |
| `src/intra_model_memval/schemas.py` | Modelli dati Pydantic v2 |
| `src/intra_model_memval/cli.py` | Interfaccia CLI |
| `scripts/build_dataset.py` | Export dataset per training |
| `tests/` | Test pipeline (pytest) |
| `docs/PROJECT_SPEC.md` | Specifiche complete del progetto |
| `docs/DEV_STATUS_ROADMAP.md` | Status sviluppo e roadmap |

## Dependencies & tooling

- **Python**: ≥3.11
- **Runtime**: `pydantic>=2.0`
- **Dev/test**: `pytest>=8.0`
- **Build backend**: `hatchling`
- Nessuna dipendenza di rete o servizi cloud

## Configuration

Nessun file `.env` richiesto per l'esecuzione base. Eventuali parametri passati via CLI o argomenti script.

Toggle chiave nel codice:
- `self_eval_enforced` in `MemoryPersistence`: se `True`, richiede che ogni record superi la regola self-eval prima del salvataggio.

## Common commands

```bash
# Installazione dev
pip install -e ".[dev]"

# CLI principale
python -m src.intra_model_memval.cli --help

# Build dataset per training
python scripts/build_dataset.py --help

# Test completi
pytest tests/ -v

# Test specifici
pytest tests/test_storage.py -v
pytest tests/test_self_eval.py -v
pytest tests/test_selection.py -v
```

## Operational notes

- Progetto **offline**: nessun server, nessuna porta, nessun servizio esterno.
- Non fa parte del ciclo deploy MCP (non va in `Binah`).
- L'output tipico è un dataset JSONL o struttura numerica per modelli di test (es. GPT-2 small).
- Modifiche a `storage.py` impattano l'atomicità: testare sempre con `test_storage.py`.

## Known issues / risks

- Versione 0.1.0: API non ancora stabile, schema potrebbe variare.
- `self_eval_enforced` è sperimentale (default off): modalità non hardened.
- Assenza di documentazione sullo schema esatto di `raw_numeric` in `docs/` (verificare `PROJECT_SPEC.md`).
- Nessuna strategia di migrazione dati definita per cambi di schema SQLite.

## Roadmap / next actions

1. **(S)** Stabilizzare e documentare schema `raw_numeric` in `PROJECT_SPEC.md`
2. **(M)** Definire strategia di migrazione SQLite per evoluzioni schema
3. **(M)** Aggiungere filtri per `provenance` e `quality_score` nella CLI
4. **(M)** Rendere `self_eval_enforced` configurabile da CLI/env senza modificare il codice
5. **(L)** Aggiungere export in formato HuggingFace Datasets per integrazione diretta con pipeline training
