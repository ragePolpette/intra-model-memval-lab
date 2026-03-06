# Correzioni P0 (Prima Cosa Da Fare)

Aggiornato al: 2026-03-06
Contesto: nuova direzione progetto con doppio profilo memoria:
- `internal` = numeric-native
- `mcp` = text-native (senza conversione numerica artificiale)

## Obiettivo P0
Allineare il codice alla direzione aggiornata.  
Queste correzioni sono la **prima cosa da fare** prima di nuovi sviluppi.

## Sequenza P0 (ordine obbligato)

1. `schemas.py` profile-aware
- Introdurre profilo esplicito (`internal` / `mcp`).
- Rimuovere assunzione globale `modality_primary=numeric`.
- Rendere i campi payload condizionali al profilo:
  - `internal`: `raw_numeric` richiesto
  - `mcp`: `text_view` richiesto

2. `storage.py` profile-aware persistence
- Separare validazione/salvataggio per profilo.
- `internal`: mantenere blob numerico + text shadow.
- `mcp`: salvataggio text-first senza obbligo `raw_numeric`.

3. Regole `train_ready` e validator
- `internal` train-ready se payload numerico valido.
- `mcp` train-ready se payload testuale valido.
- Nessuna conversione forzata testo->numerico nel path MCP.

4. `scripts/build_dataset.py` profile-aware export
- Aggiungere `--profile`.
- Export coerente al profilo:
  - `internal`: numeric-oriented (+ text shadow opzionale)
  - `mcp`: text-oriented

5. `src/intra_model_memval/cli.py` policy description
- Esporre policy per profilo.
- Evitare `require_numeric=true` come default universale.

6. Test P0 obbligatori
- Nuovi test per profilo MCP: save/load/list/search/export.
- Test di non-regressione internal.
- Test esplicito: MCP non crea payload numerico sintetico.

## Criterio di completamento P0
P0 e' completo quando:
- entrambi i profili funzionano con validazioni corrette,
- i test passano su entrambi i profili,
- non ci sono assunzioni numeric-only nel path MCP.

## Blocco successivo (dopo P0)
Solo dopo P0 si passa a:
- manifest quality gates,
- parity semantica cross-profile,
- orchestrazione analyzer e update pesi.

