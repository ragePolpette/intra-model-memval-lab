# 1. Branch used

`feature/gpt2-small-baseline-tracing`

# 2. Summary of implementation

La verticale GPT-2 small e' stata aggiunta dentro il repository esistente come
primo adapter reale, primo trace backend concreto e primo harness di baseline
evaluation. Il framework generale del repo resta separato dal materiale
esperimentale, che vive in `experiments/gpt2_small`.

# 3. Repository structure changes

Moduli aggiunti o estesi nel package:

- `src/intra_model_memval/adapters/`
- `src/intra_model_memval/tracing/`
- `src/intra_model_memval/evaluation/`
- `src/intra_model_memval/cli.py`

Materiale aggiunto per l'esperimento:

- `experiments/gpt2_small/configs/`
- `experiments/gpt2_small/datasets/`
- `experiments/gpt2_small/prompts/`
- `experiments/gpt2_small/runs/`
- `experiments/gpt2_small/notes/`

Test aggiunti:

- `tests/unit/`
- `tests/integration/`
- `tests/fixtures/`

# 4. GPT-2 adapter design

L'adapter base definisce un contratto minimo per:

- load
- tokenize
- forward
- decode token ids
- metadata/description

`GPT2SmallAdapter` usa import lazy di `torch` e `transformers`, carica il checkpoint
standard `gpt2`, mette il modello in `eval()` e restituisce logits, hidden states
e metadata riproducibili senza mescolare logica di modello e CLI.

# 5. Trace flow implemented

Il `TraceRunner`:

1. riceve un `EpisodeRecord` o un payload equivalente
2. costruisce il prompt da template semplice
3. esegue il forward pass via adapter
4. serializza un summary JSON leggibile
5. serializza un tensor dump completo
6. registra `ExperimentRun`, artifact fisici e `TraceArtifact`

Il trace salva input text, prompt, token ids, logits, hidden states e metadata di
riproducibilita' basilari.

# 6. Baseline evaluation implemented

La baseline evaluation:

- legge un dataset sintetico `country -> capital`
- costruisce un `EvaluationSpec`
- esegue inference su split `target`, `related`, `unrelated`
- misura rank/logprob del primo token atteso
- salva un `EvaluationRun` distinto dai trace artifact

Questa fase resta esplicitamente pre-update.

# 7. Tests

Test aggiunti o aggiornati per coprire:

- contract adapter e path di load mockabile per GPT-2
- parsing del dataset sintetico
- materializzazione `TraceArtifact`
- flow del baseline harness
- orchestration CLI di `run-gpt2-trace` e `run-gpt2-baseline-eval`

La suite usa adapter fake per evitare download del modello nei test ordinari.

# 8. Documentation updates

Documenti aggiunti o aggiornati:

- `README.md`
- `docs/GPT2_EXPERIMENT_PLAN.md`
- `docs/EVALUATION_PROTOCOL.md`
- `GPT2_VERTICAL_SLICE_REPORT.md`

# 9. Current limitations

Manca ancora tutto cio' che sarebbe necessario per arrivare a:

- localization reale
- pathway detection
- localized updates
- regression analysis seria pre/post update

Inoltre la baseline corrente valuta soprattutto il primo token atteso e usa un
dataset sintetico volutamente piccolo.

# 10. Next recommended steps

- aggiungere scoring multi-token e generazione controllata
- definire una prima rappresentazione dei localization candidates
- collegare trace selection e evaluation cases
- preparare un confronto pre/post update senza fingere ancora editing reale
- espandere il dataset sintetico in modo sobrio prima di passare a dati meno puliti

# Commit history summary

Commit principali creati durante il lavoro:

- `feat(adapters): add gpt2 baseline tracing pipeline`
- `feat(dataset): add synthetic capitals experiment assets`
- `test(tracing): cover gpt2 trace and baseline flows`
- `build(pyproject): add optional gpt2 runtime deps`
- `docs(gpt2): document vertical slice protocol and report`
