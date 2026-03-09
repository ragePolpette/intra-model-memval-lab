# GPT-2 Experiment Plan

## Obiettivo

Introdurre `GPT-2 small` nel repository esistente come:

- primo model adapter reale
- primo backend reale per tracing
- primo harness concreto di baseline evaluation

L'obiettivo non e' trasformare il repository in un progetto GPT-2 dedicato.
Il repository resta il laboratorio principale per esperimenti generici su trace,
evaluation e futuri update localizzati.

## Perche' dentro questo repo

Questa verticale usa direttamente i contratti del laboratorio:

- `EpisodeRecord`
- `TraceArtifact`
- `EvaluationSpec`
- `EvaluationRun`
- `ExperimentRun`

Separare GPT-2 in un altro repo renderebbe piu' debole il test dell'architettura
riallineata. Tenerlo qui permette di verificare che il framework generico sia
capace di sostenere un primo adapter reale senza specializzarsi in modo rigido.

## Struttura

Codice framework:

- `src/intra_model_memval/adapters/base.py`
- `src/intra_model_memval/adapters/gpt2_adapter.py`
- `src/intra_model_memval/tracing/runner.py`
- `src/intra_model_memval/tracing/serializers.py`
- `src/intra_model_memval/evaluation/datasets.py`
- `src/intra_model_memval/evaluation/metrics.py`
- `src/intra_model_memval/evaluation/harness.py`
- `src/intra_model_memval/cli.py`

Materiale esperimentale:

- `experiments/gpt2_small/datasets/synthetic_capitals.json`
- `experiments/gpt2_small/prompts/capital_prompt.txt`
- `experiments/gpt2_small/configs/baseline_eval.json`
- `experiments/gpt2_small/configs/trace_example.json`
- `experiments/gpt2_small/notes/README.md`

## Dataset sintetico

Il dataset iniziale e' piccolo, seedato e leggibile. Il task e':

- `country -> capital`

Gli split servono solo a dare una struttura minima alla baseline:

- `target`: casi direttamente al centro della prima valutazione
- `related`: casi dello stesso dominio per controlli vicini
- `unrelated`: casi tenuti separati come controllo piu' largo

Il formato resta semplice e trasparente: JSON con `case_id`, `country`,
`capital`, `expected_response`.

## Trace Flow

1. La CLI o il chiamante fornisce un `EpisodeRecord` o testo equivalente.
2. `TraceRunner` costruisce il prompt da template semplice.
3. L'adapter GPT-2 esegue un forward pass in `eval()`.
4. Vengono raccolti:
   - prompt text
   - token ids
   - logits
   - hidden states
5. Il runner serializza:
   - summary JSON leggibile
   - tensor dump completo
6. Lo store registra:
   - `ExperimentRun`
   - artifact fisici
   - `TraceArtifact`

## Baseline Eval Flow

1. `load_synthetic_dataset()` legge dataset e prompt template.
2. Il dataset diventa un `EvaluationSpec`.
3. `EvaluationHarness.run_baseline()` esegue inference su tutti i casi.
4. La baseline misura il comportamento pre-update con scoring del prossimo token.
5. Lo store salva:
   - `ExperimentRun`
   - `EvaluationSpec`
   - `EvaluationRun`

## Gia' implementato

- contract adapter base
- adapter GPT-2 small reale
- trace runner minimo ma reale
- serializzazione artifact
- dataset sintetico minimo
- harness baseline pre-update
- CLI per trace ed evaluation
- test unit/integration leggeri e mockabili

## Non ancora implementato

- localization
- pathway detection
- update engine
- update budget enforcement reale
- regression suite post-update
- confronto pre/post trace

## Prossimi step naturali

- aggiungere scoring multi-token e generazione controllata
- introdurre trace selection piu' mirata per casi target
- definire contratti per localization candidates
- preparare una prima pipeline di confronto pre/post update
