# intra-model-memval-lab

Repository laboratorio per esperimenti auditabili su:

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec` / `EvaluationRun`
- store / ingestion / evaluation / CLI

Il repository resta un laboratorio generale. Non e' un repo GPT-2 dedicato.
La verticale `GPT-2 small` vive qui come primo adapter reale, primo trace backend concreto
e primo harness di baseline evaluation prima di qualsiasi update dei pesi.

## Stato attuale

Il repo implementa oggi:

- persistenza SQLite + artifact registry tramite `ExperimentStore`
- ingestion di `EpisodeRecord`
- adapter base riusabile per modelli causali
- `GPT2SmallAdapter` reale basato su Hugging Face + PyTorch
- `TraceRunner` reale con salvataggio artifact e metadata di riproducibilita'
- baseline evaluation su dataset sintetico `country -> capital`
- CLI per trace ed evaluation baseline

Il repo non implementa ancora:

- localization reale
- pathway detection
- localized updates
- regression analysis seria post-update

## Struttura

- `src/intra_model_memval/domain/`: contratti dati principali
- `src/intra_model_memval/persistence/`: store transazionale e artifact registry
- `src/intra_model_memval/ingestion/`: normalizzazione e acquisizione episodi
- `src/intra_model_memval/adapters/`: contract base + adapter GPT-2 small
- `src/intra_model_memval/tracing/`: trace runner e serializzazione artifact
- `src/intra_model_memval/evaluation/`: dataset loader, metriche e baseline harness
- `src/intra_model_memval/cli.py`: entrypoint CLI
- `experiments/gpt2_small/`: dataset, prompt, config e note della verticale
- `docs/GPT2_EXPERIMENT_PLAN.md`: piano della verticale GPT-2 small
- `docs/EVALUATION_PROTOCOL.md`: protocollo baseline corrente

## Installazione

Base minima:

```bash
pip install -e .
```

Per la verticale GPT-2 small:

```bash
pip install -e .[gpt2]
```

Per sviluppo e test:

```bash
pip install -e .[dev]
```

## CLI

Help generale:

```bash
intra-model-exp --help
```

Trace GPT-2 small:

```bash
intra-model-exp --db-path data/exp.db --artifact-dir data/artifacts run-gpt2-trace ^
  --content-text "What is the capital of France?" ^
  --prompt-template "Question: {content_text}\nAnswer:"
```

Baseline evaluation GPT-2 small:

```bash
intra-model-exp --db-path data/exp.db --artifact-dir data/artifacts run-gpt2-baseline-eval
```

Comandi principali:

- `save-episode`
- `list-episodes`
- `register-trace`
- `run-gpt2-trace`
- `list-traces`
- `create-update-candidate`
- `list-update-candidates`
- `create-evaluation-spec`
- `create-evaluation-run`
- `run-gpt2-baseline-eval`
- `list-evaluation-runs`
- `export-episodes`
- `list-runs`

## Verticale GPT-2 small

Materiale esperimentale:

- `experiments/gpt2_small/datasets/synthetic_capitals.json`
- `experiments/gpt2_small/prompts/capital_prompt.txt`
- `experiments/gpt2_small/configs/baseline_eval.json`
- `experiments/gpt2_small/configs/trace_example.json`

La baseline corrente e' intenzionalmente sobria:

- forward pass riproducibile ragionevole in `eval()`
- raccolta di logits e hidden states
- trace artifact materializzato con summary JSON + tensor dump
- evaluation baseline pre-update su split `target` / `related` / `unrelated`

## Note sui documenti storici

I documenti storici in `docs/PROJECT_SPEC.md`, `docs/TARGET_OBJECTIVE.md`,
`docs/DEV_STATUS_ROADMAP.md`, `docs/correzioni_p0.md` e
`docs/MISSING_POINTS_SCHEMA_AND_PLAN.md` restano come contesto.

Per lo stato reale usare:

- `README.md`
- `docs/GPT2_EXPERIMENT_PLAN.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/ARCHITECTURE.md`
- `GPT2_VERTICAL_SLICE_REPORT.md`
