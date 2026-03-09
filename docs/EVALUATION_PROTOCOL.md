# Evaluation Protocol

## Scopo della baseline in questa fase

La baseline corrente misura il comportamento di `GPT-2 small` prima di qualsiasi
update dei pesi. Serve a stabilire un punto di partenza auditabile.

Non e':

- una valutazione completa di knowledge editing
- una prova di localization
- una prova di pathway detection
- una regression analysis seria post-update

## Input

La baseline usa il dataset sintetico:

- `experiments/gpt2_small/datasets/synthetic_capitals.json`

con prompt template:

- `experiments/gpt2_small/prompts/capital_prompt.txt`

Ogni caso produce:

- `prompt`
- `expected_response`
- metadati sul gruppo (`target`, `related`, `unrelated`)

## Significato degli split

`target`

- casi al centro del primo banco prova

`related`

- casi vicini nello stesso dominio fattuale

`unrelated`

- casi usati come controllo separato nello stesso task sintetico

In questa fase gli split non rappresentano ancora una teoria forte di generalizzazione.
Sono solo una partizione minima utile per leggere la baseline in modo piu' ordinato.

## Metrica corrente

La baseline usa scoring del prossimo token.

Per ogni prompt:

1. si esegue un forward pass
2. si prende il vettore logits dell'ultimo token del prompt
3. si tokenizza la risposta attesa nel formato coerente con next-token scoring
4. si valuta il primo token atteso

Metriche per caso:

- `expected_first_token_rank`
- `expected_first_token_logprob`
- `top1_match`
- `topk_match`

Metriche aggregate:

- accuracy top-1 per gruppo
- accuracy top-k per gruppo
- media del rank atteso
- media della logprob attesa

## Artifact distinti

La verticale distingue esplicitamente:

- `TraceArtifact`: artifact di tracing per un episodio specifico
- baseline inference: forward pass eseguiti dall'harness
- `EvaluationRun`: risultato aggregato della baseline

La baseline evaluation non salva automaticamente un trace per ogni caso.
Questo evita di confondere tracing ed evaluation.

## Limiti attuali

- scoring solo sul primo token atteso
- niente generazione multi-token come metrica primaria
- niente confronto pre/post update
- dataset sintetico piccolo
- nessuna stima seria di regressione fuori dominio
- nessun criterio di successo per localization o editing

## Uso corretto dei risultati

I risultati attuali vanno letti come baseline pre-update, non come misura conclusiva
della qualita' del sistema di editing che ancora non esiste.
