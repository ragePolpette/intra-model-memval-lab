# Modulo `consolidation` — Design Contract v0.1

> Il pezzo mancante tra `selection.py` (esiste) e `evaluation/` (esiste):
> lezione → dataset sintetico → LoRA con replay → verdetto.
> Target runtime: Gemma 3 270M IT + Gemma Scope 2 + TransformerLens.

---

## Pipeline del ciclo di sonno

```
[1 Harvest] → [2 Lesson extraction] → [3 Conflict check] →
[4 Variation generation + Verifier] → [5 Replay buffer] →
[6 Train LoRA] → [7 Evaluate] → [8 Promote / Rollback]
```

### 1. Harvest
Input: cluster `PROMOTED` dalla distillation di llm-memory (o `EpisodeRecord` del Lab)
dall'ultimo ciclo, con `selection_score ≥ soglia`.
Output: lista di `(cluster, episodi grezzi)` — gli episodi grezzi servono al punto 4,
non buttarli dopo l'estrazione della lezione.

### 2. Lesson extraction (secondo modello)
Per ogni cluster produce un record `Lesson`:

```python
class Lesson(BaseModel):
    lesson_id: str
    statement: str                    # la lezione in forma dichiarativa
    scope_tags: list[str]             # dominio di applicabilità
    boundary: str                     # quando NON si applica (obbligatorio!)
    source_cluster_ids: list[str]
    source_episode_ids: list[str]
    status: LessonStatus              # CANDIDATE / CONSOLIDATED / SUPERSEDED / CONFLICTED
    consolidation_run_id: str | None
```

`boundary` obbligatorio: una lezione senza confini espliciti genera variazioni
che overgeneralizzano, ed è il difetto n.1 dei dati sintetici.

### 3. Conflict check
Confronta `statement` con le Lesson già `CONSOLIDATED`:
- similarità embedding sopra soglia → LLM judge decide: `merge` / `supersede` / `conflict`
- `conflict` blocca il candidato e lo manda a revisione (umana, per ora)

*POC: rimandabile a v2 (con <10 lezioni i conflitti si vedono a occhio).*

### 4. Variation generation + Verifier — il cuore
Input: `Lesson` + episodi grezzi sorgente.
Output: `SyntheticDataset` (JSONL, registrato come `ArtifactRef`).

Generazione (secondo modello, N=20–50 per lezione):
- ogni esempio = traiettoria di ragionamento che **applica** la lezione
  in un contesto diverso (superficie variata, struttura profonda costante)
- ancoraggio: almeno il 30% delle variazioni parte da un episodio reale parafrasato
- **negative examples** (~20%): contesti dentro `boundary` dove la lezione
  NON va applicata, con il comportamento corretto alternativo

Verifier (passo separato, prompt diverso, idealmente modello diverso):
- per ogni esempio: "questa traiettoria applica la lezione o la recita?"
- score 0–1; sotto soglia → scartato
- se il reject rate > 50%, la lezione è mal formulata → torna al punto 2

```jsonl
{"lesson_id": "...", "kind": "positive|negative|anchor",
 "input": "...", "target": "...", "verifier_score": 0.87}
```

### 5. Replay buffer
Batch di training = mix:
- variazioni nuove (il "da imparare")
- **ancore**, in rapporto configurabile (default 1:1):
  - campione da corpus generico open (FineWeb / instruction dataset)
  - self-distillation: output del modello corrente su capability protette
    (generati PRIMA del training — sono lo snapshot del "vecchio sé")
  - variazioni delle lezioni già consolidate nei cicli precedenti

### 6. Train
- LoRA su Gemma 3 270M IT (rank 8–16 per iniziare)
- record `ConsolidationCandidate` (successore di `UpdateCandidate` per questa pipeline):

```python
class ConsolidationCandidate(BaseModel):
    consolidation_candidate_id: str
    lesson_ids: list[str]
    dataset_ref: ArtifactRef          # il JSONL verificato
    replay_config: dict               # ratio, sorgenti ancore
    train_config: dict                # rank, lr, epochs, seed
    base_model_ref: str               # incluso lo stack LoRA corrente
    status: CandidateStatus           # PREPARED / TRAINED / PROMOTED / ROLLED_BACK
    evaluation_run_ids: list[str]
```

### 7. Evaluate (riusa `EvaluationSpec` / `EvaluationRun`)
Tre famiglie di check, tutte pre/post:
1. **Apprendimento**: held-out set di variazioni mai viste in training
   → la lezione viene applicata?
2. **Regressione comportamentale**: benchmark fissi sulle capability protette
   (stesso set a ogni ciclo, è la serie storica del forgetting)
3. **Drift interno** (v2): attivazioni SAE su un set di probe prompt fissi,
   distanza feature pre/post come metrica di danno collaterale
   → registrata come `TraceArtifact`

### 8. Promote / Rollback
- tutti e tre i check sopra soglia → `PROMOTED` (LoRA in stack)
- regressione sopra tolleranza → `ROLLED_BACK`, lezione resta `CANDIDATE`
  con i dati diagnostici allegati
- decisione merge-stack: rimandata. Stack puro fino a ~5 adapter,
  poi si affronta il merge (problema aperto, v3)

---

## Iperparametri da tarare (in ordine di impatto)
| Parametro | Default iniziale |
|---|---|
| replay ratio (nuovo:ancore) | 1:1 |
| variazioni per lezione | 30 |
| quota negative examples | 20% |
| soglia verifier | 0.7 |
| LoRA rank | 8 |

---

## POC "ciclo brutto" — definizione di fatto
Un singolo ciclo end-to-end, anche manuale nei raccordi:
- [ ] 50 memorie sintetiche → 1 cluster → 1 Lesson
- [ ] 30 variazioni generate + verificate
- [ ] buffer con ancore da corpus generico (skip self-distillation)
- [ ] 1 run LoRA su Gemma 3 270M IT
- [ ] eval pre/post: held-out variations + 1 benchmark di regressione
- [ ] verdetto registrato in `EvaluationRun`

Fuori scope POC: conflict check, SAE drift, multi-ciclo, merge LoRA.
Se questo ciclo gira una volta, tutto il resto è iterazione.

---

## Stato di implementazione nel Lab

Mappatura contratto → codice (`src/intra_model_memval/`):

| Step | Dove | Stato |
|---|---|---|
| 1 Harvest | `selection.select_episodes` (riuso) | esistente |
| 2 Lesson extraction | `consolidation.interfaces.LessonExtractor` | contratto (LLM-backed, fuori repo) |
| 3 Conflict check | `consolidation.interfaces.ConflictChecker` | contratto (v2) |
| 4a Generazione | `consolidation.interfaces.VariationGenerator` | contratto (LLM-backed) |
| 4b Verifier gate | `consolidation.dataset.apply_verifier_gate` + `check_dataset_composition` | implementato (logica pura) |
| 4c Dataset JSONL | `consolidation.dataset.register_dataset` (artifact registry) | implementato |
| 5 Replay buffer | `consolidation.replay.build_replay_buffer` | implementato (mixing deterministico; sourcing ancore a carico del chiamante) |
| 6 Train | `consolidation.training.GemmaLoraTrainer` (extra `[train]`) + `ConsolidationPlanner` | implementato (LoRA reale via peft/transformers) |
| 7 Evaluate | `evaluation.EvaluationExecutor` + `HFTextGenerator`, su `EvaluationSpec` / `EvaluationRun` | implementato (containment match, pre/post) |
| 8 Promote / Rollback | `consolidation.gating.decide_promotion` + `ConsolidationPlanner.apply_decision` | implementato |

Adattamenti rispetto al draft v0.1:

- `id` → `lesson_id` / `consolidation_candidate_id`, per coerenza con la convenzione
  `<entity>_id` del resto del dominio. Il formato JSONL del dataset resta quello
  del contratto (`lesson_id`, `kind`, `input`, `target`, `verifier_score`).
- `ConsolidationCandidate` **affianca** `UpdateCandidate` invece di sostituirlo:
  la rimozione di `UpdateCandidate` tocca store, CLI e test esistenti ed è
  rimandata a una migrazione dedicata.
- Gli iperparametri della tabella vivono in `consolidation.config.ConsolidationConfig`
  con gli stessi default; le soglie di promozione in `consolidation.gating.GateThresholds`.
- Persistenza: tabelle `lessons` e `consolidation_candidates` in `ExperimentStore`,
  con `RunKind.CONSOLIDATION` per i run manifest.
- Coerentemente con il resto del repo, nessun comportamento fake: generazione e
  verifica LLM restano dietro `Protocol`; il training LoRA e la generazione HF
  sono implementati ma vivono nell'extra opzionale `[train]` (import lazy, il
  core resta senza dipendenze ML).

## Esecuzione del POC (macchina GPU)

```bash
pip install -e ".[train]"
python scripts/run_consolidation_poc.py \
    --workdir runs/poc-001 \
    --lesson-json examples/consolidation_poc/lesson.json \
    --variations-jsonl examples/consolidation_poc/variations.jsonl \
    --anchors-jsonl examples/consolidation_poc/anchors.jsonl \
    --regression-jsonl examples/consolidation_poc/regression.jsonl
```

Lo script esegue un ciclo completo: gate del verifier, split held-out, replay
buffer con ancore, eval pre, training LoRA, eval post, verdetto promote/rollback.
Tutto (store SQLite, dataset, adapter, `verdict.json`) finisce sotto `--workdir`.
I file in `examples/consolidation_poc/` sono fixture giocattolo per provare il
loop; per il POC vero vanno sostituiti con le 30 variazioni generate+verificate
da LLM. Scoring eval: containment match normalizzato — volutamente rozzo, da
sostituire quando il loop gira.

Semplificazioni note del trainer POC: loss LM full-sequence (niente masking
completion-only), un adapter per run (niente stacking sul LoRA precedente).
