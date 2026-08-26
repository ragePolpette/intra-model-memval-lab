# `consolidation` module — Design Contract v0.1

> The missing piece between `selection.py` (exists) and `evaluation/` (exists):
> lesson → synthetic dataset → LoRA with replay → verdict.
> Target runtime: Gemma 3 270M IT + Gemma Scope 2 + TransformerLens.

---

## Sleep-cycle pipeline

```
[1 Harvest] → [2 Lesson extraction] → [3 Conflict check] →
[4 Variation generation + Verifier] → [5 Replay buffer] →
[6 Train LoRA] → [7 Evaluate] → [8 Promote / Rollback]
```

### 1. Harvest
Input: `PROMOTED` cluster from llm-memory's distillation (or Lab `EpisodeRecord`)
from the last cycle, with `selection_score ≥ threshold`.
Output: list of `(cluster, raw episodes)` — the raw episodes are needed at step 4,
don't discard them after lesson extraction.

### 2. Lesson extraction (second model)
For each cluster, produces a `Lesson` record:

```python
class Lesson(BaseModel):
    lesson_id: str
    statement: str                    # the lesson in declarative form
    scope_tags: list[str]             # domain of applicability
    boundary: str                     # when it does NOT apply (mandatory!)
    source_cluster_ids: list[str]
    source_episode_ids: list[str]
    status: LessonStatus              # CANDIDATE / CONSOLIDATED / SUPERSEDED / CONFLICTED
    consolidation_run_id: str | None
```

`boundary` mandatory: a lesson without explicit boundaries generates variations
that overgeneralize, which is failure mode #1 for synthetic data.

### 3. Conflict check
Compares `statement` against the already `CONSOLIDATED` Lessons:
- embedding similarity above threshold → LLM judge decides: `merge` / `supersede` / `conflict`
- `conflict` blocks the candidate and sends it for review (human, for now)

*POC: deferrable to v2 (with <10 lessons, conflicts are visible at a glance).*

### 4. Variation generation + Verifier — the core
Input: `Lesson` + raw source episodes.
Output: `SyntheticDataset` (JSONL, registered as `ArtifactRef`).

Generation (second model, N=20–50 per lesson):
- each example = a reasoning trajectory that **applies** the lesson
  in a different context (varied surface, constant deep structure)
- anchoring: at least 30% of the variations start from a paraphrased real episode
- **negative examples** (~20%): contexts within `boundary` where the lesson
  should NOT be applied, with the correct alternative behavior

Verifier (separate step, different prompt, ideally a different model):
- for each example: "does this trajectory apply the lesson or just recite it?"
- score 0–1; below threshold → discarded
- if the reject rate > 50%, the lesson is poorly formulated → back to step 2

```jsonl
{"lesson_id": "...", "kind": "positive|negative|anchor",
 "input": "...", "target": "...", "verifier_score": 0.87}
```

### 5. Replay buffer
Training batch = mix of:
- new variations (the "to be learned")
- **anchors**, at a configurable ratio (default 1:1):
  - sample from a generic open corpus (FineWeb / instruction dataset)
  - self-distillation: current model's output on protected capabilities
    (generated BEFORE training — they are the snapshot of the "old self")
  - variations of lessons already consolidated in previous cycles

### 6. Train
- LoRA on Gemma 3 270M IT (rank 8–16 to start)
- `ConsolidationCandidate` record (successor to `UpdateCandidate` for this pipeline):

```python
class ConsolidationCandidate(BaseModel):
    consolidation_candidate_id: str
    lesson_ids: list[str]
    dataset_ref: ArtifactRef          # the verified JSONL
    replay_config: dict               # ratio, anchor sources
    train_config: dict                # rank, lr, epochs, seed
    base_model_ref: str               # including the current LoRA stack
    status: CandidateStatus           # PREPARED / TRAINED / PROMOTED / ROLLED_BACK
    evaluation_run_ids: list[str]
```

### 7. Evaluate (reuses `EvaluationSpec` / `EvaluationRun`)
Three families of checks, all pre/post:
1. **Learning**: held-out set of variations never seen in training
   → is the lesson applied?
2. **Behavioral regression**: fixed benchmark on protected capabilities
   (same set every cycle, it's the historical series for forgetting)
3. **Internal drift** (v2): SAE activations on a fixed set of probe prompts,
   pre/post feature distance as a collateral-damage metric
   → registered as `TraceArtifact`

### 8. Promote / Rollback
- all three checks above threshold → `PROMOTED` (LoRA in the stack)
- regression above tolerance → `ROLLED_BACK`, lesson stays `CANDIDATE`
  with the diagnostic data attached
- stack-merge decision: deferred. Pure stack up to ~5 adapters,
  then the merge problem is addressed (open problem, v3)

---

## Hyperparameters to tune (in order of impact)
| Parameter | Initial default |
|---|---|
| replay ratio (new:anchors) | 1:1 |
| variations per lesson | 30 |
| negative examples quota | 20% |
| verifier threshold | 0.7 |
| LoRA rank | 8 |

---

## POC "rough cycle" — definition of done
A single end-to-end cycle, even if manually wired:
- [ ] 50 synthetic memories → 1 cluster → 1 Lesson
- [ ] 30 variations generated + verified
- [ ] buffer with anchors from a generic corpus (skip self-distillation)
- [ ] 1 LoRA run on Gemma 3 270M IT
- [ ] pre/post eval: held-out variations + 1 regression benchmark
- [ ] verdict recorded in `EvaluationRun`

Out of scope for the POC: conflict check, SAE drift, multi-cycle, LoRA merge.
If this cycle runs once, everything else is iteration.

---

## Implementation status in the Lab

Contract-to-code mapping (`src/intra_model_memval/`):

| Step | Where | Status |
|---|---|---|
| 1 Harvest | `selection.select_episodes` (reused) | existing |
| 2 Lesson extraction | `consolidation.interfaces.LessonExtractor` | contract (LLM-backed, outside the repo) |
| 3 Conflict check | `consolidation.interfaces.ConflictChecker` | contract (v2) |
| 4a Generation | `consolidation.interfaces.VariationGenerator` | contract (LLM-backed) |
| 4b Verifier gate | `consolidation.dataset.apply_verifier_gate` + `check_dataset_composition` | implemented (pure logic) |
| 4c Dataset JSONL | `consolidation.dataset.register_dataset` (artifact registry) | implemented |
| 5 Replay buffer | `consolidation.replay.build_replay_buffer` | implemented (deterministic mixing; anchor sourcing is the caller's responsibility) |
| 6 Train | `consolidation.training.GemmaLoraTrainer` (extra `[train]`) + `ConsolidationPlanner` | implemented (real LoRA via peft/transformers) |
| 7 Evaluate | `evaluation.EvaluationExecutor` + `HFTextGenerator`, on `EvaluationSpec` / `EvaluationRun` | implemented (containment match, pre/post) |
| 8 Promote / Rollback | `consolidation.gating.decide_promotion` + `ConsolidationPlanner.apply_decision` | implemented |

Adaptations relative to the v0.1 draft:

- `id` → `lesson_id` / `consolidation_candidate_id`, for consistency with the
  `<entity>_id` convention used across the rest of the domain. The dataset's
  JSONL format stays the one from the contract (`lesson_id`, `kind`, `input`,
  `target`, `verifier_score`).
- `ConsolidationCandidate` **sits alongside** `UpdateCandidate` instead of replacing it:
  removing `UpdateCandidate` touches the store, CLI and existing tests and is
  deferred to a dedicated migration.
- The hyperparameters from the table live in `consolidation.config.ConsolidationConfig`
  with the same defaults; the promotion thresholds live in `consolidation.gating.GateThresholds`.
- Persistence: `lessons` and `consolidation_candidates` tables in `ExperimentStore`,
  with `RunKind.CONSOLIDATION` for the run manifests.
- Consistent with the rest of the repo, no fake behavior: LLM generation and
  verification stay behind `Protocol`; LoRA training and HF generation
  are implemented but live in the optional `[train]` extra (lazy import, the
  core stays without ML dependencies).

## Running the POC (GPU machine)

```bash
pip install -e ".[train]"
python scripts/run_consolidation_poc.py \
    --workdir runs/poc-001 \
    --lesson-json examples/consolidation_poc/lesson.json \
    --variations-jsonl examples/consolidation_poc/variations.jsonl \
    --anchors-jsonl examples/consolidation_poc/anchors.jsonl \
    --regression-jsonl examples/consolidation_poc/regression.jsonl
```

The script runs a complete cycle: verifier gate, held-out split, replay
buffer with anchors, pre eval, LoRA training, post eval, promote/rollback verdict.
Everything (SQLite store, dataset, adapter, `verdict.json`) ends up under `--workdir`.
The files in `examples/consolidation_poc/` are toy fixtures for trying the
loop; for the real POC they need to be replaced with the 30 LLM-generated-and-verified
variations. Eval scoring: normalized containment match — deliberately rough, to
be replaced once the loop runs.

Known POC-trainer simplifications: full-sequence LM loss (no completion-only
masking), one adapter per run (no stacking on the previous LoRA).
