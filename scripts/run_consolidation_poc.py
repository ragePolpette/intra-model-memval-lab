"""End-to-end consolidation POC: lesson -> verified dataset -> LoRA with replay -> verdict.

Runs the "ciclo brutto" from the design contract on a GPU (or CPU) machine:

    pip install -e ".[train]"
    python scripts/run_consolidation_poc.py \
        --workdir runs/poc-001 \
        --lesson-json examples/consolidation_poc/lesson.json \
        --variations-jsonl examples/consolidation_poc/variations.jsonl \
        --anchors-jsonl examples/consolidation_poc/anchors.jsonl \
        --regression-jsonl examples/consolidation_poc/regression.jsonl

Everything (store, artifacts, adapter, verdict) lands under --workdir, ready for
scripts/export_experiment_snapshot.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intra_model_memval.consolidation import (
    ConsolidationConfig,
    ConsolidationPlanner,
    GateThresholds,
    GemmaLoraTrainer,
    HFTextGenerator,
    apply_verifier_gate,
    build_replay_buffer,
    check_dataset_composition,
    decide_promotion,
    examples_from_jsonl_bytes,
    register_dataset,
)
from intra_model_memval.consolidation.training import DEFAULT_BASE_MODEL
from intra_model_memval.domain import EvaluationCase, EvaluationSpec, Lesson
from intra_model_memval.evaluation import EvaluationExecutor
from intra_model_memval.persistence import ExperimentStore
from intra_model_memval.utils.ids import new_id, utc_now_iso


def load_lesson(path: Path) -> Lesson:
    payload = json.loads(path.read_text(encoding="utf-8"))
    now = utc_now_iso()
    payload.setdefault("lesson_id", new_id("lesson"))
    payload.setdefault("created_at_utc", now)
    payload.setdefault("updated_at_utc", now)
    return Lesson.model_validate(payload)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one consolidation cycle end-to-end")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--lesson-json", required=True, type=Path)
    parser.add_argument("--variations-jsonl", required=True, type=Path)
    parser.add_argument("--anchors-jsonl", required=True, type=Path)
    parser.add_argument("--regression-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--verifier-threshold", type=float, default=0.7)
    parser.add_argument("--learning-min", type=float, default=0.7)
    parser.add_argument("--regression-tolerance", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(db_path=workdir / "experiment.db", artifact_dir=workdir / "artifacts")
    config = ConsolidationConfig(
        replay_ratio=args.replay_ratio,
        verifier_threshold=args.verifier_threshold,
        lora_rank=args.lora_rank,
        seed=args.seed,
    )

    # 2. Lesson (extraction is upstream; here it arrives as a file)
    lesson = store.save_lesson(load_lesson(args.lesson_json))
    print(f"[lesson] {lesson.lesson_id}: {lesson.statement}")

    # 4. Verifier gate on the generated variations
    variations = examples_from_jsonl_bytes(args.variations_jsonl.read_bytes())
    gate = apply_verifier_gate(variations, config=config)
    print(
        f"[verifier] accepted {len(gate.accepted)}/{len(variations)} "
        f"(reject rate {gate.reject_rate:.2f})"
    )
    if gate.lesson_needs_reformulation:
        print("[verifier] reject rate above maximum: lesson is badly formulated, go back to extraction")
        return 1
    composition = check_dataset_composition(gate.accepted, config=config)
    for issue in composition.issues:
        print(f"[composition] warning: {issue}")

    # Held-out split (never seen in training) + dataset registration
    rng = random.Random(args.seed)
    accepted = list(gate.accepted)
    rng.shuffle(accepted)
    holdout_count = max(1, round(len(accepted) * args.holdout_fraction))
    holdout, train_examples = accepted[:holdout_count], accepted[holdout_count:]
    if not train_examples:
        print("[dataset] not enough accepted variations to train")
        return 1
    dataset_ref = register_dataset(store, train_examples)
    print(f"[dataset] {len(train_examples)} train / {len(holdout)} held-out -> {dataset_ref.path}")

    # 5. Replay buffer with generic-corpus anchors
    anchors = load_jsonl(args.anchors_jsonl)
    buffer = build_replay_buffer(train_examples, anchors, ratio=args.replay_ratio, seed=args.seed)
    print(f"[replay] {buffer.summary()}")

    # 6. Prepare candidate + run manifest
    planner = ConsolidationPlanner(config)
    run, candidate = planner.prepare_candidate(
        [lesson],
        dataset_ref,
        base_model_ref=args.base_model,
        replay_config={"ratio": args.replay_ratio, "seed": args.seed, "anchor_source": str(args.anchors_jsonl)},
        train_config={"adapter": "lora", "rank": args.lora_rank, "epochs": args.epochs, "seed": args.seed},
    )
    store.register_run(run)
    store.save_consolidation_candidate(candidate)

    # 7. Evaluation spec: held-out variations (learning) + fixed benchmark (regression)
    spec = EvaluationSpec(
        evaluation_spec_id=new_id("evaluation-spec"),
        name=f"consolidation-{lesson.lesson_id}",
        target_facts=[
            EvaluationCase(case_id=f"holdout-{i}", prompt=item.input, expected_response=item.target)
            for i, item in enumerate(holdout)
        ],
        related_facts=[
            EvaluationCase(case_id=f"regression-{i}", prompt=row["input"], expected_response=row["target"])
            for i, row in enumerate(load_jsonl(args.regression_jsonl))
        ],
        regression_rules=["protected capabilities must not drop beyond tolerance"],
        created_at_utc=utc_now_iso(),
    )
    store.save_evaluation_spec(spec)
    executor = EvaluationExecutor()

    print(f"[eval] pre-training evaluation on {args.base_model}")
    base_generator = HFTextGenerator(args.base_model, device=args.device)
    pre_run, pre_eval = executor.execute(
        spec, base_generator, subject_type="base_model", subject_id=args.base_model
    )
    store.register_run(pre_run)
    store.save_evaluation_run(pre_eval)
    print(f"[eval] pre metrics: {pre_eval.metrics}")
    del base_generator

    # Train
    print("[train] starting LoRA training")
    trainer = GemmaLoraTrainer(
        store, base_model_id=args.base_model, output_dir=workdir / "adapters", device=args.device
    )
    adapter_ref = trainer.train(candidate, buffer.items)
    candidate = planner.mark_trained(
        candidate, adapter_ref=adapter_ref, training_summary={"replay": buffer.summary()}
    )
    store.save_consolidation_candidate(candidate)
    print(f"[train] adapter saved: {adapter_ref.metadata['adapter_dir']} (loss {adapter_ref.metadata['train_loss']:.4f})")

    # Post-training evaluation
    post_generator = HFTextGenerator(
        args.base_model, adapter_dir=adapter_ref.metadata["adapter_dir"], device=args.device
    )
    post_run, post_eval = executor.execute(
        spec,
        post_generator,
        subject_type="consolidation_candidate",
        subject_id=candidate.consolidation_candidate_id,
    )
    store.register_run(post_run)
    store.save_evaluation_run(post_eval)
    print(f"[eval] post metrics: {post_eval.metrics}")

    # 8. Promote / Rollback
    decision = decide_promotion(
        learning_score=float(post_eval.metrics["target_match_rate"]),
        regression_pre=float(pre_eval.metrics["related_match_rate"]),
        regression_post=float(post_eval.metrics["related_match_rate"]),
        thresholds=GateThresholds(
            learning_min=args.learning_min, regression_tolerance=args.regression_tolerance
        ),
    )
    candidate, lessons = planner.apply_decision(
        candidate,
        [lesson],
        decision,
        evaluation_run_ids=[pre_eval.evaluation_run_id, post_eval.evaluation_run_id],
    )
    store.save_consolidation_candidate(candidate)
    for item in lessons:
        store.save_lesson(item)

    verdict = {
        "candidate_id": candidate.consolidation_candidate_id,
        "status": candidate.status.value,
        "checks": decision.checks,
        "reasons": decision.reasons,
        "pre_metrics": pre_eval.metrics,
        "post_metrics": post_eval.metrics,
    }
    (workdir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"[verdict] {candidate.status.value.upper()} -> {workdir / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
