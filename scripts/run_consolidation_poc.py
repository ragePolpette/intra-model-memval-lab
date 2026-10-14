"""End-to-end consolidation POC: lesson -> verified dataset -> LoRA with replay -> verdict.

Runs the "ciclo brutto" from the design contract on a GPU (or CPU) machine:

    pip install -e ".[train]"
    python scripts/run_consolidation_poc.py \
        --workdir runs/poc-001 \
        --lesson-json examples/consolidation_poc/lesson.json \
        --variations-jsonl examples/consolidation_poc/variations.jsonl \
        --anchors-jsonl examples/consolidation_poc/anchors.jsonl \
        --regression-jsonl examples/consolidation_poc/regression.jsonl

Everything (store, artifacts, adapter, verdict.json, experiment_manifest.json) lands
under --workdir, ready for scripts/export_experiment_snapshot.py; inspect the outcome with
`intra-model-exp --db-path <workdir>/experiment.db --artifact-dir <workdir>/artifacts
report --run-id <run_id>`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intra_model_memval.consolidation import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    TRAINING_MODE,
    ConsolidationConfig,
    ConsolidationPlanner,
    GateThresholds,
    GemmaLoraTrainer,
    HFTextGenerator,
    LoraTrainerAdapter,
    apply_verifier_gate,
    build_replay_buffer,
    check_dataset_composition,
    decide_promotion,
    examples_from_jsonl_bytes,
    register_dataset,
    release_accelerator_memory,
    resolve_train_config,
    split_holdout,
)
from intra_model_memval.consolidation.manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    build_experiment_manifest,
    describe_base_model,
    manifest_to_json_bytes,
)
from intra_model_memval.domain import (  # noqa: E402
    EvaluationCase,
    EvaluationSpec,
    ExperimentStatus,
    Lesson,
)
from intra_model_memval.evaluation import EvaluationExecutor, TextGenerator  # noqa: E402
from intra_model_memval.persistence import ExperimentStore  # noqa: E402
from intra_model_memval.utils.ids import new_id, utc_now_iso  # noqa: E402

GeneratorFactory = Callable[..., TextGenerator]
TrainerFactory = Callable[..., LoraTrainerAdapter]


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
    parser.add_argument("--anchors-jsonl", required=True, type=Path, help="replay anchors (generic corpus)")
    parser.add_argument("--regression-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-model-revision", default=None, help="hub revision (branch, tag or commit) to pin")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--verifier-threshold", type=float, default=0.7)
    parser.add_argument("--learning-min", type=float, default=0.7)
    parser.add_argument("--regression-tolerance", type=float, default=0.05, help="max drop in match rate")
    parser.add_argument(
        "--regression-tolerance-cases",
        type=int,
        default=None,
        help="max number of lost regression cases; overrides --regression-tolerance",
    )
    parser.add_argument(
        "--min-regression-cases",
        type=int,
        default=10,
        help="roll back when the regression benchmark has fewer cases than this",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: GeneratorFactory = HFTextGenerator,
    trainer_factory: TrainerFactory = GemmaLoraTrainer,
) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        print("[config] --max-new-tokens must be positive")
        return 2
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(db_path=workdir / "experiment.db", artifact_dir=workdir / "artifacts")
    config = ConsolidationConfig(
        replay_ratio=args.replay_ratio,
        verifier_threshold=args.verifier_threshold,
        lora_rank=args.lora_rank,
        seed=args.seed,
    )
    thresholds = GateThresholds(
        learning_min=args.learning_min,
        regression_tolerance=args.regression_tolerance,
        regression_tolerance_cases=args.regression_tolerance_cases,
        min_regression_cases=args.min_regression_cases,
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
    print(
        f"[composition] variations by kind: positive={composition.positive_count} "
        f"negative={composition.negative_count} kind=anchor={composition.anchor_count}"
    )
    for issue in composition.issues:
        print(f"[composition] warning: {issue}")

    # Held-out split (never seen in training), stratified by kind; kind=anchor stays in training
    split = split_holdout(gate.accepted, fraction=args.holdout_fraction, seed=args.seed)
    if not split.train:
        print("[dataset] not enough accepted variations to train")
        return 1
    if not split.holdout:
        print("[dataset] not enough accepted variations for a held-out set")
        return 1
    dataset_ref = register_dataset(store, split.train)
    print(
        f"[dataset] {len(split.train)} train / {len(split.holdout)} held-out "
        f"{split.counts_by_kind()} -> {dataset_ref.path}"
    )

    # 5. Replay buffer: new variations (all kinds) + replay anchors from a generic corpus
    replay_anchors = load_jsonl(args.anchors_jsonl)
    buffer = build_replay_buffer(split.train, replay_anchors, ratio=args.replay_ratio, seed=args.seed)
    print(f"[replay] {buffer.summary()}")

    # 6. Prepare candidate + run manifest
    planner = ConsolidationPlanner(config)
    train_config = {"adapter": "lora", "rank": args.lora_rank, "epochs": args.epochs, "seed": args.seed}
    run, candidate = planner.prepare_candidate(
        [lesson],
        dataset_ref,
        base_model_ref=args.base_model,
        replay_config={
            "ratio": args.replay_ratio,
            "seed": args.seed,
            "replay_anchor_source": str(args.anchors_jsonl),
        },
        train_config=train_config,
    )
    store.register_run(run)
    store.save_consolidation_candidate(candidate)

    # 7. Evaluation spec: held-out variations (learning) + fixed benchmark (regression)
    regression_rows = load_jsonl(args.regression_jsonl)
    spec = EvaluationSpec(
        evaluation_spec_id=new_id("evaluation-spec"),
        name=f"consolidation-{lesson.lesson_id}",
        target_facts=[
            EvaluationCase(
                case_id=f"holdout-{i}",
                prompt=item.input,
                expected_response=item.target,
                rubric=item.rubric,
                metadata={"kind": item.kind.value},
            )
            for i, item in enumerate(split.holdout)
        ],
        related_facts=[
            EvaluationCase(
                case_id=f"regression-{i}",
                prompt=row["input"],
                expected_response=row["target"],
                rubric=row.get("rubric"),
            )
            for i, row in enumerate(regression_rows)
        ],
        regression_rules=["protected capabilities must not drop beyond tolerance"],
        created_at_utc=utc_now_iso(),
    )
    store.save_evaluation_spec(spec)
    if len(regression_rows) < args.min_regression_cases:
        print(
            f"[eval] warning: {len(regression_rows)} regression cases, below "
            f"--min-regression-cases {args.min_regression_cases}: the gate will roll back"
        )
    executor = EvaluationExecutor()

    run = store.register_run(planner.update_run(run, status=ExperimentStatus.RUNNING, mode=TRAINING_MODE))
    state: dict[str, Any] = {"pre_eval": None, "post_eval": None, "adapter_ref": None, "decision": None}
    devices: dict[str, Any] = {"requested": args.device}

    def write_manifest(status: str, error: str | None = None):
        adapter_ref = state["adapter_ref"]
        adapter_meta = adapter_ref.metadata if adapter_ref is not None else {}
        manifest = build_experiment_manifest(
            run_id=run.run_id,
            status=status,
            repo_root=ROOT,
            base_model=describe_base_model(
                args.base_model,
                requested_revision=args.base_model_revision,
                resolved_revision=adapter_meta.get("base_model_revision"),
            ),
            seed=args.seed,
            devices={**devices, "training": adapter_meta.get("device")},
            config={
                "cli_args": _jsonable_args(args),
                "consolidation": config.model_dump(mode="json"),
                "gate_thresholds": thresholds.model_dump(mode="json"),
                "train": adapter_meta.get("train_config") or resolve_train_config(train_config),
                "replay": candidate.replay_config,
                "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False},
            },
            data={
                "lesson_id": lesson.lesson_id,
                "dataset_artifact_id": dataset_ref.artifact_id,
                "dataset_artifact_hash": dataset_ref.artifact_hash,
                "split_by_kind": split.counts_by_kind(),
                "holdout_case_ids": [case.case_id for case in spec.target_facts],
                "regression_case_count": len(regression_rows),
                "replay_buffer": buffer.summary(),
                "evaluation_spec_id": spec.evaluation_spec_id,
            },
            metrics={
                "pre": state["pre_eval"].metrics if state["pre_eval"] else None,
                "post": state["post_eval"].metrics if state["post_eval"] else None,
                "pre_evaluation_run_id": state["pre_eval"].evaluation_run_id if state["pre_eval"] else None,
                "post_evaluation_run_id": state["post_eval"].evaluation_run_id if state["post_eval"] else None,
                "train_loss": adapter_meta.get("train_loss"),
            },
            adapter=None
            if adapter_ref is None
            else {
                "adapter_dir": adapter_meta.get("adapter_dir"),
                "artifact_id": adapter_ref.artifact_id,
                "artifact_hash": adapter_ref.artifact_hash,
                "artifact_path": adapter_ref.path,
            },
            decision=state["decision"],
            error=error,
        )
        data = manifest_to_json_bytes(manifest)
        manifest_path = workdir / MANIFEST_FILENAME
        manifest_path.write_bytes(data)
        ref = store.register_artifact_bytes(data, media_type="application/json", suffix=".json")
        return manifest_path, ref

    try:
        print(f"[eval] pre-training evaluation on {args.base_model}")
        base_generator = generator_factory(
            args.base_model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            revision=args.base_model_revision,
        )
        devices["evaluation"] = getattr(base_generator, "device", None)
        pre_run, pre_eval = executor.execute(
            spec, base_generator, subject_type="base_model", subject_id=args.base_model
        )
        store.register_run(pre_run)
        store.save_evaluation_run(pre_eval)
        state["pre_eval"] = pre_eval
        print(f"[eval] pre metrics: {pre_eval.metrics}")
        del base_generator
        release_accelerator_memory()

        print("[train] starting LoRA training")
        trainer = trainer_factory(
            store,
            base_model_id=args.base_model,
            output_dir=workdir / "adapters",
            device=args.device,
            revision=args.base_model_revision,
        )
        adapter_ref = trainer.train(candidate, buffer.items)
        state["adapter_ref"] = adapter_ref
        candidate = planner.mark_trained(
            candidate, adapter_ref=adapter_ref, training_summary={"replay": buffer.summary()}
        )
        store.save_consolidation_candidate(candidate)
        print(
            f"[train] adapter saved: {adapter_ref.metadata['adapter_dir']} "
            f"(loss {adapter_ref.metadata['train_loss']:.4f})"
        )
        release_accelerator_memory()

        post_generator = generator_factory(
            args.base_model,
            adapter_dir=adapter_ref.metadata["adapter_dir"],
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            revision=args.base_model_revision,
        )
        post_run, post_eval = executor.execute(
            spec,
            post_generator,
            subject_type="consolidation_candidate",
            subject_id=candidate.consolidation_candidate_id,
        )
        store.register_run(post_run)
        store.save_evaluation_run(post_eval)
        state["post_eval"] = post_eval
        print(f"[eval] post metrics: {post_eval.metrics}")
        del post_generator
        release_accelerator_memory()

        # 8. Promote / Rollback
        decision = decide_promotion(
            learning_score=float(post_eval.metrics["target_match_rate"]),
            regression_pre=float(pre_eval.metrics["related_match_rate"]),
            regression_post=float(post_eval.metrics["related_match_rate"]),
            regression_case_count=int(post_eval.metrics["related_case_count"]),
            thresholds=thresholds,
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
        state["decision"] = candidate.result_summary["decision"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        manifest_path, manifest_ref = write_manifest(ExperimentStatus.FAILED.value, error)
        store.register_run(
            planner.update_run(
                run,
                status=ExperimentStatus.FAILED,
                manifest_updates={
                    "error": error,
                    "experiment_manifest_artifact_id": manifest_ref.artifact_id,
                },
            )
        )
        print(f"[run] FAILED: {error} -> {manifest_path}")
        raise

    manifest_path, manifest_ref = write_manifest(ExperimentStatus.COMPLETED.value)
    store.register_run(
        planner.update_run(
            run,
            status=ExperimentStatus.COMPLETED,
            manifest_updates={
                "adapter_artifact_id": state["adapter_ref"].artifact_id,
                "adapter_dir": state["adapter_ref"].metadata.get("adapter_dir"),
                "pre_evaluation_run_id": pre_eval.evaluation_run_id,
                "post_evaluation_run_id": post_eval.evaluation_run_id,
                "decision_status": candidate.status.value,
                "experiment_manifest_artifact_id": manifest_ref.artifact_id,
            },
        )
    )

    verdict = {
        "run_id": run.run_id,
        "candidate_id": candidate.consolidation_candidate_id,
        "status": candidate.status.value,
        "checks": decision.checks,
        "reasons": decision.reasons,
        "pre_metrics": pre_eval.metrics,
        "post_metrics": post_eval.metrics,
    }
    (workdir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"[verdict] {candidate.status.value.upper()} -> {workdir / 'verdict.json'}")
    print(f"[manifest] {manifest_path}")
    print(
        f"[report] intra-model-exp --db-path {store.db_path} --artifact-dir {store.artifact_dir} "
        f"report --run-id {run.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
