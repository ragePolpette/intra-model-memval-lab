"""CLI for experiment-oriented repository operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import GPT2SmallAdapter
from .domain import (
    EvaluationSpec,
    ExperimentRun,
    ExperimentStatus,
    RunKind,
    TraceArtifact,
    UpdateCandidate,
)
from .evaluation import (
    EvaluationHarness,
    default_gpt2_dataset_path,
    default_gpt2_prompt_template_path,
    load_synthetic_dataset,
)
from .ingestion import EpisodeIngestionService
from .persistence import ExperimentStore
from .selection import EpisodeSelectionPolicy, select_episodes
from .tracing import TraceRunner
from .utils.ids import new_id, utc_now_iso


def _load_json(value: str | None, *, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _store_from_args(args: argparse.Namespace) -> ExperimentStore:
    return ExperimentStore(db_path=args.db_path, artifact_dir=args.artifact_dir)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _load_text_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _build_gpt2_adapter(args: argparse.Namespace) -> GPT2SmallAdapter:
    return GPT2SmallAdapter(
        model_name=args.model_name,
        adapter_id=args.adapter_id,
        device=args.device,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment infrastructure CLI")
    parser.add_argument("--db-path", required=True, type=Path, help="SQLite database path")
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Artifact directory path")
    sub = parser.add_subparsers(dest="command", required=True)

    save_episode = sub.add_parser("save-episode", help="Create or update an episode record")
    save_episode.add_argument("--episode-id")
    save_episode.add_argument("--content-text", required=True)
    save_episode.add_argument("--metadata-json")
    save_episode.add_argument("--topic-tag", action="append", default=[])
    save_episode.add_argument("--trigger-tag", action="append", default=[])
    save_episode.add_argument("--source-type", default="unknown")
    save_episode.add_argument("--source-label")
    save_episode.add_argument("--source-uri")
    save_episode.add_argument("--collected-by")
    save_episode.add_argument("--is-external", action="store_true")
    save_episode.add_argument("--observed-at-utc")
    save_episode.add_argument("--conversation-id")
    save_episode.add_argument("--session-id")
    save_episode.add_argument("--notes")

    list_episodes = sub.add_parser("list-episodes", help="List or search episodes")
    list_episodes.add_argument("--query")
    list_episodes.add_argument("--topic-tag")
    list_episodes.add_argument("--source-type")
    list_episodes.add_argument("--limit", type=int, default=50)
    list_episodes.add_argument("--offset", type=int, default=0)

    register_trace = sub.add_parser("register-trace", help="Register a trace artifact contract or artifact file")
    register_trace.add_argument("--trace-id")
    register_trace.add_argument("--episode-id", required=True)
    register_trace.add_argument("--run-id")
    register_trace.add_argument("--adapter-id", required=True)
    register_trace.add_argument("--trace-type", required=True)
    register_trace.add_argument("--trace-input-json")
    register_trace.add_argument("--summary-metrics-json")
    register_trace.add_argument("--repro-json")
    register_trace.add_argument("--status", default="placeholder")
    register_trace.add_argument("--notes")
    register_trace.add_argument("--artifact-file", action="append", default=[])

    gpt2_trace = sub.add_parser("run-gpt2-trace", help="Run a real GPT-2 small trace")
    gpt2_trace.add_argument("--episode-id")
    gpt2_trace.add_argument("--content-text")
    gpt2_trace.add_argument("--metadata-json")
    gpt2_trace.add_argument("--topic-tag", action="append", default=[])
    gpt2_trace.add_argument("--trigger-tag", action="append", default=[])
    gpt2_trace.add_argument("--source-type", default="synthetic")
    gpt2_trace.add_argument("--source-label", default="gpt2-small-trace")
    gpt2_trace.add_argument("--prompt-template")
    gpt2_trace.add_argument("--prompt-template-file", type=Path)
    gpt2_trace.add_argument("--trace-type", default="gpt2-forward-pass")
    gpt2_trace.add_argument("--model-name", default="gpt2")
    gpt2_trace.add_argument("--adapter-id", default="gpt2-small")
    gpt2_trace.add_argument("--device", default="cpu")
    gpt2_trace.add_argument("--seed", type=int, default=0)
    gpt2_trace.add_argument("--include-attentions", action="store_true")
    gpt2_trace.add_argument("--notes")

    list_traces = sub.add_parser("list-traces", help="List registered trace artifacts")
    list_traces.add_argument("--episode-id")
    list_traces.add_argument("--limit", type=int, default=50)
    list_traces.add_argument("--offset", type=int, default=0)

    create_update = sub.add_parser("create-update-candidate", help="Create an update candidate")
    create_update.add_argument("--update-candidate-id")
    create_update.add_argument("--episode-id", required=True)
    create_update.add_argument("--trace-id")
    create_update.add_argument("--run-id")
    create_update.add_argument("--target-fact-json", required=True)
    create_update.add_argument("--localization-json")
    create_update.add_argument("--budget-json")
    create_update.add_argument("--hypothesis", required=True)
    create_update.add_argument("--status", default="proposed")
    create_update.add_argument("--evaluation-spec-id")
    create_update.add_argument("--result-summary-json")
    create_update.add_argument("--lineage-json")

    list_updates = sub.add_parser("list-update-candidates", help="List update candidates")
    list_updates.add_argument("--episode-id")
    list_updates.add_argument("--limit", type=int, default=50)
    list_updates.add_argument("--offset", type=int, default=0)

    create_spec = sub.add_parser("create-evaluation-spec", help="Create an evaluation spec")
    create_spec.add_argument("--evaluation-spec-id")
    create_spec.add_argument("--name", required=True)
    create_spec.add_argument("--description")
    create_spec.add_argument("--target-facts-json", required=True)
    create_spec.add_argument("--related-facts-json", default="[]")
    create_spec.add_argument("--unrelated-facts-json", default="[]")
    create_spec.add_argument("--regression-rules-json", default="[]")
    create_spec.add_argument("--metadata-json")

    list_specs = sub.add_parser("list-evaluation-specs", help="List evaluation specs")
    list_specs.add_argument("--limit", type=int, default=50)
    list_specs.add_argument("--offset", type=int, default=0)

    create_eval_run = sub.add_parser("create-evaluation-run", help="Create a planned evaluation run")
    create_eval_run.add_argument("--evaluation-run-id")
    create_eval_run.add_argument("--evaluation-spec-id", required=True)
    create_eval_run.add_argument("--subject-type", required=True)
    create_eval_run.add_argument("--subject-id", required=True)
    create_eval_run.add_argument("--run-id")

    gpt2_eval = sub.add_parser("run-gpt2-baseline-eval", help="Run baseline evaluation for GPT-2 small")
    gpt2_eval.add_argument("--dataset-path", type=Path, default=default_gpt2_dataset_path())
    gpt2_eval.add_argument("--prompt-template-file", type=Path, default=default_gpt2_prompt_template_path())
    gpt2_eval.add_argument("--evaluation-spec-id")
    gpt2_eval.add_argument("--spec-name", default="gpt2-small-capitals-baseline")
    gpt2_eval.add_argument("--model-name", default="gpt2")
    gpt2_eval.add_argument("--adapter-id", default="gpt2-small")
    gpt2_eval.add_argument("--device", default="cpu")
    gpt2_eval.add_argument("--seed", type=int, default=0)

    list_eval_runs = sub.add_parser("list-evaluation-runs", help="List evaluation runs")
    list_eval_runs.add_argument("--evaluation-spec-id")
    list_eval_runs.add_argument("--limit", type=int, default=50)
    list_eval_runs.add_argument("--offset", type=int, default=0)

    export_episodes = sub.add_parser("export-episodes", help="Export curated episodes to JSONL")
    export_episodes.add_argument("--output", required=True, type=Path)
    export_episodes.add_argument("--query")
    export_episodes.add_argument("--include-topic-tag", action="append", default=[])
    export_episodes.add_argument("--exclude-topic-tag", action="append", default=[])
    export_episodes.add_argument("--source-type", action="append", default=[])
    export_episodes.add_argument("--sample-size", type=int)
    export_episodes.add_argument("--max-per-source-label", type=int)
    export_episodes.add_argument("--max-per-session", type=int)
    export_episodes.add_argument("--seed", type=int, default=42)

    sub.add_parser("list-runs", help="List experiment runs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    store = _store_from_args(args)

    if args.command == "save-episode":
        service = EpisodeIngestionService()
        episode = service.ingest_episode(
            {
                "episode_id": args.episode_id,
                "content_text": args.content_text,
                "metadata": _load_json(args.metadata_json, default={}),
                "topic_tags": args.topic_tag,
                "trigger_tags": args.trigger_tag,
                "provenance": {
                    "source_type": args.source_type,
                    "source_label": args.source_label,
                    "source_uri": args.source_uri,
                    "collected_by": args.collected_by,
                    "is_external": args.is_external,
                },
                "observed_at_utc": args.observed_at_utc,
                "conversation_id": args.conversation_id,
                "session_id": args.session_id,
                "notes": args.notes,
            }
        )
        saved = store.save_episode(episode)
        _print_json(saved.model_dump(mode="json"))
        return 0

    if args.command == "list-episodes":
        if args.query:
            payload = store.search_episodes(args.query, limit=args.limit, offset=args.offset)
        else:
            payload = store.list_episodes(
                limit=args.limit,
                offset=args.offset,
                topic_tag=args.topic_tag,
                source_type=args.source_type,
            )
        _print_json([item.model_dump(mode="json") for item in payload])
        return 0

    if args.command == "register-trace":
        run_id = args.run_id or new_id("run")
        now = utc_now_iso()
        run = ExperimentRun(
            run_id=run_id,
            run_kind=RunKind.TRACE,
            status=ExperimentStatus.PLANNED,
            title=f"trace:{args.trace_type}",
            manifest={
                "episode_id": args.episode_id,
                "adapter_id": args.adapter_id,
                "trace_type": args.trace_type,
                "mode": "placeholder-registration",
            },
            created_at_utc=now,
            updated_at_utc=now,
        )
        store.register_run(run)
        artifact_refs = [store.register_artifact_file(Path(path), role="trace-artifact") for path in args.artifact_file]
        trace = TraceArtifact(
            trace_id=args.trace_id or new_id("trace"),
            episode_id=args.episode_id,
            run_id=run_id,
            adapter_id=args.adapter_id,
            trace_input_spec=_load_json(args.trace_input_json, default={}),
            trace_type=args.trace_type,
            artifact_refs=artifact_refs,
            summary_metrics=_load_json(args.summary_metrics_json, default={}),
            status=args.status,
            reproducibility_metadata=_load_json(args.repro_json, default={}),
            created_at_utc=now,
            updated_at_utc=now,
            notes=args.notes,
        )
        saved = store.save_trace_artifact(trace)
        _print_json(saved.model_dump(mode="json"))
        return 0

    if args.command == "run-gpt2-trace":
        if not args.episode_id and not args.content_text:
            raise SystemExit("run-gpt2-trace requires --episode-id or --content-text")
        prompt_template = args.prompt_template or _load_text_file(args.prompt_template_file) or "{content_text}"
        if args.episode_id:
            episode = store.load_episode(args.episode_id)
            if episode is None:
                raise SystemExit(f"Episode not found: {args.episode_id}")
        else:
            service = EpisodeIngestionService()
            episode = service.ingest_episode(
                {
                    "content_text": args.content_text,
                    "metadata": _load_json(args.metadata_json, default={}),
                    "topic_tags": args.topic_tag,
                    "trigger_tags": args.trigger_tag,
                    "provenance": {
                        "source_type": args.source_type,
                        "source_label": args.source_label,
                    },
                }
            )

        adapter = _build_gpt2_adapter(args)
        runner = TraceRunner(store=store, adapter=adapter)
        result = runner.trace_episode(
            episode,
            prompt_template=prompt_template,
            trace_type=args.trace_type,
            include_attentions=args.include_attentions,
            notes=args.notes,
        )
        _print_json(
            {
                "episode": result.episode.model_dump(mode="json"),
                "run": result.experiment_run.model_dump(mode="json"),
                "trace": result.trace_artifact.model_dump(mode="json"),
            }
        )
        return 0

    if args.command == "list-traces":
        traces = store.list_trace_artifacts(episode_id=args.episode_id, limit=args.limit, offset=args.offset)
        _print_json([item.model_dump(mode="json") for item in traces])
        return 0

    if args.command == "create-update-candidate":
        run_id = args.run_id or new_id("run")
        now = utc_now_iso()
        run = ExperimentRun(
            run_id=run_id,
            run_kind=RunKind.UPDATE_CANDIDATE,
            status=ExperimentStatus.PLANNED,
            title="update-candidate",
            manifest={"episode_id": args.episode_id, "trace_id": args.trace_id, "mode": "candidate-only"},
            created_at_utc=now,
            updated_at_utc=now,
        )
        store.register_run(run)
        candidate = UpdateCandidate(
            update_candidate_id=args.update_candidate_id or new_id("update-candidate"),
            episode_id=args.episode_id,
            trace_id=args.trace_id,
            run_id=run_id,
            target_fact_spec=_load_json(args.target_fact_json, default={}),
            candidate_localization=_load_json(args.localization_json, default={}),
            update_budget=_load_json(args.budget_json, default={}),
            hypothesis=args.hypothesis,
            status=args.status,
            evaluation_spec_id=args.evaluation_spec_id,
            result_summary=_load_json(args.result_summary_json, default={}),
            lineage=_load_json(args.lineage_json, default={}),
            created_at_utc=now,
            updated_at_utc=now,
        )
        saved = store.save_update_candidate(candidate)
        _print_json(saved.model_dump(mode="json"))
        return 0

    if args.command == "list-update-candidates":
        payload = store.list_update_candidates(
            episode_id=args.episode_id,
            limit=args.limit,
            offset=args.offset,
        )
        _print_json([item.model_dump(mode="json") for item in payload])
        return 0

    if args.command == "create-evaluation-spec":
        spec = EvaluationSpec(
            evaluation_spec_id=args.evaluation_spec_id or new_id("evaluation-spec"),
            name=args.name,
            description=args.description,
            target_facts=_load_json(args.target_facts_json, default=[]),
            related_facts=_load_json(args.related_facts_json, default=[]),
            unrelated_facts=_load_json(args.unrelated_facts_json, default=[]),
            regression_rules=_load_json(args.regression_rules_json, default=[]),
            metadata=_load_json(args.metadata_json, default={}),
            created_at_utc=utc_now_iso(),
        )
        saved = store.save_evaluation_spec(spec)
        _print_json(saved.model_dump(mode="json"))
        return 0

    if args.command == "list-evaluation-specs":
        payload = store.list_evaluation_specs(limit=args.limit, offset=args.offset)
        _print_json([item.model_dump(mode="json") for item in payload])
        return 0

    if args.command == "create-evaluation-run":
        spec = store.load_evaluation_spec(args.evaluation_spec_id)
        if spec is None:
            raise SystemExit(f"Evaluation spec not found: {args.evaluation_spec_id}")
        harness = EvaluationHarness()
        run, evaluation_run = harness.prepare_run(
            spec,
            subject_type=args.subject_type,
            subject_id=args.subject_id,
            run_id=args.run_id,
            evaluation_run_id=args.evaluation_run_id,
        )
        store.register_run(run)
        saved = store.save_evaluation_run(evaluation_run)
        _print_json(saved.model_dump(mode="json"))
        return 0

    if args.command == "run-gpt2-baseline-eval":
        dataset = load_synthetic_dataset(
            dataset_path=args.dataset_path,
            prompt_template_path=args.prompt_template_file,
        )
        spec = dataset.to_evaluation_spec(
            evaluation_spec_id=args.evaluation_spec_id,
            name=args.spec_name,
        )
        store.save_evaluation_spec(spec)
        adapter = _build_gpt2_adapter(args)
        harness = EvaluationHarness()
        result = harness.run_baseline(spec, adapter=adapter, subject_id=args.model_name)
        store.register_run(result.experiment_run)
        saved = store.save_evaluation_run(result.evaluation_run)
        _print_json(
            {
                "evaluation_spec": spec.model_dump(mode="json"),
                "run": result.experiment_run.model_dump(mode="json"),
                "evaluation_run": saved.model_dump(mode="json"),
            }
        )
        return 0

    if args.command == "list-evaluation-runs":
        payload = store.list_evaluation_runs(
            evaluation_spec_id=args.evaluation_spec_id,
            limit=args.limit,
            offset=args.offset,
        )
        _print_json([item.model_dump(mode="json") for item in payload])
        return 0

    if args.command == "export-episodes":
        policy = EpisodeSelectionPolicy(
            include_topic_tags=args.include_topic_tag,
            exclude_topic_tags=args.exclude_topic_tag,
            source_types=args.source_type,
            sample_size=args.sample_size,
            max_per_source_label=args.max_per_source_label,
            max_per_session=args.max_per_session,
            seed=args.seed,
        )
        result = select_episodes(store, policy, query=args.query)
        rows = [item.model_dump(mode="json") for item in result.episodes]
        store.export_jsonl(args.output, rows)
        run = ExperimentRun(
            run_id=new_id("run"),
            run_kind=RunKind.EXPORT,
            status=ExperimentStatus.COMPLETED,
            title="episode-export",
            manifest={
                "selected_count": result.selected_count,
                "total_candidates": result.total_candidates,
                "output": str(args.output),
            },
            created_at_utc=utc_now_iso(),
            updated_at_utc=utc_now_iso(),
        )
        store.register_run(run)
        _print_json(run.model_dump(mode="json"))
        return 0

    if args.command == "list-runs":
        payload = store.list_runs()
        _print_json([item.model_dump(mode="json") for item in payload])
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
