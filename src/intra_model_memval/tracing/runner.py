"""Minimal but real trace runner for model adapter forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapters import BaseModelAdapter
from ..domain import EpisodeRecord, ExperimentRun, ExperimentStatus, RunKind, TraceArtifact, TraceStatus
from ..ingestion import EpisodeIngestionService
from ..persistence import ExperimentStore
from ..utils.ids import new_id, utc_now_iso
from .serializers import build_top_token_summary, serialize_trace_summary, serialize_trace_tensors


@dataclass(slots=True)
class TraceExecutionResult:
    episode: EpisodeRecord
    experiment_run: ExperimentRun
    trace_artifact: TraceArtifact


class TraceRunner:
    """Materialize forward traces and persist auditable artifacts."""

    def __init__(self, *, store: ExperimentStore, adapter: BaseModelAdapter) -> None:
        self.store = store
        self.adapter = adapter
        self.ingestion_service = EpisodeIngestionService()

    def trace_episode(
        self,
        episode: EpisodeRecord | dict[str, Any],
        *,
        prompt_template: str = "{content_text}",
        trace_type: str = "forward-pass",
        include_attentions: bool = False,
        run_id: str | None = None,
        trace_id: str | None = None,
        notes: str | None = None,
    ) -> TraceExecutionResult:
        saved_episode = self.store.save_episode(self.ingestion_service.ingest_episode(episode))
        prompt_text = prompt_template.format(
            content_text=saved_episode.content_text,
            episode_id=saved_episode.episode_id,
        )
        forward_result = self.adapter.forward(
            prompt_text,
            output_hidden_states=True,
            output_attentions=include_attentions,
        )
        adapter_spec = self.adapter.describe()
        now = utc_now_iso()
        actual_run_id = run_id or new_id("run")
        actual_trace_id = trace_id or new_id("trace")

        run = ExperimentRun(
            run_id=actual_run_id,
            run_kind=RunKind.TRACE,
            status=ExperimentStatus.COMPLETED,
            title=f"trace:{trace_type}:{self.adapter.adapter_id}",
            manifest={
                "episode_id": saved_episode.episode_id,
                "adapter_id": self.adapter.adapter_id,
                "model_id": self.adapter.model_id,
                "trace_type": trace_type,
                "prompt_template": prompt_template,
                "token_count": len(forward_result.token_ids),
                "artifact_roles": ["trace-summary", "trace-tensors"],
            },
            created_at_utc=now,
            updated_at_utc=now,
        )
        self.store.register_run(run)

        summary_ref = self.store.register_artifact_bytes(
            serialize_trace_summary(
                self.adapter,
                forward_result,
                input_text=saved_episode.content_text,
                prompt_template=prompt_template,
            ),
            media_type="application/json",
            suffix=".json",
        ).model_copy(update={"role": "trace-summary"})

        tensor_bytes, tensor_suffix, tensor_media_type = serialize_trace_tensors(forward_result)
        tensor_ref = self.store.register_artifact_bytes(
            tensor_bytes,
            media_type=tensor_media_type,
            suffix=tensor_suffix,
        ).model_copy(update={"role": "trace-tensors"})

        trace = TraceArtifact(
            trace_id=actual_trace_id,
            episode_id=saved_episode.episode_id,
            run_id=actual_run_id,
            adapter_id=self.adapter.adapter_id,
            trace_input_spec={
                "input_text": saved_episode.content_text,
                "prompt_text": prompt_text,
                "prompt_template": prompt_template,
                "token_ids": forward_result.token_ids,
            },
            trace_type=trace_type,
            artifact_refs=[summary_ref, tensor_ref],
            summary_metrics={
                "token_count": len(forward_result.token_ids),
                "hidden_state_count": len(forward_result.hidden_states),
                "logits_shape": list(getattr(forward_result.logits, "shape", [])),
                "top_next_tokens": build_top_token_summary(self.adapter, forward_result, top_k=5),
            },
            status=TraceStatus.MATERIALIZED,
            reproducibility_metadata={
                "adapter": adapter_spec.model_dump(mode="json"),
                "seed": adapter_spec.metadata.get("seed"),
                "episode_context_hash": saved_episode.context_hash,
                "include_attentions": include_attentions,
            },
            created_at_utc=now,
            updated_at_utc=now,
            notes=notes,
        )
        saved_trace = self.store.save_trace_artifact(trace)
        return TraceExecutionResult(
            episode=saved_episode,
            experiment_run=run,
            trace_artifact=saved_trace,
        )
