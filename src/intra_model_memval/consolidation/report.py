"""Pre/post comparison report for a consolidation run, per evaluation family and per case."""

from __future__ import annotations

from typing import Any

from ..domain import EvaluationRun, ExperimentRun, RunKind
from ..persistence import ExperimentStore

FAMILY_LABELS = {
    "target": "learning (held-out variations)",
    "related": "regression (protected capabilities)",
    "unrelated": "unrelated",
}


class ReportError(ValueError):
    pass


def _resolve_evaluation_runs(
    store: ExperimentStore, run: ExperimentRun, evaluation_run_ids: list[str]
) -> tuple[EvaluationRun | None, EvaluationRun | None]:
    pre_id = run.manifest.get("pre_evaluation_run_id")
    post_id = run.manifest.get("post_evaluation_run_id")
    pre = store.load_evaluation_run(pre_id) if pre_id else None
    post = store.load_evaluation_run(post_id) if post_id else None
    for evaluation_run_id in evaluation_run_ids:
        if pre is not None and post is not None:
            break
        loaded = store.load_evaluation_run(evaluation_run_id)
        if loaded is None:
            continue
        if pre is None and loaded.subject_type == "base_model":
            pre = loaded
        elif post is None and loaded.subject_type == "consolidation_candidate":
            post = loaded
    return pre, post


def _rate(evaluation_run: EvaluationRun | None, family: str) -> float | None:
    if evaluation_run is None:
        return None
    value = evaluation_run.metrics.get(f"{family}_match_rate")
    return None if value is None else float(value)


def _delta(pre: float | None, post: float | None) -> float | None:
    return None if pre is None or post is None else post - pre


def build_consolidation_report(store: ExperimentStore, run_id: str) -> dict[str, Any]:
    run = store.load_run(run_id)
    if run is None:
        raise ReportError(f"run not found: {run_id}")
    if run.run_kind != RunKind.CONSOLIDATION:
        raise ReportError(f"run {run_id} is a {run.run_kind.value} run, expected consolidation")

    candidates = store.list_consolidation_candidates(run_id=run_id, limit=1)
    candidate = candidates[0] if candidates else None
    pre, post = _resolve_evaluation_runs(
        store, run, list(candidate.evaluation_run_ids) if candidate else []
    )

    families: list[dict[str, Any]] = []
    for family, label in FAMILY_LABELS.items():
        counts = [
            int(item.metrics.get(f"{family}_case_count", 0)) for item in (pre, post) if item is not None
        ]
        if not any(counts):
            continue
        pre_rate, post_rate = _rate(pre, family), _rate(post, family)
        families.append(
            {
                "family": family,
                "label": label,
                "case_count": max(counts),
                "pre": pre_rate,
                "post": post_rate,
                "delta": _delta(pre_rate, post_rate),
            }
        )

    cases: dict[tuple[str, str], dict[str, Any]] = {}
    for side, evaluation_run in (("pre", pre), ("post", post)):
        if evaluation_run is None:
            continue
        for observation in evaluation_run.observations:
            key = (str(observation.get("family")), str(observation.get("case_id")))
            entry = cases.setdefault(
                key,
                {
                    "family": key[0],
                    "case_id": key[1],
                    "prompt": observation.get("prompt"),
                    "scorer": observation.get("scorer", "containment"),
                    "pre": None,
                    "post": None,
                },
            )
            entry[side] = observation.get("score")
            entry[f"{side}_generated"] = observation.get("generated")
    for entry in cases.values():
        entry["delta"] = _delta(entry["pre"], entry["post"])

    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "mode": run.manifest.get("mode"),
        "consolidation_candidate_id": candidate.consolidation_candidate_id if candidate else None,
        "candidate_status": candidate.status.value if candidate else None,
        "decision": candidate.result_summary.get("decision") if candidate else None,
        "pre_evaluation_run_id": pre.evaluation_run_id if pre else None,
        "post_evaluation_run_id": post.evaluation_run_id if post else None,
        "families": families,
        "cases": list(cases.values()),
    }


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def format_report_text(report: dict[str, Any]) -> str:
    lines = [
        f"run {report['run_id']}  status={report['status']}  mode={report['mode']}",
        f"candidate {report['consolidation_candidate_id']}  status={report['candidate_status']}",
    ]
    decision = report.get("decision")
    if decision:
        lines.append(f"decision {decision.get('status')}  checks={decision.get('checks')}")
        lines.extend(f"  reason: {reason}" for reason in decision.get("reasons", []))
    lines.append("")
    lines.append("per family:        cases    pre     post    delta")
    for item in report["families"]:
        lines.append(
            f"  {item['family']:<14} {item['case_count']:>5}  {_fmt(item['pre'])}  "
            f"{_fmt(item['post'])}  {_fmt(item['delta'], signed=True)}   {item['label']}"
        )
    lines.append("")
    lines.append("per case:")
    for item in report["cases"]:
        lines.append(
            f"  {item['family']:<9} {item['case_id']:<16} pre={_fmt(item['pre'])} "
            f"post={_fmt(item['post'])} delta={_fmt(item['delta'], signed=True)} [{item['scorer']}]"
        )
    return "\n".join(lines)
