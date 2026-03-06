"""Dataset selection policy engine for memval training export."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schemas import MemoryRecord
from .storage import MemoryPersistence


@dataclass
class SelectionPolicy:
    novelty_min: float = 0.2
    top_ratio: float = 0.60
    mid_ratio: float = 0.25
    low_ratio: float = 0.15
    external_min_ratio: float = 0.25
    max_per_writer: float = 0.20
    max_per_conversation: float = 0.05
    sample_size: int | None = None
    seed: int = 42


@dataclass
class SelectionStats:
    total_candidates: int
    after_novelty_filter: int
    selected: int
    external_ratio: float
    required_external: int
    external_target_met: bool
    top_count: int
    mid_count: int
    low_count: int


@dataclass
class SelectionResult:
    records: list[MemoryRecord]
    stats: SelectionStats


def _conversation_id(record: MemoryRecord) -> str:
    direct_value = record.metadata.get("conversation_id")
    if direct_value is not None:
        normalized = str(direct_value).strip()
        if normalized:
            return normalized

    context_fingerprint = record.metadata.get("context_fingerprint")
    if not isinstance(context_fingerprint, dict):
        return ""
    return str(context_fingerprint.get("conversation_id") or "")


def _bucket(record: MemoryRecord) -> str:
    score = int(record.importance_score)
    if score >= 70:
        return "top"
    if score >= 40:
        return "mid"
    return "low"


def _count_buckets(records: list[MemoryRecord]) -> tuple[int, int, int]:
    top = sum(1 for item in records if _bucket(item) == "top")
    mid = sum(1 for item in records if _bucket(item) == "mid")
    low = sum(1 for item in records if _bucket(item) == "low")
    return top, mid, low


def _sort_key(record: MemoryRecord) -> tuple[int, str, str]:
    return (-int(record.importance_score), str(record.created_at_utc), str(record.entry_id))


def _writer(record: MemoryRecord) -> str:
    writer = str(record.writer_model).strip()
    return writer or "unknown-model"


def _can_add(
    candidate: MemoryRecord,
    *,
    selected_ids: set[str],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
    max_writer_count: int,
    max_conversation_count: int,
) -> bool:
    if candidate.entry_id in selected_ids:
        return False

    writer = _writer(candidate)
    if writer_counts.get(writer, 0) >= max_writer_count:
        return False

    conversation_id = _conversation_id(candidate)
    if conversation_id and conversation_counts.get(conversation_id, 0) >= max_conversation_count:
        return False

    return True


def _add_candidate(
    candidate: MemoryRecord,
    *,
    selected: list[MemoryRecord],
    selected_ids: set[str],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
) -> None:
    selected.append(candidate)
    selected_ids.add(candidate.entry_id)
    writer = _writer(candidate)
    writer_counts[writer] = writer_counts.get(writer, 0) + 1
    conversation_id = _conversation_id(candidate)
    if conversation_id:
        conversation_counts[conversation_id] = conversation_counts.get(conversation_id, 0) + 1


def _remove_candidate(
    candidate: MemoryRecord,
    *,
    selected: list[MemoryRecord],
    selected_ids: set[str],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
) -> None:
    selected.remove(candidate)
    selected_ids.discard(candidate.entry_id)

    writer = _writer(candidate)
    writer_after = writer_counts.get(writer, 0) - 1
    if writer_after > 0:
        writer_counts[writer] = writer_after
    else:
        writer_counts.pop(writer, None)

    conversation_id = _conversation_id(candidate)
    if conversation_id:
        conversation_after = conversation_counts.get(conversation_id, 0) - 1
        if conversation_after > 0:
            conversation_counts[conversation_id] = conversation_after
        else:
            conversation_counts.pop(conversation_id, None)


def _target_counts(
    total_target: int,
    *,
    top_ratio: float,
    mid_ratio: float,
    low_ratio: float,
) -> tuple[int, int, int]:
    raw = {
        "top": float(total_target) * top_ratio,
        "mid": float(total_target) * mid_ratio,
        "low": float(total_target) * low_ratio,
    }
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    assigned = sum(counts.values())
    remaining = max(0, total_target - assigned)
    order = sorted(
        raw.keys(),
        key=lambda key: (raw[key] - counts[key], {"top": 0, "mid": 1, "low": 2}[key]),
        reverse=True,
    )
    for key in order[:remaining]:
        counts[key] += 1
    return counts["top"], counts["mid"], counts["low"]


def _pick_with_caps(
    candidates: list[MemoryRecord],
    needed: int,
    *,
    selected: list[MemoryRecord],
    selected_ids: set[str],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
    max_writer_count: int,
    max_conversation_count: int,
    randomize: bool,
    rng: random.Random,
) -> int:
    if needed <= 0:
        return 0

    ordered = list(candidates)
    if randomize:
        rng.shuffle(ordered)
    else:
        ordered.sort(key=_sort_key)

    for candidate in ordered:
        if needed <= 0:
            break
        if _can_add(
            candidate,
            selected_ids=selected_ids,
            writer_counts=writer_counts,
            conversation_counts=conversation_counts,
            max_writer_count=max_writer_count,
            max_conversation_count=max_conversation_count,
        ):
            _add_candidate(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                writer_counts=writer_counts,
                conversation_counts=conversation_counts,
            )
            needed -= 1
    return needed


def _fetch_all_records(persistence: MemoryPersistence, page_size: int = 500) -> list[MemoryRecord]:
    all_records: list[MemoryRecord] = []
    offset = 0
    while True:
        page = persistence.list_records(limit=page_size, offset=offset)
        if not page:
            break
        all_records.extend(page)
        offset += len(page)
    return all_records


def _count_external(records: list[MemoryRecord]) -> int:
    return sum(1 for item in records if item.is_external)


def _can_swap_incoming(
    incoming: MemoryRecord,
    outgoing: MemoryRecord,
    *,
    selected_ids: set[str],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
    max_writer_count: int,
    max_conversation_count: int,
) -> bool:
    if incoming.entry_id in selected_ids and incoming.entry_id != outgoing.entry_id:
        return False

    writer_next = dict(writer_counts)
    conv_next = dict(conversation_counts)

    outgoing_writer = _writer(outgoing)
    writer_next[outgoing_writer] = writer_next.get(outgoing_writer, 0) - 1
    if writer_next[outgoing_writer] <= 0:
        writer_next.pop(outgoing_writer, None)

    outgoing_conversation = _conversation_id(outgoing)
    if outgoing_conversation:
        conv_next[outgoing_conversation] = conv_next.get(outgoing_conversation, 0) - 1
        if conv_next[outgoing_conversation] <= 0:
            conv_next.pop(outgoing_conversation, None)

    incoming_writer = _writer(incoming)
    if writer_next.get(incoming_writer, 0) >= max_writer_count:
        return False

    incoming_conversation = _conversation_id(incoming)
    if incoming_conversation and conv_next.get(incoming_conversation, 0) >= max_conversation_count:
        return False

    return True


def _enforce_external_quota(
    *,
    selected: list[MemoryRecord],
    selected_ids: set[str],
    filtered: list[MemoryRecord],
    writer_counts: dict[str, int],
    conversation_counts: dict[str, int],
    required_external: int,
    max_writer_count: int,
    max_conversation_count: int,
) -> None:
    current_external = _count_external(selected)
    if current_external >= required_external:
        return

    while current_external < required_external:
        remaining_external = [
            item
            for item in filtered
            if item.is_external and item.entry_id not in selected_ids
        ]
        if not remaining_external:
            return

        remaining_external.sort(key=_sort_key)
        removable_internal = [item for item in selected if not item.is_external]
        removable_internal.sort(
            key=lambda item: (
                {"low": 0, "mid": 1, "top": 2}[_bucket(item)],
                int(item.importance_score),
                str(item.created_at_utc),
                str(item.entry_id),
            )
        )

        swapped = False
        for incoming in remaining_external:
            for outgoing in removable_internal:
                if not _can_swap_incoming(
                    incoming,
                    outgoing,
                    selected_ids=selected_ids,
                    writer_counts=writer_counts,
                    conversation_counts=conversation_counts,
                    max_writer_count=max_writer_count,
                    max_conversation_count=max_conversation_count,
                ):
                    continue

                _remove_candidate(
                    outgoing,
                    selected=selected,
                    selected_ids=selected_ids,
                    writer_counts=writer_counts,
                    conversation_counts=conversation_counts,
                )
                _add_candidate(
                    incoming,
                    selected=selected,
                    selected_ids=selected_ids,
                    writer_counts=writer_counts,
                    conversation_counts=conversation_counts,
                )
                current_external += 1
                swapped = True
                break

            if swapped:
                break

        if not swapped:
            return


def select_training_records(
    persistence: MemoryPersistence,
    policy: SelectionPolicy,
) -> SelectionResult:
    if abs((policy.top_ratio + policy.mid_ratio + policy.low_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios top/mid/low must sum to 1.0")

    rng = random.Random(policy.seed)
    all_records = _fetch_all_records(persistence)
    filtered = [record for record in all_records if float(record.novelty_score) >= policy.novelty_min]
    if not filtered:
        return SelectionResult(
            records=[],
            stats=SelectionStats(
                total_candidates=len(all_records),
                after_novelty_filter=0,
                selected=0,
                external_ratio=0.0,
                required_external=0,
                external_target_met=True,
                top_count=0,
                mid_count=0,
                low_count=0,
            ),
        )

    total_target = policy.sample_size or len(filtered)
    total_target = max(1, min(total_target, len(filtered)))
    max_writer_count = max(1, math.ceil(total_target * policy.max_per_writer))
    max_conversation_count = max(1, math.ceil(total_target * policy.max_per_conversation))

    top_candidates = [record for record in filtered if _bucket(record) == "top"]
    mid_candidates = [record for record in filtered if _bucket(record) == "mid"]
    low_candidates = [record for record in filtered if _bucket(record) == "low"]

    top_target, mid_target, low_target = _target_counts(
        total_target,
        top_ratio=policy.top_ratio,
        mid_ratio=policy.mid_ratio,
        low_ratio=policy.low_ratio,
    )

    selected: list[MemoryRecord] = []
    selected_ids: set[str] = set()
    writer_counts: dict[str, int] = {}
    conversation_counts: dict[str, int] = {}

    _pick_with_caps(
        top_candidates,
        top_target,
        selected=selected,
        selected_ids=selected_ids,
        writer_counts=writer_counts,
        conversation_counts=conversation_counts,
        max_writer_count=max_writer_count,
        max_conversation_count=max_conversation_count,
        randomize=False,
        rng=rng,
    )

    _pick_with_caps(
        mid_candidates,
        mid_target,
        selected=selected,
        selected_ids=selected_ids,
        writer_counts=writer_counts,
        conversation_counts=conversation_counts,
        max_writer_count=max_writer_count,
        max_conversation_count=max_conversation_count,
        randomize=True,
        rng=rng,
    )

    _pick_with_caps(
        low_candidates,
        low_target,
        selected=selected,
        selected_ids=selected_ids,
        writer_counts=writer_counts,
        conversation_counts=conversation_counts,
        max_writer_count=max_writer_count,
        max_conversation_count=max_conversation_count,
        randomize=True,
        rng=rng,
    )

    backfill_needed = total_target - len(selected)
    if backfill_needed > 0:
        remaining = [record for record in filtered if record.entry_id not in selected_ids]
        _pick_with_caps(
            remaining,
            backfill_needed,
            selected=selected,
            selected_ids=selected_ids,
            writer_counts=writer_counts,
            conversation_counts=conversation_counts,
            max_writer_count=max_writer_count,
            max_conversation_count=max_conversation_count,
            randomize=False,
            rng=rng,
        )

    required_external = math.ceil(total_target * policy.external_min_ratio)
    _enforce_external_quota(
        selected=selected,
        selected_ids=selected_ids,
        filtered=filtered,
        writer_counts=writer_counts,
        conversation_counts=conversation_counts,
        required_external=required_external,
        max_writer_count=max_writer_count,
        max_conversation_count=max_conversation_count,
    )

    selected = selected[:total_target]
    selected.sort(key=_sort_key)

    current_external = _count_external(selected)
    top_count, mid_count, low_count = _count_buckets(selected)
    external_ratio = (current_external / len(selected)) if selected else 0.0

    return SelectionResult(
        records=selected,
        stats=SelectionStats(
            total_candidates=len(all_records),
            after_novelty_filter=len(filtered),
            selected=len(selected),
            external_ratio=external_ratio,
            required_external=required_external,
            external_target_met=current_external >= required_external,
            top_count=top_count,
            mid_count=mid_count,
            low_count=low_count,
        ),
    )
