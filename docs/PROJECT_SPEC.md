# intra-model-memval-lab - Proposed Specs

## 1. Goal
- Build a standalone hardened pipeline for **categorized intra-model memory**.
- Produce training-ready datasets while mitigating self-reinforcing feedback loops.
- Support dual ingestion profiles without semantic drift:
  - `internal` profile: numeric-native memory.
  - `MCP` profile: text-native memory (no fake numeric conversion).

## 2. Scope
- Input: curated records exported from `llm-memory` (or equivalent store).
- Processing: validation, provenance scoring, category normalization, dedup policy, sampling.
- Output: dual datasets:
  - numeric-oriented dataset (for internal profile)
  - text-oriented dataset (for MCP profile and audit/debug)

## 3. Non-goals
- No MCP runtime serving in this project.
- No live memory write path for production traffic.
- No direct deployment in Binah runtime stack.

## 4. Core Data Model (dual-profile, proposed)
- Common fields:
- `entry_id`: str
- `category`: enum (`decision`, `fact`, `rule`, `assumption`, `unknown`, `invalidated`)
- `modality_primary`: enum (`numeric`, `text`)
- `importance_score`: int `0..100`
- `novelty_score`: float `0..1`
- `is_external`: bool
- `provenance_level`: enum (`trusted_runtime`, `verified_tool`, `declared_only`)
- `context_hash`: str (16 hex)
- `writer_model`: str
- `writer_agent_id`: str
- `created_at_utc`: str ISO8601
- `metadata`: object
- `schema_version`: str
- `model_fingerprint`: str

- Internal payload (`modality_primary=numeric`):
- `raw_numeric`: object (required)
  - `dtype`: str (`float32`, `float16`, `int8`, ...)
  - `shape`: list[int]
  - `encoding`: enum (`base64`, `npz_ref`, `arrow_ref`)
  - `payload_b64`: optional str
  - `blob_path`: optional str
  - `blob_hash`: optional str
- `text_view`: optional str (shadow/audit)

- MCP payload (`modality_primary=text`):
- `text_view`: required str (primary payload)
- `raw_numeric`: absent by design
- no synthetic conversion to pseudo-numeric vectors at write time

## 5. Validation Rules (Phase 1)
- Reject entries with missing common required fields.
- Internal profile: reject entries with missing `raw_numeric`.
- MCP profile: reject entries with missing `text_view`.
- Clamp score/ranges to valid intervals.
- Drop entries with `novelty_score < 0.2`.
- Mark `provenance_level=declared_only` when evidence is absent.
- Mark record as `train_ready=false` if required payload for selected profile is invalid/absent.

## 6. Feedback-Loop Mitigation Policy
- Hard novelty filter: `novelty_score >= 0.2`.
- External anchor quota: at least `25%` where `is_external=true`.
- Sampling distribution:
  - `60%` high (`importance_score >= 70`)
  - `25%` medium (`40..69`)
  - `15%` low (`0..39`) with coarse quality guard
- Diversity caps:
  - max `20%` from same `writer_model`
  - max `5%` from same `conversation_id` (if present)

## 7. Dedup/Consolidation Matrix
- High similarity + same `context_hash` -> local variant (do not consolidate for FT).
- High similarity + different `context_hash` -> transferable candidate (eligible for consolidation).

## 8. Export Format
- Internal-oriented export:
  - `record_id`, `raw_numeric`, `category`, `importance_score`, `meta`
- MCP/text export:
  - `record_id`, `text_view`, `category`, `importance_score`, `meta`
- Deterministic sort by `(importance_score desc, created_at asc, entry_id asc)` before write.

## 9. Quality Gates
- Gate A: schema validation pass rate >= 99%.
- Gate B: external quota satisfied.
- Gate C: distribution 60/25/15 within tolerance ±2%.
- Gate D: no duplicate `entry_id` in output.

## 10. Phased Delivery
- Phase 0: scaffold + policy CLI.
- Phase 1: dataset builder from sqlite/jsonl with profile-aware validation (`internal`/`MCP`).
- Phase 2: provenance hardening + richer evidence attestation.
- Phase 3: trainer adapters (GPT-2 small baseline, eval report).

## 11. Core Save Service (implemented baseline)
- Service: `MemoryPersistence`
- API:
  - `save_memory_record(record)`
  - `save_many(records)` (single-transaction batch)
  - `load_memory_record(entry_id)`
  - `list_records(...)` (filtered + paginated)
  - `search_records(query, ...)` (text/meta search + filters)
- Guarantees:
  - schema validation before persistence
  - self-evaluation validation/scoring in save path (toggleable enforcement)
  - idempotent upsert by `entry_id`
  - full rollback for batch failures (`save_many`)
  - dual persistence (blob numeric + sqlite metadata/text shadow)
  - logical rollback on transaction failure
  - indexes for `context_hash`, `importance_score`, `category`
  - query pagination order stable: `importance desc`, `created_at asc`, `entry_id asc`
