# Changelog

## Unreleased

### Fixed

- `ExperimentStore` now enables `PRAGMA foreign_keys=ON` on every connection, not only on the one used to create the schema. Writes that reference a missing run, episode, trace or evaluation spec now raise `sqlite3.IntegrityError` instead of storing orphan rows. Existing databases that already contain orphan rows can fail when those rows are updated.
- The consolidation `ExperimentRun` no longer stays `planned` with `mode=planner-no-training` after real training: the POC moves it to `running` with `mode=lora-training`, then to `completed` (with adapter, evaluation run ids, decision and manifest references) or `failed` (with the error). New `ConsolidationPlanner.update_run`.
- The POC held-out split is stratified by example kind and never puts `kind=anchor` variations in the held-out set (`split_holdout`).
- The POC frees cached CUDA memory after dropping each model (`release_accelerator_memory`).

### Changed

- Evaluation scoring: `EvaluationCase` and `SyntheticExample` accept an optional `rubric` (`required` groups of alternative terms, `forbidden` terms). Cases with a rubric are scored on it with word-boundary matching; cases without one keep the normalized containment match. Observations record the scorer used and the missing groups / forbidden hits; the evaluation run mode is now `executor-rubric-or-containment`. The example fixtures define rubrics, and `regression.jsonl` grows from 6 to 12 cases.
- Promotion gate: `GateThresholds` adds `regression_tolerance_cases` (tolerance as a number of lost cases) and `min_regression_cases` (default 0); `decide_promotion` accepts `regression_case_count` and reports a new `regression_coverage` check. The POC script defaults to `--min-regression-cases 10`, so a regression benchmark with fewer than 10 cases now rolls the candidate back.
- `ReplayBuffer.summary()` keys are renamed to `replay_anchor_count`, `requested_replay_anchor_count` and `replay_anchor_shortfall` to separate replay anchors from `kind=anchor` examples; the composition issue text now says `kind=anchor example share`. The POC replay config key `anchor_source` is now `replay_anchor_source`.
- `verdict.json` includes the consolidation `run_id`.
- The SAE spike verification notes moved from the script docstring to `docs/history/SAE_DRIFT_SPIKE.md`.

### Added

- `experiment_manifest.json` written by the POC for each run (also registered as an artifact): package versions, git commit and dirty flag, base model id with requested and resolved revision (or SHA-256 of a local model directory), resolved configuration, seed, devices, pre/post metrics, decision and adapter location.
- CLI commands `report --run-id` (pre/post match rate per family and per case, text or JSON), `list-lessons`, `show-lesson`, `list-consolidation-candidates`, `show-consolidation-candidate`, `show-verdict`. `cli.main` accepts an optional argument list.
- POC flags `--max-new-tokens`, `--base-model-revision`, `--min-regression-cases`, `--regression-tolerance-cases`; `run_consolidation_poc.main` accepts injectable generator and trainer factories, and a smoke test runs the full script with fakes.
- `ruff` in the `[dev]` extra with configuration in `pyproject.toml` and a lint step in CI; `[spike]` extra with `sae-lens` for `scripts/spikes/sae_drift_spike.py`.
