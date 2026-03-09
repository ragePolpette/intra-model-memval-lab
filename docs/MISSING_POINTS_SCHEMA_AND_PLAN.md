# MISSING_POINTS_SCHEMA_AND_PLAN.md

Documento storico superato.

Il nuovo piano non parte piu' da dual profile `internal/MCP` su `MemoryRecord`,
ma da una pipeline sperimentale fondata su:

- `EpisodeRecord`
- `TraceArtifact`
- `UpdateCandidate`
- `EvaluationSpec`
- `EvaluationRun`

Per i gap residui e i prossimi step usare:

- `docs/ARCHITECTURE.md`
- `docs/MIGRATION_NOTES.md`
- `REALIGNMENT_REPORT.md`
