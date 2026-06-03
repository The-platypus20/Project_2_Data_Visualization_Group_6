# Scripts

Reserved for standalone helper scripts (e.g. one-off data exports or refresh
jobs). The reusable dashboard logic lives in `../src/`.

Current helper:

- `precompute_openalex_dashboard_stats.py`
  Runs a one-shot OpenAlex precompute job and writes dashboard-ready outputs to
  `../data/openalex_dashboard_stats/`.
- `export_openalex_ai_works.py`
  Exports large row-level OpenAlex AI datasets in resumable CSV chunks to a
  tracked year-shard directory under `../Dataset/openalex_exports/` by default.
  It checkpoints both written chunks and the in-memory buffer, supports
  collaborator handoff via `--resume`, and defaults to a conservative per-run
  call cap so one run stays under the OpenAlex free-tier daily limit.
