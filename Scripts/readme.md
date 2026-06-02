# Scripts

Reserved for standalone helper scripts (e.g. one-off data exports or refresh
jobs). The reusable dashboard logic lives in `../src/`.

Current helper:

- `precompute_openalex_dashboard_stats.py`
  Runs a one-shot OpenAlex precompute job and writes dashboard-ready outputs to
  `../data/openalex_dashboard_stats/`.
