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

## OpenAlex crawl tutorial

### 1. Prerequisites

- Have a free OpenAlex API key from `https://openalex.org/settings/api`
- Export these variables in your shell:

```bash
export OPENALEX_API_KEY=your_key_here
export OPENALEX_EMAIL=your_email@example.com
```

- Activate the project environment:

```bash
cd /home/hoangnam/Project_2_Data_Visualization_Group_6-1
source myenv/bin/activate
```

### 2. How the crawler stores progress

Each year shard writes to its own tracked directory under:

```text
raw_dashboard/Dataset/openalex_exports/
```

For example, the `2020–2025` shard writes to:

```text
raw_dashboard/Dataset/openalex_exports/ai_works_2020_2025/
```

Inside that folder, the crawler keeps:

- `part_00001.csv`, `part_00002.csv`, ...: finished chunks
- `manifest.json`: current cursor, chunk count, budget metadata
- `_buffer_checkpoint.csv`: in-memory rows not yet flushed to a chunk

This means another person can pull the repo and continue with `--resume`
without redoing finished chunks.

### 3. Basic crawl command

To crawl one year shard completely:

```bash
python raw_dashboard/Scripts/export_openalex_ai_works.py \
  --start-year 2020 \
  --end-year 2025 \
  --target-rows 0 \
  --resume
```

Notes:

- `--target-rows 0` means: crawl all matching papers for that shard
- `--resume` is safe to use even on the first run
- the script stops at the daily free-tier safety cap and can continue the next day

### 4. Recommended team workflow

Do not have two people run the same shard at the same time.

Instead, split non-overlapping year ranges. Example:

- Person A: `2020–2025`
- Person B: `2015–2019`
- Person C: `2010–2014`
- Person D: `2000–2009`

Each person runs:

```bash
python raw_dashboard/Scripts/export_openalex_ai_works.py \
  --start-year YYYY \
  --end-year YYYY \
  --target-rows 0 \
  --resume
```

After each work session:

```bash
cd raw_dashboard
git add Dataset/openalex_exports/<your_shard_dir>
git commit -m "Update OpenAlex crawl progress for <your_shard_dir>"
git push
```

Another collaborator can then:

```bash
git pull
python Scripts/export_openalex_ai_works.py --start-year YYYY --end-year YYYY --target-rows 0 --resume
```

### 5. Your assigned shard

For the `2020–2025` shard, use:

```bash
python raw_dashboard/Scripts/export_openalex_ai_works.py \
  --start-year 2020 \
  --end-year 2025 \
  --target-rows 0 \
  --resume
```

### 6. Safety and duplicate behavior

The crawler:

- skips paper IDs already written in previous chunk files
- skips paper IDs already present in the current checkpoint buffer
- restores the checkpoint buffer on resume
- uses a conservative daily call cap by default for OpenAlex free-tier use

### 7. If a crawl stops or crashes

Just rerun the same shard command:

```bash
python raw_dashboard/Scripts/export_openalex_ai_works.py \
  --start-year 2020 \
  --end-year 2025 \
  --target-rows 0 \
  --resume
```

The script will continue from the saved state in `manifest.json` and
`_buffer_checkpoint.csv`.
