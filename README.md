# AI Research Growth & Concentration

An interactive **Shiny for Python** dashboard that tells the story of Artificial
Intelligence research from **2000 to 2025**, built on ~2.24 million papers from
the [OpenAlex](https://openalex.org/) *Artificial Intelligence* subfield (1702).

It answers three questions across three tabs:

| Tab | Question | Highlights |
|-----|----------|-----------|
| **How AI grew** | How fast did AI research scale, and who led it? | growth curve with research-wave milestones, top countries, institution types, topic diversity |
| **Where ideas moved** | Which topics rose, faded, and where did the frontier go? | topic bubbles & drill-down, growth–impact frontier, rising / fading terms |
| **What drives impact** | What makes a paper high-impact — and can we predict it? | citation-concentration funnel, logistic-regression drivers, gradient-boosting prediction, LSTM forecast |

---

## 1. How it works — the offline → cache → app architecture

The dashboard never touches the multi-gigabyte raw data at runtime. All heavy
computation happens **offline** and is written to small CSV "cache" files that
the app reads instantly:

```
 3 raw shards            merge          1 raw file              clean + EDA            1 clean file
 ai_works_merge_*.csv  ─────────────▶  ai_works_merged_raw  ──────────────────────▶  ai_works_clean.csv
 (2000-09,10-19,20-25)  merge_raw_      .csv                  build_clean_dataset.py   (+ preprocess_report.md)
                        dataset.py
                                                                                            │
                                                                                            │  build_all_cache.py
                                                                                            ▼
                                                                            Dataset/dashboard_cache/*.csv
                                                                            (small aggregates + ML results)
                                                                                            │
                                                                                            │  app.py reads cache
                                                                                            ▼
                                                                                   Interactive dashboard
```

- **Runtime (app.py)** only needs `Dataset/dashboard_cache/` — which **is committed to the repo**.
  You can run the dashboard immediately after installing the requirements, with **no data download**.
- **Rebuilding the cache** needs the big data files (see [§5](#5-data-files-google-drive)).

---

## 2. Repository structure

```
.
├── app.py                       # Shiny entry point (UI + server wiring)
├── build_all_cache.py           # Rebuild every cache file from the clean dataset (one command)
├── build_all_cache.sh           # Same, as a shell script
├── requirements.txt             # Runtime dependencies
├── requirements-build.txt       # Extra deps to rebuild the ML cache (sklearn, lightgbm, torch)
│
├── src/
│   ├── theme.py                 # Shared dark Plotly theme + global dashboard CSS
│   ├── narrative_common.py      # Small shared UI helpers (cards, badges, metrics)
│   ├── narrative_data.py        # Cached readers for the core/shared cache CSVs
│   ├── tab3_data.py             # Cached readers for the Tab 3 (tab3_*.csv) cache
│   ├── mod_narrative.py         # Assembles the three tabs into the navbar
│   ├── tab_how_ai_grew.py       # Tab 1
│   ├── tab_where_ideas_moved.py # Tab 2
│   ├── tab_what_created_impact.py# Tab 3  ("What drives impact")
│   ├── build_landscape_json.py  # Builds www/data/*.json layouts for Tab 2
│   ├── export_openalex_ai_works.py  # Resumable OpenAlex crawler (STEP 0, see §5.1)
│   └── preprocess/              # Offline builders (run once, write cache)
│       ├── merge_raw_dataset.py            # STEP 1: merge 3 shards → 1 raw file
│       ├── build_clean_dataset.py          # STEP 2: EDA + clean → ai_works_clean.csv + report
│       ├── build_core_dashboard_cache.py   # STEP 3a: yearly/topic/country/diversity/paper cache
│       ├── build_impact_ml_cache.py        # STEP 3b: Tab 3 ML cache (tab3_*.csv)
│       ├── build_institution_type_view.py  # STEP 3c: institution-type cache
│       └── build_rising_fading_terms_family.py # STEP 3d: rising/fading terms cache
│
├── Dataset/
│   ├── dashboard_cache/         # ✅ committed small cache CSVs (what the app reads)
│   └── clean/
│       └── preprocess_report/preprocess_report.md  # EDA / data-quality report
│
└── www/                         # Static assets (CSS/JS) + precomputed JSON layouts for Tab 2
    ├── landscape.css · landscape.js
    └── data/*.json
```

> Big data files (the 3 shards, `ai_works_merged_raw.csv`, `ai_works_clean.csv`) are
> **git-ignored** and live on Google Drive — see [§5](#5-data-files-google-drive).

---

## 3. Setup

Requires **Python 3.11+**.

```bash
# create + activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Runtime dependencies: `numpy`, `pandas`, `plotly`, `shiny`, `shinywidgets`.

---

## 4. Run the dashboard

The cache is committed, so this works right after setup — **no data download needed**:

```bash
shiny run app.py --reload
# or pin host/port:
shiny run app.py --host 127.0.0.1 --port 8000 --reload
```

Open the local URL Shiny prints (e.g. `http://127.0.0.1:8000`).

---

## 5. Data files (Google Drive)

The raw and intermediate datasets are too large for GitHub and are hosted here:

**📁 https://drive.google.com/drive/folders/1JOBKt8ITIpp9HxvdXlPjJefiAh2CpUeY**

| File | Size (≈) | What it is | Place it in |
|------|----------|------------|-------------|
| `ai_works_merge_2000_2009.csv` | 303 MB | Raw OpenAlex shard 2000–2009 | `Dataset/` |
| `ai_works_merge_2010_2019.csv` | 595 MB | Raw OpenAlex shard 2010–2019 | `Dataset/` |
| `ai_works_merge_2020_2025.csv` | 867 MB | Raw OpenAlex shard 2020–2025 | `Dataset/` |
| `ai_works_merged_raw.csv` | 1.7 GB | The 3 shards concatenated (output of STEP 1) | `Dataset/clean/` |
| `ai_works_clean.csv` | 1.7 GB | Cleaned, analysis-ready table (output of STEP 2) | `Dataset/clean/` |

You only need these to **rebuild** the cache. Two options:

- **Fast path** — download just `ai_works_clean.csv` into `Dataset/clean/`, then run `build_all_cache.py` ([§6](#6-rebuilding-the-cache)).
- **Full path** — download the 3 shards into `Dataset/`, then run the whole pipeline from STEP 1.

### 5.1 Crawling the raw shards yourself from OpenAlex (optional)

If you'd rather regenerate the raw shards from scratch instead of downloading them,
use `src/export_openalex_ai_works.py`. It pulls the OpenAlex *Artificial
Intelligence* subfield (`primary_topic.subfield.id:1702`, DOI-bearing, non-retracted
articles/preprints/chapters/proceedings) with cursor pagination, and is **fully
resumable**.

**1) Get an OpenAlex API key** and export it (plus a contact email) as environment
variables:

```bash
export OPENALEX_API_KEY="your_key_here"
export OPENALEX_EMAIL="you@example.com"     # optional but polite (sent as mailto)
pip install requests                         # crawler dependency (also in requirements-build.txt)
```

**2) Crawl each of the three shards** (`--target-rows 0` = all matching works for
that year range):

```bash
python src/export_openalex_ai_works.py --start-year 2000 --end-year 2009 --target-rows 0
python src/export_openalex_ai_works.py --start-year 2010 --end-year 2019 --target-rows 0
python src/export_openalex_ai_works.py --start-year 2020 --end-year 2025 --target-rows 0
```

Each shard is written as resumable chunks:
`Dataset/openalex_exports/ai_works_<start>_<end>/part_00001.csv`, `part_00002.csv`, …
plus a `manifest.json`.

- **Resuming:** if a run stops (network drop, daily limit, you close the laptop),
  re-run the *same* command with `--resume` — it continues from the last chunk and
  never re-downloads finished paper IDs:
  ```bash
  python src/export_openalex_ai_works.py --start-year 2020 --end-year 2025 --target-rows 0 --resume
  ```
- **Free-tier daily limit:** OpenAlex allows ~10,000 list calls/day (≈ 1M works).
  The script caps itself safely (`--max-daily-list-calls`, default 9500). Large
  shards (e.g. 2020–2025 ≈ 1.1M rows) therefore span **multiple days** — just
  `--resume` the next day (a new API key resets the budget sooner). Pass
  `--allow-full-budget` to use the entire estimated daily budget in one go.

**3) Concatenate each shard's chunks into the single file the pipeline expects**
(`merge_raw_dataset.py` keeps the first header and drops the rest):

```bash
python src/preprocess/merge_raw_dataset.py \
  --input Dataset/openalex_exports/ai_works_2000_2009/part_*.csv \
  --output Dataset/ai_works_merge_2000_2009.csv

python src/preprocess/merge_raw_dataset.py \
  --input Dataset/openalex_exports/ai_works_2010_2019/part_*.csv \
  --output Dataset/ai_works_merge_2010_2019.csv

python src/preprocess/merge_raw_dataset.py \
  --input Dataset/openalex_exports/ai_works_2020_2025/part_*.csv \
  --output Dataset/ai_works_merge_2020_2025.csv
```

You now have the three `Dataset/ai_works_merge_*.csv` shards — continue with the
full pipeline in [§6 Option B](#6-rebuilding-the-cache).

---

## 6. Rebuilding the cache

Install the extra ML build dependencies first:

```bash
pip install -r requirements-build.txt   # scikit-learn, lightgbm, torch
```

### Option A — from the clean file (fast)

```bash
# requires Dataset/clean/ai_works_clean.csv
python build_all_cache.py
```

This runs all four builders in order (core → Tab 3 ML → institution → rising/fading)
and writes every file into `Dataset/dashboard_cache/`.

### Option B — full pipeline from the raw shards

```bash
# STEP 1 — merge the 3 shards into one raw file
python src/preprocess/merge_raw_dataset.py \
  --input Dataset/ai_works_merge_2000_2009.csv \
          Dataset/ai_works_merge_2010_2019.csv \
          Dataset/ai_works_merge_2020_2025.csv \
  --output Dataset/clean/ai_works_merged_raw.csv

# STEP 2 — EDA + clean (writes Dataset/clean/preprocess_report/preprocess_report.md)
python src/preprocess/build_clean_dataset.py \
  --input Dataset/clean/ai_works_merged_raw.csv \
  --output Dataset/clean/ai_works_clean.csv

# STEP 3 — build all caches from the clean file
python build_all_cache.py
```

**What the clean step does** (one streaming pass, low memory): drops rows with
missing/out-of-range year or missing topic, de-duplicates by `paper_id`
(falling back to title+year+topic), fills numeric impact fields with 0, and keeps
all original columns so downstream builders are unchanged. It writes a full EDA /
data-quality report to `Dataset/clean/preprocess_report/preprocess_report.md`.

---

## 7. Cache files reference

All in `Dataset/dashboard_cache/` (committed). Structure is fixed; only the counts
reflect the cleaned data.

| Builder | Files | Used by |
|---------|-------|---------|
| `build_core_dashboard_cache.py` | `yearly_counts`, `bucket_year_counts`, `topic_year_counts`, `diversity_metrics`, `impact_topic_scatter`, `top_countries`, `country_topic_year`, `paper_lookup` | Tabs 1 & 2 |
| `build_institution_type_view.py` | `institution_type_year_summary`, `institution_type_top_institutions`, `institution_type_country_year`, `institution_type_paper_view` | Tab 1 |
| `build_rising_fading_terms_family.py` | `rising_terms`, `fading_terms`, `rising_fading_terms`, `wordcloud_terms` | Tab 2 |
| `build_impact_ml_cache.py` | `tab3_lorenz`, `tab3_funnel`, `tab3_concentration`, `tab3_drivers`, `tab3_model_metrics`, `tab3_roc_curve`, `tab3_calibration`, `tab3_forecast` | Tab 3 |

---

## 8. The machine learning behind Tab 3

**Impact label.** A paper is *high impact* if its **citation velocity** (citations
per year) is in the **top 10% of its own publication year**. Ranking within each
year removes the age bias that would otherwise always favour older papers.

The models predict that yes/no label **using only traits known at publication
time** (number of references, team size, institutions, international reach, open
access, venue type) — **no citation data is used as input**, so there is no leakage.

The tab is laid out as a top-to-bottom flow of four beats:

| Beat | Question | Method | Library | Shown as |
|------|----------|--------|---------|----------|
| 1 | How concentrated is impact? | citation-threshold funnel (Gini/Lorenz also computed in cache) | numpy | funnel with exact counts for the rare highly-cited tiers |
| 2 | Which traits drive impact? | **standardized logistic regression** | scikit-learn | per-trait coefficient tornado |
| 3 | Can we predict it? | **gradient boosting** on a 20% hold-out | LightGBM | ROC curve + 3-model comparison (ROC-AUC ≈ 0.82, lift@10 ≈ 4×) |
| 4 | Where is impact heading? | **LSTM** sequence forecast | PyTorch | every family's share of high-impact papers, all lines at once with a highlighted 2026–2028 forecast + band |

---

## 9. Notes & troubleshooting

- **macOS OpenMP deadlock.** `build_impact_ml_cache.py` can hang (0% CPU, no
  output) because numpy ships GNU OpenMP while LightGBM/torch ship LLVM OpenMP and
  the two runtimes deadlock. The fix is already baked into the script
  (`OMP_NUM_THREADS` + `KMP_DUPLICATE_LIB_OK` set before imports, and
  `torch.set_num_threads(1)`), so a normal `python build_all_cache.py` works.
- **Git ignores big data.** Everything under `Dataset/*.csv` and `Dataset/**/*.csv`
  is ignored, *except* `Dataset/dashboard_cache/*.csv` which is tracked on purpose.
- **Data scope.** OpenAlex subfield 1702 (Artificial Intelligence) only — so
  Computer Vision (a separate subfield, 1707) is intentionally absent.

---

## 10. Tech stack

Python · Shiny for Python · Plotly / shinywidgets · pandas / numpy ·
scikit-learn · LightGBM · PyTorch · data from OpenAlex.

---

## 11. Authors — Group 6

| No. | Student ID | Full Name | Contribution |
|-----|------------|-----------|--------------|
| 1 | V202401781 | Nguyen Thi Phuong Thao | EDA, preprocessing, report writing |
| 2 | V202401694 | Le Thao Vy | Shiny dashboard, UX design, report writing |
| 3 | V202401748 | Ngo Thanh An | Preprocessing, visualization, ML, dashboard |
| 4 | V202401647 | Nguyen Hoang Nam | ML integration, dashboard support, report writing |
