# AI Research Landscape Dashboard

Interactive Shiny for Python dashboard exploring how AI research grew from 2000 to 2025, where topic families moved, and which topics created visible impact.

## Requirements

- Python 3.11 or newer
- Precomputed dashboard cache files in `Dataset/dashboard_cache/`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
shiny run app.py --host 127.0.0.1 --port 8922 --reload
```

Open the local URL printed by Shiny.

## Data

The runtime dashboard reads precomputed CSV cache files from:

```text
Dataset/dashboard_cache/
```

The app does not require raw paper-level CSV files such as OpenAlex exports or `Dataset/ai_papers_processed.csv`. Raw OpenAlex exports are excluded from GitHub; rebuild cache files separately before running the dashboard if the cache folder is missing.
