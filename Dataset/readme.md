# Dataset

OpenAlex exports of AI-related scholarly works.

- `ai_papers_clean_15k.csv` — flattened, cleaned crawl (15k works).
- `ai_papers_processed.csv` — adds engineered features (paper age, citation
  velocity, author/institution/country counts). Canonical analysis table; a
  copy lives in `../data/` for the Shiny app.

See `../Notebooks/` for the crawl and preprocessing steps.
