"""Data loading and enrichment layer for the dashboard.

The processed OpenAlex export is loaded once and enriched with a few derived
columns the dashboard relies on:

* ``year``          - integer publication year
* ``topic_bucket``  - paper mapped to a single canonical AI topic (heuristic)
* ``venue_group``   - friendly grouping of ``publication_type``
* ``regions``       - list of continents the paper's countries belong to
* ``country_list``  - list of ISO-2 codes
* ``novelty_proxy`` - structural proxy in [0, 1] (see note below)

NOTE ON SAMPLING: this export is a sample of *highly cited* AI-related works
(minimum citation count is in the hundreds). Volume trends therefore describe
the growth of *impactful* AI literature, not all AI output. This caveat is
surfaced in the dashboard footer and README.

NOTE ON NOVELTY: a real novelty/similarity signal requires the NLP embedding
model (a separate workstream). Until then ``novelty_proxy`` is a transparent
structural stand-in: papers that reference more prior work score lower, those
that reference less score higher.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from . import geo
from . import sector as sectormod

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_CANDIDATES = [
    _PROJECT_ROOT / "data" / "ai_papers_processed.csv",
    _PROJECT_ROOT / "Dataset" / "ai_papers_processed.csv",
]

# Canonical AI topic buckets used by the Topic filter, the stacked-area chart
# and the topic heatmap. Order matters: classification picks the first match,
# so more specific buckets are listed before the general "Core ML" catch-all.
TOPIC_RULES = [
    ("Computer Vision", ["vision", "image", "face", "expression recognition",
                          "pattern recognition", "object detection", "segmentation",
                          "visual", "video"]),
    ("NLP", ["natural language", "language model", "text ", "speech", "translation",
             "linguistic", "sentiment", "dialogue", "question answering", " nlp"]),
    ("Reinforcement Learning", ["reinforcement"]),
    ("Robotics", ["robot", "autonomous", "manipulation", "motion planning",
                  "control system"]),
    ("Healthcare AI", ["health", "medical", "clinical", "radiolog", "biomedical",
                       "disease", "diagnos", "genomic", "drug", "cancer", "imaging"]),
    ("AI Ethics & Fairness", ["ethic", "fairness", "bias", "privacy", "explainab",
                              "interpretab", "responsible", "governance", "accountab"]),
    ("Search & Recommender", ["information retrieval", "recommend", "ranking",
                              "search engine", "retrieval"]),
    ("ML Theory & Optimization", ["optimization", "statistical learning", "bayesian",
                                  "probabil", "graph theory", "computational theory",
                                  "convex", "stochastic"]),
    ("Core ML / Deep Learning", ["machine learning", "deep learning", "neural network",
                                 "classification", "clustering", "data mining",
                                 "artificial intelligence", "data classification",
                                 "representation learning", "generative"]),
]

TOPIC_BUCKETS = [name for name, _ in TOPIC_RULES] + ["Other"]

_VENUE_MAP = {
    "article": "Article / Journal",
    "preprint": "Preprint",
    "book-chapter": "Book chapter",
}
VENUE_GROUPS = ["Article / Journal", "Preprint", "Book chapter"]


def _classify_topic(text: str) -> str:
    t = text.lower()
    for bucket, keywords in TOPIC_RULES:
        if any(k in t for k in keywords):
            return bucket
    return "Other"


def _split_codes(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [c.strip() for c in value.split(";") if c.strip()]


@functools.lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load and enrich the processed dataset (cached for the app's lifetime)."""
    path = next((p for p in _DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "ai_papers_processed.csv not found in data/ or Dataset/."
        )
    df = pd.read_csv(path)

    # --- clean / coerce ----------------------------------------------------
    df["year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2026)]
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)
    df["referenced_works_count"] = pd.to_numeric(
        df["referenced_works_count"], errors="coerce"
    ).fillna(0)

    # --- derived columns ---------------------------------------------------
    topic_text = (
        df["primary_subfield"].fillna("") + " ; "
        + df["primary_topic"].fillna("") + " ; "
        + df["topics"].fillna("")
    )
    df["topic_bucket"] = topic_text.map(_classify_topic)
    df["venue_group"] = (
        df["publication_type"].map(_VENUE_MAP).fillna("Other")
    )
    df["country_list"] = df["countries"].map(_split_codes)
    df["regions"] = df["country_list"].map(
        lambda codes: sorted({geo.region(c) for c in codes})
    )

    # Academia vs Industry collaboration category (see src/sector.py).
    df["institution_list"] = df["institutions"].map(_split_codes)
    df["sector"] = df["institution_list"].map(sectormod.paper_sector)

    # Structural novelty proxy: fewer references -> higher novelty score.
    ref_pct = df["referenced_works_count"].rank(pct=True)
    df["novelty_proxy"] = (1.0 - ref_pct).round(3)

    return df.reset_index(drop=True)


def year_bounds() -> tuple[int, int]:
    df = load_data()
    return int(df["year"].min()), int(df["year"].max())


# ---------------------------------------------------------------------------
# Aggregation helpers (operate on an already-filtered DataFrame)
# ---------------------------------------------------------------------------
def papers_per_year(df: pd.DataFrame) -> pd.Series:
    return df.groupby("year").size().sort_index()


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, country). A multi-country paper counts once per
    country involved."""
    out = df.explode("country_list").rename(columns={"country_list": "iso2"})
    out = out[out["iso2"].notna() & (out["iso2"] != "")].copy()
    out["country"] = out["iso2"].map(geo.name)
    out["iso3"] = out["iso2"].map(geo.iso3)
    out["region"] = out["iso2"].map(geo.region)
    out["pop_m"] = out["iso2"].map(geo.population_m)
    return out


def explode_institutions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, institution), tagged with its inferred sector."""
    col = "institution_list" if "institution_list" in df.columns else "institutions"
    src = df[col] if col == "institution_list" else df[col].map(_split_codes)
    out = df.assign(institution=src).explode("institution")
    out = out[out["institution"].notna() & (out["institution"] != "")].copy()
    out["inst_sector"] = out["institution"].map(sectormod.classify_institution)
    return out
