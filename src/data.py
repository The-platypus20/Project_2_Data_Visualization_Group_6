"""Load precomputed OpenAlex snapshot tables for the dashboard.

The dashboard no longer queries OpenAlex live. Instead, a one-shot precompute
job writes exact aggregate tables and a stratified paper sample to
`data/openalex_dashboard_stats/`, and this module loads that snapshot into
memory for the UI.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import pandas as pd

from . import geo

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATS_DIR = _PROJECT_ROOT / "data" / "openalex_dashboard_stats"
_STATS_DIR = Path(os.environ.get("OPENALEX_DASHBOARD_STATS_DIR", _DEFAULT_STATS_DIR)).expanduser()
EXCLUDED_YEARS = {2026}

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
VENUE_GROUPS = ["Article / Journal", "Preprint", "Book chapter", "Other"]
OA_GROUPS = ["gold", "green", "hybrid", "bronze", "diamond", "closed", "unknown"]

_SNAPSHOT_FILES = {
    "year_counts": "exact_year_counts.csv",
    "country_counts": "exact_country_counts.csv",
    "country_year_counts": "exact_country_year_counts.csv",
    "type_counts": "exact_type_counts.csv",
    "type_year_counts": "exact_type_year_counts.csv",
    "oa_counts": "exact_oa_status_counts.csv",
    "oa_year_counts": "exact_oa_status_year_counts.csv",
    "subfield_counts": "exact_subfield_counts.csv",
    "subfield_year_counts": "exact_subfield_year_counts.csv",
    "primary_topic_year_counts": "exact_primary_topic_year_counts.csv",
    "topic_bucket_year_counts": "exact_topic_bucket_year_counts.csv",
    "top_institutions_by_country": "exact_top_institutions_by_country.csv",
    "sample_plan": "sample_plan.csv",
    "sampled_papers": "sampled_papers.csv",
    "manifest": "manifest.json",
}


def stats_dir() -> Path:
    return _STATS_DIR


def _required_path(name: str) -> Path:
    path = _STATS_DIR / _SNAPSHOT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing snapshot file: {path}. "
            "Run Scripts/precompute_openalex_dashboard_stats.py first."
        )
    return path


def _optional_csv(name: str) -> pd.DataFrame:
    path = _STATS_DIR / _SNAPSHOT_FILES[name]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _split_semicolon(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _normalize_country_code(value) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if "/" in text:
        text = text.rstrip("/").split("/")[-1]
    return text.upper()


def _normalize_country_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country_code" not in df.columns:
        return df
    out = df.copy()
    out["country_code"] = out["country_code"].map(_normalize_country_code)
    out["country_name"] = out["country_code"].map(geo.name)
    out["region"] = out["country_code"].map(geo.region)
    return out


def _normalize_institutions_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "country_code" in out.columns:
        out["country_code"] = out["country_code"].map(_normalize_country_code)
        out["country_name"] = out["country_code"].map(geo.name)
    return out


def _prepare_sampled_papers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in [
        "publication_year", "citation_count", "citations_per_year",
        "referenced_works_count", "sample_weight", "paper_age",
        "author_count", "institution_count", "country_count", "novelty_proxy",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["year"] = out["publication_year"].astype("Int64")
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    out = out[~out["year"].isin(EXCLUDED_YEARS)].copy()
    out["country_list"] = out.get("countries", pd.Series(dtype=str)).map(_split_semicolon)
    out["institution_list"] = out.get("institutions", pd.Series(dtype=str)).map(_split_semicolon)
    out["regions"] = out["country_list"].map(lambda xs: sorted({geo.region(x) for x in xs}))
    out["oa_status"] = out.get("oa_status", pd.Series(dtype=str)).fillna("unknown").replace("", "unknown")
    out["venue_group"] = out.get("venue_group", pd.Series(dtype=str)).fillna("Other").replace("", "Other")
    return out.reset_index(drop=True)


def _filter_excluded_years(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "publication_year" not in df.columns:
        return df
    out = df.copy()
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    return out[~out["publication_year"].isin(EXCLUDED_YEARS)].copy()


@functools.lru_cache(maxsize=1)
def load_snapshot() -> dict[str, object]:
    _required_path("year_counts")
    manifest_path = _STATS_DIR / _SNAPSHOT_FILES["manifest"]
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    snapshot = {
        "stats_dir": _STATS_DIR,
        "manifest": manifest,
        "year_counts": _filter_excluded_years(pd.read_csv(_required_path("year_counts"), encoding="utf-8-sig")),
        "country_counts": _normalize_country_frame(_optional_csv("country_counts")),
        "country_year_counts": _filter_excluded_years(_normalize_country_frame(_optional_csv("country_year_counts"))),
        "type_counts": _optional_csv("type_counts"),
        "type_year_counts": _filter_excluded_years(_optional_csv("type_year_counts")),
        "oa_counts": _optional_csv("oa_counts"),
        "oa_year_counts": _filter_excluded_years(_optional_csv("oa_year_counts")),
        "subfield_counts": _optional_csv("subfield_counts"),
        "subfield_year_counts": _filter_excluded_years(_optional_csv("subfield_year_counts")),
        "primary_topic_year_counts": _filter_excluded_years(_optional_csv("primary_topic_year_counts")),
        "topic_bucket_year_counts": _filter_excluded_years(_optional_csv("topic_bucket_year_counts")),
        "top_institutions_by_country": _normalize_institutions_frame(
            _optional_csv("top_institutions_by_country")
        ),
        "sample_plan": _optional_csv("sample_plan"),
        "sampled_papers": _prepare_sampled_papers(_optional_csv("sampled_papers")),
    }
    return snapshot


def year_bounds(snapshot: dict[str, object] | None = None) -> tuple[int, int]:
    snap = snapshot or load_snapshot()
    years = snap["year_counts"]["publication_year"]
    return int(years.min()), int(years.max())


def snapshot_summary(snapshot: dict[str, object] | None = None) -> dict[str, object]:
    snap = snapshot or load_snapshot()
    manifest = dict(snap.get("manifest") or {})
    manifest.setdefault("sample_size_materialized", len(snap["sampled_papers"]))
    manifest.setdefault("stats_dir", str(snap["stats_dir"]))
    return manifest


def filter_years(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    if df.empty or "publication_year" not in df.columns:
        return df.copy()
    out = df.copy()
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    return out[(out["publication_year"] >= year_min) & (out["publication_year"] <= year_max)].copy()


def filter_sampled_papers(snapshot: dict[str, object], year_min: int, year_max: int) -> pd.DataFrame:
    df = snapshot["sampled_papers"]
    if df.empty:
        return df.copy()
    return df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
