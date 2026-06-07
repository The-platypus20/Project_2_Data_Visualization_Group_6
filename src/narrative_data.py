"""Cached data helpers for the AI research story dashboard.

This demo-facing layer intentionally reads precomputed CSV aggregates from
Dataset/dashboard_cache instead of scanning the raw multi-million-row exports
inside Shiny render functions.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import TOPIC_RULES
except Exception:
    TOPIC_RULES = [
        ("Language AI & NLP", [
            "natural language", "language", "text", "speech", "semantic",
            "sentiment", "dialogue", "topic modeling"
        ]),
        ("Core AI & Machine Learning", [
            "machine learning", "deep learning", "neural", "classification",
            "clustering", "graph neural", "adversarial"
        ]),
        ("AI Infrastructure & Security", [
            "optimization", "bayesian", "probabilistic", "cryptography",
            "security", "algorithm", "quantum"
        ]),
        ("Robotics & Autonomous Systems", [
            "robot", "robotics", "control", "tracking", "sensor", "planning"
        ]),
        ("Healthcare & Medical AI", [
            "health", "healthcare", "medical", "clinical", "cancer",
            "disease", "diagnosis"
        ]),
        ("Responsible AI & Governance", [
            "privacy", "fairness", "ethics", "bias", "explainable", "xai"
        ]),
        ("Reinforcement Learning & Agents", [
            "reinforcement", "agent", "policy", "reward", "multi-agent"
        ]),
    ]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "Dataset" / "dashboard_cache"
CACHE_READ_LOG = CACHE_DIR / "_cache_read_log.txt"

print(f"[narrative_data] imported from {Path(__file__).resolve()}", flush=True)


def _write_cache_log(message: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_READ_LOG.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception as exc:
        print(f"[cache_csv] log write failed: {exc}", flush=True)


def _cache_csv(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = CACHE_DIR / name
    if path.exists():
        resolved = path.resolve()
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        start = time.perf_counter()
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            message = (
                f"[cache_csv] ERROR {name}: path={resolved} size={size_bytes:,} bytes "
                f"({size_mb:.2f} MB), read_time={elapsed:.3f}s, error={exc}"
            )
            print(message, flush=True)
            _write_cache_log(message)
            raise
        elapsed = time.perf_counter() - start
        message = (
            f"[cache_csv] READ {name}: path={resolved} rows={len(df):,}, "
            f"size={size_bytes:,} bytes ({size_mb:.2f} MB), read_time={elapsed:.3f}s"
        )
        print(message, flush=True)
        _write_cache_log(message)
        return df

    message = f"[cache_csv] MISSING {name}: path={path.resolve()}"
    print(message, flush=True)
    _write_cache_log(message)
    return pd.DataFrame(columns=columns or [])

CONCEPT_PATTERNS = {
    "attention": {"label": "Attention"},
    "transformer": {"label": "Transformer"},
    "language_model_frontier": {"label": "Language Model Frontier"},
    "graph_neural_network": {"label": "Graph Neural Network"},
    "reinforcement_learning": {"label": "Reinforcement Learning"},
    "computer_vision": {"label": "Computer Vision"},
    "diffusion": {"label": "Diffusion"},
    "prompting": {"label": "Prompting"},
}
CONCEPTS = list(CONCEPT_PATTERNS)


def _classify_topic(text: object) -> str:
    haystack = str(text).lower()
    for bucket, keywords in TOPIC_RULES:
        if any(k in haystack for k in keywords):
            return bucket
    return "Applied / Interdisciplinary AI"


def concept_label(concept_key: str) -> str:
    return CONCEPT_PATTERNS.get(
        concept_key,
        {"label": str(concept_key).replace("_", " ").title()},
    )["label"]


@functools.lru_cache(maxsize=1)
def yearly_counts() -> pd.DataFrame:
    return _cache_csv("yearly_counts.csv", ["year", "count"])


@functools.lru_cache(maxsize=1)
def bucket_year_counts() -> pd.DataFrame:
    return _cache_csv("bucket_year_counts.csv", ["year", "topic_bucket", "count"])


@functools.lru_cache(maxsize=1)
def topic_year_counts() -> pd.DataFrame:
    df = _cache_csv("topic_year_counts.csv", ["year", "primary_topic", "count", "topic_bucket"])
    if "topic_bucket" not in df and "primary_topic" in df:
        df["topic_bucket"] = df["primary_topic"].map(_classify_topic)
    return df


@functools.lru_cache(maxsize=1)
def topic_family_nodes() -> pd.DataFrame:
    return _cache_csv(
        "genome_nodes.csv",
        ["theme", "family", "count", "late_count", "growth_since_2000", "fwci_attention", "citation_velocity", "keywords", "x", "y"],
    )

def growth_impact_matrix() -> pd.DataFrame:
    """
    Topic-level growth-impact data aligned with the Tab 2 bubble swarm.
    Source of truth: www/data/subtopic_layout.json.
    """
    path = PROJECT_ROOT / "www" / "data" / "subtopic_layout.json"
    if not path.exists():
        return pd.DataFrame(columns=[
            "primary_topic",
            "topic_family",
            "paper_count",
            "growth_rate",
            "median_fwci",
            "mean_fwci",
            "citation_velocity",
            "mean_citation_velocity",
            "quadrant",
        ])

    df = pd.read_json(path)

    df = df.rename(
        columns={
            "topic": "primary_topic",
            "family": "topic_family",
            "paper_count_2025": "paper_count",
        }
    )

    if "citation_velocity" not in df.columns:
        if "mean_citation_velocity" in df.columns:
            df["citation_velocity"] = df["mean_citation_velocity"]
        elif "median_velocity" in df.columns:
            df["citation_velocity"] = df["median_velocity"]
        else:
            df["citation_velocity"] = 0

    if "mean_citation_velocity" not in df.columns:
        df["mean_citation_velocity"] = df["citation_velocity"]

    if "mean_fwci" not in df.columns:
        if "median_fwci" in df.columns:
            df["mean_fwci"] = df["median_fwci"]
        else:
            df["mean_fwci"] = 0

    if "median_fwci" not in df.columns:
        df["median_fwci"] = df["mean_fwci"]

    if "entropy_family" not in df.columns and "topic_family" in df.columns:
        df["entropy_family"] = df["topic_family"]

    return df

def theme_profile(theme_name: str) -> dict[str, object]:
    nodes = topic_family_nodes()
    if nodes.empty:
        return {
            "theme": theme_name,
            "family": "Topic family",
            "count": 0,
            "growth_since_2000": np.nan,
            "fwci_attention": 0,
            "citation_velocity": 0,
            "keywords": "",
            "papers": [],
            "related": [],
        }
    if theme_name not in set(nodes["theme"]):
        theme_name = str(nodes.sort_values("count", ascending=False).iloc[0]["theme"])
    node = nodes[nodes["theme"].eq(theme_name)].iloc[0]
    related = (
        nodes[nodes["family"].eq(node["family"]) & ~nodes["theme"].eq(theme_name)]
        .sort_values("count", ascending=False)["theme"]
        .head(5)
        .tolist()
    )
    return {
        "theme": theme_name,
        "family": node.get("family", "Topic family"),
        "count": float(node.get("count", 0)),
        "growth_since_2000": float(node.get("growth_since_2000", np.nan)),
        "fwci_attention": float(node.get("fwci_attention", 0)),
        "citation_velocity": float(node.get("citation_velocity", 0)),
        "keywords": str(node.get("keywords", "")),
        "papers": representative_papers(theme_name, family=str(node.get("family", ""))),
        "related": related,
    }


@functools.lru_cache(maxsize=1)
def paper_lookup() -> pd.DataFrame:
    out = _cache_csv(
        "paper_lookup.csv",
        ["title", "year", "fwci", "citation_velocity", "citation_count", "primary_topic", "topic_bucket"],
    )
    if out.empty:
        return out
    for col in ["year", "fwci", "citation_velocity", "citation_count"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "topic_bucket" not in out and "primary_topic" in out:
        out["topic_bucket"] = out["primary_topic"].map(_classify_topic)
    return out.drop_duplicates(subset=["title", "primary_topic"], keep="first")


def representative_papers(topic: str, *, family: str = "", limit: int = 5) -> list[dict[str, object]]:
    papers = paper_lookup()
    if papers.empty:
        return []
    sub = papers[papers["primary_topic"].fillna("").astype(str).eq(str(topic))].copy()
    if sub.empty and family and "topic_bucket" in papers:
        sub = papers[papers["topic_bucket"].fillna("").astype(str).eq(str(family))].copy()
    if sub.empty and topic:
        haystack = (
            papers.get("primary_topic", pd.Series("", index=papers.index)).fillna("").astype(str)
            + " "
            + papers.get("topics", pd.Series("", index=papers.index)).fillna("").astype(str)
        ).str.lower()
        sub = papers[haystack.str.contains(str(topic).lower(), regex=False, na=False)].copy()
    if sub.empty:
        return []
    sort_cols = [c for c in ["fwci", "citation_velocity", "citation_count"] if c in sub]
    if sort_cols:
        sub["_rank"] = sub[sort_cols].fillna(0).sum(axis=1)
        sub = sub.sort_values("_rank", ascending=False)
    keep = [c for c in ["title", "year", "fwci", "citation_velocity", "citation_count", "primary_topic"] if c in sub]
    return sub[keep].head(limit).to_dict("records")


@functools.lru_cache(maxsize=1)
def diversity_metrics() -> pd.DataFrame:
    return _cache_csv("diversity_metrics.csv", ["year", "entropy", "top5_share"])


@functools.lru_cache(maxsize=1)
def concept_year_counts() -> pd.DataFrame:
    return _cache_csv("concept_year_counts.csv", ["concept", "concept_key", "year", "sample_count", "share"])


@functools.lru_cache(maxsize=1)
def impact_topic_scatter() -> pd.DataFrame:
    return _cache_csv(
        "impact_topic_scatter.csv",
        ["primary_topic", "paper_count", "mean_fwci", "median_fwci", "mean_citation_velocity", "entropy_family"],
    )


@functools.lru_cache(maxsize=1)
def genome_fitness_matrix() -> pd.DataFrame:
    """Deprecated: old Tab 3 matrix removed."""
    return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def topic_prediction_summary() -> pd.DataFrame:
    return _cache_csv(
        "topic_prediction_summary.csv",
        ["primary_topic", "paper_count", "actual_high_visibility_share", "predicted_high_visibility_share", "median_probability"],
    )


@functools.lru_cache(maxsize=1)
def top_countries(limit: int = 5) -> pd.DataFrame:
    return _cache_csv("top_countries.csv", ["country", "papers"]).head(limit)


@functools.lru_cache(maxsize=1)
def country_topic_year_counts() -> pd.DataFrame:
    return _cache_csv(
        "country_topic_year.csv",
        ["country", "year", "topic_bucket", "count"],
    )


def concept_detail(concept_key: str) -> dict[str, object]:
    label = concept_label(concept_key)
    df = concept_year_counts()
    sub = df[df["concept_key"].eq(concept_key)].copy() if "concept_key" in df else pd.DataFrame()
    if sub.empty:
        return {"label": label, "first_year": None, "peak_year": "Not observed", "peak_share": 0.0, "papers": []}
    sub = sub.sort_values("year")
    observed = sub[sub["sample_count"].fillna(0) > 0]
    first_year = int(observed["year"].iloc[0]) if not observed.empty else None
    peak = sub.sort_values("share", ascending=False).iloc[0]
    return {
        "label": label,
        "first_year": first_year,
        "peak_year": int(peak["year"]),
        "peak_share": float(peak["share"]),
        "papers": [],
    }


def mutation_cloud(year: int, top_n: int = 24) -> pd.DataFrame:
    df = _cache_csv("mutation_cloud_terms.csv", ["year", "term", "count", "score", "x", "y", "size"])
    if df.empty:
        return pd.DataFrame(columns=["term", "count", "score", "x", "y", "size"])
    sub = df[df["year"].eq(int(year))].copy()
    if sub.empty:
        sub = df[df["year"].eq(int(df["year"].max()))].copy()
    sub = sub.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    if "x" not in sub or "y" not in sub:
        positions = [(0.0, 0.0)]
        for i in range(1, len(sub)):
            ring = 1 + (i - 1) // 8
            slot = (i - 1) % 8
            angle = slot * (2 * np.pi / 8) + ring * .28
            radius = .72 * ring
            positions.append((np.cos(angle) * radius, np.sin(angle) * radius))
        sub["x"] = [p[0] for p in positions]
        sub["y"] = [p[1] for p in positions]
    if "size" not in sub:
        score = pd.to_numeric(sub.get("score", 1), errors="coerce").fillna(0)
        denom = float(score.max()) if float(score.max()) > 0 else 1.0
        sub["size"] = 14 + 20 * score / denom
    return sub


def topic_heatmap_by_year(top_n: int = 18) -> pd.DataFrame:
    counts = topic_year_counts().copy()
    if counts.empty:
        return counts.assign(share=pd.Series(dtype=float), topic_order=pd.Series(dtype=int))
    totals = counts.groupby("primary_topic")["count"].sum().sort_values(ascending=False)
    keep = totals.head(top_n).index
    out = counts[counts["primary_topic"].isin(keep)].copy()
    year_totals = out.groupby("year")["count"].transform("sum").replace(0, np.nan)
    out["share"] = 100 * out["count"] / year_totals
    out["topic_order"] = out["primary_topic"].map({topic: i for i, topic in enumerate(keep)})
    return out.sort_values(["topic_order", "year"])


def topic_opportunity_matrix() -> pd.DataFrame:
    df = growth_impact_matrix().copy()
    if df.empty:
        return pd.DataFrame(columns=["primary_topic", "paper_count", "family", "paper_growth", "citation_growth", "quadrant", "median_fwci"])
    out = df.rename(columns={"growth": "paper_growth", "median_velocity": "citation_growth"}).copy()
    out["quadrant"] = out.get("quadrant", "Low Momentum")
    return out


def publication_channel_year_counts() -> pd.DataFrame:
    return _cache_csv("publication_channel_year_counts.csv", ["year", "channel", "count"])


def rising_primary_topics(top_n: int = 8) -> pd.DataFrame:
    return _cache_csv("rising_primary_topics.csv", ["primary_topic", "early", "late", "growth", "family"]).head(top_n)



@functools.lru_cache(maxsize=1)
def open_access_period_summary() -> pd.DataFrame:
    return _cache_csv(
        "oa_period_status.csv",
        ["period", "access_status", "count"],
    )

def discovery_metrics() -> dict[str, pd.DataFrame]:
    return {
        "collaboration": _cache_csv("discovery_collaboration.csv"),
        "collaboration_impact": _cache_csv("discovery_collaboration_impact.csv"),
        "oa": _cache_csv("discovery_oa.csv"),
        "venue": _cache_csv("discovery_venue.csv"),
        "international": _cache_csv("discovery_international.csv"),
        "search_topic": _cache_csv("discovery_search_topic.csv"),
    }

@functools.lru_cache(maxsize=1)
def rising_fading_terms() -> pd.DataFrame:
    return _cache_csv(
        "rising_fading_terms.csv",
        [
            "scope",
            "term",
            "early_count",
            "late_count",
            "early_share_per_1000",
            "late_share_per_1000",
            "delta_share_per_1000",
            "growth_ratio",
            "direction",
        ],
    )


@functools.lru_cache(maxsize=1)
def rising_terms() -> pd.DataFrame:
    return _cache_csv(
        "rising_terms.csv",
        [
            "scope",
            "term",
            "early_count",
            "late_count",
            "early_share_per_1000",
            "late_share_per_1000",
            "delta_share_per_1000",
            "growth_ratio",
            "direction",
        ],
    )


@functools.lru_cache(maxsize=1)
def fading_terms() -> pd.DataFrame:
    return _cache_csv(
        "fading_terms.csv",
        [
            "scope",
            "term",
            "early_count",
            "late_count",
            "early_share_per_1000",
            "late_share_per_1000",
            "delta_share_per_1000",
            "growth_ratio",
            "direction",
        ],
    )


@functools.lru_cache(maxsize=1)
def wordcloud_terms() -> pd.DataFrame:
    return _cache_csv(
        "wordcloud_terms.csv",
        [
            "scope",
            "term",
            "early_count",
            "late_count",
            "early_share_per_1000",
            "late_share_per_1000",
            "delta_share_per_1000",
            "growth_ratio",
            "direction",
            "score",
        ],
    )


@functools.lru_cache(maxsize=1)
def institution_type_year_summary() -> pd.DataFrame:
    """Return four-group institution participation by year.

    This prefers a precomputed institution_type_year_summary.csv.
    If it is missing, it builds the four groups from institution_type_year_summary.csv.

    Counts are participation counts, not exclusive shares:
    one paper can involve more than one institution group.
    """
    grouped = _cache_csv(
        "institution_type_year_summary.csv",
        [
            "year",
            "institution_type",
            "paper_institution_rows",
            "unique_papers",
            "unique_institutions",
            "median_fwci",
            "mean_fwci",
            "median_citation_velocity",
            "mean_citation_velocity",
        ],
    )
    if not grouped.empty:
        return grouped

    base = _cache_csv(
        "institution_type_year_summary.csv",
        [
            "year",
            "institution_type",
            "paper_institution_rows",
            "unique_papers",
            "unique_institutions",
            "median_fwci",
            "mean_fwci",
            "median_citation_velocity",
            "mean_citation_velocity",
        ],
    )
    if base.empty:
        return grouped

    group_map = {
        "University": "University",
        "Business": "Business",
        "Government": "Public sector",
        "Healthcare": "Public sector",
        "Nonprofit": "Nonprofit / Other",
        "Other": "Nonprofit / Other",
        "Unknown": "Nonprofit / Other",
    }
    out = base.copy()
    out["institution_type"] = out["institution_type"].map(group_map).fillna("Nonprofit / Other")

    agg_map = {}
    for col in ["paper_institution_rows", "unique_papers", "unique_institutions"]:
        if col in out:
            agg_map[col] = "sum"
    for col in ["median_fwci", "mean_fwci", "median_citation_velocity", "mean_citation_velocity"]:
        if col in out:
            agg_map[col] = "median" if col.startswith("median") else "mean"

    return (
        out.groupby(["year", "institution_type"], as_index=False)
        .agg(agg_map)
        .sort_values(["year", "institution_type"])
    )
