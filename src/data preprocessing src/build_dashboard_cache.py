"""Build precomputed dashboard cache tables.

Run from project root:
    python src/build_dashboard_cache.py

The dashboard reads these small CSVs from Dataset/dashboard_cache/ so Shiny
does not scan the 1M+ row field CSVs or recluster conference text on every run.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

os.environ["DASHBOARD_CACHE_BYPASS"] = "1"

try:
    from src import narrative_data as nd
except ImportError:
    from . import narrative_data as nd


def _save(df: pd.DataFrame, name: str) -> None:
    path = nd.CACHE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{name}: {df.shape}", flush=True)


def _mutation_cloud_all_years() -> pd.DataFrame:
    topic_counts = nd.topic_year_counts().copy()
    rows = []
    for year in sorted(topic_counts["year"].dropna().astype(int).unique()):
        current = topic_counts[topic_counts["year"].eq(year)].copy()
        previous = topic_counts[topic_counts["year"].lt(year)].copy()
        if current.empty or previous.empty:
            continue
        current_total = max(current["count"].sum(), 1)
        previous_totals = previous.groupby("primary_topic")["count"].sum()
        previous_total = max(previous_totals.sum(), 1)
        current["current_share"] = current["count"] / current_total
        current["previous_share"] = current["primary_topic"].map(previous_totals).fillna(0) / previous_total
        current["score"] = current["current_share"] - current["previous_share"]
        out = current[current["score"] > 0].sort_values("score", ascending=False).head(22).reset_index(drop=True)
        if out.empty:
            continue
        positions = [(0.0, 0.0)]
        import numpy as np
        for i in range(1, len(out)):
            ring = 1 + (i - 1) // 8
            slot = (i - 1) % 8
            angle = slot * (2 * np.pi / 8) + ring * .22
            radius = .72 * ring
            positions.append((np.cos(angle) * radius, np.sin(angle) * radius))
        rows.append(pd.DataFrame({
            "year": year,
            "term": out["primary_topic"],
            "count": out["count"],
            "score": out["score"],
            "x": [p[0] for p in positions],
            "y": [p[1] for p in positions],
            "size": 14 + 20 * (out["score"] / out["score"].max()),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _concept_year_counts_fast() -> pd.DataFrame:
    df = nd.papers()
    text = (
        df["title"].fillna("").astype(str)
        + " ; "
        + df["keywords"].fillna("").astype(str)
        + " ; "
        + df["primary_topic"].fillna("").astype(str)
        + " ; "
        + df["topics"].fillna("").astype(str)
    ).str.lower()
    totals = df.groupby("year").size()
    rows = []
    for concept in nd.CONCEPTS:
        mask = text.str.contains(nd._concept_regex(concept), regex=True, na=False)
        by_year = df[mask].groupby("year").size()
        for year, total in totals.items():
            sample_count = float(by_year.get(year, 0))
            rows.append({
                "concept": nd.concept_label(concept),
                "concept_key": concept,
                "year": int(year),
                "sample_count": sample_count,
                "share": 100 * sample_count / max(float(total), 1.0),
            })
    return pd.DataFrame(rows)


def _conference_cluster_terms() -> pd.DataFrame:
    clusters = nd.conference_clusters(7)
    rows = []
    for cluster, terms in clusters["top_terms"].items():
        for rank, term in enumerate(terms, start=1):
            rows.append({"cluster": int(cluster), "rank": rank, "term": term})
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    nd.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _save(nd.yearly_counts(), "yearly_counts.csv")
    _save(nd.bucket_year_counts(), "bucket_year_counts.csv")
    _save(nd.topic_year_counts(), "topic_year_counts.csv")
    _save(nd.genome_nodes(), "genome_nodes.csv")
    _save(nd.diversity_metrics(), "diversity_metrics.csv")
    _save(nd.publication_channel_year_counts(), "publication_channel_year_counts.csv")
    _save(_concept_year_counts_fast(), "concept_year_counts.csv")
    _save(_mutation_cloud_all_years(), "mutation_cloud_terms.csv")

    clusters = nd.conference_clusters(7)
    _save(clusters["coords"], "conference_cluster_coords.csv")
    _save(_conference_cluster_terms(), "conference_cluster_terms.csv")
    _save(nd.rising_primary_topics(12), "rising_primary_topics.csv")
    _save(nd.top_countries(10), "top_countries.csv")
    _save(nd.frontier_echo(), "frontier_echo.csv")

    discovery = nd.discovery_metrics()
    _save(discovery["collaboration"], "discovery_collaboration.csv")
    _save(discovery["collaboration_impact"], "discovery_collaboration_impact.csv")
    _save(discovery["oa"], "discovery_oa.csv")
    _save(discovery["venue"], "discovery_venue.csv")
    _save(discovery["international"], "discovery_international.csv")
    _save(discovery["search_topic"], "discovery_search_topic.csv")

    print(f"dashboard cache built in {time.time() - start:.1f}s at {nd.CACHE_DIR}")


if __name__ == "__main__":
    main()

