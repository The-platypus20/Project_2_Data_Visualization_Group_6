from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from . import narrative_data as nd
except ImportError:
    from . import narrative_data as nd


def _matches(label: str, terms: list[str]) -> bool:
    text = str(label).lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) for term in terms)


def main() -> None:
    cache_dir = Path("Dataset/dashboard_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    topics = nd.topic_year_counts().copy()
    years = sorted(topics["year"].dropna().astype(int).unique())
    totals = topics.groupby("year")["count"].sum()

    concept_rows = []
    for key, spec in nd.CONCEPT_PATTERNS.items():
        mask = topics["primary_topic"].map(lambda value: _matches(value, spec["terms"]))
        by_year = topics[mask].groupby("year")["count"].sum()
        for year in years:
            count = float(by_year.get(year, 0))
            concept_rows.append({
                "concept": nd.concept_label(key),
                "concept_key": key,
                "year": int(year),
                "sample_count": count,
                "share": 100 * count / max(float(totals.get(year, 0)), 1.0),
            })
    pd.DataFrame(concept_rows).to_csv(cache_dir / "concept_year_counts.csv", index=False)

    cloud_rows = []
    for year, sub in topics.groupby("year"):
        total = max(float(sub["count"].sum()), 1.0)
        previous = topics[topics["year"] < year].groupby("primary_topic")["count"].sum()
        previous_total = max(float(previous.sum()), 1.0)
        for _, row in sub.sort_values("count", ascending=False).head(35).iterrows():
            term = str(row["primary_topic"])[:42]
            count = float(row["count"])
            score = count / total - float(previous.get(row["primary_topic"], 0)) / previous_total
            cloud_rows.append({"year": int(year), "term": term, "count": count, "score": max(score, 0.0001)})
    cloud = pd.DataFrame(cloud_rows)
    cloud["rank"] = cloud.groupby("year")["score"].rank(ascending=False, method="first")
    cloud = cloud[cloud["rank"] <= 24].drop(columns=["rank"])
    cloud.to_csv(cache_dir / "mutation_cloud_terms.csv", index=False)
    print(f"concept_year_counts: {len(concept_rows):,} rows")
    print(f"mutation_cloud_terms: {len(cloud):,} rows")


if __name__ == "__main__":
    main()

