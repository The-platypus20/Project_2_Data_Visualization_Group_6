from __future__ import annotations

import time
from pathlib import Path

try:
    from . import narrative_data as nd
except ImportError:
    from . import narrative_data as nd


def _save(df, name: str, cache_dir: Path) -> None:
    start = time.time()
    path = cache_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {len(df):,} rows in {time.time() - start:.1f}s")


def main() -> None:
    cache_dir = Path("Dataset/dashboard_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("Building dashboard caches for 2000-2025...")

    jobs = {
        "yearly_counts": nd.yearly_counts,
        "bucket_year_counts": nd.bucket_year_counts,
        "topic_year_counts": nd.topic_year_counts,
        "diversity_metrics": nd.diversity_metrics,
        "publication_channel_year_counts": nd.publication_channel_year_counts,
        "concept_year_counts": nd.concept_year_counts,
        "genome_fitness_matrix": nd.genome_fitness_matrix,
        "rising_primary_topics": lambda: nd.rising_primary_topics(20),
        "top_countries": lambda: nd.top_countries(10),
        "impact_topic_scatter": nd.impact_topic_scatter,
        "topic_prediction_summary": nd.topic_prediction_summary,
    }
    for name, fn in jobs.items():
        job_start = time.time()
        df = fn()
        _save(df, name, cache_dir)
        print(f"  compute: {time.time() - job_start:.1f}s")

    metrics = nd.discovery_metrics()
    name_map = {
        "collaboration": "discovery_collaboration",
        "collaboration_impact": "discovery_collaboration_impact",
        "oa": "discovery_oa",
        "venue": "discovery_venue",
        "international": "discovery_international",
        "search_topic": "discovery_search_topic",
    }
    for key, df in metrics.items():
        _save(df, name_map[key], cache_dir)

    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

