"""Build core CSV caches for the AI research dashboard.

Run from project root, for example:
python src/build_core_dashboard_cache.py \
  --input Dataset/ai_works_merge_2000_2009.csv Dataset/ai_works_merge_2010_2019.csv Dataset/ai_works_merge_2020_2025.csv \
  --output-dir Dataset/dashboard_cache \
  --min-year 2000 \
  --max-year 2025

Outputs:
- yearly_counts.csv
- bucket_year_counts.csv
- topic_year_counts.csv
- diversity_metrics.csv
- impact_topic_scatter.csv
- top_countries.csv
- paper_lookup.csv
- country_topic_year.csv
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TOPIC_RULES = [
    ("NLP", [
        "natural language", "language", "nlp", "text", "speech", "semantic",
        "sentiment", "dialogue", "topic modeling", "translation",
    ]),
    ("Core ML / Deep Learning", [
        "machine learning", "deep learning", "neural", "classification",
        "clustering", "graph neural", "adversarial", "representation learning",
    ]),
    ("ML Theory & Optimization", [
        "optimization", "bayesian", "probabilistic", "algorithm", "theory",
        "causal", "quantum", "data compression",
    ]),
    ("Robotics", [
        "robot", "robotics", "control", "tracking", "sensor", "autonomous",
        "planning", "navigation", "motion planning",
    ]),
    ("Healthcare AI", [
        "health", "healthcare", "medical", "clinical", "cancer", "disease",
        "diagnosis", "radiology", "patient", "neuroscience",
    ]),
    ("AI Ethics & Fairness", [
        "privacy", "fairness", "ethics", "bias", "explainable", "xai",
        "trust", "law", "safety", "intellectual property",
    ]),
    ("Reinforcement Learning", [
        "reinforcement", "agent", "policy", "reward", "multi-agent",
        "negotiation",
    ]),
]

COUNTRY_NAME_MAP = {
    "US": "United States",
    "USA": "United States",
    "United States of America": "United States",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "Great Britain": "United Kingdom",
    "CN": "China",
    "DE": "Germany",
    "JP": "Japan",
    "FR": "France",
    "IT": "Italy",
    "CA": "Canada",
    "IN": "India",
    "ID": "Indonesia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core dashboard cache CSV files from paper-level AI datasets.")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV files or glob patterns.")
    parser.add_argument("--output-dir", default="Dataset/dashboard_cache")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--top-countries", type=int, default=20)
    parser.add_argument("--paper-limit-per-topic", type=int, default=80)
    parser.add_argument("--paper-lookup-total-limit", type=int, default=20_000)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path().glob(pattern))
        elif path.exists():
            files.append(path)
    files = sorted(set(p.resolve() for p in files if p.exists() and p.suffix.lower() == ".csv"))
    if not files:
        raise FileNotFoundError("No CSV files found from --input.")
    return files


def find_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def choose_columns(columns: Iterable[str]) -> dict[str, str | None]:
    cols = list(columns)
    return {
        "year": find_col(cols, ["year", "publication_year", "pub_year"]),
        "title": find_col(cols, ["title", "display_name", "work_title"]),
        "topic": find_col(cols, ["primary_topic", "topic", "topic_label"]),
        "topic_bucket": find_col(cols, ["topic_bucket", "family", "primary_subfield", "primary_field"]),
        "subfield": find_col(cols, ["primary_subfield", "subfield"]),
        "field": find_col(cols, ["primary_field", "field"]),
        "domain": find_col(cols, ["primary_domain", "domain"]),
        "topics": find_col(cols, ["topics", "topic_list"]),
        "keywords": find_col(cols, ["keywords", "concepts"]),
        "fwci": find_col(cols, ["fwci", "field_weighted_citation_impact"]),
        "citation_count": find_col(cols, ["citation_count", "cited_by_count", "citations"]),
        "citation_velocity": find_col(cols, ["citation_velocity", "citations_per_year"]),
        "country": find_col(cols, ["country", "countries", "authorship_countries", "country_code", "country_codes"]),
        "venue": find_col(cols, ["venue", "venue_source", "source", "source_display_name", "journal", "conference"]),
        "id": find_col(cols, ["id", "work_id", "openalex_id", "doi"]),
    }


def normalize_topic(value: Any) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else "Unknown topic"


def classify_bucket(topic: Any, explicit_bucket: Any = None) -> str:
    bucket = str(explicit_bucket or "").strip()
    if bucket and bucket.lower() not in {"nan", "none", "null", "unknown"}:
        return bucket
    text = str(topic or "").lower()
    for label, keywords in TOPIC_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    return "Applied / Interdisciplinary AI"


def safe_parse(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def normalize_country(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return COUNTRY_NAME_MAP.get(text, text)


def extract_countries(value: Any) -> list[str]:
    parsed = safe_parse(value)
    countries: list[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                country = item.get("display_name") or item.get("name") or item.get("country") or item.get("country_code")
            else:
                country = item
            norm = normalize_country(country)
            if norm:
                countries.append(norm)
    elif isinstance(parsed, dict):
        for key in ["display_name", "name", "country", "country_code"]:
            norm = normalize_country(parsed.get(key))
            if norm:
                countries.append(norm)
        if not countries:
            for item in parsed.values():
                norm = normalize_country(item)
                if norm:
                    countries.append(norm)
    elif isinstance(parsed, str):
        # Handles "China; United States", "China|United States", "China, United States".
        for part in re.split(r"\s*[;|,]\s*", parsed):
            norm = normalize_country(part)
            if norm:
                countries.append(norm)

    # Remove duplicates inside one paper so country counts mean paper participation.
    return list(dict.fromkeys(countries))


def entropy_from_counts(counts: Iterable[float]) -> float:
    arr = np.array([float(x) for x in counts if float(x) > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())


def numeric_series(df: pd.DataFrame, col: str | None, default: float = np.nan) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def string_series(df: pd.DataFrame, col: str | None, default: str = "") -> pd.Series:
    if col and col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def first_country_text(countries: list[str]) -> str:
    return countries[0] if countries else ""


def build_cache(args: argparse.Namespace) -> None:
    files = expand_inputs(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly_counts: Counter[int] = Counter()
    bucket_year_counts: Counter[tuple[int, str]] = Counter()
    topic_year_counts: Counter[tuple[int, str, str]] = Counter()
    country_counts: Counter[str] = Counter()
    country_topic_year_counts: Counter[tuple[str, int, str]] = Counter()

    topic_metric_rows: list[pd.DataFrame] = []
    paper_candidates: list[pd.DataFrame] = []

    total_rows = 0
    usable_rows = 0

    cols_map: dict[str, str | None] | None = None
    usecols: list[str] | None = None

    for file in files:
        print(f"Reading {file}", flush=True)
        sample = pd.read_csv(file, nrows=5, low_memory=False)
        current_cols = choose_columns(sample.columns)
        if current_cols["year"] is None:
            raise ValueError(f"No year column found in {file}. Columns: {list(sample.columns)[:40]}")
        if current_cols["topic"] is None:
            raise ValueError(f"No primary_topic/topic column found in {file}. Columns: {list(sample.columns)[:40]}")

        cols_map = current_cols
        usecols = [c for c in dict.fromkeys(current_cols.values()) if c]

        for i, chunk in enumerate(pd.read_csv(file, usecols=usecols, chunksize=args.chunksize, low_memory=False)):
            total_rows += len(chunk)
            year = numeric_series(chunk, cols_map["year"])
            chunk = chunk.assign(_year=year)
            chunk = chunk.dropna(subset=["_year"])
            chunk["_year"] = chunk["_year"].astype(int)
            chunk = chunk[(chunk["_year"] >= args.min_year) & (chunk["_year"] <= args.max_year)].copy()
            if chunk.empty:
                continue
            usable_rows += len(chunk)

            topic = string_series(chunk, cols_map["topic"]).map(normalize_topic)
            explicit_bucket = string_series(chunk, cols_map["topic_bucket"]) if cols_map["topic_bucket"] else pd.Series("", index=chunk.index)
            bucket = [classify_bucket(t, b) for t, b in zip(topic, explicit_bucket)]
            chunk["_primary_topic"] = topic
            chunk["_topic_bucket"] = bucket

            # Counts.
            yearly_counts.update(chunk["_year"].tolist())
            bucket_year_counts.update(zip(chunk["_year"], chunk["_topic_bucket"]))
            topic_year_counts.update(zip(chunk["_year"], chunk["_primary_topic"], chunk["_topic_bucket"]))

            # Countries and country-topic-year.
            if cols_map["country"]:
                country_values = string_series(chunk, cols_map["country"])
                for y, b, raw_countries in zip(chunk["_year"], chunk["_topic_bucket"], country_values):
                    countries = extract_countries(raw_countries)
                    for country in countries:
                        country_counts[country] += 1
                        country_topic_year_counts[(country, int(y), b)] += 1

            # Topic impact metrics.
            metrics = pd.DataFrame({
                "primary_topic": chunk["_primary_topic"],
                "topic_bucket": chunk["_topic_bucket"],
                "year": chunk["_year"],
                "fwci": numeric_series(chunk, cols_map["fwci"], default=np.nan),
                "citation_count": numeric_series(chunk, cols_map["citation_count"], default=np.nan),
                "citation_velocity": numeric_series(chunk, cols_map["citation_velocity"], default=np.nan),
            })
            topic_metric_rows.append(metrics)

            # Paper lookup candidates. Keep only useful candidates per chunk/topic.
            if cols_map["title"]:
                citation_count = numeric_series(chunk, cols_map["citation_count"], default=0).fillna(0)
                fwci = numeric_series(chunk, cols_map["fwci"], default=0).fillna(0)
                citation_velocity = numeric_series(chunk, cols_map["citation_velocity"], default=0).fillna(0)
                paper_df = pd.DataFrame({
                    "id": string_series(chunk, cols_map["id"]) if cols_map["id"] else "",
                    "title": string_series(chunk, cols_map["title"]),
                    "year": chunk["_year"],
                    "fwci": fwci,
                    "citation_velocity": citation_velocity,
                    "citation_count": citation_count,
                    "primary_topic": chunk["_primary_topic"],
                    "topic_bucket": chunk["_topic_bucket"],
                    "country": "",
                    "venue": string_series(chunk, cols_map["venue"]) if cols_map["venue"] else "",
                })
                if cols_map["country"]:
                    paper_df["country"] = string_series(chunk, cols_map["country"]).map(lambda x: first_country_text(extract_countries(x)))
                paper_df["_rank"] = np.log1p(citation_count) + 2.0 * np.log1p(fwci.clip(lower=0)) + np.log1p(citation_velocity.clip(lower=0))
                paper_df = paper_df[paper_df["title"].str.len() >= 18].copy()
                if not paper_df.empty:
                    paper_df = (
                        paper_df.sort_values("_rank", ascending=False)
                        .groupby("primary_topic", group_keys=False)
                        .head(args.paper_limit_per_topic)
                    )
                    paper_candidates.append(paper_df)

            if (i + 1) % 10 == 0:
                print(f"  {file.name}: chunk {i + 1}, usable rows so far {usable_rows:,}", flush=True)

    print(f"Total rows read: {total_rows:,}; usable rows {usable_rows:,}", flush=True)

    # yearly_counts.csv
    yearly_df = pd.DataFrame([
        {"year": year, "count": count}
        for year, count in sorted(yearly_counts.items())
    ])
    yearly_df.to_csv(output_dir / "yearly_counts.csv", index=False)

    # bucket_year_counts.csv
    bucket_df = pd.DataFrame([
        {"year": year, "topic_bucket": bucket, "count": count}
        for (year, bucket), count in sorted(bucket_year_counts.items())
    ])
    bucket_df.to_csv(output_dir / "bucket_year_counts.csv", index=False)

    # topic_year_counts.csv
    topic_year_df = pd.DataFrame([
        {"year": year, "primary_topic": topic, "topic_bucket": bucket, "count": count}
        for (year, topic, bucket), count in sorted(topic_year_counts.items())
    ])
    topic_year_df.to_csv(output_dir / "topic_year_counts.csv", index=False)

    # diversity_metrics.csv: entropy over topic buckets + top-5 topic share.
    diversity_rows = []
    for year, sub in topic_year_df.groupby("year"):
        bucket_counts = sub.groupby("topic_bucket")["count"].sum().sort_values(ascending=False)
        topic_counts = sub.groupby("primary_topic")["count"].sum().sort_values(ascending=False)
        total = float(topic_counts.sum()) if not topic_counts.empty else 0.0
        top5_share = 100.0 * float(topic_counts.head(5).sum()) / max(total, 1.0)
        diversity_rows.append({
            "year": int(year),
            "entropy": entropy_from_counts(bucket_counts.values),
            "top5_share": top5_share,
        })
    pd.DataFrame(diversity_rows).sort_values("year").to_csv(output_dir / "diversity_metrics.csv", index=False)

    # impact_topic_scatter.csv
    if topic_metric_rows:
        metrics_all = pd.concat(topic_metric_rows, ignore_index=True)
        metrics_all["fwci"] = pd.to_numeric(metrics_all["fwci"], errors="coerce")
        metrics_all["citation_count"] = pd.to_numeric(metrics_all["citation_count"], errors="coerce")
        metrics_all["citation_velocity"] = pd.to_numeric(metrics_all["citation_velocity"], errors="coerce")

        topic_totals = (
            metrics_all.groupby("primary_topic", as_index=False)
            .agg(
                paper_count=("primary_topic", "size"),
                mean_fwci=("fwci", "mean"),
                median_fwci=("fwci", "median"),
                mean_citation_velocity=("citation_velocity", "mean"),
                median_velocity=("citation_velocity", "median"),
                mean_citation_count=("citation_count", "mean"),
                topic_bucket=("topic_bucket", lambda s: s.mode().iloc[0] if not s.mode().empty else "Applied / Interdisciplinary AI"),
            )
        )

        # Growth = late-period count / early-period count. Uses 2020-2025 vs 2000-2005 where available.
        early = metrics_all[metrics_all["year"].between(args.min_year, min(args.min_year + 5, args.max_year))]
        late = metrics_all[metrics_all["year"].between(max(args.max_year - 5, args.min_year), args.max_year)]
        early_counts = early.groupby("primary_topic").size()
        late_counts = late.groupby("primary_topic").size()
        topic_totals["early_count"] = topic_totals["primary_topic"].map(early_counts).fillna(0).astype(float)
        topic_totals["late_count"] = topic_totals["primary_topic"].map(late_counts).fillna(0).astype(float)
        topic_totals["growth"] = (topic_totals["late_count"] + 1.0) / (topic_totals["early_count"] + 1.0)
        topic_totals["entropy_family"] = topic_totals["topic_bucket"]
        topic_totals.to_csv(output_dir / "impact_topic_scatter.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "primary_topic", "paper_count", "mean_fwci", "median_fwci",
            "mean_citation_velocity", "median_velocity", "growth", "entropy_family",
        ]).to_csv(output_dir / "impact_topic_scatter.csv", index=False)

    # top_countries.csv and country_topic_year.csv
    top_country_df = pd.DataFrame([
        {"country": country, "papers": count}
        for country, count in country_counts.most_common(args.top_countries)
    ])
    top_country_df.to_csv(output_dir / "top_countries.csv", index=False)

    country_topic_year_df = pd.DataFrame([
        {"country": country, "year": year, "topic_bucket": bucket, "count": count}
        for (country, year, bucket), count in sorted(country_topic_year_counts.items())
    ])
    country_topic_year_df.to_csv(output_dir / "country_topic_year.csv", index=False)

    # paper_lookup.csv
    if paper_candidates:
        paper_lookup = pd.concat(paper_candidates, ignore_index=True)
        paper_lookup = paper_lookup.drop_duplicates(subset=["title", "primary_topic"], keep="first")
        paper_lookup = (
            paper_lookup.sort_values("_rank", ascending=False)
            .groupby("primary_topic", group_keys=False)
            .head(args.paper_limit_per_topic)
            .sort_values("_rank", ascending=False)
            .head(args.paper_lookup_total_limit)
            .drop(columns=["_rank"])
        )
    else:
        paper_lookup = pd.DataFrame(columns=[
            "id", "title", "year", "fwci", "citation_velocity", "citation_count",
            "primary_topic", "topic_bucket", "country", "venue",
        ])
    paper_lookup.to_csv(output_dir / "paper_lookup.csv", index=False)

    print("\nWrote core dashboard cache:", flush=True)
    for name in [
        "yearly_counts.csv",
        "bucket_year_counts.csv",
        "topic_year_counts.csv",
        "diversity_metrics.csv",
        "impact_topic_scatter.csv",
        "top_countries.csv",
        "country_topic_year.csv",
        "paper_lookup.csv",
    ]:
        path = output_dir / name
        print(f"- {path} ({path.stat().st_size / 1024:,.1f} KB)", flush=True)


def main() -> None:
    args = parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
