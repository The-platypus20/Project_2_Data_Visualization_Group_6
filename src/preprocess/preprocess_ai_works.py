"""Preprocess OpenAlex AI works for the Project 2 dashboard.

Purpose
-------
Create a stable, cleaned input layer before building dashboard caches.
The script handles schema normalization, duplicate removal, missing values,
country expansion, topic-family fallback, and a compact data-quality report.

Typical use from project root
-----------------------------
python src/preprocess_ai_works.py \
  --input Dataset/ai_works_merge_2000_2009.csv Dataset/ai_works_merge_2010_2019.csv Dataset/ai_works_merge_2020_2025.csv \
  --output Dataset/clean/ai_works_clean_2000_2025.csv \
  --country-output Dataset/clean/ai_work_country_long_2000_2025.csv \
  --report-dir Dataset/clean/preprocess_report \
  --min-year 2000 \
  --max-year 2025

Outputs
-------
1. Clean work-level table.
2. Country-long table for top country and country-topic aggregates.
3. Missingness profile CSV.
4. Deduplication summary CSV.
5. Markdown quality report.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "id": ["id", "work_id", "openalex_id", "doi"],
    "doi": ["doi"],
    "title": ["title", "display_name", "work_title"],
    "year": ["year", "publication_year", "pub_year"],
    "primary_topic": ["primary_topic", "topic", "topic_label"],
    "topic_bucket": ["topic_bucket", "family", "topic_family", "bucket"],
    "primary_subfield": ["primary_subfield", "subfield"],
    "primary_field": ["primary_field", "field"],
    "primary_domain": ["primary_domain", "domain"],
    "topics": ["topics", "topic_list", "concepts"],
    "keywords": ["keywords", "keyword", "keyword_list"],
    "abstract": ["abstract", "abstract_text"],
    "countries": ["countries", "authorship_countries", "country_list"],
    "country": ["country", "primary_country"],
    "institutions": ["institutions", "authorship_institutions", "institution_names"],
    "authorships": ["authorships", "raw_authorships"],
    "venue": ["venue", "venue_name", "source", "source_display_name", "journal", "conference"],
    "venue_type": ["venue_type", "source_type", "publication_type", "type"],
    "fwci": ["fwci", "field_weighted_citation_impact"],
    "citation_count": ["citation_count", "cited_by_count", "citations"],
    "citation_velocity": ["citation_velocity", "citations_per_year"],
    "referenced_works_count": ["referenced_works_count", "reference_count", "references_count"],
    "author_count": ["author_count", "authors_count"],
    "institution_count": ["institution_count", "institutions_count"],
    "country_count": ["country_count", "countries_count"],
    "is_oa": ["is_oa", "open_access", "oa", "open_access_is_oa"],
}

OUTPUT_COLUMNS = [
    "id", "doi", "title", "title_clean", "year",
    "primary_topic", "topic_bucket", "primary_subfield", "primary_field", "primary_domain",
    "topics", "keywords", "abstract", "countries", "country", "institutions", "authorships",
    "venue", "venue_type", "fwci", "citation_count", "citation_velocity",
    "referenced_works_count", "author_count", "institution_count", "country_count", "is_oa",
    "paper_age", "impact_score", "high_impact_label", "source_file",
]

NUMERIC_COLUMNS = [
    "year", "fwci", "citation_count", "citation_velocity", "referenced_works_count",
    "author_count", "institution_count", "country_count",
]

TOPIC_RULES = [
    ("NLP", ["natural language", "language", "nlp", "text", "speech", "semantic", "sentiment", "dialogue", "translation"]),
    ("Core ML / Deep Learning", ["machine learning", "deep learning", "neural", "classification", "clustering", "graph neural", "adversarial", "representation"]),
    ("ML Theory & Optimization", ["optimization", "bayesian", "probabilistic", "algorithm", "theory", "causal", "cryptography", "security"]),
    ("Robotics", ["robot", "robotics", "control", "tracking", "sensor", "autonomous", "planning", "navigation"]),
    ("Healthcare AI", ["health", "healthcare", "medical", "clinical", "cancer", "disease", "diagnosis", "patient", "radiology"]),
    ("AI Ethics & Fairness", ["privacy", "fairness", "ethics", "bias", "explainable", "xai", "law", "trust", "responsible"]),
    ("Reinforcement Learning", ["reinforcement", "policy", "reward", "agent", "multi-agent", "decision making"]),
]

COUNTRY_FIXES = {
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "peoples republic of china": "China",
    "people's republic of china": "China",
    "pr china": "China",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
}

# ---------------------------------------------------------------------------
# Basic parsers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean OpenAlex AI works before dashboard cache generation.")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV/Parquet files. Globs are supported.")
    parser.add_argument("--output", default="Dataset/clean/ai_works_clean_2000_2025.csv", help="Clean work-level output path.")
    parser.add_argument("--country-output", default="Dataset/clean/ai_work_country_long_2000_2025.csv", help="Country-long output path.")
    parser.add_argument("--report-dir", default="Dataset/clean/preprocess_report", help="Directory for data-quality reports.")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--output-format", choices=["csv", "parquet"], default=None, help="Infer from --output if omitted.")
    parser.add_argument("--keep-unknown-topic", action="store_true", help="Keep rows with missing primary_topic as Unknown topic.")
    parser.add_argument("--drop-missing-title", action="store_true", help="Drop rows with missing title. Default keeps them but excludes them from paper lookup later.")
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path().glob(pattern))
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    files = sorted(set(p.resolve() for p in files if p.exists()))
    if not files:
        raise FileNotFoundError("No input files found. Check --input paths/globs.")
    return files


def read_sample(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(5)
    return pd.read_csv(path, nrows=5, low_memory=False)


def iter_file(path: Path, usecols: list[str] | None, chunksize: int):
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=usecols)
        yield df
    elif suffix == ".csv":
        yield from pd.read_csv(path, usecols=lambda c: usecols is None or c in usecols, chunksize=chunksize, low_memory=False)
    else:
        raise ValueError(f"Unsupported input type: {path}")


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def build_schema(columns: Iterable[str]) -> dict[str, str | None]:
    return {out_col: find_column(columns, candidates) for out_col, candidates in COLUMN_CANDIDATES.items()}


def safe_parse(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def stringify(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def clean_title(value: Any) -> str:
    text = stringify(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: Any) -> str:
    text = stringify(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_topic_bucket(row: pd.Series) -> str:
    existing = normalize_label(row.get("topic_bucket"))
    if existing and existing.lower() not in {"nan", "none", "unknown"}:
        return existing
    haystack = " ".join(
        normalize_label(row.get(col)).lower()
        for col in ["primary_topic", "primary_subfield", "primary_field", "primary_domain", "topics", "keywords"]
    )
    for bucket, terms in TOPIC_RULES:
        if any(term in haystack for term in terms):
            return bucket
    return "Applied / Interdisciplinary AI"


def parse_countries(value: Any) -> list[str]:
    parsed = safe_parse(value)
    countries: list[str] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                name = item.get("display_name") or item.get("name") or item.get("country") or item.get("country_code")
            else:
                name = item
            if name is not None:
                countries.append(str(name))
    elif isinstance(parsed, dict):
        for key in ["display_name", "name", "country", "country_code"]:
            if parsed.get(key):
                countries.append(str(parsed[key]))
                break
    elif isinstance(parsed, str):
        # Handles "China; United States", "China|United States", and list-like strings.
        parts = re.split(r"\s*[;|,]\s*", parsed)
        countries.extend(parts)
    elif parsed is not None:
        countries.append(str(parsed))

    out = []
    for c in countries:
        c = re.sub(r"\s+", " ", str(c)).strip().strip("'\"")
        if not c or c.lower() in {"nan", "none", "null", "unknown"}:
            continue
        c = COUNTRY_FIXES.get(c.lower(), c)
        out.append(c)
    return list(dict.fromkeys(out))


def to_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "open"}:
        return True
    if s in {"0", "false", "f", "no", "n", "closed"}:
        return False
    return None

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def normalize_chunk(chunk: pd.DataFrame, schema: dict[str, str | None], source_file: str, args: argparse.Namespace) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    for clean_col in COLUMN_CANDIDATES:
        raw_col = schema.get(clean_col)
        if raw_col and raw_col in chunk.columns:
            out[clean_col] = chunk[raw_col]
        else:
            out[clean_col] = pd.NA

    out["source_file"] = source_file

    for col in ["title", "primary_topic", "topic_bucket", "primary_subfield", "primary_field", "primary_domain", "topics", "keywords", "abstract", "countries", "country", "institutions", "authorships", "venue", "venue_type", "id", "doi"]:
        if col in out:
            out[col] = out[col].map(normalize_label)

    out["title_clean"] = out["title"].map(clean_title)

    for col in NUMERIC_COLUMNS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out = out[(out["year"] >= args.min_year) & (out["year"] <= args.max_year)].copy()

    if args.drop_missing_title:
        out = out[out["title_clean"].ne("")].copy()

    if args.keep_unknown_topic:
        out["primary_topic"] = out["primary_topic"].replace("", pd.NA).fillna("Unknown topic")
    else:
        out = out[out["primary_topic"].replace("", pd.NA).notna()].copy()

    for col in ["fwci", "citation_count", "citation_velocity", "referenced_works_count", "author_count", "institution_count", "country_count"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["topic_bucket"] = out.apply(infer_topic_bucket, axis=1)

    # Prefer explicit country column, otherwise use the first parsed country from countries.
    parsed_countries = out["countries"].map(parse_countries)
    out["_country_list"] = parsed_countries
    country_from_col = out["country"].map(lambda x: parse_countries(x)[0] if parse_countries(x) else "")
    out["country"] = [
        c if c else (lst[0] if lst else "Unknown")
        for c, lst in zip(country_from_col, parsed_countries)
    ]
    out["countries"] = parsed_countries.map(lambda xs: "; ".join(xs) if xs else "Unknown")

    if "is_oa" in out:
        out["is_oa"] = out["is_oa"].map(to_bool)

    out["paper_age"] = args.max_year - out["year"] + 1
    out["impact_score"] = (
        out["fwci"].clip(lower=0).fillna(0) * 0.65
        + pd.Series(out["citation_velocity"]).clip(lower=0).fillna(0) * 0.25
        + pd.Series(out["citation_count"]).clip(lower=0).map(lambda x: math.log1p(float(x))) * 0.10
    )

    # High-impact label is computed later after global quantile is known.
    out["high_impact_label"] = pd.NA

    return out


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = len(df)
    working = df.copy()
    working["_row_order"] = range(len(working))
    working["_rank_keep"] = (
        working["fwci"].fillna(0) * 1000
        + working["citation_velocity"].fillna(0) * 10
        + working["citation_count"].fillna(0)
        + working["year"].fillna(0) * 0.01
    )

    parts = []
    summary_rows = []

    has_id = working["id"].replace("", pd.NA).notna()
    id_part = working[has_id].sort_values("_rank_keep", ascending=False).drop_duplicates("id", keep="first")
    parts.append(id_part)
    summary_rows.append({"dedupe_rule": "id", "input_rows": int(has_id.sum()), "kept_rows": len(id_part), "removed_rows": int(has_id.sum()) - len(id_part)})

    no_id = working[~has_id].copy()
    if not no_id.empty:
        key_cols = ["title_clean", "year", "primary_topic"]
        no_id = no_id.sort_values("_rank_keep", ascending=False).drop_duplicates(key_cols, keep="first")
        parts.append(no_id)
        summary_rows.append({"dedupe_rule": "title_clean+year+primary_topic", "input_rows": int((~has_id).sum()), "kept_rows": len(no_id), "removed_rows": int((~has_id).sum()) - len(no_id)})

    out = pd.concat(parts, ignore_index=True) if parts else working.iloc[0:0].copy()
    out = out.sort_values("_row_order").drop(columns=["_row_order", "_rank_keep"], errors="ignore")

    # One more safety pass: duplicate ids may overlap with no-id rows after string conversion edge cases.
    after = len(out)
    summary_rows.append({"dedupe_rule": "total", "input_rows": before, "kept_rows": after, "removed_rows": before - after})
    return out, pd.DataFrame(summary_rows)


def build_country_long(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        countries = parse_countries(row_dict.get("countries"))
        if not countries:
            countries = [row_dict.get("country") or "Unknown"]
        for country in countries:
            records.append({
                "work_id": row_dict.get("id") or row_dict.get("doi") or row_dict.get("title_clean"),
                "year": row_dict.get("year"),
                "country": country,
                "primary_topic": row_dict.get("primary_topic"),
                "topic_bucket": row_dict.get("topic_bucket"),
                "fwci": row_dict.get("fwci"),
                "citation_count": row_dict.get("citation_count"),
                "citation_velocity": row_dict.get("citation_velocity"),
            })
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.drop_duplicates(subset=["work_id", "country", "year", "primary_topic"])
    return out

# ---------------------------------------------------------------------------
# Reports and saving
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path: Path, output_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format or ("parquet" if path.suffix.lower() in {".parquet", ".pq"} else "csv")
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        missing = int(df[col].isna().sum() + (df[col].astype(str).str.strip().eq("").sum() if df[col].dtype == "object" else 0))
        rows.append({"column": col, "missing_rows": missing, "missing_pct": 100 * missing / n})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def write_report(df: pd.DataFrame, country_long: pd.DataFrame, missing_df: pd.DataFrame, dedupe_df: pd.DataFrame, raw_rows: int, args: argparse.Namespace, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(report_dir / "preprocess_column_missingness.csv", index=False)
    dedupe_df.to_csv(report_dir / "preprocess_deduplication_summary.csv", index=False)

    top_topics = df["primary_topic"].value_counts().head(15)
    top_countries = country_long["country"].value_counts().head(15) if not country_long.empty else pd.Series(dtype=int)
    year_min = int(df["year"].min()) if not df.empty else args.min_year
    year_max = int(df["year"].max()) if not df.empty else args.max_year

    lines = []
    lines.append("# Preprocess quality report")
    lines.append("")
    lines.append(f"Input rows read: {raw_rows:,}")
    lines.append(f"Clean work rows: {len(df):,}")
    lines.append(f"Country-long rows: {len(country_long):,}")
    lines.append(f"Year range kept: {year_min}-{year_max}")
    lines.append(f"Duplicate rows removed: {int(dedupe_df[dedupe_df['dedupe_rule'].eq('total')]['removed_rows'].iloc[0]) if not dedupe_df.empty else 0:,}")
    lines.append(f"Missing title share: {float(missing_df[missing_df['column'].eq('title')]['missing_pct'].iloc[0]) if 'title' in set(missing_df['column']) else 0:.2f}%")
    lines.append(f"Missing country share: {float(missing_df[missing_df['column'].eq('country')]['missing_pct'].iloc[0]) if 'country' in set(missing_df['column']) else 0:.2f}%")
    lines.append("")
    lines.append("## Cleaning decisions")
    lines.append("- Dropped rows with missing or out-of-range year.")
    lines.append("- Dropped missing primary_topic rows by default. Use --keep-unknown-topic to keep them.")
    lines.append("- Deduplicated first by id, then by title_clean + year + primary_topic when id is unavailable.")
    lines.append("- Filled numeric impact fields with 0 when missing.")
    lines.append("- Inferred topic_bucket from topic text when missing.")
    lines.append("- Parsed multi-country fields into a country-long table.")
    lines.append("")
    lines.append("## Top topics")
    for name, count in top_topics.items():
        lines.append(f"- {name}: {count:,}")
    lines.append("")
    lines.append("## Top countries")
    for name, count in top_countries.items():
        lines.append(f"- {name}: {count:,}")
    lines.append("")
    lines.append("## Dashboard handoff")
    lines.append("Use the clean work table as input to build_core_dashboard_cache.py. Use the country-long table for top_countries.csv and country_topic_year.csv when supported.")
    lines.append("For Tab 3 ML, use impact_score or high_impact_label as target candidates, and use topic family, collaboration counts, venue type, open access, reference count, country count, and paper age as model features.")
    (report_dir / "preprocess_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.input)
    print(f"Found {len(files)} input files")

    frames = []
    raw_rows = 0
    schema: dict[str, str | None] | None = None
    usecols: list[str] | None = None

    for path in files:
        sample = read_sample(path)
        file_schema = build_schema(sample.columns)
        if schema is None:
            schema = file_schema
        raw_cols = [c for c in file_schema.values() if c]
        usecols = sorted(set(raw_cols)) if raw_cols else None
        print(f"Reading {path.name}; columns used: {usecols}")
        for chunk in iter_file(path, usecols, args.chunksize):
            raw_rows += len(chunk)
            clean = normalize_chunk(chunk, file_schema, path.name, args)
            if not clean.empty:
                frames.append(clean)
            print(f"  rows read so far: {raw_rows:,}; clean chunks: {sum(len(f) for f in frames):,}")

    if not frames:
        raise RuntimeError("No rows left after preprocessing. Check year/topic/title filters.")

    df = pd.concat(frames, ignore_index=True)
    df, dedupe_df = dedupe(df)

    # Global high-impact label. Use FWCI first when present, otherwise impact_score.
    target_source = df["fwci"] if df["fwci"].fillna(0).gt(0).any() else df["impact_score"]
    cutoff = float(target_source.quantile(0.75)) if len(df) else 0.0
    df["high_impact_label"] = (target_source >= cutoff).astype(int)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS]

    country_long = build_country_long(df)
    missing_df = missingness_profile(df)

    output_path = Path(args.output)
    country_path = Path(args.country_output)
    report_dir = Path(args.report_dir)

    save_table(df, output_path, args.output_format)
    save_table(country_long, country_path, "csv")
    write_report(df, country_long, missing_df, dedupe_df, raw_rows, args, report_dir)

    print("\nSaved:")
    print(f"- {output_path} ({len(df):,} rows)")
    print(f"- {country_path} ({len(country_long):,} rows)")
    print(f"- {report_dir / 'preprocess_report.md'}")
    print(f"- {report_dir / 'preprocess_column_missingness.csv'}")
    print(f"- {report_dir / 'preprocess_deduplication_summary.csv'}")


if __name__ == "__main__":
    main()
