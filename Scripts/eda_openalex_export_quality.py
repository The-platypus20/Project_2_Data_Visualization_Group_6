"""EDA checks for OpenAlex export quality.

Focus areas:
- what papers with missing `countries` look like
- whether suspicious non-AI papers still contain AI signals in their metadata

The script reads finished `part_*.csv` files from a shard directory and prints
compact summaries plus a few example rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHARD_DIR = ROOT / "Dataset" / "openalex_exports" / "ai_works_2020_2025"
TEXT_COLUMNS = ["title", "primary_topic", "primary_subfield", "primary_field", "topics", "keywords"]
AI_SIGNAL_PATTERNS = [
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural network",
    r"\blarge language model",
    r"\blanguage model\b",
    r"\bnlp\b",
    r"\bnatural language\b",
    r"\bcomputer vision\b",
    r"\breinforcement learning\b",
    r"\brobot",
    r"\bpattern recognition\b",
    r"\bclassification\b",
    r"\bclustering\b",
    r"\brepresentation learning\b",
    r"\bgenerative\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DEFAULT_SHARD_DIR,
        help="Path to the export shard directory.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=0,
        help="Only inspect the last N completed parts. Default 0 means all parts.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="How many example rows to print per section.",
    )
    return parser.parse_args()


def completed_parts(shard_dir: Path, parts: int) -> list[Path]:
    files = sorted(shard_dir.glob("part_*.csv"))
    if parts > 0:
        return files[-parts:]
    return files


def load_parts(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No completed part_*.csv files found.")
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    return pd.concat(frames, ignore_index=True)


def ai_signal_text(row: pd.Series) -> str:
    values = [str(row.get(col) or "") for col in TEXT_COLUMNS]
    return " || ".join(values).lower()


def has_ai_signal(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in AI_SIGNAL_PATTERNS)


def add_ai_signal_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ai_signal_text"] = out.apply(ai_signal_text, axis=1)
    out["has_ai_signal"] = out["ai_signal_text"].map(has_ai_signal)
    out["has_ai_in_primary_topic"] = out["primary_topic"].fillna("").astype(str).str.contains(
        "artificial intelligence|machine learning|deep learning|neural|language model|computer vision|robot|reinforcement",
        case=False,
        regex=True,
    )
    out["has_ai_in_topics_or_keywords"] = (
        out["topics"].fillna("").astype(str).str.contains(
            "artificial intelligence|machine learning|deep learning|neural|language model|computer vision|robot|reinforcement",
            case=False,
            regex=True,
        )
        | out["keywords"].fillna("").astype(str).str.contains(
            "artificial intelligence|machine learning|deep learning|neural|language model|computer vision|robot|reinforcement",
            case=False,
            regex=True,
        )
    )
    return out


def print_missing_country_eda(df: pd.DataFrame, sample_rows: int) -> None:
    missing = df[df["countries"].isna() | (df["countries"].astype(str).str.strip() == "")]
    print("\n=== Missing country overview ===")
    print(f"missing countries rows: {len(missing):,} / {len(df):,} ({len(missing)/max(len(df),1):.2%})")
    if missing.empty:
        return

    print("\nTop primary topics among missing-country rows:")
    print(missing["primary_topic"].fillna("(missing)").value_counts().head(15).to_string())

    print("\nTop publication types among missing-country rows:")
    print(missing["publication_type"].fillna("(missing)").value_counts().to_string())

    print("\nInstitution count summary for missing-country rows:")
    print(missing["institution_count"].describe().to_string())

    print("\nVenue/source missingness among missing-country rows:")
    venue_missing = pd.DataFrame({
        "venue_source_missing": [missing["venue_source"].isna().sum()],
        "venue_type_missing": [missing["venue_type"].isna().sum()],
        "institutions_missing": [missing["institutions"].isna().sum()],
    })
    print(venue_missing.to_string(index=False))

    show_cols = [
        "paper_id",
        "title",
        "publication_year",
        "publication_type",
        "primary_topic",
        "topics",
        "keywords",
        "authors",
        "institutions",
        "countries",
        "venue_source",
        "venue_type",
        "doi",
    ]
    print(f"\nSample missing-country rows ({min(sample_rows, len(missing))} rows):")
    print(missing[show_cols].head(sample_rows).to_string(index=False))


def print_ai_signal_eda(df: pd.DataFrame, sample_rows: int) -> None:
    flagged = add_ai_signal_flags(df)
    suspicious = flagged[~flagged["has_ai_signal"]].copy()

    print("\n=== AI signal overview ===")
    print(f"rows with any AI signal in title/topic/keywords: {flagged['has_ai_signal'].sum():,} / {len(flagged):,} ({flagged['has_ai_signal'].mean():.2%})")
    print(f"rows with NO AI signal in title/topic/keywords: {len(suspicious):,} / {len(flagged):,} ({len(suspicious)/max(len(flagged),1):.2%})")

    if suspicious.empty:
        return

    print("\nTop primary topics among suspicious rows:")
    print(suspicious["primary_topic"].fillna("(missing)").value_counts().head(20).to_string())

    print("\nTop primary subfields among suspicious rows:")
    print(suspicious["primary_subfield"].fillna("(missing)").value_counts().head(20).to_string())

    show_cols = [
        "paper_id",
        "title",
        "publication_year",
        "primary_topic",
        "primary_subfield",
        "primary_field",
        "topics",
        "keywords",
        "venue_source",
        "doi",
        "has_ai_in_primary_topic",
        "has_ai_in_topics_or_keywords",
    ]
    print(f"\nSample suspicious rows ({min(sample_rows, len(suspicious))} rows):")
    print(suspicious[show_cols].head(sample_rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    paths = completed_parts(args.shard_dir, args.parts)
    print(f"Loading {len(paths)} completed parts from {args.shard_dir}")
    df = load_parts(paths)
    print(f"Loaded {len(df):,} rows")

    print_missing_country_eda(df, args.sample_rows)
    print_ai_signal_eda(df, args.sample_rows)


if __name__ == "__main__":
    main()
