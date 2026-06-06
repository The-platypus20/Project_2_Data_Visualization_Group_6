from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SEARCH_GROUPS = {
    "computer_vision": [
        "computer vision", "image", "visual", "video", "object detection",
        "segmentation", "recognition", "image classification", "face recognition",
        "medical imaging", "remote sensing", "convolutional", "cnn"
    ],
    "llm_transformers": [
        "large language model", "large language models", "llm", "llms",
        "transformer", "transformers", "bert", "gpt", "chatgpt",
        "prompt", "prompting", "instruction tuning", "foundation model",
        "generative pre-trained"
    ],
    "nlp": [
        "natural language", "nlp", "text", "speech", "semantic",
        "sentiment", "translation", "dialogue", "language model"
    ],
    "robotics": [
        "robot", "robotics", "autonomous", "navigation", "control",
        "motion planning", "path planning", "manipulation"
    ],
    "healthcare": [
        "health", "healthcare", "medical", "clinical", "cancer",
        "disease", "diagnosis", "radiology", "patient", "neuroscience"
    ],
    "responsible_ai": [
        "fairness", "ethics", "bias", "privacy", "explainable",
        "xai", "trust", "safety", "law", "intellectual property"
    ],
}


TEXT_COLUMNS_CANDIDATES = [
    "title",
    "topics",
    "keywords",
    "primary_topic",
    "primary_subfield",
    "primary_field",
    "primary_domain",
    "search_term",
]


def make_pattern(words: list[str]) -> re.Pattern:
    escaped = [re.escape(w).replace(r"\ ", r"\s+") for w in words]
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


def detect_csv_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return path


def safe_read_sample(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def available_columns(path: Path) -> list[str]:
    sample = safe_read_sample(path, 5)
    return sample.columns.tolist()


def audit_raw(path: Path, chunksize: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = available_columns(path)
    print("Columns:")
    print(cols)

    text_cols = [c for c in TEXT_COLUMNS_CANDIDATES if c in cols]
    print("\nText columns used:")
    print(text_cols)

    if "publication_year" in cols:
        year_col = "publication_year"
    elif "year" in cols:
        year_col = "year"
    else:
        year_col = None

    topic_col = "primary_topic" if "primary_topic" in cols else None
    bucket_cols = [c for c in ["primary_subfield", "primary_field", "primary_domain", "search_term"] if c in cols]

    patterns = {name: make_pattern(words) for name, words in SEARCH_GROUPS.items()}

    group_counts = {name: 0 for name in SEARCH_GROUPS}
    group_year_counts = []
    topic_hits = []
    sample_hits = []

    total_rows = 0

    use_cols = list(dict.fromkeys(text_cols + bucket_cols + ([year_col] if year_col else []) + ([topic_col] if topic_col else [])))

    for i, chunk in enumerate(pd.read_csv(path, usecols=use_cols, chunksize=chunksize, low_memory=False)):
        total_rows += len(chunk)
        print(f"Processing chunk {i + 1}, rows so far: {total_rows:,}")

        combined = pd.Series("", index=chunk.index, dtype="object")
        for c in text_cols:
            combined = combined + " " + chunk[c].fillna("").astype(str)

        for group_name, pattern in patterns.items():
            mask = combined.str.contains(pattern, na=False)
            hit_count = int(mask.sum())
            group_counts[group_name] += hit_count

            if hit_count == 0:
                continue

            sub = chunk.loc[mask].copy()
            sub["_group"] = group_name

            if year_col:
                y = (
                    sub.groupby(year_col, dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                y["_group"] = group_name
                group_year_counts.append(y)

            if topic_col:
                topic_summary = (
                    sub.groupby(topic_col, dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .head(30)
                )
                topic_summary["_group"] = group_name
                topic_hits.append(topic_summary)

            keep_sample_cols = [c for c in ["title", "publication_year", "year", "primary_topic", "primary_subfield", "primary_field", "primary_domain", "keywords", "topics", "search_term"] if c in sub.columns]
            sample_hits.append(sub[keep_sample_cols].head(20).assign(_group=group_name))

    print("\n=== Total rows ===")
    print(total_rows)

    print("\n=== Search group hit counts ===")
    group_df = (
        pd.DataFrame(
            [{"group": k, "matched_rows": v, "share_pct": 100 * v / max(total_rows, 1)} for k, v in group_counts.items()]
        )
        .sort_values("matched_rows", ascending=False)
    )
    print(group_df.to_string(index=False))
    group_df.to_csv(out_dir / "raw_search_group_counts.csv", index=False)

    if group_year_counts:
        year_df = pd.concat(group_year_counts, ignore_index=True)
        year_df.to_csv(out_dir / "raw_search_group_year_counts.csv", index=False)

    if topic_hits:
        topic_df = pd.concat(topic_hits, ignore_index=True)
        topic_df = (
            topic_df.groupby(["_group", topic_col], dropna=False)["count"]
            .sum()
            .reset_index()
            .sort_values(["_group", "count"], ascending=[True, False])
        )
        topic_df.to_csv(out_dir / "raw_search_group_primary_topics.csv", index=False)

        print("\n=== Top primary_topic per search group ===")
        for group in topic_df["_group"].unique():
            print(f"\n--- {group} ---")
            print(topic_df[topic_df["_group"].eq(group)].head(20).to_string(index=False))

    if sample_hits:
        sample_df = pd.concat(sample_hits, ignore_index=True)
        sample_df.to_csv(out_dir / "raw_search_group_samples.csv", index=False)

    print("\nSaved audit files to:")
    print(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to raw CSV, e.g. Dataset/ai_works_merge_2000_2025.csv")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--out", default="Dataset/dashboard_cache/audit_raw")
    args = parser.parse_args()

    audit_raw(
        path=detect_csv_path(Path(args.raw)),
        chunksize=args.chunksize,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()