from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


STRICT_GROUPS = {
    "strict_computer_vision": [
        r"\bcomputer vision\b",
        r"\bobject detection\b",
        r"\bimage segmentation\b",
        r"\bsemantic segmentation\b",
        r"\binstance segmentation\b",
        r"\bimage classification\b",
        r"\bvisual recognition\b",
        r"\bvisual representation\b",
        r"\bvisual question answering\b",
        r"\bvideo understanding\b",
        r"\bvideo recognition\b",
        r"\bface recognition\b",
        r"\bmedical imaging\b",
        r"\bremote sensing image\b",
        r"\bconvolutional neural network\b",
        r"\bconvolutional neural networks\b",
    ],
    "strict_llm_transformers": [
        r"\blarge language model\b",
        r"\blarge language models\b",
        r"\bLLM\b",
        r"\bLLMs\b",
        r"\btransformer\b",
        r"\btransformers\b",
        r"\bBERT\b",
        r"\bGPT\b",
        r"\bChatGPT\b",
        r"\bprompt engineering\b",
        r"\bprompting\b",
        r"\binstruction tuning\b",
        r"\bfoundation model\b",
        r"\bfoundation models\b",
        r"\bgenerative pre-trained\b",
    ],
}


TEXT_COLS = [
    "title",
    "topics",
    "keywords",
    "primary_topic",
    "primary_subfield",
    "primary_field",
    "primary_domain",
    "search_term",
]


def compile_group(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), flags=re.IGNORECASE)


def existing_files(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        nargs="+",
        required=True,
        help="One or more raw CSV files.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--out", default="Dataset/dashboard_cache/audit_raw_strict")
    args = parser.parse_args()

    raw_paths = existing_files(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = {name: compile_group(pats) for name, pats in STRICT_GROUPS.items()}

    counts = {name: 0 for name in STRICT_GROUPS}
    file_counts = []
    year_parts = []
    topic_parts = []
    sample_parts = []
    total = 0

    for raw_path in raw_paths:
        print(f"\n=== Reading {raw_path} ===")
        sample = pd.read_csv(raw_path, nrows=5, low_memory=False)
        cols = sample.columns.tolist()

        text_cols = [c for c in TEXT_COLS if c in cols]
        year_col = "publication_year" if "publication_year" in cols else "year" if "year" in cols else None

        use_cols = list(dict.fromkeys(text_cols + ([year_col] if year_col else [])))
        print("Text columns:", text_cols)
        print("Year column:", year_col)

        file_total = 0
        file_group_counts = {name: 0 for name in STRICT_GROUPS}

        for i, chunk in enumerate(pd.read_csv(raw_path, usecols=use_cols, chunksize=args.chunksize, low_memory=False)):
            total += len(chunk)
            file_total += len(chunk)
            print(f"{raw_path.name} chunk {i + 1}, file rows: {file_total:,}, total rows: {total:,}")

            combined = pd.Series("", index=chunk.index, dtype="object")
            for c in text_cols:
                combined = combined + " " + chunk[c].fillna("").astype(str)

            for group, pattern in patterns.items():
                mask = combined.str.contains(pattern, na=False)
                hit_count = int(mask.sum())
                counts[group] += hit_count
                file_group_counts[group] += hit_count

                if not mask.any():
                    continue

                sub = chunk.loc[mask].copy()
                sub["_group"] = group
                sub["_source_file"] = raw_path.name

                if year_col:
                    y = sub.groupby(year_col).size().reset_index(name="count")
                    y["_group"] = group
                    y["_source_file"] = raw_path.name
                    year_parts.append(y)

                if "primary_topic" in sub.columns:
                    t = (
                        sub.groupby("primary_topic", dropna=False)
                        .size()
                        .reset_index(name="count")
                    )
                    t["_group"] = group
                    t["_source_file"] = raw_path.name
                    topic_parts.append(t)

                keep = [
                    c for c in [
                        "title", "publication_year", "year", "primary_topic",
                        "primary_subfield", "primary_field", "primary_domain",
                        "keywords", "topics", "search_term"
                    ]
                    if c in sub.columns
                ]
                sample_parts.append(sub[keep].head(30).assign(_group=group, _source_file=raw_path.name))

        for group, count in file_group_counts.items():
            file_counts.append({
                "source_file": raw_path.name,
                "group": group,
                "matched_rows": count,
                "file_rows": file_total,
                "share_pct": 100 * count / max(file_total, 1),
            })

    counts_df = pd.DataFrame([
        {
            "group": group,
            "matched_rows": count,
            "total_rows": total,
            "share_pct": 100 * count / max(total, 1),
        }
        for group, count in counts.items()
    ]).sort_values("matched_rows", ascending=False)

    counts_df.to_csv(out_dir / "strict_group_counts.csv", index=False)
    pd.DataFrame(file_counts).to_csv(out_dir / "strict_group_counts_by_file.csv", index=False)

    print("\n=== Strict group counts ===")
    print(counts_df.to_string(index=False))

    print("\n=== Strict group counts by file ===")
    print(pd.DataFrame(file_counts).to_string(index=False))

    if year_parts:
        year_df = pd.concat(year_parts, ignore_index=True)
        year_name = "publication_year" if "publication_year" in year_df.columns else "year"
        year_df = year_df.groupby(["_group", year_name], as_index=False)["count"].sum()
        year_df.to_csv(out_dir / "strict_group_year_counts.csv", index=False)

    if topic_parts:
        topic_df = pd.concat(topic_parts, ignore_index=True)
        topic_df = topic_df.groupby(["_group", "primary_topic"], as_index=False)["count"].sum()
        topic_df = topic_df.sort_values(["_group", "count"], ascending=[True, False])
        topic_df.to_csv(out_dir / "strict_group_primary_topics.csv", index=False)

        print("\n=== Top primary topics ===")
        for group in topic_df["_group"].unique():
            print(f"\n--- {group} ---")
            print(topic_df[topic_df["_group"].eq(group)].head(20).to_string(index=False))

    if sample_parts:
        sample_df = pd.concat(sample_parts, ignore_index=True)
        sample_df.to_csv(out_dir / "strict_group_samples.csv", index=False)

    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()