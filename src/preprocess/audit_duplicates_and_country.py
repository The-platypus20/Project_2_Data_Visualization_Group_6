from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def pick_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_missing_country(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {
        "na", "n/a", "nan", "none", "null", "<na>", "unknown",
        "[]", "{}", "[nan]", "[none]", "[null]", "['nan']", "['none']",
    }


def parse_country_values(value: Any) -> list[str]:
    if is_missing_country(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if not is_missing_country(x)]

    text = str(value).strip()

    if text.startswith("[") or text.startswith("{"):
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(text)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if not is_missing_country(x)]
                if isinstance(obj, dict):
                    vals = []
                    for key in ["country", "country_name", "display_name", "name", "code", "country_code"]:
                        if key in obj and not is_missing_country(obj[key]):
                            vals.append(str(obj[key]).strip())
                    if vals:
                        return vals
                    return [str(x).strip() for x in obj.values() if not is_missing_country(x)]
            except Exception:
                pass

    parts = re.split(r"\s*[;|]\s*", text)
    # Do not split "United States" or comma-separated names too aggressively.
    if len(parts) == 1 and "," in text and not any(x in text for x in ["United States", "United Kingdom"]):
        parts = [p.strip() for p in text.split(",")]
    return [p.strip() for p in parts if not is_missing_country(p)]


def read_sample_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=5, low_memory=False).columns.tolist()


def audit_duplicates(inputs: list[Path], outdir: Path, sample_limit: int, chunksize: int) -> None:
    frames = []
    for path in inputs:
        cols = read_sample_columns(path)
        id_col = pick_col(cols, ["id", "paper_id", "work_id", "openalex_id", "doi"])
        title_col = pick_col(cols, ["title", "display_name", "work_title"])
        year_col = pick_col(cols, ["year", "publication_year", "pub_year"])
        topic_col = pick_col(cols, ["primary_topic", "topic", "topic_label"])
        cite_col = pick_col(cols, ["citation_count", "cited_by_count", "citations"])
        fwci_col = pick_col(cols, ["fwci", "field_weighted_citation_impact"])
        country_col = pick_col(cols, ["country", "countries", "country_names", "country_code", "countries_distinct_count"])

        usecols = [c for c in [id_col, title_col, year_col, topic_col, cite_col, fwci_col, country_col] if c]
        if not usecols:
            continue

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            out = pd.DataFrame()
            out["source_file"] = path.name
            out["row_in_chunk"] = range(len(chunk))
            out["id"] = chunk[id_col] if id_col else ""
            out["title"] = chunk[title_col] if title_col else ""
            out["year"] = chunk[year_col] if year_col else ""
            out["primary_topic"] = chunk[topic_col] if topic_col else ""
            out["citation_count"] = chunk[cite_col] if cite_col else ""
            out["fwci"] = chunk[fwci_col] if fwci_col else ""
            out["country_raw"] = chunk[country_col] if country_col else ""
            out["title_clean"] = out["title"].map(norm_text)
            out["topic_clean"] = out["primary_topic"].map(norm_text)
            out["id_clean"] = out["id"].map(norm_text)
            out["year_clean"] = pd.to_numeric(out["year"], errors="coerce")
            frames.append(out)

    if not frames:
        raise RuntimeError("No readable input columns found.")

    df = pd.concat(frames, ignore_index=True)

    # ID duplicates.
    id_dupes = pd.DataFrame()
    with_id = df[df["id_clean"].ne("")].copy()
    if not with_id.empty:
        counts = with_id["id_clean"].value_counts()
        dup_ids = set(counts[counts > 1].index)
        id_dupes = with_id[with_id["id_clean"].isin(dup_ids)].copy()
        id_dupes["duplicate_rule"] = "same_id"

    # Title-year-topic duplicates only where id is missing.
    no_id = df[df["id_clean"].eq("")].copy()
    no_id["title_year_topic_key"] = (
        no_id["title_clean"] + "|" +
        no_id["year_clean"].astype("Int64").astype(str) + "|" +
        no_id["topic_clean"]
    )
    key_counts = no_id["title_year_topic_key"].value_counts()
    dup_keys = set(key_counts[key_counts > 1].index)
    title_dupes = no_id[no_id["title_year_topic_key"].isin(dup_keys)].copy()
    title_dupes["duplicate_rule"] = "same_title_year_topic_no_id"

    duplicate_candidates = pd.concat([id_dupes, title_dupes], ignore_index=True)
    duplicate_candidates = duplicate_candidates.sort_values(
        ["duplicate_rule", "id_clean", "title_clean", "year_clean", "topic_clean"]
    )

    summary_rows = [
        {"metric": "input_rows", "value": len(df)},
        {"metric": "same_id_duplicate_rows", "value": len(id_dupes)},
        {"metric": "same_title_year_topic_no_id_duplicate_rows", "value": len(title_dupes)},
        {"metric": "total_duplicate_candidate_rows", "value": len(duplicate_candidates)},
    ]
    pd.DataFrame(summary_rows).to_csv(outdir / "duplicate_audit_summary.csv", index=False)

    keep_cols = [
        "duplicate_rule", "source_file", "id", "title", "year", "primary_topic",
        "citation_count", "fwci", "country_raw", "title_clean"
    ]
    duplicate_candidates[keep_cols].head(sample_limit).to_csv(
        outdir / "duplicate_candidates_sample.csv", index=False
    )

    # Group-level compact view for human inspection.
    if not duplicate_candidates.empty:
        duplicate_candidates["group_key"] = duplicate_candidates.apply(
            lambda r: r["id_clean"] if r["duplicate_rule"] == "same_id" else r.get("title_year_topic_key", ""),
            axis=1,
        )
        group_view = (
            duplicate_candidates
            .groupby(["duplicate_rule", "group_key"], dropna=False)
            .agg(
                rows=("title", "size"),
                titles=("title", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
                years=("year", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
                topics=("primary_topic", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
                ids=("id", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
                citations=("citation_count", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
                countries=("country_raw", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(4))),
            )
            .reset_index()
            .sort_values(["duplicate_rule", "rows"], ascending=[True, False])
        )
        group_view.head(sample_limit).to_csv(outdir / "duplicate_groups_sample.csv", index=False)


def audit_country(clean_path: Path, country_long_path: Path | None, outdir: Path, chunksize: int, sample_limit: int) -> None:
    cols = read_sample_columns(clean_path)
    year_col = pick_col(cols, ["year", "publication_year"])
    country_col = pick_col(cols, ["country", "countries", "country_names", "country_code"])
    topic_col = pick_col(cols, ["topic_bucket", "family", "primary_topic", "topic_family", "primary_subfield"])
    title_col = pick_col(cols, ["title", "display_name"])
    id_col = pick_col(cols, ["id", "paper_id", "work_id", "openalex_id", "doi"])

    if country_col is None:
        raise ValueError("No country/countries column found in clean file.")

    usecols = [c for c in [id_col, title_col, year_col, topic_col, country_col] if c]
    total = 0
    missing_raw = 0
    parsed_empty = 0
    samples = []
    country_counts = {}

    for chunk in pd.read_csv(clean_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        total += len(chunk)
        miss_mask = chunk[country_col].map(is_missing_country)
        missing_raw += int(miss_mask.sum())

        parsed = chunk[country_col].map(parse_country_values)
        empty_mask = parsed.map(len).eq(0)
        parsed_empty += int(empty_mask.sum())

        for countries in parsed:
            for c in countries:
                country_counts[c] = country_counts.get(c, 0) + 1

        bad = chunk[empty_mask].head(max(0, sample_limit - len(samples)))
        if not bad.empty:
            samples.append(bad)

    country_summary = pd.DataFrame([
        {"metric": "clean_rows", "value": total},
        {"metric": "raw_missing_country_rows", "value": missing_raw},
        {"metric": "parsed_empty_country_rows", "value": parsed_empty},
        {"metric": "raw_missing_country_share_pct", "value": 100 * missing_raw / max(total, 1)},
        {"metric": "parsed_empty_country_share_pct", "value": 100 * parsed_empty / max(total, 1)},
    ])
    country_summary.to_csv(outdir / "country_missing_summary.csv", index=False)

    top = (
        pd.DataFrame([{"country": k, "rows": v} for k, v in country_counts.items()])
        .sort_values("rows", ascending=False)
    )
    top.to_csv(outdir / "country_value_counts_from_clean.csv", index=False)

    if samples:
        pd.concat(samples, ignore_index=True).head(sample_limit).to_csv(
            outdir / "country_missing_samples.csv", index=False
        )

    if country_long_path and country_long_path.exists() and country_long_path.is_file():
        long = pd.read_csv(country_long_path, low_memory=False)
        pd.DataFrame([
            {"metric": "country_long_rows", "value": len(long)},
            {"metric": "country_long_columns", "value": ", ".join(long.columns.astype(str))},
        ]).to_csv(outdir / "country_long_file_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", nargs="+", required=True, help="Raw CSV files used before preprocessing.")
    parser.add_argument("--clean-input", required=True, help="Clean work CSV produced by preprocess_ai_works.py.")
    parser.add_argument("--country-long", default="", help="Optional country-long CSV from preprocess_ai_works.py.")
    parser.add_argument("--outdir", default="Dataset/clean/audit_duplicates_country")
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--sample-limit", type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_paths = [Path(p) for p in args.raw_input]
    for p in raw_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    clean_path = Path(args.clean_input)
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)

    country_long_path = Path(args.country_long) if args.country_long else None

    audit_duplicates(raw_paths, outdir, args.sample_limit, args.chunksize)
    audit_country(clean_path, country_long_path, outdir, args.chunksize, args.sample_limit)

    print(f"Audit written to {outdir}")
    for p in sorted(outdir.glob("*.csv")):
        print("-", p)


if __name__ == "__main__":
    main()
