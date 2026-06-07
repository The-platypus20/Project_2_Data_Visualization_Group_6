"""EDA + preprocess + clean the merged raw dataset into one analysis-ready file.

This is STEP 2 of the offline data pipeline:

    Step 1  merge_raw_dataset.py   3 shards            -> 1 raw merged CSV
    Step 2  build_clean_dataset.py 1 raw merged CSV    -> 1 clean CSV + EDA report   <-- this file
    Step 3  build_*_cache.py       1 clean CSV         -> Dataset/dashboard_cache/*.csv
    Step 4  app.py                 dashboard_cache/*   -> visualisation (fast)

What this script does, in one streaming pass over the merged file:
  * EDA  : row count, dtypes, missing-value shares, duplicate count, year range,
           citation distribution, top topics / countries / venues, OA share.
  * CHECK: detects duplicates and missing/out-of-range values.
  * CLEAN: drops bad-year and missing-topic rows, removes duplicates, fills the
           numeric impact fields with 0.

CRITICAL: the clean file keeps EXACTLY the same columns as the raw shards, so
every existing cache builder (build_core_dashboard_cache, build_impact_ml_cache,
build_institution_type_view, build_rising_fading_terms_family) works unchanged --
they just read fewer, cleaner rows. Only the row content changes, never the
structure of the downstream cache CSVs.

Usage:
    python src/preprocess/build_clean_dataset.py \
        --input Dataset/clean/ai_works_merged_raw.csv \
        --output Dataset/clean/ai_works_clean.csv \
        --report-dir Dataset/clean/preprocess_report
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that represent numeric impact / collaboration signals. Missing values
# in these are filled with 0 (a missing citation count means "no citations yet").
NUMERIC_FILL_COLS = [
    "citation_count", "citations_per_year", "fwci", "referenced_works_count",
    "paper_age", "citation_velocity", "author_count", "institution_count",
    "country_count",
]

_WS = re.compile(r"\s+")


def _title_key(value: object) -> str:
    """Normalise a title for fuzzy de-duplication (lowercase, collapse spaces)."""
    return _WS.sub(" ", str(value).strip().lower())


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "<na>"}


def build_clean(args: argparse.Namespace) -> None:
    src = Path(args.input)
    out = Path(args.output)
    report_dir = Path(args.report_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # we append chunk by chunk, so start fresh

    # ---- running accumulators (memory-bounded) -----------------------------
    rows_read = 0
    dropped_year = 0
    dropped_topic = 0
    dups_removed = 0
    clean_rows = 0

    seen_keys: set[str] = set()                 # global de-dup state
    missing_counts: Counter[str] = Counter()    # missing per column (clean rows)
    topic_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    venue_type_counter: Counter[str] = Counter()
    year_counter: Counter[int] = Counter()
    oa_true = 0
    citation_values: list[np.ndarray] = []
    velocity_values: list[np.ndarray] = []
    dtypes_seen: dict[str, str] = {}

    first_write = True
    for ci, chunk in enumerate(pd.read_csv(
        src, dtype=str, chunksize=args.chunksize, encoding="utf-8-sig", low_memory=False
    )):
        rows_read += len(chunk)
        if not dtypes_seen:
            dtypes_seen = {c: str(t) for c, t in chunk.dtypes.items()}

        # --- year: parse, range-filter -------------------------------------
        year = pd.to_numeric(chunk.get("publication_year"), errors="coerce")
        year_ok = year.between(args.min_year, args.max_year)
        dropped_year += int((~year_ok).sum())
        chunk = chunk.loc[year_ok].copy()
        chunk["publication_year"] = year.loc[year_ok].astype(int).astype(str)

        # --- primary_topic: required ---------------------------------------
        topic_blank = chunk["primary_topic"].map(_is_blank)
        dropped_topic += int(topic_blank.sum())
        chunk = chunk.loc[~topic_blank].copy()

        # --- de-duplicate: id first, else title+year+topic -----------------
        # Vectorised key build (a per-row Python loop over 2.2M rows is too slow).
        ids = (chunk["paper_id"] if "paper_id" in chunk.columns
               else pd.Series("", index=chunk.index)).fillna("").astype(str).str.strip()
        id_blank = ids.str.lower().isin(["", "nan", "none", "null", "<na>"])
        titles = chunk["title"].map(_title_key)
        years = chunk["publication_year"].astype(str)
        topics = chunk["primary_topic"].astype(str)
        key_id = "id::" + ids
        key_tt = "tt::" + titles + "||" + years + "||" + topics
        keys = key_id.where(~id_blank, key_tt)
        is_dup = keys.duplicated(keep="first") | keys.isin(seen_keys)
        dups_removed += int(is_dup.sum())
        chunk = chunk.loc[~is_dup].copy()
        seen_keys.update(keys.loc[~is_dup].tolist())

        # --- fill numeric impact fields ------------------------------------
        for col in NUMERIC_FILL_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        # --- accumulate EDA on the surviving (clean) rows ------------------
        clean_rows += len(chunk)
        for col in ["title", "primary_topic", "countries", "venue_type", "is_oa"]:
            if col in chunk.columns:
                missing_counts[col] += int(chunk[col].map(_is_blank).sum())
        topic_counter.update(chunk["primary_topic"].astype(str))
        year_counter.update(chunk["publication_year"].astype(int))
        if "venue_type" in chunk.columns:
            venue_type_counter.update(
                chunk["venue_type"].fillna("(unknown)").replace("", "(unknown)").astype(str))
        if "is_oa" in chunk.columns:
            oa_true += int(chunk["is_oa"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        if "country_names" in chunk.columns:
            primary_country = chunk["country_names"].map(_first_country)
            country_counter.update(c for c in primary_country if c)
        citation_values.append(pd.to_numeric(chunk.get("citation_count"), errors="coerce")
                               .fillna(0).to_numpy())
        velocity_values.append(pd.to_numeric(chunk.get("citation_velocity"), errors="coerce")
                               .fillna(0).to_numpy())

        # --- write clean chunk ---------------------------------------------
        chunk.to_csv(out, mode="w" if first_write else "a",
                     header=first_write, index=False)
        first_write = False
        print(f"  chunk {ci + 1}: read so far {rows_read:,} | clean {clean_rows:,}", flush=True)

    citations = np.concatenate(citation_values) if citation_values else np.array([0.0])
    velocity = np.concatenate(velocity_values) if velocity_values else np.array([0.0])
    _write_report(
        report_dir=report_dir, out=out,
        rows_read=rows_read, clean_rows=clean_rows,
        dropped_year=dropped_year, dropped_topic=dropped_topic, dups_removed=dups_removed,
        missing_counts=missing_counts, topic_counter=topic_counter,
        country_counter=country_counter, venue_type_counter=venue_type_counter,
        year_counter=year_counter, oa_true=oa_true,
        citations=citations, velocity=velocity, dtypes_seen=dtypes_seen,
        min_year=args.min_year, max_year=args.max_year,
    )
    print(f"\nDone. Clean file: {out}", flush=True)
    print(f"Rows read {rows_read:,} -> clean {clean_rows:,} "
          f"(dups {dups_removed:,}, bad-year {dropped_year:,}, no-topic {dropped_topic:,})",
          flush=True)


def _first_country(value: object) -> str:
    if _is_blank(value):
        return ""
    first = re.split(r"\s*[;,|]\s*", str(value).strip())[0]
    return first.strip()


def _write_report(*, report_dir: Path, out: Path, rows_read: int, clean_rows: int,
                  dropped_year: int, dropped_topic: int, dups_removed: int,
                  missing_counts: Counter, topic_counter: Counter,
                  country_counter: Counter, venue_type_counter: Counter,
                  year_counter: Counter, oa_true: int,
                  citations: np.ndarray, velocity: np.ndarray,
                  dtypes_seen: dict, min_year: int, max_year: int) -> None:
    denom = max(clean_rows, 1)
    miss = lambda c: 100.0 * missing_counts.get(c, 0) / denom
    years_present = sorted(year_counter)
    lines: list[str] = []
    lines.append("# Preprocess quality report\n")
    lines.append(f"Input rows read: {rows_read:,}")
    lines.append(f"Clean work rows: {clean_rows:,}")
    lines.append(f"Year range kept: {min_year}-{max_year}")
    if years_present:
        lines.append(f"Years actually present: {years_present[0]}-{years_present[-1]}")
    lines.append(f"Duplicate rows removed: {dups_removed:,}")
    lines.append(f"Rows dropped (bad/out-of-range year): {dropped_year:,}")
    lines.append(f"Rows dropped (missing primary_topic): {dropped_topic:,}")
    lines.append(f"Missing title share: {miss('title'):.2f}%")
    lines.append(f"Missing country share: {miss('countries'):.2f}%")
    lines.append(f"Missing venue_type share: {miss('venue_type'):.2f}%")
    lines.append(f"Open-access share: {100.0 * oa_true / denom:.2f}%")

    lines.append("\n## Citation distribution (clean rows)")
    lines.append(f"- citation_count: min {citations.min():.0f}, median {np.median(citations):.0f}, "
                 f"mean {citations.mean():.2f}, max {citations.max():.0f}")
    lines.append(f"- never cited (citation_count == 0): "
                 f"{100.0 * (citations == 0).mean():.2f}%")
    lines.append(f"- citation_velocity: median {np.median(velocity):.3f}, "
                 f"mean {velocity.mean():.3f}, max {velocity.max():.2f}")

    lines.append("\n## Cleaning decisions")
    lines.append("- Dropped rows with missing or out-of-range publication_year.")
    lines.append("- Dropped rows with missing primary_topic.")
    lines.append("- Deduplicated by paper_id first, then by title+year+primary_topic when id is unavailable.")
    lines.append("- Filled numeric impact fields with 0 when missing "
                 f"({', '.join(NUMERIC_FILL_COLS)}).")
    lines.append("- Kept all original columns so downstream cache builders are unchanged.")

    lines.append("\n## Column dtypes (as read)")
    for col, dt in dtypes_seen.items():
        lines.append(f"- {col}: {dt}")

    lines.append("\n## Top 15 primary topics")
    for name, n in topic_counter.most_common(15):
        lines.append(f"- {name}: {n:,}")

    lines.append("\n## Top 15 primary countries")
    for name, n in country_counter.most_common(15):
        lines.append(f"- {name}: {n:,}")

    lines.append("\n## Venue type distribution")
    for name, n in venue_type_counter.most_common(15):
        lines.append(f"- {name}: {n:,}")

    lines.append("\n## Dashboard handoff")
    lines.append(f"Use `{out}` as the single `--input` to every build_*_cache.py script.")
    lines.append("Cache CSV structure is identical to before; only counts reflect the cleaned data.")

    report_path = report_dir / "preprocess_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="Dataset/clean/ai_works_merged_raw.csv",
                        help="Merged raw CSV from merge_raw_dataset.py.")
    parser.add_argument("--output", default="Dataset/clean/ai_works_clean.csv")
    parser.add_argument("--report-dir", default="Dataset/clean/preprocess_report")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    build_clean(args)


if __name__ == "__main__":
    main()
