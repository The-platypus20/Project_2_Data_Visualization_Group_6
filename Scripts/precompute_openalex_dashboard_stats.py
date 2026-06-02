"""Precompute dashboard-ready OpenAlex statistics in one offline run.

This script uses exact OpenAlex group-by queries wherever possible, and falls
back to a stratified row-level sample only for dashboard elements that require
paper-level distributions or heuristics.

Outputs are written to `data/openalex_dashboard_stats/` by default:

* `exact_year_counts.csv`
* `exact_country_counts.csv`
* `exact_country_year_counts.csv`
* `exact_type_counts.csv`
* `exact_type_year_counts.csv`
* `exact_oa_status_counts.csv`
* `exact_oa_status_year_counts.csv`
* `exact_subfield_counts.csv`
* `exact_subfield_year_counts.csv`
* `exact_topic_bucket_year_counts.csv`
* `exact_top_institutions_by_country.csv`
* `sample_plan.csv`
* `sampled_papers.csv`
* `manifest.json`
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import geo
from src import sector as sectormod
from src.data import TOPIC_RULES

BASE_URL = "https://api.openalex.org/works"
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025
DEFAULT_SAMPLE_SIZE = 30000
DEFAULT_TOP_COUNTRIES = 25
DEFAULT_TOP_INSTITUTIONS = 15
DEFAULT_OUTPUT_DIR = ROOT / "data" / "openalex_dashboard_stats"

SELECT_FIELDS = ",".join([
    "id",
    "doi",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "cited_by_count",
    "referenced_works_count",
    "authorships",
    "topics",
    "primary_topic",
    "primary_location",
    "language",
    "open_access",
    "is_retracted",
    "is_paratext",
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--top-countries", type=int, default=DEFAULT_TOP_COUNTRIES)
    parser.add_argument("--top-institutions", type=int, default=DEFAULT_TOP_INSTITUTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-filter",
        default=None,
        help="Override the default OpenAlex filter string.",
    )
    parser.add_argument(
        "--mailto",
        default=os.environ.get("OPENALEX_EMAIL", "").strip(),
        help="Contact email to pass to OpenAlex via mailto/User-Agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260602,
        help="Base seed used for reproducible per-year sampling.",
    )
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help="Only compute exact grouped statistics and skip row-level sample export.",
    )
    return parser.parse_args()


def api_key() -> str:
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENALEX_API_KEY is required. Set it in the shell before running this script."
        )
    return key


def build_base_filter(start_year: int, end_year: int) -> str:
    return ",".join([
        "primary_topic.subfield.id:1702",
        "has_doi:true",
        "is_retracted:false",
        "is_paratext:false",
        "type:article|preprint|book-chapter|proceedings-article",
        f"from_publication_date:{start_year}-01-01",
        f"to_publication_date:{end_year}-12-31",
    ])


def request_openalex(
    session: requests.Session,
    params: dict,
    *,
    max_retries: int = 6,
    sleep_base: float = 2.0,
) -> dict:
    params = dict(params)
    params["api_key"] = api_key()

    for attempt in range(max_retries):
        response = session.get(BASE_URL, params=params, timeout=90)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {401, 403}:
            raise RuntimeError(
                f"OpenAlex authentication failed ({response.status_code}). "
                "Check OPENALEX_API_KEY."
            )
        if response.status_code == 429:
            wait = sleep_base ** attempt
            print(f"Rate limited by OpenAlex. Sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue
        if 500 <= response.status_code < 600:
            wait = sleep_base ** attempt
            print(f"OpenAlex server error {response.status_code}. Sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"OpenAlex request failed with status {response.status_code}: "
            f"{response.text[:300]}"
        )

    raise RuntimeError("OpenAlex request failed after retries.")


def _classify_topic(text: str) -> str:
    value = text.lower()
    for bucket, keywords in TOPIC_RULES:
        if any(keyword in value for keyword in keywords):
            return bucket
    return "Other"


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _join(values: list[str]) -> str:
    return "; ".join(_unique_preserve(values))


def _topic_name(obj: dict | None, key: str) -> str:
    if not isinstance(obj, dict):
        return ""
    value = obj.get(key)
    if isinstance(value, dict):
        return str(value.get("display_name") or "")
    return str(value or "")


def _normalize_country_code(value) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if "/" in text:
        text = text.rstrip("/").split("/")[-1]
    return text.upper()


def _to_int(value, default: int = 0) -> int:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return int(num)


def _to_float(value, default: float = 0.0) -> float:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return float(num)


def parse_group_rows(payload: dict, group_col: str, name_col: str = "display_name") -> pd.DataFrame:
    rows = []
    for group in payload.get("group_by", []):
        rows.append({
            group_col: group.get("key"),
            name_col: group.get("key_display_name") or group.get("display_name") or group.get("key"),
            "paper_count": int(group.get("count", 0)),
        })
    return pd.DataFrame(rows)


def fetch_group_counts(
    session: requests.Session,
    *,
    filter_str: str,
    group_by: str,
    group_col: str,
    name_col: str = "display_name",
    extra_params: dict | None = None,
) -> pd.DataFrame:
    params = {
        "filter": filter_str,
        "group_by": group_by,
        "per-page": 200,
    }
    if extra_params:
        params.update(extra_params)
    payload = request_openalex(session, params)
    return parse_group_rows(payload, group_col=group_col, name_col=name_col)


def fetch_year_counts(session: requests.Session, base_filter: str, start_year: int, end_year: int) -> pd.DataFrame:
    df = fetch_group_counts(
        session,
        filter_str=base_filter,
        group_by="publication_year",
        group_col="publication_year",
        name_col="publication_year_label",
    )
    if df.empty:
        years = pd.DataFrame({"publication_year": list(range(start_year, end_year + 1))})
        years["paper_count"] = 0
        return years
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df.dropna(subset=["publication_year"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)
    years = pd.DataFrame({"publication_year": list(range(start_year, end_year + 1))})
    df = years.merge(df[["publication_year", "paper_count"]], on="publication_year", how="left")
    df["paper_count"] = df["paper_count"].fillna(0).astype(int)
    return df.sort_values("publication_year").reset_index(drop=True)


def fetch_counts_by_year(
    session: requests.Session,
    *,
    base_filter: str,
    years: list[int],
    group_by: str,
    group_col: str,
    name_col: str = "display_name",
) -> pd.DataFrame:
    frames = []
    for year in years:
        filter_str = f"{base_filter},publication_year:{year}"
        frame = fetch_group_counts(
            session,
            filter_str=filter_str,
            group_by=group_by,
            group_col=group_col,
            name_col=name_col,
        )
        if frame.empty:
            continue
        frame["publication_year"] = year
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["publication_year", group_col, name_col, "paper_count"])
    return pd.concat(frames, ignore_index=True)


def allocate_stratified_sample(year_counts: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    plan = year_counts.copy()
    total_population = int(plan["paper_count"].sum())
    if total_population == 0:
        plan["target_n"] = 0
        plan["sample_weight"] = np.nan
        return plan

    plan["target_n"] = (
        plan["paper_count"] / total_population * sample_size
    ).round().astype(int)
    plan.loc[(plan["paper_count"] > 0) & (plan["target_n"] == 0), "target_n"] = 1
    plan["target_n"] = np.minimum(plan["target_n"], plan["paper_count"])

    diff = sample_size - int(plan["target_n"].sum())
    if diff != 0:
        idxs = plan.sort_values("paper_count", ascending=False).index.tolist()
        step = 1 if diff > 0 else -1
        moved = 0
        pointer = 0
        while moved < abs(diff) and idxs:
            idx = idxs[pointer % len(idxs)]
            new_value = int(plan.loc[idx, "target_n"]) + step
            upper = int(plan.loc[idx, "paper_count"])
            if 0 <= new_value <= upper:
                plan.loc[idx, "target_n"] = new_value
                moved += 1
            pointer += 1
            if pointer > len(idxs) * max(sample_size, 1):
                break

    plan["sample_weight"] = np.where(
        plan["target_n"] > 0,
        plan["paper_count"] / plan["target_n"],
        np.nan,
    )
    return plan


def fetch_sample_for_year(
    session: requests.Session,
    *,
    filter_str: str,
    year: int,
    n: int,
    sample_weight: float,
    seed: int,
) -> list[dict]:
    if n <= 0:
        return []

    params = {
        "filter": f"{filter_str},publication_year:{year}",
        "sample": n,
        "seed": seed,
        "select": SELECT_FIELDS,
        "per_page": 100,
        "cursor": "*",
    }

    works: list[dict] = []
    while len(works) < n:
        payload = request_openalex(session, params)
        batch = payload.get("results") or []
        if not batch:
            break
        works.extend(batch)
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor

    rows = []
    for work in works[:n]:
        row = normalize_work(work)
        row["sample_year"] = year
        row["sample_weight"] = sample_weight
        rows.append(row)
    return rows


def normalize_work(work: dict) -> dict:
    primary_topic = work.get("primary_topic") or {}
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    authorships = work.get("authorships") or []
    topics = work.get("topics") or []

    author_names: list[str] = []
    institution_names: list[str] = []
    institution_ids: list[str] = []
    country_codes: list[str] = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            author_names.append(str(author["display_name"]))
        for inst in authorship.get("institutions") or []:
            name = str(inst.get("display_name") or "")
            inst_id = str(inst.get("id") or "")
            code = str(inst.get("country_code") or "")
            if name:
                institution_names.append(name)
            if inst_id:
                institution_ids.append(inst_id)
            if code:
                country_codes.append(code)

    authors = _unique_preserve(author_names)
    institutions = _unique_preserve(institution_names)
    countries = _unique_preserve(country_codes)
    topic_names = [str(topic.get("display_name") or "") for topic in topics]

    publication_year = _to_int(work.get("publication_year"), default=0)
    citation_count = _to_float(work.get("cited_by_count"), default=0.0)
    referenced_count = _to_float(work.get("referenced_works_count"), default=0.0)
    current_year = datetime.now(UTC).year
    paper_age = max(0, current_year - publication_year)

    topic_text = " ; ".join([
        _topic_name(primary_topic, "subfield"),
        str(primary_topic.get("display_name") or ""),
        _join(topic_names),
    ])

    publication_type = str(work.get("type") or "")
    if publication_type == "article":
        venue_group = "Article / Journal"
    elif publication_type == "preprint":
        venue_group = "Preprint"
    elif publication_type == "book-chapter":
        venue_group = "Book chapter"
    else:
        venue_group = "Other"

    return {
        "paper_id": str(work.get("id") or ""),
        "title": str(work.get("display_name") or ""),
        "publication_year": publication_year,
        "publication_date": str(work.get("publication_date") or ""),
        "publication_type": publication_type,
        "citation_count": citation_count,
        "citations_per_year": citation_count / float(paper_age + 1),
        "referenced_works_count": referenced_count,
        "topics": _join(topic_names),
        "primary_topic": str(primary_topic.get("display_name") or ""),
        "primary_subfield": _topic_name(primary_topic, "subfield"),
        "primary_field": _topic_name(primary_topic, "field"),
        "primary_domain": _topic_name(primary_topic, "domain"),
        "venue_source": str(source.get("display_name") or ""),
        "venue_type": str(source.get("type") or ""),
        "authors": _join(authors),
        "institutions": _join(institutions),
        "institution_ids": _join(_unique_preserve(institution_ids)),
        "countries": _join(countries),
        "doi": str(work.get("doi") or ""),
        "language": str(work.get("language") or ""),
        "is_oa": bool(open_access.get("is_oa", False)),
        "oa_status": str(open_access.get("oa_status") or ""),
        "paper_age": paper_age,
        "author_count": len(authors),
        "institution_count": len(institutions),
        "country_count": len(countries),
        "topic_bucket": _classify_topic(topic_text),
        "venue_group": venue_group,
        "sector": sectormod.paper_sector(institutions),
    }


def enrich_sample_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(novelty_proxy=pd.Series(dtype=float))
    ref_pct = df["referenced_works_count"].rank(pct=True)
    df = df.copy()
    df["novelty_proxy"] = (1.0 - ref_pct).round(3)
    return df


def write_manifest(out_dir: Path, payload: dict) -> None:
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_filter = args.base_filter or build_base_filter(args.start_year, args.end_year)
    years = list(range(args.start_year, args.end_year + 1))

    session = requests.Session()
    headers = {"User-Agent": "openalex-dashboard-precompute/1.0"}
    if args.mailto:
        headers["User-Agent"] = f"mailto:{args.mailto}"
    session.headers.update(headers)

    print("Fetching exact yearly counts...")
    year_counts = fetch_year_counts(session, base_filter, args.start_year, args.end_year)
    year_counts.to_csv(out_dir / "exact_year_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact type counts...")
    type_counts = fetch_group_counts(
        session,
        filter_str=base_filter,
        group_by="type",
        group_col="publication_type",
        name_col="publication_type_label",
    )
    type_counts.to_csv(out_dir / "exact_type_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact type-by-year counts...")
    type_year_counts = fetch_counts_by_year(
        session,
        base_filter=base_filter,
        years=years,
        group_by="type",
        group_col="publication_type",
        name_col="publication_type_label",
    )
    type_year_counts.to_csv(out_dir / "exact_type_year_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact country counts...")
    country_counts = fetch_group_counts(
        session,
        filter_str=base_filter,
        group_by="institutions.country_code",
        group_col="country_code",
        name_col="country_name",
    )
    country_counts["country_code"] = country_counts["country_code"].map(_normalize_country_code)
    country_counts["region"] = country_counts["country_code"].map(geo.region)
    country_counts["country_name"] = country_counts["country_code"].map(geo.name).fillna(country_counts["country_name"])
    country_counts.to_csv(out_dir / "exact_country_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact country-by-year counts...")
    country_year_counts = fetch_counts_by_year(
        session,
        base_filter=base_filter,
        years=years,
        group_by="institutions.country_code",
        group_col="country_code",
        name_col="country_name",
    )
    country_year_counts["country_code"] = country_year_counts["country_code"].map(_normalize_country_code)
    country_year_counts["region"] = country_year_counts["country_code"].map(geo.region)
    country_year_counts["country_name"] = (
        country_year_counts["country_code"].map(geo.name).fillna(country_year_counts["country_name"])
    )
    country_year_counts.to_csv(out_dir / "exact_country_year_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact OA-status counts...")
    oa_counts = fetch_group_counts(
        session,
        filter_str=base_filter,
        group_by="open_access.oa_status",
        group_col="oa_status",
        name_col="oa_status_label",
    )
    oa_counts.to_csv(out_dir / "exact_oa_status_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact OA-status-by-year counts...")
    oa_year_counts = fetch_counts_by_year(
        session,
        base_filter=base_filter,
        years=years,
        group_by="open_access.oa_status",
        group_col="oa_status",
        name_col="oa_status_label",
    )
    oa_year_counts.to_csv(out_dir / "exact_oa_status_year_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact subfield counts...")
    subfield_counts = fetch_group_counts(
        session,
        filter_str=base_filter,
        group_by="primary_topic.subfield.id",
        group_col="subfield_id",
        name_col="subfield_name",
    )
    subfield_counts["topic_bucket"] = subfield_counts["subfield_name"].fillna("").map(_classify_topic)
    subfield_counts.to_csv(out_dir / "exact_subfield_counts.csv", index=False, encoding="utf-8-sig")

    print("Fetching exact subfield-by-year counts...")
    subfield_year_counts = fetch_counts_by_year(
        session,
        base_filter=base_filter,
        years=years,
        group_by="primary_topic.subfield.id",
        group_col="subfield_id",
        name_col="subfield_name",
    )
    subfield_year_counts["topic_bucket"] = (
        subfield_year_counts["subfield_name"].fillna("").map(_classify_topic)
    )
    subfield_year_counts.to_csv(
        out_dir / "exact_subfield_year_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Fetching exact primary-topic-by-year counts...")
    primary_topic_year_counts = fetch_counts_by_year(
        session,
        base_filter=base_filter,
        years=years,
        group_by="primary_topic.id",
        group_col="primary_topic_id",
        name_col="primary_topic_name",
    )
    primary_topic_year_counts["topic_bucket"] = (
        primary_topic_year_counts["primary_topic_name"].fillna("").map(_classify_topic)
    )
    topic_bucket_year = (
        primary_topic_year_counts.groupby(["publication_year", "topic_bucket"], as_index=False)["paper_count"]
        .sum()
        .sort_values(["publication_year", "paper_count"], ascending=[True, False])
    )
    topic_bucket_year.to_csv(
        out_dir / "exact_topic_bucket_year_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_topic_year_counts.to_csv(
        out_dir / "exact_primary_topic_year_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Fetching exact top institutions for top countries...")
    top_country_codes = (
        country_counts.sort_values("paper_count", ascending=False)["country_code"]
        .dropna()
        .astype(str)
        .head(args.top_countries)
        .tolist()
    )
    institution_frames = []
    for code in top_country_codes:
        frame = fetch_group_counts(
            session,
            filter_str=f"{base_filter},institutions.country_code:{code}",
            group_by="institutions.id",
            group_col="institution_id",
            name_col="institution_name",
        )
        if frame.empty:
            continue
        frame["country_code"] = code
        frame["country_name"] = geo.name(code)
        institution_frames.append(frame.sort_values("paper_count", ascending=False).head(args.top_institutions))
    top_institutions = (
        pd.concat(institution_frames, ignore_index=True)
        if institution_frames
        else pd.DataFrame(columns=["institution_id", "institution_name", "paper_count", "country_code", "country_name"])
    )
    top_institutions.to_csv(
        out_dir / "exact_top_institutions_by_country.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_plan = allocate_stratified_sample(year_counts, args.sample_size)
    sample_plan.to_csv(out_dir / "sample_plan.csv", index=False, encoding="utf-8-sig")

    sampled_papers = pd.DataFrame()
    if not args.skip_sample:
        print("Fetching stratified row-level sample...")
        rows: list[dict] = []
        for _, rec in sample_plan.iterrows():
            year = int(rec["publication_year"])
            n = int(rec["target_n"])
            weight = float(rec["sample_weight"]) if not pd.isna(rec["sample_weight"]) else math.nan
            rows.extend(
                fetch_sample_for_year(
                    session,
                    filter_str=base_filter,
                    year=year,
                    n=n,
                    sample_weight=weight,
                    seed=args.seed + year,
                )
            )
            print(f"  {year}: fetched {n} sampled works")
        sampled_papers = enrich_sample_frame(pd.DataFrame(rows))
        sampled_papers.to_csv(out_dir / "sampled_papers.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "base_filter": base_filter,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "sample_size_requested": args.sample_size,
        "sample_size_materialized": int(len(sampled_papers)),
        "top_countries_for_institution_drilldown": args.top_countries,
        "top_institutions_per_country": args.top_institutions,
        "files": sorted([path.name for path in out_dir.iterdir() if path.is_file()]),
    }
    write_manifest(out_dir, manifest)

    print(f"Done. Wrote dashboard stats to {out_dir}")


if __name__ == "__main__":
    main()
