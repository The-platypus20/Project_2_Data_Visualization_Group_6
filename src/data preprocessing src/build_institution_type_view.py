"""
Build institution-type view from 3 raw OpenAlex dashboard files.

Goal:
- Combine 3 raw files.
- Extract institutions per paper.
- Classify institutions into University, Business, Government, Healthcare, Nonprofit, Other.
- Save clean CSV files for dashboard / manual inspection.

Example:
python src/build_institution_type_view.py \
  --inputs Dataset/raw/ai_works_2000_2009.csv Dataset/raw/ai_works_2010_2019.csv Dataset/raw/ai_works_2020_2025.csv \
  --outdir Dataset/dashboard_cache

Outputs:
- institution_type_paper_view.csv
- institution_type_year_summary.csv
- institution_type_top_institutions.csv
- institution_type_country_year.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


UNIVERSITY_KEYWORDS = [
    "university", "université", "universidad", "universita", "universität",
    "college", "school of", "institute of technology", "polytechnic",
    "academy", "faculty of", "department of", "higher education",
]

BUSINESS_KEYWORDS = [
    "inc", "ltd", "llc", "corp", "corporation", "company", "co.", "limited",
    "gmbh", "s.a.", "sa", "plc", "ag", "pte", "bv", "nv",
    "google", "microsoft", "meta", "facebook", "amazon", "apple", "ibm", "intel",
    "nvidia", "openai", "deepmind", "anthropic", "huawei", "tencent", "alibaba",
    "baidu", "samsung", "sony", "siemens", "bosch", "oracle", "salesforce",
    "adobe", "nec", "fujitsu", "qualcomm", "tesla",
]

GOVERNMENT_KEYWORDS = [
    "government", "ministry", "department of defense", "national laboratory",
    "national lab", "army", "navy", "air force", "nasa", "nih", "nsf",
    "national institute", "agency", "commission", "council",
]

HEALTHCARE_KEYWORDS = [
    "hospital", "medical center", "clinic", "health system", "healthcare",
    "cancer center", "children's hospital", "nhs", "medical school",
]

NONPROFIT_KEYWORDS = [
    "foundation", "nonprofit", "non-profit", "charity", "association",
    "society", "institute for", "research institute", "laboratory for",
]

OPENALEX_TYPE_MAP = {
    "education": "University",
    "company": "Business",
    "government": "Government",
    "healthcare": "Healthcare",
    "nonprofit": "Nonprofit",
    "facility": "Other",
    "archive": "Other",
    "other": "Other",
}


# ---------- File loading ----------


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported file type: {path}")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------- Parsing helpers ----------


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

    # JSON first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Python literal second
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def split_text_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parsed = safe_parse(value)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, str):
        # Handles strings like "A; B; C" or "A|B|C"
        parts = re.split(r"\s*[;|]\s*", parsed)
        return [p.strip() for p in parts if p.strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def normalize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    return name


def normalize_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    return s.replace("https://openalex.org/", "")


# ---------- Institution extraction ----------


def extract_from_authorships(row: pd.Series, authorships_col: str) -> list[dict[str, Any]]:
    parsed = safe_parse(row.get(authorships_col))
    if not isinstance(parsed, list):
        return []

    institutions: list[dict[str, Any]] = []
    for author in parsed:
        if not isinstance(author, dict):
            continue
        for inst in author.get("institutions", []) or []:
            if not isinstance(inst, dict):
                continue
            institutions.append({
                "institution_id": normalize_id(inst.get("id") or inst.get("openalex_id")),
                "institution_name": normalize_name(inst.get("display_name") or inst.get("name") or ""),
                "openalex_type": str(inst.get("type") or "").strip().lower() or None,
                "country_code": inst.get("country_code") or inst.get("country") or None,
            })
    return institutions


def extract_from_institutions_col(row: pd.Series, institutions_col: str) -> list[dict[str, Any]]:
    parsed = safe_parse(row.get(institutions_col))
    institutions: list[dict[str, Any]] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                institutions.append({
                    "institution_id": normalize_id(item.get("id") or item.get("openalex_id")),
                    "institution_name": normalize_name(item.get("display_name") or item.get("name") or ""),
                    "openalex_type": str(item.get("type") or "").strip().lower() or None,
                    "country_code": item.get("country_code") or item.get("country") or None,
                })
            else:
                name = normalize_name(str(item))
                if name:
                    institutions.append({
                        "institution_id": None,
                        "institution_name": name,
                        "openalex_type": None,
                        "country_code": None,
                    })
        return institutions

    if isinstance(parsed, str):
        for name in split_text_list(parsed):
            institutions.append({
                "institution_id": None,
                "institution_name": normalize_name(name),
                "openalex_type": None,
                "country_code": None,
            })

    return institutions


def extract_institutions(row: pd.Series, cols: dict[str, str | None]) -> list[dict[str, Any]]:
    if cols.get("authorships"):
        out = extract_from_authorships(row, cols["authorships"])
        if out:
            return out

    if cols.get("institutions"):
        out = extract_from_institutions_col(row, cols["institutions"])
        if out:
            return out

    return []


# ---------- Classification ----------


def has_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def classify_institution(name: str, openalex_type: str | None = None) -> str:
    typ = (openalex_type or "").strip().lower()
    if typ in OPENALEX_TYPE_MAP:
        return OPENALEX_TYPE_MAP[typ]

    n = name.lower().strip()
    if not n:
        return "Unknown"

    # Order matters: medical schools are education unless name is clearly hospital/center.
    if has_keyword(n, HEALTHCARE_KEYWORDS):
        return "Healthcare"
    if has_keyword(n, UNIVERSITY_KEYWORDS):
        return "University"
    if has_keyword(n, GOVERNMENT_KEYWORDS):
        return "Government"
    if has_keyword(n, BUSINESS_KEYWORDS):
        return "Business"


# ---------- Build view ----------


def build_long_view(inputs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in inputs:
        df = read_any_table(path)
        df["_source_file"] = path.name

        cols = {
            "work_id": find_col(df, ["id", "work_id", "openalex_id", "doi"]),
            "year": find_col(df, ["year", "publication_year", "pub_year"]),
            "title": find_col(df, ["title", "display_name", "work_title"]),
            "fwci": find_col(df, ["fwci", "field_weighted_citation_impact"]),
            "citation_count": find_col(df, ["citation_count", "cited_by_count", "citations"]),
            "citation_velocity": find_col(df, ["citation_velocity", "citations_per_year"]),
            "topic": find_col(df, ["primary_topic", "topic", "topic_label"]),
            "subfield": find_col(df, ["primary_subfield", "subfield"]),
            "field": find_col(df, ["primary_field", "field"]),
            "countries": find_col(df, ["countries", "country", "authorship_countries"]),
            "authorships": find_col(df, ["authorships", "raw_authorships"]),
            "institutions": find_col(df, ["institutions", "authorship_institutions", "institution_names"]),
        }

        if cols["year"] is None:
            raise ValueError(f"No year column found in {path}. Columns: {list(df.columns)[:30]}")

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            institutions = extract_institutions(row, cols)
            if not institutions:
                continue

            work_id = row.get(cols["work_id"]) if cols["work_id"] else None
            year = row.get(cols["year"])
            title = row.get(cols["title"]) if cols["title"] else None
            fwci = row.get(cols["fwci"]) if cols["fwci"] else None
            citation_count = row.get(cols["citation_count"]) if cols["citation_count"] else None
            citation_velocity = row.get(cols["citation_velocity"]) if cols["citation_velocity"] else None
            topic = row.get(cols["topic"]) if cols["topic"] else None
            subfield = row.get(cols["subfield"]) if cols["subfield"] else None
            field = row.get(cols["field"]) if cols["field"] else None

            for inst in institutions:
                name = normalize_name(inst.get("institution_name", ""))
                if not name:
                    continue
                inst_type = classify_institution(name, inst.get("openalex_type"))
                records.append({
                    "work_id": work_id,
                    "year": year,
                    "title": title,
                    "institution_id": inst.get("institution_id"),
                    "institution_name": name,
                    "openalex_type": inst.get("openalex_type"),
                    "institution_type": inst_type,
                    "country_code": inst.get("country_code"),
                    "topic": topic,
                    "subfield": subfield,
                    "field": field,
                    "fwci": fwci,
                    "citation_count": citation_count,
                    "citation_velocity": citation_velocity,
                    "source_file": path.name,
                })

        part = pd.DataFrame(records)
        print(f"{path.name}: {len(df):,} works -> {len(part):,} work-institution rows")
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for col in ["fwci", "citation_count", "citation_velocity"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Deduplicate same paper-institution pairs.
    out["_dedupe_key"] = (
        out["work_id"].astype(str).fillna("")
        + "|" + out["institution_id"].astype(str).fillna("")
        + "|" + out["institution_name"].str.lower().fillna("")
    )
    out = out.drop_duplicates("_dedupe_key").drop(columns=["_dedupe_key"])
    return out


def save_outputs(long_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    paper_view_path = outdir / "institution_type_paper_view.csv"
    long_df.to_csv(paper_view_path, index=False)

    year_summary = (
        long_df
        .dropna(subset=["year"])
        .groupby(["year", "institution_type"], dropna=False)
        .agg(
            paper_institution_rows=("work_id", "count"),
            unique_papers=("work_id", "nunique"),
            unique_institutions=("institution_name", "nunique"),
            median_fwci=("fwci", "median"),
            mean_fwci=("fwci", "mean"),
            median_citation_velocity=("citation_velocity", "median"),
            mean_citation_velocity=("citation_velocity", "mean"),
        )
        .reset_index()
    )
    year_summary.to_csv(outdir / "institution_type_year_summary.csv", index=False)

    top_inst = (
        long_df
        .groupby(["institution_type", "institution_name"], dropna=False)
        .agg(
            paper_institution_rows=("work_id", "count"),
            unique_papers=("work_id", "nunique"),
            active_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            median_fwci=("fwci", "median"),
            mean_fwci=("fwci", "mean"),
            median_citation_velocity=("citation_velocity", "median"),
        )
        .reset_index()
        .sort_values(["institution_type", "unique_papers"], ascending=[True, False])
    )
    top_inst.to_csv(outdir / "institution_type_top_institutions.csv", index=False)

    country_year = (
        long_df
        .dropna(subset=["year"])
        .groupby(["year", "country_code", "institution_type"], dropna=False)
        .agg(
            paper_institution_rows=("work_id", "count"),
            unique_papers=("work_id", "nunique"),
            unique_institutions=("institution_name", "nunique"),
            median_fwci=("fwci", "median"),
        )
        .reset_index()
    )
    country_year.to_csv(outdir / "institution_type_country_year.csv", index=False)

    print("\nSaved:")
    for name in [
        "institution_type_paper_view.csv",
        "institution_type_year_summary.csv",
        "institution_type_top_institutions.csv",
        "institution_type_country_year.csv",
    ]:
        print(f"- {outdir / name}")

    print("\nInstitution type counts:")
    print(long_df["institution_type"].value_counts(dropna=False).to_string())


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Three raw OpenAlex files. CSV, Parquet, JSON, and JSONL are supported.",
    )
    parser.add_argument(
        "--outdir",
        default="Dataset/dashboard_cache",
        help="Output directory for institution type CSV files.",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    long_df = build_long_view(inputs)
    if long_df.empty:
        raise RuntimeError("No institutions extracted. Check institution/authorship columns in raw files.")

    save_outputs(long_df, Path(args.outdir))


if __name__ == "__main__":
    main()
