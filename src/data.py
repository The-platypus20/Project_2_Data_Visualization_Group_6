"""OpenAlex-backed data loading and enrichment for the dashboard.

The app now queries OpenAlex live instead of reading local CSV files. The
resulting dataframe preserves the shape expected by the dashboard modules:

* ``year``              - integer publication year
* ``topic_bucket``      - canonical topic bucket derived from OpenAlex topics
* ``venue_group``       - friendly grouping of the OpenAlex work type
* ``regions``           - list of continents for the paper's countries
* ``country_list``      - list of ISO-2 codes
* ``institution_list``  - list of institution names
* ``sector``            - academia / industry collaboration heuristic
* ``novelty_proxy``     - structural proxy in [0, 1]

The live query intentionally fetches a capped, citation-sorted sample rather
than attempting a full snapshot download through the API.
"""
from __future__ import annotations

from datetime import date
import os

import numpy as np
import pandas as pd
import requests

from . import geo
from . import sector as sectormod

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_YEAR_MIN = 2000
DEFAULT_YEAR_MAX = date.today().year
DEFAULT_QUERY = os.environ.get("OPENALEX_DEFAULT_QUERY", "artificial intelligence")
DEFAULT_MAX_WORKS = int(os.environ.get("OPENALEX_DEFAULT_MAX_WORKS", "2000"))
PER_PAGE = 100

# Canonical AI topic buckets used by the Topic filter, stacked-area chart and
# topic heatmap. Order matters: classification picks the first match.
TOPIC_RULES = [
    ("Computer Vision", ["vision", "image", "face", "expression recognition",
                          "pattern recognition", "object detection", "segmentation",
                          "visual", "video"]),
    ("NLP", ["natural language", "language model", "text ", "speech", "translation",
             "linguistic", "sentiment", "dialogue", "question answering", " nlp"]),
    ("Reinforcement Learning", ["reinforcement"]),
    ("Robotics", ["robot", "autonomous", "manipulation", "motion planning",
                  "control system"]),
    ("Healthcare AI", ["health", "medical", "clinical", "radiolog", "biomedical",
                       "disease", "diagnos", "genomic", "drug", "cancer", "imaging"]),
    ("AI Ethics & Fairness", ["ethic", "fairness", "bias", "privacy", "explainab",
                              "interpretab", "responsible", "governance", "accountab"]),
    ("Search & Recommender", ["information retrieval", "recommend", "ranking",
                              "search engine", "retrieval"]),
    ("ML Theory & Optimization", ["optimization", "statistical learning", "bayesian",
                                  "probabil", "graph theory", "computational theory",
                                  "convex", "stochastic"]),
    ("Core ML / Deep Learning", ["machine learning", "deep learning", "neural network",
                                 "classification", "clustering", "data mining",
                                 "artificial intelligence", "data classification",
                                 "representation learning", "generative"]),
]

TOPIC_BUCKETS = [name for name, _ in TOPIC_RULES] + ["Other"]

_VENUE_MAP = {
    "article": "Article / Journal",
    "preprint": "Preprint",
    "book-chapter": "Book chapter",
}
VENUE_GROUPS = ["Article / Journal", "Preprint", "Book chapter"]

_EMPTY_COLUMNS = [
    "paper_id", "title", "publication_year", "publication_date", "publication_type",
    "citation_count", "citations_per_year", "referenced_works_count", "query_term",
    "primary_topic", "primary_domain", "primary_field", "primary_subfield", "topics",
    "authors", "institutions", "countries", "year", "topic_bucket", "venue_group",
    "country_list", "regions", "institution_list", "sector", "author_count",
    "institution_count", "country_count", "paper_age", "venue_source", "is_oa",
    "oa_status", "doi", "novelty_proxy",
]


def _classify_topic(text: str) -> str:
    t = text.lower()
    for bucket, keywords in TOPIC_RULES:
        if any(k in t for k in keywords):
            return bucket
    return "Other"


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _join(items: list[str]) -> str:
    return "; ".join(_unique_preserve(items))


def _topic_name(obj: dict | None, key: str) -> str:
    if not isinstance(obj, dict):
        return ""
    value = obj.get(key)
    if isinstance(value, dict):
        return str(value.get("display_name") or "")
    return str(value or "")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def _api_key() -> str:
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set OPENALEX_API_KEY before running the dashboard. "
            "Get a free key from https://openalex.org/settings/api."
        )
    return key


def _fetch_page(session: requests.Session, params: dict) -> dict:
    resp = session.get(OPENALEX_WORKS_URL, params=params, timeout=60)
    if resp.status_code == 401:
        raise RuntimeError("OpenAlex rejected OPENALEX_API_KEY (401 Unauthorized).")
    if resp.status_code == 429:
        raise RuntimeError("OpenAlex rate limit reached (429). Reduce max works or retry later.")
    resp.raise_for_status()
    return resp.json()


def _work_to_row(work: dict, query: str) -> dict:
    primary_topic = work.get("primary_topic") or {}
    topics = work.get("topics") or []
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    authorships = work.get("authorships") or []

    author_names: list[str] = []
    institution_names: list[str] = []
    country_codes: list[str] = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            author_names.append(str(author["display_name"]))
        for inst in authorship.get("institutions") or []:
            name = str(inst.get("display_name") or "")
            code = str(inst.get("country_code") or "")
            if name:
                institution_names.append(name)
            if code:
                country_codes.append(code)

    topic_names = [str(topic.get("display_name") or "") for topic in topics]
    countries = _unique_preserve(country_codes)
    institutions = _unique_preserve(institution_names)
    authors = _unique_preserve(author_names)

    publication_year = pd.to_numeric(work.get("publication_year"), errors="coerce")
    if pd.isna(publication_year):
        publication_year = DEFAULT_YEAR_MAX
    publication_year = int(publication_year)
    citation_count = float(pd.to_numeric(work.get("cited_by_count"), errors="coerce") or 0)
    referenced_count = float(
        pd.to_numeric(work.get("referenced_works_count"), errors="coerce") or 0
    )
    paper_age = max(0, date.today().year - publication_year)
    citations_per_year = citation_count / float(paper_age + 1)

    row = {
        "paper_id": str(work.get("id") or ""),
        "title": str(work.get("display_name") or ""),
        "publication_year": publication_year,
        "publication_date": str(work.get("publication_date") or ""),
        "publication_type": str(work.get("type") or ""),
        "citation_count": citation_count,
        "citations_per_year": citations_per_year,
        "referenced_works_count": referenced_count,
        "query_term": query,
        "primary_topic": str(primary_topic.get("display_name") or ""),
        "primary_domain": _topic_name(primary_topic, "domain"),
        "primary_field": _topic_name(primary_topic, "field"),
        "primary_subfield": _topic_name(primary_topic, "subfield"),
        "topics": _join(topic_names),
        "authors": _join(authors),
        "institutions": _join(institutions),
        "countries": _join(countries),
        "year": publication_year,
        "country_list": countries,
        "regions": sorted({geo.region(code) for code in countries}),
        "institution_list": institutions,
        "sector": sectormod.paper_sector(institutions),
        "author_count": len(authors),
        "institution_count": len(institutions),
        "country_count": len(countries),
        "paper_age": paper_age,
        "venue_source": str(source.get("display_name") or ""),
        "is_oa": bool(open_access.get("is_oa", False)),
        "oa_status": str(open_access.get("oa_status") or ""),
        "doi": str(work.get("doi") or ""),
    }

    topic_text = " ; ".join([
        row["primary_subfield"],
        row["primary_topic"],
        row["topics"],
    ])
    row["topic_bucket"] = _classify_topic(topic_text)
    row["venue_group"] = _VENUE_MAP.get(row["publication_type"], "Other")
    return row


def load_data(
    query: str,
    year_min: int = DEFAULT_YEAR_MIN,
    year_max: int = DEFAULT_YEAR_MAX,
    max_records: int = DEFAULT_MAX_WORKS,
) -> pd.DataFrame:
    """Fetch a citation-sorted sample of works from OpenAlex and enrich it."""
    query = (query or "").strip()
    if not query:
        raise RuntimeError("Enter an OpenAlex search query before loading data.")

    max_records = max(1, int(max_records))
    session = requests.Session()
    session.headers.update({"User-Agent": "raw_dashboard-openalex/1.0"})

    params = {
        "api_key": _api_key(),
        "search": query,
        "filter": f"publication_year:{year_min}-{year_max}",
        "sort": "cited_by_count:desc",
        "select": ",".join([
            "id",
            "display_name",
            "doi",
            "publication_year",
            "publication_date",
            "type",
            "cited_by_count",
            "referenced_works_count",
            "topics",
            "primary_topic",
            "authorships",
            "primary_location",
            "open_access",
        ]),
        "per_page": min(PER_PAGE, max_records),
        "cursor": "*",
    }

    rows: list[dict] = []
    while len(rows) < max_records:
        payload = _fetch_page(session, params)
        results = payload.get("results") or []
        if not results:
            break
        for work in results:
            rows.append(_work_to_row(work, query))
            if len(rows) >= max_records:
                break
        next_cursor = ((payload.get("meta") or {}).get("next_cursor"))
        if not next_cursor:
            break
        params["cursor"] = next_cursor

    if not rows:
        return empty_frame()

    df = pd.DataFrame(rows)
    ref_pct = df["referenced_works_count"].rank(pct=True)
    df["novelty_proxy"] = (1.0 - ref_pct).round(3)

    for col in _EMPTY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[_EMPTY_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers (operate on an already-filtered DataFrame)
# ---------------------------------------------------------------------------
def papers_per_year(df: pd.DataFrame) -> pd.Series:
    return df.groupby("year").size().sort_index()


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, country)."""
    out = df.explode("country_list").rename(columns={"country_list": "iso2"})
    out = out[out["iso2"].notna() & (out["iso2"] != "")].copy()
    out["country"] = out["iso2"].map(geo.name)
    out["iso3"] = out["iso2"].map(geo.iso3)
    out["region"] = out["iso2"].map(geo.region)
    out["pop_m"] = out["iso2"].map(geo.population_m)
    return out


def explode_institutions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, institution), tagged with inferred sector."""
    out = df.assign(institution=df["institution_list"]).explode("institution")
    out = out[out["institution"].notna() & (out["institution"] != "")].copy()
    out["inst_sector"] = out["institution"].map(sectormod.classify_institution)
    return out
