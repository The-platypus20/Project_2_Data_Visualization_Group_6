"""Export OpenAlex AI works in resumable chunks for dashboard analysis.

This script uses OpenAlex list+filter pagination rather than search. It is
designed for large offline exports such as the first 1,000,000 AI works with
dashboard-compatible columns.

Output layout by default:

* `Dataset/openalex_exports/ai_works_<start>_<end>/part_00001.csv`
* `Dataset/openalex_exports/ai_works_<start>_<end>/part_00002.csv`
* ...
* `Dataset/openalex_exports/ai_works_<start>_<end>/manifest.json`

The export is resumable. If interrupted, re-run the same command and it will
continue from the last completed chunk. The buffer between chunk writes is also
checkpointed, so a collaborator can pull the repo and resume without
re-crawling or duplicating finished paper IDs.
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

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import geo

BASE_URL = "https://api.openalex.org/works"
RATE_LIMIT_URL = "https://api.openalex.org/rate-limit"
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025
DEFAULT_TARGET_ROWS = 1_000_000
DEFAULT_CHUNK_ROWS = 10_000
DEFAULT_EXPORT_BASE_DIR = ROOT / "Dataset" / "openalex_exports"
DEFAULT_QUERY_LABEL = "primary_topic.subfield.id:1702"
DEFAULT_MAX_DAILY_LIST_CALLS = 9500
DEFAULT_HEARTBEAT_PAGES = 25
BUFFER_CHECKPOINT_NAME = "_buffer_checkpoint.csv"
PER_PAGE = 100

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
    "keywords",
    "language",
    "open_access",
    "fwci",
    "is_retracted",
    "is_paratext",
])

EXPORT_COLUMNS = [
    "paper_id",
    "title",
    "publication_year",
    "publication_date",
    "publication_type",
    "citation_count",
    "citations_per_year",
    "fwci",
    "referenced_works_count",
    "topics",
    "keywords",
    "primary_topic",
    "primary_subfield",
    "primary_field",
    "primary_domain",
    "venue_source",
    "venue_type",
    "authors",
    "institutions",
    "institution_ids",
    "countries",
    "country_names",
    "doi",
    "language",
    "is_oa",
    "oa_status",
    "search_term_used",
    "paper_age",
    "citation_velocity",
    "author_count",
    "institution_count",
    "country_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="Row target. Use 0 to mean all matching works for the shard.",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a tracked year-shard path under Dataset/openalex_exports/.",
    )
    parser.add_argument(
        "--job-name",
        default="",
        help="Optional shard/job name. Defaults to ai_works_<start>_<end>.",
    )
    parser.add_argument(
        "--base-filter",
        default=None,
        help="Override the default OpenAlex filter string.",
    )
    parser.add_argument(
        "--query-label",
        default=DEFAULT_QUERY_LABEL,
        help="Value written to `search_term_used` in the exported rows.",
    )
    parser.add_argument(
        "--mailto",
        default=os.environ.get("OPENALEX_EMAIL", "").strip(),
        help="Contact email to pass to OpenAlex via User-Agent.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing manifest/state in the output directory.",
    )
    parser.add_argument(
        "--max-daily-list-calls",
        type=int,
        default=DEFAULT_MAX_DAILY_LIST_CALLS,
        help=(
            "Safety cap for list+filter calls in one run. "
            "Default leaves headroom under the free-tier 10,000-call/day limit."
        ),
    )
    parser.add_argument(
        "--allow-full-budget",
        action="store_true",
        help="Disable the safety cap and allow use of the full estimated daily budget.",
    )
    parser.add_argument(
        "--heartbeat-pages",
        type=int,
        default=DEFAULT_HEARTBEAT_PAGES,
        help="Print progress every N API pages.",
    )
    return parser.parse_args()


def api_key() -> str:
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENALEX_API_KEY is required.")
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


def default_job_name(start_year: int, end_year: int, job_name: str = "") -> str:
    text = str(job_name or "").strip()
    if text:
        return text
    return f"ai_works_{start_year}_{end_year}"


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    return DEFAULT_EXPORT_BASE_DIR / default_job_name(args.start_year, args.end_year, args.job_name)


def request_openalex(
    session: requests.Session,
    params: dict,
    *,
    max_retries: int = 8,
    sleep_base: float = 2.0,
) -> dict:
    payload = dict(params)
    payload["api_key"] = api_key()

    for attempt in range(max_retries):
        response = session.get(BASE_URL, params=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {401, 403}:
            raise RuntimeError(
                f"OpenAlex authentication failed ({response.status_code}). Check OPENALEX_API_KEY."
            )
        if response.status_code == 429:
            wait = sleep_base ** attempt
            print(f"Rate limited. Sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue
        if 500 <= response.status_code < 600:
            wait = sleep_base ** attempt
            print(f"Server error {response.status_code}. Sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"OpenAlex request failed with status {response.status_code}: {response.text[:300]}"
        )

    raise RuntimeError("OpenAlex request failed after retries.")


def request_rate_limit(session: requests.Session) -> dict:
    response = session.get(
        RATE_LIMIT_URL,
        params={"api_key": api_key()},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAlex rate-limit check failed with status {response.status_code}: "
            f"{response.text[:300]}"
        )
    return response.json()


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _join(values: list[str]) -> str:
    return "; ".join(_unique_preserve(values))


def _topic_name(obj: dict | None, key: str) -> str:
    if not isinstance(obj, dict):
        return ""
    value = obj.get(key)
    if isinstance(value, dict):
        return str(value.get("display_name") or "")
    return str(value or "")


def normalize_country_code(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "/" in text:
        text = text.rstrip("/").split("/")[-1]
    return text.upper()

def _to_int(value, default: int = 0) -> int:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return default
    return int(number)


def _to_float(value, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return default
    return float(number)


def normalize_work(work: dict, *, query_label: str, current_year: int) -> dict:
    primary_topic = work.get("primary_topic") or {}
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    authorships = work.get("authorships") or []
    topics = work.get("topics") or []
    keywords = work.get("keywords") or []

    author_names: list[str] = []
    institution_names: list[str] = []
    institution_ids: list[str] = []
    country_codes: list[str] = []
    country_names: list[str] = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        author_name = str(author.get("display_name") or "").strip()
        if author_name:
            author_names.append(author_name)
        for institution in authorship.get("institutions") or []:
            inst_name = str(institution.get("display_name") or "").strip()
            inst_id = str(institution.get("id") or "").strip()
            code = normalize_country_code(str(institution.get("country_code") or ""))
            if inst_name:
                institution_names.append(inst_name)
            if inst_id:
                institution_ids.append(inst_id)
            if code:
                country_codes.append(code)
            country_name = str(
                institution.get("country")
                or institution.get("country_display_name")
                or geo.name(code)
                or ""
            ).strip()
            if country_name:
                country_names.append(country_name)

    topic_names = [str(topic.get("display_name") or "").strip() for topic in topics if topic.get("display_name")]
    keyword_names = [str(keyword.get("display_name") or "").strip() for keyword in keywords if keyword.get("display_name")]
    authors = _unique_preserve(author_names)
    institutions = _unique_preserve(institution_names)
    countries = _unique_preserve(country_codes)
    countries_named = _unique_preserve(country_names)
    keyword_names = _unique_preserve(keyword_names)

    publication_year = _to_int(work.get("publication_year"), default=0)
    citation_count = _to_float(work.get("cited_by_count"), default=0.0)
    referenced_count = _to_float(work.get("referenced_works_count"), default=0.0)
    paper_age = max(0, current_year - publication_year) if publication_year > 0 else 0
    citations_per_year = citation_count / float(paper_age + 1)
    publication_type = str(work.get("type") or "")

    return {
        "paper_id": str(work.get("id") or ""),
        "title": str(work.get("display_name") or ""),
        "publication_year": publication_year,
        "publication_date": str(work.get("publication_date") or ""),
        "publication_type": publication_type,
        "citation_count": citation_count,
        "citations_per_year": citations_per_year,
        "fwci": _to_float(work.get("fwci"), default=math.nan),
        "referenced_works_count": referenced_count,
        "topics": _join(topic_names),
        "keywords": _join(keyword_names),
        "primary_topic": str(primary_topic.get("display_name") or ""),
        "primary_subfield": _topic_name(primary_topic, "subfield"),
        "primary_field": _topic_name(primary_topic, "field"),
        "primary_domain": _topic_name(primary_topic, "domain"),
        "venue_source": str(source.get("display_name") or ""),
        "venue_type": str(source.get("type") or ""),
        "authors": _join(authors),
        "institutions": _join(institutions),
        "institution_ids": _join(institution_ids),
        "countries": _join(countries),
        "country_names": _join(countries_named),
        "doi": str(work.get("doi") or ""),
        "language": str(work.get("language") or ""),
        "is_oa": bool(open_access.get("is_oa", False)),
        "oa_status": str(open_access.get("oa_status") or ""),
        "search_term_used": query_label,
        "paper_age": paper_age,
        "citation_velocity": citations_per_year,
        "author_count": len(authors),
        "institution_count": len(institutions),
        "country_count": len(countries),
    }


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPORT_COLUMNS)


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def buffer_checkpoint_path(out_dir: Path) -> Path:
    return out_dir / BUFFER_CHECKPOINT_NAME


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def load_manifest(out_dir: Path) -> dict | None:
    path = manifest_path(out_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(out_dir: Path, payload: dict) -> None:
    _atomic_write_text(manifest_path(out_dir), json.dumps(payload, indent=2))


def load_buffer_checkpoint(out_dir: Path) -> list[dict]:
    path = buffer_checkpoint_path(out_dir)
    if not path.exists():
        return []
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return []
    frame = frame.reindex(columns=EXPORT_COLUMNS)
    return frame.to_dict(orient="records")


def write_buffer_checkpoint(out_dir: Path, rows: list[dict]) -> None:
    path = buffer_checkpoint_path(out_dir)
    if not rows:
        if path.exists():
            path.unlink()
        return
    frame = pd.DataFrame(rows).reindex(columns=EXPORT_COLUMNS)
    _atomic_write_csv(path, frame)


def existing_chunk_paths(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("part_*.csv"))


def load_seen_ids_from_chunks(paths: list[Path]) -> tuple[set[str], int]:
    seen_ids: set[str] = set()
    total_rows = 0
    for path in paths:
        frame = pd.read_csv(path, usecols=["paper_id"], encoding="utf-8-sig")
        ids = frame["paper_id"].dropna().astype(str).tolist()
        total_rows += len(ids)
        seen_ids.update(ids)
    return seen_ids, total_rows


def next_chunk_path(out_dir: Path, chunk_index: int) -> Path:
    return out_dir / f"part_{chunk_index:05d}.csv"


def write_chunk(
    out_dir: Path,
    chunk_index: int,
    rows: list[dict],
    seen_ids: set[str],
) -> tuple[Path | None, int, set[str]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None, 0, set()
    frame = frame.reindex(columns=EXPORT_COLUMNS)
    frame["paper_id"] = frame["paper_id"].fillna("").astype(str)
    frame = frame[frame["paper_id"] != ""].copy()
    frame = frame.drop_duplicates(subset=["paper_id"], keep="first")
    if seen_ids:
        frame = frame[~frame["paper_id"].isin(seen_ids)].copy()
    if frame.empty:
        return None, 0, set()
    path = next_chunk_path(out_dir, chunk_index)
    _atomic_write_csv(path, frame)
    written_ids = set(frame["paper_id"].tolist())
    return path, int(len(frame)), written_ids


def validate_resume_manifest(manifest: dict, args: argparse.Namespace, base_filter: str) -> None:
    expected = {
        "base_filter": base_filter,
        "job_name": default_job_name(args.start_year, args.end_year, args.job_name),
        "query_label": args.query_label,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "target_rows_requested": args.target_rows,
        "chunk_rows": args.chunk_rows,
        "max_daily_list_calls": None if args.allow_full_budget else max(1, int(args.max_daily_list_calls)),
        "allow_full_budget": bool(args.allow_full_budget),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Cannot resume: manifest {key}={manifest.get(key)!r} does not match current {value!r}."
            )


def persist_state(
    out_dir: Path,
    manifest: dict,
    *,
    rows_written: int,
    pages_fetched: int,
    chunk_index: int,
    cursor: str | None,
    chunk_files: list[str],
    buffer_rows: list[dict],
    status: str,
) -> None:
    write_buffer_checkpoint(out_dir, buffer_rows)
    manifest.update(
        {
            "rows_written": rows_written,
            "pages_fetched": pages_fetched,
            "next_chunk_index": chunk_index,
            "next_cursor": cursor,
            "chunk_files": chunk_files,
            "status": status,
            "updated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "buffered_rows_checkpointed": len(buffer_rows),
        }
    )
    write_manifest(out_dir, manifest)


def main() -> None:
    args = parse_args()
    args.out_dir = resolve_out_dir(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_filter = args.base_filter or build_base_filter(args.start_year, args.end_year)
    current_year = datetime.now(UTC).year
    job_name = default_job_name(args.start_year, args.end_year, args.job_name)

    manifest = load_manifest(args.out_dir) if args.resume else None
    if manifest:
        validate_resume_manifest(manifest, args, base_filter)
        actual_chunk_paths = existing_chunk_paths(args.out_dir)
        seen_ids, actual_rows_written = load_seen_ids_from_chunks(actual_chunk_paths)
        rows_written = actual_rows_written
        pages_fetched = int(manifest.get("pages_fetched", 0))
        chunk_files = [path.name for path in actual_chunk_paths]
        chunk_index = len(chunk_files) + 1
        cursor = manifest.get("next_cursor") or "*"
        buffer = load_buffer_checkpoint(args.out_dir)
        buffered_ids = {
            str(row.get("paper_id") or "")
            for row in buffer
            if str(row.get("paper_id") or "").strip()
        }
    else:
        seen_ids = set()
        rows_written = 0
        pages_fetched = 0
        chunk_index = 1
        cursor = "*"
        chunk_files: list[str] = []
        buffer = []
        buffered_ids: set[str] = set()
        manifest = {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "base_filter": base_filter,
            "job_name": job_name,
            "query_label": args.query_label,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "target_rows_requested": args.target_rows,
            "chunk_rows": args.chunk_rows,
            "select_fields": SELECT_FIELDS.split(","),
            "columns": EXPORT_COLUMNS,
            "max_daily_list_calls": None if args.allow_full_budget else max(1, int(args.max_daily_list_calls)),
            "allow_full_budget": bool(args.allow_full_budget),
            "rows_written": 0,
            "pages_fetched": 0,
            "next_chunk_index": 1,
            "next_cursor": "*",
            "chunk_files": [],
            "status": "running",
        }
        write_manifest(args.out_dir, manifest)

    session = requests.Session()
    headers = {"User-Agent": "openalex-ai-export/1.0"}
    if args.mailto:
        headers["User-Agent"] = f"mailto:{args.mailto}"
    session.headers.update(headers)

    estimated_calls_for_target = None if args.target_rows <= 0 else math.ceil(args.target_rows / PER_PAGE)
    effective_call_cap = None if args.allow_full_budget else max(1, int(args.max_daily_list_calls))
    if (
        effective_call_cap is not None
        and estimated_calls_for_target is not None
        and estimated_calls_for_target > effective_call_cap
        and not args.resume
    ):
        safe_target_rows = effective_call_cap * PER_PAGE
        raise RuntimeError(
            "Requested target exceeds the configured safe free-tier budget. "
            f"target_rows={args.target_rows:,} needs about {estimated_calls_for_target:,} list calls, "
            f"but the safety cap allows {effective_call_cap:,}. "
            f"Use --target-rows {safe_target_rows:,}, split the export across days with --resume, "
            "or pass --allow-full-budget if you intentionally want to use the entire estimated budget."
        )

    rate_limit = request_rate_limit(session).get("rate_limit") or {}
    daily_remaining_usd = float(rate_limit.get("daily_remaining_usd") or 0.0)
    endpoint_costs = rate_limit.get("endpoint_costs_usd") or {}
    list_call_cost = float(endpoint_costs.get("list") or 0.0001)
    remaining_list_calls_by_budget = math.floor(daily_remaining_usd / list_call_cost) if list_call_cost > 0 else 0

    print(f"Writing to {args.out_dir}")
    print(f"Job shard: {job_name} ({args.start_year}-{args.end_year})")
    print(f"Starting at rows_written={rows_written}, next_chunk={chunk_index}, cursor={cursor!r}")
    print(
        f"Free-tier view: daily_remaining_usd=${daily_remaining_usd:.4f}, "
        f"list_call_cost=${list_call_cost:.4f}/call, "
        f"remaining_list_calls_by_budget={remaining_list_calls_by_budget:,}"
    )

    finished = False
    heartbeat_pages = max(1, int(args.heartbeat_pages))
    started_at = time.time()
    effective_target_rows = args.target_rows if args.target_rows > 0 else None

    while effective_target_rows is None or rows_written + len(buffer) < effective_target_rows:
        if effective_call_cap is not None and pages_fetched >= effective_call_cap:
            print(
                f"Stopping at safety cap: pages_fetched={pages_fetched:,} "
                f">= max_daily_list_calls={effective_call_cap:,}"
            )
            break
        remaining = None if effective_target_rows is None else effective_target_rows - rows_written - len(buffer)
        if remaining is not None and remaining <= 0:
            break

        payload = request_openalex(
            session,
            {
                "filter": base_filter,
                "select": SELECT_FIELDS,
                "per-page": PER_PAGE,
                "cursor": cursor,
            },
        )
        batch = payload.get("results") or []
        if not batch:
            finished = True
            break
        if "total_available_count" not in manifest:
            manifest["total_available_count"] = int((payload.get("meta") or {}).get("count") or 0)
            if effective_target_rows is None:
                effective_target_rows = int(manifest["total_available_count"])
            manifest["target_rows_effective"] = effective_target_rows

        batch_limit = len(batch) if remaining is None else min(len(batch), remaining)
        for work in batch[:batch_limit]:
            row = normalize_work(work, query_label=args.query_label, current_year=current_year)
            paper_id = str(row.get("paper_id") or "").strip()
            if not paper_id or paper_id in seen_ids or paper_id in buffered_ids:
                continue
            buffer.append(row)
            buffered_ids.add(paper_id)

        pages_fetched += 1
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            finished = True
            break
        cursor = next_cursor

        if pages_fetched % heartbeat_pages == 0:
            elapsed = time.time() - started_at
            approx_rows_seen = rows_written + len(buffer)
            target_label = (
                f"{effective_target_rows:,}"
                if effective_target_rows is not None
                else str(manifest.get("total_available_count", "unknown"))
            )
            print(
                f"Progress: pages={pages_fetched:,}, buffered_rows={len(buffer):,}, "
                f"rows_seen≈{approx_rows_seen:,}/{target_label}, elapsed={elapsed/60:.1f}m"
            )
            persist_state(
                args.out_dir,
                manifest,
                rows_written=rows_written,
                pages_fetched=pages_fetched,
                chunk_index=chunk_index,
                cursor=cursor,
                chunk_files=chunk_files,
                buffer_rows=buffer,
                status="running",
            )

        if len(buffer) >= args.chunk_rows:
            chunk_rows = buffer[:args.chunk_rows]
            path, count, written_ids = write_chunk(args.out_dir, chunk_index, chunk_rows, seen_ids)
            rows_written += count
            if path is not None:
                chunk_files.append(path.name)
                chunk_index += 1
            seen_ids.update(written_ids)
            buffer = buffer[args.chunk_rows:]
            buffered_ids = {
                str(row.get("paper_id") or "")
                for row in buffer
                if str(row.get("paper_id") or "").strip()
            }
            persist_state(
                args.out_dir,
                manifest,
                rows_written=rows_written,
                pages_fetched=pages_fetched,
                chunk_index=chunk_index,
                cursor=cursor,
                chunk_files=chunk_files,
                buffer_rows=buffer,
                status="running",
            )
            if path is not None:
                target_label = (
                    f"{effective_target_rows:,}"
                    if effective_target_rows is not None
                    else str(manifest.get("total_available_count", "unknown"))
                )
                print(f"Wrote {path.name} with {count:,} rows ({rows_written:,}/{target_label})")

    if buffer:
        path, count, written_ids = write_chunk(args.out_dir, chunk_index, buffer, seen_ids)
        rows_written += count
        if path is not None:
            chunk_files.append(path.name)
            chunk_index += 1
            target_label = (
                f"{effective_target_rows:,}"
                if effective_target_rows is not None
                else str(manifest.get("total_available_count", "unknown"))
            )
            print(f"Wrote {path.name} with {count:,} rows ({rows_written:,}/{target_label})")
        seen_ids.update(written_ids)
        buffer = []
        buffered_ids = set()

    status = "complete" if (effective_target_rows is not None and rows_written >= effective_target_rows) or finished else "running"
    persist_state(
        args.out_dir,
        manifest,
        rows_written=rows_written,
        pages_fetched=pages_fetched,
        chunk_index=chunk_index,
        cursor=None if status == "complete" else cursor,
        chunk_files=chunk_files,
        buffer_rows=buffer,
        status=status,
    )
    manifest.update(
        {
            "job_name": job_name,
            "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "max_daily_list_calls": effective_call_cap,
            "allow_full_budget": bool(args.allow_full_budget),
            "target_rows_effective": effective_target_rows,
            "rate_limit_snapshot": {
                "daily_remaining_usd": daily_remaining_usd,
                "list_call_cost_usd": list_call_cost,
                "remaining_list_calls_by_budget": remaining_list_calls_by_budget,
            },
            "unique_paper_ids_written": len(seen_ids),
        }
    )
    write_manifest(args.out_dir, manifest)

    estimated_requests = math.ceil(rows_written / PER_PAGE)
    print(
        "Done. "
        f"rows_written={rows_written:,}, pages_fetched={pages_fetched:,}, "
        f"estimated_filter_calls={estimated_requests:,}, status={status}"
    )


if __name__ == "__main__":
    main()
