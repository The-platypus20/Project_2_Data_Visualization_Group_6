"""Build rising/fading AI terms from raw 2000-2025 CSV files.

Input: one or more paper-level CSV files.
Expected useful columns, if present:
- year
- title
- abstract / abstract_inverted_index / display_name
- primary_topic / topic / topics
- topic_bucket / family / primary_subfield

Output:
- rising_fading_terms.csv
- rising_terms.csv
- fading_terms.csv
- wordcloud_terms.csv

Example:
python src/build_rising_fading_terms.py \
  --input "Dataset/raw/ai_works_2000_2009.csv" "Dataset/raw/ai_works_2010_2019.csv" "Dataset/raw/ai_works_2020_2025.csv" \
  --output-dir "Dataset/dashboard_cache" \
  --target-start 2020 \
  --target-end 2025
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "based", "be", "because", "been", "before", "being", "between", "both", "but",
    "by", "can", "case", "cases", "could", "data", "do", "does", "doing", "during", "each", "effect",
    "effects", "for", "from", "further", "had", "has", "have", "having", "how", "in", "into", "is", "it",
    "its", "itself", "method", "methods", "model", "models", "more", "most", "new", "no", "nor", "not",
    "of", "on", "one", "or", "other", "our", "out", "paper", "papers", "proposed", "research", "result",
    "results", "same", "should", "show", "shows", "study", "studies", "such", "system", "systems", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "under", "use", "used", "using", "via", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "within", "without", "work", "works", "would", "using", "approach", "analysis",
    "application", "applications", "framework", "towards", "toward", "review", "survey", "overview", "problem",
    "problems", "performance", "learning", "artificial", "intelligence", "ai", "machine", "deep"
}

TERM_ALLOWLIST_HINTS = {
    "transformer", "attention", "diffusion", "prompt", "prompting", "llm", "language", "vision", "graph",
    "neural", "reinforcement", "federated", "privacy", "xai", "explainable", "robustness", "adversarial",
    "alignment", "agent", "agents", "multimodal", "foundation", "retrieval", "generation", "generative",
    "embedding", "embeddings", "semantic", "reasoning", "robotics", "medical", "healthcare", "clinical"
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+(?:-[a-z0-9]+)?")
HTML_RE = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rising/fading term cache from raw AI paper CSV files.")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV file paths. Supports globs.")
    parser.add_argument("--output-dir", default="Dataset/dashboard_cache", help="Output directory.")
    parser.add_argument("--target-start", type=int, default=2020, help="Late/selected period start year.")
    parser.add_argument("--target-end", type=int, default=2025, help="Late/selected period end year.")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--min-docs", type=int, default=25, help="Minimum total docs containing term.")
    parser.add_argument("--top-n", type=int, default=80, help="Rows per direction in rising/fading files.")
    parser.add_argument("--max-ngram", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--text-cols", nargs="*", default=None, help="Optional text columns to use.")
    parser.add_argument("--family-col", default=None, help="Optional family/group column.")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            files.extend(Path().glob(pattern))
        elif p.exists():
            files.append(p)
    files = sorted(set(f.resolve() for f in files if f.exists() and f.suffix.lower() == ".csv"))
    if not files:
        raise FileNotFoundError("No input CSV files found. Check --input paths/globs.")
    return files


def choose_columns(columns: Iterable[str], requested_text_cols: list[str] | None, requested_family_col: str | None):
    cols = list(columns)
    lower = {c.lower(): c for c in cols}

    year_col = next((lower[c] for c in ["year", "publication_year", "pub_year"] if c in lower), None)
    if year_col is None:
        raise ValueError("No year column found. Expected one of: year, publication_year, pub_year.")

    if requested_text_cols:
        text_cols = [c for c in requested_text_cols if c in cols]
    else:
        candidates = [
            "title", "display_name", "abstract", "abstract_text", "primary_topic", "topic", "topics",
            "primary_subfield", "primary_field", "primary_domain", "keywords", "concepts",
        ]
        text_cols = [lower[c] for c in candidates if c in lower]

    if not text_cols:
        raise ValueError("No text columns found. Pass --text-cols title abstract primary_topic etc.")

    family_col = None
    if requested_family_col and requested_family_col in cols:
        family_col = requested_family_col
    else:
        family_col = next((lower[c] for c in ["topic_bucket", "family", "primary_subfield", "primary_field"] if c in lower), None)

    usecols = [year_col] + text_cols + ([family_col] if family_col else [])
    return year_col, text_cols, family_col, list(dict.fromkeys(usecols))


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""

    # OpenAlex abstract_inverted_index sometimes appears as JSON-like dict.
    if text.startswith("{") and "InvertedIndex" in text:
        try:
            obj = json.loads(text)
            inv = obj.get("InvertedIndex", obj)
            words = []
            for word, positions in inv.items():
                if isinstance(positions, list):
                    for pos in positions:
                        words.append((int(pos), word))
            if words:
                text = " ".join(w for _, w in sorted(words))
        except Exception:
            pass

    text = HTML_RE.sub(" ", text)
    text = text.replace("&amp;", " and ").replace("&", " and ")
    return text.lower()


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    out = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    return out


def ngrams(tokens: list[str], max_ngram: int) -> set[str]:
    terms: set[str] = set()
    for tok in tokens:
        if tok not in STOPWORDS:
            terms.add(tok)
    if max_ngram >= 2:
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a in STOPWORDS or b in STOPWORDS:
                continue
            terms.add(f"{a} {b}")
    if max_ngram >= 3:
        for i in range(len(tokens) - 2):
            a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
            if a in STOPWORDS or b in STOPWORDS or c in STOPWORDS:
                continue
            terms.add(f"{a} {b} {c}")
    return terms


def period_for_year(year: int, target_start: int, target_end: int, min_year: int, max_year: int) -> str | None:
    if year < min_year or year > max_year:
        return None
    if target_start <= year <= target_end:
        return "late"
    if min_year <= year < target_start:
        return "early"
    return None


def default_family(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "All AI"
    text = str(value).strip()
    return text if text else "All AI"


def process_files(args: argparse.Namespace) -> pd.DataFrame:
    files = expand_inputs(args.input)
    print(f"Found {len(files)} CSV files")

    term_counts = {"early": Counter(), "late": Counter()}
    family_term_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    total_docs = {"early": 0, "late": 0}
    family_docs: Counter[tuple[str, str]] = Counter()

    initialized = False
    year_col = ""
    text_cols: list[str] = []
    family_col: str | None = None
    usecols: list[str] | None = None

    for file in files:
        print(f"Reading {file}")
        if not initialized:
            sample = pd.read_csv(file, nrows=5)
            year_col, text_cols, family_col, usecols = choose_columns(sample.columns, args.text_cols, args.family_col)
            initialized = True
            print(f"Using year_col={year_col}; text_cols={text_cols}; family_col={family_col}")

        assert usecols is not None
        for chunk in pd.read_csv(file, usecols=lambda c: c in usecols, chunksize=args.chunksize, low_memory=False):
            chunk[year_col] = pd.to_numeric(chunk[year_col], errors="coerce")
            chunk = chunk.dropna(subset=[year_col])
            if chunk.empty:
                continue
            chunk[year_col] = chunk[year_col].astype(int)
            chunk = chunk[(chunk[year_col] >= args.min_year) & (chunk[year_col] <= args.max_year)]
            if chunk.empty:
                continue

            for row in chunk.itertuples(index=False):
                row_dict = row._asdict()
                year = int(row_dict[year_col])
                period = period_for_year(year, args.target_start, args.target_end, args.min_year, args.max_year)
                if period is None:
                    continue

                family = default_family(row_dict.get(family_col)) if family_col else "All AI"
                text = " ".join(clean_text(row_dict.get(c)) for c in text_cols)
                tokens = tokenize(text)
                if not tokens:
                    continue

                terms = ngrams(tokens, args.max_ngram)
                if not terms:
                    continue

                total_docs[period] += 1
                family_docs[(family, period)] += 1
                term_counts[period].update(terms)
                family_term_counts[(family, period)].update(terms)

    rows = []
    all_terms = set(term_counts["early"]) | set(term_counts["late"])
    eps = 0.5
    early_total = max(total_docs["early"], 1)
    late_total = max(total_docs["late"], 1)

    for term in all_terms:
        early_count = term_counts["early"][term]
        late_count = term_counts["late"][term]
        total = early_count + late_count
        if total < args.min_docs:
            continue
        early_share = 1000 * early_count / early_total
        late_share = 1000 * late_count / late_total
        delta = late_share - early_share
        growth_ratio = (late_share + eps) / (early_share + eps)
        direction = "rising" if delta > 0 else "fading"
        score = abs(delta) * math.log1p(total)
        rows.append({
            "scope": "All AI",
            "term": term,
            "early_count": early_count,
            "late_count": late_count,
            "early_share_per_1000": early_share,
            "late_share_per_1000": late_share,
            "delta_share_per_1000": delta,
            "growth_ratio": growth_ratio,
            "direction": direction,
            "score": score,
            "total_docs": total,
            "early_period": f"{args.min_year}-{args.target_start - 1}",
            "late_period": f"{args.target_start}-{args.target_end}",
        })

    families = sorted({family for family, _ in family_docs.keys()})
    for family in families:
        early_docs = max(family_docs[(family, "early")], 1)
        late_docs = max(family_docs[(family, "late")], 1)
        terms = set(family_term_counts[(family, "early")]) | set(family_term_counts[(family, "late")])
        for term in terms:
            early_count = family_term_counts[(family, "early")][term]
            late_count = family_term_counts[(family, "late")][term]
            total = early_count + late_count
            if total < args.min_docs:
                continue
            early_share = 1000 * early_count / early_docs
            late_share = 1000 * late_count / late_docs
            delta = late_share - early_share
            growth_ratio = (late_share + eps) / (early_share + eps)
            direction = "rising" if delta > 0 else "fading"
            score = abs(delta) * math.log1p(total)
            rows.append({
                "scope": family,
                "term": term,
                "early_count": early_count,
                "late_count": late_count,
                "early_share_per_1000": early_share,
                "late_share_per_1000": late_share,
                "delta_share_per_1000": delta,
                "growth_ratio": growth_ratio,
                "direction": direction,
                "score": score,
                "total_docs": total,
                "early_period": f"{args.min_year}-{args.target_start - 1}",
                "late_period": f"{args.target_start}-{args.target_end}",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Light cleanup: remove phrases made only of generic words unless they contain a useful AI hint.
    def useful(term: str) -> bool:
        parts = term.split()
        if len(parts) == 1 and term not in TERM_ALLOWLIST_HINTS and len(term) < 5:
            return False
        return True

    out = out[out["term"].map(useful)].copy()
    return out.sort_values(["scope", "direction", "score"], ascending=[True, True, False])


def write_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "rising_fading_terms.csv"
    rising_path = output_dir / "rising_terms.csv"
    fading_path = output_dir / "fading_terms.csv"
    cloud_path = output_dir / "wordcloud_terms.csv"

    df.to_csv(full_path, index=False)

    rising = (
        df[df["direction"].eq("rising")]
        .sort_values(["scope", "score"], ascending=[True, False])
        .groupby("scope", group_keys=False)
        .head(args.top_n)
    )
    fading = (
        df[df["direction"].eq("fading")]
        .sort_values(["scope", "score"], ascending=[True, False])
        .groupby("scope", group_keys=False)
        .head(args.top_n)
    )
    rising.to_csv(rising_path, index=False)
    fading.to_csv(fading_path, index=False)

    cloud = df.copy()
    cloud["weight"] = cloud["score"]
    cloud = cloud.sort_values(["scope", "direction", "weight"], ascending=[True, True, False])
    cloud.groupby(["scope", "direction"], group_keys=False).head(args.top_n).to_csv(cloud_path, index=False)

    print(f"Wrote {full_path} ({len(df):,} rows)")
    print(f"Wrote {rising_path} ({len(rising):,} rows)")
    print(f"Wrote {fading_path} ({len(fading):,} rows)")
    print(f"Wrote {cloud_path} ({len(cloud):,} rows before top filter write)")


def main() -> None:
    args = parse_args()
    if args.target_start <= args.min_year:
        # Full-range selection has no prior baseline. Split inside the range.
        midpoint = (args.min_year + args.target_end) // 2
        print(f"target-start <= min-year. Using internal split: early={args.min_year}-{midpoint}, late={midpoint + 1}-{args.target_end}")
        args.target_start = midpoint + 1
    df = process_files(args)
    if df.empty:
        raise RuntimeError("No terms were produced. Check text columns, min_docs, and year range.")
    write_outputs(df, args)


if __name__ == "__main__":
    main()
