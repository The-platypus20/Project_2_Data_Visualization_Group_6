"""Build cleaned rising/fading AI terms by dashboard topic family.

This version fixes three issues from the first family script:
1. Avoids n-grams across column / semicolon boundaries, which created junk like
   "techniques anomaly".
2. Canonicalizes near-duplicates, e.g. "xai explainable" -> "explainable xai".
3. Filters generic / misleading terms and phrases before writing dashboard cache.

Outputs:
- rising_fading_terms.csv
- rising_terms.csv
- fading_terms.csv
- wordcloud_terms.csv
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

BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "between", "both", "but",
    "by", "can", "case", "cases", "could", "do", "does", "doing", "during", "each", "effect",
    "effects", "for", "from", "further", "had", "has", "have", "having", "how", "in", "into", "is", "it",
    "its", "itself", "more", "most", "new", "no", "nor", "not", "of", "on", "one", "or", "other", "our",
    "out", "proposed", "same", "should", "show", "shows", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "under", "use", "used", "using",
    "via", "was", "we", "were", "what", "when", "where", "which", "while", "who", "will", "with", "within",
    "without", "would", "towards", "toward",
}

# Generic research words. We usually remove them, especially as single words.
GENERIC_WORDS = {
    "algorithm", "algorithms", "analysis", "application", "applications", "approach", "approaches",
    "artificial", "based", "computer", "computational", "data", "deep", "engineering", "framework",
    "intelligence", "learning", "machine", "mathematics", "method", "methods", "methodologies",
    "model", "modeling", "models", "network", "networks", "novel", "optimization", "paper", "papers",
    "performance", "problem", "problems", "programming", "research", "result", "results", "review",
    "software", "study", "studies", "survey", "system", "systems", "technique", "techniques", "technology",
    "theory", "topic", "topics", "work", "works",
}

STOPWORDS = BASE_STOPWORDS | GENERIC_WORDS

# Keep these as single words if they are strong and interpretable.
SINGLE_TERM_ALLOWLIST = {
    "adversarial", "alignment", "attention", "cryptography", "detection", "diffusion", "fairness",
    "federated", "hallucination", "privacy", "prompting", "reinforcement", "robotics", "transformer",
    "xai",
}

# Canonicalize duplicates and awkward n-grams.
TERM_REPLACEMENTS = {
    "xai explainable": "explainable xai",
    "explainable artificial": "explainable ai",
    "ai explainable": "explainable ai",
    "privacy preserving": "privacy-preserving",
    "privacy-preserving learning": "privacy-preserving learning",
    "federated privacy": "federated privacy-preserving",
    "federated privacy-preserving": "federated privacy-preserving",
    "privacy-preserving federated": "federated privacy-preserving",
    "large language": "large language model",
    "language model": "language model",
    "foundation models": "foundation model",
    "large language models": "large language model",
    "few shot": "few-shot",
    "few-shot learning": "few-shot learning",
    "domain adaptation": "domain adaptation",
    "anomaly detection": "anomaly detection",
    "detection anomaly": "anomaly detection",
    "techniques anomaly": "anomaly detection",
    "anomaly techniques": "anomaly detection",
    "detection techniques": "anomaly detection",
    "robustness adversarial": "adversarial robustness",
    "adversarial robustness": "adversarial robustness",
    "web ontologies": "semantic web",
    "semantic web": "semantic web",
    "ontologies semantic": "semantic web",
    "logic reasoning": "logic reasoning",
    "reasoning logic": "logic reasoning",
    "security privacy": "privacy security",
    "privacy security": "privacy security",
    "cryptographic security": "cryptography security",
    "security cryptography": "cryptography security",
}

# Remove these even if statistically strong. They are too generic or misleading for the demo.
BAD_TERMS = {
    "algorithm", "algorithms", "analysis", "application", "applications", "artificial intelligence",
    "computational techniques", "computer", "computer science", "data", "deep learning", "engineering methodologies",
    "gradient optimization", "learning", "machine learning", "mathematics", "model", "modeling", "models",
    "natural", "neural", "network", "networks", "optimization techniques", "programming", "software engineering",
    "stochastic gradient", "techniques", "theory", "topic", "topics", "xai", "artificial", "intelligence",
}

BAD_SUBSTRINGS = {
    "advanced software", "methodologies advanced", "service-oriented architecture", "techniques advanced",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+(?:-[a-z0-9]+)?")
HTML_RE = re.compile(r"<[^>]+>")
SPLIT_RE = re.compile(r"[;|•\n\r]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned rising/fading term cache from raw AI paper CSV files.")
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
    parser.add_argument("--family-col", default=None, help="Optional family/group column. Only used when no topic-family map matches.")
    parser.add_argument("--topic-family-json", default="www/data/subtopic_layout.json", help="JSON mapping primary_topic/topic to dashboard family. Use empty string to disable.")
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


def normalize_label(value: object) -> str:
    return " ".join(str(value or "").lower().replace("&", "and").split())


def load_topic_family_map(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        print(f"Topic-family map not found: {path}. Falling back to --family-col/default grouping.")
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read topic-family map {path}: {exc}. Falling back to --family-col/default grouping.")
        return {}
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "").strip()
        if not family:
            continue
        for key in ["topic", "primary_topic", "name", "label"]:
            norm = normalize_label(row.get(key))
            if norm:
                out[norm] = family
    print(f"Loaded {len(out):,} topic->family mappings from {path}")
    return out


def choose_columns(columns: Iterable[str], requested_text_cols: list[str] | None, requested_family_col: str | None, topic_family_map: dict[str, str] | None = None):
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
    elif not topic_family_map:
        family_col = next((lower[c] for c in ["topic_bucket", "family"] if c in lower), None)

    topic_col = next((lower[c] for c in ["primary_topic", "topic"] if c in lower), None)
    usecols = [year_col] + text_cols + ([family_col] if family_col else []) + ([topic_col] if topic_col else [])
    return year_col, text_cols, family_col, topic_col, list(dict.fromkeys(usecols))


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if not text or text.lower() == "nan":
        return ""

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
    text = re.sub(r"[/_,:()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def tokenize_segment(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    out = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok in BASE_STOPWORDS:
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    return out


def simple_singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    return word


def canonical_term(term: str) -> str | None:
    t = normalize_label(term)
    if not t:
        return None
    t = t.replace("few shot", "few-shot")
    words = [simple_singular(w) for w in t.split()]
    t = " ".join(words)
    t = TERM_REPLACEMENTS.get(t, t)

    # One more normalization pass after replacement.
    t = normalize_label(t)
    if not t:
        return None
    if t in BAD_TERMS:
        return None
    if any(bad in t for bad in BAD_SUBSTRINGS):
        return None

    parts = t.split()
    if len(parts) == 1:
        if t not in SINGLE_TERM_ALLOWLIST:
            return None
    else:
        # Drop phrases that are only generic words.
        if all(p in GENERIC_WORDS or p in BASE_STOPWORDS for p in parts):
            return None
        # Drop phrases with generic endings unless already canonical/meaningful.
        if parts[-1] in {"technique", "method", "approach", "system", "model", "topic", "algorithm"} and t not in TERM_REPLACEMENTS.values():
            return None

    return t


def segment_terms(text: str, max_ngram: int) -> set[str]:
    terms: set[str] = set()
    for raw_segment in SPLIT_RE.split(text):
        segment = clean_text(raw_segment)
        if not segment:
            continue
        tokens = tokenize_segment(segment)
        if not tokens:
            continue

        # Unigrams.
        for tok in tokens:
            term = canonical_term(tok)
            if term:
                terms.add(term)

        # Bigrams.
        if max_ngram >= 2:
            for i in range(len(tokens) - 1):
                term = canonical_term(f"{tokens[i]} {tokens[i + 1]}")
                if term:
                    terms.add(term)

        # Trigrams.
        if max_ngram >= 3:
            for i in range(len(tokens) - 2):
                term = canonical_term(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
                if term:
                    terms.add(term)
    return terms


def row_terms(row_dict: dict, text_cols: list[str], max_ngram: int) -> set[str]:
    # Important: process each column and each semicolon-separated topic separately.
    # This prevents fake n-grams across field boundaries.
    terms: set[str] = set()
    for col in text_cols:
        value = row_dict.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        terms.update(segment_terms(str(value), max_ngram=max_ngram))
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
    topic_col: str | None = None
    usecols: list[str] | None = None
    topic_family_map = load_topic_family_map(args.topic_family_json)

    for file in files:
        print(f"Reading {file}")
        if not initialized:
            sample = pd.read_csv(file, nrows=5)
            year_col, text_cols, family_col, topic_col, usecols = choose_columns(sample.columns, args.text_cols, args.family_col, topic_family_map)
            initialized = True
            print(f"Using year_col={year_col}; text_cols={text_cols}; family_col={family_col}; topic_col={topic_col}")

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

                family = None
                if topic_family_map and topic_col:
                    family = topic_family_map.get(normalize_label(row_dict.get(topic_col)))
                if not family and family_col:
                    family = default_family(row_dict.get(family_col))
                if family == "All AI":
                    family = None

                terms = row_terms(row_dict, text_cols, args.max_ngram)
                if not terms:
                    continue

                total_docs[period] += 1
                term_counts[period].update(terms)
                if family:
                    family_docs[(family, period)] += 1
                    family_term_counts[(family, period)].update(terms)

    rows = []
    eps = 0.5

    def append_scope(scope: str, early_counter: Counter, late_counter: Counter, early_total_raw: int, late_total_raw: int):
        early_total = max(early_total_raw, 1)
        late_total = max(late_total_raw, 1)
        all_terms = set(early_counter) | set(late_counter)
        for term in all_terms:
            early_count = early_counter[term]
            late_count = late_counter[term]
            total = early_count + late_count
            if total < args.min_docs:
                continue
            early_share = 1000 * early_count / early_total
            late_share = 1000 * late_count / late_total
            delta = late_share - early_share
            if abs(delta) < 1e-12:
                continue
            growth_ratio = (late_share + eps) / (early_share + eps)
            direction = "rising" if delta > 0 else "fading"
            score = abs(delta) * math.log1p(total)
            rows.append({
                "scope": scope,
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

    append_scope("All AI", term_counts["early"], term_counts["late"], total_docs["early"], total_docs["late"])

    families = sorted({family for family, _ in family_docs.keys()})
    for family in families:
        append_scope(
            family,
            family_term_counts[(family, "early")],
            family_term_counts[(family, "late")],
            family_docs[(family, "early")],
            family_docs[(family, "late")],
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Final duplicate cleanup after scoring: keep the strongest row per scope/term/direction.
    out = out.sort_values(["scope", "direction", "score"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["scope", "term", "direction"], keep="first")
    return out


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
    df = process_files(args)
    write_outputs(df, args)


if __name__ == "__main__":
    main()
