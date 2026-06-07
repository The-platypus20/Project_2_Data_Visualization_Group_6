"""Build compact JSON files for the Tab 2 topic landscape.

The browser-facing landscape reads these JSON files instead of scanning the
large OpenAlex exports in the app or loading paper-level data client-side.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import narrative_data as nd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WWW_DIR = PROJECT_ROOT / "www"
DATA_DIR = WWW_DIR / "data"
YEARS = list(range(2000, 2026))
PAPER_SAMPLE_LIMIT = 5_000

FAMILY_SPECS = [
    {
        "family": "NLP & Language AI",
        "color": "#7c3aed",
        "x": 405,
        "y": 145,
        "bucket": "NLP",
        "description": "Text, speech, sentiment, semantics, and language-oriented AI research.",
        "terms": [
            "natural language", "language", "nlp", "text", "speech",
            "semantic", "sentiment", "opinion", "dialogue", "readability",
            "authorship", "topic modeling"
        ],
    },
    {
        "family": "Core ML & Deep Learning",
        "color": "#2563eb",
        "x": 185,
        "y": 255,
        "bucket": "Core ML / Deep Learning",
        "description": "Neural networks, graph learning, robustness, classification, clustering, and core ML methods.",
        "terms": [
            "deep learning", "neural", "graph neural", "classification",
            "clustering", "machine learning", "adversarial", "reservoir",
            "wireless signal"
        ],
    },
    {
        "family": "Optimization, Theory & Security",
        "color": "#dc2626",
        "x": 600,
        "y": 600,
        "bucket": "ML Theory & Optimization",
        "description": "Optimization, probabilistic methods, algorithms, cryptography, and security-oriented AI infrastructure.",
        "terms": [
            "optimization", "bayesian", "probabilistic", "algorithm",
            "theory", "causal", "cryptography", "security",
            "data compression", "quantum"
        ],
    },
    {
        "family": "Robotics & Control",
        "color": "#059669",
        "x": 155,
        "y": 475,
        "bucket": "Robotics",
        "description": "Robotics, control systems, tracking, sensor fusion, and autonomous systems.",
        "terms": [
            "robot", "robotics", "control", "tracking", "sensor",
            "fusion", "autonomous", "planning", "fuzzy logic"
        ],
    },
    {
        "family": "Healthcare AI",
        "color": "#be185d",
        "x": 325,
        "y": 390,
        "bucket": "Healthcare AI",
        "description": "Medical, clinical, cancer, mental-health, and healthcare machine-learning applications.",
        "terms": [
            "health", "healthcare", "medical", "clinical", "cancer",
            "disease", "diagnosis", "psychiatry", "mental health", "neuroscience"
        ],
    },
    {
        "family": "Responsible AI",
        "color": "#65a30d",
        "x": 785,
        "y": 335,
        "bucket": "AI Ethics & Fairness",
        "description": "Privacy, fairness, explainability, law, trust, and responsible AI.",
        "terms": [
            "privacy", "fairness", "ethics", "bias", "explainable",
            "xai", "law", "intellectual property", "trust"
        ],
    },
    {
        "family": "Reinforcement Learning & Agents",
        "color": "#d97706",
        "x": 785,
        "y": 520,
        "bucket": "Reinforcement Learning",
        "description": "Reinforcement learning, multi-agent systems, planning, negotiation, and agentic decision-making.",
        "terms": [
            "reinforcement", "policy", "reward", "agent",
            "multi-agent", "negotiation", "problem solving", "planning"
        ],
    },
    {
    "family": "Knowledge, Logic & Reasoning",
    "color": "#0f766e",
    "x": 775,
    "y": 160,
    "bucket": "",
    "description": "Symbolic AI, semantic web, logic, reasoning, knowledge representation, and formal methods.",
    "terms": [
        "semantic web", "ontology", "ontologies",
        "knowledge graph", "knowledge representation", "knowledge-based", "knowledge base",
        "logic", "reasoning", "symbolic", "formal",
        "programming language", "type systems", "verification",
        "problem solving", "planning"
    ],
    },
    {
    "family": "Applied / Interdisciplinary AI",
    "color": "#0891b2",
    "x": 575,
    "y": 320,
    "bucket": "",
    "description": "Applied and cross-domain AI topics, including education, science, service systems, software engineering, anomaly detection, and interdisciplinary domains.",
    "terms": [
        "education", "educational", "service", "software engineering",
        "computational physics", "geochemistry", "solar", "seismology",
        "interdisciplinary", "employee performance", "cognitive science",
        "anomaly detection", "data stream", "data analysis"
        ],
    },
]


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 4)


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = _clean_number(value)
            if value is None:
                continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        out[key] = value
    return out

def _add_quadrants_to_subtopics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    growth_values = pd.to_numeric(
        pd.Series([row.get("growth_rate", 0) for row in rows]),
        errors="coerce",
    ).fillna(0)

    velocity_values = pd.to_numeric(
        pd.Series([row.get("citation_velocity", 0) for row in rows]),
        errors="coerce",
    ).fillna(0)

    growth_mid = float(growth_values.median())
    velocity_mid = float(velocity_values.median())

    for row in rows:
        growth = float(row.get("growth_rate", 0) or 0)
        velocity = float(row.get("citation_velocity", 0) or 0)

        high_growth = growth >= growth_mid
        high_velocity = velocity >= velocity_mid

        if high_growth and high_velocity:
            row["quadrant"] = "High growth / High velocity"
        elif high_growth and not high_velocity:
            row["quadrant"] = "High growth / Low velocity"
        elif not high_growth and high_velocity:
            row["quadrant"] = "Low growth / High velocity"
        else:
            row["quadrant"] = "Low growth / Low velocity"

    return rows

def _pattern(terms: list[str]) -> str:
    return "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in terms)


def _topic_family(topic: str, bucket: str, specs: list[dict[str, Any]]) -> str:
    topic_l = str(topic).lower()
    bucket_l = str(bucket).lower()

    applied_exact_topics = {
        "geochemistry and geologic mapping",
        "solar radiation and photovoltaics",
        "seismology and earthquake studies",
        "computational physics and python applications",
    }

    knowledge_exact_topics = {
        "semantic web and ontologies",
        "logic, programming, and type systems",
        "logic, reasoning, and knowledge",
        "experience-based knowledge management",
    }

    if topic_l in applied_exact_topics:
        return "Applied / Interdisciplinary AI"

    if topic_l in knowledge_exact_topics:
        return "Knowledge, Logic & Reasoning"

    bucket_map = {
        "nlp": "NLP & Language AI",
        "core ml / deep learning": "Core ML & Deep Learning",
        "ml theory & optimization": "Optimization, Theory & Security",
        "robotics": "Robotics & Control",
        "healthcare ai": "Healthcare AI",
        "ai ethics & fairness": "Responsible AI",
        "reinforcement learning": "Reinforcement Learning & Agents",
    }

    if bucket_l in bucket_map:
        return bucket_map[bucket_l]

    for spec in specs:
        if re.search(_pattern(spec["terms"]), topic_l):
            return str(spec["family"])

    return "Applied / Interdisciplinary AI"


def _cumulative_counts(frame: pd.DataFrame, topic_col: str, topic_value: str) -> dict[str, int]:
    if frame.empty:
        return {str(year): 0 for year in YEARS}
    sub = frame[frame[topic_col].astype(str).eq(str(topic_value))].copy()
    by_year = sub.groupby("year")["count"].sum()
    running = 0
    out = {}
    for year in YEARS:
        running += int(by_year.get(year, 0))
        out[str(year)] = int(running)
    return out


def _bad_title(title: str) -> bool:
    value = str(title or "").strip().lower()
    if len(value) < 18:
        return True
    bad_patterns = [
        r"<[^>]+>",
        r"\bbook review\b",
        r"\bproceedings\b",
        r"\bconference\b",
        r"\bcoronavirus pandemic\b",
        r"\bhandbook of\b",
        r"\bintroduction to\b",
    ]
    return any(re.search(pattern, value) for pattern in bad_patterns)


def _representative_papers(papers: pd.DataFrame, *, topic: str = "", family: str = "", limit: int = 5) -> list[dict[str, Any]]:
    if papers.empty:
        return []

    sub = papers.copy()

    if topic:
        sub = sub[sub["topic"].astype(str).eq(topic)].copy()
    elif family:
        sub = sub[sub["family"].astype(str).eq(family)].copy()

    if sub.empty:
        return []

    if "title" in sub:
        sub = sub[~sub["title"].astype(str).map(_bad_title)].copy()

    if sub.empty:
        return []

    citation = pd.to_numeric(sub.get("citation_count"), errors="coerce").fillna(0)
    fwci = pd.to_numeric(sub.get("fwci"), errors="coerce").fillna(0)
    year = pd.to_numeric(sub.get("year"), errors="coerce").fillna(0)

    positive_fwci = fwci[fwci.gt(0)]
    if positive_fwci.empty:
        fwci_cap = 0
    else:
        fwci_cap = min(float(positive_fwci.quantile(0.99)), 500.0)

    clipped_fwci = fwci.clip(lower=0, upper=fwci_cap)

    sub["_rank"] = (
        np.log1p(citation)
        + 2.0 * np.log1p(clipped_fwci)
        + 0.03 * year
    )

    sub = sub.sort_values("_rank", ascending=False)

    keep = ["id", "title", "year", "family", "topic", "fwci", "citation_count", "country", "venue"]
    records = []
    for rec in sub[[col for col in keep if col in sub]].head(limit).to_dict("records"):
        records.append(_clean_record(rec))
    return records


def _load_papers(topic_to_family: dict[str, str]) -> pd.DataFrame:
    lookup = nd.paper_lookup().copy()
    if lookup.empty:
        return pd.DataFrame(columns=["id", "title", "year", "family", "topic"])
    lookup = lookup.rename(columns={"primary_topic": "topic"})
    lookup["topic"] = lookup["topic"].fillna("").astype(str)
    lookup["family"] = lookup["topic"].map(topic_to_family).fillna("")
    lookup = lookup[lookup["family"].ne("")].copy()
    if lookup.empty:
        return lookup
    lookup["year"] = pd.to_numeric(lookup.get("year"), errors="coerce")
    for col in ["fwci", "citation_velocity", "citation_count"]:
        if col in lookup:
            lookup[col] = pd.to_numeric(lookup[col], errors="coerce")
    lookup = lookup.drop_duplicates(subset=["title", "topic"], keep="first")
    lookup = lookup.reset_index(drop=True)
    lookup["id"] = ["paper-" + str(i + 1) for i in range(len(lookup))]
    return lookup


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    topic_counts = nd.topic_year_counts().copy()
    metrics = nd.impact_topic_scatter().copy()
    if topic_counts.empty:
        raise RuntimeError("Dataset/dashboard_cache/topic_year_counts.csv is required.")

    topic_counts["year"] = pd.to_numeric(topic_counts["year"], errors="coerce").astype("Int64")
    topic_counts["count"] = pd.to_numeric(topic_counts["count"], errors="coerce").fillna(0)
    topic_counts = topic_counts[topic_counts["year"].between(2000, 2025)].copy()
    topic_counts["year"] = topic_counts["year"].astype(int)

    topic_counts["family"] = [
        _topic_family(str(row["primary_topic"]), str(row.get("topic_bucket", "")), FAMILY_SPECS)
        for _, row in topic_counts.iterrows()
    ]
    topic_counts = topic_counts[topic_counts["family"].ne("")].copy()

    topic_to_family = (
        topic_counts.groupby(["primary_topic", "family"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .drop_duplicates("primary_topic")
        .set_index("primary_topic")["family"]
        .to_dict()
    )
    papers = _load_papers(topic_to_family)

    metric_map = {}
    if not metrics.empty:
        metric_map = metrics.set_index("primary_topic").to_dict("index")

    family_rows: list[dict[str, Any]] = []
    subtopic_rows: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        family = str(spec["family"])
        fam_counts = topic_counts[topic_counts["family"].eq(family)].copy()
        by_year = fam_counts.groupby("year")["count"].sum()
        running = 0
        cumulative: dict[str, int] = {}
        for year in YEARS:
            running += int(by_year.get(year, 0))
            cumulative[str(year)] = int(running)

        topic_totals = (
            fam_counts.groupby("primary_topic", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )
        top_topics = topic_totals.head(8)["primary_topic"].astype(str).tolist()
        weight_source = topic_totals.head(12).set_index("primary_topic")["count"]
        fwci_values = []
        fwci_weights = []
        velocity_values = []
        velocity_weights = []

        for topic, count in weight_source.items():
            row = metric_map.get(topic, {})
            weight = float(count or 1)

            fwci = _clean_number(row.get("median_fwci") or row.get("mean_fwci"))
            velocity = _clean_number(
                row.get("median_velocity")
                or row.get("citation_velocity")
                or row.get("mean_citation_velocity")
            )

            if fwci is not None and fwci > 0:
                fwci_values.append(float(fwci))
                fwci_weights.append(weight)

            if velocity is not None and velocity > 0:
                velocity_values.append(float(velocity))
                velocity_weights.append(weight)

        median_fwci = float(np.average(fwci_values, weights=fwci_weights)) if fwci_values else None
        citation_velocity = float(np.average(velocity_values, weights=velocity_weights)) if velocity_values else None

        family_rows.append(_clean_record({
            "family": family,
            "x": spec["x"],
            "y": spec["y"],
            "color": spec["color"],
            "description": spec["description"],
            "cumulative_paper_count_by_year": cumulative,
            "paper_count_2025": cumulative["2025"],
            "median_fwci": median_fwci,
            "citation_velocity": citation_velocity,
            "top_subtopics": top_topics[:5],
            "representative_papers": _representative_papers(papers, family=family, limit=5),
        }))

        offsets = [(-42, -14), (44, -10), (-26, 32), (38, 34), (0, 0), (-58, 38), (56, -42), (12, 54)]
        for index, topic in enumerate(top_topics):
            tx, ty = offsets[index % len(offsets)]
            counts = _cumulative_counts(fam_counts, "primary_topic", topic)
            metric_row = metric_map.get(topic, {})
            early = sum(int(counts[str(year)]) for year in range(2000, 2006))
            late = int(counts["2025"]) - int(counts["2017"])

            growth_rate = float(metric_row.get("growth", np.nan))
            if not np.isfinite(growth_rate):
                growth_rate = (late / max(early, 1)) if late else 0

            fwci = _clean_number(metric_row.get("median_fwci") or metric_row.get("mean_fwci"))

            velocity = _clean_number(
                metric_row.get("median_velocity")
                or metric_row.get("citation_velocity")
                or metric_row.get("mean_citation_velocity")
            )

            if velocity is None:
                velocity = 0

            frontier_score = float((fwci or 0) * 0.55 + min(growth_rate, 25) * 0.03 + (velocity or 0) * 0.25)
            subtopic_rows.append(_clean_record({
                "family": family,
                "topic": topic,
                "x": int(spec["x"] + tx),
                "y": int(spec["y"] + ty),
                "cumulative_paper_count_by_year": counts,
                "paper_count_2025": counts["2025"],
                "median_fwci": fwci,
                "growth_rate": _clean_number(growth_rate),
                "citation_velocity": velocity,
                "frontier_score": _clean_number(frontier_score),
                "representative_papers": _representative_papers(papers, topic=topic, limit=5),
            }))

    sample = papers.copy()
    if not sample.empty:
        sample["_rank"] = (
            pd.to_numeric(sample.get("fwci"), errors="coerce").fillna(0) * 1000
            + pd.to_numeric(sample.get("citation_velocity"), errors="coerce").fillna(0) * 10
            + pd.to_numeric(sample.get("citation_count"), errors="coerce").fillna(0)
            + pd.to_numeric(sample.get("year"), errors="coerce").fillna(0) * 0.05
        )
        sample = sample.sort_values("_rank", ascending=False).head(PAPER_SAMPLE_LIMIT)
    keep_cols = ["id", "title", "year", "family", "topic", "fwci", "citation_count", "country", "venue"]
    paper_rows = [
        _clean_record(rec)
        for rec in sample[[col for col in keep_cols if col in sample]].to_dict("records")
    ]

    subtopic_rows = _add_quadrants_to_subtopics(subtopic_rows)
    outputs = {
        "topic_family_layout.json": family_rows,
        "subtopic_layout.json": subtopic_rows,
        "papers_sample.json": paper_rows,
    }
    for name, rows in outputs.items():
        path = DATA_DIR / name
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        size_kb = path.stat().st_size / 1024
        print(f"{name}: {len(rows):,} rows, {size_kb:,.1f} KB")


if __name__ == "__main__":
    build()

