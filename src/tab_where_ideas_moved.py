"""Tab 2: topic-family landscape and movement drill-down."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import narrative_data as nd
from . import theme
from .narrative_common import badge, card_header, metric, notice, paper_list, section_label


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WWW_DIR = PROJECT_ROOT / "www"
LANDSCAPE_JS = WWW_DIR / "landscape.js"
LANDSCAPE_CSS = WWW_DIR / "landscape.css"
LANDSCAPE_DATA_DIR = WWW_DIR / "data"
COUNTRY_TOPIC_PATH = PROJECT_ROOT / "Dataset" / "dashboard_cache" / "country_topic_year.csv"
COUNTRY_CHOICES = [
    "Global",
    "China",
    "United States",
    "United Kingdom",
    "India",
    "Germany",
    "Japan",
    "France",
    "Italy",
    "Canada",
    "Indonesia",
]


QUADRANT_COLORS = {
    "Rising stars": "#059669",
    "Hidden gems": "#1d4ed8",
    "Fast growth, lower impact": "#d97706",
    "Mature or crowded": "#64748b",
}


def _clean_metric(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number

def _count_at_year(record: dict, year: int) -> int:
    counts = record.get("cumulative_paper_count_by_year", {}) or {}
    return int(counts.get(str(year), counts.get(year, 0)) or 0)


def _count_through(record: dict, start_year: int, end_year: int) -> int:
    end = _count_at_year(record, end_year)
    before = _count_at_year(record, start_year - 1) if start_year > 2000 else 0
    return max(0, end - before)

def _fmt_int(value: object) -> str:
    number = _clean_metric(value)
    return f"{number:,.0f}" if number is not None else ""


def _fmt_float(value: object, digits: int = 2) -> str:
    number = _clean_metric(value)
    return f"{number:,.{digits}f}" if number is not None else ""


def _add_frontier_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    growth_rank = np.log1p(out["growth"]).rank(pct=True)
    impact_rank = out["median_fwci"].rank(pct=True)
    velocity_rank = out["median_velocity"].rank(pct=True)
    out["frontier_score"] = 100 * (.40 * growth_rank + .35 * impact_rank + .25 * velocity_rank)
    out["top_signal"] = np.select(
        [
            growth_rank >= impact_rank.combine(velocity_rank, max),
            impact_rank >= growth_rank.combine(velocity_rank, max),
        ],
        ["Recent growth", "Normalized impact"],
        default="Citation velocity",
    )
    return out


def _normalize_topic_label(value: object) -> str:
    return " ".join(str(value or "").lower().replace("&", "and").split())


def _clean_frontier_topics() -> pd.DataFrame:
    """Fallback matrix from the old cache.

    This is only used if the landscape JSON is missing. The Tab 2 matrix should
    normally use _landscape_subtopics() as the source of truth so every topic
    matches the bubble map.
    """
    df = nd.growth_impact_matrix().copy()
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["primary_topic", "family", "growth", "median_fwci", "paper_count"]
    )
    for col in ["growth", "median_fwci", "median_velocity", "paper_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["growth"] = df["growth"].clip(lower=.01)
    df["median_velocity"] = df["median_velocity"].fillna(0)
    return _add_frontier_score(df.dropna(subset=["growth", "median_fwci", "paper_count"]))


def _metric_from_record(record: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = _clean_metric(record.get(name))
        if value is not None:
            return float(value)
    return float(default)


def _frontier_topics_for_period(start_year: int, end_year: int) -> pd.DataFrame:
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    # Source of truth: the same 2000-2025 topic JSON used by the bubble map.
    # This prevents the matrix from showing an older / different topic taxonomy.
    landscape_rows = _landscape_subtopics()
    if landscape_rows:
        old_cache = nd.growth_impact_matrix().copy()
        old_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if not old_cache.empty:
            for _, row in old_cache.iterrows():
                old_lookup[
                    (
                        _normalize_topic_label(row.get("family")),
                        _normalize_topic_label(row.get("primary_topic")),
                    )
                ] = row.to_dict()

        rows = []
        for record in landscape_rows:
            family = str(record.get("family", ""))
            topic = str(record.get("topic", ""))
            period_count = _count_through(record, start_year, end_year)
            if not family or not topic or period_count <= 0:
                continue

            fallback = old_lookup.get((_normalize_topic_label(family), _normalize_topic_label(topic)), {})
            growth = _metric_from_record(record, "growth_rate", "growth", default=np.nan)
            if not np.isfinite(growth):
                growth = _metric_from_record(fallback, "growth", "growth_rate", default=0.01)

            rows.append({
                "primary_topic": topic,
                "family": family,
                "paper_count": float(period_count),
                "growth": max(float(growth), 0.01),
                "median_fwci": _metric_from_record(record, "median_fwci", "fwci", default=_metric_from_record(fallback, "median_fwci", default=0.0)),
                "median_velocity": _metric_from_record(record, "citation_velocity", "median_velocity", default=_metric_from_record(fallback, "median_velocity", default=0.0)),
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return _add_frontier_score(out.dropna(subset=["primary_topic", "family", "growth", "median_fwci", "paper_count"]))

    # Fallback if www/data/subtopic_layout.json has not been generated yet.
    topics = _clean_frontier_topics()
    if topics.empty:
        return topics
    topics = topics.copy()
    topics = topics[pd.to_numeric(topics["paper_count"], errors="coerce").fillna(0) > 0]
    return topics


def _frontier_family_frame(topics: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Family-level matrix rows using the same family JSON as the bubble swarm."""
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    family_records = {str(row.get("family", "")): row for row in _landscape_families()}
    rows = []

    for family, sub in topics.groupby("family"):
        family_record = family_records.get(str(family), {})
        weight = sub["paper_count"].clip(lower=1)

        family_count = _count_through(family_record, start_year, end_year) if family_record else 0
        if family_count <= 0:
            family_count = float(sub["paper_count"].sum())

        rows.append({
            "name": family,
            "family": family,
            "primary_topic": family,
            "paper_count": float(family_count),
            "growth": float(np.average(sub["growth"], weights=weight)),
            "median_fwci": _metric_from_record(
                family_record,
                "median_fwci",
                "fwci",
                default=float(np.average(sub["median_fwci"], weights=weight)),
            ),
            "median_velocity": _metric_from_record(
                family_record,
                "citation_velocity",
                "median_velocity",
                default=float(np.average(sub["median_velocity"], weights=weight)),
            ),
            "topics_shown": int(len(sub)),
            "top_topics": ", ".join(sub.sort_values("frontier_score", ascending=False)["primary_topic"].head(5)),
        })

    return _add_frontier_score(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _apply_quadrants(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Assign quadrants on rank-scaled display positions.

    The raw growth/FWCI values are still kept for hover and detail cards.
    The plotted x/y coordinates use percent ranks so family bubbles spread out
    more evenly and the four quadrants look balanced in the demo.
    """
    out = df.copy()
    if out.empty:
        return out, .5, .5

    out["display_growth"] = out["growth"].rank(pct=True, method="first")
    out["display_fwci"] = out["median_fwci"].rank(pct=True, method="first")

    # Keep every point away from the borders so labels have room.
    out["display_growth"] = .08 + .84 * out["display_growth"]
    out["display_fwci"] = .08 + .84 * out["display_fwci"]

    x_cut = .5
    y_cut = .5
    out["quadrant"] = np.select(
        [
            (out["display_growth"] >= x_cut) & (out["display_fwci"] >= y_cut),
            (out["display_growth"] < x_cut) & (out["display_fwci"] >= y_cut),
            (out["display_growth"] >= x_cut) & (out["display_fwci"] < y_cut),
        ],
        ["Rising stars", "Hidden gems", "Fast growth, lower impact"],
        default="Mature or crowded",
    )
    return out, x_cut, y_cut


def _size_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    quantiles = out["paper_count"].quantile([.25, .5, .75]).to_dict()

    def bucket(count):
        if count <= quantiles[.25]:
            return "Small", 18
        if count <= quantiles[.5]:
            return "Medium", 28
        if count <= quantiles[.75]:
            return "Large", 38
        return "XL", 52

    buckets = out["paper_count"].map(bucket)
    out["size_tier"] = [label for label, _ in buckets]
    out["bubble_size"] = [size for _, size in buckets]
    return out


def _metric_group(*items):
    visible = [item for item in items if item is not None]
    return ui.div(*visible, class_="metric-grid") if visible else ui.div()


@lru_cache(maxsize=1)
def _landscape_families() -> list[dict[str, Any]]:
    path = LANDSCAPE_DATA_DIR / "topic_family_layout.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _landscape_subtopics() -> list[dict[str, Any]]:
    path = LANDSCAPE_DATA_DIR / "subtopic_layout.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _find_family(name: str) -> dict[str, Any] | None:
    if not name:
        return None
    return next((row for row in _landscape_families() if row.get("family") == name), None)


def _find_topic(family: str, topic: str) -> dict[str, Any] | None:
    if not family or not topic:
        return None
    return next(
        (
            row
            for row in _landscape_subtopics()
            if row.get("family") == family and row.get("topic") == topic
        ),
        None,
    )


def _metric_cards(items: list[tuple[str, str, str | None]]):
    cards = [metric(label, value, note) for label, value, note in items if value]
    if not cards:
        return None
    return ui.div(*cards, class_="metric-grid")


def _paper_records(record: dict[str, Any], year: int) -> list[dict[str, Any]]:
    papers = record.get("representative_papers") or []
    out = []
    for paper in papers:
        try:
            paper_year = int(paper.get("year", 0))
        except (TypeError, ValueError):
            paper_year = 0
        if paper_year <= year:
            out.append(paper)
    return out or papers


def _default_detail():
    return ui.div(
        ui.div("All AI", class_="landscape-breadcrumb"),
        ui.div("How to read this map", class_="landscape-detail-title"),
        ui.p(
            "Large circles are primary topic families. Inner dots are leading subtopics. Use the year slider to see both family and subtopic growth. Click a family or subtopic to inspect evidence.",
            class_="landscape-detail-copy",
        ),
        ui.tags.ol(
            ui.tags.li("The map covers all cached papers without double counting."),
            ui.tags.li("Circle size follows cumulative paper volume through the selected year."),
            ui.tags.li("Computer vision and LLMs are cross-cutting signals, not standalone primary families."),
            class_="small",
        ),
    )


def _empty_detail(family: str, topic: str, year: int):
    crumb = "All AI"
    if family:
        crumb += f" > {family}"
    if topic:
        crumb += f" > {topic}"
    return ui.div(
        ui.div(crumb, class_="landscape-breadcrumb"),
        ui.div(topic or family or "Selection", class_="landscape-detail-title"),
        ui.div(
            f"No papers available for this selection through {year}. Move the slider forward or reset the view.",
            class_="landscape-empty",
        ),
    )


def _family_detail(record: dict[str, Any], start_year: int, end_year: int):
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    family = str(record.get("family", ""))
    count = _count_through(record, start_year, end_year)

    if count <= 0:
        return _empty_detail(family, "", end_year)

    metrics = [
        ("Papers", _fmt_int(count), f"{start_year}–{end_year}"),
        ("Median FWCI", _fmt_float(record.get("median_fwci"), 2), None),
        ("Citation velocity", _fmt_float(record.get("citation_velocity"), 2), "per year"),
    ]

    subs = [s for s in _landscape_subtopics() if s.get("family") == family]
    subs = sorted(
        subs,
        key=lambda s: _count_through(s, start_year, end_year),
        reverse=True,
    )[:8]

    max_count = max([_count_through(s, start_year, end_year) for s in subs] or [1])

    subtopic_bars = ui.div(
        ui.tags.div(f"Top subtopics, {start_year}–{end_year}", class_="landscape-section-title"),
        *[
            ui.div(
                ui.div(
                    ui.span(str(s.get("topic", "")), class_="subtopic-rank-name"),
                    ui.span(f"{_count_through(s, start_year, end_year):,}", class_="subtopic-rank-value"),
                    class_="subtopic-rank-head",
                ),
                ui.div(
                    ui.div(
                        class_="subtopic-rank-fill",
                        style=f"width:{100 * _count_through(s, start_year, end_year) / max_count:.1f}%;",
                    ),
                    class_="subtopic-rank-track",
                ),
                class_="subtopic-rank-row",
            )
            for s in subs
            if _count_through(s, start_year, end_year) > 0
        ],
        class_="subtopic-rank-card",
    )

    return ui.div(
        ui.div(f"All AI > {family}", class_="landscape-breadcrumb"),
        ui.div(family, class_="landscape-detail-title"),
        ui.p(str(record.get("description", "")), class_="landscape-detail-copy"),
        _metric_cards(metrics),
        subtopic_bars,
        ui.p(ui.tags.strong("Representative papers"), class_="panel-label"),
        paper_list(_paper_records(record, end_year), limit=4),
    )


def _topic_detail(record: dict[str, Any], start_year: int, end_year: int):
    family = str(record.get("family", ""))
    topic = str(record.get("topic", ""))
    count = _count_through(record, start_year, end_year)
    if count <= 0:
        return _empty_detail(family, topic, end_year)
    metrics = [
        ("Papers", _fmt_int(count), f"Through {end_year}"),
        ("Growth rate", _fmt_float(record.get("growth_rate"), 2), None),
        ("Median FWCI", _fmt_float(record.get("median_fwci"), 2), None),
        ("Citation velocity", _fmt_float(record.get("citation_velocity"), 2), "per year"),
        ("Frontier score", _fmt_float(record.get("frontier_score"), 2), None),
    ]
    return ui.div(
        ui.div(f"All AI > {family} > {topic}", class_="landscape-breadcrumb"),
        ui.div(topic, class_="landscape-detail-title"),
        ui.p(f"Parent family: {family}", class_="landscape-detail-copy"),
        _metric_cards(metrics),
        ui.p(ui.tags.strong("Representative papers"), class_="panel-label"),
        paper_list(_paper_records(record, end_year), limit=4),
    )


def _landscape_container():
    return ui.div(
        ui.div(
            ui.div(
                ui.div(
                    ui.span("Selected period", class_="landscape-slider-title"),
                    ui.span("2000–2025", id="landscape-year-range-value"),
                    class_="landscape-slider-label",
                ),
                ui.div(
                    ui.div(class_="landscape-range-rail"),
                    ui.div(id="landscape-range-fill", class_="landscape-range-fill"),
                    ui.tags.input(
                        id="landscape-year-start",
                        type="range",
                        min="2000",
                        max="2025",
                        step="1",
                        value="2000",
                        aria_label="Landscape start year",
                        class_="landscape-range-input landscape-range-start",
                    ),
                    ui.tags.input(
                        id="landscape-year-end",
                        type="range",
                        min="2000",
                        max="2025",
                        step="1",
                        value="2025",
                        aria_label="Landscape end year",
                        class_="landscape-range-input landscape-range-end",
                    ),
                    class_="landscape-range-slider-stack",
                ),
                class_="landscape-range-control",
            ),
            ui.tags.button("Play", id="landscape-play", type="button", class_="btn btn-outline-secondary btn-sm"),
            ui.tags.button("Reset view", id="landscape-reset", type="button", class_="btn btn-outline-secondary btn-sm"),
            class_="landscape-toolbar",
        ),
        ui.div(
            ui.tags.svg(id="landscape-svg", role="img", aria_label="AI topic family bubble swarm"),
            ui.div("Loading topic landscape...", id="landscape-loading", class_="landscape-loading"),
            ui.div(id="landscape-tooltip", class_="landscape-tooltip"),
            id="landscape-shell",
            class_="landscape-svg-shell",
        ),
        id="landscape-root",
    )



TERM_SHIFT_BAD_TERMS = {
    "topic", "topics", "model", "models", "method", "methods",
    "approach", "approaches", "system", "systems", "study", "studies",
    "analysis", "research", "paper", "papers", "data", "based",
    "using", "use", "used", "application", "applications",
    "technique", "techniques", "algorithm", "algorithms",
    "neural", "natural", "language", "learning", "machine",
    "deep", "artificial", "intelligence", "network", "networks",
    "control", "theory", "programming", "mathematics",
    "classification", "framework", "performance", "information",
}

TERM_SHIFT_KEEP_SINGLE = {
    "detection", "adversarial", "transformer", "diffusion",
    "prompting", "robotics", "privacy", "fairness",
    "explainability", "cryptography", "optimization",
    "reinforcement", "hallucination", "reasoning", "alignment",
}


def _normalize_term(value: object) -> str:
    text = str(value or "").lower().strip()
    text = " ".join(text.split())
    replacements = {
        "algorithms": "algorithm",
        "models": "model",
        "networks": "network",
        "methods": "method",
        "techniques": "technique",
    }
    return replacements.get(text, text)


def _keep_term(value: object) -> bool:
    term = _normalize_term(value)
    if not term or term in TERM_SHIFT_BAD_TERMS:
        return False
    if len(term) <= 2 or term.isdigit():
        return False
    words = term.split()
    if len(words) >= 2:
        # Phrases carry the story better than broad one-word terms.
        return not all(word in TERM_SHIFT_BAD_TERMS for word in words)
    return term in TERM_SHIFT_KEEP_SINGLE


def _prepare_term_shift(df: pd.DataFrame, direction: str, limit: int = 8) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out.rename(columns={
        "early_share_per_1000": "early_share",
        "late_share_per_1000": "late_share",
        "delta_share_per_1000": "delta_share",
    })

    needed = ["term", "early_count", "late_count", "early_share", "late_share", "delta_share", "growth_ratio"]
    for col in needed:
        if col not in out.columns:
            out[col] = 0

    for col in ["early_count", "late_count", "early_share", "late_share", "delta_share", "growth_ratio"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["term"] = out["term"].map(_normalize_term)
    out = out[out["term"].map(_keep_term)].copy()
    out = out.drop_duplicates(subset=["term"], keep="first")

    if direction == "rising":
        out = out.sort_values("delta_share", ascending=False)
    else:
        out = out.sort_values("delta_share", ascending=True)

    out = out.head(limit).copy()
    if direction == "rising":
        return out.sort_values("delta_share")
    return out.assign(loss_positive=out["delta_share"].abs()).sort_values("loss_positive")


def movement_ui():
    return ui.nav_panel(
        "Where ideas moved",
        ui.include_css(LANDSCAPE_CSS),
        ui.include_js(LANDSCAPE_JS, method="link_files"),
        ui.div(
            ui.div(
                badge("OpenAlex 2000-2025"),
                badge("Topic family grouping"),
                badge("Topic drill-down"),
                class_="badge-row",
            ),
            ui.h2("Where ideas moved"),
            ui.p(
                "AI ideas concentrated in topic families, then shifted toward newer research language after 2020.",
                class_="tab-insight",
            ),
            class_="tab-heading",
        ),
        section_label("Topic-family drill-down"),
        ui.layout_columns(
            ui.card(
                card_header(
                    "AI topic families over time",
                    "Drag the year slider to see how each family grows. Click a family to inspect subtopics and representative papers.",
                ),
                _landscape_container(),
                notice("Families are single-label groups to avoid double counting. Computer vision and LLMs are treated as cross-cutting signals across healthcare, robotics, NLP, and core ML."),
                class_="hero-card landscape-card",
            ),
            ui.card(
                card_header("Selected Bubble Detail"),
                ui.output_ui("landscape_profile"),
                class_="landscape-detail-card",
            ),
            col_widths=[8, 4],
        ),
        section_label("Topic momentum and selected-family language shift"),
        ui.layout_columns(
            ui.card(
                card_header(
                    "Frontier map: growth vs normalized impact",
                    "Click a family bubble or matrix point. The term charts on the right switch to that family when cached data exists.",
                ),
                output_widget("movement_growth_impact_matrix"),
                ui.tags.script("""
                (function() {
                  function bindMovementMatrix() {
                    const wrapper = document.getElementById("movement_growth_impact_matrix");
                    const el = wrapper && (wrapper.classList.contains("js-plotly-plot") ? wrapper : wrapper.querySelector(".js-plotly-plot"));
                    if (!el || el.dataset.movementFrontierBound || !window.Shiny || typeof el.on !== "function") return false;
                    el.dataset.movementFrontierBound = "1";
                    el.on("plotly_click", function(eventData) {
                      const point = eventData && eventData.points && eventData.points[0];
                      const d = point && point.customdata;
                      if (d) Shiny.setInputValue("movement_frontier_click", {
                        kind: d[0], family: d[1], topic: d[2], name: d[3]
                      }, {priority: "event"});
                    });
                    return true;
                  }
                  let tries = 0;
                  const retry = setInterval(function() {
                    bindMovementMatrix();
                    tries += 1;
                    if (tries > 80) clearInterval(retry);
                  }, 250);
                  document.addEventListener("shiny:value", function(e) {
                    if (e.target && e.target.id === "movement_growth_impact_matrix") {
                      setTimeout(bindMovementMatrix, 80);
                    }
                  });
                })();
                """),
                notice("Right means faster recent growth. Up means stronger normalized impact. Larger bubbles mean more papers in the selected range."),
            ),
            ui.div(
                ui.card(
                    card_header(
                        "Terms gaining attention",
                        "Filtered by selected family when available; otherwise shows All AI.",
                    ),
                    output_widget("rising_terms_bar"),
                    class_="term-shift-card",
                ),
                ui.card(
                    card_header(
                        "Terms losing relative attention",
                        "Filtered by selected family when available; otherwise shows All AI.",
                    ),
                    output_widget("fading_terms_bar"),
                    class_="term-shift-card",
                ),
                class_="side-stack",
            ),
            col_widths=[8, 4],
        ),
    )

def movement_server(input, output, session):
    selected_family = reactive.Value("")
    selected_topic = reactive.Value("")
    selected_year = reactive.Value(2025)
    selected_start_year = reactive.Value(2000)
    selected_end_year = reactive.Value(2025)

    @reactive.effect
    @reactive.event(input.landscape_year_current)
    def _sync_landscape_year():
        try:
            year = int(input.landscape_year_current())
        except (TypeError, ValueError):
            year = 2025
        selected_year.set(max(2000, min(2025, year)))

    @reactive.effect
    @reactive.event(input.landscape_year_start)
    def _sync_landscape_start_year():
        try:
            year = int(input.landscape_year_start())
        except (TypeError, ValueError):
            year = 2000
        selected_start_year.set(max(2000, min(2025, year)))


    @reactive.effect
    @reactive.event(input.landscape_year_end)
    def _sync_landscape_end_year():
        try:
            year = int(input.landscape_year_end())
        except (TypeError, ValueError):
            year = 2025
        selected_end_year.set(max(2000, min(2025, year)))
    @reactive.effect
    @reactive.event(input.landscape_family_click)
    def _select_landscape_family():
        selected_family.set(str(input.landscape_family_click() or ""))
        if not selected_family():
            selected_topic.set("")

    @reactive.effect
    @reactive.event(input.landscape_topic_click)
    def _select_landscape_topic():
        selected_topic.set(str(input.landscape_topic_click() or ""))

    @reactive.effect
    @reactive.event(input.landscape_reset)
    def _reset_landscape_selection():
        selected_family.set("")
        selected_topic.set("")

    @render.ui
    def landscape_profile():
        family = selected_family()
        topic = selected_topic()
        start_year = selected_start_year()
        end_year = selected_end_year()

        if start_year > end_year:
            start_year, end_year = end_year, start_year

        if topic and family:
            record = _find_topic(family, topic)
            if record:
                return _topic_detail(record, start_year, end_year)

        if family:
            record = _find_family(family)
            if record:
                return _family_detail(record, start_year, end_year)

        return _default_detail()

    def _term_scope_for(df: pd.DataFrame) -> str:
        family = str(selected_family() or "").strip()
        if "scope" not in df.columns:
            return "All AI"
        scopes = set(df["scope"].dropna().astype(str))
        if family and family in scopes:
            return family
        if "All AI" in scopes:
            return "All AI"
        return sorted(scopes)[0] if scopes else "All AI"

    def _terms_for_selected_scope(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        scope = _term_scope_for(df)
        if "scope" in df.columns:
            sub = df[df["scope"].astype(str).eq(scope)].copy()
            if sub.empty and scope != "All AI":
                sub = df[df["scope"].astype(str).eq("All AI")].copy()
                scope = "All AI"
            return sub, scope
        return df.copy(), scope
    
    @reactive.effect
    @reactive.event(input.landscape_year_start)
    def _sync_landscape_start_year():
        try:
            year = int(input.landscape_year_start())
        except (TypeError, ValueError):
            year = 2000
        selected_start_year.set(max(2000, min(2025, year)))


    @reactive.effect
    @reactive.event(input.landscape_year_end)
    def _sync_landscape_end_year():
        try:
            year = int(input.landscape_year_end())
        except (TypeError, ValueError):
            year = 2025
        selected_end_year.set(max(2000, min(2025, year)))


    @render_widget
    def rising_terms_bar():
        raw, scope = _terms_for_selected_scope(nd.rising_terms().copy())
        df = _prepare_term_shift(raw, "rising", limit=8)
        if df.empty:
            return theme.empty_figure(f"No rising terms found for {scope}.")

        fig = go.Figure(
            go.Bar(
                x=df["delta_share"],
                y=df["term"],
                orientation="h",
                marker=dict(color=theme.ACCENT),
                customdata=df[["early_count", "late_count", "early_share", "late_share", "growth_ratio"]].fillna(0),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Share gain %{x:.2f} per 1,000 papers<br>"
                    "Early docs %{customdata[0]:,.0f}<br>"
                    "Late docs %{customdata[1]:,.0f}<br>"
                    "Early share %{customdata[2]:.2f} / 1,000<br>"
                    "Late share %{customdata[3]:.2f} / 1,000<br>"
                    "Growth ratio %{customdata[4]:.2f}x<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title_text="", zeroline=True, zerolinecolor="rgba(255,255,255,.28)")
        fig.update_yaxes(title_text="", automargin=True)
        fig.update_layout(
            title=dict(text=scope, font=dict(size=11, color=theme.SUBTLE_TEXT), y=0.98),
            showlegend=False,
            margin=dict(l=8, r=10, t=24, b=24),
        )
        return theme.style(fig, height=225)

    @render_widget
    def fading_terms_bar():
        raw, scope = _terms_for_selected_scope(nd.fading_terms().copy())
        df = _prepare_term_shift(raw, "fading", limit=8)
        if df.empty:
            return theme.empty_figure(f"No fading terms found for {scope}.")

        fig = go.Figure(
            go.Bar(
                x=df["loss_positive"],
                y=df["term"],
                orientation="h",
                marker=dict(color="#FF8CA1"),
                customdata=df[["early_count", "late_count", "early_share", "late_share", "growth_ratio", "delta_share"]].fillna(0),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Share loss %{x:.2f} per 1,000 papers<br>"
                    "Early docs %{customdata[0]:,.0f}<br>"
                    "Late docs %{customdata[1]:,.0f}<br>"
                    "Early share %{customdata[2]:.2f} / 1,000<br>"
                    "Late share %{customdata[3]:.2f} / 1,000<br>"
                    "Growth ratio %{customdata[4]:.2f}x<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="", automargin=True)
        fig.update_layout(
            title=dict(text=scope, font=dict(size=11, color=theme.SUBTLE_TEXT), y=0.98),
            showlegend=False,
            margin=dict(l=8, r=10, t=24, b=24),
        )
        return theme.style(fig, height=225)


    @reactive.effect
    @reactive.event(input.movement_frontier_click)
    def _select_movement_frontier():
        payload = input.movement_frontier_click()
        if not payload:
            return
        kind = payload.get("kind")
        family = str(payload.get("family") or "")
        topic = str(payload.get("topic") or "")
        if kind == "family":
            selected_family.set(family)
            selected_topic.set("")
        elif kind == "topic":
            selected_family.set(family)
            selected_topic.set(topic)

    @render_widget
    def movement_growth_impact_matrix():
        start_year = selected_start_year()
        end_year = selected_end_year()
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        topics = _frontier_topics_for_period(start_year, end_year)
        if topics.empty:
            return theme.empty_figure("No growth-impact matrix data")

        family = selected_family()
        topic = selected_topic()

        # Keep the matrix at family level. Subtopic drill-down stays in the
        # bubble detail panel, which keeps this chart readable.
        plot_df = _frontier_family_frame(topics, start_year, end_year)
        kind = "family"
        name_col = "name"

        if plot_df.empty:
            return theme.empty_figure("No growth-impact matrix data")

        plot_df, x_cut, y_cut = _apply_quadrants(plot_df)
        plot_df = _size_buckets(plot_df)
        plot_df["name"] = plot_df[name_col].astype(str)
        selected_name = topic if kind == "topic" else family
        top_labels = set(plot_df.sort_values("frontier_score", ascending=False).head(4)["name"])
        top_labels.update(plot_df.sort_values("growth", ascending=False).head(1)["name"])
        top_labels.update(plot_df.sort_values("median_fwci", ascending=False).head(1)["name"])
        if selected_name:
            top_labels.add(str(selected_name))

        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
        fig = go.Figure()
        bg = {
            "Rising stars": "rgba(5,150,105,.09)",
            "Hidden gems": "rgba(29,78,216,.08)",
            "Fast growth, lower impact": "rgba(217,119,6,.08)",
            "Mature or crowded": "rgba(100,116,139,.07)",
        }
        for x0, x1, y0, y1, label in [
            (x_cut, x_max, y_cut, y_max, "Rising stars"),
            (x_min, x_cut, y_cut, y_max, "Hidden gems"),
            (x_cut, x_max, y_min, y_cut, "Fast growth, lower impact"),
            (x_min, x_cut, y_min, y_cut, "Mature or crowded"),
        ]:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=bg[label], line=dict(width=0), layer="below")

        for quad, sub in plot_df.groupby("quadrant"):
            is_selected = sub["name"].eq(str(selected_name))
            show_label = sub["name"].isin(top_labels)
            custom = pd.DataFrame({
                "kind": kind,
                "family": sub["family"],
                "topic": sub["primary_topic"],
                "name": sub["name"],
                "paper_count": sub["paper_count"],
                "growth": sub["growth"],
                "median_fwci": sub["median_fwci"],
                "median_velocity": sub["median_velocity"],
                "frontier_score": sub["frontier_score"],
                "size_tier": sub["size_tier"],
            })
            fig.add_trace(go.Scatter(
                x=sub["display_growth"],
                y=sub["display_fwci"],
                mode="markers+text",
                name=quad,
                text=sub["name"].where(show_label, ""),
                textposition="top center",
                textfont=dict(size=10, color="#0f172a"),
                marker=dict(
                    size=sub["bubble_size"],
                    color=QUADRANT_COLORS[quad],
                    opacity=np.where(show_label | is_selected, .86, .48),
                    line=dict(color=np.where(is_selected, "#020617", "#ffffff"), width=np.where(is_selected, 3, 1.2)),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[3]}</b><br>"
                    f"{start_year}-{end_year}: " + "%{customdata[4]:,.0f} papers<br>"
                    "Growth ratio %{customdata[5]:.2f}x<br>"
                    "Median FWCI %{customdata[6]:.2f}<br>"
                    "Citation velocity %{customdata[7]:.1f}/yr<br>"
                    "Frontier score %{customdata[8]:.1f}<extra></extra>"
                ),
            ))

        fig.add_vline(x=x_cut, line_color="#0f172a", line_dash="dash", line_width=1.5)
        fig.add_hline(y=y_cut, line_color="#0f172a", line_dash="dash", line_width=1.5)

        fig.update_xaxes(
            title_text="Growth momentum rank",
            range=[x_min, x_max],
            tickmode="array",
            tickvals=[0, .25, .5, .75, 1],
            ticktext=["Low", "", "Median", "", "High"],
            showgrid=True,
            gridcolor="#edf2f7",
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="Normalized impact rank",
            range=[y_min, y_max],
            tickmode="array",
            tickvals=[0, .25, .5, .75, 1],
            ticktext=["Low", "", "Median", "", "High"],
            showgrid=True,
            gridcolor="#edf2f7",
            zeroline=False,
        )
        fig.update_layout(margin=dict(l=60, r=24, t=54, b=56), legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)))
        return theme.style(fig, height=470)

    @render.ui
    def movement_frontier_detail():
        start_year = selected_start_year()
        end_year = selected_end_year()
        if start_year > end_year:
            start_year, end_year = end_year, start_year

        topics = _frontier_topics_for_period(start_year, end_year)
        if topics.empty:
            return ui.p("No matrix detail data is available.", class_="text-muted small")

        family = selected_family()
        topic = selected_topic()
        if topic:
            row = topics[topics["primary_topic"].eq(topic)]
            if not row.empty:
                r = row.iloc[0]
                return ui.div(
                    _metric_group(
                        metric("Selected topic", str(r["primary_topic"]), str(r["family"])),
                        metric("Papers", f"{float(r['paper_count']):,.0f}", f"{start_year}-{end_year}"),
                        metric("Growth ratio", f"{float(r['growth']):.2f}x"),
                        metric("Median FWCI", f"{float(r['median_fwci']):.2f}"),
                        metric("Citation velocity", f"{float(r['median_velocity']):.1f}/yr"),
                        metric("Frontier score", f"{float(r['frontier_score']):.1f}"),
                    ),
                    ui.p("This topic is matched to the same topic label used in the bubble drill-down.", class_="interpretation"),
                )

        if family:
            sub = topics[topics["family"].eq(family)].sort_values("frontier_score", ascending=False)
            if not sub.empty:
                items = [
                    ui.tags.li(
                        ui.tags.strong(str(row["primary_topic"])),
                        ui.br(),
                        ui.span(
                            f"Score {float(row['frontier_score']):.1f} | {float(row['growth']):.2f}x growth | FWCI {float(row['median_fwci']):.2f}",
                            class_="text-muted small",
                        ),
                    )
                    for _, row in sub.head(5).iterrows()
                ]
                return ui.div(
                    _metric_group(
                        metric("Selected family", family),
                        metric("Topics shown", f"{len(sub):,.0f}"),
                        metric("Family papers", f"{float(sub['paper_count'].sum()):,.0f}", f"{start_year}-{end_year}"),
                        metric("Median FWCI", f"{float(sub['median_fwci'].median()):.2f}"),
                    ),
                    ui.p("Click a matrix point to sync the selected bubble detail above.", class_="interpretation"),
                    ui.p(ui.tags.strong("Top 5 subtopics"), class_="panel-label"),
                    ui.tags.ul(*items, class_="paper-list"),
                )

        return ui.div(
            ui.p(
                "Each bubble is a topic family. Click one family to convert the matrix into its subtopics.",
                class_="interpretation",
            ),
            ui.p("The selected year range controls bubble size and which topics appear.", class_="text-muted small"),
        )

    @render_widget
    def top_countries():
        df = nd.top_countries(20).copy()
        if df.empty:
            return theme.empty_figure("Country output cache is empty.")
        df = df[df["country"].isin(COUNTRY_CHOICES[1:])].copy()
        if df.empty:
            df = nd.top_countries(10).copy()
        df = df.sort_values("papers", ascending=True)
        selected = str(input.country_focus() or "Global")
        top3 = set(df.sort_values("papers", ascending=False).head(3)["country"])
        if selected == "Global":
            colors = np.where(df["country"].isin(top3), "#2563eb", "#cbd5e1")
        else:
            colors = np.where(df["country"].eq(selected), "#2563eb", "#cbd5e1")
        fig = go.Figure(go.Bar(
            x=df["papers"],
            y=df["country"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#ffffff", width=1.2)),
            customdata=df[["country", "papers"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} papers<extra></extra>",
        ))
        fig.update_xaxes(title_text="Paper count, 2000-2025", showgrid=True, gridcolor="#edf2f7")
        fig.update_yaxes(title_text="", automargin=True)
        fig.update_layout(showlegend=False, margin=dict(l=8, r=14, t=12, b=46))
        return theme.style(fig, height=430)

    @render_widget
    def mutation_river():
        selected = str(input.country_focus() or "Global")
        if selected == "Global" or not COUNTRY_TOPIC_PATH.exists():
            df = nd.bucket_year_counts().copy()
            label_col = "topic_bucket"
            count_col = "count"
            chart_note = "Global topic-family share"
        else:
            df = nd.country_topic_year_counts()
            df = df[df["country"].eq(selected)].copy()
            label_col = "topic_bucket"
            count_col = "count"
            chart_note = f"{selected} topic-family share from country-topic aggregates"
        if df.empty:
            return theme.empty_figure(f"No topic stream data for {selected}")
        totals = df.groupby("year")[count_col].transform("sum").replace(0, np.nan)
        df["share"] = 100 * df[count_col] / totals
        latest = df[df["year"].eq(df["year"].max())].groupby(label_col)["share"].sum().sort_values(ascending=False)
        keep = list(latest.head(6).index)
        stream = df[df[label_col].isin(keep)].copy()
        other = df[~df[label_col].isin(keep)].groupby("year", as_index=False)["share"].sum()
        if not other.empty:
            other[label_col] = "Other"
            stream = pd.concat([stream[["year", label_col, "share"]], other[["year", label_col, "share"]]], ignore_index=True)
        fig = go.Figure()
        for i, topic in enumerate(stream[label_col].drop_duplicates()):
            sub = stream[stream[label_col].eq(topic)].sort_values("year")
            color = "#94a3b8" if topic == "Other" else theme.PALETTE[i % len(theme.PALETTE)]
            fig.add_trace(go.Scatter(
                x=sub["year"],
                y=sub["share"],
                stackgroup="one",
                mode="lines",
                name=topic,
                line=dict(width=.7, color=color),
            ))
        fig.add_annotation(
            x=.01,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=chart_note,
            font=dict(size=11, color="#475569"),
            align="left",
        )
        fig.update_yaxes(title_text="Share of AI papers (%)", range=[0, 100])
        fig.update_xaxes(title_text="Year")
        fig.update_layout(margin=dict(l=52, r=8, t=42, b=42), autosize=True)
        return theme.style(fig, height=620)

