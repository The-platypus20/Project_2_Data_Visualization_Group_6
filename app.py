"""Exploring Growth and Concentration in AI Research — snapshot dashboard."""
from __future__ import annotations

from shiny import App, reactive, render, ui

from src import data as datamod
from src import theme
from src.mod_concentration import concentration_ui, concentration_server
from src.mod_growth import growth_ui, growth_server
from src.mod_impact import impact_ui, impact_server
from src.mod_pressure import pressure_ui, pressure_server

SNAPSHOT = datamod.load_snapshot()
YEAR_MIN, YEAR_MAX = datamod.year_bounds(SNAPSHOT)
SUMMARY = datamod.snapshot_summary(SNAPSHOT)

_sidebar = ui.sidebar(
    ui.h5("Snapshot"),
    ui.output_ui("snapshot_notice"),
    ui.hr(),
    ui.h5("Filters"),
    ui.input_slider(
        "f_years",
        "Year range",
        min=YEAR_MIN,
        max=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX),
        step=1,
        sep="",
    ),
    ui.input_action_button("f_reset", "Reset filters", class_="btn-sm btn-outline-secondary"),
    ui.hr(),
    ui.p(
        "Counts and time-series use precomputed OpenAlex statistics. "
        "Sample-based views are explicitly labeled in the cards.",
        class_="text-muted small",
    ),
    width=300,
)

app_ui = ui.page_navbar(
    growth_ui(),
    impact_ui(),
    concentration_ui(),
    pressure_ui(),
    sidebar=_sidebar,
    title="AI Research: Growth & Concentration",
    id="navbar",
    fillable=False,
    header=ui.head_content(ui.tags.style(theme.DASHBOARD_CSS)),
    footer=ui.div(
        ui.tags.small(
            "Data: precomputed OpenAlex snapshot. Exact grouped counts are used wherever "
            "possible; paper-level distribution views come from the stratified sample.",
            class_="text-muted",
        ),
        class_="px-3 py-2",
    ),
)


def server(input, output, session):

    @render.ui
    def snapshot_notice():
        parts = []
        generated = SUMMARY.get("generated_at_utc")
        if generated:
            parts.append(f"Generated: {generated}")
        sample_n = SUMMARY.get("sample_size_materialized")
        if sample_n is not None:
            parts.append(f"Sampled papers: {int(sample_n):,}")
        stats_path = SUMMARY.get("stats_dir")
        if stats_path:
            parts.append(f"Source: {stats_path}")
        return ui.div(
            *(ui.tags.div(part) for part in parts),
            class_="text-muted small",
        )

    @reactive.effect
    @reactive.event(input.f_reset)
    def _reset():
        ui.update_slider("f_years", value=(YEAR_MIN, YEAR_MAX))

    @reactive.calc
    def year_range() -> tuple[int, int]:
        return tuple(input.f_years())

    @reactive.calc
    def sampled_filtered():
        lo, hi = year_range()
        return datamod.filter_sampled_papers(SNAPSHOT, lo, hi)

    growth_server(input, output, session, SNAPSHOT, year_range)
    impact_server(input, output, session, SNAPSHOT, sampled_filtered, year_range)
    concentration_server(input, output, session, SNAPSHOT, sampled_filtered, year_range)
    pressure_server(input, output, session, SNAPSHOT, sampled_filtered, year_range)


app = App(app_ui, server)
