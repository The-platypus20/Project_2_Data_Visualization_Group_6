"""Exploring Growth and Concentration in AI Research — Shiny dashboard.

Run with:   shiny run app.py --reload     (or)     python -m shiny run app.py

The app is organised as a navbar with four analytical tabs (Growth, Impact,
Concentration, Pressure Index) that share one sidebar of filters. Each tab's
UI and server logic lives in its own ``src/mod_*.py`` module for easy tracking;
shared data, geography, metrics and styling live in the other ``src`` modules.
"""
from __future__ import annotations

from shiny import App, reactive, ui

from src import data as datamod
from src import theme
from src.data import TOPIC_BUCKETS, VENUE_GROUPS
from src.geo import REGIONS
from src.mod_growth import growth_ui, growth_server
from src.mod_impact import impact_ui, impact_server
from src.mod_concentration import concentration_ui, concentration_server
from src.mod_pressure import pressure_ui, pressure_server
from src.mod_mlnlp import mlnlp_ui, mlnlp_server

YEAR_MIN, YEAR_MAX = datamod.year_bounds()

_sidebar = ui.sidebar(
    ui.h5("Filters"),
    ui.input_slider("f_years", "Year range", min=YEAR_MIN, max=YEAR_MAX,
                    value=(YEAR_MIN, YEAR_MAX), step=1, sep=""),
    ui.input_checkbox_group("f_regions", "Region", REGIONS, selected=REGIONS),
    ui.input_checkbox_group("f_topics", "Topic", TOPIC_BUCKETS, selected=TOPIC_BUCKETS),
    ui.input_checkbox_group("f_venues", "Venue / source", VENUE_GROUPS, selected=VENUE_GROUPS),
    ui.input_action_button("f_reset", "Reset filters", class_="btn-sm btn-outline-secondary"),
    ui.hr(),
    ui.p("Selections apply to every tab.", class_="text-muted small"),
    width=300,
)

app_ui = ui.page_navbar(
    growth_ui(),
    impact_ui(),
    concentration_ui(),
    pressure_ui(),
    mlnlp_ui(),
    sidebar=_sidebar,
    title="AI Research: Growth & Concentration",
    id="navbar",
    fillable=False,
    header=ui.head_content(ui.tags.style(theme.DASHBOARD_CSS)),
    footer=ui.div(
        ui.tags.small(
            "Data: OpenAlex (sample of highly-cited AI-related works, 2000–2026). "
            "All metrics and charts reflect the current filter selections.",
            class_="text-muted"),
        class_="px-3 py-2",
    ),
)


def server(input, output, session):

    @reactive.effect
    @reactive.event(input.f_reset)
    def _reset():
        ui.update_slider("f_years", value=(YEAR_MIN, YEAR_MAX))
        ui.update_checkbox_group("f_regions", selected=REGIONS)
        ui.update_checkbox_group("f_topics", selected=TOPIC_BUCKETS)
        ui.update_checkbox_group("f_venues", selected=VENUE_GROUPS)

    @reactive.calc
    def filtered():
        """Single shared filtered frame driven by the sidebar inputs."""
        df = datamod.load_data()
        lo, hi = input.f_years()
        df = df[(df["year"] >= lo) & (df["year"] <= hi)]

        regions = set(input.f_regions())
        if regions and regions != set(REGIONS):
            df = df[df["regions"].map(lambda rs: bool(regions.intersection(rs)))]

        topics = set(input.f_topics())
        if topics and topics != set(TOPIC_BUCKETS):
            df = df[df["topic_bucket"].isin(topics)]

        venues = set(input.f_venues())
        if venues and venues != set(VENUE_GROUPS):
            df = df[df["venue_group"].isin(venues)]

        return df

    growth_server(input, output, session, filtered)
    impact_server(input, output, session, filtered)
    concentration_server(input, output, session, filtered)
    pressure_server(input, output, session, filtered)
    mlnlp_server(input, output, session)


app = App(app_ui, server)
