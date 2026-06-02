"""Exploring Growth and Concentration in AI Research — live OpenAlex dashboard."""
from __future__ import annotations

from shiny import App, reactive, render, ui

from src import data as datamod
from src import theme
from src.data import TOPIC_BUCKETS, VENUE_GROUPS
from src.geo import REGIONS
from src.mod_growth import growth_ui, growth_server
from src.mod_impact import impact_ui, impact_server
from src.mod_concentration import concentration_ui, concentration_server
from src.mod_pressure import pressure_ui, pressure_server

YEAR_MIN, YEAR_MAX = datamod.DEFAULT_YEAR_MIN, datamod.DEFAULT_YEAR_MAX

_sidebar = ui.sidebar(
    ui.h5("Filters"),
    ui.input_text("f_query", "OpenAlex search", datamod.DEFAULT_QUERY,
                  placeholder="e.g. artificial intelligence"),
    ui.input_numeric("f_limit", "Max works to load", datamod.DEFAULT_MAX_WORKS,
                     min=100, max=10000, step=100),
    ui.input_action_button("f_reload", "Load from OpenAlex", class_="btn-sm btn-primary"),
    ui.output_ui("f_notice"),
    ui.hr(),
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
    sidebar=_sidebar,
    title="AI Research: Growth & Concentration",
    id="navbar",
    fillable=False,
    header=ui.head_content(ui.tags.style(theme.DASHBOARD_CSS)),
    footer=ui.div(
        ui.tags.small(
            "Data: live OpenAlex API results, sorted by citation count and capped by "
            "the current 'Max works to load' setting. Charts reflect the loaded sample "
            "under the current sidebar filters.",
            class_="text-muted"),
        class_="px-3 py-2",
    ),
)


def server(input, output, session):
    dataset_state = reactive.Value(datamod.empty_frame())
    notice_state = reactive.Value("Waiting for OpenAlex data.")

    def _load_from_openalex() -> None:
        with reactive.isolate():
            query = (input.f_query() or "").strip()
            try:
                max_works = int(input.f_limit() or datamod.DEFAULT_MAX_WORKS)
            except (TypeError, ValueError):
                max_works = datamod.DEFAULT_MAX_WORKS
        notice_state.set(
            f"Loading up to {max_works:,} works for '{query or datamod.DEFAULT_QUERY}'..."
        )
        try:
            df = datamod.load_data(
                query=query or datamod.DEFAULT_QUERY,
                year_min=YEAR_MIN,
                year_max=YEAR_MAX,
                max_records=max_works,
            )
        except Exception as exc:
            dataset_state.set(datamod.empty_frame())
            notice_state.set(str(exc))
            return
        dataset_state.set(df)
        notice_state.set(
            f"Loaded {len(df):,} works from OpenAlex for '{query or datamod.DEFAULT_QUERY}'."
        )

    @reactive.effect
    def _initial_load():
        _load_from_openalex()

    @reactive.effect
    @reactive.event(input.f_reload)
    def _reload():
        _load_from_openalex()

    @render.ui
    def f_notice():
        msg = notice_state.get()
        css = "text-muted small"
        if "Set OPENALEX_API_KEY" in msg or "rejected" in msg or "rate limit" in msg:
            css = "text-danger small"
        return ui.div(msg, class_=css)

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
        df = dataset_state.get()
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


app = App(app_ui, server)
