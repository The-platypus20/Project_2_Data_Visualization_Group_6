"""AI research narrative dashboard.

Run with: shiny run app.py --reload
"""
from __future__ import annotations

from shiny import App, ui

from src import theme
from src.mod_narrative import narrative_server, narrative_ui

app_ui = ui.page_navbar(
    *narrative_ui(),
    title="AI Research Landscape Map",
    id="navbar",
    fillable=False,
    header=ui.head_content(ui.tags.style(theme.DASHBOARD_CSS)),
    footer=ui.div(
        ui.tags.small(
            "Data: precomputed OpenAlex dashboard cache files in Dataset/dashboard_cache/. "
            "Topic groups are metadata-derived from OpenAlex topic labels. Raw OpenAlex exports are not required at runtime.",
            class_="text-muted",
        ),
        class_="px-3 py-2",
    ),
)


def server(input, output, session):
    narrative_server(input, output, session)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
WWW_DIR = PROJECT_ROOT / "www"

app = App(app_ui, server, static_assets=WWW_DIR)
