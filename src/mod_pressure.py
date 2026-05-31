"""Pressure Index tab: a composite "research pressure" indicator over time.

No wireframe was supplied for this tab, so it is designed to synthesise the
signals from the first three tabs into one explainable index. Four per-year
components are min-max normalised across the visible period (so the index is
*relative to the selected window*) and averaged:

* Volume growth     - YoY growth in papers (faster growth = more pressure)
* Impact dilution   - share of papers below the selection's median citations
* Geographic concn. - top-5 country share of that year's papers
* Topic crowding     - 1 − normalised topic entropy (fewer topics dominating)

Index = 100 × mean(normalised components). It is a descriptive composite, not
a validated metric, and is labelled as such in the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import data as datamod
from . import metrics, theme

_COMPONENTS = ["Volume growth", "Impact dilution", "Geographic concentration",
               "Topic crowding"]


def pressure_ui():
    return ui.nav_panel(
        "Pressure Index",
        ui.layout_columns(
            ui.value_box("Pressure index (latest)", ui.output_text("p_latest"),
                         ui.span("0–100, relative to selected period", class_="text-muted small")),
            ui.value_box("Change vs first year", ui.output_text("p_change"),
                         ui.span("index points", class_="text-muted small")),
            ui.value_box("Dominant driver", ui.output_text("p_driver"),
                         ui.span("largest component in latest year", class_="text-muted small")),
            col_widths=[4, 4, 4], fill=False,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Composite research pressure over time"),
                output_widget("p_index"),
                ui.card_footer(ui.span(
                    "Average of four normalised pressure components (0–100). "
                    "Higher = more competitive / crowded conditions for the period.",
                    class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Pressure components over time"),
                output_widget("p_components"),
                ui.card_footer(ui.span("Each component min-max normalised across the visible years.",
                                       class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Latest-year pressure profile"),
                output_widget("p_radar"),
            ),
            ui.card(
                ui.card_header("How the index is built"),
                ui.markdown(
                    "**Research Pressure Index** combines four signals, each rescaled "
                    "to 0–1 across the selected years and then averaged:\n\n"
                    "- **Volume growth** — year-over-year growth in publications.\n"
                    "- **Impact dilution** — share of papers below the median citation count.\n"
                    "- **Geographic concentration** — top-5 country share of output.\n"
                    "- **Topic crowding** — inverse of topic entropy (few topics dominating).\n\n"
                    "It is a *descriptive composite* for exploration, not a validated "
                    "benchmark. Adjust the sidebar filters to see how pressure differs "
                    "across regions, topics and venues."
                ),
            ),
            col_widths=[5, 7],
        ),
    )


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def pressure_server(input, output, session, filtered):

    @reactive.calc
    def components() -> pd.DataFrame:
        df = filtered()
        if df.empty or df["year"].nunique() < 2:
            return pd.DataFrame(columns=_COMPONENTS)
        years = sorted(df["year"].unique())
        median_cit = df["citation_count"].median()
        n_buckets = max(2, df["topic_bucket"].nunique())
        counts = df.groupby("year").size().sort_index()
        growth = metrics.yoy_growth(counts).reindex(years)

        rows = {}
        for y in years:
            sub = df[df["year"] == y]
            dilution = (sub["citation_count"] < median_cit).mean()
            ex = datamod.explode_countries(sub)
            concn = (metrics.top_n_share(ex.groupby("iso2").size().values, 5) / 100.0
                     if not ex.empty else np.nan)
            ent = metrics.shannon_entropy(sub["topic_bucket"].value_counts().values)
            crowding = 1.0 - ent / np.log2(n_buckets)
            rows[y] = [growth.get(y, np.nan), dilution, concn, crowding]

        raw = pd.DataFrame.from_dict(rows, orient="index", columns=_COMPONENTS)
        norm = raw.apply(_minmax)
        norm["Index"] = norm.mean(axis=1) * 100.0
        return norm

    @render.text
    def p_latest():
        c = components()
        return "—" if c.empty else f"{c['Index'].iloc[-1]:.0f}"

    @render.text
    def p_change():
        c = components()
        if c.empty:
            return "—"
        return f"{c['Index'].iloc[-1] - c['Index'].iloc[0]:+.0f}"

    @render.text
    def p_driver():
        c = components()
        if c.empty:
            return "—"
        return c[_COMPONENTS].iloc[-1].idxmax()

    @render_widget
    def p_index():
        c = components()
        if c.empty:
            return theme.empty_figure("Need at least two years of data")
        fig = go.Figure(go.Scatter(x=c.index, y=c["Index"], mode="lines+markers",
                                   line=dict(color=theme.ACCENT, width=3), fill="tozeroy"))
        fig.update_yaxes(title_text="Pressure index", range=[0, 100])
        return theme.style(fig, height=360)

    @render_widget
    def p_components():
        c = components()
        if c.empty:
            return theme.empty_figure("Need at least two years of data")
        fig = go.Figure()
        for i, comp in enumerate(_COMPONENTS):
            fig.add_trace(go.Scatter(x=c.index, y=c[comp], mode="lines", name=comp,
                                     line=dict(color=theme.PALETTE[i], width=2)))
        fig.update_yaxes(title_text="Normalised (0–1)", range=[0, 1])
        return theme.style(fig, height=360)

    @render_widget
    def p_radar():
        c = components()
        if c.empty:
            return theme.empty_figure("Need at least two years of data")
        latest = c[_COMPONENTS].iloc[-1]
        fig = go.Figure(go.Scatterpolar(
            r=list(latest.values) + [latest.values[0]],
            theta=_COMPONENTS + [_COMPONENTS[0]],
            fill="toself", line=dict(color=theme.ACCENT)))
        fig.update_layout(template="g6", height=340,
                          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          showlegend=False)
        return fig
