"""Pressure Index tab using snapshot-based exact and sample-weighted components."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import metrics, theme

_COMPONENTS = [
    "Volume growth",
    "High-impact scarcity",
    "Geographic concentration",
    "Topic crowding",
]


def pressure_ui():
    return ui.nav_panel(
        "Pressure Index",
        ui.layout_columns(
            ui.value_box("Pressure index (latest)", ui.output_text("p_latest"),
                         ui.span("0-100, relative to selected period", class_="text-muted small")),
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
                    "Average of four normalized components; three are exact grouped metrics, "
                    "one comes from the stratified sample.", class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Pressure components over time"),
                output_widget("p_components"),
                ui.card_footer(ui.span("Each component is min-max normalized within the selected years.",
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
                    "to 0-1 across the selected years and then averaged:\n\n"
                    "- **Volume growth**: year-over-year growth in publications.\n"
                    "- **High-impact scarcity**: inverse of the sample-weighted share of papers "
                    "with at least 100 citations.\n"
                    "- **Geographic concentration**: top-5 country share of output.\n"
                    "- **Topic crowding**: inverse of topic entropy.\n\n"
                    "The index is descriptive rather than normative; it shows where output is "
                    "growing fast while remaining concentrated in countries/topics and without "
                    "a matching broad share of high-impact work."
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


def _weights(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([])
    w = pd.to_numeric(df.get("sample_weight"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    w[w <= 0] = 1.0
    return w


def _weighted_share(df: pd.DataFrame, mask: pd.Series) -> float:
    if df.empty:
        return float("nan")
    w = _weights(df)
    m = mask.to_numpy(dtype=bool)
    if w.sum() == 0:
        return float("nan")
    return float(w[m].sum() / w.sum())


def _year_slice(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    return out[(out["publication_year"] >= lo) & (out["publication_year"] <= hi)].copy()


def pressure_server(input, output, session, snapshot, sampled_filtered, year_range):

    @reactive.calc
    def components() -> pd.DataFrame:
        lo, hi = year_range()
        year_counts = _year_slice(snapshot["year_counts"], lo, hi)
        topic_counts = _year_slice(snapshot["topic_bucket_year_counts"], lo, hi)
        country_counts = _year_slice(snapshot["country_year_counts"], lo, hi)
        sample = sampled_filtered()

        if year_counts.empty or year_counts["publication_year"].nunique() < 2:
            return pd.DataFrame(columns=_COMPONENTS + ["Index"])

        years = sorted(year_counts["publication_year"].astype(int).unique())
        counts = year_counts.set_index("publication_year")["paper_count"].sort_index().astype(float)
        growth = metrics.yoy_growth(counts).reindex(years)

        rows = {}
        for year in years:
            sample_year = sample[sample["year"] == year]
            high_share = _weighted_share(
                sample_year,
                pd.to_numeric(sample_year["citation_count"], errors="coerce") >= 100,
            )
            scarcity = 1.0 - high_share if np.isfinite(high_share) else np.nan

            countries = country_counts[country_counts["publication_year"] == year]
            concn = (
                metrics.top_n_share(countries["paper_count"].values, 5) / 100.0
                if not countries.empty else np.nan
            )

            topics = topic_counts[topic_counts["publication_year"] == year]
            n_topics = max(2, int(topics["topic_bucket"].nunique()))
            ent = metrics.shannon_entropy(topics["paper_count"].values)
            crowding = 1.0 - ent / np.log2(n_topics)

            rows[year] = [growth.get(year, np.nan), scarcity, concn, crowding]

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
            return theme.empty_figure("Need at least two years of snapshot data")
        fig = go.Figure(go.Scatter(
            x=c.index, y=c["Index"], mode="lines+markers",
            line=dict(color=theme.ACCENT, width=3), fill="tozeroy",
        ))
        fig.update_yaxes(title_text="Pressure index", range=[0, 100])
        return theme.style(fig, height=360)

    @render_widget
    def p_components():
        c = components()
        if c.empty:
            return theme.empty_figure("Need at least two years of snapshot data")
        fig = go.Figure()
        for i, comp in enumerate(_COMPONENTS):
            fig.add_trace(go.Scatter(
                x=c.index, y=c[comp], mode="lines", name=comp,
                line=dict(color=theme.PALETTE[i], width=2),
            ))
        fig.update_yaxes(title_text="Normalized (0-1)", range=[0, 1])
        return theme.style(fig, height=360)

    @render_widget
    def p_radar():
        c = components()
        if c.empty:
            return theme.empty_figure("Need at least two years of snapshot data")
        latest = c[_COMPONENTS].iloc[-1]
        fig = go.Figure(go.Scatterpolar(
            r=list(latest.values) + [latest.values[0]],
            theta=_COMPONENTS + [_COMPONENTS[0]],
            fill="toself", line=dict(color=theme.ACCENT),
        ))
        fig.update_layout(
            template="g6",
            height=340,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
        )
        return fig
