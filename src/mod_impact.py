"""Impact tab using exact volume counts plus sample-based citation distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import theme

_THRESHOLD_CHOICES = {
    "5": ">= 5 citations",
    "10": ">= 10 citations",
    "50": ">= 50 citations",
    "100": ">= 100 citations",
    "250": ">= 250 citations",
}

_COHORTS = [
    ("2000-2009", 2000, 2009),
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2025", 2020, 2025),
]


def impact_ui():
    return ui.nav_panel(
        "Impact",
        ui.layout_columns(
            ui.value_box("Mean citations", ui.output_text("i_mean"),
                         ui.span("sample-weighted estimate", class_="text-muted small")),
            ui.value_box("High-impact share", ui.output_text("i_highshare"),
                         ui.span("above chosen threshold", class_="text-muted small")),
            ui.value_box("Mean citations/year", ui.output_text("i_velocity"),
                         ui.span("sample-weighted estimate", class_="text-muted small")),
            ui.value_box("Mean references", ui.output_text("i_refs"),
                         ui.span("sample-weighted estimate", class_="text-muted small")),
            col_widths=[3, 3, 3, 3], fill=False,
        ),
        ui.input_select("imp_threshold", "High-impact threshold", _THRESHOLD_CHOICES,
                        selected="100", width="300px"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Paper count vs citation impact"),
                output_widget("i_dual"),
                ui.card_footer(ui.span(
                    "Exact yearly paper counts on the left axis, sample-weighted mean "
                    "citations on the right axis.", class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("High-impact share over time"),
                output_widget("i_hightrend"),
                ui.card_footer(ui.span(
                    "Sample-weighted share of papers above the selected citation threshold.",
                    class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Citation distribution by cohort"),
                output_widget("i_hist"),
                ui.card_footer(ui.span(
                    "Weighted histogram from the stratified sample; x-axis is log10(citations + 1).",
                    class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Reference intensity trend"),
                output_widget("i_reftrend"),
                ui.card_footer(ui.span(
                    "Sample-weighted average number of references per paper by year.",
                    class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Sampled paper explorer"),
            ui.input_text("imp_search", None, placeholder="Search titles...", width="320px"),
            ui.output_data_frame("i_table"),
            ui.card_footer(ui.span(
                "Explorer uses the saved stratified sample, not the full corpus.",
                class_="text-muted small")),
        ),
    )


def _weights(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([])
    w = np.array(
        pd.to_numeric(df.get("sample_weight"), errors="coerce").fillna(1.0),
        dtype=float,
        copy=True,
    )
    w[w <= 0] = 1.0
    return w


def _weighted_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return float("nan")
    values = np.array(pd.to_numeric(df[column], errors="coerce"), dtype=float, copy=True)
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    w = _weights(df)[mask]
    values = values[mask]
    return float(np.average(values, weights=w))


def _weighted_share(df: pd.DataFrame, mask: pd.Series) -> float:
    if df.empty:
        return float("nan")
    w = _weights(df)
    m = mask.to_numpy(dtype=bool)
    if w.size == 0 or w.sum() == 0:
        return float("nan")
    return float(w[m].sum() / w.sum())


def _exact_year_counts(snapshot: dict, lo: int, hi: int) -> pd.Series:
    df = snapshot["year_counts"].copy()
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df[(df["publication_year"] >= lo) & (df["publication_year"] <= hi)]
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("publication_year")["paper_count"].sort_index().astype(float)


def impact_server(input, output, session, snapshot, sampled_filtered, year_range):

    @reactive.calc
    def threshold() -> float:
        return float(input.imp_threshold())

    @render.text
    def i_mean():
        df = sampled_filtered()
        val = _weighted_mean(df, "citation_count")
        return "—" if np.isnan(val) else f"{val:,.0f}"

    @render.text
    def i_highshare():
        df = sampled_filtered()
        if df.empty:
            return "—"
        share = _weighted_share(df, pd.to_numeric(df["citation_count"], errors="coerce") >= threshold())
        return "—" if np.isnan(share) else f"{share * 100:.0f}%"

    @render.text
    def i_velocity():
        df = sampled_filtered()
        val = _weighted_mean(df, "citations_per_year")
        return "—" if np.isnan(val) else f"{val:.1f}"

    @render.text
    def i_refs():
        df = sampled_filtered()
        val = _weighted_mean(df, "referenced_works_count")
        return "—" if np.isnan(val) else f"{val:,.0f}"

    @render_widget
    def i_dual():
        lo, hi = year_range()
        counts = _exact_year_counts(snapshot, lo, hi)
        df = sampled_filtered()
        if counts.empty or df.empty:
            return theme.empty_figure()
        yearly = (
            df.groupby("year")
            .apply(lambda sub: _weighted_mean(sub, "citation_count"))
            .dropna()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=counts.index, y=counts.values, name="Papers",
            line=dict(color=theme.ACCENT, width=3),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=yearly.index, y=yearly.values, name="Mean citations",
            line=dict(color=theme.PALETTE[3], width=2, dash="dash"),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Papers", secondary_y=False)
        fig.update_yaxes(title_text="Mean citations", secondary_y=True, showgrid=False)
        return theme.style(fig, height=360)

    @render_widget
    def i_hightrend():
        df = sampled_filtered()
        if df.empty:
            return theme.empty_figure()
        yearly = (
            df.groupby("year")
            .apply(lambda sub: _weighted_share(
                sub, pd.to_numeric(sub["citation_count"], errors="coerce") >= threshold()
            ) * 100.0)
            .dropna()
        )
        fig = go.Figure(go.Scatter(
            x=yearly.index, y=yearly.values, mode="lines",
            fill="tozeroy", line=dict(color=theme.ACCENT, width=2),
        ))
        fig.update_yaxes(title_text="High-impact share (%)", range=[0, 100])
        return theme.style(fig, height=360)

    @render_widget
    def i_hist():
        df = sampled_filtered()
        if df.empty:
            return theme.empty_figure()
        fig = go.Figure()
        bins = np.linspace(0, 5, 31)
        for i, (label, lo, hi) in enumerate(_COHORTS):
            sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
            if sub.empty:
                continue
            values = np.log10(pd.to_numeric(sub["citation_count"], errors="coerce").clip(lower=0) + 1.0)
            weights = _weights(sub)
            hist, edges = np.histogram(values, bins=bins, weights=weights)
            centers = (edges[:-1] + edges[1:]) / 2
            share = hist / hist.sum() if hist.sum() > 0 else hist
            fig.add_trace(go.Bar(
                x=centers, y=share, name=label, opacity=0.65,
                marker_color=theme.PALETTE[i % len(theme.PALETTE)],
            ))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="log10(citations + 1)")
        fig.update_yaxes(title_text="Weighted share")
        return theme.style(fig, height=360)

    @render_widget
    def i_reftrend():
        df = sampled_filtered()
        if df.empty:
            return theme.empty_figure()
        avg = (
            df.groupby("year")
            .apply(lambda sub: _weighted_mean(sub, "referenced_works_count"))
            .dropna()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=avg.index, y=avg.values, mode="lines+markers",
            name="Mean references", line=dict(color=theme.ACCENT, width=2),
        ))
        fig.update_yaxes(title_text="References per paper")
        return theme.style(fig, height=360)

    @render.data_frame
    def i_table():
        df = sampled_filtered()
        if df.empty:
            return render.DataGrid(pd.DataFrame({"Title": [], "Year": [], "Citations": []}))
        q = (input.imp_search() or "").strip().lower()
        if q:
            df = df[df["title"].fillna("").str.lower().str.contains(q)]
        cols = ["title", "year", "citation_count", "citations_per_year",
                "referenced_works_count", "venue_source", "sample_weight"]
        out = (
            df[cols]
            .sort_values("citation_count", ascending=False)
            .head(500)
            .rename(columns={
                "title": "Title",
                "year": "Year",
                "citation_count": "Citations",
                "citations_per_year": "Citations/yr",
                "referenced_works_count": "References",
                "venue_source": "Venue",
                "sample_weight": "Weight",
            })
        )
        out["Citations/yr"] = pd.to_numeric(out["Citations/yr"], errors="coerce").round(1)
        out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").round(1)
        return render.DataGrid(out, height="360px", summary=True)
