"""Impact tab: does citation impact keep up as volume grows?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import theme

_THRESHOLD_CHOICES = {
    "median": "Below median of selection",
    "300": "< 300 citations",
    "500": "< 500 citations",
    "1000": "< 1,000 citations",
    "2000": "< 2,000 citations",
}

_COHORTS = [("2000–2009", 2000, 2009), ("2010–2014", 2010, 2014),
            ("2015–2019", 2015, 2019), ("2020–2026", 2020, 2026)]


def impact_ui():
    return ui.nav_panel(
        "Impact",
        ui.layout_columns(
            ui.value_box("Median citations", ui.output_text("i_median"),
                         ui.span("typical paper in selection", class_="text-muted small")),
            ui.value_box("Low-citation share", ui.output_text("i_lowshare"),
                         ui.span("below the chosen threshold", class_="text-muted small")),
            ui.value_box("Top 1% citation share", ui.output_text("i_top1"),
                         ui.span("citations held by top 1% of papers", class_="text-muted small")),
            ui.value_box("Novelty proxy", ui.output_text("i_novelty"),
                         ui.span("mean structural novelty (0–1)", class_="text-muted small")),
            col_widths=[3, 3, 3, 3], fill=False,
        ),
        ui.input_select("imp_threshold", "Citation threshold (low-citation definition)",
                        _THRESHOLD_CHOICES, selected="median", width="320px"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Paper count vs citation impact"),
                output_widget("i_dual"),
                ui.card_footer(ui.span(
                    "Solid: papers per year. Dashed: median citations per year (right axis). "
                    "Diverging lines suggest impact dilution.", class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Low-citation share over time"),
                output_widget("i_lowtrend"),
                ui.card_footer(ui.span(
                    "Share of each year's papers falling below the citation threshold.",
                    class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Citation distribution by cohort"),
                output_widget("i_hist"),
                ui.card_footer(ui.span("Citations on a log scale, normalised within each cohort.",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Novelty proxy trend"),
                output_widget("i_noveltytrend"),
                ui.card_footer(ui.span("Annual average with linear trend. Lower = builds more on prior work.",
                                       class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Paper Explorer"),
            ui.input_text("imp_search", None, placeholder="Search titles…", width="320px"),
            ui.output_data_frame("i_table"),
            ui.card_footer(ui.span("Top 500 papers by citations for the current filters.",
                                   class_="text-muted small")),
        ),
    )


def impact_server(input, output, session, filtered):

    @reactive.calc
    def threshold():
        df = filtered()
        sel = input.imp_threshold()
        if sel == "median":
            return float(df["citation_count"].median()) if not df.empty else 0.0
        return float(sel)

    @render.text
    def i_median():
        df = filtered()
        return "—" if df.empty else f"{df['citation_count'].median():,.0f}"

    @render.text
    def i_lowshare():
        df = filtered()
        if df.empty:
            return "—"
        return f"{(df['citation_count'] < threshold()).mean() * 100:.0f}%"

    @render.text
    def i_top1():
        df = filtered()
        if df.empty:
            return "—"
        c = np.sort(df["citation_count"].values)[::-1]
        n = max(1, int(np.ceil(len(c) * 0.01)))
        total = c.sum()
        return "—" if total == 0 else f"{c[:n].sum() / total * 100:.1f}%"

    @render.text
    def i_novelty():
        df = filtered()
        return "—" if df.empty else f"{df['novelty_proxy'].mean():.2f}"

    @render_widget
    def i_dual():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        counts = df.groupby("year").size().sort_index()
        med = df.groupby("year")["citation_count"].median().sort_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=counts.index, y=counts.values, name="Papers",
                                 line=dict(color=theme.ACCENT, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=med.index, y=med.values, name="Median citations",
                                 line=dict(color=theme.PALETTE[3], width=2, dash="dash")),
                      secondary_y=True)
        fig.update_yaxes(title_text="Papers", secondary_y=False)
        fig.update_yaxes(title_text="Median citations", secondary_y=True, showgrid=False)
        return theme.style(fig, height=360)

    @render_widget
    def i_lowtrend():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        thr = threshold()
        share = (df.assign(low=df["citation_count"] < thr)
                 .groupby("year")["low"].mean().sort_index() * 100)
        fig = go.Figure(go.Scatter(x=share.index, y=share.values, mode="lines",
                                   fill="tozeroy", line=dict(color=theme.ACCENT, width=2)))
        fig.update_yaxes(title_text="Low-citation share (%)", range=[0, 100])
        return theme.style(fig, height=360)

    @render_widget
    def i_hist():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        fig = go.Figure()
        for i, (label, lo, hi) in enumerate(_COHORTS):
            sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
            if sub.empty:
                continue
            vals = np.log10(sub["citation_count"].clip(lower=1))
            fig.add_trace(go.Histogram(x=vals, name=label, histnorm="probability",
                                       opacity=0.6, nbinsx=30,
                                       marker_color=theme.PALETTE[i]))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Citations (log₁₀)")
        fig.update_yaxes(title_text="Share of cohort")
        return theme.style(fig, height=360)

    @render_widget
    def i_noveltytrend():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        avg = df.groupby("year")["novelty_proxy"].mean().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avg.index, y=avg.values, mode="markers",
                                 name="Annual average", marker=dict(color=theme.ACCENT, size=8)))
        if len(avg) >= 2:
            coef = np.polyfit(avg.index, avg.values, 1)
            trend = np.poly1d(coef)(avg.index)
            fig.add_trace(go.Scatter(x=avg.index, y=trend, mode="lines", name="Trend",
                                     line=dict(color=theme.MUTED, dash="dash")))
        fig.update_yaxes(title_text="Novelty proxy", range=[0, 1])
        return theme.style(fig, height=360)

    @render.data_frame
    def i_table():
        df = filtered()
        q = (input.imp_search() or "").strip().lower()
        if q:
            df = df[df["title"].fillna("").str.lower().str.contains(q)]
        cols = ["title", "year", "citation_count", "citations_per_year",
                "novelty_proxy", "venue_source"]
        out = (df[cols].sort_values("citation_count", ascending=False)
               .head(500)
               .rename(columns={"title": "Title", "year": "Year",
                                "citation_count": "Citations",
                                "citations_per_year": "Citations/yr",
                                "novelty_proxy": "Novelty",
                                "venue_source": "Venue"}))
        out["Citations/yr"] = out["Citations/yr"].round(1)
        return render.DataGrid(out, height="360px", summary=True)
