"""Growth tab: how the volume of (highly-cited) AI research expands over time."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import render, ui
from shinywidgets import output_widget, render_widget

from . import metrics, theme

_INFO = ("Volume reflects the sample of highly-cited AI works in the dataset, "
         "not all AI output.")


def growth_ui():
    return ui.nav_panel(
        "Growth",
        ui.layout_columns(
            ui.value_box("Total AI papers", ui.output_text("g_total"),
                         ui.span("in current selection", class_="text-muted small")),
            ui.value_box("Growth since start", ui.output_text("g_growth"),
                         ui.span("first vs last year in range", class_="text-muted small")),
            ui.value_box("Avg annual growth (CAGR)", ui.output_text("g_cagr"),
                         ui.span("compound annual growth rate", class_="text-muted small")),
            ui.value_box("Top growth topic", ui.output_text("g_toptopic"),
                         ui.span("highest CAGR in range", class_="text-muted small")),
            col_widths=[3, 3, 3, 3], fill=False,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("AI papers published per year"),
                output_widget("g_line"),
                ui.card_footer(ui.span(
                    "Solid line: papers per year. Dashed line: year-over-year growth (%). "
                    + _INFO, class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Year-over-year acceleration"),
                output_widget("g_accel"),
                ui.card_footer(ui.span(
                    "Change in YoY growth vs the previous year (percentage points). "
                    "Positive = accelerating, negative = decelerating.",
                    class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Topic contribution to publication growth"),
            output_widget("g_area"),
            ui.card_footer(ui.span(
                "Each band is a canonical AI topic; band height is papers per year. "
                "Shows which topics drive the overall expansion.",
                class_="text-muted small")),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Top research subfields"),
                output_widget("g_subfields"),
                ui.card_footer(ui.span("Most common OpenAlex primary subfields in the selection.",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Publication-type mix"),
                output_widget("g_venuemix"),
                ui.card_footer(ui.span("Share of works by venue type.", class_="text-muted small")),
            ),
            col_widths=[8, 4],
        ),
    )


def growth_server(input, output, session, filtered):

    @render.text
    def g_total():
        return f"{len(filtered()):,}"

    @render.text
    def g_growth():
        s = metrics_series(filtered())
        if len(s) < 2 or s.iloc[0] == 0:
            return "—"
        return f"{(s.iloc[-1] / s.iloc[0] - 1) * 100:+.0f}%"

    @render.text
    def g_cagr():
        s = metrics_series(filtered())
        if len(s) < 2:
            return "—"
        val = metrics.cagr(s.iloc[0], s.iloc[-1], len(s) - 1)
        return "—" if np.isnan(val) else f"{val * 100:.1f}%"

    @render.text
    def g_toptopic():
        df = filtered()
        if df.empty:
            return "—"
        best, best_cagr = "—", -np.inf
        for topic, sub in df.groupby("topic_bucket"):
            s = sub.groupby("year").size().sort_index()
            if len(s) < 2 or s.sum() < 20:
                continue
            c = metrics.cagr(s.iloc[0], s.iloc[-1], len(s) - 1)
            if not np.isnan(c) and c > best_cagr:
                best, best_cagr = topic, c
        return best

    @render_widget
    def g_line():
        s = metrics_series(filtered())
        if s.empty:
            return theme.empty_figure()
        yoy = metrics.yoy_growth(s)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers",
                                 name="Papers", line=dict(color=theme.ACCENT, width=3)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=yoy.index, y=yoy.values, mode="lines",
                                 name="YoY growth (%)",
                                 line=dict(color=theme.MUTED, width=2, dash="dash")),
                      secondary_y=True)
        fig.update_yaxes(title_text="Papers", secondary_y=False)
        fig.update_yaxes(title_text="YoY growth (%)", secondary_y=True, showgrid=False)
        return theme.style(fig, height=360)

    @render_widget
    def g_accel():
        s = metrics_series(filtered())
        accel = metrics.yoy_acceleration(s).dropna()
        if accel.empty:
            return theme.empty_figure()
        colors = [theme.ACCENT if v >= 0 else theme.PALETTE[3] for v in accel.values]
        fig = go.Figure(go.Bar(x=accel.index, y=accel.values, marker_color=colors))
        fig.update_yaxes(title_text="Acceleration (pp)")
        return theme.style(fig, height=360)

    @render_widget
    def g_area():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        pivot = (df.groupby(["year", "topic_bucket"]).size()
                 .unstack(fill_value=0).sort_index())
        # Order bands by total volume so the biggest topics sit at the bottom.
        order = pivot.sum().sort_values(ascending=False).index
        fig = go.Figure()
        for i, topic in enumerate(order):
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[topic], name=topic, mode="lines",
                stackgroup="one", line=dict(width=0.5, color=theme.PALETTE[i % len(theme.PALETTE)]),
            ))
        fig.update_yaxes(title_text="Papers")
        return theme.style(fig, height=380)

    @render_widget
    def g_subfields():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        top = df["primary_subfield"].value_counts().head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=theme.ACCENT))
        fig.update_xaxes(title_text="Papers")
        return theme.style(fig, height=380)

    @render_widget
    def g_venuemix():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        mix = df["venue_group"].value_counts()
        fig = go.Figure(go.Pie(labels=mix.index, values=mix.values, hole=0.5))
        fig.update_traces(textinfo="percent+label", showlegend=False)
        return theme.style(fig, height=380)


def metrics_series(df):
    """Papers-per-year series for the filtered frame."""
    return df.groupby("year").size().sort_index()
