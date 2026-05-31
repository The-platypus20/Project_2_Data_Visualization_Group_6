"""Concentration tab: where AI research clusters by geography and topic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import data as datamod
from . import metrics, theme

_MAP_METRICS = {
    "raw": "Raw paper count",
    "per_million": "Papers per million people",
    "impact": "Citation impact per paper",
    "cagr": "Growth rate (CAGR)",
}


def concentration_ui():
    return ui.nav_panel(
        "Concentration",
        ui.layout_columns(
            ui.value_box("Top 5 country share", ui.output_text("c_top5"),
                         ui.span("of papers in selection", class_="text-muted small")),
            ui.value_box("Topic entropy", ui.output_text("c_entropy"),
                         ui.span("bits — higher = more spread", class_="text-muted small")),
            ui.value_box("Citation Gini", ui.output_text("c_gini"),
                         ui.span("0 equal · 1 concentrated", class_="text-muted small")),
            ui.value_box("Map metric (global)", ui.output_text("c_mapval"),
                         ui.span("current map metric value", class_="text-muted small")),
            col_widths=[3, 3, 3, 3], fill=False,
        ),
        ui.input_radio_buttons("con_metric", "Map metric", _MAP_METRICS,
                               selected="raw", inline=True),
        ui.layout_columns(
            ui.card(
                ui.card_header("World map of AI research concentration"),
                output_widget("c_map"),
                ui.card_footer(ui.span("Choropleth shaded by the selected map metric.",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Topic growth heatmap"),
                output_widget("c_heatmap"),
                ui.card_footer(ui.span("Paper count per topic per year (darker = more papers).",
                                       class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Citation concentration (Lorenz curve)"),
                output_widget("c_lorenz"),
            ),
            ui.card(
                ui.card_header("Top countries"),
                ui.output_data_frame("c_topcountries"),
            ),
            ui.card(
                ui.card_header("Country drill-down: top institutions"),
                ui.input_select("con_country", None, choices=[], width="100%"),
                ui.output_data_frame("c_drill"),
            ),
            col_widths=[4, 4, 4],
        ),
        ui.h6("Academia vs Industry", class_="mt-2"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Sector composition"),
                output_widget("c_sector_donut"),
                ui.card_footer(ui.span("Each paper labelled by the sectors of its institutions "
                                       "(rule-based heuristic — see src/sector.py).",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Academia–Industry collaboration over time"),
                output_widget("c_sector_trend"),
                ui.card_footer(ui.span("Share of each year's papers by collaboration type.",
                                       class_="text-muted small")),
            ),
            col_widths=[4, 8],
        ),
    )


def concentration_server(input, output, session, filtered):

    @reactive.calc
    def country_counts():
        ex = datamod.explode_countries(filtered())
        return ex.groupby(["iso2", "country", "iso3"]).size().rename("papers").reset_index()

    @render.text
    def c_top5():
        cc = country_counts()
        if cc.empty:
            return "—"
        return f"{metrics.top_n_share(cc['papers'].values, 5):.0f}%"

    @render.text
    def c_entropy():
        df = filtered()
        if df.empty:
            return "—"
        return f"{metrics.shannon_entropy(df['topic_bucket'].value_counts().values):.2f}"

    @render.text
    def c_gini():
        df = filtered()
        return "—" if df.empty else f"{metrics.gini(df['citation_count'].values):.2f}"

    @reactive.calc
    def map_frame():
        """Per-country dataframe with the selected map metric as column `value`."""
        ex = datamod.explode_countries(filtered())
        if ex.empty:
            return pd.DataFrame(columns=["iso3", "country", "value"])
        metric = input.con_metric()
        g = ex.groupby(["iso3", "country"])
        if metric == "raw":
            out = g.size().rename("value").reset_index()
        elif metric == "impact":
            out = g["citation_count"].mean().rename("value").reset_index()
        elif metric == "per_million":
            base = g.size().rename("papers").reset_index()
            pop = (ex.groupby("iso3")["pop_m"].first())
            base["value"] = base.apply(
                lambda r: r["papers"] / pop.get(r["iso3"], np.nan), axis=1)
            out = base[["iso3", "country", "value"]]
        else:  # cagr
            rows = []
            for (iso3, country), sub in g:
                s = sub.groupby("year").size().sort_index()
                if len(s) >= 2 and s.sum() >= 10:
                    val = metrics.cagr(s.iloc[0], s.iloc[-1], len(s) - 1)
                    rows.append((iso3, country, val * 100))
            out = pd.DataFrame(rows, columns=["iso3", "country", "value"])
        return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])

    @render.text
    def c_mapval():
        mf, df, metric = map_frame(), filtered(), input.con_metric()
        if df.empty:
            return "—"
        if metric == "raw":
            return f"{len(df):,}"
        if metric == "impact":
            return f"{df['citation_count'].mean():,.0f}"
        if metric == "per_million":
            return "—" if mf.empty else f"{mf['value'].mean():.2f}"
        s = df.groupby("year").size().sort_index()
        val = metrics.cagr(s.iloc[0], s.iloc[-1], len(s) - 1) if len(s) >= 2 else np.nan
        return "—" if np.isnan(val) else f"{val * 100:.1f}%"

    @render_widget
    def c_map():
        mf = map_frame()
        if mf.empty:
            return theme.empty_figure()
        label = _MAP_METRICS[input.con_metric()]
        fig = go.Figure(go.Choropleth(
            locations=mf["iso3"], z=mf["value"], text=mf["country"],
            colorscale=theme.SEQUENTIAL, colorbar_title=None,
            hovertemplate="%{text}<br>" + label + ": %{z:,.2f}<extra></extra>",
        ))
        fig.update_geos(showframe=False, showcoastlines=False,
                        projection_type="natural earth", bgcolor="white")
        fig.update_layout(template="g6", height=420, margin=dict(l=0, r=0, t=10, b=0))
        return fig

    @render_widget
    def c_heatmap():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        pivot = (df.groupby(["topic_bucket", "year"]).size()
                 .unstack(fill_value=0).sort_index())
        # Order topics by total volume (largest at top of the heatmap).
        pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[str(y) for y in pivot.columns], y=list(pivot.index),
            colorscale=theme.SEQUENTIAL, colorbar_title=None,
            hovertemplate="%{y}<br>%{x}: %{z} papers<extra></extra>"))
        fig.update_layout(template="g6", height=420, margin=dict(l=8, r=8, t=10, b=40))
        return fig

    @render_widget
    def c_lorenz():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        x, y = metrics.lorenz_curve(df["citation_count"].values)
        g = metrics.gini(df["citation_count"].values)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality",
                                 line=dict(color=theme.MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Lorenz",
                                 fill="tonexty", line=dict(color=theme.ACCENT, width=2)))
        fig.add_annotation(x=0.05, y=0.9, text=f"Gini = {g:.2f}", showarrow=False,
                           font=dict(size=14, color="#111827"), xref="paper", yref="paper")
        fig.update_xaxes(title_text="Cumulative share of papers", range=[0, 1])
        fig.update_yaxes(title_text="Cumulative share of citations", range=[0, 1])
        return theme.style(fig, height=340)

    @render.data_frame
    def c_topcountries():
        cc = country_counts()
        if cc.empty:
            return render.DataGrid(pd.DataFrame({"Country": [], "Papers": [], "Share %": []}))
        total = cc["papers"].sum()
        top = cc.sort_values("papers", ascending=False).head(15).copy()
        top["Share %"] = (top["papers"] / total * 100).round(1)
        top.insert(0, "Rank", range(1, len(top) + 1))
        out = top[["Rank", "country", "papers", "Share %"]].rename(
            columns={"country": "Country", "papers": "Papers"})
        return render.DataGrid(out, height="320px")

    # Keep the drill-down country selector populated with the current top countries.
    @reactive.effect
    def _sync_country_choices():
        cc = country_counts()
        choices = cc.sort_values("papers", ascending=False)["country"].head(25).tolist()
        ui.update_select("con_country", choices=choices,
                         selected=(choices[0] if choices else None))

    @render.data_frame
    def c_drill():
        country = input.con_country()
        if not country:
            return render.DataGrid(pd.DataFrame({"Institution": [], "Papers": []}))
        ex = datamod.explode_countries(filtered())
        ex = ex[ex["country"] == country]
        inst = datamod.explode_institutions(ex)
        if inst.empty:
            return render.DataGrid(pd.DataFrame({"Institution": [], "Papers": []}))
        agg = (inst.groupby("institution")
               .agg(Papers=("paper_id", "size"), Citations=("citation_count", "sum"))
               .reset_index())
        agg["Cites/Paper"] = (agg["Citations"] / agg["Papers"]).round(0)
        total = agg["Papers"].sum()
        agg["Share %"] = (agg["Papers"] / total * 100).round(1)
        out = (agg.sort_values("Papers", ascending=False).head(15)
               .rename(columns={"institution": "Institution"}))
        return render.DataGrid(out, height="300px")

    _SECTOR_COLORS = {"Academia": theme.PALETTE[0], "Industry": theme.PALETTE[3],
                      "Academia–Industry": theme.PALETTE[2], "Other / Mixed": theme.MUTED}

    @render_widget
    def c_sector_donut():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        mix = df["sector"].value_counts()
        fig = go.Figure(go.Pie(
            labels=mix.index, values=mix.values, hole=0.5,
            marker=dict(colors=[_SECTOR_COLORS.get(s, theme.MUTED) for s in mix.index])))
        fig.update_traces(textinfo="percent")
        return theme.style(fig, height=360)

    @render_widget
    def c_sector_trend():
        df = filtered()
        if df.empty:
            return theme.empty_figure()
        counts = df.groupby(["year", "sector"]).size()
        share = (counts / counts.groupby(level=0).transform("sum") * 100
                 ).unstack(fill_value=0).sort_index()
        fig = go.Figure()
        for s in ["Academia", "Academia–Industry", "Industry", "Other / Mixed"]:
            if s in share.columns:
                fig.add_trace(go.Scatter(x=share.index, y=share[s], name=s, mode="lines",
                                         stackgroup="one",
                                         line=dict(width=0.5, color=_SECTOR_COLORS[s])))
        fig.update_yaxes(title_text="Share of papers (%)", range=[0, 100])
        return theme.style(fig, height=360)
