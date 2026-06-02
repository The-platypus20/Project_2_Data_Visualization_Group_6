"""Concentration tab backed primarily by exact grouped statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import geo
from . import metrics, theme

_MAP_METRICS = {
    "raw": "Raw paper count",
    "per_million": "Papers per million people",
    "cagr": "Growth rate (CAGR)",
}


def concentration_ui():
    return ui.nav_panel(
        "Concentration",
        ui.layout_columns(
            ui.value_box("Top 5 country share", ui.output_text("c_top5"),
                         ui.span("of exact paper counts", class_="text-muted small")),
            ui.value_box("Topic entropy", ui.output_text("c_entropy"),
                         ui.span("exact topic-bucket spread", class_="text-muted small")),
            ui.value_box("Top 10 institution share", ui.output_text("c_instshare"),
                         ui.span("within saved top-institution table", class_="text-muted small")),
            ui.value_box("Map metric", ui.output_text("c_mapval"),
                         ui.span("global summary for current map view", class_="text-muted small")),
            col_widths=[3, 3, 3, 3], fill=False,
        ),
        ui.input_radio_buttons("con_metric", "Map metric", _MAP_METRICS,
                               selected="raw", inline=True),
        ui.layout_columns(
            ui.card(
                ui.card_header("World map of AI research concentration"),
                output_widget("c_map"),
                ui.card_footer(ui.span(
                    "Map uses exact grouped OpenAlex counts from the snapshot.",
                    class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Topic growth heatmap"),
                output_widget("c_heatmap"),
                ui.card_footer(ui.span(
                    "Paper count per topic bucket per year from exact grouped stats.",
                    class_="text-muted small")),
            ),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Country concentration curve"),
                output_widget("c_curve"),
                ui.card_footer(ui.span(
                    "Cumulative paper share after sorting countries by output.",
                    class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Top countries"),
                ui.output_data_frame("c_topcountries"),
            ),
            ui.card(
                ui.card_header("Country drill-down: top institutions"),
                ui.input_select("con_country", None, choices=[], width="100%"),
                ui.output_data_frame("c_drill"),
                ui.card_footer(ui.span(
                    "Institution table is an all-period exact snapshot for each top country.",
                    class_="text-muted small")),
            ),
            col_widths=[4, 4, 4],
        ),
        ui.h6("Research Structure", class_="mt-2"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Open-access composition"),
                output_widget("c_oa_donut"),
                ui.card_footer(ui.span("Exact OA-status counts from OpenAlex.",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("OA status over time"),
                output_widget("c_oa_trend"),
                ui.card_footer(ui.span("Exact yearly OA-status mix.",
                                       class_="text-muted small")),
            ),
            col_widths=[4, 8],
        ),
    )


def _year_slice(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    return out[(out["publication_year"] >= lo) & (out["publication_year"] <= hi)].copy()


def concentration_server(input, output, session, snapshot, sampled_filtered, year_range):

    @reactive.calc
    def country_counts():
        lo, hi = year_range()
        df = _year_slice(snapshot["country_year_counts"], lo, hi)
        if df.empty:
            return pd.DataFrame(columns=["country_code", "country_name", "paper_count", "region"])
        out = (
            df.groupby(["country_code", "country_name", "region"], as_index=False)["paper_count"]
            .sum()
            .sort_values("paper_count", ascending=False)
        )
        return out

    @render.text
    def c_top5():
        cc = country_counts()
        if cc.empty:
            return "—"
        return f"{metrics.top_n_share(cc['paper_count'].values, 5):.0f}%"

    @render.text
    def c_entropy():
        lo, hi = year_range()
        df = _year_slice(snapshot["topic_bucket_year_counts"], lo, hi)
        if df.empty:
            return "—"
        counts = df.groupby("topic_bucket")["paper_count"].sum().values
        return f"{metrics.shannon_entropy(counts):.2f}"

    @render.text
    def c_instshare():
        df = snapshot["top_institutions_by_country"]
        if df.empty:
            sample = sampled_filtered()
            if sample.empty:
                return "—"
            rows = []
            for _, row in sample[["country_list", "institution_list"]].iterrows():
                for code in row["country_list"]:
                    for inst in row["institution_list"]:
                        rows.append((code, inst))
            if not rows:
                return "—"
            df = pd.DataFrame(rows, columns=["country_code", "institution_name"])
            df = df.groupby(["country_code", "institution_name"]).size().reset_index(name="paper_count")
        total = df["paper_count"].sum()
        top10 = df.sort_values("paper_count", ascending=False).head(10)["paper_count"].sum()
        return "—" if total == 0 else f"{top10 / total * 100:.0f}%"

    @reactive.calc
    def map_frame():
        cc = country_counts()
        if cc.empty:
            return pd.DataFrame(columns=["iso3", "country_name", "value"])
        metric = input.con_metric()
        out = cc.copy()
        out["iso3"] = out["country_code"].map(geo.iso3)
        out = out[out["iso3"].notna()].copy()
        if metric == "raw":
            out["value"] = out["paper_count"]
        elif metric == "per_million":
            out["pop_m"] = out["country_code"].map(geo.population_m)
            out["value"] = out.apply(
                lambda row: row["paper_count"] / row["pop_m"] if row["pop_m"] else np.nan,
                axis=1,
            )
        else:
            lo, hi = year_range()
            rows = []
            yearly = _year_slice(snapshot["country_year_counts"], lo, hi)
            for (code, name), sub in yearly.groupby(["country_code", "country_name"]):
                s = sub.groupby("publication_year")["paper_count"].sum().sort_index()
                if len(s) < 2 or s.iloc[0] <= 0:
                    continue
                val = metrics.cagr(s.iloc[0], s.iloc[-1], len(s) - 1)
                if np.isnan(val):
                    continue
                rows.append({
                    "country_code": code,
                    "country_name": name,
                    "iso3": geo.iso3(code),
                    "value": val * 100.0,
                })
            out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["iso3", "country_name", "value"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=["value", "iso3", "country_name"]).copy()
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out[np.isfinite(out["value"])].copy()
        return out

    @render.text
    def c_mapval():
        mf = map_frame()
        cc = country_counts()
        metric = input.con_metric()
        if cc.empty:
            return "—"
        if metric == "raw":
            return f"{int(cc['paper_count'].sum()):,}"
        if metric == "per_million":
            return "—" if mf.empty else f"{mf['value'].mean():.2f}"
        return "—" if mf.empty else f"{mf['value'].mean():.1f}%"

    @render_widget
    def c_map():
        mf = map_frame()
        if mf.empty:
            return theme.empty_figure()
        label = _MAP_METRICS[input.con_metric()]
        fig = go.Figure(go.Choropleth(
            locations=mf["iso3"], z=mf["value"], text=mf["country_name"],
            colorscale=theme.SEQUENTIAL, colorbar_title=None,
            hovertemplate="%{text}<br>" + label + ": %{z:,.2f}<extra></extra>",
        ))
        fig.update_geos(showframe=False, showcoastlines=False,
                        projection_type="natural earth", bgcolor="white")
        fig.update_layout(template="g6", height=420, margin=dict(l=0, r=0, t=10, b=0))
        return fig

    @render_widget
    def c_heatmap():
        lo, hi = year_range()
        df = _year_slice(snapshot["topic_bucket_year_counts"], lo, hi)
        df = df[df["topic_bucket"] != "Other"].copy()
        if df.empty:
            return theme.empty_figure()
        pivot = (
            df.groupby(["topic_bucket", "publication_year"])["paper_count"]
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[str(y) for y in pivot.columns], y=list(pivot.index),
            colorscale=theme.SEQUENTIAL, colorbar_title=None,
            hovertemplate="%{y}<br>%{x}: %{z} papers<extra></extra>",
        ))
        fig.update_layout(template="g6", height=420, margin=dict(l=8, r=8, t=10, b=40))
        return fig

    @render_widget
    def c_curve():
        cc = country_counts()
        if cc.empty:
            return theme.empty_figure()
        shares = cc["paper_count"].sort_values(ascending=False).to_numpy(dtype=float)
        shares = shares / shares.sum()
        cum = np.cumsum(shares)
        x = np.linspace(0, 1, len(cum) + 1)
        y = np.insert(cum, 0, 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name="Country concentration",
            fill="tozeroy", line=dict(color=theme.ACCENT, width=2),
        ))
        fig.update_xaxes(title_text="Cumulative share of countries", range=[0, 1])
        fig.update_yaxes(title_text="Cumulative share of papers", range=[0, 1])
        return theme.style(fig, height=340)

    @render.data_frame
    def c_topcountries():
        cc = country_counts()
        if cc.empty:
            return render.DataGrid(pd.DataFrame({"Country": [], "Papers": [], "Share %": []}))
        total = cc["paper_count"].sum()
        top = cc.head(15).copy()
        top["Share %"] = (top["paper_count"] / total * 100).round(1)
        top.insert(0, "Rank", range(1, len(top) + 1))
        out = top[["Rank", "country_name", "paper_count", "Share %"]].rename(
            columns={"country_name": "Country", "paper_count": "Papers"}
        )
        return render.DataGrid(out, height="320px")

    @reactive.effect
    def _sync_country_choices():
        cc = country_counts()
        choices = cc["country_name"].head(25).tolist()
        ui.update_select("con_country", choices=choices,
                         selected=(choices[0] if choices else None))

    @render.data_frame
    def c_drill():
        country = input.con_country()
        if not country:
            return render.DataGrid(pd.DataFrame({"Institution": [], "Papers": []}))
        df = snapshot["top_institutions_by_country"]
        df = df[df["country_name"] == country].copy()
        if df.empty:
            sample = sampled_filtered()
            rows = []
            for _, row in sample[["country_list", "institution_list"]].iterrows():
                if country not in [geo.name(code) for code in row["country_list"]]:
                    continue
                for inst in row["institution_list"]:
                    rows.append({"institution_name": inst})
            if rows:
                df = pd.DataFrame(rows).groupby("institution_name").size().reset_index(name="paper_count")
                df["country_name"] = country
        if df.empty:
            return render.DataGrid(pd.DataFrame({"Institution": [], "Papers": []}))
        total = df["paper_count"].sum()
        df["Share %"] = (df["paper_count"] / total * 100).round(1)
        df = df.sort_values("paper_count", ascending=False).head(15)
        out = df[["institution_name", "paper_count", "Share %"]].rename(
            columns={"institution_name": "Institution", "paper_count": "Papers"}
        )
        return render.DataGrid(out, height="300px")

    @render_widget
    def c_oa_donut():
        lo, hi = year_range()
        df = _year_slice(snapshot["oa_year_counts"], lo, hi)
        if df.empty:
            return theme.empty_figure()
        mix = (
            df.groupby("oa_status_label")["paper_count"]
            .sum()
            .sort_values(ascending=False)
        )
        fig = go.Figure(go.Pie(labels=mix.index, values=mix.values, hole=0.5))
        fig.update_traces(textinfo="percent")
        return theme.style(fig, height=360)

    @render_widget
    def c_oa_trend():
        lo, hi = year_range()
        df = _year_slice(snapshot["oa_year_counts"], lo, hi)
        if df.empty:
            return theme.empty_figure()
        counts = df.groupby(["publication_year", "oa_status_label"])["paper_count"].sum()
        share = (counts / counts.groupby(level=0).transform("sum") * 100).unstack(fill_value=0).sort_index()
        fig = go.Figure()
        for i, status in enumerate(share.columns):
            fig.add_trace(go.Scatter(
                x=share.index, y=share[status], name=status, mode="lines", stackgroup="one",
                line=dict(width=0.5, color=theme.PALETTE[i % len(theme.PALETTE)]),
            ))
        fig.update_yaxes(title_text="Share of papers (%)", range=[0, 100])
        return theme.style(fig, height=360)
