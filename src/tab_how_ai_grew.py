"""Tab 1: the 2000-2025 AI growth story."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget, render_widget

from . import narrative_data as nd
from . import theme
from .narrative_common import badge, card_header, notice, section_label

BREAKPOINT_YEARS = [2012, 2017, 2020, 2022]
BREAKPOINT_LABELS = {
    2012: "Deep learning",
    2017: "Transformer",
    2020: "Research surge",
    2022: "GenAI wave",
}
TAB1_ACCENT = "#7CC9FF"
SECONDARY_ACCENT = "#F59E0B"
HIGHLIGHT_COUNTRIES = {"China", "United States"}
INSTITUTION_TYPE_ORDER = ["University", "Business", "Public sector", "Nonprofit / Other"]


def _compact_count(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _format_delta(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):+.1f}{suffix}"


def _yearly_growth() -> pd.DataFrame:
    df = nd.yearly_counts().sort_values("year").copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "paper_count", "yoy_growth"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["paper_count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["year", "paper_count"])
    df["year"] = df["year"].astype(int)
    df["yoy_growth"] = (
        df["paper_count"].pct_change().replace([float("inf"), -float("inf")], pd.NA) * 100
    )
    return df[["year", "paper_count", "yoy_growth"]]


def _breakpoint_rows() -> pd.DataFrame:
    df = _yearly_growth()
    points = df[df["year"].isin(BREAKPOINT_YEARS)].copy()
    points["label"] = points["year"].map(BREAKPOINT_LABELS)
    return points


def _latest_entropy() -> tuple[str, str]:
    df = nd.diversity_metrics().copy()
    if df.empty or "entropy" not in df:
        return "N/A", "Topic diversity"
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    df = df.dropna(subset=["year", "entropy"]).sort_values("year")
    if df.empty:
        return "N/A", "Topic diversity"
    last = df.iloc[-1]
    first = df.iloc[0]
    delta = float(last["entropy"] - first["entropy"])
    return f"{float(last['entropy']):.2f}", f"{_format_delta(delta)} since {int(first['year'])}"


def _kpi_card(label: str, value: str, note: str, accent: str = TAB1_ACCENT) -> ui.Tag:
    return ui.div(
        ui.div(label, style="font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#64748B;"),
        ui.div(value, style="font-size:28px; line-height:1.05; font-weight:800; color:#0B172A; margin-top:8px;"),
        ui.div(note, style="font-size:12px; color:#64748B; margin-top:7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"),
        style=(
            "min-height:96px; padding:16px 18px; border-radius:18px; "
            "background:linear-gradient(180deg, rgba(255,255,255,.96), rgba(248,251,255,.96)); "
            "border:1px solid rgba(209,222,238,.95); "
            "box-shadow:0 10px 24px rgba(15,23,42,.06); "
            f"border-top:3px solid {accent};"
        ),
    )


def _closest_years(available_years: list[int], targets: list[int]) -> list[int]:
    if not available_years:
        return []
    years: list[int] = []
    for target in targets:
        closest = min(available_years, key=lambda y: abs(y - target))
        if closest not in years:
            years.append(closest)
    return years


def growth_ui():
    return ui.nav_panel(
        "How AI grew",
        ui.div(
            ui.div(
                ui.div(
                    ui.h2("How AI grew"),
                    ui.p(
                        "AI scaled fast, while topic diversity and research leadership changed with it.",
                        class_="tab-insight",
                    ),
                ),
                ui.div(badge("OpenAlex 2000-2025"), badge("Scale and structure"), class_="badge-row"),
                class_="growth-header-row",
            ),
            section_label("At a glance"),
            ui.output_ui("growth_kpi_cards"),
            section_label("Growth curve"),
            ui.card(
                card_header(
                    "AI papers grew sharply after key research waves",
                    "Annual paper count with turning points annotated on the line.",
                ),
                output_widget("paper_timeline"),
                notice("Milestone labels are historical anchors for reading the curve, not causal estimates."),
            ),
            section_label("Leadership and structure"),
            ui.layout_columns(
                ui.card(
                    card_header(
                        "Country output is concentrated at the top",
                        "Top countries by AI paper count. The two largest producers are highlighted.",
                    ),
                    output_widget("growth_top_countries"),
                    ui.output_ui("country_concentration_note"),
                ),
                ui.card(
                    card_header(
                        "University remained the institutional anchor",
                        "Papers involving each institution type over time. University is highlighted.",
                    ),
                    output_widget("institution_type_participation"),
                    ui.output_ui("institution_lead_note"),
                ),
                ui.card(
                    card_header(
                        "Topic diversity increased",
                        "Entropy measures how evenly papers spread across topic families.",
                    ),
                    output_widget("theme_diversity"),
                    notice("Higher entropy means AI papers are spread more evenly across topic families."),
                ),
                col_widths=[4, 4, 4],
            ),
            class_="growth-tab",
        ),
    )


def growth_server(input, output, session):
    @render.ui
    def growth_kpi_cards():
        yearly = _yearly_growth()
        countries = nd.top_countries(10).copy()

        if yearly.empty:
            return ui.div(
                _kpi_card("Total papers", "N/A", "No yearly cache"),
                _kpi_card("Growth multiplier", "N/A", "No yearly cache"),
                _kpi_card("Country leaders", "N/A", "No country cache"),
                _kpi_card("Topic entropy", "N/A", "No diversity cache"),
                style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin-bottom:10px;",
            )

        first = yearly.iloc[0]
        last = yearly.iloc[-1]
        total_papers = float(yearly["paper_count"].sum())
        growth_multiplier = (
            float(last["paper_count"] / first["paper_count"])
            if float(first["paper_count"]) > 0
            else np.nan
        )

        country_value = "China + US"
        country_note = "Largest output producers"
        if not countries.empty:
            countries["papers"] = pd.to_numeric(countries["papers"], errors="coerce")
            highlights = countries[countries["country"].isin(HIGHLIGHT_COUNTRIES)].copy()
            if not highlights.empty:
                highlight_total = int(highlights["papers"].sum())
                country_note = f"{highlight_total:,} papers combined"

        entropy_value, entropy_note = _latest_entropy()

        return ui.div(
            _kpi_card("Total papers", f"{_compact_count(total_papers)}", f"{int(first['year'])}-{int(last['year'])}"),
            _kpi_card(
                "Growth multiplier",
                f"{growth_multiplier:.1f}×" if pd.notna(growth_multiplier) else "N/A",
                f"{int(first['year'])} to {int(last['year'])}",
                accent="#A78BFA",
            ),
            _kpi_card("Country leaders", country_value, country_note, accent="#F59E0B"),
            _kpi_card("Topic entropy", entropy_value, entropy_note, accent="#22C55E"),
            style="display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; margin-bottom:10px;",
        )

    @render_widget
    def paper_timeline():
        df = _yearly_growth()
        if df.empty:
            return theme.empty_figure("Yearly paper count cache is empty.")

        fig = go.Figure(
            go.Scatter(
                x=df["year"],
                y=df["paper_count"],
                customdata=df["yoy_growth"],
                mode="lines+markers",
                name="Paper count",
                line=dict(color=TAB1_ACCENT, width=3),
                marker=dict(size=6, color=TAB1_ACCENT, line=dict(color="#ffffff", width=1)),
                hovertemplate=(
                    "Year %{x}<br>"
                    "Papers %{y:,.0f}<br>"
                    "YoY growth %{customdata:+.1f}%<extra></extra>"
                ),
            )
        )

        ymax = float(df["paper_count"].max()) * 1.20
        annotation_offsets = {2012: 34, 2017: 52, 2020: 34, 2022: 56}
        for _, row in _breakpoint_rows().iterrows():
            year = int(row["year"])
            paper_count = float(row["paper_count"])
            label = str(row["label"])
            fig.add_vline(x=year, line_color="#9AAFC7", line_width=1.1, line_dash="dot")
            fig.add_annotation(
                x=year,
                y=min(paper_count * 1.08, ymax * .91),
                text=f"{year}<br>{label}",
                showarrow=False,
                yshift=annotation_offsets.get(year, 40),
                bgcolor="rgba(255,255,255,.94)",
                bordercolor="#D8E4F2",
                borderwidth=1,
                font=dict(size=10, color="#0B172A"),
                align="center",
            )

        max_count = float(df["paper_count"].max())
        tick_step = 50_000 if max_count > 150_000 else 25_000
        tickvals = [v for v in range(0, int(ymax) + tick_step, tick_step) if v <= ymax]
        fig.update_xaxes(title_text="", tickmode="linear", dtick=2, range=[1999.5, 2025.5])
        fig.update_yaxes(
            title_text="Papers",
            range=[0, ymax],
            tickvals=tickvals,
            ticktext=[_compact_count(v) for v in tickvals],
            showgrid=True,
            gridcolor="#EAF1F8",
        )
        fig.update_layout(showlegend=False, margin=dict(l=54, r=16, t=38, b=34))
        return theme.style(fig, height=365)

    @render_widget
    def growth_top_countries():
        df = nd.top_countries(10).copy()
        if df.empty:
            return theme.empty_figure("Country output cache is empty.")

        df["papers"] = pd.to_numeric(df["papers"], errors="coerce")
        df = df.dropna(subset=["papers"]).sort_values("papers", ascending=True)
        colors = np.where(df["country"].isin(HIGHLIGHT_COUNTRIES), theme.ACCENT, "rgba(159,178,204,0.38)")

        fig = go.Figure(
            go.Bar(
                x=df["papers"],
                y=df["country"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.18)", width=1)),
                customdata=df[["country", "papers"]],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} papers<extra></extra>",
            )
        )

        latest_max = float(df["papers"].max()) if not df.empty else 0
        top_desc = df.sort_values("papers", ascending=False).head(2)
        for _, row in top_desc.iterrows():
            country = str(row["country"])
            papers = float(row["papers"])
            fig.add_annotation(
                x=papers,
                y=country,
                text=f"{_compact_count(papers)} papers",
                showarrow=False,
                xshift=42,
                font=dict(size=10, color=theme.TEXT),
                bgcolor="rgba(255,255,255,.88)",
                bordercolor="rgba(203,213,225,.8)",
                borderwidth=1,
            )

        fig.update_xaxes(title_text="Papers", showgrid=True, gridcolor=theme.GRID, range=[0, latest_max * 1.23])
        fig.update_yaxes(title_text="", automargin=True)
        fig.update_layout(showlegend=False, margin=dict(l=8, r=14, t=12, b=42))
        return theme.style(fig, height=330)

    @render_widget
    def institution_type_participation():
        df = nd.institution_type_year_summary().copy()
        if df.empty:
            return theme.empty_figure("Institution type cache is empty.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        value_col = "unique_papers" if "unique_papers" in df else "paper_institution_rows"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["year", value_col]).copy()
        df["year"] = df["year"].astype(int)

        pivot = (
            df.pivot_table(
                index="year",
                columns="institution_type",
                values=value_col,
                aggfunc="sum",
            )
            .fillna(0)
            .sort_index()
        )

        fig = go.Figure()
        draw_order = [g for g in INSTITUTION_TYPE_ORDER if g in pivot.columns]
        draw_order += [g for g in pivot.columns if g not in draw_order]

        for group in draw_order:
            is_university = group == "University"
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[group],
                    mode="lines+markers" if is_university else "lines",
                    name=group,
                    line=dict(
                        color=(TAB1_ACCENT if is_university else "rgba(148,163,184,.48)"),
                        width=(3.4 if is_university else 1.6),
                    ),
                    marker=dict(size=(6 if is_university else 0), color=TAB1_ACCENT),
                    customdata=np.column_stack([
                        np.repeat(group, len(pivot)),
                        pivot.index.to_numpy(),
                        pivot[group].to_numpy(),
                    ]),
                    hovertemplate=(
                        "Group %{customdata[0]}<br>"
                        "Year %{customdata[1]}<br>"
                        "Papers involving group %{customdata[2]:,.0f}<extra></extra>"
                    ),
                )
            )

        if "University" in pivot.columns and not pivot.empty:
            last_year = int(pivot.index.max())
            last_value = float(pivot.loc[last_year, "University"])
            fig.add_annotation(
                x=last_year,
                y=last_value,
                text="University leads",
                showarrow=True,
                arrowhead=2,
                ax=-70,
                ay=-32,
                bgcolor="rgba(255,255,255,.92)",
                bordercolor="#D8E4F2",
                borderwidth=1,
                font=dict(size=10, color="#0B172A"),
            )

        fig.update_xaxes(title_text="", tickmode="linear", dtick=4)
        fig.update_yaxes(title_text="Papers involving group", tickformat="~s", showgrid=True, gridcolor=theme.GRID)
        fig.update_layout(
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)),
            hovermode="x unified",
            margin=dict(l=52, r=8, t=42, b=34),
        )
        return theme.style(fig, height=330)

    @render.ui
    def institution_lead_note():
        df = nd.institution_type_year_summary().copy()
        if df.empty:
            return notice("Institution type cache is empty.")
        value_col = "unique_papers" if "unique_papers" in df else "paper_institution_rows"
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df[value_col] = pd.to_numeric(df.get(value_col), errors="coerce")
        df = df.dropna(subset=["year", value_col])
        if df.empty:
            return notice("Institution type cache has no readable yearly values.")
        leaders = (
            df.groupby(["year", "institution_type"], as_index=False)[value_col]
            .sum()
            .sort_values(["year", value_col], ascending=[True, False])
            .groupby("year")
            .first()
            .reset_index()
        )
        university_years = int(leaders[leaders["institution_type"].eq("University")]["year"].nunique())
        total_years = int(leaders["year"].nunique())
        if university_years == total_years and total_years > 0:
            return notice(f"University is the leading institution group in every observed year ({total_years}/{total_years}).")
        return notice(f"University leads in {university_years}/{total_years} observed years; this chart measures participation, not exclusive ownership.")

    @render.ui
    def country_concentration_note():
        df = nd.top_countries(10).copy()
        if df.empty:
            return notice("Country output cache is empty.")
        df["papers"] = pd.to_numeric(df["papers"], errors="coerce")
        df = df.dropna(subset=["papers"]).sort_values("papers", ascending=False).reset_index(drop=True)
        if len(df) < 3:
            return notice("Country output is concentrated at the top of the ranking.")
        first = df.iloc[0]
        second = df.iloc[1]
        third = df.iloc[2]
        gap = float(second["papers"] - third["papers"])
        return notice(
            f"{first['country']} has {int(first['papers']):,} papers and {second['country']} has {int(second['papers']):,}. "
            f"The gap between #{2} {second['country']} and #{3} {third['country']} is {int(gap):,} papers."
        )

    @render_widget
    def theme_diversity():
        df = nd.diversity_metrics().copy()
        if df.empty:
            return theme.empty_figure("Diversity cache is empty.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
        df["top5_share"] = pd.to_numeric(df.get("top5_share"), errors="coerce")
        df = df.dropna(subset=["year", "entropy"]).sort_values("year")

        custom = df[["entropy", "top5_share"]].to_numpy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["entropy"],
                customdata=custom,
                mode="lines+markers",
                name="Entropy",
                line=dict(color=TAB1_ACCENT, width=3),
                marker=dict(size=5, color=TAB1_ACCENT),
                hovertemplate=(
                    "Year %{x}<br>"
                    "Topic entropy %{customdata[0]:.2f}<br>"
                    "Top-5 share %{customdata[1]:.1f}%<extra></extra>"
                ),
            )
        )

        if "top5_share" in df and df["top5_share"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=df["top5_share"],
                    customdata=custom,
                    mode="lines",
                    name="Top-5 share",
                    yaxis="y2",
                    line=dict(color=SECONDARY_ACCENT, width=2.2),
                    hovertemplate=(
                        "Year %{x}<br>"
                        "Topic entropy %{customdata[0]:.2f}<br>"
                        "Top-5 share %{customdata[1]:.1f}%<extra></extra>"
                    ),
                )
            )

        if len(df) >= 2:
            first = df.iloc[0]
            last = df.iloc[-1]
            delta = float(last["entropy"] - first["entropy"])
            fig.add_annotation(
                x=float(last["year"]),
                y=float(last["entropy"]),
                text=f"Entropy {_format_delta(delta)}",
                showarrow=True,
                arrowhead=2,
                ax=-52,
                ay=-34,
                bgcolor="rgba(255,255,255,.9)",
                bordercolor="#D8E4F2",
                borderwidth=1,
                font=dict(size=10, color="#0B172A"),
            )

        fig.update_xaxes(title_text="", tickmode="linear", dtick=4)
        fig.update_layout(
            yaxis=dict(title="Entropy", gridcolor=theme.GRID),
            yaxis2=dict(title="Top-5 share", overlaying="y", side="right", ticksuffix="%", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)),
            hovermode="x unified",
            margin=dict(l=48, r=54, t=42, b=34),
        )
        return theme.style(fig, height=330)