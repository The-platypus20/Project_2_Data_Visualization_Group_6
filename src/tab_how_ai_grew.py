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
        return pd.DataFrame(columns=["year", "paper_count", "yoy_growth", "mean_fwci", "median_fwci"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["paper_count"] = pd.to_numeric(df["count"], errors="coerce")
    for col in ["mean_fwci", "median_fwci"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.dropna(subset=["year", "paper_count"])
    df["year"] = df["year"].astype(int)
    df["yoy_growth"] = (
        df["paper_count"].pct_change().replace([float("inf"), -float("inf")], pd.NA) * 100
    )
    return df[["year", "paper_count", "yoy_growth", "mean_fwci", "median_fwci"]]


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
    return (
    f"{float(last['entropy']):.2f}",
    f"{_format_delta(delta)} since {int(first['year'])}. Higher = broader topic spread.",
)


def _kpi_card(label: str, value: str, note: str, accent: str = TAB1_ACCENT) -> ui.Tag:
    return ui.div(
        ui.div(label, class_="kpi-label"),
        ui.div(value, class_="kpi-value"),
        ui.div(note, class_="kpi-note"),
        class_="kpi-card",
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
                        "AI research output grew steadily in the 2000s, then surged after 2017. The field's growth was anchored by universities and concentrated in a few leading countries, but open access expansion made research more accessible globally.",
                        class_="tab-insight",
                    ),
                ),
                ui.div(badge("OpenAlex 2000-2025"), badge("Scale and structure"), class_="badge-row"),
                class_="growth-header-row",
            ),
            ui.output_ui("growth_kpi_cards"),
            ui.card(
                card_header(
                    "AI output surged after 2017, but field impact did not rise at the same pace",
                    "Annual paper count with mean field-normalized impact overlay. A value of 1.0 means world average.",
                ),
                output_widget("paper_timeline"),
            ),
            ui.layout_columns(
                ui.card(
                    card_header(
                        "China and the United States became the clear output leaders",
                        "Yearly AI paper output for the top 8 countries. The US and China lead by a wide margin, with a growing gap between #2 and #3.",
                    ),
                    output_widget("growth_top_countries"),
                ),
                ui.card(
                    card_header(
                        "Universities remained the main institutional anchor of AI research",
                        "Share of institutional participation by year. University involvement stays dominant across the full period.",
                    ),
                    output_widget("institution_type_participation"),
                ),
                col_widths=[6, 6],
            ),
            ui.card(
                card_header(
                    "AI research became more accessible as open access expanded.",
                    "Open and closed access status by publication period." ,
                ),
                output_widget("open_access_period_chart"),
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
                _kpi_card("AI Papers Indexed", "N/A", "No yearly cache"),
                _kpi_card("Output Growth", "N/A", "No yearly cache"),
                _kpi_card("Leading Countries", "N/A", "No country cache"),
                _kpi_card("Topic Diversity", "N/A", "No diversity cache"),
                class_="kpi-grid",
            )

        first = yearly.iloc[0]
        last = yearly.iloc[-1]
        total_papers = float(yearly["paper_count"].sum())
        growth_multiplier = (
            float(last["paper_count"] / first["paper_count"])
            if float(first["paper_count"]) > 0
            else np.nan
        )

        country_value = "USA + China"
        country_note = "Largest output producers"
        if not countries.empty:
            countries["papers"] = pd.to_numeric(countries["papers"], errors="coerce")
            highlights = countries[countries["country"].isin(HIGHLIGHT_COUNTRIES)].copy()
            if not highlights.empty:
                highlight_total = int(highlights["papers"].sum())
                country_note = f"{highlight_total:,} papers combined"

        entropy_value, entropy_note = _latest_entropy()

        return ui.div(
            _kpi_card("AI Papers Indexed", f"{_compact_count(total_papers)}", f"{int(first['year'])}-{int(last['year'])}"),
            _kpi_card(
                "Output Growth",
                f"{growth_multiplier:.1f}×" if pd.notna(growth_multiplier) else "N/A",
                f"{int(first['year'])} to {int(last['year'])}",
                accent="#A78BFA",
            ),
            _kpi_card("Leading Countries", country_value, country_note, accent="#F59E0B"),
            _kpi_card("Topic Diversity", entropy_value, entropy_note, accent="#22C55E"),
            class_="kpi-grid",
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

        impact_col = "mean_fwci"
        if impact_col in df and df[impact_col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=df[impact_col],
                    mode="lines+markers",
                    name="Mean field impact",
                    yaxis="y2",
                    line=dict(color=SECONDARY_ACCENT, width=2.4, dash="dot"),
                    marker=dict(size=5, color=SECONDARY_ACCENT),
                    hovertemplate="Year %{x}<br>Mean field-normalized impact %{y:.2f}<extra></extra>",
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
                font=dict(size=12, color="#0B172A"),
                align="center",
            )

        max_count = float(df["paper_count"].max())
        tick_step = 50_000 if max_count > 150_000 else 25_000
        tickvals = [v for v in range(0, int(ymax) + tick_step, tick_step) if v <= ymax]
        fig.update_xaxes(title_text="", tickmode="linear", dtick=2, range=[1999.5, 2025.5], tickfont=dict(size=12))
        fig.update_yaxes(
            title=dict(text="Papers", font=dict(size=13)),
            tickfont=dict(size=12),
            range=[0, ymax],
            tickvals=tickvals,
            ticktext=[_compact_count(v) for v in tickvals],
            showgrid=True,
            gridcolor="#EAF1F8",
        )
        if df[["mean_fwci", "median_fwci"]].notna().any().any():
            fig.update_layout(
                yaxis2=dict(title=dict(text="Mean field impact", font=dict(size=13)), overlaying="y", side="right", showgrid=False, rangemode="tozero"),
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)),
            )
        fig.update_layout(showlegend=bool(df[["mean_fwci", "median_fwci"]].notna().any().any()), font=dict(size=12), margin=dict(l=62, r=70, t=42, b=38))
        return theme.style(fig, height=430)

    @render_widget
    def growth_top_countries():
        df = nd.country_topic_year_counts().copy()
        if df.empty:
            return theme.empty_figure("Country-year output cache is empty.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df = df.dropna(subset=["year", "count", "country"]).copy()
        df["year"] = df["year"].astype(int)
        if df.empty:
            return theme.empty_figure("Country-year output cache has no readable values.")

        yearly_country = (
            df.groupby(["year", "country"], as_index=False)["count"]
            .sum()
            .sort_values(["country", "year"])
        )
        totals = yearly_country.groupby("country")["count"].sum().sort_values(ascending=False)
        keep = list(totals.head(8).index)
        top_two = list(totals.head(2).index)
        plot_df = yearly_country[yearly_country["country"].isin(keep)].copy()

        fig = go.Figure()
        highlight_colors = [TAB1_ACCENT, SECONDARY_ACCENT]
        color_lookup = {country: highlight_colors[i] for i, country in enumerate(top_two[:2])}

        # Draw muted countries first, then highlighted leaders on top.
        draw_order = [c for c in keep if c not in top_two] + top_two
        for country in draw_order:
            sub = plot_df[plot_df["country"].eq(country)].sort_values("year")
            if sub.empty:
                continue
            is_highlight = country in top_two
            fig.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub["count"],
                    mode="lines+markers" if is_highlight else "lines",
                    name=country,
                    line=dict(
                        color=color_lookup.get(country, "rgba(159,178,204,0.42)"),
                        width=3.6 if is_highlight else 1.4,
                    ),
                    marker=dict(size=6 if is_highlight else 0, color=color_lookup.get(country, "rgba(159,178,204,0.42)")),
                    opacity=1.0 if is_highlight else 0.45,
                    customdata=np.column_stack([
                        np.repeat(country, len(sub)),
                        sub["count"].to_numpy(),
                    ]),
                    hovertemplate="<b>%{customdata[0]}</b><br>Year %{x}<br>Papers %{customdata[1]:,.0f}<extra></extra>",
                )
            )

        for country in top_two:
            sub = plot_df[plot_df["country"].eq(country)].sort_values("year")
            if sub.empty:
                continue
            last = sub.iloc[-1]
            fig.add_annotation(
                x=int(last["year"]),
                y=float(last["count"]),
                text=f"{country}, {int(last['year'])}: {_compact_count(last['count'])}",
                showarrow=True,
                arrowhead=2,
                ax=-78,
                ay=-26 if country == top_two[0] else 24,
                bgcolor="rgba(255,255,255,.94)",
                bordercolor="rgba(203,213,225,.85)",
                borderwidth=1,
                font=dict(size=12, color="#0B172A"),
            )

        max_count = float(plot_df["count"].max()) if not plot_df.empty else 0
        fig.update_xaxes(title_text="", tickmode="linear", dtick=4, tickfont=dict(size=12))
        fig.update_yaxes(
            title=dict(text="Papers", font=dict(size=13)),
            tickformat="~s",
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor=theme.GRID,
            range=[0, max_count * 1.16 if max_count > 0 else 1],
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)),
            hovermode="x unified",
            font=dict(size=12),
            margin=dict(l=54, r=18, t=48, b=36),
        )
        return theme.style(fig, height=390)

    @render_widget
    def institution_type_participation():
        df = nd.institution_type_year_summary().copy()
        if df.empty:
            return theme.empty_figure("Institution type cache is empty.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        value_col = "unique_papers" if "unique_papers" in df else "paper_institution_rows"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["year", value_col, "institution_type"]).copy()
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

        if pivot.empty:
            return theme.empty_figure("Institution type cache has no readable yearly values.")

        draw_order = [g for g in INSTITUTION_TYPE_ORDER if g in pivot.columns]
        draw_order += [g for g in pivot.columns if g not in draw_order]
        # Put University first so its band is easy to compare across time.
        draw_order = [g for g in draw_order if g == "University"] + [g for g in draw_order if g != "University"]

        totals = pivot.sum(axis=1).replace(0, np.nan)
        share = pivot.div(totals, axis=0) * 100
        share = share.fillna(0)

        fig = go.Figure()
        muted_fills = [
            "rgba(148,163,184,0.24)",
            "rgba(148,163,184,0.19)",
            "rgba(148,163,184,0.14)",
            "rgba(148,163,184,0.10)",
            "rgba(148,163,184,0.08)",
            "rgba(148,163,184,0.06)",
        ]

        cumulative = pd.Series(0.0, index=share.index)

        for i, group in enumerate(draw_order):
            is_university = group == "University"
            y_share = share[group]
            y_count = pivot[group]
            upper = cumulative + y_share
            lower = cumulative.copy()

            fig.add_trace(
                go.Scatter(
                    x=share.index,
                    y=upper,
                    mode="lines",
                    name=group,
                    fill="tonexty" if i > 0 else "tozeroy",
                    line=dict(
                        color=TAB1_ACCENT if is_university else "rgba(159,178,204,0.62)",
                        width=3.4 if is_university else 1.1,
                    ),
                    fillcolor="rgba(124,201,255,0.68)" if is_university else muted_fills[i % len(muted_fills)],
                    opacity=1.0 if is_university else 0.82,
                    customdata=np.column_stack([
                        np.repeat(group, len(share)),
                        share.index.to_numpy(),
                        y_count.to_numpy(),
                        y_share.to_numpy(),
                        lower.to_numpy(),
                        upper.to_numpy(),
                    ]),
                    hovertemplate=(
                        "Group %{customdata[0]}<br>"
                        "Year %{customdata[1]}<br>"
                        "Share %{customdata[3]:.1f}%<br>"
                        "Papers involving group %{customdata[2]:,.0f}<extra></extra>"
                    ),
                )
            )

            cumulative = upper

        if "University" in share.columns and not share.empty:
            last_year = int(share.index.max())
            last_share = float(share.loc[last_year, "University"])
            last_count = float(pivot.loc[last_year, "University"])
            fig.add_annotation(
                x=last_year,
                y=last_share,
                text=f"University, {last_year}: {last_share:.0f}%",
                showarrow=True,
                arrowhead=2,
                ax=-90,
                ay=-24,
                bgcolor="rgba(255,255,255,.94)",
                bordercolor="#D8E4F2",
                borderwidth=1,
                font=dict(size=12, color="#0B172A"),
                hovertext=f"{last_count:,.0f} papers",
            )

        fig.update_xaxes(title_text="", tickmode="linear", dtick=4, tickfont=dict(size=12))
        fig.update_yaxes(
            title=dict(text="Share of institutional participation", font=dict(size=13)),
            range=[0, 100],
            ticksuffix="%",
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor=theme.GRID,
        )
        fig.update_layout(
            legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
            hovermode="x unified",
            font=dict(size=12),
            margin=dict(l=56, r=8, t=46, b=36),
        )
        return theme.style(fig, height=390)

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
    def open_access_period_chart():
        df = nd.open_access_period_summary().copy()
        if df.empty:
            return theme.empty_figure("Open access cache is empty. Rebuild the dashboard cache to create oa_period_status.csv.")

        needed = {"period", "access_status", "count"}
        if not needed.issubset(df.columns):
            return theme.empty_figure("Open access cache is missing required columns.")

        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
        period_order = ["2000-2009", "2010-2019", "2020-2025"]
        status_order = ["Open access", "Closed"]

        df = df[df["access_status"].astype(str).isin(status_order)].copy()

        df["period"] = pd.Categorical(df["period"], categories=period_order, ordered=True)
        df["access_status"] = pd.Categorical(df["access_status"], categories=status_order, ordered=True)
        df = df.dropna(subset=["period", "access_status"])

        totals = df.groupby("period", observed=False)["count"].transform("sum").replace(0, np.nan)
        df["share"] = 100 * df["count"] / totals
        df["share"] = df["share"].fillna(0)

        colors = {
            "Open access": TAB1_ACCENT,
            "Closed": "rgba(159,178,204,0.68)"
        }
        fig = go.Figure()
        for status in status_order:
            sub = df[df["access_status"].astype(str).eq(status)].sort_values("period")
            if sub.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=sub["period"].astype(str),
                    y=sub["share"],
                    name=status,
                    marker=dict(color=colors[status], line=dict(color="rgba(255,255,255,0.18)", width=1)),
                    customdata=sub[["count", "share"]].to_numpy(),
                    text=sub["share"].map(lambda v: f"{v:.0f}%" if v >= 6 else ""),
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=13, color="#07111F" if status == "Open access" else "#0F172A"),
                    hovertemplate=(
                        f"<b>{status}</b><br>"
                        "Period %{x}<br>"
                        "Share %{customdata[1]:.1f}%<br>"
                        "Papers %{customdata[0]:,.0f}<extra></extra>"
                    ),
                )
            )

        fig.update_xaxes(title_text="", tickfont=dict(size=12))
        fig.update_yaxes(
            title=dict(text="Share of papers", font=dict(size=13)),
            range=[0, 100],
            ticksuffix="%",
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor=theme.GRID,
        )
        fig.update_layout(
            barmode="stack",
            uniformtext_minsize=11,
            uniformtext_mode="hide",
            legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
            font=dict(size=12),
            margin=dict(l=54, r=8, t=46, b=36),
        )
        return theme.style(fig, height=390)
