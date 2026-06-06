"""Tab 3: growth-impact frontier ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import narrative_data as nd
from . import theme
from .narrative_common import badge, card_header, metric, notice, paper_list


QUADRANT_COLORS = {
    "Rising stars": "#059669",
    "Hidden gems": "#1d4ed8",
    "Fast growth, lower impact": "#d97706",
    "Mature or crowded": "#64748b",
}


def _clean_topics() -> pd.DataFrame:
    df = nd.growth_impact_matrix().copy()
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["primary_topic", "family", "growth", "median_fwci", "paper_count"]
    )
    for col in ["growth", "median_fwci", "median_velocity", "paper_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["growth"] = df["growth"].clip(lower=.01)
    df["median_velocity"] = df["median_velocity"].fillna(0)
    return _add_frontier_score(df.dropna(subset=["growth", "median_fwci", "paper_count"]))


def _add_frontier_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    growth_rank = np.log1p(out["growth"]).rank(pct=True)
    impact_rank = out["median_fwci"].rank(pct=True)
    velocity_rank = out["median_velocity"].rank(pct=True)
    out["frontier_score"] = 100 * (.40 * growth_rank + .35 * impact_rank + .25 * velocity_rank)
    out["top_signal"] = np.select(
        [
            growth_rank >= impact_rank.combine(velocity_rank, max),
            impact_rank >= growth_rank.combine(velocity_rank, max),
        ],
        ["Recent growth", "Normalized impact"],
        default="Citation velocity",
    )
    return out


def _family_frame(topics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, sub in topics.groupby("family"):
        weight = sub["paper_count"].clip(lower=1)
        rows.append({
            "name": family,
            "family": family,
            "primary_topic": family,
            "paper_count": float(sub["paper_count"].sum()),
            "growth": float(np.average(sub["growth"], weights=weight)),
            "median_fwci": float(np.average(sub["median_fwci"], weights=weight)),
            "median_velocity": float(np.average(sub["median_velocity"], weights=weight)),
            "topics_shown": int(len(sub)),
            "top_topics": ", ".join(sub.sort_values("frontier_score", ascending=False)["primary_topic"].head(5)),
        })
    return _add_frontier_score(pd.DataFrame(rows))


def _apply_quadrants(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    out = df.copy()
    x_cut = float(out["growth"].quantile(.75))
    y_cut = float(out["median_fwci"].quantile(.75))
    out["quadrant"] = np.select(
        [
            (out["growth"] >= x_cut) & (out["median_fwci"] >= y_cut),
            (out["growth"] < x_cut) & (out["median_fwci"] >= y_cut),
            (out["growth"] >= x_cut) & (out["median_fwci"] < y_cut),
        ],
        ["Rising stars", "Hidden gems", "Fast growth, lower impact"],
        default="Mature or crowded",
    )
    return out, x_cut, y_cut


def _size_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    quantiles = out["paper_count"].quantile([.25, .5, .75]).to_dict()

    def bucket(count):
        if count <= quantiles[.25]:
            return "Small", 18
        if count <= quantiles[.5]:
            return "Medium", 28
        if count <= quantiles[.75]:
            return "Large", 38
        return "XL", 52

    buckets = out["paper_count"].map(bucket)
    out["size_tier"] = [label for label, _ in buckets]
    out["bubble_size"] = [size for _, size in buckets]
    return out


def _metric_if(label: str, value, note: str | None = None, fmt=None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return None
    return metric(label, fmt(value) if fmt else str(value), note)


def _metric_group(*items):
    visible = [item for item in items if item is not None]
    return ui.div(*visible, class_="metric-grid") if visible else ui.div()


def impact_ui():
    return ui.nav_panel(
        "What created impact",
        ui.div(
            ui.div(
                badge("Growth-impact matrix"),
                badge("Frontier score"),
                badge("Interpretable ranking"),
                class_="badge-row",
            ),
            ui.h2("What created impact"),
            ui.p(
                "Growth alone does not explain visibility. We compare growth, normalized impact, and frontier signals.",
                class_="tab-insight",
            ),
            class_="tab-heading",
        ),
        ui.layout_columns(
            ui.card(
                card_header(
                    "Frontier map: growth vs normalized impact",
                    "Read right for growth, upward for FWCI, and bubble size for topic volume.",
                ),
                ui.output_ui("frontier_view_controls"),
                output_widget("growth_impact_matrix"),
                ui.tags.script("""
                (function() {
                  function bindPlot(outputId, flag, inputId) {
                    const wrapper = document.getElementById(outputId);
                    const el = wrapper && (wrapper.classList.contains("js-plotly-plot") ? wrapper : wrapper.querySelector(".js-plotly-plot"));
                    if (!el || el.dataset[flag] || !window.Shiny || typeof el.on !== "function") return false;
                    el.dataset[flag] = "1";
                    el.on("plotly_click", function(eventData) {
                      const point = eventData && eventData.points && eventData.points[0];
                      const d = point && point.customdata;
                      if (d) Shiny.setInputValue(inputId, {
                        kind: d[0], family: d[1], topic: d[2], name: d[3]
                      }, {priority: "event"});
                    });
                    return true;
                  }
                  function bindAll() {
                    bindPlot("growth_impact_matrix", "frontierMapBound", "frontier_map_click");
                    bindPlot("frontier_ranking", "frontierRankingBound", "frontier_ranking_click");
                  }
                  let tries = 0;
                  const retry = setInterval(function() {
                    bindAll();
                    tries += 1;
                    if (tries > 80) clearInterval(retry);
                  }, 250);
                  document.addEventListener("shiny:value", function(e) {
                    if (e.target && (e.target.id === "growth_impact_matrix" || e.target.id === "frontier_ranking")) {
                      setTimeout(bindAll, 80);
                    }
                  });
                })();
                """),
            ),
            ui.card(
                ui.output_ui("selected_frontier_detail"),
            ),
            col_widths=[8, 4],
        ),
        ui.layout_columns(
            ui.card(
                card_header(
                    "Top frontier topics",
                    "Topics ranked by combined growth, field-normalized impact, and citation velocity.",
                ),
                output_widget("frontier_ranking"),
                notice(
                    "The score is directional, not causal. It surfaces topics with recent growth and normalized impact."
                ),
            ),
            ui.card(
                card_header("Topic passport"),
                ui.output_ui("topic_passport"),
            ),
            col_widths=[7, 5],
        ),
    )


def impact_server(input, output, session):
    selected_family = reactive.Value(None)
    selected_topic = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.frontier_map_click)
    def _frontier_map_click():
        payload = input.frontier_map_click()
        if not payload:
            return
        kind = payload.get("kind")
        if kind == "family":
            selected_family.set(str(payload.get("family")))
            selected_topic.set(None)
        elif kind == "topic":
            selected_family.set(str(payload.get("family")))
            selected_topic.set(str(payload.get("topic")))

    @reactive.effect
    @reactive.event(input.frontier_ranking_click)
    def _frontier_ranking_click():
        payload = input.frontier_ranking_click()
        if not payload:
            return
        selected_family.set(str(payload.get("family")))
        selected_topic.set(str(payload.get("topic")))

    @reactive.effect
    @reactive.event(input.back_to_families)
    def _back_to_families():
        selected_family.set(None)
        selected_topic.set(None)

    @reactive.effect
    @reactive.event(input.reset_frontier_view)
    def _reset_frontier_view():
        selected_family.set(None)
        selected_topic.set(None)

    @render.ui
    def frontier_view_controls():
        family = selected_family()
        topic = selected_topic()
        breadcrumb = "All AI"
        if family:
            breadcrumb += f" > {family}"
        if topic:
            breadcrumb += f" > {topic}"
        return ui.div(
            ui.div(ui.tags.strong("Viewing:"), f" {breadcrumb}", class_="text-muted small"),
            ui.div(
                ui.input_action_button("back_to_families", "Back to families", class_="btn btn-outline-secondary btn-sm"),
                ui.input_action_button("reset_frontier_view", "Reset view", class_="btn btn-outline-secondary btn-sm"),
                class_="d-flex gap-2 flex-wrap my-2",
            ),
        )

    @render_widget
    def growth_impact_matrix():
        topics = _clean_topics()
        if topics.empty:
            return theme.empty_figure("No growth-impact matrix data")

        family = selected_family()
        topic = selected_topic()
        if family:
            plot_df = topics[topics["family"].eq(family)].copy()
            kind = "topic"
            name_col = "primary_topic"
            if plot_df.empty:
                selected_family.set(None)
                plot_df = _family_frame(topics)
                kind = "family"
                name_col = "name"
        else:
            plot_df = _family_frame(topics)
            kind = "family"
            name_col = "name"

        plot_df, x_cut, y_cut = _apply_quadrants(plot_df)
        plot_df = _size_buckets(plot_df)
        plot_df["name"] = plot_df[name_col].astype(str)
        selected_name = topic if kind == "topic" else family
        top_labels = set(plot_df.sort_values("frontier_score", ascending=False).head(4)["name"])
        top_labels.update(plot_df.sort_values("growth", ascending=False).head(1)["name"])
        top_labels.update(plot_df.sort_values("median_fwci", ascending=False).head(1)["name"])
        if selected_name:
            top_labels.add(str(selected_name))

        x_min = max(float(plot_df["growth"].min()) * .72, .01)
        x_max = max(float(plot_df["growth"].max()) * 1.4, x_min * 1.4)
        y_min = 0.0
        y_max = max(float(plot_df["median_fwci"].max()) * 1.18, .8)
        fig = go.Figure()
        bg = {
            "Rising stars": "rgba(5,150,105,.09)",
            "Hidden gems": "rgba(29,78,216,.08)",
            "Fast growth, lower impact": "rgba(217,119,6,.08)",
            "Mature or crowded": "rgba(100,116,139,.07)",
        }
        for x0, x1, y0, y1, label in [
            (x_cut, x_max, y_cut, y_max, "Rising stars"),
            (x_min, x_cut, y_cut, y_max, "Hidden gems"),
            (x_cut, x_max, y_min, y_cut, "Fast growth, lower impact"),
            (x_min, x_cut, y_min, y_cut, "Mature or crowded"),
        ]:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=bg[label], line=dict(width=0), layer="below")

        for quad, sub in plot_df.groupby("quadrant"):
            is_selected = sub["name"].eq(str(selected_name))
            show_label = sub["name"].isin(top_labels)
            custom = pd.DataFrame({
                "kind": kind,
                "family": sub["family"],
                "topic": sub["primary_topic"],
                "name": sub["name"],
                "paper_count": sub["paper_count"],
                "growth": sub["growth"],
                "median_fwci": sub["median_fwci"],
                "median_velocity": sub["median_velocity"],
                "frontier_score": sub["frontier_score"],
                "size_tier": sub["size_tier"],
            })
            fig.add_trace(go.Scatter(
                x=sub["growth"],
                y=sub["median_fwci"],
                mode="markers+text",
                name=quad,
                text=sub["name"].where(show_label, ""),
                textposition="top center",
                textfont=dict(size=10, color="#0f172a"),
                marker=dict(
                    size=sub["bubble_size"],
                    color=QUADRANT_COLORS[quad],
                    opacity=np.where(show_label | is_selected, .86, .48),
                    line=dict(color=np.where(is_selected, "#020617", "#ffffff"), width=np.where(is_selected, 3, 1.2)),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[3]}</b><br>"
                    "%{customdata[4]:,.0f} papers<br>"
                    "Growth ratio %{customdata[5]:.2f}x<br>"
                    "Median FWCI %{customdata[6]:.2f}<br>"
                    "Citation velocity %{customdata[7]:.1f}/yr<br>"
                    "Frontier score %{customdata[8]:.1f}<extra></extra>"
                ),
            ))

        fig.add_vline(x=x_cut, line_color="#0f172a", line_dash="dash", line_width=1.7)
        fig.add_hline(y=y_cut, line_color="#0f172a", line_dash="dash", line_width=1.7)
        fig.add_annotation(x=x_cut, y=y_max * .98, showarrow=False, text="Top 25 percent growth cutoff",
                           textangle=-90, xanchor="left", yanchor="top", font=dict(size=10, color="#0f172a"),
                           bgcolor="rgba(255,255,255,.86)")
        fig.add_annotation(x=x_min * 1.05, y=y_cut, showarrow=False, text="Top 25 percent impact cutoff",
                           xanchor="left", yanchor="bottom", font=dict(size=10, color="#0f172a"),
                           bgcolor="rgba(255,255,255,.86)")

        tickvals = [v for v in [.01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if x_min <= v <= x_max]
        fig.update_xaxes(
            title_text="Recent growth ratio",
            type="log",
            range=[np.log10(x_min), np.log10(x_max)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=[f"{v:g}x" for v in tickvals],
            showgrid=True,
            gridcolor="#edf2f7",
        )
        fig.update_yaxes(title_text="Median FWCI", range=[y_min, y_max], showgrid=True, gridcolor="#edf2f7")
        fig.update_layout(margin=dict(l=56, r=20, t=54, b=54), legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)))
        return theme.style(fig, height=470)

    @render.ui
    def selected_frontier_detail():
        topics = _clean_topics()
        if topics.empty:
            return ui.p("No matrix detail data is available.", class_="text-muted small")
        family = selected_family()
        topic = selected_topic()
        if topic:
            row = topics[topics["primary_topic"].eq(topic)]
            if row.empty:
                topic = None
            else:
                r = row.iloc[0]
                return ui.div(
                    card_header("Selected frontier detail"),
                    _metric_group(
                        metric("Selected topic", str(r["primary_topic"]), str(r["family"])),
                        metric("Paper count", f"{float(r['paper_count']):,.0f}"),
                        metric("Growth ratio", f"{float(r['growth']):.2f}x"),
                        metric("Median FWCI", f"{float(r['median_fwci']):.2f}"),
                        metric("Citation velocity", f"{float(r['median_velocity']):.1f}/yr"),
                        metric("Frontier score", f"{float(r['frontier_score']):.1f}"),
                    ),
                )
        if family:
            sub = topics[topics["family"].eq(family)].sort_values("frontier_score", ascending=False)
            if not sub.empty:
                items = [
                    ui.tags.li(
                        ui.tags.strong(str(row["primary_topic"])),
                        ui.br(),
                        ui.span(f"Score {float(row['frontier_score']):.1f} | {float(row['growth']):.2f}x growth | FWCI {float(row['median_fwci']):.2f}",
                                class_="text-muted small"),
                    )
                    for _, row in sub.head(5).iterrows()
                ]
                return ui.div(
                    card_header("Selected frontier detail"),
                    _metric_group(
                        metric("Selected family", family),
                        metric("Topics shown", f"{len(sub):,.0f}"),
                        metric("Family papers", f"{float(sub['paper_count'].sum()):,.0f}"),
                        metric("Median FWCI", f"{float(sub['median_fwci'].median()):.2f}"),
                    ),
                    ui.p("This family view separates its faster-growing subtopics from the subtopics with stronger normalized impact.",
                         class_="interpretation"),
                    ui.p(ui.tags.strong("Top 5 subtopics"), class_="panel-label"),
                    ui.tags.ul(*items, class_="paper-list"),
                )
        return ui.div(
            card_header("How to read the matrix"),
            ui.p(
                "Each bubble is a topic family. Right means faster recent growth. Up means stronger normalized impact. Larger bubbles mean more papers.",
                class_="interpretation",
            ),
            ui.p("Click a family to inspect its subtopics.", class_="text-muted small"),
        )

    @render_widget
    def frontier_ranking():
        topics = _clean_topics()
        if topics.empty:
            return theme.empty_figure("No frontier ranking data")
        family = selected_family()
        df = topics[topics["family"].eq(family)].copy() if family else topics.copy()
        if df.empty:
            df = topics.copy()
        df = df.sort_values("frontier_score", ascending=False).head(8).sort_values("frontier_score")
        selected = selected_topic()
        colors = np.where(df["primary_topic"].eq(str(selected)), "#0f172a", "#1d4ed8")
        fig = go.Figure(go.Bar(
            x=df["frontier_score"],
            y=df["primary_topic"],
            orientation="h",
            marker_color=colors,
            customdata=pd.DataFrame({
                "kind": "topic",
                "family": df["family"],
                "topic": df["primary_topic"],
                "name": df["primary_topic"],
                "paper_count": df["paper_count"],
                "growth": df["growth"],
                "median_fwci": df["median_fwci"],
                "median_velocity": df["median_velocity"],
                "frontier_score": df["frontier_score"],
            }),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Frontier score %{x:.1f}<br>"
                "%{customdata[4]:,.0f} papers<br>"
                "Growth ratio %{customdata[5]:.2f}x<br>"
                "Median FWCI %{customdata[6]:.2f}<br>"
                "Citation velocity %{customdata[7]:.1f}/yr<extra></extra>"
            ),
        ))
        fig.update_xaxes(title_text="Frontier score")
        fig.update_yaxes(title_text="", automargin=True)
        fig.update_layout(margin=dict(l=8, r=16, t=12, b=46), showlegend=False)
        return theme.style(fig, height=390)

    @render.ui
    def topic_passport():
        topics = _clean_topics()
        if topics.empty:
            return ui.p("No topic passport data is available.", class_="text-muted small")
        topic = selected_topic()
        rows = topics[topics["primary_topic"].eq(topic)] if topic else pd.DataFrame()
        if rows.empty:
            rows = topics.sort_values("frontier_score", ascending=False).head(1)
        row = rows.iloc[0]
        papers = nd.representative_papers(str(row["primary_topic"]), family=str(row["family"]), limit=4)
        return ui.div(
            _metric_group(
                metric("Selected topic", str(row["primary_topic"]), str(row["family"])),
                metric("Parent family", str(row["family"])),
                metric("Paper count", f"{float(row['paper_count']):,.0f}"),
                metric("Frontier score", f"{float(row['frontier_score']):.1f}"),
                metric("Growth ratio", f"{float(row['growth']):.2f}x"),
                metric("Median FWCI", f"{float(row['median_fwci']):.2f}"),
                metric("Citation velocity", f"{float(row['median_velocity']):.1f}/yr"),
                metric("Top signal", str(row["top_signal"])),
            ),
            ui.p(
                "The frontier score combines recent growth, normalized impact, and citation velocity. It is a ranking signal, not a causal explanation.",
                class_="interpretation",
            ),
            ui.p(ui.tags.strong("Representative papers"), class_="panel-label"),
            paper_list(papers, topic=True, limit=4),
        )

