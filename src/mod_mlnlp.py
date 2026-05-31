"""ML & NLP tab: title clustering, publication forecast, impact prediction."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

from . import mlnlp, theme


def mlnlp_ui():
    return ui.nav_panel(
        "ML & NLP",
        ui.h5("1 · Unsupervised NLP — clustering paper titles (TF-IDF + KMeans)"),
        ui.layout_columns(
            ui.value_box("Titles clustered", ui.output_text("nlp_n")),
            ui.value_box("Silhouette score", ui.output_text("nlp_sil"),
                         ui.span("cluster separation (−1…1)", class_="text-muted small")),
            ui.input_slider("nlp_k", "Number of clusters (k)", min=3, max=10, value=8, step=1),
            col_widths=[3, 3, 6], fill=False,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Title clusters (2-D TF-IDF projection)"),
                output_widget("nlp_scatter"),
                ui.card_footer(ui.span("Each point is a paper, positioned by TF-IDF similarity "
                                       "(TruncatedSVD) and coloured by KMeans cluster.",
                                       class_="text-muted small")),
            ),
            ui.card(
                ui.card_header("Cluster sizes"),
                output_widget("nlp_sizes"),
            ),
            col_widths=[8, 4],
        ),
        ui.card(
            ui.card_header(
                ui.span("Top terms in cluster "),
                ui.input_select("nlp_cluster", None, choices=[], width="120px"),
            ),
            output_widget("nlp_terms"),
        ),
        ui.hr(),
        ui.h5("2 · Time-series forecast — projecting publication volume (Holt's trend)"),
        ui.layout_columns(
            ui.input_slider("fc_cutoff", "Train up to year", min=2010, max=2024,
                            value=2019, step=1, sep=""),
            ui.input_slider("fc_horizon", "Forecast horizon (years)", min=1, max=6, value=4, step=1),
            col_widths=[6, 6], fill=False,
        ),
        ui.card(
            ui.card_header("Publications per year — actual vs forecast"),
            output_widget("fc_chart"),
            ui.card_footer(ui.span(
                "Holt's linear-trend model fitted on counts up to the cutoff, projected ahead "
                "with a ±95% band. NOTE: this is a sample of highly-cited works, so recent years "
                "are right-censored (papers haven't accrued citations yet) — read the forecast as "
                "a trend extrapolation, not a literal prediction of all output.",
                class_="text-muted small")),
        ),
        ui.hr(),
        ui.h5("3 · Supervised ML — what predicts high citation velocity?"),
        ui.layout_columns(
            ui.value_box("Model ROC AUC", ui.output_text("ml_auc"),
                         ui.span("Random Forest, held-out test", class_="text-muted small")),
            ui.value_box("Accuracy", ui.output_text("ml_acc"),
                         ui.span("top-quartile velocity vs rest", class_="text-muted small")),
            col_widths=[6, 6], fill=False,
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Feature importance"), output_widget("ml_importance")),
            ui.card(ui.card_header("ROC curve"), output_widget("ml_roc")),
            ui.card(ui.card_header("Confusion matrix"), output_widget("ml_confusion")),
            col_widths=[5, 4, 3],
        ),
        ui.div(ui.tags.small(
            "Target: whether a paper reaches the top 25% of citations-per-year. Using velocity "
            "(not raw citations) controls for paper age. Features: references, team size, "
            "openness, topic, sector and venue.", class_="text-muted"), class_="px-2"),
    )


def mlnlp_server(input, output, session):

    @reactive.calc
    def clustering():
        return mlnlp.cluster_titles(int(input.nlp_k()))

    @reactive.effect
    def _sync_clusters():
        k = int(input.nlp_k())
        ui.update_select("nlp_cluster", choices=[str(c) for c in range(k)], selected="0")

    @render.text
    def nlp_n():
        return f"{clustering()['n']:,}"

    @render.text
    def nlp_sil():
        return f"{clustering()['silhouette']:.2f}"

    @render_widget
    def nlp_scatter():
        cl = clustering()
        coords = cl["coords"]
        fig = go.Figure()
        for c in sorted(coords["cluster"].unique()):
            sub = coords[coords["cluster"] == c]
            fig.add_trace(go.Scattergl(
                x=sub["x"], y=sub["y"], mode="markers", name=f"Cluster {c}",
                marker=dict(size=5, opacity=0.6,
                            color=theme.PALETTE[int(c) % len(theme.PALETTE)]),
                customdata=sub[["title", "year", "citations"]],
                hovertemplate="<b>%{customdata[0]}</b><br>Year %{customdata[1]} · "
                              "%{customdata[2]:,} citations<extra></extra>"))
        fig.update_xaxes(title_text="SVD component 1")
        fig.update_yaxes(title_text="SVD component 2")
        return theme.style(fig, height=440)

    @render_widget
    def nlp_sizes():
        cl = clustering()
        sizes = cl["sizes"]
        fig = go.Figure(go.Pie(
            labels=[f"Cluster {c}" for c in sizes.index], values=sizes.values,
            hole=0.5, marker=dict(colors=[theme.PALETTE[int(c) % len(theme.PALETTE)]
                                          for c in sizes.index])))
        fig.update_traces(textinfo="percent")
        return theme.style(fig, height=440)

    @render_widget
    def nlp_terms():
        cl = clustering()
        try:
            c = int(input.nlp_cluster())
        except (ValueError, TypeError):
            c = 0
        terms = cl["top_terms"].get(c, [])[::-1]
        if not terms:
            return theme.empty_figure("No terms")
        weights = list(range(1, len(terms) + 1))
        fig = go.Figure(go.Bar(x=weights, y=terms, orientation="h",
                               marker_color=theme.PALETTE[c % len(theme.PALETTE)]))
        fig.update_xaxes(title_text="Relative importance (rank)")
        return theme.style(fig, height=320)

    @render_widget
    def fc_chart():
        fc = mlnlp.forecast_publications(int(input.fc_cutoff()), int(input.fc_horizon()))
        hist = fc["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers",
                                 name="Actual", line=dict(color=theme.ACCENT, width=3)))
        f = fc["forecast"]
        if not f.empty:
            up, lo = fc["upper"], fc["lower"]
            fig.add_trace(go.Scatter(x=list(up.index) + list(lo.index[::-1]),
                                     y=list(up.values) + list(lo.values[::-1]),
                                     fill="toself", fillcolor="rgba(245,158,11,0.18)",
                                     line=dict(width=0), name="95% band", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=f.index, y=f.values, mode="lines+markers",
                                     name="Forecast",
                                     line=dict(color=theme.PALETTE[2], width=3, dash="dash")))
            fig.add_vline(x=fc["cutoff"] + 0.5, line=dict(color=theme.MUTED, dash="dot"))
        fig.update_yaxes(title_text="Papers")
        return theme.style(fig, height=380)

    @render.text
    def ml_auc():
        return f"{mlnlp.citation_model()['auc']:.2f}"

    @render.text
    def ml_acc():
        return f"{mlnlp.citation_model()['accuracy'] * 100:.0f}%"

    @render_widget
    def ml_importance():
        imp = mlnlp.citation_model()["importance"].iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color=theme.ACCENT))
        fig.update_xaxes(title_text="Importance")
        return theme.style(fig, height=380)

    @render_widget
    def ml_roc():
        m = mlnlp.citation_model()
        fpr, tpr = m["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(color=theme.MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {m['auc']:.2f}",
                                 fill="tozeroy", line=dict(color=theme.ACCENT, width=2)))
        fig.update_xaxes(title_text="False positive rate", range=[0, 1])
        fig.update_yaxes(title_text="True positive rate", range=[0, 1])
        return theme.style(fig, height=380)

    @render_widget
    def ml_confusion():
        cm = mlnlp.citation_model()["confusion"]
        labels = ["Not high", "High"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, colorscale=theme.SEQUENTIAL,
            text=cm, texttemplate="%{text}", showscale=False,
            hovertemplate="Predicted %{x}<br>Actual %{y}: %{z}<extra></extra>"))
        fig.update_xaxes(title_text="Predicted")
        fig.update_yaxes(title_text="Actual", autorange="reversed")
        return theme.style(fig, height=380)
