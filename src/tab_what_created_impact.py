"""Tab 3: "What drives impact".

A machine-learning tour of what separates a high-impact AI paper from the rest.
Heavy compute lives offline in src/preprocess/build_impact_ml_cache.py; this tab
only reads the small tab3_*.csv cache files (see src/tab3_data.py), so every
chart renders instantly.

Impact label used everywhere on this tab: a paper is "high impact" if its
citation_velocity (citations per year) sits in the TOP 10% of its own
publication year. Ranking within each year removes the age bias that would
otherwise punish recent papers (a 2024 paper has had less time to collect
citations than a 2008 paper).

Four beats:
  1. How concentrated is impact?   citation funnel
  2. Which traits drive impact?    standardized logistic-regression tornado
  3. Can we predict it?            gradient boosting ROC
  4. Where is impact heading?      LSTM forecast of every family's share
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import render, ui
from shinywidgets import output_widget, render_widget

from . import theme
from . import tab3_data as t3
from .narrative_common import badge, card_header, metric, notice

ACCENT = theme.ACCENT          # blue
GOOD = "#7BE0B5"               # green  -> raises the odds of impact
BAD = "#FF8CA1"                # rose   -> lowers the odds of impact
MUTED = "rgba(159,178,204,0.45)"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _section(num: str, text: str) -> ui.Tag:
    """A large, readable section heading (bigger than the default label)."""
    return ui.div(
        ui.span(num + " · ", style=f"color:{ACCENT};"),
        text,
        class_="story-section-label tab3-section-title",
        style="font-size:1.5rem; font-weight:850; margin:1.6rem 0 .6rem; letter-spacing:.01em;",
    )


def _taskline(*body) -> ui.Tag:
    """States the prediction problem the models on beats 2-3 are solving."""
    return ui.div(
        ui.tags.strong("The task — "), *body,
        class_="tab3-taskline",
        style="font-size:1.08rem; margin:.1rem .2rem .8rem; padding:.55rem .85rem; border-radius:8px; line-height:1.5;",
    )


def _explain(title: str, *body) -> ui.Tag:
    """A themed explanation box (reuses the .interpretation style, enlarged)."""
    return ui.div(
        ui.span(ui.tags.strong(title + " "), *body),
        class_="interpretation tab3-explain",
        style="font-size:1.04rem; line-height:1.55; padding:.85rem 1rem;",
    )


def _flow() -> ui.Tag:
    """A top-to-bottom roadmap of the four beats on this tab."""
    steps = [
        ("1", "How concentrated is impact?", "citation funnel"),
        ("2", "Which traits drive impact?", "logistic-regression drivers"),
        ("3", "Can we predict it?", "gradient-boosting model"),
        ("4", "Where is impact heading?", "LSTM forecast"),
    ]
    rows = []
    for i, (num, question, method) in enumerate(steps):
        rows.append(ui.div(
            ui.span(num, style="display:inline-grid; place-items:center; min-width:30px; height:30px; "
                               "border-radius:50%; background:#7CC9FF; color:#06111F; font-weight:850;"),
            ui.div(
                ui.span(question, style="font-weight:800; color:#EAF2FF; font-size:1.06rem;"),
                ui.span("  →  " + method, style="color:#9FB2CC; font-size:.96rem;"),
            ),
            style="display:flex; align-items:center; gap:.75rem; padding:.3rem .1rem;",
        ))
        if i < len(steps) - 1:
            rows.append(ui.div("↓", style="color:#7CC9FF; font-size:1.05rem; margin:0 0 0 13px; line-height:1;"))
    return ui.div(
        ui.div("What this tab does, step by step",
               style="font-weight:850; color:#EAF2FF; font-size:1.02rem; margin-bottom:.55rem;"),
        *rows,
        ui.div("Impact = a paper in the top 10% citation velocity of its own publication year.",
               style="color:#9FB2CC; font-size:.9rem; margin-top:.6rem; "
                     "border-top:1px solid rgba(255,255,255,0.08); padding-top:.5rem;"),
        class_="interpretation",
        style="padding:.95rem 1.1rem;",
    )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _family_palette(families: list[str]) -> dict[str, str]:
    return {fam: theme.PALETTE[i % len(theme.PALETTE)] for i, fam in enumerate(families)}


def _forecast_families() -> list[str]:
    """Families ordered by their average recent share of high-impact papers."""
    df = t3.forecast()
    if df.empty or "family" not in df:
        return []
    df = df.copy()
    df["share"] = pd.to_numeric(df["share"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    recent = df[df["year"] <= 2025]
    order = recent.groupby("family")["share"].mean().sort_values(ascending=False).index.tolist()
    return order or sorted(df["family"].dropna().unique().tolist())


def _pct(x, digits: int = 1) -> str:
    return "N/A" if x is None or pd.isna(x) else f"{float(x):.{digits}f}%"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def impact_ui():
    fam_choices = _forecast_families()
    return ui.nav_panel(
        "What drives impact",
        ui.div(
            # ---- header -------------------------------------------------- #
            ui.div(
                ui.div(
                    ui.h2("What drives impact"),
                    ui.p(
                        "Most AI papers are barely cited; a tiny minority shape the field. "
                        "Here we dissect what a high-impact paper is made of — and train "
                        "machine-learning models to see whether impact can be explained and predicted.",
                        class_="tab-insight",
                    ),
                ),
                ui.div(
                    badge("OpenAlex 2000-2025"),
                    badge("Machine learning"),
                    badge("Top-10% citation velocity"),
                    class_="badge-row",
                ),
                class_="growth-header-row",
            ),
            _flow(),
            ui.output_ui("tab3_headline_kpis"),

            # ---- Beat 1: concentration ---------------------------------- #
            _section("1", "How concentrated is impact?"),
            ui.card(
                card_header(
                    "From all papers down to the rare giants",
                    "How many papers survive each citation threshold.",
                ),
                output_widget("tab3_funnel"),
                notice("The highly-cited tiers are so rare their bars almost vanish — the white callouts give the exact counts."),
            ),
            _explain("A few proportion of papers hold most of the citations"),

            # ---- Beat 2: drivers (logistic regression) ------------------ #
            _section("2", "Which traits drive impact?"),
            _taskline(
                "is this paper in the ",
                ui.tags.strong("top 10% citation velocity of its own publication year?"),
                " The model learns this yes/no answer from traits known at publication time "
                "(no citation data), then we read off which traits matter most.",
            ),
            ui.card(
                card_header(
                    "Standardized logistic-regression drivers",
                    "Each bar is one trait's effect on the odds of being high impact, holding the others fixed.",
                ),
                output_widget("tab3_drivers"),
            ),
            _explain(
                "How this model works —",
                "a ",
                ui.tags.strong("logistic regression"),
                " weighs all traits together; each standardized bar is that trait's own pull on the odds of "
                "high impact (longer = stronger, green = raises, rose = lowers).",
            ),

            # ---- Beat 3: prediction (gradient boosting) ----------------- #
            _section("3", "Can we predict it?"),
            _taskline(
                "the same yes/no question — ",
                ui.tags.strong("is this paper in the top 10% citation velocity of its own publication year?"),
                " — but here we measure how accurately the model predicts it on papers it has never seen.",
            ),
            ui.layout_columns(
                ui.card(
                    card_header(
                        "ROC curve — ranking power on unseen papers",
                        "Tested on a 20% hold-out set the model never trained on.",
                    ),
                    output_widget("tab3_roc"),
                ),
                ui.card(
                    card_header(
                        "Model comparison",
                        "Three models scored on the same hold-out set.",
                    ),
                    ui.output_ui("tab3_model_cards"),
                    notice("Gradient boosting ranks impact best; baseline is random guessing at the 10% rate."),
                ),
                col_widths=[6, 6],
            ),
            _explain(
                "How this model works —",
                ui.tags.strong("gradient boosting (LightGBM)"),
                " stacks hundreds of small decision trees. ",
                ui.tags.strong("ROC-AUC"),
                " ≈ chance it ranks a high-impact paper above an ordinary one (0.5 = guessing, 1.0 = perfect); ",
                ui.tags.strong("lift@10%"),
                " = how many more hits you get from its top 10% versus random.",
            ),

            # ---- Beat 4: forecast (LSTM) -------------------------------- #
            _section("4", "Where is impact heading?"),
            ui.card(
                card_header(
                    "Forecasting each family's share of high-impact papers",
                    "All families at once; pick one to highlight its 2026-2028 LSTM forecast and uncertainty band.",
                ),
                ui.input_select("tab3_family", "Highlight a research family",
                                choices=fam_choices, selected=(fam_choices[0] if fam_choices else None)),
                output_widget("tab3_forecast"),
            ),
            _explain(
                "How this model works —",
                "an ",
                ui.tags.strong("LSTM (Long Short-Term Memory) neural network"),
                " reads each family's recent history as a sequence and learns the temporal pattern to predict the "
                "next year. We forecast each family's ",
                ui.tags.strong("share of all high-impact papers"),
                " (not raw counts) because the most recent years are still accumulating citations — shares are "
                "far more stable than absolute impact numbers near the present. The shaded band is an approximate "
                "80% confidence range that widens further into the future.",
            ),
            ui.div(
                ui.tags.small(
                    "Models: scikit-learn (logistic regression), LightGBM (gradient boosting), PyTorch (LSTM). "
                    "All trained offline; this tab reads only the cached results.",
                    class_="text-muted",
                ),
                style="margin:.3rem .25rem 1rem;",
            ),
            class_="growth-tab",
        ),
    )


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
def impact_server(input, output, session):

    # ---- headline KPIs ---------------------------------------------------- #
    @render.ui
    def tab3_headline_kpis():
        c = t3.concentration()
        mm = t3.model_metrics().copy()
        roc = np.nan
        if not mm.empty and "roc_auc" in mm:
            mm["roc_auc"] = pd.to_numeric(mm["roc_auc"], errors="coerce")
            roc = float(mm["roc_auc"].max())
        never = c.get("never_cited_pct")
        top1 = c.get("top1pct_citation_share")
        return ui.div(
            ui.div(ui.div("Never cited", class_="kpi-label"), ui.div(_pct(never, 1), class_="kpi-value"), ui.div("Papers with zero citations", class_="kpi-note"), class_="kpi-card"),
            ui.div(ui.div("Top 1% citation share", class_="kpi-label"), ui.div(_pct(top1, 0), class_="kpi-value"), ui.div("Citations held by top 1%", class_="kpi-note"), class_="kpi-card"),
            ui.div(ui.div("Best model ROC-AUC", class_="kpi-label"), ui.div(f"{roc:.3f}" if not pd.isna(roc) else "N/A", class_="kpi-value"), ui.div("Ranking power on unseen papers", class_="kpi-note"), class_="kpi-card"),
            class_="kpi-grid tab3-kpi-grid",
        )

    # ---- Beat 1: Funnel --------------------------------------------------- #
    @render_widget
    def tab3_funnel():
        df = t3.funnel().copy()
        if df.empty:
            return theme.empty_figure("Funnel cache is empty.")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        total = float(df["count"].iloc[0]) if len(df) else 1.0

        fig = go.Figure(go.Funnel(
            y=df["stage"], x=df["count"],
            textposition="inside", textinfo="value+percent initial",
            marker=dict(color=theme.PALETTE[:len(df)],
                        line=dict(color="rgba(255,255,255,0.18)", width=1)),
            connector=dict(line=dict(color=MUTED, width=1)),
            hovertemplate="%{y}<br>%{x:,.0f} papers (%{percentInitial:.2%} of all)<extra></extra>"))

        # The ≥100 and ≥1000 tiers are so tiny their bars vanish: add white
        # callouts with an arrow pointing at the (near-zero-width) bar.
        for stage in ["Cited ≥ 100", "Cited ≥ 1000"]:
            row = df[df["stage"] == stage]
            if row.empty:
                continue
            cnt = float(row["count"].iloc[0])
            pct = 100.0 * cnt / max(total, 1)
            fig.add_annotation(
                x=0.5, xref="paper", xanchor="left",
                y=stage, yref="y",
                text=f"<b>{cnt:,.0f}</b> papers ({pct:.2f}%)",
                showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1.6,
                arrowcolor="#FFFFFF", ax=110, ay=0,
                font=dict(color="#FFFFFF", size=13),
                bgcolor="rgba(6,17,31,0.78)", bordercolor="rgba(255,255,255,0.35)", borderwidth=1,
            )
        fig.update_layout(margin=dict(l=20, r=28, t=20, b=20))
        return theme.style(fig, height=340)

    # ---- Beat 2: drivers -------------------------------------------------- #
    @render_widget
    def tab3_drivers():
        df = t3.drivers().copy()
        if df.empty:
            return theme.empty_figure("Driver cache is empty.")
        df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
        df = df.dropna(subset=["coef"]).sort_values("coef")
        top_idx = df["coef"].abs().idxmax()
        colors = [ACCENT if idx == top_idx else "rgba(148,163,184,0.58)" for idx in df.index]
        fig = go.Figure(go.Bar(
            x=df["coef"], y=df["feature"], orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.16)", width=1)),
            hovertemplate="<b>%{y}</b><br>coefficient %{x:+.2f}<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_width=1.2)
        fig.update_xaxes(title_text="Effect on log-odds of high impact (standardized)", showgrid=True, gridcolor=theme.GRID, zeroline=False)
        fig.update_yaxes(title_text="")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=18, t=18, b=44))
        return theme.style(fig, height=360)

    # ---- Beat 3: model cards --------------------------------------------- #
    @render.ui
    def tab3_model_cards():
        df = t3.model_metrics().copy()
        if df.empty:
            return ui.div("Model-metrics cache is empty.", class_="text-muted small")
        for col in ["roc_auc", "pr_auc", "lift_at_10"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Fixed display order: best model first, random baseline last.
        order = ["gradient boosting", "logistic regression", "baseline"]
        df["__o"] = df["model"].apply(
            lambda m: next((i for i, o in enumerate(order) if o in str(m).lower()), len(order)))
        df = df.sort_values("__o")
        cards = []
        for _, r in df.iterrows():
            cards.append(metric(
                str(r["model"]),
                f"AUC {r['roc_auc']:.3f}",
                f"PR-AUC {r['pr_auc']:.3f} · lift@10 {r['lift_at_10']:.2f}×",
            ))
        return ui.div(*cards, class_="metric-grid one-wide")

    # ---- Beat 3: ROC ------------------------------------------------------ #
    @render_widget
    def tab3_roc():
        df = t3.roc_curve().copy()
        if df.empty:
            return theme.empty_figure("ROC cache is empty.")
        df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
        df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")
        mm = t3.model_metrics().copy()
        auc = np.nan
        if not mm.empty:
            mm["roc_auc"] = pd.to_numeric(mm["roc_auc"], errors="coerce")
            gbm = mm[mm["model"].str.contains("oost", case=False, na=False)]
            auc = float(gbm["roc_auc"].iloc[0]) if not gbm.empty else float(mm["roc_auc"].max())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color=MUTED, width=1.6, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["fpr"], y=df["tpr"], mode="lines", name="Gradient boosting",
                                 line=dict(color=ACCENT, width=3), fill="tozeroy",
                                 fillcolor="rgba(124,201,255,0.10)",
                                 hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        if not pd.isna(auc):
            fig.add_annotation(x=0.62, y=0.18, text=f"ROC-AUC = {auc:.3f}", showarrow=False,
                               font=dict(size=15, color=theme.TEXT),
                               bgcolor="rgba(6,17,31,0.7)", bordercolor="rgba(255,255,255,0.16)", borderwidth=1)
        fig.update_xaxes(title_text="False positive rate", range=[0, 1], showgrid=True, gridcolor=theme.GRID)
        fig.update_yaxes(title_text="True positive rate", range=[0, 1])
        fig.update_layout(showlegend=True, margin=dict(l=56, r=18, t=20, b=44))
        return theme.style(fig, height=330)

    # ---- Beat 4: forecast (all families, highlight one) ------------------ #
    @render_widget
    def tab3_forecast():
        df = t3.forecast().copy()
        if df.empty:
            return theme.empty_figure("Forecast cache is empty — run build_impact_ml_cache.py")
        for col in ["year", "share", "lo", "hi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        families = _forecast_families()
        palette = _family_palette(families)
        sel = input.tab3_family() or (families[0] if families else None)

        fig = go.Figure()
        # Draw non-selected families first (faded), selected last so it sits on top.
        for fam in [f for f in families if f != sel] + ([sel] if sel in families else []):
            sub = df[df["family"] == fam].sort_values("year")
            if sub.empty:
                continue
            hist = sub[sub["kind"] == "history"]
            fore = sub[sub["kind"] == "forecast"]
            # Bridge history into forecast for a continuous line.
            if not hist.empty and not fore.empty:
                fore = pd.concat([hist.iloc[[-1]].assign(kind="forecast"), fore], ignore_index=True)
            color = palette.get(fam, ACCENT)
            is_sel = fam == sel

            if is_sel:
                # confidence band
                if not fore.empty and fore["hi"].notna().any():
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fore["year"], fore["year"][::-1]]),
                        y=pd.concat([fore["hi"], fore["lo"][::-1]]),
                        fill="toself", fillcolor=_rgba(color, 0.16),
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=hist["year"], y=hist["share"], mode="lines+markers", name=fam,
                    legendgroup=fam, line=dict(color=color, width=4), marker=dict(size=6, color=color),
                    hovertemplate=f"<b>{fam}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
                if not fore.empty:
                    fig.add_trace(go.Scatter(
                        x=fore["year"], y=fore["share"], mode="lines+markers", name=f"{fam} (forecast)",
                        legendgroup=fam, showlegend=False,
                        line=dict(color=color, width=4, dash="dash"), marker=dict(size=6, color=color),
                        hovertemplate=f"<b>{fam}</b><br>%{{x}}: %{{y:.1f}}% (forecast)<extra></extra>"))
            else:
                full = pd.concat([hist, fore[fore["kind"] == "forecast"]], ignore_index=True).sort_values("year")
                fig.add_trace(go.Scatter(
                    x=full["year"], y=full["share"], mode="lines", name=fam, legendgroup=fam,
                    line=dict(color=_rgba(color, 0.32), width=1.6),
                    hovertemplate=f"{fam}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))

        fig.add_vline(x=2025, line_dash="dot", line_color=MUTED, line_width=1,
                      annotation_text="last data (2025) · forecast →", annotation_position="top",
                      annotation_font=dict(color=theme.SUBTLE_TEXT, size=11))
        fig.update_xaxes(title_text="", tickmode="linear", dtick=4)
        fig.update_yaxes(title_text="Share of high-impact papers (%)", showgrid=True, gridcolor=theme.GRID, rangemode="tozero")
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
                          margin=dict(l=56, r=18, t=44, b=34))
        return theme.style(fig, height=420)
