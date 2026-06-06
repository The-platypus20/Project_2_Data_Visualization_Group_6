"""Shared Plotly styling so every chart looks consistent across tabs."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

PALETTE = [
    "#7CC9FF", "#7BE0B5", "#FFD27A", "#FF8CA1", "#B79CFF",
    "#88E1FF", "#FFB778", "#9FD97D", "#C084FC", "#94A3B8",
]

TOPIC_FAMILY_ORDER = [
    "Core ML / Deep Learning",
    "NLP",
    "Computer Vision",
    "ML Theory & Optimization",
    "Reinforcement Learning",
    "Robotics",
    "Data Mining & Pattern Recognition",
    "AI Ethics & Society",
    "Other",
]

SEQUENTIAL = "Blues"
ACCENT = "#7CC9FF"
MUTED = "#9FB2CC"
GRID = "rgba(255,255,255,0.08)"
TEXT = "#EAF2FF"
SUBTLE_TEXT = "#9FB2CC"
SURFACE = "rgba(13,25,46,0.88)"
PAGE_BG = "#07111F"

_TEMPLATE = go.layout.Template()
_TEMPLATE.layout = go.Layout(
    font=dict(
        family='Aptos, "Segoe UI", Inter, Helvetica, Arial, sans-serif',
        size=12,
        color=TEXT,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,25,46,0.28)",
    colorway=PALETTE,
    margin=dict(l=60, r=20, t=70, b=50),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        color=TEXT,
        automargin=True,
        title_standoff=10,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        color=TEXT,
        automargin=True,
        title_standoff=10,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="left",
        x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color=TEXT),
    ),
    hoverlabel=dict(
        bgcolor="#050C18",
        font_size=12,
        font_color=TEXT,
        bordercolor="rgba(255,255,255,0.14)",
    ),
    title=dict(
        font=dict(size=14, color=TEXT),
        x=0.0,
        xanchor="left",
        y=0.97,
        yanchor="top",
    ),
)
pio.templates["g6"] = _TEMPLATE

DASHBOARD_CSS = """
/* AI Observatory global dark theme */

:root {
    --bs-body-bg: #07111F;
    --bs-body-color: #EAF2FF;
    --bs-primary: #7CC9FF;
    --bs-secondary-color: #9FB2CC;
    --bs-border-color: rgba(255,255,255,0.10);
    --bs-link-color: #BFE4FF;
    --bs-link-hover-color: #FFFFFF;
    --bs-font-sans-serif: Aptos, "Segoe UI", Inter, Helvetica, Arial, sans-serif;

    --ai-bg: #07111F;
    --ai-bg-2: #0A1730;
    --ai-panel: rgba(13,25,46,0.88);
    --ai-panel-2: rgba(17,32,57,0.94);
    --ai-panel-3: rgba(23,40,68,0.82);
    --ai-border: rgba(255,255,255,0.10);
    --ai-border-2: rgba(124,201,255,0.22);
    --ai-text: #EAF2FF;
    --ai-muted: #9FB2CC;
    --ai-muted-2: #7E91AD;
    --ai-accent: #7CC9FF;
    --ai-blue: #4A99FF;
    --ai-green: #7BE0B5;
    --ai-yellow: #FFD27A;
    --ai-rose: #FF8CA1;
    --ai-violet: #B79CFF;
}

html,
body,
.bslib-page-dashboard,
.container-fluid,
.bslib-sidebar-layout,
.bslib-sidebar-layout > .main,
.sidebar {
    background:
        radial-gradient(circle at top left, rgba(96,142,255,0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(78,224,190,0.10), transparent 30%),
        linear-gradient(180deg, #07111F 0%, #0A1730 48%, #07111F 100%) !important;
    color: var(--ai-text) !important;
    font-family: Aptos, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
}

.container-fluid {
    max-width: 1480px;
    margin-inline: auto;
}

.navbar {
    background: rgba(7,17,31,0.92) !important;
    border-bottom: 1px solid var(--ai-border) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18) !important;
    backdrop-filter: blur(12px);
}

.navbar-brand,
.navbar .nav-link,
.navbar .navbar-text {
    color: var(--ai-text) !important;
    font-weight: 750;
}

.navbar-brand {
    font-size: clamp(1.05rem, 1.4vw, 1.45rem);
}

.navbar .nav-link {
    border-radius: 999px;
    padding-inline: 0.85rem;
    margin-inline: 2px;
}

.navbar .nav-link:hover,
.navbar .nav-link:focus {
    background: rgba(124,201,255,0.10) !important;
    color: #FFFFFF !important;
}

.navbar .nav-link.active,
.navbar-nav .show > .nav-link,
.nav-tabs .nav-link.active {
    background: rgba(124,201,255,0.18) !important;
    color: #FFFFFF !important;
    box-shadow: inset 0 0 0 1px rgba(124,201,255,0.22);
}

.nav-tabs .nav-link {
    color: var(--ai-muted) !important;
    font-weight: 650;
}

.tab-heading,
.growth-tab {
    background: transparent !important;
    color: var(--ai-text) !important;
}

.tab-heading {
    padding: clamp(.8rem, 1.4vw, 1.25rem) clamp(.7rem, 1.5vw, 1.1rem) .55rem;
}

.tab-heading h2,
.growth-tab h2,
.chart-title,
.card-header,
.card-header *,
.story-hero h1 {
    color: var(--ai-text) !important;
}

.tab-heading h2 {
    font-size: clamp(1.55rem, 2.2vw, 2.15rem);
    line-height: 1.08;
    font-weight: 850;
    margin: 0 0 .25rem;
}

.tab-insight,
.chart-subtitle,
.text-muted,
.small,
.growth-tab .tab-insight,
.growth-tab .chart-subtitle,
.growth-tab .text-muted,
.story-section-label {
    color: var(--ai-muted) !important;
}

.tab-insight {
    font-size: clamp(.92rem, 1.05vw, 1.08rem);
    max-width: 72rem;
    margin: 0;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin-bottom: .45rem;
}

.truth-badge,
.growth-tab .truth-badge {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(124,201,255,0.18) !important;
    background: rgba(124,201,255,0.10) !important;
    color: #BFE4FF !important;
    border-radius: 999px;
    padding: .16rem .55rem;
    font-size: .72rem;
    font-weight: 750;
    white-space: nowrap;
}

.card,
.bslib-card,
.growth-tab .card {
    background: var(--ai-panel) !important;
    border: 1px solid var(--ai-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 50px rgba(0,0,0,0.22) !important;
    color: var(--ai-text) !important;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.card-header,
.growth-tab .card-header {
    background: rgba(17,32,57,0.72) !important;
    border-bottom: 1px solid var(--ai-border) !important;
    color: var(--ai-text) !important;
    font-weight: 750;
}

.card-footer,
.growth-tab .card-footer {
    background: rgba(17,32,57,0.48) !important;
    border-top: 1px solid var(--ai-border) !important;
    color: var(--ai-muted) !important;
    font-size: .85rem;
}

.card-body {
    min-width: 0;
    overflow: hidden;
}

.hero-card .card-body {
    min-height: clamp(430px, 62vh, 720px);
}

.bslib-value-box,
.value-box {
    min-height: 118px;
    background: var(--ai-panel) !important;
    border: 1px solid var(--ai-border) !important;
    border-left: 4px solid var(--ai-accent) !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 50px rgba(0,0,0,0.18) !important;
}

.bslib-value-box .value-box-value,
.bslib-value-box .value-box-title,
.value-box-value,
.value-box-title {
    color: var(--ai-text) !important;
}

.bslib-value-box .value-box-value {
    font-size: 1.55rem;
    line-height: 1.15;
}

.bslib-value-box .value-box-title {
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: uppercase;
}

.bslib-value-box .small,
.metric-note {
    color: var(--ai-muted) !important;
}

.side-stack {
    display: grid;
    gap: 1rem;
    align-content: start;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .65rem;
    margin-bottom: .8rem;
}

.metric-grid.one-wide {
    grid-template-columns: 1fr;
}

.metric,
.growth-tab .breakpoint-card,
.breakpoint-card {
    background: rgba(255,255,255,0.055) !important;
    border: 1px solid var(--ai-border) !important;
    border-radius: 14px !important;
    color: var(--ai-text) !important;
}

.metric {
    padding: .65rem .7rem;
    min-width: 0;
}

.metric-label,
.breakpoint-note,
.breakpoint-label,
.panel-label {
    color: var(--ai-muted) !important;
}

.metric-label {
    font-size: .68rem;
    font-weight: 780;
    text-transform: uppercase;
}

.metric-value,
.breakpoint-year {
    color: var(--ai-text) !important;
}

.metric-value {
    font-size: 1.05rem;
    font-weight: 760;
    line-height: 1.15;
    overflow-wrap: anywhere;
}

.breakpoint-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: .75rem;
    width: 100%;
}

.breakpoint-card,
.growth-tab .breakpoint-card {
    display: block;
    width: 100%;
    min-height: 118px;
    text-align: left;
    padding: 0;
    box-shadow: none;
}

.growth-tab .breakpoint-card-inner {
    padding: .85rem;
}

.breakpoint-year {
    font-size: 1.35rem;
    font-weight: 900;
    line-height: 1;
}

.breakpoint-count,
.growth-tab .breakpoint-count {
    margin-top: .35rem;
    color: var(--ai-accent) !important;
    font-weight: 850;
}

.growth-tab .breakpoint-card:hover,
.growth-tab .breakpoint-card:focus {
    background: rgba(124,201,255,0.10) !important;
    border-color: rgba(124,201,255,0.28) !important;
    color: var(--ai-text) !important;
}

.growth-tab .breakpoint-card.selected {
    background: rgba(124,201,255,0.16) !important;
    border-color: rgba(124,201,255,0.52) !important;
    box-shadow: 0 0 0 1px rgba(124,201,255,0.18) !important;
}

.breakpoint-insight,
.interpretation {
    background: rgba(124,201,255,0.08) !important;
    border: 1px solid rgba(124,201,255,0.18) !important;
    border-left: 3px solid var(--ai-accent) !important;
    border-radius: 8px;
    color: #D8EFFF !important;
    padding: .65rem .75rem;
    font-size: .9rem;
    line-height: 1.35;
    margin: .35rem 0 .75rem;
}

.form-label,
.control-label,
.form-check-label,
.checkbox label,
.radio label,
.shiny-input-container label,
.sidebar h5,
.sidebar .control-label,
.btn,
.action-button {
    color: var(--ai-text) !important;
}

.sidebar h5 {
    font-weight: 750;
    margin-bottom: 1rem;
}

.sidebar .shiny-input-container {
    margin-bottom: 1.35rem;
}

.form-control,
.form-select,
.selectize-input,
.selectize-dropdown {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: var(--ai-text) !important;
}

.selectize-dropdown-content,
.selectize-dropdown .option {
    background: #0B1730 !important;
    color: var(--ai-text) !important;
}

.form-control:focus,
.form-select:focus,
.selectize-input.focus {
    border-color: var(--ai-accent) !important;
    box-shadow: 0 0 0 .18rem rgba(124,201,255,0.16) !important;
}

.btn,
.action-button {
    border-radius: 12px !important;
    font-weight: 850 !important;
}

.btn-outline-secondary {
    color: var(--ai-text) !important;
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

.btn-outline-secondary:hover,
.btn-outline-secondary:focus {
    color: #06111F !important;
    background: var(--ai-accent) !important;
    border-color: var(--ai-accent) !important;
}

.irs,
.irs-grid-text {
    color: var(--ai-muted) !important;
}

.irs--shiny .irs-line {
    background: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.10) !important;
}

.irs--shiny .irs-bar {
    background: linear-gradient(90deg, var(--ai-blue), var(--ai-accent)) !important;
    border-color: transparent !important;
}

.irs--shiny .irs-handle {
    border-color: var(--ai-accent) !important;
    background: #FFFFFF !important;
}

.irs--shiny .irs-from,
.irs--shiny .irs-to,
.irs--shiny .irs-single {
    background: var(--ai-accent) !important;
    color: #06111F !important;
}

.shiny-data-grid,
.table,
table {
    color: var(--ai-text) !important;
}

.paper-list,
.gap-panel ul {
    padding-left: 1.1rem;
    margin-bottom: 0;
}

.paper-list li,
.gap-panel li {
    color: var(--ai-text) !important;
    margin-bottom: .55rem;
    line-height: 1.35;
}

.plotly,
.js-plotly-plot {
    background: transparent !important;
}

.story-hero {
    margin: 1rem 1rem .8rem;
    min-height: 300px;
    border-radius: 18px;
    padding: 2rem;
    color: #ffffff;
    background:
        radial-gradient(circle at 84% 24%, rgba(125,211,252,.22), transparent 28%),
        linear-gradient(135deg, #06111f 0%, #0d2a48 56%, #07111f 100%);
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(320px, .75fr);
    gap: 1rem;
    box-shadow: 0 20px 54px rgba(0,0,0,.22);
}

.story-hero p {
    color: #C8D9EF !important;
    max-width: 820px;
    font-size: 1.02rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.hero-eyebrow {
    color: #9CC7EF;
    font-size: .74rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: .7rem;
}

.hero-mini-panel,
.hero-signal,
.hero-year-control,
.hero-data-note {
    background: rgba(6,17,31,.68) !important;
    border: 1px solid rgba(255,255,255,.16) !important;
    color: var(--ai-text) !important;
}

.hero-scene {
    background:
        radial-gradient(circle at 22% 80%, rgba(5,150,105,.18), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,.05), rgba(14,165,233,.09)) !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    border-radius: 14px;
}

@media (max-width: 1100px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }

    .story-hero {
        grid-template-columns: 1fr;
        padding: 1.35rem;
    }

    .story-hero h1 {
        font-size: 2.05rem;
    }
}

@media (max-width: 900px) {
    .metric-grid,
    .breakpoint-grid {
        grid-template-columns: 1fr;
    }

    .growth-header-row {
        display: block;
    }
}
"""

def style(fig: go.Figure, *, title: str | None = None, height: int | None = None) -> go.Figure:
    """Apply the shared template plus optional title/height in one call."""
    fig.update_layout(template="g6")
    if title is not None:
        fig.update_layout(title=title)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color=SUBTLE_TEXT),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        template="g6",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig
