"""Shared Plotly styling so every chart looks consistent across tabs."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Qualitative palette (color-blind friendly, ordered for topic stacks).
PALETTE = [
    "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#9333ea", "#94a3b8",
]
SEQUENTIAL = "Blues"
ACCENT = "#2563eb"
MUTED = "#94a3b8"
GRID = "#e5e7eb"

_TEMPLATE = go.layout.Template()
_TEMPLATE.layout = go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
              size=12, color="#1f2937"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    colorway=PALETTE,
    # Generous top margin so the horizontal legend never sits on the title;
    # automargin on both axes prevents tick/title overlap on narrow cards.
    margin=dict(l=60, r=20, t=70, b=50),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID, ticks="outside",
               tickcolor=GRID, automargin=True, title_standoff=10),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
               automargin=True, title_standoff=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
    title=dict(font=dict(size=14, color="#111827"), x=0.0, xanchor="left",
               y=0.97, yanchor="top"),
)
pio.templates["g6"] = _TEMPLATE

# CSS injected once in app.py to fix value-box crowding and tighten cards.
DASHBOARD_CSS = """
.bslib-value-box .value-box-value { font-size: 1.55rem; line-height: 1.15; }
.bslib-value-box .value-box-title { font-size: .8rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .02em; color: #6b7280; }
.bslib-value-box { min-height: 118px; }
.card-header { font-weight: 600; font-size: .95rem; }
.shiny-data-grid { font-size: .85rem; }
.nav-tabs .nav-link { font-weight: 600; }
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
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color=MUTED), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(template="g6", xaxis=dict(visible=False),
                      yaxis=dict(visible=False), height=300)
    return fig
