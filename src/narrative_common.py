"""Small shared UI/data helpers for the narrative dashboard tabs."""
from __future__ import annotations

from shiny import ui

from . import theme


def family_colors(values):
    families = list(dict.fromkeys(values))
    return {fam: theme.PALETTE[i % len(theme.PALETTE)] for i, fam in enumerate(families)}


def topic_family_colors(values):
    families = list(dict.fromkeys(values))
    ordered = [fam for fam in theme.TOPIC_FAMILY_ORDER if fam in families]
    ordered += [fam for fam in families if fam not in ordered]
    return {
        fam: ("#94a3b8" if fam == "Other" else theme.PALETTE[i % len(theme.PALETTE)])
        for i, fam in enumerate(ordered)
    }


def metric(label, value, note=None):
    return ui.div(
        ui.div(label, class_="metric-label"),
        ui.div(value, class_="metric-value"),
        ui.div(note or "", class_="metric-note"),
        class_="metric",
    )


def badge(text):
    return ui.span(text, class_="truth-badge")


def card_header(title, subtitle=None):
    return ui.card_header(
        ui.div(
            ui.div(title, class_="chart-title"),
            ui.div(subtitle or "", class_="chart-subtitle"),
        )
    )


def notice(text):
    return ui.card_footer(
        ui.span(ui.tags.strong("What to notice: "), text, class_="text-muted small")
    )


def section_label(text):
    return ui.div(text, class_="story-section-label")


def paper_list(records, *, topic=False, limit=4):
    if hasattr(records, "to_dict"):
        records = records.to_dict("records")
    if not records:
        return ui.p(
            "Paper lookup cache not built or no top papers found for this selection.",
            class_="text-muted small",
        )
    items = []
    for rec in records[:limit]:
        meta = f"{int(rec.get('year', 2019))}"
        if "fwci" in rec and rec.get("fwci") == rec.get("fwci"):
            meta += f" | FWCI {float(rec.get('fwci', 0)):,.2f}"
        elif "citation_velocity" in rec and rec.get("citation_velocity") == rec.get("citation_velocity"):
            meta += f" | velocity {float(rec.get('citation_velocity', 0)):,.1f}/yr"
        elif "citation_count" in rec:
            meta += f" | {float(rec.get('citation_count', 0)):,.0f} citations"
        if topic and rec.get("primary_topic"):
            meta += f" | {rec['primary_topic']}"
        items.append(
            ui.tags.li(
                ui.tags.strong(str(rec.get("title", "Untitled"))),
                ui.br(),
                ui.span(meta, class_="text-muted small"),
            )
        )
    return ui.tags.ul(*items, class_="paper-list")

