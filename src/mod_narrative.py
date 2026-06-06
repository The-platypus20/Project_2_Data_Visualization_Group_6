"""Narrative dashboard entry point.

The three tabs live in separate modules so each section can be edited or
parallelized without waking old render outputs from another tab.
"""
from __future__ import annotations

from .tab_how_ai_grew import growth_server, growth_ui
from .tab_where_ideas_moved import movement_server, movement_ui
from .tab_what_created_impact import impact_server, impact_ui

def narrative_ui():
    return [
        growth_ui(),
        movement_ui(),
        impact_ui(),
    ]


def narrative_server(input, output, session):
    growth_server(input, output, session)
    movement_server(input, output, session)
    impact_server(input, output, session)

