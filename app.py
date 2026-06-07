"""AI research narrative dashboard.

Run with: shiny run app.py --reload
"""
from __future__ import annotations

from shiny import App, ui

from src import theme
from src.mod_narrative import narrative_server, narrative_ui

BACK_TO_TOP_CSS = """
#back-to-top {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 9999;
  opacity: 0;
  transform: translateY(10px);
  pointer-events: none;
  border: 1px solid rgba(124, 201, 255, 0.32);
  border-radius: 999px;
  padding: 10px 14px;
  color: #06111F;
  background: linear-gradient(180deg, #BFE4FF, #7CC9FF);
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.26);
  font-size: 12px;
  font-weight: 850;
  letter-spacing: 0.02em;
  transition: opacity 160ms ease, transform 160ms ease;
}
#back-to-top.is-visible {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}
#back-to-top:hover {
  transform: translateY(-2px);
}
"""

BACK_TO_TOP_JS = """
(function () {
  function bindBackToTop() {
    var btn = document.getElementById("back-to-top");
    if (!btn || btn.dataset.bound) return;
    btn.dataset.bound = "1";
    function sync() {
      if (window.scrollY > 420) btn.classList.add("is-visible");
      else btn.classList.remove("is-visible");
    }
    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    window.addEventListener("scroll", sync, { passive: true });
    sync();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindBackToTop);
  } else {
    bindBackToTop();
  }
  document.addEventListener("shiny:connected", bindBackToTop);
})();
"""


NAV_POLISH_CSS = """
.navbar {
  padding-top: 0.7rem !important;
  padding-bottom: 0.65rem !important;
}
.navbar-brand {
  font-weight: 900 !important;
  letter-spacing: -0.02em !important;
}
.navbar-nav {
  gap: 0.45rem !important;
}
.navbar-nav .nav-link {
  margin: 0 0.08rem !important;
  padding: 0.56rem 0.92rem !important;
  border-radius: 14px !important;
  color: #9FB2CC !important;
  font-weight: 760 !important;
  transition: background 140ms ease, color 140ms ease, box-shadow 140ms ease, transform 140ms ease !important;
}
.navbar-nav .nav-link:hover {
  color: #EAF2FF !important;
  background: rgba(124, 201, 255, 0.08) !important;
}
.navbar-nav .nav-link.active,
.navbar-nav .show > .nav-link {
  color: #06111F !important;
  background: linear-gradient(180deg, #BFE4FF, #7CC9FF) !important;
  box-shadow: 0 10px 24px rgba(124, 201, 255, 0.18) !important;
}
.tab-heading,
.growth-header-row {
  min-height: 116px;
  padding: 1.15rem 1.25rem !important;
  border-radius: 24px !important;
  background: rgba(13,25,46,0.78) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: 0 18px 50px rgba(0,0,0,0.18) !important;
}
.tab-heading h2,
.growth-header-row h2 {
  margin: 0 0 0.32rem !important;
  color: #EAF2FF !important;
  font-size: clamp(1.55rem, 2.2vw, 2.05rem) !important;
  font-weight: 920 !important;
  letter-spacing: -0.04em !important;
}
.tab-insight {
  color: #9FB2CC !important;
  font-size: 1rem !important;
  line-height: 1.45 !important;
  max-width: 980px !important;
}
"""



SIDEBAR_UX_CSS = """
/* Sidebar navigation + theme polish */
body {
  transition: background 180ms ease, color 180ms ease;
}

@media (min-width: 992px) {
  body {
    padding-left: 248px;
  }

  .navbar {
    position: fixed !important;
    inset: 0 auto 0 0 !important;
    width: 248px !important;
    height: 100vh !important;
    z-index: 1040 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    padding: 1.1rem 0.9rem !important;
    border-right: 1px solid rgba(124, 201, 255, 0.18) !important;
    border-bottom: 0 !important;
    background:
      radial-gradient(circle at 18% 8%, rgba(124,201,255,0.16), transparent 30%),
      linear-gradient(180deg, rgba(7,17,31,0.98), rgba(10,23,48,0.98)) !important;
    box-shadow: 18px 0 54px rgba(0,0,0,0.22) !important;
    backdrop-filter: blur(14px);
  }

  .navbar > .container-fluid,
  .navbar > .container,
  .navbar-collapse,
  .navbar-nav {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
  }

  .navbar-brand {
    white-space: normal !important;
    line-height: 1.04 !important;
    font-size: 1.1rem !important;
    margin: 0 0 1.2rem 0 !important;
    padding: 0.35rem 0.35rem 1rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.12) !important;
  }

  .navbar-brand::before {
    content: "AI";
    display: grid;
    place-items: center;
    width: 48px;
    height: 48px;
    margin-bottom: 0.75rem;
    border-radius: 16px;
    color: #06111F;
    font-weight: 950;
    letter-spacing: -0.06em;
    background: linear-gradient(180deg, #BFE4FF, #7CC9FF);
    box-shadow: 0 14px 32px rgba(124,201,255,0.24);
  }

  .navbar-nav::before {
    content: "NAVIGATE";
    color: #7E91AD;
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    margin: 0.4rem 0.45rem 0.65rem;
  }

  .navbar-nav .nav-link {
    width: 100% !important;
    text-align: left !important;
    margin: 0.12rem 0 !important;
    padding: 0.72rem 0.82rem !important;
    border-radius: 16px !important;
    border: 1px solid transparent !important;
  }

  .navbar-nav .nav-link.active,
  .navbar-nav .show > .nav-link {
    color: #06111F !important;
    background: linear-gradient(180deg, #DFF2FF, #7CC9FF) !important;
    box-shadow: 0 12px 28px rgba(124,201,255,0.22) !important;
  }

  .navbar .sidebar-actions {
    width: 100%;
    margin-top: auto;
    padding: 1rem 0.25rem 0;
    border-top: 1px solid rgba(255,255,255,0.12);
    display: grid;
    gap: 0.55rem;
  }

  .sidebar-actions-title {
    color: #7E91AD;
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    margin: 0.3rem 0.2rem 0.1rem;
  }

  .sidebar-action-btn {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.075);
    color: #EAF2FF;
    font-size: 0.82rem;
    font-weight: 850;
    padding: 0.62rem 0.72rem;
    text-align: left;
    transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
  }

  .sidebar-action-btn:hover {
    transform: translateY(-1px);
    background: rgba(124,201,255,0.13);
    border-color: rgba(124,201,255,0.30);
  }

  .theme-segmented-control {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
    padding: 0.28rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
  }

  .theme-choice-btn {
    border: 0;
    border-radius: 12px;
    padding: 0.56rem 0.45rem;
    color: #9FB2CC;
    background: transparent;
    font-size: 0.8rem;
    font-weight: 880;
    text-align: center;
    transition: background 140ms ease, color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
  }

  .theme-choice-btn:hover {
    color: #EAF2FF;
    background: rgba(124,201,255,0.10);
    transform: translateY(-1px);
  }

  .theme-choice-btn.is-active {
    color: #06111F;
    background: linear-gradient(180deg, #BFE4FF, #7CC9FF);
    box-shadow: 0 10px 24px rgba(124,201,255,0.22);
  }

  .container-fluid {
    max-width: 1520px;
  }
}

/* Hero banner: gradient-only background, no image dependency */
.tab-heading,
.growth-header-row {
  position: relative;
  overflow: hidden;
  isolation: isolate;
  background:
    radial-gradient(circle at 82% 18%, rgba(124,201,255,0.20), transparent 28%),
    radial-gradient(circle at 12% 86%, rgba(74,153,255,0.14), transparent 30%),
    linear-gradient(135deg, rgba(7,17,31,0.96), rgba(13,42,72,0.92) 58%, rgba(7,17,31,0.96)) !important;
}

.tab-heading::after,
.growth-header-row::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  opacity: 0.22;
  background-image:
    linear-gradient(rgba(191,228,255,0.14) 1px, transparent 1px),
    linear-gradient(90deg, rgba(191,228,255,0.14) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at 74% 26%, black, transparent 70%);
}

.metadata-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.42rem;
  margin-top: 0.65rem;
}

.metadata-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.34rem;
  border-radius: 999px;
  padding: 0.28rem 0.62rem;
  background: rgba(124,201,255,0.10);
  border: 1px solid rgba(124,201,255,0.20);
  color: #BFE4FF;
  font-size: 0.74rem;
  font-weight: 820;
  white-space: nowrap;
}

.metadata-chip strong {
  color: #F8FBFF;
  font-weight: 900;
}

/* Light theme keeps the same blue identity */
body.ai-light-theme,
body.ai-light-theme .bslib-page-dashboard,
body.ai-light-theme .container-fluid,
body.ai-light-theme .bslib-sidebar-layout,
body.ai-light-theme .bslib-sidebar-layout > .main,
body.ai-light-theme .sidebar {
  --bs-body-bg: #F5F9FF;
  --bs-body-color: #0F172A;
  --ai-bg: #F5F9FF;
  --ai-bg-2: #EAF3FF;
  --ai-panel: rgba(255,255,255,0.94);
  --ai-panel-2: rgba(248,251,255,0.96);
  --ai-panel-3: rgba(239,246,255,0.88);
  --ai-border: rgba(15,23,42,0.10);
  --ai-border-2: rgba(37,99,235,0.18);
  --ai-text: #0F172A;
  --ai-muted: #475569;
  --ai-muted-2: #64748B;
  --ai-accent: #2563EB;
  color: #0F172A !important;
  background:
    radial-gradient(circle at top left, rgba(37,99,235,0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 30%),
    linear-gradient(180deg, #F5F9FF 0%, #EAF3FF 48%, #F8FBFF 100%) !important;
}

body.ai-light-theme .navbar {
  background:
    radial-gradient(circle at 20% 8%, rgba(37,99,235,0.12), transparent 30%),
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(236,246,255,0.96)) !important;
  border-right-color: rgba(37,99,235,0.18) !important;
  box-shadow: 18px 0 54px rgba(15,23,42,0.10) !important;
}

body.ai-light-theme .navbar-brand,
body.ai-light-theme .navbar .nav-link,
body.ai-light-theme .navbar .navbar-text,
body.ai-light-theme .tab-heading h2,
body.ai-light-theme .growth-header-row h2,
body.ai-light-theme .chart-title,
body.ai-light-theme .card-header,
body.ai-light-theme .card-header *,
body.ai-light-theme .form-label,
body.ai-light-theme .control-label,
body.ai-light-theme .form-check-label,
body.ai-light-theme .shiny-input-container label {
  color: #0F172A !important;
}

body.ai-light-theme .navbar-nav .nav-link {
  color: #475569 !important;
}

body.ai-light-theme .navbar-nav .nav-link:hover {
  color: #0F172A !important;
  background: rgba(37,99,235,0.08) !important;
}

body.ai-light-theme .navbar-nav .nav-link.active,
body.ai-light-theme .navbar-nav .show > .nav-link {
  color: #FFFFFF !important;
  background: linear-gradient(180deg, #60A5FA, #2563EB) !important;
}

body.ai-light-theme .card,
body.ai-light-theme .bslib-card,
body.ai-light-theme .growth-tab .card,
body.ai-light-theme .bslib-value-box,
body.ai-light-theme .value-box,
body.ai-light-theme .metric,
body.ai-light-theme .breakpoint-card {
  background: rgba(255,255,255,0.94) !important;
  border-color: rgba(15,23,42,0.10) !important;
  box-shadow: 0 18px 48px rgba(15,23,42,0.10) !important;
  color: #0F172A !important;
}

body.ai-light-theme .card-header,
body.ai-light-theme .growth-tab .card-header,
body.ai-light-theme .card-footer,
body.ai-light-theme .growth-tab .card-footer {
  background: rgba(239,246,255,0.80) !important;
  border-color: rgba(15,23,42,0.09) !important;
}

body.ai-light-theme .tab-insight,
body.ai-light-theme .chart-subtitle,
body.ai-light-theme .text-muted,
body.ai-light-theme .small,
body.ai-light-theme .metric-label,
body.ai-light-theme .metric-note,
body.ai-light-theme .card-footer,
body.ai-light-theme .growth-tab .card-footer,
body.ai-light-theme .navbar-nav::before,
body.ai-light-theme .sidebar-actions-title {
  color: #475569 !important;
}

body.ai-light-theme .tab-heading,
body.ai-light-theme .growth-header-row {
  background:
    radial-gradient(circle at 82% 18%, rgba(37,99,235,0.16), transparent 28%),
    radial-gradient(circle at 12% 86%, rgba(14,165,233,0.12), transparent 30%),
    linear-gradient(135deg, rgba(255,255,255,0.96), rgba(235,246,255,0.94) 58%, rgba(255,255,255,0.96)) !important;
}

body.ai-light-theme .metadata-chip,
body.ai-light-theme .truth-badge,
body.ai-light-theme .growth-tab .truth-badge {
  color: #1D4ED8 !important;
  background: rgba(37,99,235,0.08) !important;
  border-color: rgba(37,99,235,0.18) !important;
}

body.ai-light-theme .metadata-chip strong {
  color: #0F172A;
}

body.ai-light-theme .sidebar-action-btn {
  color: #0F172A;
  background: rgba(37,99,235,0.06);
  border-color: rgba(37,99,235,0.14);
}

body.ai-light-theme .theme-segmented-control {
  background: rgba(37,99,235,0.06);
  border-color: rgba(37,99,235,0.14);
}

body.ai-light-theme .theme-choice-btn {
  color: #475569;
}

body.ai-light-theme .theme-choice-btn:hover {
  color: #0F172A;
  background: rgba(37,99,235,0.08);
}

body.ai-light-theme .theme-choice-btn.is-active {
  color: #FFFFFF;
  background: linear-gradient(180deg, #60A5FA, #2563EB);
  box-shadow: 0 10px 24px rgba(37,99,235,0.20);
}

body.ai-light-theme .form-control,
body.ai-light-theme .form-select,
body.ai-light-theme .selectize-input,
body.ai-light-theme .selectize-dropdown {
  background: rgba(255,255,255,0.94) !important;
  border-color: rgba(15,23,42,0.12) !important;
  color: #0F172A !important;
}

@media (max-width: 991px) {
  .navbar .sidebar-actions {
    display: none;
  }
}
"""

SIDEBAR_UX_JS = """
(function () {
  function setPressed(id, active) {
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("is-active", active);
    el.setAttribute("aria-pressed", active ? "true" : "false");
  }

  function relayoutPlotlyTheme(isLight) {
    if (!window.Plotly) return;
    var layout = isLight ? {
      "paper_bgcolor": "rgba(0,0,0,0)",
      "plot_bgcolor": "rgba(0,0,0,0)",
      "font.color": "#334155",
      "legend.font.color": "#334155",
      "xaxis.color": "#475569",
      "yaxis.color": "#475569",
      "xaxis.gridcolor": "#E2E8F0",
      "yaxis.gridcolor": "#E2E8F0",
      "yaxis2.color": "#475569",
      "yaxis2.gridcolor": "rgba(0,0,0,0)"
    } : {
      "paper_bgcolor": "rgba(0,0,0,0)",
      "plot_bgcolor": "rgba(0,0,0,0)",
      "font.color": "#EAF2FF",
      "legend.font.color": "#EAF2FF",
      "xaxis.color": "#EAF2FF",
      "yaxis.color": "#EAF2FF",
      "xaxis.gridcolor": "rgba(255,255,255,0.08)",
      "yaxis.gridcolor": "rgba(255,255,255,0.08)",
      "yaxis2.color": "#EAF2FF",
      "yaxis2.gridcolor": "rgba(0,0,0,0)"
    };
    document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
      try {
        window.Plotly.relayout(plot, layout);
        window.Plotly.restyle(plot, {
          "textfont.color": isLight ? "#0F172A" : "#EAF2FF",
          "marker.line.color": isLight ? "rgba(15,23,42,0.18)" : "rgba(255,255,255,0.55)"
        });
      } catch (e) {}
    });
  }

  function applyTheme(theme) {
    var normalized = theme === "light" ? "light" : "dark";
    var isLight = normalized === "light";
    document.body.classList.toggle("ai-light-theme", isLight);
    document.documentElement.classList.toggle("ai-light-theme", isLight);
    setPressed("ai-theme-dark", !isLight);
    setPressed("ai-theme-light", isLight);
    try { localStorage.setItem("ai-dashboard-theme", normalized); } catch (e) {}
    setTimeout(function () { relayoutPlotlyTheme(isLight); }, 80);
    setTimeout(function () { relayoutPlotlyTheme(isLight); }, 500);
  }

  function schedulePlotlyRestyle(delay) {
    var isLight = document.body.classList.contains("ai-light-theme");
    setTimeout(function () { relayoutPlotlyTheme(isLight); }, delay || 60);
  }

  document.addEventListener("shiny:value", function () { schedulePlotlyRestyle(60); });
  document.addEventListener("plotly_afterplot", function () { schedulePlotlyRestyle(30); });
  document.addEventListener("shown.bs.tab", function () { schedulePlotlyRestyle(80); });
  window.addEventListener("resize", function () { schedulePlotlyRestyle(120); });

  function mountSidebarActions() {
    var nav = document.querySelector(".navbar");
    if (!nav || nav.querySelector(".sidebar-actions")) return;
    var container = nav.querySelector(".container-fluid") || nav.querySelector(".container") || nav;
    var actions = document.createElement("div");
    actions.className = "sidebar-actions";
    actions.innerHTML = '<div class="sidebar-actions-title">DISPLAY</div>' +
      '<div class="theme-segmented-control" role="group" aria-label="Theme selector">' +
      '<button type="button" class="theme-choice-btn" id="ai-theme-dark" aria-pressed="true">Dark</button>' +
      '<button type="button" class="theme-choice-btn" id="ai-theme-light" aria-pressed="false">Light</button>' +
      '</div>';
    container.appendChild(actions);

    document.getElementById("ai-theme-dark").addEventListener("click", function () {
      applyTheme("dark");
    });
    document.getElementById("ai-theme-light").addEventListener("click", function () {
      applyTheme("light");
    });

    var saved = "dark";
    try { saved = localStorage.getItem("ai-dashboard-theme") || "dark"; } catch (e) {}
    applyTheme(saved);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountSidebarActions);
  } else {
    mountSidebarActions();
  }
  document.addEventListener("shiny:connected", mountSidebarActions);
})();
"""

app_ui = ui.page_navbar(
    *narrative_ui(),
    title="AI Research Growth & Concentration",
    id="navbar",
    fillable=False,
    header=ui.head_content(
        ui.tags.style(theme.DASHBOARD_CSS),
        ui.tags.style(BACK_TO_TOP_CSS),
        ui.tags.style(NAV_POLISH_CSS),
        ui.tags.style(SIDEBAR_UX_CSS),
        ui.tags.script(BACK_TO_TOP_JS),
        ui.tags.script(SIDEBAR_UX_JS),
    ),
    footer=ui.div(
        ui.tags.small(
            "Data: precomputed OpenAlex dashboard cache files in Dataset/dashboard_cache/. "
            "Topic groups are metadata-derived from OpenAlex topic labels. Raw OpenAlex exports are not required at runtime.",
            class_="text-muted",
        ),
        ui.tags.button("↑ Top", id="back-to-top", type="button", title="Back to top"),
        class_="px-3 py-2",
    ),
)


def server(input, output, session):
    narrative_server(input, output, session)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
WWW_DIR = PROJECT_ROOT / "www"

app = App(app_ui, server, static_assets=WWW_DIR)
