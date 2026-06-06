(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const MIN_YEAR = 2000;
  const MAX_YEAR = 2025;
  const VIEWBOX = { width: 1080, height: 680 };

  const FAMILY_RADIUS_MIN = 28;
  const FAMILY_RADIUS_MAX = 116;
  const SUBTOPIC_RADIUS_MIN = 7;
  const SUBTOPIC_RADIUS_MAX = 34;
  const TRANSITION_MS = 420;

  const PARTICLE_PAPER_UNIT = 2800;
  const PARTICLE_MAX_PER_FAMILY = 150;
  const PARTICLE_MIN_VISIBLE = 6;

  const FALLBACK_COLORS = [
    "#7cc9ff", "#7be0b5", "#ffd27a", "#ff8ca1", "#b79cff",
    "#88e1ff", "#ffb778", "#9fd97d", "#c084fc", "#94a3b8"
  ];

  const state = {
    families: [],
    subtopics: [],
    selectedFamily: "",
    selectedTopic: "",
    yearStart: MIN_YEAR,
    yearEnd: MAX_YEAR,
    timer: null,
    previousFamilyR: new Map(),
    previousSubtopicR: new Map(),
    previousParticleCounts: new Map(),
  };

  function $(id) {
    return document.getElementById(id);
  }

  function svgEl(name, attrs) {
    const el = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value !== null && value !== undefined) el.setAttribute(key, String(value));
    });
    return el;
  }

  function sendInput(id, value) {
    if (window.Shiny && typeof window.Shiny.setInputValue === "function") {
      window.Shiny.setInputValue(id, value, { priority: "event" });
    }
  }

  function fmt(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "";
    return Math.round(n).toLocaleString("en-US");
  }

  function metric(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "";
    return n.toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function normalizeColor(d, i) {
    return d.color || d.colour || d.family_color || FALLBACK_COLORS[i % FALLBACK_COLORS.length];
  }

  function cumulativeAtYear(d, year) {
    const counts = d.cumulative_paper_count_by_year || d.counts_by_year || {};
    return Number(counts[String(year)] || counts[year] || 0);
  }

  function countThrough(d, startYear, endYear) {
    const start = Math.min(startYear, endYear);
    const end = Math.max(startYear, endYear);
    const endCount = cumulativeAtYear(d, end);
    const beforeStart = start > MIN_YEAR ? cumulativeAtYear(d, start - 1) : 0;
    return Math.max(0, endCount - beforeStart);
  }

  function radiusFor(count, minR, maxR, maxCount) {
    if (!count || !maxCount) return minR;
    return minR + (Math.sqrt(count) / Math.sqrt(maxCount)) * (maxR - minR);
  }

  function labelFor(value, maxLen) {
    const text = String(value || "");
    return text.length > maxLen ? `${text.slice(0, maxLen - 1)}…` : text;
  }

  function hashUnit(seed) {
    let h = 2166136261;
    const text = String(seed);
    for (let i = 0; i < text.length; i += 1) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) / 4294967295;
  }

  function hexToRgb(hex) {
    const clean = String(hex || "").replace("#", "").trim();
    if (!/^[0-9a-fA-F]{6}$/.test(clean)) return { r: 124, g: 201, b: 255 };
    const value = parseInt(clean, 16);
    return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
  }

  function rgba(hex, alpha) {
    const { r, g, b } = hexToRgb(hex);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function liftedColor(hex, lift = 42) {
    const { r, g, b } = hexToRgb(hex);
    return `rgb(${Math.min(255, r + lift)}, ${Math.min(255, g + lift)}, ${Math.min(255, b + lift)})`;
  }

  function particleCountFor(count) {
    if (count <= 0) return 0;
    return Math.max(
      PARTICLE_MIN_VISIBLE,
      Math.min(PARTICLE_MAX_PER_FAMILY, Math.round(count / PARTICLE_PAPER_UNIT))
    );
  }

  function particlePositions(seed, radius, count, dotR) {
    const points = [];
    const maxR = Math.max(0, radius * 0.72);
    const minGap = dotR * 2.55;
    const maxAttempts = 90;

    for (let i = 0; i < count; i += 1) {
      let placed = false;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        const a = hashUnit(`${seed}:a:${i}:${attempt}`) * Math.PI * 2;
        const rr = Math.sqrt(hashUnit(`${seed}:r:${i}:${attempt}`)) * maxR;
        const x = Math.cos(a) * rr;
        const y = Math.sin(a) * rr;
        const ok = points.every((p) => Math.hypot(x - p.x, y - p.y) >= minGap);
        if (ok) {
          points.push({ x, y });
          placed = true;
          break;
        }
      }
      if (!placed) break;
    }
    return points;
  }

  function addRadiusAnimation(circle, fromR, toR) {
    const from = Number.isFinite(Number(fromR)) ? Number(fromR) : Number(toR);
    const to = Number(toR);
    circle.setAttribute("r", String(from));
    if (!Number.isFinite(to) || Math.abs(from - to) < 0.25) {
      circle.setAttribute("r", String(to));
      return;
    }
    const anim = svgEl("animate", {
      attributeName: "r",
      from,
      to,
      dur: `${TRANSITION_MS}ms`,
      fill: "freeze",
      calcMode: "spline",
      keySplines: ".2 .8 .2 1",
    });
    circle.appendChild(anim);
  }

  function hideLoading() {
    const loading = $("landscape-loading");
    if (loading) loading.style.display = "none";
  }

  function showError(message) {
    hideLoading();
    const shell = $("landscape-shell");
    if (!shell) return;
    shell.querySelectorAll(".landscape-error").forEach((el) => el.remove());
    const error = document.createElement("div");
    error.className = "landscape-error";
    error.textContent = message;
    shell.appendChild(error);
  }

  function tooltipHtml(type, d) {
    const count = countThrough(d, state.yearStart, state.yearEnd);
    const title = type === "family" ? d.family : d.topic;
    const parts = [
      `<strong>${title}</strong>`,
      `${fmt(count)} papers in ${state.yearStart}–${state.yearEnd}`,
    ];
    if (type === "subtopic") parts.push(`Family: ${d.family}`);
    if (Number.isFinite(Number(d.median_fwci))) parts.push(`Median FWCI: ${metric(d.median_fwci)}`);
    if (Number.isFinite(Number(d.citation_velocity))) parts.push(`Citation velocity: ${metric(d.citation_velocity)}/yr`);
    if (Number.isFinite(Number(d.growth_rate))) parts.push(`Growth rate: ${metric(d.growth_rate)}`);
    return parts.join("<br>");
  }

  function showTooltip(event, html) {
    const tip = $("landscape-tooltip");
    const shell = $("landscape-shell");
    if (!tip || !shell) return;
    tip.innerHTML = html;
    const bounds = shell.getBoundingClientRect();
    const x = clamp(event.clientX - bounds.left + 14, 10, Math.max(10, bounds.width - 310));
    const y = clamp(event.clientY - bounds.top + 14, 10, Math.max(10, bounds.height - 120));
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
    tip.classList.add("show");
  }

  function hideTooltip() {
    const tip = $("landscape-tooltip");
    if (tip) tip.classList.remove("show");
  }

  function updateYearRangeLabel() {
    const label = $("landscape-year-range-value");
    if (label) label.textContent = `${state.yearStart}–${state.yearEnd}`;
  }

  function updateYearRangeFill() {
    const fill = $("landscape-range-fill");
    if (!fill) return;
    const total = MAX_YEAR - MIN_YEAR;
    const startPct = ((state.yearStart - MIN_YEAR) / total) * 100;
    const endPct = ((state.yearEnd - MIN_YEAR) / total) * 100;
    fill.style.left = `${startPct}%`;
    fill.style.width = `${Math.max(0, endPct - startPct)}%`;
  }

  function setYearRange(startYear, endYear) {
    let start = clamp(Number(startYear) || MIN_YEAR, MIN_YEAR, MAX_YEAR);
    let end = clamp(Number(endYear) || MAX_YEAR, MIN_YEAR, MAX_YEAR);
    if (start > end) [start, end] = [end, start];

    state.yearStart = start;
    state.yearEnd = end;

    const startInput = $("landscape-year-start");
    const endInput = $("landscape-year-end");
    if (startInput && Number(startInput.value) !== start) startInput.value = String(start);
    if (endInput && Number(endInput.value) !== end) endInput.value = String(end);

    updateYearRangeLabel();
    updateYearRangeFill();

    sendInput("landscape_year_start", start);
    sendInput("landscape_year_end", end);
    sendInput("landscape_year_current", end);
    sendInput("landscape_year_range", { start, end });

    render();
  }

  function setSelectedFamily(family) {
    state.selectedFamily = family || "";
    state.selectedTopic = "";
    state.previousParticleCounts = new Map();
    sendInput("landscape_family_click", state.selectedFamily);
    sendInput("landscape_topic_click", "");
    render();
  }

  function setSelectedTopic(d) {
    state.selectedFamily = d.family || "";
    state.selectedTopic = d.topic || "";
    state.previousParticleCounts = new Map();
    sendInput("landscape_family_click", state.selectedFamily);
    sendInput("landscape_topic_click", state.selectedTopic);
    render();
  }

  function resetView() {
    stopPlay();
    state.selectedFamily = "";
    state.selectedTopic = "";
    state.previousFamilyR = new Map();
    state.previousSubtopicR = new Map();
    state.previousParticleCounts = new Map();
    sendInput("landscape_reset", Date.now());
    sendInput("landscape_family_click", "");
    sendInput("landscape_topic_click", "");
    setYearRange(MIN_YEAR, MAX_YEAR);
  }

  function stopPlay() {
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    const btn = $("landscape-play");
    if (btn) btn.textContent = "Play";
  }

  function togglePlay() {
    if (state.timer) {
      stopPlay();
      return;
    }
    const btn = $("landscape-play");
    if (btn) btn.textContent = "Pause";

    if (state.yearEnd >= MAX_YEAR) setYearRange(state.yearStart, state.yearStart);

    state.timer = window.setInterval(() => {
      if (state.yearEnd >= MAX_YEAR) {
        stopPlay();
        return;
      }
      setYearRange(state.yearStart, state.yearEnd + 1);
    }, 850);
  }

  function familyOpacity(family) {
    if (!state.selectedFamily) return 1;
    return family === state.selectedFamily ? 1 : 0.18;
  }

  function subtopicKey(d) {
    return `${d.family}::${d.topic}`;
  }

  function selectedSubtopicRecord() {
    if (!state.selectedFamily || !state.selectedTopic) return null;
    return state.subtopics.find((d) => d.family === state.selectedFamily && d.topic === state.selectedTopic) || null;
  }

  function particleAnimationStart(centerX, centerY, endX, endY, seed, index, mode) {
    const dx = endX - centerX;
    const dy = endY - centerY;
    const baseScale = mode === "topic" ? 0.35 : 0.16;
    const jitter = (hashUnit(`${seed}:move:${index}`) - 0.5) * 10;
    const angle = Math.atan2(dy, dx) + Math.PI / 2;
    return {
      x: centerX + dx * baseScale + Math.cos(angle) * jitter,
      y: centerY + dy * baseScale + Math.sin(angle) * jitter,
    };
  }

  function buildDefs(svg) {
    const defs = svgEl("defs");

    const glow = svgEl("filter", {
      id: "landscapeGlow",
      x: "-45%",
      y: "-45%",
      width: "190%",
      height: "190%",
    });
    glow.appendChild(svgEl("feGaussianBlur", { stdDeviation: "5.5", result: "blur" }));
    const merge = svgEl("feMerge");
    merge.appendChild(svgEl("feMergeNode", { in: "blur" }));
    merge.appendChild(svgEl("feMergeNode", { in: "SourceGraphic" }));
    glow.appendChild(merge);
    defs.appendChild(glow);

    const paperGlow = svgEl("filter", {
      id: "paperDotGlow",
      x: "-60%",
      y: "-60%",
      width: "220%",
      height: "220%",
    });
    paperGlow.appendChild(svgEl("feGaussianBlur", { stdDeviation: "2.2", result: "blur" }));
    const merge2 = svgEl("feMerge");
    merge2.appendChild(svgEl("feMergeNode", { in: "blur" }));
    merge2.appendChild(svgEl("feMergeNode", { in: "SourceGraphic" }));
    paperGlow.appendChild(merge2);
    defs.appendChild(paperGlow);

    svg.appendChild(defs);
  }

  function drawStars(svg) {
    const stars = svgEl("g", { class: "landscape-stars", opacity: "0.34" });
    for (let i = 0; i < 95; i += 1) {
      const x = (i * 137) % VIEWBOX.width;
      const y = (i * 89) % VIEWBOX.height;
      stars.appendChild(svgEl("circle", {
        cx: x,
        cy: y,
        r: 0.65 + ((i * 7) % 16) / 13,
        fill: "rgba(203,213,225,.48)",
      }));
    }
    svg.appendChild(stars);
  }

  function drawFamilyNode(layer, d, index, maxFamilyCount, newFamilyR, newParticleCounts) {
    const familyCount = countThrough(d, state.yearStart, state.yearEnd);
    if (familyCount <= 0) return;

    const color = normalizeColor(d, index);
    const r = radiusFor(familyCount, FAMILY_RADIUS_MIN, FAMILY_RADIUS_MAX, maxFamilyCount);
    const selected = d.family === state.selectedFamily;
    const opacity = familyOpacity(d.family);
    const previousR = state.previousFamilyR.get(d.family) ?? r;
    newFamilyR.set(d.family, r);

    const focusedTopic = selected ? selectedSubtopicRecord() : null;
    const topicCount = focusedTopic ? countThrough(focusedTopic, state.yearStart, state.yearEnd) : 0;
    const particleMode = focusedTopic && topicCount > 0 ? "topic" : "family";
    const particleSeed = particleMode === "topic" ? `${d.family}::${focusedTopic.topic}` : d.family;
    const particleBasisCount = particleMode === "topic" ? topicCount : familyCount;
    const particleCount = particleCountFor(particleBasisCount);
    const particleKey = particleMode === "topic" ? particleSeed : d.family;
    const previousParticleCount = state.previousParticleCounts.get(particleKey) || 0;
    newParticleCounts.set(particleKey, particleCount);

    const group = svgEl("g", {
      class: `family-node${selected ? " is-selected" : ""}`,
      opacity,
    });

    const outer = svgEl("circle", {
      class: "family-halo",
      cx: d.x,
      cy: d.y,
      fill: color,
      "fill-opacity": selected ? 0.17 : 0.075,
      stroke: color,
      "stroke-opacity": selected ? 0.9 : 0.26,
      "stroke-width": selected ? 3.2 : 1.35,
      filter: selected ? "url(#landscapeGlow)" : null,
    });
    addRadiusAnimation(outer, previousR + 18, r + 18);

    const bubble = svgEl("circle", {
      class: "family-bubble",
      cx: d.x,
      cy: d.y,
      fill: color,
      "fill-opacity": selected ? 0.34 : 0.19,
      stroke: selected ? "#fef3c7" : color,
      "stroke-opacity": selected ? 0.98 : 0.58,
      "stroke-width": selected ? 3.6 : 1.8,
    });
    addRadiusAnimation(bubble, previousR, r);

    group.appendChild(outer);
    group.appendChild(bubble);

    const particleLayer = svgEl("g", {
      class: `family-particle-layer ${particleMode === "topic" ? "is-topic-filtered" : "is-family-total"}`,
    });
    const dotR = particleMode === "topic" ? 3.05 : (selected ? 3.2 : 2.9);
    const spreadRadius = particleMode === "topic" ? Math.max(34, r * 0.48) : r;
    const focusX = particleMode === "topic" ? d.x + (Number(focusedTopic.x) - Number(d.x)) * 0.48 : Number(d.x);
    const focusY = particleMode === "topic" ? d.y + (Number(focusedTopic.y) - Number(d.y)) * 0.48 : Number(d.y);
    const targetOpacity = selected ? 0.58 : 0.38;
    const delayCap = particleMode === "topic" ? 0.5 : 0.75;

    particlePositions(particleSeed, spreadRadius, particleCount, dotR).forEach((pos, i) => {
      const isNew = i >= previousParticleCount || particleMode === "topic";
      const endX = focusX + pos.x;
      const endY = focusY + pos.y;
      const start = particleAnimationStart(focusX, focusY, endX, endY, particleSeed, i, particleMode);
      const delay = `${Math.min(delayCap, i * 0.005)}s`;

      const dot = svgEl("circle", {
        class: "family-paper-particle",
        cx: isNew ? start.x : endX,
        cy: isNew ? start.y : endY,
        r: isNew ? 0.25 : dotR,
        fill: liftedColor(color),
        "fill-opacity": isNew ? 0 : targetOpacity,
        filter: "url(#paperDotGlow)",
      });

      if (isNew) {
        dot.appendChild(svgEl("animate", {
          attributeName: "cx",
          from: start.x,
          to: endX,
          dur: particleMode === "topic" ? "560ms" : "680ms",
          begin: delay,
          fill: "freeze",
          calcMode: "spline",
          keySplines: ".2 .8 .2 1",
        }));
        dot.appendChild(svgEl("animate", {
          attributeName: "cy",
          from: start.y,
          to: endY,
          dur: particleMode === "topic" ? "560ms" : "680ms",
          begin: delay,
          fill: "freeze",
          calcMode: "spline",
          keySplines: ".2 .8 .2 1",
        }));
        dot.appendChild(svgEl("animate", {
          attributeName: "r",
          from: 0.25,
          to: dotR,
          dur: particleMode === "topic" ? "440ms" : "540ms",
          begin: delay,
          fill: "freeze",
          calcMode: "spline",
          keySplines: ".2 .8 .2 1",
        }));
        dot.appendChild(svgEl("animate", {
          attributeName: "fill-opacity",
          from: 0,
          to: targetOpacity,
          dur: "360ms",
          begin: delay,
          fill: "freeze",
        }));
      }

      particleLayer.appendChild(dot);
    });
    group.appendChild(particleLayer);

    const labelY = Math.max(24, d.y - r - 20);
    const label = svgEl("text", { class: "family-label", x: d.x, y: labelY });
    label.textContent = labelFor(d.family, 31);

    const countLabel = svgEl("text", { class: "family-count", x: d.x, y: labelY + 18 });
    countLabel.textContent = `${fmt(familyCount)} papers`;

    const rangeLabel = svgEl("text", { class: "family-range", x: d.x, y: labelY + 34 });
    rangeLabel.textContent = particleMode === "topic"
      ? `${labelFor(focusedTopic.topic, 22)} · ${fmt(topicCount)}`
      : `${state.yearStart}–${state.yearEnd}`;

    group.appendChild(label);
    group.appendChild(countLabel);
    group.appendChild(rangeLabel);

    group.addEventListener("click", () => setSelectedFamily(d.family));
    group.addEventListener("mousemove", (event) => showTooltip(event, tooltipHtml("family", d)));
    group.addEventListener("mouseleave", hideTooltip);

    layer.appendChild(group);
  }

  function topSubtopicKeys(rows) {
    return new Set(
      [...rows]
        .sort((a, b) => countThrough(b, state.yearStart, state.yearEnd) - countThrough(a, state.yearStart, state.yearEnd))
        .slice(0, 16)
        .map(subtopicKey)
    );
  }

  function drawSubtopics(svg, maxSubtopicCount, newSubtopicR) {
    if (!state.selectedFamily) return;

    const rows = state.subtopics.filter((d) => d.family === state.selectedFamily && countThrough(d, state.yearStart, state.yearEnd) > 0);
    if (!rows.length) return;

    const family = state.families.find((d) => d.family === state.selectedFamily);
    const familyColor = family ? normalizeColor(family, 0) : "#7cc9ff";
    const labelKeys = topSubtopicKeys(rows);

    const layer = svgEl("g", { class: "subtopic-layer" });

    rows.forEach((d, i) => {
      const count = countThrough(d, state.yearStart, state.yearEnd);
      const color = normalizeColor(d, i) || familyColor;
      const key = subtopicKey(d);
      const r = radiusFor(count, SUBTOPIC_RADIUS_MIN, SUBTOPIC_RADIUS_MAX, maxSubtopicCount);
      const previousR = state.previousSubtopicR.get(key) ?? r;
      const selected = state.selectedTopic === d.topic;
      const showLabel = selected || labelKeys.has(key);
      newSubtopicR.set(key, r);

      const group = svgEl("g", {
        class: `subtopic-group${selected ? " is-selected" : ""}`,
        opacity: selected || !state.selectedTopic ? 1 : 0.35,
      });

      const halo = svgEl("circle", {
        class: "subtopic-halo",
        cx: d.x,
        cy: d.y,
        fill: color,
        "fill-opacity": selected ? 0.24 : 0.12,
        stroke: color,
        "stroke-opacity": selected ? 0.92 : 0.34,
        "stroke-width": selected ? 2.6 : 1.1,
      });
      addRadiusAnimation(halo, previousR + 8, r + 8);

      const dot = svgEl("circle", {
        class: `subtopic-dot${selected ? " is-selected" : ""}`,
        cx: d.x,
        cy: d.y,
        fill: color,
        "fill-opacity": selected ? 0.88 : 0.64,
        stroke: selected ? "#ffffff" : "rgba(255,255,255,.52)",
        "stroke-width": selected ? 2.6 : 1.2,
      });
      addRadiusAnimation(dot, previousR, r);

      const label = svgEl("text", {
        class: `subtopic-label${showLabel ? "" : " hover-only"}`,
        x: d.x,
        y: d.y - r - 10,
      });
      label.textContent = labelFor(d.topic, 24);

      group.appendChild(halo);
      group.appendChild(dot);
      group.appendChild(label);

      group.addEventListener("click", (event) => {
        event.stopPropagation();
        setSelectedTopic(d);
      });
      group.addEventListener("mousemove", (event) => showTooltip(event, tooltipHtml("subtopic", d)));
      group.addEventListener("mouseleave", hideTooltip);

      layer.appendChild(group);
    });

    svg.appendChild(layer);
  }

  function drawWatermark(svg) {
    const watermark = svgEl("text", {
      class: "watermark",
      x: VIEWBOX.width - 34,
      y: VIEWBOX.height - 36,
    });
    watermark.textContent = `${state.yearStart}–${state.yearEnd}`;
    svg.appendChild(watermark);
  }

  function render() {
    const svg = $("landscape-svg");
    if (!svg || !state.families.length) return;

    svg.replaceChildren();
    buildDefs(svg);

    svg.appendChild(svgEl("rect", {
      x: 0,
      y: 0,
      width: VIEWBOX.width,
      height: VIEWBOX.height,
      fill: "transparent",
    }));
    drawStars(svg);

    const familyCounts = state.families.map((d) => countThrough(d, state.yearStart, state.yearEnd));
    const subtopicCounts = state.subtopics.map((d) => countThrough(d, state.yearStart, state.yearEnd));
    const maxFamilyCount = Math.max(1, ...familyCounts);
    const maxSubtopicCount = Math.max(1, ...subtopicCounts);

    const newFamilyR = new Map();
    const newSubtopicR = new Map();
    const newParticleCounts = new Map();

    const familyLayer = svgEl("g", { class: "family-layer" });
    state.families.forEach((d, i) => drawFamilyNode(familyLayer, d, i, maxFamilyCount, newFamilyR, newParticleCounts));
    svg.appendChild(familyLayer);

    drawSubtopics(svg, maxSubtopicCount, newSubtopicR);
    drawWatermark(svg);

    state.previousFamilyR = newFamilyR;
    state.previousSubtopicR = newSubtopicR;
    state.previousParticleCounts = newParticleCounts;
  }

  async function fetchJsonFromCandidates(paths, label) {
    const errors = [];
    for (const path of paths) {
      try {
        const response = await fetch(path);
        if (response.ok) return await response.json();
        errors.push(`${path}: ${response.status}`);
      } catch (error) {
        errors.push(`${path}: ${error.message}`);
      }
    }
    throw new Error(`${label} could not be loaded. Tried ${errors.join(" | ")}`);
  }

  function normalizeRows(rows, type) {
    return (rows || []).map((d, i) => {
      const color = normalizeColor(d, i);
      return {
        ...d,
        x: Number(d.x ?? d.cx ?? 540),
        y: Number(d.y ?? d.cy ?? 340),
        color,
        family: String(d.family || d.topic_family || d.name || ""),
        topic: type === "subtopic" ? String(d.topic || d.primary_topic || d.name || "") : d.topic,
      };
    });
  }

  function bindControls() {
    const startInput = $("landscape-year-start");
    const endInput = $("landscape-year-end");
    const play = $("landscape-play");
    const reset = $("landscape-reset");

    if (startInput && !startInput.dataset.bound) {
      startInput.dataset.bound = "1";
      startInput.addEventListener("input", (event) => {
        stopPlay();
        setYearRange(Number(event.target.value), state.yearEnd);
      });
    }

    if (endInput && !endInput.dataset.bound) {
      endInput.dataset.bound = "1";
      endInput.addEventListener("input", (event) => {
        stopPlay();
        setYearRange(state.yearStart, Number(event.target.value));
      });
    }

    if (play && !play.dataset.bound) {
      play.dataset.bound = "1";
      play.addEventListener("click", togglePlay);
    }

    if (reset && !reset.dataset.bound) {
      reset.dataset.bound = "1";
      reset.addEventListener("click", resetView);
    }
  }

  async function init() {
    const root = $("landscape-root");
    const svg = $("landscape-svg");
    if (!root || !svg) return;

    svg.setAttribute("viewBox", `0 0 ${VIEWBOX.width} ${VIEWBOX.height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    try {
      const [families, subtopics] = await Promise.all([
        fetchJsonFromCandidates(
          [
            "data/topic_family_layout.json",
            "/data/topic_family_layout.json",
            "www/data/topic_family_layout.json",
            "/www/data/topic_family_layout.json",
          ],
          "family JSON"
        ),
        fetchJsonFromCandidates(
          [
            "data/subtopic_layout.json",
            "/data/subtopic_layout.json",
            "www/data/subtopic_layout.json",
            "/www/data/subtopic_layout.json",
          ],
          "subtopic JSON"
        ),
      ]);

      state.families = normalizeRows(families, "family").filter((d) => d.family);
      state.subtopics = normalizeRows(subtopics, "subtopic").filter((d) => d.family && d.topic);

      hideLoading();
      bindControls();
      setYearRange(MIN_YEAR, MAX_YEAR);
    } catch (error) {
      console.error("Landscape init failed:", error);
      showError(`Landscape data could not be loaded: ${error.message}`);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
