#!/usr/bin/env python3
"""Generate a standalone interactive explorer for statistical-analysis results."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "temp" / "statistical_analysis"
REQUIRED_FILES = {
    "manifest": "manifest.json",
    "pairwise": "pairwise_distances.csv",
    "profiles": "user_profiles.csv",
    "mismatch": "calibration_test_mismatch.csv",
    "features": "feature_catalog.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone HTML explorer for pairwise statistical "
            "distances, longitudinal profiles, and calibration/test mismatch."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Statistical report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: REPORT_DIR/statistical_analysis_explorer.html).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated explorer in the default browser.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.report_dir = args.report_dir.expanduser().resolve()
    if not args.report_dir.is_dir():
        parser.error(f"Report directory not found: {args.report_dir}")
    missing = [
        filename
        for filename in REQUIRED_FILES.values()
        if not (args.report_dir / filename).is_file()
    ]
    if missing:
        parser.error("Report directory is missing: " + ", ".join(sorted(missing)))
    if args.output is None:
        args.output = args.report_dir / "statistical_analysis_explorer.html"
    else:
        args.output = args.output.expanduser().resolve()
    return args, parser


def _read_csv(path: Path, identity_columns=()) -> pd.DataFrame:
    dtypes = {column: "string" for column in identity_columns}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def read_report(report_dir: Path) -> dict:
    manifest = json.loads((report_dir / REQUIRED_FILES["manifest"]).read_text(encoding="utf-8"))
    pairwise = _read_csv(
        report_dir / REQUIRED_FILES["pairwise"],
        ("user_a", "user_b"),
    )
    profiles = _read_csv(
        report_dir / REQUIRED_FILES["profiles"],
        ("user_id",),
    )
    mismatch = _read_csv(
        report_dir / REQUIRED_FILES["mismatch"],
        ("user_id", "session_id"),
    )
    features = _read_csv(report_dir / REQUIRED_FILES["features"])

    required_pairwise = {
        "view",
        "task_type",
        "family",
        "metric",
        "metric_method",
        "user_a",
        "user_b",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
    }
    missing = sorted(required_pairwise.difference(pairwise.columns))
    if missing:
        raise ValueError(
            "pairwise_distances.csv is missing columns: " + ", ".join(missing)
        )

    return {
        "manifest": manifest,
        "pairwise": pairwise,
        "profiles": profiles,
        "mismatch": mismatch,
        "features": features,
    }


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    selected = frame[[column for column in columns if column in frame]].copy()
    selected = selected.astype(object).where(pd.notna(selected), None)
    return selected.to_dict(orient="records")


def _json_for_script(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


def build_html(report: dict, source_name: str) -> str:
    pairwise_records = _records(
        report["pairwise"],
        [
            "view",
            "task_type",
            "family",
            "metric",
            "metric_method",
            "user_a",
            "user_b",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "q_value",
            "n_sessions_a",
            "n_sessions_b",
            "n_samples_a",
            "n_samples_b",
        ],
    )
    profile_records = _records(
        report["profiles"],
        [
            "user_id",
            "task_type",
            "view",
            "family",
            "metric",
            "estimate",
            "n_sessions",
            "n_samples",
        ],
    )
    mismatch_records = _records(
        report["mismatch"],
        [
            "user_id",
            "session_id",
            "family",
            "metric",
            "aggregation",
            "estimate",
            "ci_low",
            "ci_high",
            "n_sessions",
        ],
    )
    feature_records = _records(
        report["features"],
        ["family", "dimensions", "supported_views", "notes"],
    )
    metadata = report["manifest"].get("metadata", {})

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReCalib Statistical Analysis Explorer</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #637089;
      --border: #d8dfeb;
      --accent: #275dad;
      --accent-2: #0b877d;
      --chart-low: #edf2fa;
      --chart-mid: #78a9de;
      --chart-high: #183f78;
      --warning-bg: #fff5d9;
      --warning-text: #684d00;
      --shadow: 0 8px 24px rgba(24, 40, 72, 0.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111722;
        --panel: #182131;
        --text: #edf2fa;
        --muted: #aeb9cc;
        --border: #344258;
        --accent: #78a9f5;
        --accent-2: #62c9bd;
        --chart-low: #202b3d;
        --chart-mid: #477cb9;
        --chart-high: #9cc4fa;
        --warning-bg: #3b321a;
        --warning-text: #f3d887;
        --shadow: none;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1500px, 100%); margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap; }
    h1 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.3rem); font-weight: 500; }
    h2 { margin: 0 0 8px; font-size: 1.05rem; font-weight: 500; }
    p { margin: 6px 0 0; color: var(--muted); }
    .note { font-size: 0.82rem; color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
      margin: 22px 0 12px;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 0.84rem; }
    select {
      width: 100%; min-height: 38px; padding: 7px 10px;
      color: var(--text); background: var(--bg);
      border: 1px solid var(--border); border-radius: 8px; font: inherit;
    }
    select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .warning {
      display: none; margin: 0 0 12px; padding: 10px 12px;
      color: var(--warning-text); background: var(--warning-bg);
      border-radius: 8px;
    }
    .metrics {
      display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr));
      gap: 12px; margin-bottom: 12px;
    }
    .metric, .panel {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; box-shadow: var(--shadow);
    }
    .metric { padding: 14px 16px; }
    .metric span { display: block; color: var(--muted); font-size: 0.82rem; }
    .metric strong { display: block; margin-top: 4px; font-size: 1.35rem; font-weight: 500; }
    .grid {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px; margin-top: 12px;
    }
    .panel { min-width: 0; padding: 16px; }
    .wide { grid-column: 1 / -1; }
    .chart { width: 100%; min-height: 390px; }
    .heatmap { min-height: 570px; }
    .empty { display: grid; place-items: center; min-height: 320px; color: var(--muted); text-align: center; }
    .selection { min-height: 22px; margin-top: 8px; color: var(--muted); }
    .table-wrap { overflow-x: auto; max-height: 500px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--border); text-align: right; }
    th { position: sticky; top: 0; z-index: 1; background: var(--panel); color: var(--muted); font-weight: 500; }
    th:first-child, td:first-child { text-align: left; }
    td { font-variant-numeric: tabular-nums; }
    @media (max-width: 920px) {
      main { padding: 16px; }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .wide { grid-column: auto; }
    }
    @media (max-width: 520px) {
      .controls, .metrics { grid-template-columns: 1fr; }
      .heatmap { min-height: 420px; }
    }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
<main>
  <header>
    <div>
      <h1>ReCalib Statistical Analysis Explorer</h1>
      <p>Pairwise domain distances, longitudinal variability, and calibration/test mismatch.</p>
    </div>
    <p class="note">Source: __SOURCE_NAME__ · __ACCEPTED__ accepted samples · __USERS__ users · __SESSIONS__ sessions</p>
  </header>

  <section class="controls" aria-label="Statistical result filters">
    <label>Feature family<select id="family-filter"></select></label>
    <label>KPI<select id="metric-filter"></select></label>
    <label>Task<select id="task-filter"></select></label>
    <label>View<select id="view-filter"></select></label>
    <label>Matrix value<select id="statistic-filter"></select></label>
  </section>

  <div id="warning" class="warning" role="status"></div>

  <section class="metrics" aria-label="Selected matrix summary">
    <div class="metric"><span id="extreme-pair-label">Largest participant pair</span><strong id="max-pair">—</strong></div>
    <div class="metric"><span id="extreme-value-label">Largest value</span><strong id="max-value">—</strong></div>
    <div class="metric"><span>Median pairwise value</span><strong id="median-value">—</strong></div>
  </section>

  <section class="grid">
    <article class="panel wide">
      <h2>Participant × participant matrix</h2>
      <div id="matrix-chart" class="chart heatmap" role="img" aria-label="Pairwise participant distance heatmap"></div>
      <p id="pair-selection" class="selection" aria-live="polite">Select a matrix cell to inspect a participant pair.</p>
    </article>
    <article class="panel">
      <h2>Mean selected matrix value by participant</h2>
      <div id="ranking-chart" class="chart" role="img" aria-label="Participant mean distance ranking"></div>
    </article>
    <article class="panel">
      <h2>Longitudinal session instability</h2>
      <div id="profile-chart" class="chart" role="img" aria-label="Within-participant session instability"></div>
    </article>
    <article class="panel wide">
      <h2>Calibration-to-test mismatch by session</h2>
      <div id="mismatch-chart" class="chart" role="img" aria-label="Calibration-to-test mismatch distributions"></div>
    </article>
    <article class="panel wide">
      <h2>Pairwise values</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Pair</th><th>Estimate</th><th>95% CI</th><th>p</th><th>q</th><th>Sessions</th><th>Samples</th>
          </tr></thead>
          <tbody id="pair-table"></tbody>
        </table>
      </div>
    </article>
  </section>
</main>
<script>
const PAIRS = __PAIRWISE_JSON__;
const PROFILES = __PROFILES_JSON__;
const MISMATCH = __MISMATCH_JSON__;
const FEATURES = __FEATURES_JSON__;

const familyFilter = document.getElementById("family-filter");
const metricFilter = document.getElementById("metric-filter");
const taskFilter = document.getElementById("task-filter");
const viewFilter = document.getElementById("view-filter");
const statisticFilter = document.getElementById("statistic-filter");
const warning = document.getElementById("warning");
const numericFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 });
const integerFormat = new Intl.NumberFormat();
const statisticColumns = ["estimate", "ci_low", "ci_high", "p_value", "q_value"];

function unique(values) { return [...new Set(values)].sort(); }
function isNumber(value) { return typeof value === "number" && Number.isFinite(value); }
function formatValue(value) { return isNumber(value) ? numericFormat.format(value) : "—"; }
function label(value) { return String(value).replaceAll("_", " "); }

function setOptions(select, values, preferred = null) {
  const current = preferred ?? select.value;
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label(value);
    select.appendChild(option);
  }
  if (values.includes(current)) select.value = current;
}

function initializeControls() {
  const families = unique(PAIRS.map(row => row.family));
  setOptions(familyFilter, families, families.includes("head_rotation") ? "head_rotation" : families[0]);
  refreshDependentControls();
}

function refreshDependentControls() {
  const family = familyFilter.value;
  const familyRows = PAIRS.filter(row => row.family === family);
  setOptions(metricFilter, unique(familyRows.map(row => row.metric)), metricFilter.value || "distribution_shift");
  setOptions(taskFilter, unique(familyRows.map(row => row.task_type)), taskFilter.value || "16-point");
  setOptions(viewFilter, unique(familyRows.map(row => row.view)), viewFilter.value || "target_conditioned");
  const availableStatistics = statisticColumns.filter(column => familyRows.some(row => isNumber(row[column])));
  setOptions(statisticFilter, availableStatistics, statisticFilter.value || "estimate");
}

function selectedRows() {
  return PAIRS.filter(row => row.family === familyFilter.value
    && row.metric === metricFilter.value
    && row.task_type === taskFilter.value
    && row.view === viewFilter.value);
}

function theme() {
  const styles = getComputedStyle(document.documentElement);
  const get = name => styles.getPropertyValue(name).trim();
  return {
    text: get("--text"), muted: get("--muted"), border: get("--border"),
    panel: get("--panel"), accent: get("--accent"), accent2: get("--accent-2"),
    low: get("--chart-low"), mid: get("--chart-mid"), high: get("--chart-high")
  };
}

function layout(extra = {}) {
  const colors = theme();
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: colors.text, family: "Inter, system-ui, sans-serif" },
    margin: { l: 68, r: 24, t: 20, b: 60 },
    hoverlabel: { bgcolor: colors.panel, bordercolor: colors.border, font: { color: colors.text } },
    ...extra
  };
}

const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d"] };

function renderEmpty(target, message) {
  document.getElementById(target).innerHTML = `<div class="empty">${message}</div>`;
}

function summary(rows, statistic) {
  const valid = rows.filter(row => isNumber(row[statistic]));
  if (!valid.length) return null;
  const sorted = valid.map(row => row[statistic]).sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const significance = ["p_value", "q_value"].includes(statistic);
  const extreme = valid.reduce((best, row) => significance
    ? (row[statistic] < best[statistic] ? row : best)
    : (row[statistic] > best[statistic] ? row : best));
  return { extreme, median };
}

function renderMatrix(rows) {
  const statistic = statisticFilter.value;
  const significance = ["p_value", "q_value"].includes(statistic);
  document.getElementById("extreme-pair-label").textContent = significance
    ? "Smallest participant pair" : "Largest participant pair";
  document.getElementById("extreme-value-label").textContent = significance
    ? "Smallest value" : "Largest value";
  const users = unique(rows.flatMap(row => [row.user_a, row.user_b]));
  const z = users.map(() => users.map(() => null));
  const details = users.map(() => users.map(() => ""));
  if (statistic === "estimate") users.forEach((_, index) => { z[index][index] = 0; });
  for (const row of rows) {
    const a = users.indexOf(row.user_a);
    const b = users.indexOf(row.user_b);
    z[a][b] = row[statistic]; z[b][a] = row[statistic];
    const detail = `Estimate ${formatValue(row.estimate)}<br>95% CI ${formatValue(row.ci_low)}–${formatValue(row.ci_high)}<br>p ${formatValue(row.p_value)} · q ${formatValue(row.q_value)}`;
    details[a][b] = detail; details[b][a] = detail;
  }
  if (!rows.length || !z.some(row => row.some(isNumber))) {
    renderEmpty("matrix-chart", `No ${label(statistic)} values are available for this selection.`);
    document.getElementById("max-pair").textContent = "—";
    document.getElementById("max-value").textContent = "—";
    document.getElementById("median-value").textContent = "—";
    return;
  }
  const colors = theme();
  Plotly.react("matrix-chart", [{
    type: "heatmap", x: users, y: users, z, customdata: details,
    colorscale: [[0, colors.low], [0.5, colors.mid], [1, colors.high]],
    colorbar: { title: label(statistic), thickness: 14 },
    hovertemplate: "User %{y} vs user %{x}<br>" + label(statistic) + " %{z:.4f}<br>%{customdata}<extra></extra>"
  }], layout({
    xaxis: { title: "Participant", side: "bottom" },
    yaxis: { title: "Participant", autorange: "reversed", scaleanchor: "x" },
    margin: { l: 72, r: 30, t: 24, b: 62 }
  }), plotConfig);
  const stats = summary(rows, statistic);
  document.getElementById("max-pair").textContent = stats ? `${stats.extreme.user_a}–${stats.extreme.user_b}` : "—";
  document.getElementById("max-value").textContent = stats ? formatValue(stats.extreme[statistic]) : "—";
  document.getElementById("median-value").textContent = stats ? formatValue(stats.median) : "—";
  const chart = document.getElementById("matrix-chart");
  chart.removeAllListeners?.("plotly_click");
  chart.on?.("plotly_click", event => {
    const point = event.points[0];
    if (point.x === point.y) return;
    const row = rows.find(item => (item.user_a === point.x && item.user_b === point.y)
      || (item.user_a === point.y && item.user_b === point.x));
    if (!row) return;
    document.getElementById("pair-selection").textContent =
      `Users ${row.user_a}–${row.user_b}: estimate ${formatValue(row.estimate)}, ` +
      `95% CI ${formatValue(row.ci_low)}–${formatValue(row.ci_high)}, ` +
      `p ${formatValue(row.p_value)}, q ${formatValue(row.q_value)}.`;
  });
}

function renderRanking(rows) {
  const statistic = statisticFilter.value;
  const values = new Map();
  for (const row of rows) {
    if (!isNumber(row[statistic])) continue;
    for (const user of [row.user_a, row.user_b]) {
      if (!values.has(user)) values.set(user, []);
      values.get(user).push(row[statistic]);
    }
  }
  const ranking = [...values.entries()].map(([user, items]) => ({
    user, value: items.reduce((sum, item) => sum + item, 0) / items.length
  })).sort((a, b) => a.value - b.value);
  if (!ranking.length) { renderEmpty("ranking-chart", "No participant ranking is available."); return; }
  Plotly.react("ranking-chart", [{
    type: "bar", orientation: "h",
    y: ranking.map(row => `User ${row.user}`), x: ranking.map(row => row.value),
    marker: { color: theme().accent },
    text: ranking.map(row => formatValue(row.value)), textposition: "auto",
    hovertemplate: "%{y}<br>Mean " + label(statistic) + " %{x:.4f}<extra></extra>"
  }], layout({
    xaxis: { title: `Mean ${label(statistic)}`, gridcolor: theme().border },
    yaxis: { automargin: true }
  }), plotConfig);
}

function renderProfiles() {
  const rows = PROFILES.filter(row => row.family === familyFilter.value
    && row.task_type === taskFilter.value && row.view === viewFilter.value);
  const medianRows = rows.filter(row => row.metric === "longitudinal_instability_median").sort((a, b) => a.user_id.localeCompare(b.user_id));
  const iqrMap = new Map(rows.filter(row => row.metric === "longitudinal_instability_iqr").map(row => [row.user_id, row.estimate]));
  if (!medianRows.length) { renderEmpty("profile-chart", "Longitudinal profiles are unavailable for this selection."); return; }
  Plotly.react("profile-chart", [{
    type: "bar", x: medianRows.map(row => `User ${row.user_id}`), y: medianRows.map(row => row.estimate),
    error_y: { type: "data", array: medianRows.map(row => iqrMap.get(row.user_id) || 0), visible: true },
    marker: { color: theme().accent2 },
    customdata: medianRows.map(row => row.n_sessions),
    hovertemplate: "%{x}<br>Median instability %{y:.4f}<br>IQR %{error_y.array:.4f}<br>%{customdata} sessions<extra></extra>"
  }], layout({
    xaxis: { title: "Participant" },
    yaxis: { title: "Session distance", gridcolor: theme().border }
  }), plotConfig);
}

function renderMismatch() {
  const rows = MISMATCH.filter(row => row.family === familyFilter.value
    && row.metric === metricFilter.value && row.aggregation === "session" && isNumber(row.estimate));
  if (!rows.length) { renderEmpty("mismatch-chart", "Calibration/test mismatch is unavailable for this KPI."); return; }
  const users = unique(rows.map(row => row.user_id));
  const colors = theme();
  const traces = users.map(user => {
    const selected = rows.filter(row => row.user_id === user);
    return {
      type: "box", name: `User ${user}`, y: selected.map(row => row.estimate),
      boxpoints: "all", jitter: 0.32, pointpos: 0,
      marker: { size: 5, color: colors.accent },
      line: { width: 1.5, color: colors.accent },
      text: selected.map(row => `Session ${row.session_id}`),
      hovertemplate: "%{text}<br>Mismatch %{y:.4f}<extra>User " + user + "</extra>"
    };
  });
  Plotly.react("mismatch-chart", traces, layout({
    xaxis: { title: "Participant" },
    yaxis: { title: `${label(metricFilter.value)} mismatch`, gridcolor: theme().border },
    showlegend: false,
    margin: { l: 72, r: 20, t: 20, b: 60 }
  }), plotConfig);
}

function renderTable(rows) {
  const body = document.getElementById("pair-table");
  body.replaceChildren();
  const sorted = [...rows].sort((a, b) => (b.estimate ?? -Infinity) - (a.estimate ?? -Infinity));
  for (const item of sorted) {
    const row = body.insertRow();
    const values = [
      `${item.user_a}–${item.user_b}`,
      formatValue(item.estimate),
      isNumber(item.ci_low) || isNumber(item.ci_high) ? `${formatValue(item.ci_low)}–${formatValue(item.ci_high)}` : "—",
      formatValue(item.p_value), formatValue(item.q_value),
      `${integerFormat.format(item.n_sessions_a)} / ${integerFormat.format(item.n_sessions_b)}`,
      `${integerFormat.format(item.n_samples_a)} / ${integerFormat.format(item.n_samples_b)}`
    ];
    for (const value of values) row.insertCell().textContent = value;
  }
}

function renderWarning(rows) {
  const messages = [];
  if (statisticFilter.value !== "estimate" && !rows.some(row => isNumber(row[statisticFilter.value]))) {
    messages.push("This report was generated without the selected inferential statistic.");
  }
  if (familyFilter.value === "gaze_point" && viewFilter.value === "target_conditioned") {
    messages.push("Target-conditioned gaze-point distances are expected to be zero because the feature is determined by target position.");
  }
  if (["gaze_point", "target_coverage"].includes(familyFilter.value)) {
    messages.push("Calibration/test mismatch for this family can contain near-zero-scale artifacts and should not be interpreted.");
  }
  warning.textContent = messages.join(" ");
  warning.style.display = messages.length ? "block" : "none";
}

function update() {
  const rows = selectedRows();
  renderWarning(rows);
  renderMatrix(rows);
  renderRanking(rows);
  renderProfiles();
  renderMismatch();
  renderTable(rows);
}

familyFilter.addEventListener("change", () => { refreshDependentControls(); update(); });
for (const control of [metricFilter, taskFilter, viewFilter, statisticFilter]) control.addEventListener("change", update);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", update);
initializeControls();
update();
</script>
</body>
</html>
"""

    return (
        template.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__SOURCE_NAME__", html.escape(source_name))
        .replace("__ACCEPTED__", f"{int(metadata.get('accepted_rows', 0)):,}")
        .replace("__USERS__", str(metadata.get("users", "—")))
        .replace("__SESSIONS__", str(metadata.get("user_sessions", "—")))
        .replace("__PAIRWISE_JSON__", _json_for_script(pairwise_records))
        .replace("__PROFILES_JSON__", _json_for_script(profile_records))
        .replace("__MISMATCH_JSON__", _json_for_script(mismatch_records))
        .replace("__FEATURES_JSON__", _json_for_script(feature_records))
    )


def main(argv=None) -> int:
    args, parser = parse_args(argv)
    try:
        report = read_report(args.report_dir)
        explorer = build_html(report, args.report_dir.name)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(explorer, encoding="utf-8")
    print(
        f"Created statistical explorer: {args.output} "
        f"({len(report['pairwise']):,} pairwise rows; "
        f"{len(report['profiles']):,} profile rows; "
        f"{len(report['mismatch']):,} mismatch rows)"
    )
    if args.open:
        webbrowser.open(args.output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
