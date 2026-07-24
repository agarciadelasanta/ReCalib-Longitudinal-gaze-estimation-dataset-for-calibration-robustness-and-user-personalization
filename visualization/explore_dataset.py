#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate an interactive, self-contained explorer for a ReCalib summary CSV."""

import argparse
import html
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "dataset_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "temp" / "recalib_dataset_explorer.html"

IDENTITY_COLUMNS = ["user_id", "session_id", "task_id", "task_type"]
OPTIONAL_COLUMNS = [
    "discarded",
    "discard_info",
    "head_rot_x",
    "head_rot_y",
    "head_rot_z",
    "head_trans_x",
    "head_trans_y",
    "head_trans_z",
    "gaze_vector_x",
    "gaze_vector_y",
    "gaze_vector_z",
    "pog_px_x",
    "pog_px_y",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone HTML explorer for ReCalib participants, "
            "sessions, tasks, quality flags, head pose, gaze, and targets."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Dataset summary CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Generated HTML report (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--id-user",
        type=int,
        help="Participant selected when the report first opens.",
    )
    parser.add_argument(
        "--id-session",
        type=int,
        help="Session selected when the report first opens; requires --id-user.",
    )
    parser.add_argument(
        "--id-task",
        type=int,
        help="Task selected when the report first opens; requires --id-session.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12_000,
        help=(
            "Maximum deterministic sample used by continuous plots. "
            "Counts remain exact (default: 12000)."
        ),
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in the default browser.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.id_session is not None and args.id_user is None:
        parser.error("--id-session requires --id-user")
    if args.id_task is not None and args.id_session is None:
        parser.error("--id-task requires --id-session")
    if args.max_points < 100:
        parser.error("--max-points must be at least 100")

    args.csv = args.csv.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    return args, parser


def read_summary(csv_path, parser):
    header = pd.read_csv(csv_path, nrows=0)
    missing = [column for column in IDENTITY_COLUMNS if column not in header.columns]
    if missing:
        parser.error(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    use_columns = IDENTITY_COLUMNS + [
        column for column in OPTIONAL_COLUMNS if column in header.columns
    ]
    frame = pd.read_csv(csv_path, usecols=use_columns, low_memory=False)

    for column in ["user_id", "session_id", "task_id"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_identity = frame[["user_id", "session_id", "task_id"]].isna().any(axis=1)
    if invalid_identity.all():
        parser.error("CSV contains no rows with valid user/session/task identifiers")
    if invalid_identity.any():
        print(f"Skipped {int(invalid_identity.sum())} rows with invalid identifiers.")
        frame = frame.loc[~invalid_identity].copy()

    for column in ["user_id", "session_id", "task_id"]:
        frame[column] = frame[column].astype(int)
    frame["task_type"] = frame["task_type"].fillna("Unknown").astype(str)

    for column in OPTIONAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan

    discarded_text = frame["discarded"].astype(str).str.lower()
    discarded_flag = discarded_text.isin({"true", "1", "yes"})
    info = frame["discard_info"].fillna("Unknown").astype(str)
    accepted = info.eq("NO_DISCARDED") | (
        info.eq("Unknown") & ~discarded_flag
    )
    frame["_status"] = np.where(accepted, "Accepted", "Discarded")
    frame["_reason"] = np.where(accepted, "Accepted", info)

    numeric_columns = [
        column
        for column in OPTIONAL_COLUMNS
        if column not in {"discarded", "discard_info"}
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_initial_selection(frame, args, parser):
    selected = frame
    if args.id_user is not None:
        selected = selected[selected["user_id"] == args.id_user]
        if selected.empty:
            parser.error(f"participant {args.id_user} is not present in the CSV")
    if args.id_session is not None:
        selected = selected[selected["session_id"] == args.id_session]
        if selected.empty:
            parser.error(
                f"session {args.id_session} is not present for participant "
                f"{args.id_user}"
            )
    if args.id_task is not None:
        selected = selected[selected["task_id"] == args.id_task]
        if selected.empty:
            parser.error(
                f"task {args.id_task} is not present for participant "
                f"{args.id_user}, session {args.id_session}"
            )


def make_group_records(frame):
    group_columns = [
        "user_id",
        "session_id",
        "task_id",
        "task_type",
        "_status",
        "_reason",
    ]
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    grouped.columns = ["user", "session", "task", "task_type", "status", "reason", "count"]
    return grouped.to_dict(orient="records")


def make_point_records(frame, max_points):
    if len(frame) > max_points:
        point_frame = frame.sample(max_points, random_state=2026)
    else:
        point_frame = frame.copy()

    columns = {
        "user_id": "user",
        "session_id": "session",
        "task_id": "task",
        "task_type": "task_type",
        "_status": "status",
        "head_rot_x": "head_pitch",
        "head_rot_y": "head_yaw",
        "pog_px_x": "target_x",
        "pog_px_y": "target_y",
    }
    point_frame = point_frame[list(columns)].rename(columns=columns)
    point_frame = point_frame.astype(object).where(pd.notna(point_frame), None)
    return point_frame.to_dict(orient="records")


def json_for_script(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


def build_report(frame, args):
    groups = make_group_records(frame)
    points = make_point_records(frame, args.max_points)
    initial = {
        "user": args.id_user,
        "session": args.id_session,
        "task": args.id_task,
    }
    source_name = html.escape(args.csv.name)

    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReCalib Dataset Explorer</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647089;
      --border: #d9dfeb;
      --accent: #275dad;
      --accent-2: #0b877d;
      --danger: #b94848;
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
        --danger: #f08a8a;
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
    main { width: min(1440px, 100%); margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap; }
    h1 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.3rem); font-weight: 500; }
    h2 { margin: 0 0 8px; font-size: 1.05rem; font-weight: 500; }
    p { margin: 6px 0 0; color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
      margin: 22px 0;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 0.84rem; }
    select {
      width: 100%;
      min-height: 38px;
      padding: 7px 10px;
      color: var(--text);
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      font: inherit;
    }
    select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .metric { padding: 16px; }
    .metric span { display: block; color: var(--muted); font-size: 0.82rem; }
    .metric strong { display: block; margin-top: 4px; font-size: 1.55rem; font-weight: 500; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .panel { min-width: 0; padding: 16px; }
    .wide { grid-column: 1 / -1; }
    .chart { width: 100%; min-height: 360px; }
    .empty {
      display: grid;
      place-items: center;
      min-height: 300px;
      color: var(--muted);
      text-align: center;
    }
    .table-wrap { overflow-x: auto; max-height: 520px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; }
    th { position: sticky; top: 0; background: var(--panel); color: var(--muted); font-weight: 500; }
    th:first-child, td:first-child { text-align: left; }
    .note { font-size: 0.82rem; color: var(--muted); }
    @media (max-width: 820px) {
      main { padding: 16px; }
      .controls, .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .wide { grid-column: auto; }
    }
    @media (max-width: 480px) {
      .controls, .metrics { grid-template-columns: 1fr; }
    }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
<main>
  <header>
    <div>
      <h1>ReCalib Dataset Explorer</h1>
      <p>Longitudinal coverage, quality outcomes, head pose, and gaze targets.</p>
    </div>
    <p class="note">Source: __SOURCE_NAME__ · Continuous plots use a deterministic sample; counts are exact.</p>
  </header>

  <section class="controls" aria-label="Dataset filters">
    <label>Participant<select id="user-filter"></select></label>
    <label>Session<select id="session-filter"></select></label>
    <label>Task<select id="task-filter"></select></label>
    <label>Quality<select id="status-filter">
      <option value="">All samples</option>
      <option value="Accepted">Accepted</option>
      <option value="Discarded">Discarded</option>
    </select></label>
  </section>

  <section class="metrics" aria-label="Filtered summary">
    <div class="metric"><span>Samples</span><strong id="metric-samples">—</strong></div>
    <div class="metric"><span>Accepted</span><strong id="metric-accepted">—</strong></div>
    <div class="metric"><span>Participants</span><strong id="metric-users">—</strong></div>
    <div class="metric"><span>Sessions / tasks</span><strong id="metric-sessions">—</strong></div>
  </section>

  <section class="grid">
    <article class="panel">
      <h2>Task composition</h2>
      <div id="task-chart" class="chart" role="img" aria-label="Samples grouped by task type"></div>
    </article>
    <article class="panel">
      <h2>Quality outcomes</h2>
      <div id="quality-chart" class="chart" role="img" aria-label="Samples grouped by quality outcome"></div>
    </article>
    <article class="panel wide">
      <h2>Longitudinal session coverage</h2>
      <div id="session-chart" class="chart" role="img" aria-label="Session coverage by participant and task type"></div>
    </article>
    <article class="panel">
      <h2>Head pose coverage</h2>
      <div id="head-chart" class="chart" role="img" aria-label="Sampled head pitch and yaw coverage"></div>
    </article>
    <article class="panel">
      <h2>Screen target coverage</h2>
      <div id="target-chart" class="chart" role="img" aria-label="Sampled point of gaze target density"></div>
    </article>
    <article class="panel wide">
      <h2>Session breakdown</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Participant / session</th>
            <th>Samples</th>
            <th>Accepted</th>
            <th>Tasks</th>
            <th>9-point</th>
            <th>16-point</th>
          </tr></thead>
          <tbody id="session-table"></tbody>
        </table>
      </div>
    </article>
  </section>
</main>
<script>
const GROUPS = __GROUPS_JSON__;
const POINTS = __POINTS_JSON__;
const INITIAL = __INITIAL_JSON__;
const userFilter = document.getElementById("user-filter");
const sessionFilter = document.getElementById("session-filter");
const taskFilter = document.getElementById("task-filter");
const statusFilter = document.getElementById("status-filter");
const numberFormat = new Intl.NumberFormat();
const percentFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const taskColors = ["#2f6fbb", "#16a394", "#9b6bc7", "#d0833f", "#d65f73", "#718096"];

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => Number(a) - Number(b));
}

function setOptions(select, values, current, allLabel) {
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = allLabel;
  select.appendChild(all);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value).padStart(2, "0");
    select.appendChild(option);
  }
  if (current !== null && current !== undefined && values.includes(Number(current))) {
    select.value = String(current);
  }
}

function selectedNumber(select) {
  return select.value === "" ? null : Number(select.value);
}

function matches(row) {
  const user = selectedNumber(userFilter);
  const session = selectedNumber(sessionFilter);
  const task = selectedNumber(taskFilter);
  const status = statusFilter.value;
  return (user === null || row.user === user)
    && (session === null || row.session === session)
    && (task === null || row.task === task)
    && (status === "" || row.status === status);
}

function refreshSessions(preferred = null) {
  const user = selectedNumber(userFilter);
  const relevant = GROUPS.filter(row => user === null || row.user === user);
  const current = preferred === null ? selectedNumber(sessionFilter) : preferred;
  setOptions(sessionFilter, uniqueSorted(relevant.map(row => row.session)), current, "All sessions");
}

function refreshTasks(preferred = null) {
  const user = selectedNumber(userFilter);
  const session = selectedNumber(sessionFilter);
  const relevant = GROUPS.filter(row =>
    (user === null || row.user === user) && (session === null || row.session === session)
  );
  const current = preferred === null ? selectedNumber(taskFilter) : preferred;
  setOptions(taskFilter, uniqueSorted(relevant.map(row => row.task)), current, "All tasks");
}

function sumBy(rows, key) {
  const totals = new Map();
  for (const row of rows) totals.set(row[key], (totals.get(row[key]) || 0) + row.count);
  return [...totals.entries()].sort((a, b) => b[1] - a[1]);
}

function themeLayout(extra = {}) {
  const styles = getComputedStyle(document.documentElement);
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: styles.getPropertyValue("--text").trim(), family: "Inter, system-ui, sans-serif" },
    margin: { l: 58, r: 18, t: 18, b: 52 },
    hoverlabel: {
      bgcolor: styles.getPropertyValue("--panel").trim(),
      bordercolor: styles.getPropertyValue("--border").trim(),
      font: { color: styles.getPropertyValue("--text").trim() }
    },
    ...extra
  };
}

const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d"] };

function renderBar(target, entries, color, xTitle) {
  if (!entries.length) {
    document.getElementById(target).innerHTML = '<div class="empty">No data for this filter.</div>';
    return;
  }
  Plotly.react(target, [{
    type: "bar",
    orientation: "h",
    y: entries.map(entry => entry[0]).reverse(),
    x: entries.map(entry => entry[1]).reverse(),
    marker: { color },
    text: entries.map(entry => numberFormat.format(entry[1])).reverse(),
    textposition: "auto",
    hovertemplate: "%{y}<br>%{x:,} samples<extra></extra>"
  }], themeLayout({
    xaxis: { title: xTitle, gridcolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim() },
    yaxis: { automargin: true }
  }), plotConfig);
}

function renderTaskChart(groups) {
  renderBar("task-chart", sumBy(groups, "task_type"), taskColors[0], "Samples");
}

function renderQualityChart(groups) {
  const entries = sumBy(groups, "reason").slice(0, 12);
  renderBar("quality-chart", entries, taskColors[1], "Samples");
}

function renderSessionChart(groups) {
  const aggregated = new Map();
  for (const row of groups) {
    const key = `${row.user}|${row.session}|${row.task_type}`;
    aggregated.set(key, (aggregated.get(key) || 0) + row.count);
  }
  const taskTypes = [...new Set(groups.map(row => row.task_type))].sort();
  const traces = taskTypes.map((taskType, index) => {
    const rows = [...aggregated.entries()]
      .filter(([key]) => key.endsWith(`|${taskType}`))
      .map(([key, count]) => {
        const [user, session] = key.split("|").map(Number);
        return { user, session, count };
      });
    return {
      type: "scatter",
      mode: "markers",
      name: taskType,
      x: rows.map(row => row.session),
      y: rows.map(row => `User ${String(row.user).padStart(2, "0")}`),
      customdata: rows.map(row => row.count),
      marker: {
        color: taskColors[index % taskColors.length],
        size: rows.map(row => Math.max(7, Math.min(28, Math.sqrt(row.count) / 1.6))),
        opacity: 0.76,
        line: { width: 1, color: "rgba(255,255,255,0.45)" }
      },
      hovertemplate: "Session %{x}<br>%{y}<br>%{customdata:,} samples<extra>" + taskType + "</extra>"
    };
  });
  Plotly.react("session-chart", traces, themeLayout({
    xaxis: {
      title: "Session ID",
      dtick: 1,
      gridcolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim()
    },
    yaxis: { title: "Participant", automargin: true },
    legend: { orientation: "h", y: 1.08 },
    margin: { l: 82, r: 20, t: 48, b: 52 }
  }), plotConfig);
}

function renderHeadChart(points) {
  const valid = points.filter(row => Number.isFinite(row.head_yaw) && Number.isFinite(row.head_pitch));
  const taskTypes = [...new Set(valid.map(row => row.task_type))].sort();
  const traces = taskTypes.map((taskType, index) => {
    const rows = valid.filter(row => row.task_type === taskType);
    return {
      type: "scattergl",
      mode: "markers",
      name: taskType,
      x: rows.map(row => row.head_yaw),
      y: rows.map(row => row.head_pitch),
      marker: { size: 4, opacity: 0.38, color: taskColors[index % taskColors.length] },
      text: rows.map(row => `User ${row.user}, session ${row.session}, task ${row.task}`),
      hovertemplate: "%{text}<br>Yaw %{x:.3f} rad<br>Pitch %{y:.3f} rad<extra>" + taskType + "</extra>"
    };
  });
  if (!valid.length) {
    document.getElementById("head-chart").innerHTML = '<div class="empty">Head rotation columns are unavailable for this filter.</div>';
    return;
  }
  Plotly.react("head-chart", traces, themeLayout({
    xaxis: {
      title: "Yaw (rad)",
      gridcolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim(),
      zerolinecolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim()
    },
    yaxis: {
      title: "Pitch (rad)",
      gridcolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim(),
      zerolinecolor: getComputedStyle(document.documentElement).getPropertyValue("--border").trim()
    },
    legend: { orientation: "h", y: 1.08 },
    margin: { l: 62, r: 18, t: 48, b: 56 }
  }), plotConfig);
}

function renderTargetChart(points) {
  const valid = points.filter(row => Number.isFinite(row.target_x) && Number.isFinite(row.target_y));
  if (!valid.length) {
    document.getElementById("target-chart").innerHTML = '<div class="empty">Target-coordinate columns are unavailable for this filter.</div>';
    return;
  }
  Plotly.react("target-chart", [{
    type: "histogram2d",
    x: valid.map(row => row.target_x),
    y: valid.map(row => row.target_y),
    nbinsx: 28,
    nbinsy: 20,
    colorscale: [[0, "rgba(47,111,187,0)"], [0.25, "#2f6fbb"], [1, "#16a394"]],
    colorbar: { title: "Samples", thickness: 12 },
    hovertemplate: "Target x %{x}<br>Target y %{y}<br>%{z} sampled rows<extra></extra>"
  }], themeLayout({
    xaxis: { title: "Screen x (px)" },
    yaxis: { title: "Screen y (px)", autorange: "reversed" },
    margin: { l: 62, r: 18, t: 18, b: 56 }
  }), plotConfig);
}

function renderTable(groups) {
  const sessions = new Map();
  for (const row of groups) {
    const key = `${row.user}|${row.session}`;
    if (!sessions.has(key)) {
      sessions.set(key, { user: row.user, session: row.session, samples: 0, accepted: 0, tasks: new Set(), calibration: 0, test: 0 });
    }
    const item = sessions.get(key);
    item.samples += row.count;
    if (row.status === "Accepted") item.accepted += row.count;
    item.tasks.add(row.task);
    if (row.task_type.toLowerCase().includes("9")) item.calibration += row.count;
    if (row.task_type.toLowerCase().includes("16")) item.test += row.count;
  }
  const rows = [...sessions.values()].sort((a, b) => a.user - b.user || a.session - b.session);
  const body = document.getElementById("session-table");
  body.replaceChildren();
  if (!rows.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 6;
    cell.textContent = "No sessions match this filter.";
    return;
  }
  for (const item of rows) {
    const row = body.insertRow();
    const values = [
      `User ${String(item.user).padStart(2, "0")} / session ${String(item.session).padStart(2, "0")}`,
      numberFormat.format(item.samples),
      `${percentFormat.format(item.accepted / item.samples * 100)}%`,
      numberFormat.format(item.tasks.size),
      numberFormat.format(item.calibration),
      numberFormat.format(item.test)
    ];
    for (const value of values) row.insertCell().textContent = value;
  }
}

function update() {
  const groups = GROUPS.filter(matches);
  const points = POINTS.filter(matches);
  const total = groups.reduce((sum, row) => sum + row.count, 0);
  const accepted = groups.filter(row => row.status === "Accepted").reduce((sum, row) => sum + row.count, 0);
  const users = new Set(groups.map(row => row.user)).size;
  const sessions = new Set(groups.map(row => `${row.user}|${row.session}`)).size;
  const tasks = new Set(groups.map(row => `${row.user}|${row.session}|${row.task}`)).size;

  document.getElementById("metric-samples").textContent = numberFormat.format(total);
  document.getElementById("metric-accepted").textContent = total ? `${percentFormat.format(accepted / total * 100)}%` : "—";
  document.getElementById("metric-users").textContent = numberFormat.format(users);
  document.getElementById("metric-sessions").textContent = `${numberFormat.format(sessions)} / ${numberFormat.format(tasks)}`;

  renderTaskChart(groups);
  renderQualityChart(groups);
  renderSessionChart(groups);
  renderHeadChart(points);
  renderTargetChart(points);
  renderTable(groups);
}

setOptions(userFilter, uniqueSorted(GROUPS.map(row => row.user)), INITIAL.user, "All participants");
refreshSessions(INITIAL.session);
refreshTasks(INITIAL.task);

userFilter.addEventListener("change", () => {
  refreshSessions();
  refreshTasks();
  update();
});
sessionFilter.addEventListener("change", () => {
  refreshTasks();
  update();
});
taskFilter.addEventListener("change", update);
statusFilter.addEventListener("change", update);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", update);
update();
</script>
</body>
</html>
"""
    return (
        template.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__SOURCE_NAME__", source_name)
        .replace("__GROUPS_JSON__", json_for_script(groups))
        .replace("__POINTS_JSON__", json_for_script(points))
        .replace("__INITIAL_JSON__", json_for_script(initial))
    )


def main(argv=None):
    args, parser = parse_args(argv)
    frame = read_summary(args.csv, parser)
    validate_initial_selection(frame, args, parser)
    report = build_report(frame, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        f"Created explorer: {args.output} "
        f"({len(frame):,} rows; {min(len(frame), args.max_points):,} plotted points)"
    )
    if args.open:
        webbrowser.open(args.output.as_uri())


if __name__ == "__main__":
    main()
