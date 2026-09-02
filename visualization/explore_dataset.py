#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate an interactive, self-contained explorer for a ReCalib summary CSV."""

import argparse
import html
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "dataset_summary.csv"
DEFAULT_SETUP = REPO_ROOT / "docs" / "setup_config.json"
DEFAULT_OUTPUT = REPO_ROOT / "temp" / "recalib_dataset_explorer.html"

IDENTITY_COLUMNS = ["user_id", "session_id", "task_id", "task_type"]
OPTIONAL_COLUMNS = [
    "head_rot_x",
    "head_rot_y",
    "head_rot_z",
    "head_trans_x",
    "head_trans_y",
    "head_trans_z",
    "pog_px_x",
    "pog_px_y",
]


@dataclass(frozen=True)
class ScreenGeometry:
    """Physical and rendered screen dimensions used for target conversion."""

    width_cm: float
    height_cm: float
    width_px: float
    height_px: float


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone HTML explorer for ReCalib participants, "
            "sessions, tasks, head-pose coverage, and screen targets."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Dataset summary CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--setup-config",
        type=Path,
        default=DEFAULT_SETUP,
        help=(
            "Acquisition setup JSON used to convert target pixels to centimeters "
            f"(default: {DEFAULT_SETUP})."
        ),
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
            "Maximum deterministic sample used by head-pose plots. "
            "Counts and target coverage remain exact (default: 12000)."
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
    args.setup_config = args.setup_config.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    if not args.setup_config.is_file():
        parser.error(f"setup config not found: {args.setup_config}")
    return args, parser


def read_screen_geometry(setup_path, parser):
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        screen_mm = setup["screen_mm"]
        screen_pixels = setup["screen_pixels"]
        zoom = float(setup.get("screen_zoom", 1.0))
        orientation = int(setup.get("screen_orientation", 0))
        width_mm = float(screen_mm["width"])
        height_mm = float(screen_mm["height"])
        width_px = float(screen_pixels["width"])
        height_px = float(screen_pixels["height"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid setup config {setup_path}: {error}")

    if orientation in (2, 3):
        width_mm, height_mm = height_mm, width_mm
        width_px, height_px = height_px, width_px
    width_px *= zoom
    height_px *= zoom
    if min(width_mm, height_mm, width_px, height_px) <= 0:
        parser.error("screen dimensions and screen_zoom must be positive")

    return ScreenGeometry(
        width_cm=width_mm / 10.0,
        height_cm=height_mm / 10.0,
        width_px=width_px,
        height_px=height_px,
    )


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


def _target_key(x_value, y_value):
    if pd.isna(x_value) or pd.isna(y_value):
        return None
    return f"{float(x_value):.12g}|{float(y_value):.12g}"


def _with_target_coordinates(frame, screen):
    prepared = frame.copy()
    prepared["_target_key"] = [
        _target_key(x_value, y_value)
        for x_value, y_value in zip(prepared["pog_px_x"], prepared["pog_px_y"])
    ]
    prepared["_target_cm_x"] = (
        prepared["pog_px_x"] / screen.width_px * screen.width_cm
    )
    prepared["_target_cm_y"] = (
        prepared["pog_px_y"] / screen.height_px * screen.height_cm
    )
    return prepared


def make_group_records(frame, screen):
    prepared = _with_target_coordinates(frame, screen)
    group_columns = [
        "user_id",
        "session_id",
        "task_id",
        "task_type",
        "pog_px_x",
        "pog_px_y",
        "_target_cm_x",
        "_target_cm_y",
        "_target_key",
    ]
    grouped = (
        prepared.groupby(group_columns, dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    grouped.columns = [
        "user",
        "session",
        "task",
        "task_type",
        "target_px_x",
        "target_px_y",
        "target_cm_x",
        "target_cm_y",
        "target_key",
        "count",
    ]
    grouped = grouped.astype(object).where(pd.notna(grouped), None)
    return grouped.to_dict(orient="records")


def make_point_records(frame, max_points, screen):
    if len(frame) > max_points:
        point_frame = frame.sample(max_points, random_state=2026)
    else:
        point_frame = frame.copy()
    point_frame = _with_target_coordinates(point_frame, screen)

    columns = {
        "user_id": "user",
        "session_id": "session",
        "task_id": "task",
        "task_type": "task_type",
        "head_rot_x": "head_pitch",
        "head_rot_y": "head_yaw",
        "head_rot_z": "head_roll",
        "head_trans_x": "head_translation_x",
        "head_trans_y": "head_translation_y",
        "head_trans_z": "head_translation_z",
        "_target_key": "target_key",
    }
    point_frame = point_frame[list(columns)].rename(columns=columns)
    for column in ["head_pitch", "head_yaw", "head_roll"]:
        point_frame[column] = np.degrees(point_frame[column])
    for column in [
        "head_translation_x",
        "head_translation_y",
        "head_translation_z",
    ]:
        point_frame[column] = point_frame[column] / 10.0
    point_frame = point_frame.astype(object).where(pd.notna(point_frame), None)
    return point_frame.to_dict(orient="records")


def json_for_script(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")


def build_report(frame, args, screen):
    groups = make_group_records(frame, screen)
    points = make_point_records(frame, args.max_points, screen)
    initial = {
        "user": args.id_user,
        "session": args.id_session,
        "task": args.id_task,
    }
    screen_record = {
        "width_cm": screen.width_cm,
        "height_cm": screen.height_cm,
        "width_px": screen.width_px,
        "height_px": screen.height_px,
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
    main { width: min(1480px, 100%); margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: end; flex-wrap: wrap; }
    h1 { margin: 0; font-size: clamp(1.55rem, 3vw, 2.3rem); font-weight: 500; }
    h2 { margin: 0; font-size: 1.05rem; font-weight: 500; }
    p { margin: 6px 0 0; color: var(--muted); }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 12px;
      margin: 22px 0;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 0.84rem; }
    select, button {
      min-height: 38px;
      padding: 7px 10px;
      color: var(--text);
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      font: inherit;
    }
    select { width: 100%; }
    select:focus, button:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    button { cursor: pointer; }
    button:disabled { cursor: default; opacity: 0.5; }
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
    .chart { width: 100%; min-height: 370px; }
    .chart-tall { min-height: 450px; }
    .panel-heading { display: flex; align-items: start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .panel-heading p { font-size: 0.84rem; }
    .coverage-controls { display: flex; gap: 10px; align-items: end; flex-wrap: wrap; }
    .coverage-controls label { min-width: 180px; }
    .chart-switch { display: flex; gap: 6px; align-items: center; }
    .chart-switch button[aria-pressed="true"] { box-shadow: inset 0 0 0 2px var(--accent); background: var(--panel); }
    .tabs { display: flex; gap: 4px; margin: 0 0 16px; border-bottom: 1px solid var(--border); }
    .tab-button { border: 0; border-bottom: 3px solid transparent; border-radius: 0; background: transparent; }
    .tab-button[aria-selected="true"] { border-bottom-color: var(--accent); color: var(--accent); }
    .collapsible-heading { cursor: pointer; }
    .collapse-toggle { margin-left: auto; white-space: nowrap; }
    .collapsible-content { min-width: 0; }
    .panel.is-collapsed .panel-heading p { display: none; }
    .panel.is-collapsed .panel-heading > :not(:first-child):not(.collapse-toggle) { display: none; }
    [hidden] { display: none !important; }
    .target-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .target-actions span { color: var(--muted); font-size: 0.84rem; }
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
      <p>Task coverage, head-pose variation, and interactive screen targets.</p>
    </div>
    <p class="note">Source: __SOURCE_NAME__ &middot; Head-pose plots use a deterministic sample; counts are exact.</p>
  </header>

  <section class="controls" aria-label="Dataset filters">
    <label>Participant<select id="user-filter"></select></label>
    <label>Session<select id="session-filter"></select></label>
    <label>Task<select id="task-filter"></select></label>
  </section>

  <nav class="tabs" role="tablist" aria-label="Explorer sections">
    <button id="overview-tab" class="tab-button" type="button" role="tab" aria-selected="true" aria-controls="overview-panel">Overview</button>
    <button id="distributions-tab" class="tab-button" type="button" role="tab" aria-selected="false" aria-controls="distributions-panel">Sample distributions</button>
  </nav>

  <section id="overview-panel" role="tabpanel" aria-labelledby="overview-tab">
    <section class="metrics" aria-label="Filtered summary">
      <div class="metric"><span>Samples</span><strong id="metric-samples">&mdash;</strong></div>
      <div class="metric"><span>Participants</span><strong id="metric-users">&mdash;</strong></div>
      <div class="metric"><span>Sessions</span><strong id="metric-sessions">&mdash;</strong></div>
      <div class="metric"><span>Tasks</span><strong id="metric-tasks">&mdash;</strong></div>
    </section>
    <section class="grid">
      <article class="panel">
        <h2>Task composition</h2>
        <div id="task-chart" class="chart" role="img" aria-label="Samples grouped by task"></div>
      </article>
      <article class="panel">
        <h2>Longitudinal session coverage</h2>
        <div id="session-chart" class="chart" role="img" aria-label="Session coverage by participant and task"></div>
      </article>
      <article class="panel wide">
        <h2>Session breakdown</h2>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>Participant / session</th>
              <th>Samples</th>
              <th>Tasks</th>
              <th>Targets</th>
              <th>9-point</th>
              <th>16-point</th>
            </tr></thead>
            <tbody id="session-table"></tbody>
          </table>
        </div>
      </article>
    </section>
  </section>

  <section id="distributions-panel" role="tabpanel" aria-labelledby="distributions-tab" hidden>
    <section class="grid">
    <article class="panel wide">
      <div class="panel-heading">
        <div>
          <h2>Screen target cross-filter</h2>
          <p>Click one or more targets to filter every other view. Click a selected target again to remove it.</p>
        </div>
        <div class="target-actions">
          <span id="target-selection" aria-live="polite"></span>
          <button id="clear-targets" type="button">Clear target selection</button>
        </div>
      </div>
      <div id="target-chart" class="chart chart-tall" role="img" aria-label="Interactive screen targets in centimeters"></div>
    </article>
    <article class="panel wide">
      <div class="panel-heading">
        <div>
          <h2 id="rotation-coverage-title">Yaw and pitch coverage by task</h2>
          <p>Population regions show sample-size-normalized density contours.</p>
        </div>
        <div class="coverage-controls">
          <label>Axes
            <select id="rotation-axes">
              <option value="yaw-pitch">Yaw and pitch</option>
              <option value="roll-yaw">Roll and yaw</option>
            </select>
          </label>
          <div class="chart-switch" aria-label="Rotation coverage display">
            <button id="rotation-points" type="button" aria-pressed="true">Points</button>
            <button id="rotation-regions" type="button" aria-pressed="false">Population regions</button>
          </div>
        </div>
      </div>
      <div id="head-scatter-chart" class="chart chart-tall" role="img" aria-label="Head yaw and pitch coverage split into task panels"></div>
    </article>
    <article class="panel wide">
      <h2>Head rotation distributions by task</h2>
      <div id="head-distribution-chart" class="chart chart-tall" role="img" aria-label="Pitch yaw and roll distributions for each task"></div>
    </article>
    <article class="panel wide">
      <h2>Central 90% head-pose span</h2>
      <div id="head-span-chart" class="chart" role="img" aria-label="Fifth to ninety-fifth percentile head rotation span by task"></div>
    </article>
    <article class="panel wide">
      <div class="panel-heading">
        <div>
          <h2 id="translation-coverage-title">Horizontal and depth translation coverage by task</h2>
          <p>Population regions show sample-size-normalized density contours.</p>
        </div>
        <div class="coverage-controls">
          <label>Axes
            <select id="translation-axes">
              <option value="horizontal-depth">Horizontal and depth</option>
              <option value="horizontal-vertical">Horizontal and vertical</option>
            </select>
          </label>
          <div class="chart-switch" aria-label="Translation coverage display">
            <button id="translation-points" type="button" aria-pressed="true">Points</button>
            <button id="translation-regions" type="button" aria-pressed="false">Population regions</button>
          </div>
        </div>
      </div>
      <div id="translation-scatter-chart" class="chart chart-tall" role="img" aria-label="Head horizontal and depth translation coverage split into task panels"></div>
    </article>
    <article class="panel wide">
      <h2>Head translation distributions by task</h2>
      <div id="translation-distribution-chart" class="chart chart-tall" role="img" aria-label="Head translation distributions for each task"></div>
    </article>
    <article class="panel wide">
      <h2>Central 90% head-translation span</h2>
      <div id="translation-span-chart" class="chart" role="img" aria-label="Fifth to ninety-fifth percentile head translation span by task"></div>
    </article>
    </section>
  </section>
</main>
<script>
const GROUPS = __GROUPS_JSON__;
const POINTS = __POINTS_JSON__;
const INITIAL = __INITIAL_JSON__;
const SCREEN = __SCREEN_JSON__;
const userFilter = document.getElementById("user-filter");
const sessionFilter = document.getElementById("session-filter");
const taskFilter = document.getElementById("task-filter");
const clearTargetsButton = document.getElementById("clear-targets");
const numberFormat = new Intl.NumberFormat();
const taskColors = ["#2f6fbb", "#16a394", "#9b6bc7", "#d0833f", "#d65f73", "#718096"];
const rotationColors = { Pitch: "#2f6fbb", Yaw: "#16a394", Roll: "#9b6bc7" };
const translationColors = { Horizontal: "#2f6fbb", Vertical: "#16a394", Depth: "#9b6bc7" };
const coverageModes = { rotation: "points", translation: "points" };
const coverageAxes = { rotation: "yaw-pitch", translation: "horizontal-depth" };
const axisPairs = {
  rotation: {
    "yaw-pitch": { xField: "head_yaw", yField: "head_pitch", xName: "Yaw", yName: "Pitch", unit: "°", title: "Yaw and pitch coverage by task" },
    "roll-yaw": { xField: "head_roll", yField: "head_yaw", xName: "Roll", yName: "Yaw", unit: "°", title: "Roll and yaw coverage by task" }
  },
  translation: {
    "horizontal-depth": { xField: "head_translation_x", yField: "head_translation_z", xName: "Horizontal x", yName: "Depth z", unit: "cm", title: "Horizontal and depth translation coverage by task" },
    "horizontal-vertical": { xField: "head_translation_x", yField: "head_translation_y", xName: "Horizontal x", yName: "Vertical y", unit: "cm", title: "Horizontal and vertical translation coverage by task" }
  }
};
const selectedTargets = new Set();

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => Number(a) - Number(b));
}

function taskLabel(row) {
  return `Task ${String(row.task).padStart(2, "0")} · ${row.task_type}`;
}

function taskKey(row) {
  return `${row.task}|${row.task_type}`;
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

function matchesBase(row) {
  const user = selectedNumber(userFilter);
  const session = selectedNumber(sessionFilter);
  const task = selectedNumber(taskFilter);
  return (user === null || row.user === user)
    && (session === null || row.session === session)
    && (task === null || row.task === task);
}

function matchesTargets(row) {
  return selectedTargets.size === 0 || selectedTargets.has(row.target_key);
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

function pruneTargetSelection(baseGroups) {
  const available = new Set(baseGroups.map(row => row.target_key).filter(Boolean));
  for (const key of selectedTargets) {
    if (!available.has(key)) selectedTargets.delete(key);
  }
}

function sumBy(rows, keyFunction) {
  const totals = new Map();
  for (const row of rows) {
    const key = keyFunction(row);
    totals.set(key, (totals.get(key) || 0) + row.count);
  }
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

function axisStyle(title) {
  const border = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
  return { title, gridcolor: border, zerolinecolor: border };
}

const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function renderTaskChart(groups) {
  const entries = sumBy(groups, taskLabel);
  const target = document.getElementById("task-chart");
  if (!entries.length) {
    target.innerHTML = '<div class="empty">No data for this filter.</div>';
    return;
  }
  Plotly.react(target, [{
    type: "bar",
    orientation: "h",
    y: entries.map(entry => entry[0]).reverse(),
    x: entries.map(entry => entry[1]).reverse(),
    marker: { color: taskColors[0] },
    text: entries.map(entry => numberFormat.format(entry[1])).reverse(),
    textposition: "auto",
    hovertemplate: "%{y}<br>%{x:,} samples<extra></extra>"
  }], themeLayout({
    xaxis: axisStyle("Samples"),
    yaxis: { automargin: true },
    margin: { l: 145, r: 18, t: 18, b: 52 }
  }), plotConfig);
}

function renderSessionChart(groups) {
  const aggregated = new Map();
  for (const row of groups) {
    const key = `${row.user}|${row.session}|${taskKey(row)}`;
    aggregated.set(key, (aggregated.get(key) || 0) + row.count);
  }
  const tasks = [...new Map(groups.map(row => [taskKey(row), taskLabel(row)])).entries()]
    .sort((a, b) => Number(a[0].split("|")[0]) - Number(b[0].split("|")[0]));
  const traces = tasks.map(([key, label], index) => {
    const rows = [...aggregated.entries()]
      .filter(([entryKey]) => entryKey.endsWith(`|${key}`))
      .map(([entryKey, count]) => {
        const [user, session] = entryKey.split("|").slice(0, 2).map(Number);
        return { user, session, count };
      });
    return {
      type: "scatter",
      mode: "markers",
      name: label,
      x: rows.map(row => row.session),
      y: rows.map(row => `User ${String(row.user).padStart(2, "0")}`),
      customdata: rows.map(row => row.count),
      marker: {
        color: taskColors[index % taskColors.length],
        size: rows.map(row => Math.max(7, Math.min(26, Math.sqrt(row.count) / 1.6))),
        opacity: 0.76
      },
      hovertemplate: "Session %{x}<br>%{y}<br>%{customdata:,} samples<extra>" + label + "</extra>"
    };
  });
  Plotly.react("session-chart", traces, themeLayout({
    xaxis: { ...axisStyle("Session ID"), dtick: 1 },
    yaxis: { title: "Participant", automargin: true },
    legend: { orientation: "h", y: 1.18 },
    margin: { l: 82, r: 20, t: 64, b: 52 }
  }), plotConfig);
}

function orderedTasks(points) {
  return [...new Map(points.map(row => [taskKey(row), taskLabel(row)])).entries()]
    .sort((a, b) => Number(a[0].split("|")[0]) - Number(b[0].split("|")[0]));
}

function renderHeadScatter(points) {
  const pair = axisPairs.rotation[coverageAxes.rotation];
  const valid = points.filter(row => Number.isFinite(row[pair.xField]) && Number.isFinite(row[pair.yField]));
  const target = document.getElementById("head-scatter-chart");
  document.getElementById("rotation-coverage-title").textContent = pair.title;
  target.setAttribute("aria-label", `${pair.xName} and ${pair.yName} head rotation coverage split into task panels`);
  if (!valid.length) {
    target.innerHTML = '<div class="empty">Head rotation columns are unavailable for this filter.</div>';
    return;
  }
  const tasks = orderedTasks(valid);
  const columnWidth = 1 / tasks.length;
  const traces = [];
  const layout = themeLayout({
    showlegend: false,
    margin: { l: 62, r: 18, t: 54, b: 58 },
    annotations: []
  });
  tasks.forEach(([key, label], index) => {
    const rows = valid.filter(row => taskKey(row) === key);
    const axisIndex = index === 0 ? "" : String(index + 1);
    if (coverageModes.rotation === "regions") {
      traces.push({
        type: "histogram2dcontour",
        name: label,
        xaxis: `x${axisIndex}`,
        yaxis: `y${axisIndex}`,
        x: rows.map(row => row[pair.xField]),
        y: rows.map(row => row[pair.yField]),
        histnorm: "probability density",
        ncontours: 7,
        contours: { coloring: "fill", showlines: true },
        colorscale: [[0, "rgba(0,0,0,0)"], [1, taskColors[index % taskColors.length]]],
        line: { width: 1, color: taskColors[index % taskColors.length] },
        showscale: false,
        hovertemplate: `${pair.xName} %{x:.1f}${pair.unit}<br>${pair.yName} %{y:.1f}${pair.unit}<br>Density %{z:.4f}<extra>${label}</extra>`
      });
    } else {
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: label,
        xaxis: `x${axisIndex}`,
        yaxis: `y${axisIndex}`,
        x: rows.map(row => row[pair.xField]),
        y: rows.map(row => row[pair.yField]),
        marker: { size: 4, opacity: 0.34, color: taskColors[index % taskColors.length] },
        text: rows.map(row => `User ${row.user}, session ${row.session}`),
        hovertemplate: `%{text}<br>${pair.xName} %{x:.1f}${pair.unit}<br>${pair.yName} %{y:.1f}${pair.unit}<extra>${label}</extra>`
      });
    }
    const start = index * columnWidth + 0.025;
    const end = (index + 1) * columnWidth - 0.025;
    layout[`xaxis${axisIndex}`] = {
      ...axisStyle(`${pair.xName} (${pair.unit})`),
      domain: [start, end],
      matches: index === 0 ? undefined : "x"
    };
    layout[`yaxis${axisIndex}`] = {
      ...axisStyle(index === 0 ? `${pair.yName} (${pair.unit})` : ""),
      anchor: `x${axisIndex}`,
      matches: index === 0 ? undefined : "y",
      showticklabels: index === 0
    };
    layout.annotations.push({
      text: label,
      x: (start + end) / 2,
      y: 1.08,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 12 }
    });
  });
  Plotly.react(target, traces, layout, plotConfig);
}

function renderHeadDistributions(points) {
  const rotations = [["Pitch", "head_pitch"], ["Yaw", "head_yaw"], ["Roll", "head_roll"]];
  const valid = points.filter(row => rotations.some(([, field]) => Number.isFinite(row[field])));
  const target = document.getElementById("head-distribution-chart");
  if (!valid.length) {
    target.innerHTML = '<div class="empty">Head rotation columns are unavailable for this filter.</div>';
    return;
  }
  const tasks = orderedTasks(valid);
  const traces = rotations.map(([axis, field]) => ({
    type: "box",
    name: axis,
    x: tasks.flatMap(([key, label]) => valid.filter(row => taskKey(row) === key && Number.isFinite(row[field])).map(() => label)),
    y: tasks.flatMap(([key]) => valid.filter(row => taskKey(row) === key && Number.isFinite(row[field])).map(row => row[field])),
    marker: { color: rotationColors[axis], size: 2, opacity: 0.22 },
    line: { color: rotationColors[axis] },
    boxpoints: "outliers",
    hovertemplate: `${axis} %{y:.1f}°<extra>%{x}</extra>`
  }));
  Plotly.react(target, traces, themeLayout({
    boxmode: "group",
    xaxis: { title: "Task", automargin: true },
    yaxis: axisStyle("Head rotation (°)"),
    legend: { orientation: "h", y: 1.08 },
    margin: { l: 62, r: 18, t: 48, b: 78 }
  }), plotConfig);
}

function quantile(sorted, probability) {
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function renderHeadSpan(points) {
  const rotations = [["Pitch", "head_pitch"], ["Yaw", "head_yaw"], ["Roll", "head_roll"]];
  const tasks = orderedTasks(points);
  const rows = [];
  tasks.forEach(([key, label]) => {
    rotations.forEach(([axis, field]) => {
      const values = points
        .filter(row => taskKey(row) === key && Number.isFinite(row[field]))
        .map(row => row[field])
        .sort((a, b) => a - b);
      if (values.length) {
        rows.push({ label: `${label} · ${axis}`, axis, low: quantile(values, 0.05), median: quantile(values, 0.5), high: quantile(values, 0.95) });
      }
    });
  });
  const target = document.getElementById("head-span-chart");
  if (!rows.length) {
    target.innerHTML = '<div class="empty">Head rotation columns are unavailable for this filter.</div>';
    return;
  }
  Plotly.react(target, [{
    type: "scatter",
    mode: "markers",
    x: rows.map(row => row.median),
    y: rows.map(row => row.label),
    marker: { color: rows.map(row => rotationColors[row.axis]), size: 8 },
    error_x: {
      type: "data",
      symmetric: false,
      array: rows.map(row => row.high - row.median),
      arrayminus: rows.map(row => row.median - row.low),
      color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),
      thickness: 2,
      width: 5
    },
    customdata: rows.map(row => [row.low, row.high]),
    hovertemplate: "Median %{x:.1f}°<br>5th–95th: %{customdata[0]:.1f}° to %{customdata[1]:.1f}°<extra>%{y}</extra>"
  }], themeLayout({
    xaxis: axisStyle("Head rotation (°)"),
    yaxis: { automargin: true, autorange: "reversed" },
    margin: { l: 190, r: 24, t: 18, b: 56 },
    showlegend: false
  }), plotConfig);
}

function renderTranslationScatter(points) {
  const pair = axisPairs.translation[coverageAxes.translation];
  const valid = points.filter(row => Number.isFinite(row[pair.xField]) && Number.isFinite(row[pair.yField]));
  const target = document.getElementById("translation-scatter-chart");
  document.getElementById("translation-coverage-title").textContent = pair.title;
  target.setAttribute("aria-label", `${pair.xName} and ${pair.yName} head translation coverage split into task panels`);
  if (!valid.length) {
    target.innerHTML = '<div class="empty">Head translation columns are unavailable for this filter.</div>';
    return;
  }
  const tasks = orderedTasks(valid);
  const columnWidth = 1 / tasks.length;
  const traces = [];
  const layout = themeLayout({
    showlegend: false,
    margin: { l: 68, r: 18, t: 54, b: 58 },
    annotations: []
  });
  tasks.forEach(([key, label], index) => {
    const rows = valid.filter(row => taskKey(row) === key);
    const axisIndex = index === 0 ? "" : String(index + 1);
    if (coverageModes.translation === "regions") {
      traces.push({
        type: "histogram2dcontour",
        name: label,
        xaxis: `x${axisIndex}`,
        yaxis: `y${axisIndex}`,
        x: rows.map(row => row[pair.xField]),
        y: rows.map(row => row[pair.yField]),
        histnorm: "probability density",
        ncontours: 7,
        contours: { coloring: "fill", showlines: true },
        colorscale: [[0, "rgba(0,0,0,0)"], [1, taskColors[index % taskColors.length]]],
        line: { width: 1, color: taskColors[index % taskColors.length] },
        showscale: false,
        hovertemplate: `${pair.xName} %{x:.1f} ${pair.unit}<br>${pair.yName} %{y:.1f} ${pair.unit}<br>Density %{z:.4f}<extra>${label}</extra>`
      });
    } else {
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: label,
        xaxis: `x${axisIndex}`,
        yaxis: `y${axisIndex}`,
        x: rows.map(row => row[pair.xField]),
        y: rows.map(row => row[pair.yField]),
        marker: { size: 4, opacity: 0.34, color: taskColors[index % taskColors.length] },
        text: rows.map(row => `User ${row.user}, session ${row.session}`),
        hovertemplate: `%{text}<br>${pair.xName} %{x:.1f} ${pair.unit}<br>${pair.yName} %{y:.1f} ${pair.unit}<extra>${label}</extra>`
      });
    }
    const start = index * columnWidth + 0.025;
    const end = (index + 1) * columnWidth - 0.025;
    layout[`xaxis${axisIndex}`] = {
      ...axisStyle(`${pair.xName} (${pair.unit})`),
      domain: [start, end],
      matches: index === 0 ? undefined : "x"
    };
    layout[`yaxis${axisIndex}`] = {
      ...axisStyle(index === 0 ? `${pair.yName} (${pair.unit})` : ""),
      anchor: `x${axisIndex}`,
      matches: index === 0 ? undefined : "y",
      showticklabels: index === 0
    };
    layout.annotations.push({
      text: label,
      x: (start + end) / 2,
      y: 1.08,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 12 }
    });
  });
  Plotly.react(target, traces, layout, plotConfig);
}

function renderTranslationDistributions(points) {
  const dimensions = [
    ["Horizontal", "head_translation_x"],
    ["Vertical", "head_translation_y"],
    ["Depth", "head_translation_z"]
  ];
  const valid = points.filter(row => dimensions.some(([, field]) => Number.isFinite(row[field])));
  const target = document.getElementById("translation-distribution-chart");
  if (!valid.length) {
    target.innerHTML = '<div class="empty">Head translation columns are unavailable for this filter.</div>';
    return;
  }
  const tasks = orderedTasks(valid);
  const traces = dimensions.map(([axis, field]) => ({
    type: "box",
    name: axis,
    x: tasks.flatMap(([key, label]) => valid.filter(row => taskKey(row) === key && Number.isFinite(row[field])).map(() => label)),
    y: tasks.flatMap(([key]) => valid.filter(row => taskKey(row) === key && Number.isFinite(row[field])).map(row => row[field])),
    marker: { color: translationColors[axis], size: 2, opacity: 0.22 },
    line: { color: translationColors[axis] },
    boxpoints: "outliers",
    hovertemplate: `${axis} %{y:.1f} cm<extra>%{x}</extra>`
  }));
  Plotly.react(target, traces, themeLayout({
    boxmode: "group",
    xaxis: { title: "Task", automargin: true },
    yaxis: axisStyle("Head translation (cm)"),
    legend: { orientation: "h", y: 1.08 },
    margin: { l: 68, r: 18, t: 48, b: 78 }
  }), plotConfig);
}

function renderTranslationSpan(points) {
  const dimensions = [
    ["Horizontal", "head_translation_x"],
    ["Vertical", "head_translation_y"],
    ["Depth", "head_translation_z"]
  ];
  const tasks = orderedTasks(points);
  const rows = [];
  tasks.forEach(([key, label]) => {
    dimensions.forEach(([axis, field]) => {
      const values = points
        .filter(row => taskKey(row) === key && Number.isFinite(row[field]))
        .map(row => row[field])
        .sort((a, b) => a - b);
      if (values.length) {
        rows.push({ label: `${label} · ${axis}`, axis, low: quantile(values, 0.05), median: quantile(values, 0.5), high: quantile(values, 0.95) });
      }
    });
  });
  const target = document.getElementById("translation-span-chart");
  if (!rows.length) {
    target.innerHTML = '<div class="empty">Head translation columns are unavailable for this filter.</div>';
    return;
  }
  Plotly.react(target, [{
    type: "scatter",
    mode: "markers",
    x: rows.map(row => row.median),
    y: rows.map(row => row.label),
    marker: { color: rows.map(row => translationColors[row.axis]), size: 8 },
    error_x: {
      type: "data",
      symmetric: false,
      array: rows.map(row => row.high - row.median),
      arrayminus: rows.map(row => row.median - row.low),
      color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),
      thickness: 2,
      width: 5
    },
    customdata: rows.map(row => [row.low, row.high]),
    hovertemplate: "Median %{x:.1f} cm<br>5th–95th: %{customdata[0]:.1f} to %{customdata[1]:.1f} cm<extra>%{y}</extra>"
  }], themeLayout({
    xaxis: axisStyle("Head translation (cm)"),
    yaxis: { automargin: true, autorange: "reversed" },
    margin: { l: 205, r: 24, t: 18, b: 56 },
    showlegend: false
  }), plotConfig);
}

function aggregateTargets(groups) {
  const targets = new Map();
  for (const row of groups) {
    if (!row.target_key) continue;
    if (!targets.has(row.target_key)) {
      targets.set(row.target_key, { key: row.target_key, pxX: row.target_px_x, pxY: row.target_px_y, cmX: row.target_cm_x, cmY: row.target_cm_y, count: 0, taskTypes: new Set() });
    }
    const target = targets.get(row.target_key);
    target.count += row.count;
    target.taskTypes.add(row.task_type);
  }
  return [...targets.values()].sort((a, b) => a.cmY - b.cmY || a.cmX - b.cmX);
}

function renderTargetChart(baseGroups) {
  const targets = aggregateTargets(baseGroups);
  const targetElement = document.getElementById("target-chart");
  if (!targets.length) {
    targetElement.innerHTML = '<div class="empty">Target-coordinate columns are unavailable for this filter.</div>';
    document.getElementById("target-selection").textContent = "No targets available";
    clearTargetsButton.disabled = true;
    return;
  }
  const hasSelection = selectedTargets.size > 0;
  const taskTypes = [...new Set(targets.flatMap(target => [...target.taskTypes]))].sort();
  const traces = taskTypes.map((taskType, index) => {
    const rows = targets.filter(target => target.taskTypes.has(taskType));
    return {
      type: "scatter",
      mode: "markers",
      name: taskType,
      x: rows.map(row => row.cmX),
      y: rows.map(row => row.cmY),
      customdata: rows.map(row => [row.key, row.pxX, row.pxY, row.count]),
      marker: {
        color: taskColors[index % taskColors.length],
        size: rows.map(row => selectedTargets.has(row.key) ? 18 : 13),
        opacity: rows.map(row => !hasSelection || selectedTargets.has(row.key) ? 0.9 : 0.28),
        line: {
          color: rows.map(row => selectedTargets.has(row.key) ? getComputedStyle(document.documentElement).getPropertyValue("--text").trim() : getComputedStyle(document.documentElement).getPropertyValue("--panel").trim()),
          width: rows.map(row => selectedTargets.has(row.key) ? 3 : 1)
        }
      },
      hovertemplate: "x %{x:.2f} cm · y %{y:.2f} cm<br>Pixel position %{customdata[1]:.0f}, %{customdata[2]:.0f}<br>%{customdata[3]:,} samples<extra>" + taskType + "</extra>"
    };
  });
  Plotly.react(targetElement, traces, themeLayout({
    xaxis: { ...axisStyle("Screen x (cm)"), range: [-0.5, SCREEN.width_cm + 0.5], constrain: "domain" },
    yaxis: { ...axisStyle("Screen y (cm)"), range: [SCREEN.height_cm + 0.5, -0.5], scaleanchor: "x", scaleratio: 1 },
    legend: { orientation: "h", y: 1.08 },
    margin: { l: 66, r: 22, t: 48, b: 58 },
    clickmode: "event"
  }), { ...plotConfig, scrollZoom: false });

  if (!targetElement.__targetHandlerBound) {
    targetElement.on("plotly_click", event => {
      const key = event.points[0].customdata[0];
      if (selectedTargets.has(key)) selectedTargets.delete(key);
      else selectedTargets.add(key);
      update();
    });
    targetElement.__targetHandlerBound = true;
  }
  document.getElementById("target-selection").textContent = hasSelection
    ? `${selectedTargets.size} of ${targets.length} targets selected`
    : `All ${targets.length} targets included`;
  clearTargetsButton.disabled = !hasSelection;
}

function renderTable(groups) {
  const sessions = new Map();
  for (const row of groups) {
    const key = `${row.user}|${row.session}`;
    if (!sessions.has(key)) {
      sessions.set(key, { user: row.user, session: row.session, samples: 0, tasks: new Set(), targets: new Set(), calibration: 0, test: 0 });
    }
    const item = sessions.get(key);
    item.samples += row.count;
    item.tasks.add(row.task);
    if (row.target_key) item.targets.add(row.target_key);
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
      numberFormat.format(item.tasks.size),
      numberFormat.format(item.targets.size),
      numberFormat.format(item.calibration),
      numberFormat.format(item.test)
    ];
    for (const value of values) row.insertCell().textContent = value;
  }
}

function update() {
  const baseGroups = GROUPS.filter(matchesBase);
  const basePoints = POINTS.filter(matchesBase);
  pruneTargetSelection(baseGroups);
  const groups = baseGroups.filter(matchesTargets);
  const points = basePoints.filter(matchesTargets);
  const total = groups.reduce((sum, row) => sum + row.count, 0);
  const users = new Set(groups.map(row => row.user)).size;
  const sessions = new Set(groups.map(row => `${row.user}|${row.session}`)).size;
  const tasks = new Set(groups.map(row => `${row.user}|${row.session}|${row.task}`)).size;

  document.getElementById("metric-samples").textContent = numberFormat.format(total);
  document.getElementById("metric-users").textContent = numberFormat.format(users);
  document.getElementById("metric-sessions").textContent = numberFormat.format(sessions);
  document.getElementById("metric-tasks").textContent = numberFormat.format(tasks);

  renderTargetChart(baseGroups);
  renderTaskChart(groups);
  renderSessionChart(groups);
  renderHeadScatter(points);
  renderHeadDistributions(points);
  renderHeadSpan(points);
  renderTranslationScatter(points);
  renderTranslationDistributions(points);
  renderTranslationSpan(points);
  renderTable(groups);
}

function setCoverageMode(kind, mode) {
  coverageModes[kind] = mode;
  document.getElementById(`${kind}-points`).setAttribute("aria-pressed", String(mode === "points"));
  document.getElementById(`${kind}-regions`).setAttribute("aria-pressed", String(mode === "regions"));
  const points = POINTS.filter(matchesBase).filter(matchesTargets);
  if (kind === "rotation") renderHeadScatter(points);
  else renderTranslationScatter(points);
}

function setCoverageAxes(kind, axes) {
  if (!axisPairs[kind][axes]) return;
  coverageAxes[kind] = axes;
  const points = POINTS.filter(matchesBase).filter(matchesTargets);
  if (kind === "rotation") renderHeadScatter(points);
  else renderTranslationScatter(points);
}

const tabButtons = [...document.querySelectorAll('[role="tab"]')];

function activateTab(button) {
  for (const tab of tabButtons) {
    const selected = tab === button;
    tab.setAttribute("aria-selected", String(selected));
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  const panel = document.getElementById(button.getAttribute("aria-controls"));
  requestAnimationFrame(() => {
    for (const plot of panel.querySelectorAll(".js-plotly-plot")) {
      Plotly.Plots.resize(plot);
    }
  });
}

function initializeCollapsiblePanels() {
  const panels = [...document.querySelectorAll("#distributions-panel > .grid > .panel")];
  panels.forEach((panel, index) => {
    let heading = [...panel.children].find(child => child.classList.contains("panel-heading"));
    if (!heading) {
      heading = document.createElement("div");
      heading.className = "panel-heading";
      const title = panel.querySelector(":scope > h2");
      panel.insertBefore(heading, panel.firstChild);
      heading.appendChild(title);
    }
    heading.classList.add("collapsible-heading");

    const content = document.createElement("div");
    content.className = "collapsible-content";
    content.id = `distribution-panel-content-${index + 1}`;
    for (const child of [...panel.children]) {
      if (child !== heading) content.appendChild(child);
    }
    panel.appendChild(content);

    const title = heading.querySelector("h2");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "collapse-toggle";
    toggle.setAttribute("aria-controls", content.id);
    toggle.setAttribute("aria-expanded", "true");
    toggle.textContent = "Minimize";
    heading.appendChild(toggle);

    function setExpanded(expanded) {
      content.hidden = !expanded;
      panel.classList.toggle("is-collapsed", !expanded);
      toggle.setAttribute("aria-expanded", String(expanded));
      toggle.textContent = expanded ? "Minimize" : "Open";
      toggle.setAttribute("aria-label", `${expanded ? "Minimize" : "Open"} ${title.textContent}`);
      if (expanded) {
        requestAnimationFrame(() => {
          for (const plot of content.querySelectorAll(".js-plotly-plot")) {
            Plotly.Plots.resize(plot);
          }
        });
      }
    }

    toggle.addEventListener("click", event => {
      event.stopPropagation();
      setExpanded(toggle.getAttribute("aria-expanded") !== "true");
    });
    heading.addEventListener("click", event => {
      if (event.target.closest("button, select, label")) return;
      setExpanded(toggle.getAttribute("aria-expanded") !== "true");
    });
  });
}

tabButtons.forEach((button, index) => {
  button.addEventListener("click", () => activateTab(button));
  button.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let nextIndex = index;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabButtons.length) % tabButtons.length;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabButtons.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabButtons.length - 1;
    tabButtons[nextIndex].focus();
    activateTab(tabButtons[nextIndex]);
  });
});

initializeCollapsiblePanels();

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
document.getElementById("rotation-axes").addEventListener("change", event => setCoverageAxes("rotation", event.target.value));
document.getElementById("rotation-points").addEventListener("click", () => setCoverageMode("rotation", "points"));
document.getElementById("rotation-regions").addEventListener("click", () => setCoverageMode("rotation", "regions"));
document.getElementById("translation-axes").addEventListener("change", event => setCoverageAxes("translation", event.target.value));
document.getElementById("translation-points").addEventListener("click", () => setCoverageMode("translation", "points"));
document.getElementById("translation-regions").addEventListener("click", () => setCoverageMode("translation", "regions"));
clearTargetsButton.addEventListener("click", () => {
  selectedTargets.clear();
  update();
});
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
        .replace("__SCREEN_JSON__", json_for_script(screen_record))
    )


def main(argv=None):
    args, parser = parse_args(argv)
    screen = read_screen_geometry(args.setup_config, parser)
    frame = read_summary(args.csv, parser)
    validate_initial_selection(frame, args, parser)
    report = build_report(frame, args, screen)

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
