#!/usr/bin/env python3
"""Generate a standalone explorer for an independent-dataset comparison report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import webbrowser

import pandas as pd
from plotly.offline import get_plotlyjs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "temp" / "dataset_comparison"
METRICS = (
    "location_shift",
    "dispersion_shift",
    "distribution_shift",
    "dependence_shift",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML explorer for a two-dataset domain-gap report."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Dataset-comparison report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: REPORT_DIR/dataset_comparison_explorer.html).",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated explorer.")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.report_dir = args.report_dir.expanduser().resolve()
    required = (
        "manifest.json",
        "dataset_distances.csv",
        "comparison_coverage.csv",
        "column_compatibility.csv",
    )
    missing = [name for name in required if not (args.report_dir / name).is_file()]
    if missing:
        parser.error("Report directory is missing: " + ", ".join(missing))
    args.output = (
        args.output.expanduser().resolve()
        if args.output
        else args.report_dir / "dataset_comparison_explorer.html"
    )
    return args, parser


def read_report(report_dir: Path) -> dict:
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("report_type") != "independent_dataset_comparison":
        raise ValueError("manifest.json is not an independent-dataset comparison report")
    distances = pd.read_csv(report_dir / "dataset_distances.csv", low_memory=False)
    coverage = pd.read_csv(report_dir / "comparison_coverage.csv", low_memory=False)
    compatibility = pd.read_csv(
        report_dir / "column_compatibility.csv", low_memory=False
    )
    required = {
        "dataset_a", "dataset_b", "view", "family", "metric",
        "estimate", "ci_low", "ci_high", "p_value", "q_value", "status",
    }
    missing = sorted(required - set(distances.columns))
    if missing:
        raise ValueError("dataset_distances.csv is missing: " + ", ".join(missing))
    return {
        "manifest": manifest,
        "distances": distances,
        "coverage": coverage,
        "compatibility": compatibility,
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    values = frame.astype(object).where(pd.notna(frame), None)
    return values.to_dict(orient="records")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )


def build_html(report: dict, source_name: str) -> str:
    metadata = report["manifest"].get("metadata", {})
    meta_a = metadata.get("dataset_a", {})
    meta_b = metadata.get("dataset_b", {})
    label_a = str(meta_a.get("label", "dataset A"))
    label_b = str(meta_b.get("label", "dataset B"))
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Independent Dataset Domain-Gap Explorer</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fb; --panel: #ffffff; --text: #172033; --muted: #637089;
      --border: #d8dfeb; --series: #275dad; --warning-bg: #fff5d9;
      --warning-text: #684d00; --shadow: 0 8px 24px rgba(24,40,72,.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111722; --panel: #182131; --text: #edf2fa; --muted: #aeb9cc;
        --border: #344258; --series: #78a9f5; --warning-bg: #3b321a;
        --warning-text: #f3d887; --shadow: none;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    main { width: min(1450px, 100%); margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 20px; flex-wrap: wrap; }
    h1 { margin: 0; font-size: clamp(1.5rem, 3vw, 2.25rem); font-weight: 500; }
    h2 { margin: 0 0 8px; font-size: 1.05rem; font-weight: 500; }
    p { margin: 6px 0 0; color: var(--muted); }
    .note { font-size: .84rem; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0 12px; }
    label { display: grid; min-width: 180px; gap: 6px; color: var(--muted); font-size: .84rem; }
    select { min-height: 38px; padding: 7px 10px; color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 8px; font: inherit; }
    select:focus { outline: 2px solid var(--series); outline-offset: 1px; }
    .warning { display: none; margin: 0 0 14px; padding: 10px 12px; color: var(--warning-text); background: var(--warning-bg); border-radius: 8px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .panel { min-width: 0; padding: 16px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }
    .wide { grid-column: 1 / -1; }
    .chart { width: 100%; min-height: 430px; }
    .empty { min-height: 360px; display: grid; place-items: center; color: var(--muted); text-align: center; }
    .table-wrap { overflow-x: auto; max-height: 520px; }
    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--border); text-align: right; }
    th { position: sticky; top: 0; z-index: 1; background: var(--panel); color: var(--muted); font-weight: 500; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    td { font-variant-numeric: tabular-nums; }
    @media (max-width: 850px) { main { padding: 16px; } .grid { grid-template-columns: 1fr; } .wide { grid-column: auto; } }
  </style>
  <script>__PLOTLY__</script>
</head>
<body>
<main>
  <header>
    <div><h1>__LABEL_A__ vs __LABEL_B__</h1><p>Independent-dataset, session-balanced domain-gap vector</p></div>
    <p class="note">__A_SAMPLES__ / __B_SAMPLES__ accepted samples · __A_SESSIONS__ / __B_SESSIONS__ sessions</p>
  </header>
  <section class="controls" aria-label="Dataset comparison filters">
    <label>View<select id="view-filter"></select></label>
    <label>Displayed value<select id="stat-filter"></select></label>
  </section>
  <div id="warning" class="warning" role="status"></div>
  <section class="grid">
    <article class="panel"><h2>Location shift</h2><div id="location-chart" class="chart" role="img" aria-label="Location-shift domain gaps"></div></article>
    <article class="panel"><h2>Dispersion shift</h2><div id="dispersion-chart" class="chart" role="img" aria-label="Dispersion-shift domain gaps"></div></article>
    <article class="panel"><h2>Distribution shift</h2><div id="distribution-chart" class="chart" role="img" aria-label="Distribution-shift domain gaps"></div></article>
    <article class="panel"><h2>Dependence shift</h2><div id="dependence-chart" class="chart" role="img" aria-label="Dependence-shift domain gaps"></div></article>
    <article class="panel wide"><h2>Comparison coverage</h2><div class="table-wrap"><table><thead><tr><th>Family</th><th>Dataset</th><th>Users</th><th>Sessions</th><th>Samples</th><th>Session descriptors</th><th>User descriptors</th><th>Common targets</th></tr></thead><tbody id="coverage-table"></tbody></table></div></article>
    <article class="panel wide"><h2>Column compatibility</h2><div class="table-wrap"><table><thead><tr><th>Header</th><th>Role</th><th>Status</th><th>Non-null A</th><th>Non-null B</th><th>Type A</th><th>Type B</th></tr></thead><tbody id="compatibility-table"></tbody></table></div></article>
    <article class="panel wide"><h2>Exact KPI vector</h2><div class="table-wrap"><table><thead><tr><th>Family</th><th>KPI</th><th>Estimate</th><th>95% CI</th><th>p</th><th>q</th><th>Status</th></tr></thead><tbody id="result-table"></tbody></table></div></article>
  </section>
</main>
<script>
const DISTANCES = __DISTANCES__;
const COVERAGE = __COVERAGE__;
const COMPATIBILITY = __COMPATIBILITY__;
const METRICS = ["location_shift", "dispersion_shift", "distribution_shift", "dependence_shift"];
const CHART_IDS = {location_shift:"location-chart", dispersion_shift:"dispersion-chart", distribution_shift:"distribution-chart", dependence_shift:"dependence-chart"};
const STATISTIC_LABELS = {estimate:"Estimate", ci_low:"95% CI lower bound", ci_high:"95% CI upper bound", p_value:"p-value", q_value:"FDR-adjusted q-value"};
const viewFilter = document.getElementById("view-filter");
const statFilter = document.getElementById("stat-filter");
const number = new Intl.NumberFormat(undefined, {maximumFractionDigits:4});
const integer = new Intl.NumberFormat();
const isNumber = value => typeof value === "number" && Number.isFinite(value);
const format = value => isNumber(value) ? number.format(value) : "—";
const label = value => String(value).replaceAll("_", " ");
const statisticLabel = value => STATISTIC_LABELS[value] || label(value);
const unique = values => [...new Set(values)].sort();

function options(select, values, preferred) {
  const current = select.value || preferred;
  select.replaceChildren();
  for (const value of values) { const item = document.createElement("option"); item.value=value; item.textContent=statisticLabel(value); select.appendChild(item); }
  if (values.includes(current)) select.value=current;
}
function theme() { const s=getComputedStyle(document.documentElement); const get=n=>s.getPropertyValue(n).trim(); return {text:get("--text"),muted:get("--muted"),border:get("--border"),panel:get("--panel"),series:get("--series")}; }
function layout(title) { const c=theme(); return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:c.text,family:"Inter,system-ui,sans-serif"},margin:{l:145,r:24,t:18,b:62},xaxis:{title,gridcolor:c.border,zerolinecolor:c.border},yaxis:{automargin:true},hoverlabel:{bgcolor:c.panel,bordercolor:c.border,font:{color:c.text}},showlegend:false}; }
const plotConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"]};

function selectedRows() { return DISTANCES.filter(row => row.view===viewFilter.value); }
function hoverDetails(row, statistic) {
  const lines=[label(row.family), `${statisticLabel(statistic)} ${format(row[statistic])}`];
  if (statistic!=="estimate" && isNumber(row.estimate)) lines.push(`Estimate ${format(row.estimate)}`);
  if (isNumber(row.ci_low) && isNumber(row.ci_high)) lines.push(`95% CI ${format(row.ci_low)}–${format(row.ci_high)}`);
  if (isNumber(row.p_value)) lines.push(`p-value ${format(row.p_value)}`);
  if (isNumber(row.q_value)) lines.push(`q-value ${format(row.q_value)}`);
  if (row.status && row.status!=="ok") lines.push(`Status: ${label(row.status)}`);
  return lines.join("<br>");
}
function renderMetric(metric, rows) {
  const target=CHART_IDS[metric]; const statistic=statFilter.value;
  const values=rows.filter(row => row.metric===metric && isNumber(row[statistic])).sort((a,b)=>a[statistic]-b[statistic]);
  if (!values.length) { document.getElementById(target).innerHTML='<div class="empty">No values are available for this selection.</div>'; return; }
  const trace={type:"bar",orientation:"h",y:values.map(row=>label(row.family)),x:values.map(row=>row[statistic]),marker:{color:theme().series},customdata:values.map(row=>hoverDetails(row,statistic)),hovertemplate:"%{customdata}<extra></extra>"};
  if (statistic==="estimate" && values.some(row=>isNumber(row.ci_low)&&isNumber(row.ci_high))) trace.error_x={type:"data",symmetric:false,array:values.map(row=>isNumber(row.ci_high)?Math.max(0,row.ci_high-row.estimate):0),arrayminus:values.map(row=>isNumber(row.ci_low)?Math.max(0,row.estimate-row.ci_low):0),visible:true};
  Plotly.react(target,[trace],layout(statisticLabel(statistic)),plotConfig);
}
function renderCoverage() {
  const body=document.getElementById("coverage-table"); body.replaceChildren();
  const rows=COVERAGE.filter(row=>row.view===viewFilter.value).sort((a,b)=>a.family.localeCompare(b.family)||a.dataset.localeCompare(b.dataset));
  for (const item of rows) { const row=body.insertRow(); const values=[label(item.family),item.dataset,integer.format(item.n_users),integer.format(item.n_sessions),integer.format(item.n_samples),integer.format(item.n_session_descriptor_units),integer.format(item.n_user_descriptor_units),isNumber(item.n_common_targets)?integer.format(item.n_common_targets):"—"]; for (const value of values) row.insertCell().textContent=value; }
}
function renderCompatibility() {
  const body=document.getElementById("compatibility-table"); body.replaceChildren();
  const rows=[...COMPATIBILITY].sort((a,b)=>(a.comparable===b.comparable?0:a.comparable?1:-1)||a.column.localeCompare(b.column));
  for (const item of rows) { const row=body.insertRow(); const values=[item.column,label(item.role),label(item.status),integer.format(item.accepted_non_null_a),integer.format(item.accepted_non_null_b),item.dtype_a,item.dtype_b]; for (const value of values) row.insertCell().textContent=value; }
}
function renderResults(rows) {
  const body=document.getElementById("result-table"); body.replaceChildren();
  for (const item of [...rows].sort((a,b)=>a.family.localeCompare(b.family)||a.metric.localeCompare(b.metric))) { const row=body.insertRow(); const values=[label(item.family),label(item.metric),format(item.estimate),isNumber(item.ci_low)||isNumber(item.ci_high)?`${format(item.ci_low)}–${format(item.ci_high)}`:"—",format(item.p_value),format(item.q_value),label(item.status)]; for (const value of values) row.insertCell().textContent=value; }
}
function renderWarning(rows) {
  const messages=[];
  if (!rows.some(row=>isNumber(row.ci_low)&&isNumber(row.ci_high)) && !rows.some(row=>isNumber(row.p_value)||isNumber(row.q_value))) messages.push("95% confidence intervals and p/q values are unavailable because this report was generated without bootstrap or permutation iterations.");
  if (statFilter.value!=="estimate"&&!rows.some(row=>isNumber(row[statFilter.value]))) messages.push("This report was generated without the selected inferential statistic.");
  if (viewFilter.value==="target_conditioned"&&rows.some(row=>row.family==="gaze_point")) messages.push("Target-conditioned gaze-point gaps are expected to be zero because gaze point is determined by target position.");
  if (rows.some(row=>row.status!=="ok")) messages.push("Some families lack sufficient data or common targets; inspect the status and coverage tables.");
  const warning=document.getElementById("warning"); warning.textContent=messages.join(" "); warning.style.display=messages.length?"block":"none";
}
function update() { const rows=selectedRows(); renderWarning(rows); for (const metric of METRICS) renderMetric(metric,rows); renderCoverage(); renderResults(rows); }
options(viewFilter,unique(DISTANCES.map(row=>row.view)),"raw");
options(statFilter,["estimate","ci_low","ci_high","p_value","q_value"].filter(stat=>DISTANCES.some(row=>isNumber(row[stat]))),"estimate");
for (const control of [viewFilter,statFilter]) control.addEventListener("change",update);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",update);
renderCompatibility();
update();
</script>
</body>
</html>"""
    return (
        template.replace("__PLOTLY__", get_plotlyjs())
        .replace("__LABEL_A__", html.escape(label_a))
        .replace("__LABEL_B__", html.escape(label_b))
        .replace("__A_SAMPLES__", f"{int(meta_a.get('accepted_rows', 0)):,}")
        .replace("__B_SAMPLES__", f"{int(meta_b.get('accepted_rows', 0)):,}")
        .replace("__A_SESSIONS__", str(meta_a.get("user_sessions", "—")))
        .replace("__B_SESSIONS__", str(meta_b.get("user_sessions", "—")))
        .replace("__DISTANCES__", _json(_records(report["distances"])))
        .replace("__COVERAGE__", _json(_records(report["coverage"])))
        .replace("__COMPATIBILITY__", _json(_records(report["compatibility"])))
        .replace("__SOURCE__", html.escape(source_name))
    )


def main(argv=None) -> int:
    args, parser = parse_args(argv)
    try:
        report = read_report(args.report_dir)
        document = build_html(report, args.report_dir.name)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Created dataset-comparison explorer: {args.output} ({len(report['distances']):,} KPI rows)")
    if args.open:
        webbrowser.open(args.output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
