#!/usr/bin/env python3
"""Generate a standalone explorer for a gaze-vector consistency report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import webbrowser

import pandas as pd
from plotly.offline import get_plotlyjs


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "temp" / "statistical_reports" / "gaze_consistency"
REQUIRED_FILES = (
    "manifest.json",
    "consistency_summary.csv",
    "consistency_units.csv",
    "label_offsets.csv",
    "session_deviations.csv",
)
VECTOR_LABELS = {
    "recalib": "ReCalib label",
    "eth": "ETH-XGaze label",
    "eth_pred": "ETH-XGaze prediction",
}
IDENTITY_DTYPES = {"user_id": "string", "session_id": "string", "task_type": "string"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML explorer for a gaze-consistency report."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Gaze-consistency report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: REPORT_DIR/gaze_consistency_explorer.html).",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated explorer.")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.report_dir = args.report_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (args.report_dir / name).is_file()]
    if missing:
        parser.error("Report directory is missing: " + ", ".join(missing))
    args.output = (
        args.output.expanduser().resolve()
        if args.output
        else args.report_dir / "gaze_consistency_explorer.html"
    )
    return args, parser


def read_report(report_dir: Path) -> dict:
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("report_type") != "gaze_consistency":
        raise ValueError("manifest.json is not a gaze-consistency report")
    tables = {
        name: pd.read_csv(report_dir / f"{name}.csv", dtype=IDENTITY_DTYPES, low_memory=False)
        for name in ("consistency_summary", "consistency_units", "label_offsets", "session_deviations")
    }
    required = {
        "level", "view", "task_type", "vector_a", "vector_b", "component",
        "pearson_r", "spearman_rho", "sd_ratio", "slope", "diff_sd_deg",
    }
    missing = sorted(required - set(tables["consistency_summary"].columns))
    if missing:
        raise ValueError("consistency_summary.csv is missing: " + ", ".join(missing))
    return {
        "manifest": manifest,
        "summary": tables["consistency_summary"],
        "units": tables["consistency_units"],
        "offsets": tables["label_offsets"],
        "deviations": tables["session_deviations"],
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
    vectors = list(metadata.get("vectors", {}))
    excluded = [
        f"{int(metadata[key]):,} {text}"
        for key, text in (
            ("discarded_rows_excluded", "discarded"),
            ("eth_unavailable_rows_excluded", "without ETH-XGaze output"),
            ("invalid_vector_rows_excluded", "with invalid vectors"),
            ("small_session_rows_excluded", "in sessions below the minimum size"),
        )
        if metadata.get(key)
    ]
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gaze Vector Consistency</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fb; --panel: #ffffff; --text: #172033; --muted: #637089;
      --border: #d8dfeb; --series: #275dad; --reference: #8a94a6; --warning-bg: #fff5d9;
      --warning-text: #684d00; --shadow: 0 8px 24px rgba(24,40,72,.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111722; --panel: #182131; --text: #edf2fa; --muted: #aeb9cc;
        --border: #344258; --series: #78a9f5; --reference: #7d889b; --warning-bg: #3b321a;
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
    .panel { position: relative; min-width: 0; padding: 16px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }
    .panel-heading { display: flex; align-items: center; gap: 7px; margin-bottom: 8px; }
    .panel-heading h2 { margin: 0; }
    .help-button { display: inline-grid; place-items: center; width: 21px; height: 21px; padding: 0; border: 1px solid var(--border); border-radius: 50%; color: var(--text); background: transparent; font: 500 .78rem/1 system-ui, sans-serif; cursor: help; }
    .help-button:focus { outline: 2px solid var(--series); outline-offset: 2px; }
    .help-popup { position: absolute; z-index: 10; top: 48px; left: 16px; right: 16px; visibility: hidden; opacity: 0; padding: 10px 12px; color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 8px; box-shadow: var(--shadow); font-size: .84rem; line-height: 1.4; transition: opacity .12s ease; }
    .help-button:hover + .help-popup, .help-button:focus + .help-popup { visibility: visible; opacity: 1; }
    .wide { grid-column: 1 / -1; }
    .chart { width: 100%; min-height: 430px; }
    .empty { min-height: 360px; display: grid; place-items: center; color: var(--muted); text-align: center; }
    .table-wrap { overflow-x: auto; max-height: 520px; }
    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--border); text-align: right; }
    th { position: sticky; top: 0; z-index: 1; background: var(--panel); color: var(--muted); font-weight: 500; }
    th:first-child, td:first-child { text-align: left; }
    td { font-variant-numeric: tabular-nums; }
    @media (max-width: 850px) { main { padding: 16px; } .grid { grid-template-columns: 1fr; } .wide { grid-column: auto; } }
  </style>
  <script>__PLOTLY__</script>
</head>
<body>
<main>
  <header>
    <div><h1>Gaze vector consistency</h1><p>Do the label sets vary together once each one's own offset is removed?</p></div>
    <p class="note">__SAMPLES__ accepted samples · __USERS__ users · __SESSIONS__ sessions · excluded: __EXCLUDED__</p>
  </header>
  <section class="controls" aria-label="Gaze consistency filters">
    <label>Task type<select id="task-filter"></select></label>
    <label>Vector pair<select id="pair-filter"></select></label>
    <label>Angle<select id="component-filter"></select></label>
  </section>
  <div id="warning" class="warning" role="status"></div>
  <section class="grid">
    <article class="panel wide"><div class="panel-heading"><h2>Agreement by level</h2><button class="help-button" type="button" aria-label="Explain agreement levels" aria-describedby="help-summary">?</button><div id="help-summary" class="help-popup" role="tooltip">Each vector is centred on its own mean at the level shown, so constant offsets do not count. Within session, raw: variation includes the target layout (a sanity check). Within session, per target: frame-to-frame variation while one target is fixated, i.e. head motion moving the gaze origin. Across sessions: each session's per-target means relative to the user's mean for that target. Consistent variation means r and the SD ratio near 1 and a small difference SD. Pearson r is sensitive to a few extreme samples; compare it with Spearman ρ and the per-unit median.</div></div><div class="table-wrap"><table><thead><tr><th>Level</th><th>Sessions</th><th>Samples</th><th>Pearson r (95% CI)</th><th>Spearman ρ</th><th>Median unit r</th><th>SD A (°)</th><th>SD B (°)</th><th>SD ratio B/A (95% CI)</th><th>Slope B on A (95% CI)</th><th>SD of B − A (°)</th></tr></thead><tbody id="summary-table"></tbody></table></div></article>
    <article class="panel"><div class="panel-heading"><h2>Agreement per unit</h2><button class="help-button" type="button" aria-label="Explain per-unit agreement" aria-describedby="help-units">?</button><div id="help-units" class="help-popup" role="tooltip">Pearson r computed separately for every session (within-session levels) or every user across their sessions (across-session level). One dot per unit; hover for the unit and its spreads. Units far to the left are where the two vectors stop moving together.</div></div><div id="units-chart" class="chart" role="img" aria-label="Per-unit Pearson correlation by level"></div></article>
    <article class="panel"><div class="panel-heading"><h2>Session-to-session drift</h2><button class="help-button" type="button" aria-label="Explain session drift" aria-describedby="help-drift">?</button><div id="help-drift" class="help-popup" role="tooltip">One dot per session: how far that session's per-target mean angle sits from the same user's mean for those targets, for vector A (x) and vector B (y). Dots on the dashed identity line drift by the same amount in both vectors.</div></div><div id="drift-chart" class="chart" role="img" aria-label="Session deviation of vector B against vector A"></div></article>
    <article class="panel wide"><div class="panel-heading"><h2>Constant offset per session</h2><button class="help-button" type="button" aria-label="Explain offsets" aria-describedby="help-offset">?</button><div id="help-offset" class="help-popup" role="tooltip">Mean of B − A within each session, with whiskers at ± one within-session SD. A stable offset is expected because the label sets use different gaze origins; what matters for consistency is that the whiskers stay short and the offset does not jump between sessions.</div></div><div id="offset-chart" class="chart" role="img" aria-label="Mean offset between vectors per session, grouped by user"></div></article>
    <article class="panel wide"><div class="panel-heading"><h2>Least consistent sessions</h2><button class="help-button" type="button" aria-label="Explain session ranking" aria-describedby="help-worst">?</button><div id="help-worst" class="help-popup" role="tooltip">Sessions ranked by within-session, per-target Pearson r, lowest first. A much larger SD in one vector usually means frames where that vector's origin estimate failed.</div></div><div class="table-wrap"><table><thead><tr><th>User / session</th><th>Samples</th><th>Pearson r</th><th>Spearman ρ</th><th>SD A (°)</th><th>SD B (°)</th><th>SD of B − A (°)</th></tr></thead><tbody id="worst-table"></tbody></table></div></article>
  </section>
</main>
<script>
const SUMMARY = __SUMMARY__;
const UNITS = __UNITS__;
const OFFSETS = __OFFSETS__;
const DEVIATIONS = __DEVIATIONS__;
const VECTOR_LABELS = __VECTOR_LABELS__;
const LEVELS = [["within_session","raw","Within session, raw"],["within_session","target_conditioned","Within session, per target"],["across_sessions","target_conditioned","Across sessions"]];
const taskFilter = document.getElementById("task-filter");
const pairFilter = document.getElementById("pair-filter");
const componentFilter = document.getElementById("component-filter");
const number = new Intl.NumberFormat(undefined, {maximumFractionDigits:3});
const integer = new Intl.NumberFormat();
const isNumber = value => typeof value === "number" && Number.isFinite(value);
const format = value => isNumber(value) ? number.format(value) : "—";
const label = value => String(value).replaceAll("_", " ");
const vectorLabel = value => VECTOR_LABELS[value] || label(value);
const pairKey = row => `${row.vector_a}|${row.vector_b}`;
const unique = values => [...new Set(values)];
const withCi = (row, name) => isNumber(row[`${name}_ci_low`]) ? `${format(row[name])} (${format(row[`${name}_ci_low`])}–${format(row[`${name}_ci_high`])})` : format(row[name]);

function options(select, items, preferred) {
  const current = select.value || preferred;
  select.replaceChildren();
  for (const [value, text] of items) { const item=document.createElement("option"); item.value=value; item.textContent=text; select.appendChild(item); }
  if (items.some(([value]) => value===current)) select.value=current;
}
function theme() { const s=getComputedStyle(document.documentElement); const get=n=>s.getPropertyValue(n).trim(); return {text:get("--text"),muted:get("--muted"),border:get("--border"),panel:get("--panel"),series:get("--series"),reference:get("--reference")}; }
function layout(xTitle, yTitle) { const c=theme(); return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:c.text,family:"Inter,system-ui,sans-serif"},margin:{l:70,r:24,t:18,b:62},xaxis:{title:{text:xTitle},gridcolor:c.border,zerolinecolor:c.border,automargin:true},yaxis:{title:{text:yTitle},gridcolor:c.border,zerolinecolor:c.border,automargin:true},hoverlabel:{bgcolor:c.panel,bordercolor:c.border,font:{color:c.text}},showlegend:false}; }
const plotConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"]};
const marker = () => ({color:theme().series,size:8,opacity:.75,line:{color:theme().panel,width:1}});
const empty = target => { Plotly.purge(target); document.getElementById(target).innerHTML='<div class="empty">No values are available for this selection.</div>'; };
const selected = row => row.task_type===taskFilter.value && pairKey(row)===pairFilter.value;
const names = () => { const [a,b]=pairFilter.value.split("|"); return [vectorLabel(a), vectorLabel(b)]; };

function renderSummary() {
  const body=document.getElementById("summary-table"); body.replaceChildren();
  for (const [level, view, text] of LEVELS) {
    const item=SUMMARY.find(row=>selected(row)&&row.component===componentFilter.value&&row.level===level&&row.view===view);
    if (!item) continue;
    const row=body.insertRow();
    const values=[text,integer.format(item.n_sessions),integer.format(item.n_samples),withCi(item,"pearson_r"),format(item.spearman_rho),format(item.unit_r_median),format(item.sd_a_deg),format(item.sd_b_deg),withCi(item,"sd_ratio"),withCi(item,"slope"),withCi(item,"diff_sd_deg")];
    for (const value of values) row.insertCell().textContent=value;
  }
}
function renderUnits() {
  const target="units-chart"; document.getElementById(target).innerHTML="";
  const traces=[];
  for (const [level, view, text] of [...LEVELS].reverse()) {
    const rows=UNITS.filter(row=>selected(row)&&row.component===componentFilter.value&&row.level===level&&row.view===view&&isNumber(row.pearson_r));
    if (!rows.length) continue;
    traces.push({type:"box",orientation:"h",name:text,x:rows.map(row=>row.pearson_r),boxpoints:"all",jitter:.5,pointpos:0,fillcolor:"rgba(0,0,0,0)",line:{color:theme().muted,width:1},marker:marker(),
      customdata:rows.map(row=>[row.user_id,row.session_id??"all",row.n_samples,row.sd_a_deg,row.sd_b_deg]),
      hovertemplate:"User %{customdata[0]} · session %{customdata[1]}<br>Pearson r %{x:.3f}<br>%{customdata[2]} points<br>SD A %{customdata[3]:.3f}° · SD B %{customdata[4]:.3f}°<extra></extra>"});
  }
  if (!traces.length) return empty(target);
  const base=layout("Pearson r per unit",""); base.margin.l=170; base.xaxis.range=[Math.min(-.05,...traces.flatMap(t=>t.x))-.05,1.05];
  Plotly.react(target,traces,base,plotConfig);
}
function renderDrift() {
  const target="drift-chart"; document.getElementById(target).innerHTML="";
  const [a,b]=pairFilter.value.split("|"); const component=componentFilter.value;
  const rows=DEVIATIONS.filter(row=>row.task_type===taskFilter.value&&isNumber(row[`${a}_${component}`])&&isNumber(row[`${b}_${component}`]));
  if (!rows.length) return empty(target);
  const x=rows.map(row=>row[`${a}_${component}`]), y=rows.map(row=>row[`${b}_${component}`]);
  const limit=Math.max(...x.map(Math.abs),...y.map(Math.abs))*1.08;
  const [nameA,nameB]=names();
  const identity={type:"scatter",mode:"lines",x:[-limit,limit],y:[-limit,limit],line:{color:theme().reference,width:1.5,dash:"dash"},hoverinfo:"skip"};
  const points={type:"scatter",mode:"markers",x,y,marker:marker(),customdata:rows.map(row=>[row.user_id,row.session_id,row.n_targets,row.n_samples]),
    hovertemplate:`User %{customdata[0]} · session %{customdata[1]}<br>${nameA} %{x:.2f}°<br>${nameB} %{y:.2f}°<br>%{customdata[2]} targets · %{customdata[3]} samples<extra></extra>`};
  const base=layout(`${nameA}: session deviation (°)`,`${nameB}: session deviation (°)`);
  base.xaxis.range=[-limit,limit]; base.yaxis.range=[-limit,limit]; base.yaxis.scaleanchor="x";
  Plotly.react(target,[identity,points],base,plotConfig);
}
function renderOffsets() {
  const target="offset-chart"; document.getElementById(target).innerHTML="";
  const component=componentFilter.value;
  const rows=OFFSETS.filter(row=>selected(row)&&isNumber(row[`mean_offset_${component}_deg`])).sort((p,q)=>String(p.user_id).localeCompare(String(q.user_id),undefined,{numeric:true})||String(p.session_id).localeCompare(String(q.session_id),undefined,{numeric:true}));
  if (!rows.length) return empty(target);
  const [nameA,nameB]=names();
  const trace={type:"scatter",mode:"markers",x:rows.map(row=>`User ${row.user_id}`),y:rows.map(row=>row[`mean_offset_${component}_deg`]),marker:marker(),
    error_y:{type:"data",array:rows.map(row=>row[`sd_offset_${component}_deg`]??0),color:theme().muted,thickness:1,width:0},
    customdata:rows.map(row=>[row.user_id,row.session_id,row[`sd_offset_${component}_deg`],row.median_angle_deg,row.n_samples]),
    hovertemplate:"User %{customdata[0]} · session %{customdata[1]}<br>Mean offset %{y:.2f}° (SD %{customdata[2]:.2f}°)<br>Median 3D angle %{customdata[3]:.2f}°<br>%{customdata[4]} samples<extra></extra>"};
  Plotly.react(target,[trace],layout("",`${nameB} − ${nameA}, ${component} (°)`),plotConfig);
}
function renderWorst() {
  const body=document.getElementById("worst-table"); body.replaceChildren();
  const rows=UNITS.filter(row=>selected(row)&&row.component===componentFilter.value&&row.level==="within_session"&&row.view==="target_conditioned"&&isNumber(row.pearson_r)).sort((p,q)=>p.pearson_r-q.pearson_r).slice(0,15);
  for (const item of rows) { const row=body.insertRow(); const values=[`${item.user_id} / ${item.session_id}`,integer.format(item.n_samples),format(item.pearson_r),format(item.spearman_rho),format(item.sd_a_deg),format(item.sd_b_deg),format(item.diff_sd_deg)]; for (const value of values) row.insertCell().textContent=value; }
}
function renderWarning() {
  const messages=[];
  const rows=SUMMARY.filter(row=>selected(row)&&row.component===componentFilter.value);
  if (!rows.some(row=>isNumber(row.pearson_r_ci_low))) messages.push("95% confidence intervals are unavailable because this report was generated without bootstrap iterations.");
  if (rows.some(row=>isNumber(row.pearson_r)&&isNumber(row.spearman_rho)&&row.spearman_rho-row.pearson_r>.15)) messages.push("Spearman ρ is well above Pearson r on some levels: a few extreme samples or sessions dominate the linear correlation. Check the per-unit chart and the least consistent sessions.");
  const warning=document.getElementById("warning"); warning.textContent=messages.join(" "); warning.style.display=messages.length?"block":"none";
}
function update() { renderWarning(); renderSummary(); renderUnits(); renderDrift(); renderOffsets(); renderWorst(); }
options(taskFilter,unique(SUMMARY.map(row=>row.task_type)).map(value=>[value,value==="all"?"All tasks":label(value)]),"all");
options(pairFilter,unique(SUMMARY.map(pairKey)).map(value=>{const [a,b]=value.split("|"); return [value,`${vectorLabel(a)} (A) vs ${vectorLabel(b)} (B)`];}),"");
options(componentFilter,[["pitch","Pitch"],["yaw","Yaw"]],"pitch");
for (const control of [taskFilter,pairFilter,componentFilter]) control.addEventListener("change",update);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",update);
update();
</script>
</body>
</html>"""
    return (
        template.replace("__PLOTLY__", get_plotlyjs())
        .replace("__SAMPLES__", f"{int(metadata.get('accepted_rows', 0)):,}")
        .replace("__USERS__", str(metadata.get("users", "—")))
        .replace("__SESSIONS__", str(metadata.get("user_sessions", "—")))
        .replace("__EXCLUDED__", html.escape(", ".join(excluded) or "none"))
        .replace("__VECTOR_LABELS__", _json({vector: VECTOR_LABELS.get(vector, vector) for vector in vectors}))
        .replace("__SUMMARY__", _json(_records(report["summary"])))
        .replace("__UNITS__", _json(_records(report["units"])))
        .replace("__OFFSETS__", _json(_records(report["offsets"])))
        .replace("__DEVIATIONS__", _json(_records(report["deviations"])))
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
    print(f"Created gaze-consistency explorer: {args.output} ({len(report['summary']):,} summary rows)")
    if args.open:
        webbrowser.open(args.output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
