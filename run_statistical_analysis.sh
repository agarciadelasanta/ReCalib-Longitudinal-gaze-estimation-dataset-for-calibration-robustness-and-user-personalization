#!/usr/bin/env bash
# Run with Bash (Git Bash on Windows), from any directory.
# Optional overrides:
#   PYTHON="/path/to/python" BOOTSTRAP=20 PERMUTATIONS=50 bash run_statistical_analysis.sh
#   VIEWS="raw target_conditioned" OUTPUT_ROOT=temp/full_analysis bash run_statistical_analysis.sh
# Relative CSV and output paths are resolved from the repository root.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "C:/Users/irisbond/.conda/envs/mamu_data_curation/python.exe" ]]; then
        PYTHON="C:/Users/irisbond/.conda/envs/mamu_data_curation/python.exe"
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        PYTHON=python3
    fi
fi

RECALIB_CSV="${RECALIB_CSV:-dataset_summary_recalib.csv}"
TARGETS_CSV="${TARGETS_CSV:-dataset_summary_targets.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-temp/statistical_reports}"
BOOTSTRAP="${BOOTSTRAP:-20}"
PERMUTATIONS="${PERMUTATIONS:-50}"
read -r -a views <<< "${VIEWS:-raw}"
settings=(--bootstrap "$BOOTSTRAP" --permutations "$PERMUTATIONS" --views "${views[@]}")

for csv in "$RECALIB_CSV" "$TARGETS_CSV"; do
    if [[ ! -f "$csv" ]]; then
        printf 'Input CSV not found: %s\n' "$csv" >&2
        exit 1
    fi
done

# Check dependencies before starting a potentially long analysis.
"$PYTHON" -c 'import numpy, pandas, scipy, plotly, tqdm'
printf 'Settings: bootstrap=%s; permutations=%s; views=%s\n' "$BOOTSTRAP" "$PERMUTATIONS" "${views[*]}"

printf '\n[1/8] Analyzing ReCalib\n'
"$PYTHON" -m statistical_analysis "$RECALIB_CSV" --output "$OUTPUT_ROOT/analysis_recalib" "${settings[@]}"

printf '\n[2/8] Creating ReCalib statistical HTML\n'
"$PYTHON" visualization/explore_statistical_analysis.py --report-dir "$OUTPUT_ROOT/analysis_recalib"

printf '\n[3/8] Analyzing Targets\n'
"$PYTHON" -m statistical_analysis "$TARGETS_CSV" --output "$OUTPUT_ROOT/analysis_targets" "${settings[@]}"

printf '\n[4/8] Creating Targets statistical HTML\n'
"$PYTHON" visualization/explore_statistical_analysis.py --report-dir "$OUTPUT_ROOT/analysis_targets"

printf '\n[5/8] Comparing ReCalib and Targets\n'
"$PYTHON" -m statistical_analysis compare "$RECALIB_CSV" "$TARGETS_CSV" \
    --name-a ReCalib --name-b Targets --output "$OUTPUT_ROOT/recalib_vs_targets" "${settings[@]}"

printf '\n[6/8] Creating comparison HTML\n'
"$PYTHON" visualization/explore_dataset_comparison.py --report-dir "$OUTPUT_ROOT/recalib_vs_targets"

printf '\n[7/8] Creating ReCalib dataset explorer\n'
"$PYTHON" visualization/explore_dataset.py --csv "$RECALIB_CSV" --output "$OUTPUT_ROOT/recalib_explorer.html"

printf '\n[8/8] Creating Targets dataset explorer\n'
"$PYTHON" visualization/explore_dataset.py --csv "$TARGETS_CSV" --output "$OUTPUT_ROOT/targets_explorer.html"

printf '\nFinished. HTML reports:\n'
printf '  %s\n' \
    "$OUTPUT_ROOT/analysis_recalib/statistical_analysis_explorer.html" \
    "$OUTPUT_ROOT/analysis_targets/statistical_analysis_explorer.html" \
    "$OUTPUT_ROOT/recalib_vs_targets/dataset_comparison_explorer.html" \
    "$OUTPUT_ROOT/recalib_explorer.html" \
    "$OUTPUT_ROOT/targets_explorer.html"
