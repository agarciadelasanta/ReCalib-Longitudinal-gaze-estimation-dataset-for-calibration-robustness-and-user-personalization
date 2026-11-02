#!/usr/bin/env bash
# Run with Bash (Git Bash on Windows), from any directory.
# Optional overrides:
#   PYTHON="/path/to/python" BOOTSTRAP=20 PERMUTATIONS=50 bash run_statistical_analysis.sh
#   VIEWS="raw target_conditioned" OUTPUT_ROOT=temp/full_analysis bash run_statistical_analysis.sh
#   LABELS="recalib eth" bash run_statistical_analysis.sh
# LABELS selects the label sets: recalib (released labels, default) and/or eth (ETH-XGaze
# baseline columns from create_dataset_summary.py --eth-csv). ETH reports are written next
# to the ReCalib ones with an _eth suffix; Targets and the comparison are skipped for eth
# when the Targets CSV has no ETH columns. With eth, the ReCalib and ETH-XGaze label vectors
# are also checked for consistent variation (gaze_consistency/). With both label sets, the
# individual analyses use every sample each set has, and recalib_vs_eth_labels/ compares
# the two label sets on the families and samples that have both labels.
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
read -r -a label_sets <<< "${LABELS:-recalib}"
settings=(--bootstrap "$BOOTSTRAP" --permutations "$PERMUTATIONS" --views "${views[@]}")

for csv in "$RECALIB_CSV" "$TARGETS_CSV"; do
    if [[ ! -f "$csv" ]]; then
        printf 'Input CSV not found: %s\n' "$csv" >&2
        exit 1
    fi
done

has_eth_columns() {
    head -n 1 "$1" | tr -d '\r' | tr ',' '\n' | grep -qx 'eth_ok'
}

for labels in "${label_sets[@]}"; do
    case "$labels" in
        recalib) ;;
        eth)
            if ! has_eth_columns "$RECALIB_CSV"; then
                printf 'No ETH-XGaze columns in %s; run create_dataset_summary.py with --eth-csv\n' "$RECALIB_CSV" >&2
                exit 1
            fi
            ;;
        *) printf 'Unknown label set: %s (use recalib or eth)\n' "$labels" >&2; exit 1 ;;
    esac
done

# Check dependencies before starting a potentially long analysis.
"$PYTHON" -c 'import numpy, pandas, scipy, plotly, tqdm'
printf 'Settings: bootstrap=%s; permutations=%s; views=%s; labels=%s\n' \
    "$BOOTSTRAP" "$PERMUTATIONS" "${views[*]}" "${label_sets[*]}"

reports=()
for labels in "${label_sets[@]}"; do
    suffix=""
    [[ "$labels" == eth ]] && suffix="_eth"
    run_settings=("${settings[@]}" --labels "$labels")

    printf '\n[%s] Analyzing ReCalib\n' "$labels"
    "$PYTHON" -m statistical_analysis "$RECALIB_CSV" --output "$OUTPUT_ROOT/analysis_recalib$suffix" "${run_settings[@]}"

    printf '\n[%s] Creating ReCalib statistical HTML\n' "$labels"
    "$PYTHON" visualization/explore_statistical_analysis.py --report-dir "$OUTPUT_ROOT/analysis_recalib$suffix"
    reports+=("$OUTPUT_ROOT/analysis_recalib$suffix/statistical_analysis_explorer.html")

    if [[ "$labels" == eth ]]; then
        printf '\n[%s] Checking ReCalib vs ETH-XGaze label consistency\n' "$labels"
        "$PYTHON" -m statistical_analysis consistency "$RECALIB_CSV" --output "$OUTPUT_ROOT/gaze_consistency" \
            --vectors recalib eth --bootstrap "$BOOTSTRAP"
        "$PYTHON" visualization/explore_gaze_consistency.py --report-dir "$OUTPUT_ROOT/gaze_consistency"
        reports+=("$OUTPUT_ROOT/gaze_consistency/gaze_consistency_explorer.html")
    fi

    if [[ "$labels" == eth ]] && ! has_eth_columns "$TARGETS_CSV"; then
        printf '\n[%s] Skipping Targets and the comparison: no ETH-XGaze columns in %s\n' "$labels" "$TARGETS_CSV"
        continue
    fi

    printf '\n[%s] Analyzing Targets\n' "$labels"
    "$PYTHON" -m statistical_analysis "$TARGETS_CSV" --output "$OUTPUT_ROOT/analysis_targets$suffix" "${run_settings[@]}"

    printf '\n[%s] Creating Targets statistical HTML\n' "$labels"
    "$PYTHON" visualization/explore_statistical_analysis.py --report-dir "$OUTPUT_ROOT/analysis_targets$suffix"

    printf '\n[%s] Comparing ReCalib and Targets\n' "$labels"
    "$PYTHON" -m statistical_analysis compare "$RECALIB_CSV" "$TARGETS_CSV" \
        --name-a ReCalib --name-b Targets --output "$OUTPUT_ROOT/recalib_vs_targets$suffix" "${run_settings[@]}"

    printf '\n[%s] Creating comparison HTML\n' "$labels"
    "$PYTHON" visualization/explore_dataset_comparison.py --report-dir "$OUTPUT_ROOT/recalib_vs_targets$suffix"
    reports+=(
        "$OUTPUT_ROOT/analysis_targets$suffix/statistical_analysis_explorer.html"
        "$OUTPUT_ROOT/recalib_vs_targets$suffix/dataset_comparison_explorer.html"
    )
done

if [[ " ${label_sets[*]} " == *" recalib "* && " ${label_sets[*]} " == *" eth "* ]]; then
    printf '\nComparing ReCalib and ETH-XGaze labels (samples with both labels)\n'
    "$PYTHON" -m statistical_analysis compare-labels "$RECALIB_CSV" \
        --output "$OUTPUT_ROOT/recalib_vs_eth_labels" "${settings[@]}"
    "$PYTHON" visualization/explore_dataset_comparison.py --report-dir "$OUTPUT_ROOT/recalib_vs_eth_labels"
    reports+=("$OUTPUT_ROOT/recalib_vs_eth_labels/dataset_comparison_explorer.html")
fi

printf '\nCreating ReCalib dataset explorer\n'
"$PYTHON" visualization/explore_dataset.py --csv "$RECALIB_CSV" --output "$OUTPUT_ROOT/recalib_explorer.html"

printf '\nCreating Targets dataset explorer\n'
"$PYTHON" visualization/explore_dataset.py --csv "$TARGETS_CSV" --output "$OUTPUT_ROOT/targets_explorer.html"

printf '\nFinished. HTML reports:\n'
printf '  %s\n' \
    "${reports[@]}" \
    "$OUTPUT_ROOT/recalib_explorer.html" \
    "$OUTPUT_ROOT/targets_explorer.html"
