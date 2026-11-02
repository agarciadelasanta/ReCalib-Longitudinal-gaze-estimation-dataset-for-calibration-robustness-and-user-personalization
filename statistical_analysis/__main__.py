from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .core import (
    AnalysisConfig, FEATURE_COLUMNS, DEFAULT_FEATURE_FAMILIES, LABEL_SETS, LANDMARK_FAMILIES,
    available_families,
)
from .comparison import DATASET_COMPARISON_FAMILIES, DEFAULT_COMPARISON_FAMILIES, compare_csvs
from .reporting import analyze_csv
from .progress import progress_enabled


def _add_labels_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels",
        choices=LABEL_SETS,
        default="recalib",
        help=(
            "Label set: 'recalib' (released labels, default) or 'eth' (ETH-XGaze "
            "baseline columns: gaze direction from eth_label_*, gaze origin from "
            "eth_face_center_*, only rows with eth_ok == 1; head pose unavailable)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ReCalib users and sessions with hierarchical, task-stratified "
            "statistical distance KPIs."
        )
    )
    parser.add_argument("csv", type=Path, help="Flattened dataset summary CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for tables, pairwise matrices, and the run manifest.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FEATURE_COLUMNS),
        help="Feature families to analyze (default: non-landmark families available for --labels).",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("target_conditioned", "raw"),
        default=("target_conditioned", "raw"),
        help="Analysis views (default: target-conditioned and raw).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Session-bootstrap iterations; use 0 to disable (default: 500).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1_000,
        help="Session-label permutation iterations; use 0 to disable (default: 1000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level in (0, 1) (default: 0.95).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--include-landmarks", action="store_true",
                        help="Add landmark position, scale, and shape using image-normalized coordinates.")
    _add_labels_argument(parser)
    parser.add_argument("--no-progress", action="store_true", help="Hide terminal progress bars.")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statistical_analysis compare",
        description=(
            "Measure the overall hierarchical statistical domain gap between "
            "two independent, comparison-compatible dataset-summary CSVs. "
            "Source task labels are not used to stratify the comparison."
        ),
    )
    parser.add_argument("csv_a", type=Path, help="First dataset-summary CSV.")
    parser.add_argument("csv_b", type=Path, help="Second dataset-summary CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the domain-gap report bundle.",
    )
    parser.add_argument("--name-a", help="Display label for the first dataset.")
    parser.add_argument("--name-b", help="Display label for the second dataset.")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=DATASET_COMPARISON_FAMILIES,
        help=(
            "Feature families to compare (default: non-quality, non-landmark "
            "families available for --labels)."
        ),
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("target_conditioned", "raw"),
        default=("raw",),
        help="Analysis views (default: raw overall comparison).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Session-bootstrap iterations; use 0 to disable (default: 500).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1_000,
        help="Session-label permutation iterations; use 0 to disable (default: 1000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level in (0, 1) (default: 0.95).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--include-landmarks", action="store_true",
                        help="Add landmark position, scale, and shape using image-normalized coordinates.")
    _add_labels_argument(parser)
    parser.add_argument("--no-progress", action="store_true", help="Hide terminal progress bars.")
    return parser


def _config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    default_families: tuple[str, ...],
) -> AnalysisConfig:
    if args.bootstrap < 0 or args.permutations < 0:
        parser.error("--bootstrap and --permutations must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.families is None:
        families = available_families(default_families, args.labels)
    else:
        families = tuple(args.families)
    if args.include_landmarks:
        families = tuple(dict.fromkeys((*families, *LANDMARK_FAMILIES)))
    try:
        return AnalysisConfig(
            feature_families=families,
            views=tuple(args.views),
            bootstrap_iterations=args.bootstrap,
            permutation_iterations=args.permutations,
            confidence_level=args.confidence,
            random_seed=args.seed,
            label_set=args.labels,
        )
    except ValueError as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        parser = build_compare_parser()
        args = parser.parse_args(argv[1:])
        config = _config_from_args(args, parser, DEFAULT_COMPARISON_FAMILIES)
        try:
            with progress_enabled(not args.no_progress):
                manifest = compare_csvs(
                    args.csv_a,
                    args.csv_b,
                    args.output,
                    config,
                    label_a=args.name_a,
                    label_b=args.name_b,
                )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "report_type": "independent_dataset_comparison",
                    "output_dir": manifest.output_dir,
                    "files": len(manifest.files),
                    "metadata": manifest.metadata,
                },
                indent=2,
            )
        )
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args, parser, DEFAULT_FEATURE_FAMILIES)
    try:
        with progress_enabled(not args.no_progress):
            manifest = analyze_csv(args.csv, args.output, config)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output_dir": manifest.output_dir,
                "files": len(manifest.files),
                "metadata": manifest.metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
