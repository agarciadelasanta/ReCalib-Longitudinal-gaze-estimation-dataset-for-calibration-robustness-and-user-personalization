from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .core import AnalysisConfig, FEATURE_COLUMNS
from .comparison import DATASET_COMPARISON_FAMILIES, compare_csvs
from .reporting import analyze_csv


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
        default=tuple(FEATURE_COLUMNS),
        help="Feature families to analyze (default: all usable families).",
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
        default=DATASET_COMPARISON_FAMILIES,
        help="Feature families to compare (default: all non-quality families).",
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
    return parser


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> AnalysisConfig:
    if args.bootstrap < 0 or args.permutations < 0:
        parser.error("--bootstrap and --permutations must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    return AnalysisConfig(
        feature_families=tuple(args.families),
        views=tuple(args.views),
        bootstrap_iterations=args.bootstrap,
        permutation_iterations=args.permutations,
        confidence_level=args.confidence,
        random_seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        parser = build_compare_parser()
        args = parser.parse_args(argv[1:])
        config = _config_from_args(args, parser)
        try:
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
    config = _config_from_args(args, parser)
    try:
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
