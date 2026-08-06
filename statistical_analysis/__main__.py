from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import AnalysisConfig, FEATURE_COLUMNS
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.permutations < 0:
        parser.error("--bootstrap and --permutations must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")

    config = AnalysisConfig(
        feature_families=tuple(args.families),
        views=tuple(args.views),
        bootstrap_iterations=args.bootstrap,
        permutation_iterations=args.permutations,
        confidence_level=args.confidence,
        random_seed=args.seed,
    )
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
