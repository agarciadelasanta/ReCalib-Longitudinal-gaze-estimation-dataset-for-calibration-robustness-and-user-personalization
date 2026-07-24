#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze and visualize the bundled ReCalib dataset summary."""

import argparse
from pathlib import Path

import gaze_analyzer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "dataset_summary.csv"
DEFAULT_PARAMS = Path(__file__).resolve().parent / "gaze_analyisis_params.yaml"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze gaze and head-pose distributions in a ReCalib summary CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Dataset summary CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument("--sep", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument(
        "--params-yaml",
        type=Path,
        default=DEFAULT_PARAMS,
        help=f"Plot limits and display parameters (default: {DEFAULT_PARAMS}).",
    )
    parser.add_argument(
        "--n-regions",
        type=int,
        default=4,
        help="Grid size used for spatial gaze coverage (default: 4).",
    )
    parser.add_argument(
        "--outlier-q",
        type=float,
        default=0.01,
        help="Quantile removed from each coordinate tail (default: 0.01).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run the analysis without creating or displaying plots.",
    )

    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument(
        "--id-user",
        type=int,
        default=7,
        help="Participant ID to analyze (default: 7).",
    )
    user_group.add_argument(
        "--all-users",
        action="store_true",
        help="Analyze all participants instead of the default participant.",
    )
    parser.add_argument(
        "--id-session",
        type=int,
        help="Session ID to analyze; requires --id-user.",
    )
    parser.add_argument(
        "--id-task",
        type=int,
        help="Task ID to analyze; requires --id-user and --id-session.",
    )

    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--export-dir",
        type=Path,
        help="Output directory. Relative paths are resolved from the current directory.",
    )
    export_group.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export plots or the augmented CSV.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.outlier_q < 0.5:
        parser.error("--outlier-q must be at least 0 and less than 0.5")
    if args.n_regions < 1:
        parser.error("--n-regions must be at least 1")

    if args.all_users:
        args.id_user = None
    if args.id_session is not None and args.id_user is None:
        parser.error("--id-session requires --id-user")
    if args.id_task is not None and args.id_session is None:
        parser.error("--id-task requires --id-session")

    args.csv = args.csv.expanduser().resolve()
    args.params_yaml = args.params_yaml.expanduser().resolve()
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    if not args.params_yaml.is_file():
        parser.error(f"parameters YAML not found: {args.params_yaml}")

    if args.no_export:
        args.export_dir = None
    elif args.export_dir is not None:
        args.export_dir = args.export_dir.expanduser().resolve()
    elif args.id_user is None:
        args.export_dir = REPO_ROOT / "temp" / "all_users"
    elif args.id_session is None:
        args.export_dir = REPO_ROOT / "temp" / f"user{args.id_user}"
    else:
        args.export_dir = (
            REPO_ROOT / "temp" / f"user{args.id_user}_{args.id_session}"
        )

    # gaze_analyzer uses the legacy attribute name ``task``.
    args.task = args.id_task
    return args


def main(argv=None):
    args = parse_args(argv)
    gaze_analyzer.analyze(args)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
