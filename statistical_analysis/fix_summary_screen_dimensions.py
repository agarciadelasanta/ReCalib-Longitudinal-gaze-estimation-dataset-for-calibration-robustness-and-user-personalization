"""Make screen dimensions consistent with POG pixel coordinates in summary CSVs.

The command defaults to inspecting ``dataset_summary_*.csv`` in the current
directory. Pass ``--apply`` to rewrite them atomically. Backups are created by
default before any original is replaced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence


DEFAULT_WIDTH = 2736
DEFAULT_HEIGHT = 1824
DEFAULT_PATTERN = "dataset_summary_*.csv"


@dataclass(frozen=True)
class CsvInspection:
    path: Path
    rows: int
    changed_rows: int
    min_x: float | None
    max_x: float | None
    min_y: float | None
    max_y: float | None
    adds_width_column: bool
    adds_height_column: bool


def _encoding_for(path: Path) -> str:
    with path.open("rb") as handle:
        return "utf-8-sig" if handle.read(3) == b"\xef\xbb\xbf" else "utf-8"


def _number(value: str, *, column: str, row_number: int, path: Path) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"{path}: row {row_number} has non-numeric {column}={value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(
            f"{path}: row {row_number} has non-finite {column}={value!r}"
        )
    return result


def _matches_dimension(value: str, expected: int) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
    except ValueError:
        return False


def inspect_csv(
    path: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> CsvInspection:
    """Validate one summary and report how many rows need corrected dimensions."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")

    encoding = _encoding_for(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {path}") from error

        required = ("pog_px_x", "pog_px_y")
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        x_index = header.index("pog_px_x")
        y_index = header.index("pog_px_y")
        width_index = header.index("screen_width_px") if "screen_width_px" in header else None
        height_index = header.index("screen_height_px") if "screen_height_px" in header else None
        rows = changed_rows = 0
        xs: list[float] = []
        ys: list[float] = []

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            rows += 1
            x = _number(row[x_index], column="pog_px_x", row_number=row_number, path=path)
            y = _number(row[y_index], column="pog_px_y", row_number=row_number, path=path)
            if x is not None:
                if x < 0 or x > width:
                    raise ValueError(
                        f"{path}: row {row_number} has pog_px_x={x:g} outside [0, {width}]"
                    )
                xs.append(x)
            if y is not None:
                if y < 0 or y > height:
                    raise ValueError(
                        f"{path}: row {row_number} has pog_px_y={y:g} outside [0, {height}]"
                    )
                ys.append(y)

            width_matches = (
                width_index is not None and _matches_dimension(row[width_index], width)
            )
            height_matches = (
                height_index is not None and _matches_dimension(row[height_index], height)
            )
            changed_rows += int(not width_matches or not height_matches)

    return CsvInspection(
        path=path,
        rows=rows,
        changed_rows=changed_rows,
        min_x=min(xs) if xs else None,
        max_x=max(xs) if xs else None,
        min_y=min(ys) if ys else None,
        max_y=max(ys) if ys else None,
        adds_width_column=width_index is None,
        adds_height_column=height_index is None,
    )


def rewrite_csv(
    path: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> None:
    """Rewrite one validated CSV through a temporary file and atomic replace."""

    path = Path(path).resolve()
    encoding = _encoding_for(path)
    temporary_path: Path | None = None
    try:
        with path.open("r", encoding=encoding, newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            width_index = (
                header.index("screen_width_px")
                if "screen_width_px" in header
                else len(header)
            )
            if width_index == len(header):
                header.append("screen_width_px")
            height_index = (
                header.index("screen_height_px")
                if "screen_height_px" in header
                else len(header)
            )
            if height_index == len(header):
                header.append("screen_height_px")

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            with temporary_path.open("w", encoding=encoding, newline="") as target:
                writer = csv.writer(target)
                writer.writerow(header)
                for row in reader:
                    while len(row) < len(header):
                        row.append("")
                    row[width_index] = str(width)
                    row[height_index] = str(height)
                    writer.writerow(row)

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _resolve_paths(values: Sequence[Path]) -> list[Path]:
    paths = list(values) if values else sorted(Path.cwd().glob(DEFAULT_PATTERN))
    resolved = []
    seen: set[Path] = set()
    for path in paths:
        candidate = path.expanduser().resolve()
        if candidate not in seen:
            resolved.append(candidate)
            seen.add(candidate)
    if not resolved:
        raise FileNotFoundError(
            f"No input files supplied and no {DEFAULT_PATTERN!r} files found in {Path.cwd()}"
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate POG coordinates and set screen_width_px/screen_height_px "
            "to the effective coordinate-space dimensions in dataset summaries."
        )
    )
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help=f"Summary CSVs (default: {DEFAULT_PATTERN} in the current directory).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the files. Without this flag, perform validation only.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create FILE.csv.bak copies when applying changes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")

    try:
        paths = _resolve_paths(args.csv)
        inspections = [
            inspect_csv(path, width=args.width, height=args.height) for path in paths
        ]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    for item in inspections:
        x_range = f"[{item.min_x:g}, {item.max_x:g}]" if item.min_x is not None else "empty"
        y_range = f"[{item.min_y:g}, {item.max_y:g}]" if item.min_y is not None else "empty"
        print(
            f"{item.path}: rows={item.rows}, rows_to_change={item.changed_rows}, "
            f"pog_x={x_range}, pog_y={y_range}"
        )

    if not args.apply:
        print(
            f"Dry run only. Re-run with --apply to set every row to "
            f"{args.width}x{args.height}."
        )
        return 0

    changed = [item for item in inspections if item.changed_rows]
    if not changed:
        print("No changes required.")
        return 0

    backup_paths = {item.path: Path(f"{item.path}.bak") for item in changed}
    if not args.no_backup:
        existing = [backup for backup in backup_paths.values() if backup.exists()]
        if existing:
            parser.error(
                "refusing to overwrite existing backups: "
                + ", ".join(str(path) for path in existing)
            )
        for source, backup in backup_paths.items():
            shutil.copy2(source, backup)
            print(f"Created backup: {backup}")

    for item in changed:
        rewrite_csv(item.path, width=args.width, height=args.height)
        print(f"Updated atomically: {item.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
