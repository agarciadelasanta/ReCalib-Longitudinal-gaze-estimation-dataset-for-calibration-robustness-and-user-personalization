from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .core import AnalysisConfig, analyze_frame


@dataclass(frozen=True)
class AnalysisManifest:
    input_path: str
    output_dir: str
    files: tuple[str, ...]
    metadata: dict[str, Any]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def _write_matrices(pairwise: pd.DataFrame, output_dir: Path) -> tuple[list[str], pd.DataFrame]:
    files: list[str] = []
    index_rows = []
    group_columns = ["view", "task_type", "family", "metric"]
    for key, group in pairwise.groupby(group_columns, sort=True, dropna=False):
        users = sorted(set(group["user_a"]).union(group["user_b"]))
        view, task_type, family, metric = key
        for statistic in ("estimate", "ci_low", "ci_high", "p_value", "q_value"):
            matrix = pd.DataFrame(np.nan, index=users, columns=users, dtype=float)
            if statistic == "estimate":
                np.fill_diagonal(matrix.values, 0.0)
            for row in group.itertuples(index=False):
                value = getattr(row, statistic)
                matrix.loc[row.user_a, row.user_b] = value
                matrix.loc[row.user_b, row.user_a] = value
            relative = (
                Path("matrices")
                / _slug(view)
                / _slug(task_type)
                / f"{_slug(family)}__{_slug(metric)}__{statistic}.csv"
            )
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            matrix.to_csv(target, index_label="user_id")
            relative_text = relative.as_posix()
            files.append(relative_text)
            index_rows.append(
                {
                    "view": view,
                    "task_type": task_type,
                    "family": family,
                    "metric": metric,
                    "statistic": statistic,
                    "path": relative_text,
                }
            )
    return files, pd.DataFrame(index_rows)


def analyze_csv(
    input_path: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
) -> AnalysisManifest:
    """Analyze a summary CSV and write the complete Phase 1 report bundle."""

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    config = config or AnalysisConfig()
    identity_dtypes = {
        "user_id": "string",
        "session_id": "string",
        "task_id": "string",
        "sample_id": "string",
    }
    frame = pd.read_csv(input_path, dtype=identity_dtypes, low_memory=False)
    result = analyze_frame(frame, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "pairwise_distances.csv": result.pairwise_distances,
        "user_profiles.csv": result.user_profiles,
        "calibration_test_mismatch.csv": result.calibration_test_mismatch,
        "feature_catalog.csv": result.feature_catalog,
        "column_catalog.csv": result.column_catalog,
    }
    files = []
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
        files.append(name)

    matrix_files, matrix_index = _write_matrices(result.pairwise_distances, output_dir)
    matrix_index.to_csv(output_dir / "matrix_index.csv", index=False)
    files.append("matrix_index.csv")
    files.extend(matrix_files)

    manifest_data = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "metadata": result.metadata,
        "files": ["manifest.json", *files],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)

    return AnalysisManifest(
        input_path=str(input_path),
        output_dir=str(output_dir),
        files=tuple(manifest_data["files"]),
        metadata=result.metadata,
    )
