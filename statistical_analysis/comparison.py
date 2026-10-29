"""Independent, session-balanced comparison of two dataset-summary CSVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .progress import progress, stage
from .core import (
    AnalysisConfig,
    FEATURE_COLUMNS,
    LANDMARK_FAMILIES,
    FEATURE_NOTES,
    FEATURE_SOURCES,
    _compare_users,
    _describe_units,
    _fdr_bh,
    _metric_method,
    _permutation_p_values,
    _prepare_accepted_frame,
)
from .metrics import METRICS, descriptor_scales


_DATASET_A = "__dataset_a__"
_DATASET_B = "__dataset_b__"
DATASET_COMPARISON_FAMILIES = tuple(
    family for family in FEATURE_COLUMNS if family != "quality"
)
DEFAULT_COMPARISON_FAMILIES = tuple(
    family for family in DATASET_COMPARISON_FAMILIES if family not in LANDMARK_FAMILIES
)
_EXCLUDED_AUDIT_COLUMNS = {"discard_info", "discard_reason"}


def _excluded_from_comparison_audit(column: str) -> bool:
    return column in _EXCLUDED_AUDIT_COLUMNS or column.startswith("qa_")


@dataclass
class DatasetComparisonResult:
    """In-memory outputs for an independent two-dataset comparison."""

    metadata: dict[str, Any]
    dataset_distances: pd.DataFrame
    comparison_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    column_compatibility: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class DatasetComparisonManifest:
    """Files and metadata written by :func:`compare_csvs`."""

    input_a: str
    input_b: str
    output_dir: str
    files: tuple[str, ...]
    metadata: dict[str, Any]


def _assert_comparison_schema(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    required = {"user_id", "session_id", "discarded"}
    for family in config.feature_families:
        if family not in FEATURE_SOURCES:
            raise ValueError(f"Unknown feature family: {family}")
        required.update(FEATURE_SOURCES[family])
    if "target_conditioned" in config.views:
        required.update({"pog_px_x", "pog_px_y"})

    missing_a = sorted(required - set(frame_a.columns))
    missing_b = sorted(required - set(frame_b.columns))
    if not missing_a and not missing_b:
        return
    details = []
    if missing_a:
        details.append("missing from dataset A: " + ", ".join(missing_a))
    if missing_b:
        details.append("missing from dataset B: " + ", ".join(missing_b))
    raise ValueError(
        "Required comparison columns are unavailable (" + "; ".join(details) + ")"
    )


def _dataset_metadata(frame: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    sessions_per_user = accepted.groupby("user_id")["session_id"].nunique()
    samples_per_session = accepted.groupby(["user_id", "session_id"]).size()

    def balance(values: pd.Series) -> dict[str, float]:
        return {
            "min": int(values.min()), "median": float(values.median()),
            "max": int(values.max()), "mean": float(values.mean()),
        }

    metadata = {
        "input_rows": int(len(frame)),
        "accepted_rows": int(len(accepted)),
        "discarded_rows_excluded": int(len(frame) - len(accepted)),
        "users": int(accepted["user_id"].nunique()),
        "user_sessions": int(
            accepted[["user_id", "session_id"]].drop_duplicates().shape[0]
        ),
        "sessions_per_user": balance(sessions_per_user),
        "samples_per_session": balance(samples_per_session),
    }
    if "task_type" in accepted:
        metadata["source_task_counts_not_used_for_stratification"] = {
            str(task): int(count)
            for task, count in accepted.groupby("task_type").size().items()
        }
    return metadata


def _column_compatibility(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    accepted_a: pd.DataFrame,
    accepted_b: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    source_to_families: dict[str, list[str]] = {}
    for family in config.feature_families:
        for column in FEATURE_SOURCES[family]:
            source_to_families.setdefault(column, []).append(family)
    if "target_conditioned" in config.views:
        source_to_families.setdefault("pog_px_x", []).append("target_conditioning")
        source_to_families.setdefault("pog_px_y", []).append("target_conditioning")

    columns_a = set(frame_a.columns)
    columns_b = set(frame_b.columns)
    ordered_columns = list(frame_a.columns) + [
        column for column in frame_b.columns if column not in columns_a
    ]
    ordered_columns = [
        column for column in ordered_columns if not _excluded_from_comparison_audit(column)
    ]
    rows = []
    for column in ordered_columns:
        present_a = column in columns_a
        present_b = column in columns_b
        families = sorted(set(source_to_families.get(column, [])))
        if column in {"user_id", "session_id", "discarded"}:
            role = "comparison_structure"
        elif column in {"task_id", "task_type"}:
            role = "source_task_metadata_not_used"
        elif families:
            role = "feature_source"
        elif column in {"screen_width_px", "screen_height_px"}:
            role = "optional_target_normalization"
        else:
            role = "not_selected"

        used_in_comparison = role in {"comparison_structure", "feature_source"}
        numeric_required = bool(families) or role == "optional_target_normalization"
        series_a = (
            accepted_a[column]
            if present_a
            else pd.Series(np.nan, index=accepted_a.index)
        )
        series_b = (
            accepted_b[column]
            if present_b
            else pd.Series(np.nan, index=accepted_b.index)
        )
        non_null_a = int(series_a.notna().sum())
        non_null_b = int(series_b.notna().sum())
        if not present_a and not used_in_comparison and role != "optional_target_normalization":
            status = "only_in_dataset_b_ignored"
            comparable = True
        elif not present_b and not used_in_comparison and role != "optional_target_normalization":
            status = "only_in_dataset_a_ignored"
            comparable = True
        elif role == "optional_target_normalization" and non_null_a == 0 and non_null_b == 0:
            status = "fallback_2736x1824_used_in_both"
            comparable = True
        elif role == "optional_target_normalization" and non_null_a == 0:
            status = "fallback_2736x1824_used_in_dataset_a"
            comparable = True
        elif role == "optional_target_normalization" and non_null_b == 0:
            status = "fallback_2736x1824_used_in_dataset_b"
            comparable = True
        elif numeric_required and non_null_a == 0 and non_null_b == 0:
            status = "missing_in_both"
            comparable = False
        elif numeric_required and non_null_a == 0:
            status = "missing_in_dataset_a"
            comparable = False
        elif numeric_required and non_null_b == 0:
            status = "missing_in_dataset_b"
            comparable = False
        else:
            status = "compatible"
            comparable = True
        rows.append(
            {
                "column": column,
                "role": role,
                "feature_families": ",".join(families),
                "present_a": present_a,
                "present_b": present_b,
                "dtype_a": str(frame_a[column].dtype) if present_a else "<absent>",
                "dtype_b": str(frame_b[column].dtype) if present_b else "<absent>",
                "accepted_non_null_a": non_null_a,
                "accepted_non_null_b": non_null_b,
                "missing_fraction_a": float(series_a.isna().mean()),
                "missing_fraction_b": float(series_b.isna().mean()),
                "numeric_required": numeric_required,
                "used_in_comparison": used_in_comparison,
                "comparable": comparable,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _subject_key(dataset_id: str, user: Any) -> str:
    user_text = str(user)
    return f"{dataset_id}|{len(user_text)}:{user_text}"


def _aggregate_user_units(
    session_units: pd.DataFrame,
    dataset_id: str,
    *,
    conditioned: bool,
) -> pd.DataFrame:
    """Give sessions equal weight within users and users equal weight in a dataset."""

    keys = ["user_id"]
    if conditioned:
        keys.extend(["_target_x", "_target_y"])
    rows = []
    for key, group in session_units.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        original_user = key[0]
        row = {
            "user_id": dataset_id,
            "session_id": _subject_key(dataset_id, original_user),
            "location": np.stack(group["location"]).mean(axis=0),
            "dispersion": np.median(np.stack(group["dispersion"]), axis=0),
            "distribution": np.stack(group["distribution"]).mean(axis=0),
            "dependence": np.stack(group["dependence"]).mean(axis=0),
            "n_samples": int(group["n_samples"].sum()),
            "n_sessions": int(group["session_id"].nunique()),
        }
        if conditioned:
            row["_target_x"] = key[1]
            row["_target_y"] = key[2]
        rows.append(row)
    return pd.DataFrame(rows)


def _resample_sessions_within_users(
    session_units: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    parts = []
    for user, user_units in session_units.groupby("user_id", sort=False):
        sessions = user_units["session_id"].drop_duplicates().to_numpy()
        for draw, session in enumerate(
            rng.choice(sessions, size=len(sessions), replace=True)
        ):
            part = user_units[user_units["session_id"].eq(session)].copy()
            part["session_id"] = f"bootstrap_{draw}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _bootstrap_dataset_intervals(
    session_units_a: pd.DataFrame,
    session_units_b: pd.DataFrame,
    *,
    conditioned: bool,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
    scales: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    if iterations <= 0:
        return {}
    samples: dict[str, list[float]] = {}
    for _ in progress(range(iterations), "Dataset bootstrap", unit="draw"):
        users_a = _aggregate_user_units(
            _resample_sessions_within_users(session_units_a, rng),
            _DATASET_A,
            conditioned=conditioned,
        )
        users_b = _aggregate_user_units(
            _resample_sessions_within_users(session_units_b, rng),
            _DATASET_B,
            conditioned=conditioned,
        )
        estimates = _compare_users(
            pd.concat([users_a, users_b], ignore_index=True),
            _DATASET_A,
            _DATASET_B,
            conditioned=conditioned,
            scales=scales,
        )
        for metric, estimate in estimates.items():
            samples.setdefault(metric, []).append(estimate)
    tail = (1.0 - confidence_level) / 2.0
    return {
        metric: (
            float(np.nanquantile(values, tail)),
            float(np.nanquantile(values, 1.0 - tail)),
        )
        for metric, values in samples.items()
        if np.isfinite(values).any()
    }


def _target_set(units: pd.DataFrame) -> set[tuple[float, float]]:
    return set(
        units[["_target_x", "_target_y"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def _catalog(config: AnalysisConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": family,
                "dimensions": len(FEATURE_COLUMNS[family]),
                "source_columns": ",".join(FEATURE_SOURCES[family]),
                "supported_views": (
                    "raw"
                    if family == "target_coverage"
                    else ",".join(config.views)
                ),
                "notes": FEATURE_NOTES[family],
            }
            for family in config.feature_families
        ]
    )


def compare_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    config: AnalysisConfig | None = None,
    *,
    label_a: str = "dataset_a",
    label_b: str = "dataset_b",
) -> DatasetComparisonResult:
    """Compare two independent datasets without pairing users or samples.

    Task labels and task schedules are ignored. Sessions receive equal weight
    within each user, and users receive equal weight within each dataset. For
    the target-conditioned view, comparisons are calculated at targets present
    in both datasets and then averaged equally across those common targets.
    """

    config = config or AnalysisConfig(
        feature_families=DEFAULT_COMPARISON_FAMILIES,
        views=("raw",),
    )
    if "quality" in config.feature_families:
        raise ValueError(
            "The quality family is excluded from independent-dataset comparisons"
        )
    if not label_a.strip() or not label_b.strip():
        raise ValueError("Dataset labels must not be empty")
    if label_a == label_b:
        raise ValueError("Dataset labels must be different")
    _assert_comparison_schema(frame_a, frame_b, config)

    stage("Preparing dataset comparison...")
    accepted_a = _prepare_accepted_frame(frame_a, config, require_task_type=False)
    accepted_b = _prepare_accepted_frame(frame_b, config, require_task_type=False)
    if accepted_a.empty or accepted_b.empty:
        raise ValueError("Both datasets must contain at least one accepted sample")

    rng = np.random.default_rng(config.random_seed)
    distance_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for family in progress(config.feature_families, "Comparing features", unit="family"):
        columns = list(FEATURE_COLUMNS[family])
        complete_a = accepted_a.dropna(subset=columns)
        complete_b = accepted_b.dropna(subset=columns)
        family_views = tuple(
            view
            for view in config.views
            if not (family == "target_coverage" and view == "target_conditioned")
        )
        for view in progress(family_views, family, unit="view"):
            conditioned = view == "target_conditioned"
            session_units_a = _describe_units(
                complete_a, columns, conditioned=conditioned
            ) if not complete_a.empty else pd.DataFrame()
            session_units_b = _describe_units(
                complete_b, columns, conditioned=conditioned
            ) if not complete_b.empty else pd.DataFrame()

            units_a = _aggregate_user_units(
                session_units_a, _DATASET_A, conditioned=conditioned
            ) if not session_units_a.empty else pd.DataFrame()
            units_b = _aggregate_user_units(
                session_units_b, _DATASET_B, conditioned=conditioned
            ) if not session_units_b.empty else pd.DataFrame()

            common_targets: int | float = np.nan
            if conditioned and not units_a.empty and not units_b.empty:
                common_targets = len(_target_set(units_a) & _target_set(units_b))

            for dataset_label, complete, session_units, user_units in (
                (label_a, complete_a, session_units_a, units_a),
                (label_b, complete_b, session_units_b, units_b),
            ):
                coverage_rows.append(
                    {
                        "dataset": dataset_label,
                        "analysis_scope": "all_accepted_samples",
                        "view": view,
                        "family": family,
                        "n_users": int(complete["user_id"].nunique()),
                        "n_sessions": int(
                            complete[["user_id", "session_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_samples": int(len(complete)),
                        "n_session_descriptor_units": int(len(session_units)),
                        "n_user_descriptor_units": int(len(user_units)),
                        "n_common_targets": common_targets,
                    }
                )

            enough_data = (
                not units_a.empty
                and not units_b.empty
                and (not conditioned or common_targets > 0)
            )
            if enough_data:
                units = pd.concat([units_a, units_b], ignore_index=True)
                scales = descriptor_scales(units)
                estimates = _compare_users(
                    units,
                    _DATASET_A,
                    _DATASET_B,
                    conditioned=conditioned,
                    scales=scales,
                )
                intervals = _bootstrap_dataset_intervals(
                    session_units_a,
                    session_units_b,
                    conditioned=conditioned,
                    iterations=config.bootstrap_iterations,
                    confidence_level=config.confidence_level,
                    rng=rng,
                    scales=scales,
                )
                p_values = _permutation_p_values(
                    units,
                    _DATASET_A,
                    _DATASET_B,
                    estimates,
                    conditioned=conditioned,
                    iterations=config.permutation_iterations,
                    rng=rng,
                    scales=scales,
                )
                status = "ok"
            else:
                estimates = {metric: np.nan for metric in METRICS}
                intervals = {}
                p_values = {}
                status = (
                    "no_common_targets"
                    if conditioned and not units_a.empty and not units_b.empty
                    else "insufficient_data"
                )

            for metric in METRICS:
                ci_low, ci_high = intervals.get(metric, (np.nan, np.nan))
                distance_rows.append(
                    {
                        "dataset_a": label_a,
                        "dataset_b": label_b,
                        "analysis_scope": "all_accepted_samples",
                        "view": view,
                        "family": family,
                        "metric": metric,
                        "metric_method": _metric_method(family, metric),
                        "estimate": estimates.get(metric, np.nan),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_values.get(metric, np.nan),
                        "q_value": np.nan,
                        "n_users_a": int(complete_a["user_id"].nunique()),
                        "n_users_b": int(complete_b["user_id"].nunique()),
                        "n_sessions_a": int(
                            complete_a[["user_id", "session_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_sessions_b": int(
                            complete_b[["user_id", "session_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_samples_a": int(len(complete_a)),
                        "n_samples_b": int(len(complete_b)),
                        "n_common_targets": common_targets,
                        "status": status,
                    }
                )

    distances = pd.DataFrame(distance_rows)
    if not distances.empty and distances["p_value"].notna().any():
        distances["q_value"] = distances.groupby(
            ["view"], sort=False
        )["p_value"].transform(_fdr_bh)

    return DatasetComparisonResult(
        metadata={
            "dataset_a": {"label": label_a, **_dataset_metadata(frame_a, accepted_a)},
            "dataset_b": {"label": label_b, **_dataset_metadata(frame_b, accepted_b)},
            "comparison_unit": "user",
            "session_weighting": "equal_within_user",
            "user_weighting": "equal_within_dataset",
            "bootstrap_unit": "sessions_resampled_within_user",
            "permutation_unit": "whole_user",
            "row_pairing": "none_independent_datasets",
            "task_stratification": "none_all_accepted_samples_combined",
            "target_conditioning": "equal_mean_over_targets_present_in_both_datasets",
            "target_normalization": "per-row screen dimensions when valid; otherwise 2736x1824",
            "fdr_scope": "all_family_kpi_tests_within_each_view",
        },
        dataset_distances=distances,
        comparison_coverage=pd.DataFrame(coverage_rows),
        column_compatibility=_column_compatibility(
            frame_a, frame_b, accepted_a, accepted_b, config
        ),
        feature_catalog=_catalog(config),
    )


def _read_summary(path: Path) -> pd.DataFrame:
    identity_dtypes = {
        "user_id": "string",
        "session_id": "string",
        "task_id": "string",
        "sample_id": "string",
    }
    stage(f"Reading {path}...")
    return pd.read_csv(path, dtype=identity_dtypes, low_memory=False)


def compare_csvs(
    input_a: str | Path,
    input_b: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
) -> DatasetComparisonManifest:
    """Compare two CSVs and write a reproducible domain-gap report bundle."""

    input_a = Path(input_a).expanduser().resolve()
    input_b = Path(input_b).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (input_a, input_b):
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")

    resolved_label_a = label_a or input_a.stem
    resolved_label_b = label_b or input_b.stem
    if resolved_label_a == resolved_label_b:
        resolved_label_a = f"{resolved_label_a}_a"
        resolved_label_b = f"{resolved_label_b}_b"

    config = config or AnalysisConfig(
        feature_families=DEFAULT_COMPARISON_FAMILIES,
        views=("raw",),
    )
    result = compare_frames(
        _read_summary(input_a),
        _read_summary(input_b),
        config,
        label_a=resolved_label_a,
        label_b=resolved_label_b,
    )
    stage(f"Writing comparison report to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_distances.csv": result.dataset_distances,
        "comparison_coverage.csv": result.comparison_coverage,
        "column_compatibility.csv": result.column_compatibility,
        "feature_catalog.csv": result.feature_catalog,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)

    files = ("manifest.json", *tables.keys())
    manifest_data = {
        "report_type": "independent_dataset_comparison",
        "input_a": str(input_a),
        "input_b": str(input_b),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "metadata": result.metadata,
        "files": list(files),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)

    return DatasetComparisonManifest(
        input_a=str(input_a),
        input_b=str(input_b),
        output_dir=str(output_dir),
        files=tuple(files),
        metadata=result.metadata,
    )
