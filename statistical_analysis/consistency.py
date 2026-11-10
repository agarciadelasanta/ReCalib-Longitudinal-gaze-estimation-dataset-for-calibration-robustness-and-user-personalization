"""Consistency between gaze-vector label sets: do they vary together?

Each label set gives one unit gaze vector per sample, compared as pitch and yaw in
degrees. A constant offset between label sets is expected (they use different gaze
origins); the question is whether their variations agree. Before comparing, each
label set is centred on its own mean at the level under test:

- within_session / raw: centred on the session mean. The variation includes the
  target layout, which every label set shares, so agreement here is a sanity check.
- within_session / target_conditioned: centred on the (session, target) mean. What
  remains is the frame-to-frame variation while one target is fixated, i.e. how
  each label set registers the head moving the gaze origin.
- across_sessions / target_conditioned: per-(session, target) means centred on the
  user's mean for that target across sessions, then averaged over targets, giving
  one deviation per session (session-to-session drift of each label set).

Agreement is summarized per unit (session, or user for across_sessions) and pooled.
Pooled confidence intervals resample sessions with replacement within each user.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .progress import progress, stage
from .core import ETH_DEPTH_COLUMN, ETH_FILTER_COLUMN, _not_discarded, eth_usable
from .reporting import AnalysisManifest


VECTOR_SOURCES = {
    "recalib": ("gaze_vector_x", "gaze_vector_y", "gaze_vector_z"),
    "eth": ("eth_label_x", "eth_label_y", "eth_label_z"),
    "eth_pred": ("eth_pred_vec_x", "eth_pred_vec_y", "eth_pred_vec_z"),
}
VECTOR_NOTES = {
    "recalib": "ReCalib label: gaze origin to target.",
    "eth": "ETH-XGaze label: ETH face centre (solvePnP) to target.",
    "eth_pred": "ETH-XGaze baseline prediction from the image.",
}
ETH_VECTORS = ("eth", "eth_pred")
COMPONENTS = ("pitch", "yaw")
TARGET_COLUMNS = ("pog_px_x", "pog_px_y")
LEVELS = (
    ("within_session", "raw"),
    ("within_session", "target_conditioned"),
    ("across_sessions", "target_conditioned"),
)
ALL_TASKS = "all"
_MIN_POINTS = 3
# Order of the per-session sufficient statistics used for pooling and the bootstrap.
_SUMS = ("n", "sx", "sy", "sxx", "syy", "sxy")
_POOLED = ("pearson_r", "sd_ratio", "slope", "diff_sd_deg")


@dataclass(frozen=True)
class ConsistencyConfig:
    vectors: tuple[str, ...] = ("recalib", "eth")
    min_session_samples: int = 100
    bootstrap_iterations: int = 500
    confidence_level: float = 0.95
    random_seed: int = 42

    def __post_init__(self) -> None:
        unknown = [vector for vector in self.vectors if vector not in VECTOR_SOURCES]
        if unknown:
            raise ValueError("Unknown gaze vectors: " + ", ".join(unknown))
        if len(set(self.vectors)) < 2:
            raise ValueError("At least two distinct gaze vectors are required")
        if self.min_session_samples < 0:
            raise ValueError("min_session_samples must be non-negative")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass
class ConsistencyResult:
    metadata: dict[str, Any]
    summary: pd.DataFrame
    units: pd.DataFrame
    offsets: pd.DataFrame
    session_deviations: pd.DataFrame


def _pitch_yaw_deg(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as the gaze_direction family in core.py, in degrees."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.full_like(vectors, np.nan), where=norms > 1e-12)
    pitch = -np.arcsin(np.clip(unit[:, 1], -1.0, 1.0))
    yaw = np.arctan2(-unit[:, 0], -unit[:, 2])
    return np.degrees(pitch), np.degrees(yaw)


def _angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=1), -1.0, 1.0)))


def _prepare_frame(
    frame: pd.DataFrame, config: ConsistencyConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep accepted samples with every selected vector and derive pitch/yaw columns.

    Sessions left with fewer than ``min_session_samples`` such samples are dropped: a
    handful of frames cannot anchor a session mean, and the across-session level gives
    every session equal weight.
    """

    uses_eth = any(vector in ETH_VECTORS for vector in config.vectors)
    required = {"user_id", "session_id", "task_type", "discarded", *TARGET_COLUMNS}
    for vector in config.vectors:
        required.update(VECTOR_SOURCES[vector])
    if uses_eth:
        required.update({ETH_FILTER_COLUMN, ETH_DEPTH_COLUMN})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    not_discarded = _not_discarded(frame["discarded"])
    mask = not_discarded.copy()
    if uses_eth:
        mask &= eth_usable(frame)
    accepted = frame.loc[mask].copy()
    for column in ("user_id", "session_id", "task_type"):
        accepted[column] = accepted[column].astype("string")
    for column in TARGET_COLUMNS:
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce")

    valid = accepted[list(TARGET_COLUMNS)].notna().all(axis=1).to_numpy()
    for vector in config.vectors:
        values = accepted[list(VECTOR_SOURCES[vector])].apply(pd.to_numeric, errors="coerce")
        pitch, yaw = _pitch_yaw_deg(values.to_numpy(float))
        accepted[f"{vector}_pitch"] = pitch
        accepted[f"{vector}_yaw"] = yaw
        valid &= np.isfinite(pitch) & np.isfinite(yaw)
    counts = {"discarded_rows_excluded": int((~not_discarded).sum())}
    if uses_eth:
        counts["eth_unavailable_rows_excluded"] = int(len(frame) - len(accepted) - counts["discarded_rows_excluded"])
    counts["invalid_vector_rows_excluded"] = int((~valid).sum())
    accepted = accepted.loc[valid]
    session_size = accepted.groupby(["user_id", "session_id"])["user_id"].transform("size")
    small = session_size < config.min_session_samples
    counts["small_session_rows_excluded"] = int(small.sum())
    counts["small_sessions_excluded"] = int(
        accepted.loc[small, ["user_id", "session_id"]].drop_duplicates().shape[0]
    )
    return accepted.loc[~small].copy(), counts


def _strata(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    strata = [(ALL_TASKS, frame)]
    strata.extend((str(task), group) for task, group in frame.groupby("task_type", sort=True))
    return strata


def _angle_columns(vectors: tuple[str, ...]) -> list[str]:
    return [f"{vector}_{component}" for vector in vectors for component in COMPONENTS]


def _within_session(frame: pd.DataFrame, vectors: tuple[str, ...], view: str) -> pd.DataFrame:
    columns = _angle_columns(vectors)
    keys = ["user_id", "session_id"]
    if view == "target_conditioned":
        keys.extend(TARGET_COLUMNS)
    centred = frame[columns] - frame.groupby(keys, sort=False)[columns].transform("mean")
    return pd.concat([frame[["user_id", "session_id"]], centred], axis=1)


def _session_deviations(frame: pd.DataFrame, vectors: tuple[str, ...]) -> pd.DataFrame:
    """One deviation per session: per-target session means minus the user's per-target mean."""

    columns = _angle_columns(vectors)
    cells = frame.groupby(["user_id", "session_id", *TARGET_COLUMNS], sort=True)
    means = cells[columns].mean()
    means["_samples"] = cells.size()
    by_target = means.groupby(level=["user_id", *TARGET_COLUMNS])
    repeated = by_target["_samples"].transform("size") >= 2
    deviations = (means[columns] - by_target[columns].transform("mean"))[repeated]
    deviations["_samples"] = means.loc[repeated, "_samples"]
    sessions = deviations.groupby(level=["user_id", "session_id"])
    result = sessions[columns].mean()
    result["n_targets"] = sessions.size()
    result["n_samples"] = sessions["_samples"].sum()
    return result.reset_index()


def _sums(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([len(x), x.sum(), y.sum(), x @ x, y @ y, x @ y], dtype=float)


def _from_sums(sums: np.ndarray) -> dict[str, float]:
    n, sx, sy, sxx, syy, sxy = sums
    if n < _MIN_POINTS:
        return dict.fromkeys(("sd_a_deg", "sd_b_deg", *_POOLED), np.nan)
    var_a = (sxx - sx * sx / n) / (n - 1)
    var_b = (syy - sy * sy / n) / (n - 1)
    cov = (sxy - sx * sy / n) / (n - 1)
    sd_a, sd_b = np.sqrt(max(var_a, 0.0)), np.sqrt(max(var_b, 0.0))
    degenerate = sd_a <= 1e-12 or sd_b <= 1e-12
    return {
        "sd_a_deg": float(sd_a),
        "sd_b_deg": float(sd_b),
        "pearson_r": np.nan if degenerate else float(cov / (sd_a * sd_b)),
        "sd_ratio": np.nan if sd_a <= 1e-12 else float(sd_b / sd_a),
        "slope": np.nan if sd_a <= 1e-12 else float(cov / var_a),
        "diff_sd_deg": float(np.sqrt(max(var_a + var_b - 2.0 * cov, 0.0))),
    }


def _agreement(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    result = {"n_samples": int(len(x)), **_from_sums(_sums(x, y))}
    result["spearman_rho"] = (
        float(spearmanr(x, y).statistic) if np.isfinite(result["pearson_r"]) else np.nan
    )
    result["status"] = "ok" if np.isfinite(result["pearson_r"]) else "insufficient_data"
    return result


def _bootstrap(
    session_sums: np.ndarray,
    session_users: np.ndarray,
    config: ConsistencyConfig,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals from resampling sessions with replacement within each user."""

    if config.bootstrap_iterations <= 0:
        return {}
    groups = [np.flatnonzero(session_users == user) for user in np.unique(session_users)]
    draws = np.empty((config.bootstrap_iterations, len(_POOLED)))
    for iteration in range(config.bootstrap_iterations):
        index = np.concatenate([rng.choice(group, size=len(group)) for group in groups])
        estimate = _from_sums(session_sums[index].sum(axis=0))
        draws[iteration] = [estimate[name] for name in _POOLED]
    tail = (1.0 - config.confidence_level) / 2.0
    intervals = {}
    for column, name in enumerate(_POOLED):
        values = draws[:, column]
        if np.isfinite(values).any():
            intervals[name] = (
                float(np.nanquantile(values, tail)),
                float(np.nanquantile(values, 1.0 - tail)),
            )
    return intervals


def _compare_pair(
    points: pd.DataFrame,
    pair: tuple[str, str],
    component: str,
    *,
    level: str,
    config: ConsistencyConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pooled and per-unit agreement for one vector pair and angle component."""

    column_a, column_b = (f"{vector}_{component}" for vector in pair)
    points = points.dropna(subset=[column_a, column_b])
    unit_keys = ["user_id"] if level == "across_sessions" else ["user_id", "session_id"]
    unit_rows = []
    for key, group in points.groupby(unit_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(unit_keys, key))
        row.update(_agreement(group[column_a].to_numpy(float), group[column_b].to_numpy(float)))
        unit_rows.append(row)

    session_groups = list(points.groupby(["user_id", "session_id"], sort=True))
    session_sums = np.array(
        [_sums(g[column_a].to_numpy(float), g[column_b].to_numpy(float)) for _, g in session_groups]
    ).reshape(-1, len(_SUMS))
    session_users = np.array([key[0] for key, _ in session_groups], dtype=object)
    pooled = _agreement(points[column_a].to_numpy(float), points[column_b].to_numpy(float))
    unit_r = np.array([row["pearson_r"] for row in unit_rows], dtype=float)
    pooled.update(
        n_users=int(points["user_id"].nunique()),
        n_sessions=len(session_groups),
        n_units=int(np.isfinite(unit_r).sum()),
        unit_r_median=float(np.nanmedian(unit_r)) if np.isfinite(unit_r).any() else np.nan,
        unit_r_min=float(np.nanmin(unit_r)) if np.isfinite(unit_r).any() else np.nan,
    )
    intervals = _bootstrap(session_sums, session_users, config, rng)
    for name in _POOLED:
        pooled[f"{name}_ci_low"], pooled[f"{name}_ci_high"] = intervals.get(name, (np.nan, np.nan))
    return pooled, unit_rows


def _offset_rows(frame: pd.DataFrame, pair: tuple[str, str], task_type: str) -> list[dict[str, Any]]:
    """Per-session constant offset (b - a) and its within-session spread."""

    vector_a, vector_b = pair
    difference = pd.DataFrame(
        {
            component: frame[f"{vector_b}_{component}"] - frame[f"{vector_a}_{component}"]
            for component in COMPONENTS
        }
    )
    difference["angle"] = _angle_deg(
        frame[list(VECTOR_SOURCES[vector_a])].to_numpy(float),
        frame[list(VECTOR_SOURCES[vector_b])].to_numpy(float),
    )
    difference[["user_id", "session_id"]] = frame[["user_id", "session_id"]]
    rows = []
    for (user, session), group in difference.groupby(["user_id", "session_id"], sort=True):
        row = {
            "task_type": task_type, "vector_a": vector_a, "vector_b": vector_b,
            "user_id": user, "session_id": session, "n_samples": int(len(group)),
        }
        for component in COMPONENTS:
            row[f"mean_offset_{component}_deg"] = float(group[component].mean())
            row[f"sd_offset_{component}_deg"] = float(group[component].std(ddof=1))
        row["median_angle_deg"] = float(group["angle"].median())
        rows.append(row)
    return rows


def analyze_consistency_frame(
    frame: pd.DataFrame, config: ConsistencyConfig | None = None
) -> ConsistencyResult:
    config = config or ConsistencyConfig()
    stage("Preparing gaze vectors...")
    accepted, exclusions = _prepare_frame(frame, config)
    if accepted.empty:
        raise ValueError("No accepted samples with every selected gaze vector")
    rng = np.random.default_rng(config.random_seed)
    pairs = list(combinations(config.vectors, 2))
    strata = _strata(accepted)

    summary_rows, unit_rows, offset_rows, deviation_frames = [], [], [], []
    tasks = [
        (task_type, stratum, level, view)
        for task_type, stratum in strata
        for level, view in LEVELS
    ]
    for task_type, stratum, level, view in progress(tasks, "Consistency", unit="level"):
        if level == "across_sessions":
            points = _session_deviations(stratum, config.vectors)
            deviation_frames.append(points.assign(task_type=task_type))
        else:
            points = _within_session(stratum, config.vectors, view)
        context = {"level": level, "view": view, "task_type": task_type}
        for pair in pairs:
            for component in COMPONENTS:
                labels = {**context, "vector_a": pair[0], "vector_b": pair[1], "component": component}
                pooled, units = _compare_pair(
                    points, pair, component, level=level, config=config, rng=rng
                )
                summary_rows.append({**labels, **pooled})
                unit_rows.extend({**labels, **row} for row in units)
    for task_type, stratum in strata:
        for pair in pairs:
            offset_rows.extend(_offset_rows(stratum, pair, task_type))

    deviations = pd.concat(deviation_frames, ignore_index=True)
    deviations = deviations[
        ["task_type", "user_id", "session_id", "n_targets", "n_samples", *_angle_columns(config.vectors)]
    ]
    summary_columns = [
        "level", "view", "task_type", "vector_a", "vector_b", "component",
        "n_users", "n_sessions", "n_units", "n_samples",
        "pearson_r", "pearson_r_ci_low", "pearson_r_ci_high", "spearman_rho",
        "unit_r_median", "unit_r_min", "sd_a_deg", "sd_b_deg",
        "sd_ratio", "sd_ratio_ci_low", "sd_ratio_ci_high",
        "slope", "slope_ci_low", "slope_ci_high",
        "diff_sd_deg", "diff_sd_deg_ci_low", "diff_sd_deg_ci_high", "status",
    ]
    unit_columns = [
        "level", "view", "task_type", "vector_a", "vector_b", "component",
        "user_id", "session_id", "n_samples", "pearson_r", "spearman_rho",
        "sd_a_deg", "sd_b_deg", "sd_ratio", "slope", "diff_sd_deg", "status",
    ]
    metadata = {
        "input_rows": int(len(frame)),
        "accepted_rows": int(len(accepted)),
        **exclusions,
        "users": int(accepted["user_id"].nunique()),
        "user_sessions": int(accepted[["user_id", "session_id"]].drop_duplicates().shape[0]),
        "task_types": sorted(accepted["task_type"].dropna().unique().tolist()),
        "vectors": {vector: {"columns": list(VECTOR_SOURCES[vector]), "note": VECTOR_NOTES[vector]}
                    for vector in config.vectors},
        "angle_units": "degrees",
        "offset_sign": "vector_b - vector_a",
        "slope": "ordinary least squares of vector_b on vector_a",
        "bootstrap_unit": "session, resampled within user",
    }
    return ConsistencyResult(
        metadata=metadata,
        summary=pd.DataFrame(summary_rows).reindex(columns=summary_columns),
        units=pd.DataFrame(unit_rows).reindex(columns=unit_columns),
        offsets=pd.DataFrame(offset_rows),
        session_deviations=deviations,
    )


def analyze_consistency_csv(
    input_path: str | Path,
    output_dir: str | Path,
    config: ConsistencyConfig | None = None,
) -> AnalysisManifest:
    """Analyze a summary CSV and write the gaze-consistency report bundle."""

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    config = config or ConsistencyConfig()
    identity_dtypes = {"user_id": "string", "session_id": "string", "task_id": "string", "sample_id": "string"}
    stage(f"Reading {input_path}...")
    frame = pd.read_csv(input_path, dtype=identity_dtypes, low_memory=False)
    result = analyze_consistency_frame(frame, config)
    stage(f"Writing gaze-consistency report to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "consistency_summary.csv": result.summary,
        "consistency_units.csv": result.units,
        "label_offsets.csv": result.offsets,
        "session_deviations.csv": result.session_deviations,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    files = ("manifest.json", *tables)
    manifest_data = {
        "report_type": "gaze_consistency",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "metadata": result.metadata,
        "files": list(files),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)
    return AnalysisManifest(
        input_path=str(input_path),
        output_dir=str(output_dir),
        files=files,
        metadata=result.metadata,
    )
