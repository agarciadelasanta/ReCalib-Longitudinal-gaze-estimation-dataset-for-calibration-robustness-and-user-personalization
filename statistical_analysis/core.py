from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .progress import progress, stage
from .metrics import compare_unit_sets, describe_samples, descriptor_scales


LANDMARK_IDS = (
    4, 33, 61, 129, 133, 168, 263, 308, 358,
    362, 469, 470, 471, 472, 474, 475, 476, 477,
)
LANDMARK_SOURCE_COLUMNS = tuple(
    column
    for landmark_id in LANDMARK_IDS
    for column in (f"lm_{landmark_id}_x", f"lm_{landmark_id}_y")
)

FEATURE_COLUMNS = {
    "target_coverage": ("_pog_norm_x", "_pog_norm_y"),
    "head_rotation": ("head_rot_x", "head_rot_y", "head_rot_z"),
    "gaze_direction": ("_gaze_pitch", "_gaze_yaw"),
    "gaze_origin": ("gaze_origin_x", "gaze_origin_y", "gaze_origin_z"),
    "gaze_point": ("gaze_pog_x", "gaze_pog_y"),
    "head_translation": ("head_trans_x", "head_trans_y", "head_trans_z"),
    "landmark_position": ("_landmark_center_x", "_landmark_center_y"),
    "landmark_scale": ("_landmark_scale",),
    "landmark_shape": tuple(f"_landmark_shape_{index}" for index in range(36)),
    "quality": (
        "qa_max_x", "qa_min_x", "qa_max_y", "qa_min_y",
        "qa_mean_x", "qa_mean_y", "qa_head_bbox_surface",
        "qa_ear_left", "qa_ear_right",
    ),
}

LANDMARK_FAMILIES = ("landmark_position", "landmark_scale", "landmark_shape")
DEFAULT_FEATURE_FAMILIES = tuple(
    family for family in FEATURE_COLUMNS if family not in LANDMARK_FAMILIES
)
LANDMARK_INPUT_COLUMNS = (*LANDMARK_SOURCE_COLUMNS, "img_res_width", "img_res_height")

FEATURE_SOURCES = {
    "target_coverage": ("pog_px_x", "pog_px_y"),
    "head_rotation": FEATURE_COLUMNS["head_rotation"],
    "gaze_direction": ("gaze_vector_x", "gaze_vector_y", "gaze_vector_z"),
    "gaze_origin": FEATURE_COLUMNS["gaze_origin"],
    "gaze_point": ("gaze_pog_x", "gaze_pog_y"),
    "head_translation": FEATURE_COLUMNS["head_translation"],
    "landmark_position": LANDMARK_INPUT_COLUMNS,
    "landmark_scale": LANDMARK_INPUT_COLUMNS,
    "landmark_shape": LANDMARK_INPUT_COLUMNS,
    "quality": FEATURE_COLUMNS["quality"],
}

FEATURE_NOTES = {
    "target_coverage": "Screen target coordinates normalized to [0, 1].",
    "gaze_direction": "Unit gaze vector represented as pitch and yaw in radians.",
    "gaze_origin": "Three-dimensional gaze origin in millimetres.",
    "gaze_point": "Screen-plane gaze point in millimetres; constant z excluded.",
    "head_rotation": "Joint Euler-angle head rotation in radians.",
    "head_translation": "Three-dimensional head translation in millimetres.",
    "landmark_position": "Centroid of facial landmarks normalized by per-sample image width and height.",
    "landmark_scale": "RMS radius around the facial centroid in image-normalized coordinates.",
    "landmark_shape": "Image-normalized landmarks centered and divided by RMS facial scale.",
    "quality": "Joint quality-assurance measurements in native units.",
}

# Label sets: "recalib" uses the released labels; "eth" replaces the label-derived
# families with the ETH-XGaze baseline columns appended by create_dataset_summary.py
# --eth-csv. Families absent from ETH_FEATURE_SOURCES do not depend on the labels and
# are shared; head pose has no ETH counterpart.
LABEL_SETS = ("recalib", "eth")
ETH_ORIGIN_COLUMNS = ("eth_face_center_x", "eth_face_center_y", "eth_face_center_z")
ETH_FEATURE_SOURCES = {
    "gaze_direction": ("eth_label_x", "eth_label_y", "eth_label_z"),
    "gaze_origin": ETH_ORIGIN_COLUMNS,
}
ETH_FEATURE_COLUMNS = {"gaze_origin": ETH_ORIGIN_COLUMNS}
ETH_FEATURE_NOTES = {
    "gaze_direction": "ETH-XGaze label: unit vector from the ETH face centre to the target, as pitch and yaw in radians.",
    "gaze_origin": "ETH-XGaze face centre (solvePnP) in millimetres.",
}
ETH_UNAVAILABLE_FAMILIES = ("head_rotation", "head_translation")
ETH_FILTER_COLUMN = "eth_ok"
# as the released gaze_eth_xgaze (dataset_conversion/add_eth_xgaze_origin.py): a face centre farther
# than this from the camera is a false dlib detection and has no ETH-XGaze label in the release
ETH_DEPTH_COLUMN = "eth_face_center_z"
ETH_MAX_DEPTH_MM = 900.0
# task_type names of the calibration and test tasks: released summaries use
# "calibration"/"test", older ones "9-point"/"16-point".
CALIBRATION_TASK_TYPES = ("calibration", "9-point")
TEST_TASK_TYPES = ("test", "16-point")


def feature_sources(family: str, label_set: str = "recalib") -> tuple[str, ...]:
    if label_set == "eth" and family in ETH_FEATURE_SOURCES:
        return ETH_FEATURE_SOURCES[family]
    return FEATURE_SOURCES[family]


def feature_columns(family: str, label_set: str = "recalib") -> tuple[str, ...]:
    if label_set == "eth" and family in ETH_FEATURE_COLUMNS:
        return ETH_FEATURE_COLUMNS[family]
    return FEATURE_COLUMNS[family]


def feature_note(family: str, label_set: str = "recalib") -> str:
    if label_set == "eth" and family in ETH_FEATURE_NOTES:
        return ETH_FEATURE_NOTES[family]
    return FEATURE_NOTES[family]


def available_families(families: tuple[str, ...], label_set: str) -> tuple[str, ...]:
    """Drop the families a label set cannot provide (used for CLI defaults)."""

    if label_set == "eth":
        return tuple(family for family in families if family not in ETH_UNAVAILABLE_FAMILIES)
    return tuple(families)


def _metric_method(family: str, metric: str) -> str:
    if metric == "location_shift":
        return "robust_standardized_median_location"
    if metric == "dispersion_shift":
        return "absolute_log_iqr_ratio"
    if metric == "distribution_shift":
        return (
            "normalized_wasserstein_1"
            if len(FEATURE_COLUMNS[family]) == 1
            else "robust_scaled_energy_distance"
        )
    if metric == "dependence_shift":
        return "spearman_correlation_matrix_distance"
    raise ValueError(f"Unknown metric: {metric}")


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the public in-memory analysis interface."""

    feature_families: tuple[str, ...] = DEFAULT_FEATURE_FAMILIES
    views: tuple[str, ...] = ("target_conditioned", "raw")
    bootstrap_iterations: int = 500
    permutation_iterations: int = 1_000
    confidence_level: float = 0.95
    random_seed: int = 42
    label_set: str = "recalib"

    def __post_init__(self) -> None:
        if self.label_set not in LABEL_SETS:
            raise ValueError(f"label_set must be one of: {', '.join(LABEL_SETS)}")
        if self.label_set == "eth":
            unavailable = [f for f in self.feature_families if f in ETH_UNAVAILABLE_FAMILIES]
            if unavailable:
                raise ValueError(
                    "Feature families without ETH-XGaze labels: " + ", ".join(unavailable)
                )
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative")
        if self.permutation_iterations < 0:
            raise ValueError("permutation_iterations must be non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
        if not self.feature_families:
            raise ValueError("feature_families must not be empty")
        if not self.views:
            raise ValueError("views must not be empty")


@dataclass
class AnalysisResult:
    """Structured Phase 1 outputs returned by :func:`analyze_frame`."""

    metadata: dict[str, Any]
    pairwise_distances: pd.DataFrame
    user_profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_test_mismatch: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    column_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)


def _prepare_accepted_frame(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    require_task_type: bool = True,
) -> pd.DataFrame:
    """Validate one summary frame, exclude discarded rows, and derive features."""

    label_set = config.label_set
    required = {"user_id", "session_id", "discarded"}
    if require_task_type:
        required.add("task_type")
    if label_set == "eth":
        required.update({ETH_FILTER_COLUMN, ETH_DEPTH_COLUMN})
    if "target_conditioned" in config.views or "target_coverage" in config.feature_families:
        required.update({"pog_px_x", "pog_px_y"})
    for family in config.feature_families:
        if family not in FEATURE_COLUMNS:
            raise ValueError(f"Unknown feature family: {family}")
        required.update(feature_sources(family, label_set))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    accepted = frame.loc[_accepted_mask(frame, label_set)].copy()
    identity_columns = ["user_id", "session_id"]
    if "task_type" in accepted:
        identity_columns.append("task_type")
    for column in identity_columns:
        accepted[column] = accepted[column].astype("string")

    numeric_columns = set()
    for family in config.feature_families:
        numeric_columns.update(feature_sources(family, label_set))
    if "target_conditioned" in config.views:
        numeric_columns.update({"pog_px_x", "pog_px_y"})
        numeric_columns.update(
            column
            for column in ("screen_width_px", "screen_height_px")
            if column in accepted
        )
    for column in sorted(numeric_columns):
        original = accepted[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"Column {column!r} contains {int(invalid.sum())} non-numeric "
                "accepted values"
            )
        accepted[column] = numeric
    accepted = accepted.copy()

    if "gaze_direction" in config.feature_families:
        gaze = accepted[list(feature_sources("gaze_direction", label_set))].to_numpy(float)
        norms = np.linalg.norm(gaze, axis=1, keepdims=True)
        normalized = np.divide(
            gaze,
            norms,
            out=np.full_like(gaze, np.nan),
            where=norms > 1e-12,
        )
        accepted["_gaze_pitch"] = -np.arcsin(np.clip(normalized[:, 1], -1.0, 1.0))
        accepted["_gaze_yaw"] = np.arctan2(-normalized[:, 0], -normalized[:, 2])
    if any(family.startswith("landmark_") for family in config.feature_families):
        landmark_values = accepted[list(LANDMARK_SOURCE_COLUMNS)].to_numpy(float)
        points = landmark_values.reshape(len(accepted), len(LANDMARK_IDS), 2)
        dimensions = accepted[["img_res_width", "img_res_height"]].to_numpy(float)
        if not (np.isfinite(dimensions) & (dimensions > 0)).all():
            raise ValueError(
                "Landmark analysis requires finite, positive img_res_width and "
                "img_res_height for every accepted sample"
            )
        points = points / dimensions[:, None, :]
        centers = np.nanmean(points, axis=1)
        centered = points - centers[:, None, :]
        scales = np.sqrt(np.nanmean(np.sum(centered**2, axis=2), axis=1))
        normalized_shape = np.divide(
            centered,
            scales[:, None, None],
            out=np.full_like(centered, np.nan),
            where=scales[:, None, None] > 1e-12,
        )
        accepted["_landmark_center_x"] = centers[:, 0]
        accepted["_landmark_center_y"] = centers[:, 1]
        accepted["_landmark_scale"] = scales
        shape_frame = pd.DataFrame(
            normalized_shape.reshape(len(accepted), -1),
            index=accepted.index,
            columns=FEATURE_COLUMNS["landmark_shape"],
        )
        accepted = pd.concat([accepted, shape_frame], axis=1)
    if "target_conditioned" in config.views or "target_coverage" in config.feature_families:
        width = pd.Series(2736.0, index=accepted.index)
        height = pd.Series(1824.0, index=accepted.index)
        if "screen_width_px" in accepted:
            observed_width = pd.to_numeric(accepted["screen_width_px"], errors="coerce")
            width = observed_width.where(observed_width > 0, width)
        if "screen_height_px" in accepted:
            observed_height = pd.to_numeric(accepted["screen_height_px"], errors="coerce")
            height = observed_height.where(observed_height > 0, height)
        accepted["_pog_norm_x"] = accepted["pog_px_x"] / width
        accepted["_pog_norm_y"] = accepted["pog_px_y"] / height
        accepted["_target_x"] = accepted["_pog_norm_x"].round(4)
        accepted["_target_y"] = accepted["_pog_norm_y"].round(4)
    return accepted


def _not_discarded(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return ~values.fillna(False)
    normalized = values.astype("string").str.strip().str.lower()
    return ~normalized.isin({"true", "1", "yes"})


def eth_usable(frame: pd.DataFrame) -> pd.Series:
    """Rows with an ETH-XGaze label: processed by the baseline, face centre within ETH_MAX_DEPTH_MM."""

    ok = pd.to_numeric(frame[ETH_FILTER_COLUMN], errors="coerce").eq(1)
    return ok & pd.to_numeric(frame[ETH_DEPTH_COLUMN], errors="coerce").le(ETH_MAX_DEPTH_MM)


def _accepted_mask(frame: pd.DataFrame, label_set: str = "recalib") -> pd.Series:
    """Non-discarded rows; with ETH labels, only those that have one (see eth_usable)."""

    mask = _not_discarded(frame["discarded"])
    if label_set == "eth":
        mask &= eth_usable(frame)
    return mask


def exclusion_counts(frame: pd.DataFrame, accepted: pd.DataFrame, label_set: str) -> dict[str, int]:
    discarded = int((~_not_discarded(frame["discarded"])).sum())
    counts = {"discarded_rows_excluded": discarded}
    if label_set == "eth":
        counts["eth_unavailable_rows_excluded"] = int(len(frame) - len(accepted) - discarded)
    return counts


def _column_catalog(
    frame: pd.DataFrame,
    accepted: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    structural = {
        "image", "sample_id", "task_id", "session_id", "user_id", "task_type",
        "discarded", "discard_code", "discard_info",
    }
    filters = {"discarded"}
    if config.label_set == "eth":
        structural.add(ETH_FILTER_COLUMN)
        filters.update({ETH_FILTER_COLUMN, ETH_DEPTH_COLUMN})
    source_to_families: dict[str, list[str]] = {}
    for family in config.feature_families:
        for column in feature_sources(family, config.label_set):
            source_to_families.setdefault(column, []).append(family)
    if "target_conditioned" in config.views:
        source_to_families.setdefault("pog_px_x", []).append("target_conditioning")
        source_to_families.setdefault("pog_px_y", []).append("target_conditioning")

    rows = []
    for column in frame.columns:
        series = accepted[column]
        missing_fraction = float(series.isna().mean()) if len(series) else np.nan
        unique = int(series.nunique(dropna=True))
        if column in source_to_families:
            role = "feature_source"
            used = True
            reason = ",".join(sorted(set(source_to_families[column])))
        elif column in structural:
            role = "structure"
            used = column in {"session_id", "user_id", "task_type", *filters}
            reason = "grouping/filtering" if used else "identifier_or_audit_field"
        elif missing_fraction == 1.0:
            role = "unusable"
            used = False
            reason = "missing_on_all_accepted_rows"
        elif unique <= 1:
            role = "unusable"
            used = False
            reason = "constant_on_accepted_rows"
        else:
            role = "not_in_phase_1_contract"
            used = False
            reason = "retained_in_source_but_not_selected"
        rows.append(
            {
                "column": column,
                "role": role,
                "used_in_analysis": used,
                "missing_fraction_accepted": missing_fraction,
                "unique_values_accepted": unique,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _describe_units(
    frame: pd.DataFrame, columns: list[str], *, conditioned: bool
) -> pd.DataFrame:
    keys = ["user_id", "session_id"]
    if conditioned:
        keys.extend(["_target_x", "_target_y"])
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        descriptor = describe_samples(group[columns].to_numpy(float))
        row = dict(zip(keys, key))
        row.update(descriptor)
        row["n_samples"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _compare_users(
    units: pd.DataFrame,
    user_a: Any,
    user_b: Any,
    *,
    conditioned: bool,
    scales: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    scales = scales or descriptor_scales(units)
    left = units[units["user_id"].eq(user_a)]
    right = units[units["user_id"].eq(user_b)]
    if not conditioned:
        return compare_unit_sets(left, right, scales)

    target_columns = ["_target_x", "_target_y"]
    left_groups = {
        target: group for target, group in left.groupby(target_columns, sort=False)
    }
    right_groups = {
        target: group for target, group in right.groupby(target_columns, sort=False)
    }
    target_metrics = []
    for target in sorted(left_groups.keys() & right_groups.keys()):
        target_metrics.append(
            compare_unit_sets(left_groups[target], right_groups[target], scales)
        )
    if not target_metrics:
        return {metric: np.nan for metric in (
            "location_shift", "dispersion_shift", "distribution_shift", "dependence_shift"
        )}
    result = {}
    for metric in target_metrics[0]:
        values = np.asarray([item[metric] for item in target_metrics], dtype=float)
        result[metric] = float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan
    return result


def _resample_user_sessions(
    units: pd.DataFrame,
    user: Any,
    rng: np.random.Generator,
) -> pd.DataFrame:
    source = units[units["user_id"].eq(user)]
    sessions = source["session_id"].drop_duplicates().to_numpy()
    if not len(sessions):
        return source.copy()
    parts = []
    for draw, session in enumerate(rng.choice(sessions, size=len(sessions), replace=True)):
        part = source[source["session_id"].eq(session)].copy()
        part["session_id"] = f"bootstrap_{draw}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _bootstrap_intervals(
    units: pd.DataFrame,
    user_a: Any,
    user_b: Any,
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
    for _ in progress(range(iterations), "Bootstrap", unit="draw"):
        boot = pd.concat(
            [
                _resample_user_sessions(units, user_a, rng),
                _resample_user_sessions(units, user_b, rng),
            ],
            ignore_index=True,
        )
        estimates = _compare_users(
            boot, user_a, user_b, conditioned=conditioned, scales=scales
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


def _permutation_p_values(
    units: pd.DataFrame,
    user_a: Any,
    user_b: Any,
    observed: dict[str, float],
    *,
    conditioned: bool,
    iterations: int,
    rng: np.random.Generator,
    scales: dict[str, np.ndarray],
) -> dict[str, float]:
    if iterations <= 0:
        return {}
    pair = units[units["user_id"].isin([user_a, user_b])].copy()
    pair["_cluster"] = list(zip(pair["user_id"], pair["session_id"]))
    clusters = pair["_cluster"].drop_duplicates().tolist()
    n_left = pair.loc[pair["user_id"].eq(user_a), "_cluster"].nunique()
    exceedances = {metric: 0 for metric in observed}
    valid = {metric: 0 for metric in observed}
    for _ in progress(range(iterations), "Permutations", unit="shuffle"):
        shuffled = rng.permutation(len(clusters))
        left_clusters = {clusters[index] for index in shuffled[:n_left]}
        permuted = pair.copy()
        permuted["user_id"] = [
            user_a if cluster in left_clusters else user_b
            for cluster in permuted["_cluster"]
        ]
        estimates = _compare_users(
            permuted,
            user_a,
            user_b,
            conditioned=conditioned,
            scales=scales,
        )
        for metric, estimate in estimates.items():
            if np.isfinite(estimate) and np.isfinite(observed[metric]):
                valid[metric] += 1
                exceedances[metric] += int(estimate >= observed[metric] - 1e-12)
    return {
        metric: (exceedances[metric] + 1.0) / (valid[metric] + 1.0)
        if valid[metric]
        else np.nan
        for metric in observed
    }


def _fdr_bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid.to_numpy(float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.maximum(adjusted, valid.to_numpy(float))
    result.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def _longitudinal_profile_rows(
    units: pd.DataFrame,
    *,
    task_type: str,
    family: str,
    view: str,
    conditioned: bool,
) -> list[dict[str, Any]]:
    scale = descriptor_scales(units)["location"]
    rows = []
    for user, user_units in units.groupby("user_id", sort=True):
        if conditioned:
            references = user_units.groupby(["_target_x", "_target_y"])["location"].apply(
                lambda values: np.stack(values).mean(axis=0)
            )
            cell_distances = []
            for _, unit in user_units.iterrows():
                reference = references.loc[(unit["_target_x"], unit["_target_y"])]
                distance = np.linalg.norm((unit["location"] - reference) / scale) / np.sqrt(len(scale))
                cell_distances.append((unit["session_id"], distance))
            distances = (
                pd.DataFrame(cell_distances, columns=["session_id", "distance"])
                .groupby("session_id")["distance"]
                .mean()
                .to_numpy(float)
            )
        else:
            locations = np.stack(user_units["location"])
            reference = locations.mean(axis=0)
            distances = np.linalg.norm((locations - reference) / scale, axis=1) / np.sqrt(len(scale))

        q75, q25 = np.percentile(distances, [75, 25])
        common = {
            "user_id": str(user),
            "task_type": task_type,
            "view": view,
            "family": family,
            "n_sessions": int(user_units["session_id"].nunique()),
            "n_samples": int(user_units["n_samples"].sum()),
        }
        rows.extend(
            [
                {**common, "metric": "longitudinal_instability_median", "estimate": float(np.median(distances))},
                {**common, "metric": "longitudinal_instability_iqr", "estimate": float(q75 - q25)},
            ]
        )
    return rows


def _task_units(
    raw_units: dict[tuple[str, str], pd.DataFrame],
    family: str,
    task_types: tuple[str, ...],
) -> pd.DataFrame | None:
    return next((raw_units[(family, task)] for task in task_types if (family, task) in raw_units), None)


def _calibration_test_rows(
    raw_units: dict[tuple[str, str], pd.DataFrame],
    config: AnalysisConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for family in config.feature_families:
        calibration = _task_units(raw_units, family, CALIBRATION_TASK_TYPES)
        test = _task_units(raw_units, family, TEST_TASK_TYPES)
        if calibration is None or test is None:
            continue
        scales = descriptor_scales(pd.concat([calibration, test], ignore_index=True))
        keys = calibration[["user_id", "session_id"]].drop_duplicates().merge(
            test[["user_id", "session_id"]].drop_duplicates(),
            on=["user_id", "session_id"],
        )
        family_rows = []
        for key in keys.itertuples(index=False):
            left = calibration[
                calibration["user_id"].eq(key.user_id)
                & calibration["session_id"].eq(key.session_id)
            ]
            right = test[
                test["user_id"].eq(key.user_id)
                & test["session_id"].eq(key.session_id)
            ]
            for metric, estimate in compare_unit_sets(left, right, scales).items():
                family_rows.append(
                    {
                        "user_id": str(key.user_id),
                        "session_id": str(key.session_id),
                        "family": family,
                        "metric": metric,
                        "aggregation": "session",
                        "estimate": estimate,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_sessions": 1,
                    }
                )
        rows.extend(family_rows)
        session_frame = pd.DataFrame(family_rows)
        for (user, metric), group in session_frame.groupby(["user_id", "metric"], sort=True):
            values = group["estimate"].dropna().to_numpy(float)
            if config.bootstrap_iterations > 0 and len(values) > 1:
                boot = np.asarray([
                    rng.choice(values, size=len(values), replace=True).mean()
                    for _ in range(config.bootstrap_iterations)
                ])
                tail = (1.0 - config.confidence_level) / 2.0
                ci_low, ci_high = np.quantile(boot, [tail, 1.0 - tail])
            else:
                ci_low = ci_high = np.nan
            rows.append(
                {
                    "user_id": str(user),
                    "session_id": "ALL",
                    "family": family,
                    "metric": metric,
                    "aggregation": "user_equal_session_mean",
                    "estimate": float(values.mean()) if len(values) else np.nan,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_sessions": int(len(values)),
                }
            )
    return rows


def analyze_frame(frame: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Analyze an in-memory ReCalib summary frame through one stable seam."""

    config = config or AnalysisConfig()
    stage("Preparing individual analysis...")
    accepted = _prepare_accepted_frame(frame, config)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    raw_units: dict[tuple[str, str], pd.DataFrame] = {}
    for task_type, task_frame in accepted.groupby("task_type", sort=True):
        for family in progress(config.feature_families, f"Task {task_type}: features", unit="family"):
            columns = list(feature_columns(family, config.label_set))
            complete = task_frame.dropna(subset=columns)
            if complete.empty:
                stage(f"Skipping {task_type}/{family}: no complete feature rows.")
                continue
            session_counts = complete.groupby("user_id")["session_id"].nunique()
            sample_counts = complete.groupby("user_id").size()
            family_views = tuple(
                view for view in config.views
                if not (family == "target_coverage" and view == "target_conditioned")
            )
            for view in progress(family_views, family, unit="view"):
                if view == "raw":
                    units = _describe_units(complete, columns, conditioned=False)
                    raw_units[(family, str(task_type))] = units
                    users = units["user_id"].drop_duplicates().tolist()
                    conditioned = False

                elif view == "target_conditioned":
                    units = _describe_units(complete, columns, conditioned=True)
                    users = units["user_id"].drop_duplicates().tolist()
                    conditioned = True

                else:
                    raise ValueError(f"Unknown analysis view: {view}")

                profile_rows.extend(
                    _longitudinal_profile_rows(
                        units,
                        task_type=str(task_type),
                        family=family,
                        view=view,
                        conditioned=conditioned,
                    )
                )
                scales = descriptor_scales(units)

                for user_a, user_b in progress(
                    combinations(users, 2), f"{family}/{view}: user pairs",
                    total=len(users) * (len(users) - 1) // 2, unit="pair",
                ):
                    estimates = _compare_users(
                        units,
                        user_a,
                        user_b,
                        conditioned=conditioned,
                        scales=scales,
                    )
                    intervals = _bootstrap_intervals(
                        units,
                        user_a,
                        user_b,
                        conditioned=conditioned,
                        iterations=config.bootstrap_iterations,
                        confidence_level=config.confidence_level,
                        rng=rng,
                        scales=scales,
                    )
                    p_values = _permutation_p_values(
                        units,
                        user_a,
                        user_b,
                        estimates,
                        conditioned=conditioned,
                        iterations=config.permutation_iterations,
                        rng=rng,
                        scales=scales,
                    )
                    for metric, estimate in estimates.items():
                        ci_low, ci_high = intervals.get(metric, (np.nan, np.nan))
                        rows.append({
                            "view": view,
                            "task_type": task_type,
                            "family": family,
                            "metric": metric,
                            "metric_method": _metric_method(family, metric),
                            "user_a": str(user_a),
                            "user_b": str(user_b),
                            "estimate": estimate,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_values.get(metric, np.nan),
                            "q_value": np.nan,
                            "n_sessions_a": int(session_counts.loc[user_a]),
                            "n_sessions_b": int(session_counts.loc[user_b]),
                            "n_samples_a": int(sample_counts.loc[user_a]),
                            "n_samples_b": int(sample_counts.loc[user_b]),
                        })

    pairwise = pd.DataFrame(rows)
    if not pairwise.empty and pairwise["p_value"].notna().any():
        group_columns = ["view", "task_type", "family", "metric"]
        pairwise["q_value"] = pairwise.groupby(
            group_columns, sort=False, group_keys=False
        )["p_value"].apply(_fdr_bh)

    stage("Computing calibration/test mismatch...")
    mismatch = pd.DataFrame(_calibration_test_rows(raw_units, config, rng))

    catalog = pd.DataFrame(
        [
            {
                "family": family,
                "dimensions": len(FEATURE_COLUMNS[family]),
                "source_columns": ",".join(feature_sources(family, config.label_set)),
                "supported_views": "raw" if family == "target_coverage" else ",".join(config.views),
                "notes": feature_note(family, config.label_set),
            }
            for family in config.feature_families
        ]
    )
    column_catalog = _column_catalog(frame, accepted, config)

    return AnalysisResult(
        metadata={
            "label_set": config.label_set,
            "input_rows": int(len(frame)),
            "accepted_rows": int(len(accepted)),
            **exclusion_counts(frame, accepted, config.label_set),
            "users": int(accepted["user_id"].nunique()),
            "user_sessions": int(
                accepted[["user_id", "session_id"]].drop_duplicates().shape[0]
            ),
            "task_counts": {
                str(task): int(count)
                for task, count in accepted.groupby("task_type").size().items()
            },
            "session_weighting": "equal_within_user",
        },
        pairwise_distances=pairwise,
        user_profiles=pd.DataFrame(profile_rows),
        calibration_test_mismatch=mismatch,
        feature_catalog=catalog,
        column_catalog=column_catalog,
    )
