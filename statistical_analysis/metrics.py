from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


METRICS = (
    "location_shift",
    "dispersion_shift",
    "distribution_shift",
    "dependence_shift",
)


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    q75, q25 = np.nanpercentile(values, [75, 25], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    return np.where(scale > 1e-12, scale, 1.0)


def describe_samples(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return robust descriptors for one equally weighted session cell."""

    values = np.asarray(values, dtype=float)
    location = np.nanmedian(values, axis=0)
    q10, q25, q50, q75, q90 = np.nanpercentile(
        values, [10, 25, 50, 75, 90], axis=0
    )
    dispersion = q75 - q25
    distribution = np.concatenate([q10, q25, q50, q75, q90])

    if values.shape[1] < 2:
        dependence = np.empty(0, dtype=float)
    else:
        correlation = pd.DataFrame(values).corr(method="spearman").to_numpy(float)
        correlation = np.nan_to_num(correlation, nan=0.0)
        dependence = correlation[np.triu_indices(values.shape[1], k=1)]

    return {
        "location": location,
        "dispersion": dispersion,
        "distribution": distribution,
        "dependence": dependence,
    }


def descriptor_scales(units: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "location": robust_scale(np.stack(units["location"])),
        "distribution": robust_scale(np.stack(units["distribution"])),
    }


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return np.nan
    cross = cdist(left, right).mean()
    within_left = cdist(left, left).mean()
    within_right = cdist(right, right).mean()
    squared = max(0.0, 2.0 * cross - within_left - within_right)
    return float(np.sqrt(squared))


def compare_unit_sets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    scales: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the fixed KPI vector between two sets of session descriptors."""

    left_location = np.stack(left["location"])
    right_location = np.stack(right["location"])
    location_delta = (
        left_location.mean(axis=0) - right_location.mean(axis=0)
    ) / scales["location"]

    epsilon = 1e-9
    left_dispersion = np.stack(left["dispersion"])
    right_dispersion = np.stack(right["dispersion"])
    dispersion_delta = np.log(
        (np.median(left_dispersion, axis=0) + epsilon)
        / (np.median(right_dispersion, axis=0) + epsilon)
    )

    left_distribution_raw = np.stack(left["distribution"])
    right_distribution_raw = np.stack(right["distribution"])
    left_distribution = left_distribution_raw / scales["distribution"]
    right_distribution = right_distribution_raw / scales["distribution"]

    left_dependence = np.stack(left["dependence"])
    right_dependence = np.stack(right["dependence"])
    if left_dependence.shape[1] == 0:
        dependence_shift = np.nan
    else:
        dependence_shift = float(
            np.linalg.norm(
                left_dependence.mean(axis=0) - right_dependence.mean(axis=0)
            )
            / np.sqrt(left_dependence.shape[1])
        )

    if left_location.shape[1] == 1:
        distribution_shift = float(
            wasserstein_distance(
                left_distribution_raw.ravel() / scales["location"][0],
                right_distribution_raw.ravel() / scales["location"][0],
            )
        )
    else:
        distribution_shift = _energy_distance(left_distribution, right_distribution)

    return {
        "location_shift": float(
            np.linalg.norm(location_delta) / np.sqrt(location_delta.size)
        ),
        "dispersion_shift": float(
            np.linalg.norm(dispersion_delta) / np.sqrt(dispersion_delta.size)
        ),
        "distribution_shift": distribution_shift,
        "dependence_shift": dependence_shift,
    }
