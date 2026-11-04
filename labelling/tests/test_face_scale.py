"""Per-user face scale from the iris size."""
import numpy as np
import pytest

from recalib_labelling.face_scale import (IRIS_DIAMETER_MM, RECALIB_FACTORS, iris_depth_mm, iris_diameter_px,
                                          load_factors, user_factors)

K = np.array([[1475.0, 0, 960], [0, 1475.0, 540], [0, 0, 1.0]])


def ring(cx, cy, r, ids):
    a, b, c, d = ids     # right, top, left, bottom
    return {a: (cx + r, cy), b: (cx, cy - r), c: (cx - r, cy), d: (cx, cy + r)}


def eyes(r):
    return {**ring(900, 500, r, (469, 470, 471, 472)), **ring(1020, 500, r, (474, 475, 476, 477))}


def test_iris_diameter_and_depth():
    lms = eyes(17.25)
    assert iris_diameter_px(lms) == pytest.approx(34.5)
    assert iris_diameter_px({str(k): v for k, v in lms.items()}) == pytest.approx(34.5)   # JSON keys
    assert iris_diameter_px(lms, K, np.zeros(5)) == pytest.approx(34.5)                 # no distortion
    assert iris_depth_mm(34.5, K) == pytest.approx(1475 * IRIS_DIAMETER_MM / 34.5)
    assert iris_diameter_px({k: v for k, v in lms.items() if k != 470}) is None


def test_user_factor_is_median_ratio_of_kept_images():
    z_iris = 1475 * IRIS_DIAMETER_MM / 34.5                      # 500.2 mm
    recs = [{"user_id": "00", "origin_z": z_iris / f, "kept": True, "landmarks": eyes(17.25)}
            for f in (0.98, 1.02, 1.03, 1.04, 1.10)]
    recs.append({"user_id": "00", "origin_z": z_iris / 2.0, "kept": False, "landmarks": eyes(17.25)})
    factor, n, d_px = user_factors(recs, K, np.zeros(5))["00"]
    assert factor == pytest.approx(1.03) and n == 5 and d_px == pytest.approx(34.5)


def test_recalib_factors_file():
    f = load_factors(RECALIB_FACTORS)
    assert sorted(f) == [f"{i:02d}" for i in range(9)]
    assert all(0.9 < v < 1.1 for v in f.values())
