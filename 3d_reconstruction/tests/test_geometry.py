"""Geometry of the ReCalib labels (no MediaPipe needed)."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recalib_reconstruction.geometry import (FACE_CENTROID_HEAD_MM, eye_midpoint_cam,
                                        mediapipe_translation_to_real_camera, orthonormalize,
                                        screen_target_mm, virtual_focal_length_px)

SETUP = {
    "screen": {"width_mm": 260, "height_mm": 173, "width_px": 2736, "height_px": 1824, "orientation": 0},
    "camera": {"position_mm": {"x": 0, "y": 10, "z": 0}, "rotation_rad": {"x": 0.0, "y": np.pi, "z": 0.0}},
}
K = np.array([[1475.1831, 0, 962.2621], [0, 1471.5504, 576.2511], [0, 0, 1.0]])
DIST = np.array([0.191645, -0.576145, 0.005257, 0.003231, 0.629024])
SIZE = (1920, 1080)


@pytest.mark.parametrize("px, py", [(10, 10), (1368, 912), (2726, 1814), (915, 612)])
def test_screen_target_matches_recalib_geometry(px, py):
    # camera 10 mm above the top edge, horizontally centred, facing the user
    expected = [130 - px * 260 / 2736, 10 + py * 173 / 1824, 0.0]
    np.testing.assert_allclose(screen_target_mm((px, py), SETUP), expected, atol=1e-9)


def test_virtual_focal_length():
    assert virtual_focal_length_px(1080) == pytest.approx(881.1, abs=0.1)


@pytest.mark.parametrize("dist", [np.zeros(5), DIST], ids=["no-distortion", "recalib-distortion"])
def test_recovers_head_position(dist):
    """Simulate what MediaPipe reports for a known head pose and check it is recovered."""
    r = Rotation.from_euler("yx", [20, 10], degrees=True).as_matrix()
    t_true = np.array([30.0, 40.0, 560.0])
    pivot_true = r @ FACE_CENTROID_HEAD_MM + t_true
    uv = cv2.projectPoints(pivot_true.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(2)
    f_v = virtual_focal_length_px(SIZE[1])
    z_mp = pivot_true[2] * f_v / K[1, 1]
    pivot_mp = np.array([*(uv - np.array(SIZE) / 2) * z_mp / f_v, z_mp])
    t_mp = pivot_mp - r @ FACE_CENTROID_HEAD_MM
    t_real = mediapipe_translation_to_real_camera(t_mp, SIZE, K, dist, r)
    np.testing.assert_allclose(t_real, t_true, atol=0.5)


def test_orthonormalize_returns_rotation():
    m = Rotation.from_euler("zyx", [5, 10, 15], degrees=True).as_matrix() + 1e-3
    r = orthonormalize(m)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_eye_midpoint_identity_rotation():
    origin = eye_midpoint_cam(np.eye(3), np.array([0.0, 0.0, 500.0]))
    assert origin[2] < 500.0          # the eyes lie in front of the canonical face origin
    assert abs(origin[0]) < 1.0       # and are horizontally centred
