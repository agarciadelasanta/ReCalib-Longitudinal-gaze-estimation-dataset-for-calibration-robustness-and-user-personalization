"""Geometry of the ReCalib labels: screen targets, real-camera head pose and gaze origin.

All 3D quantities are in mm in the camera frame: +x towards the right of the image, +y downwards,
+z away from the camera (OpenCV convention).
"""
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .canonical_face import FACE_MODEL

MEDIAPIPE_VERTICAL_FOV_DEG = 63.0               # MediaPipe's virtual camera (face geometry module)
CORRECTION_MATRIX = np.array([[1, -1, -1],      # MediaPipe (OpenGL axes) -> OpenCV axes, applied
                              [-1, 1, 1],        # element-wise to the rotation: equals D R D with
                              [-1, 1, 1]])       # D = diag(1, -1, -1)
CORRECTION_VECTOR = np.array([10, -10, -10])    # translation: cm -> mm and OpenGL -> OpenCV axes
EYE_CORNERS = (33, 133, 263, 362)                # canonical-face eye corners (gaze origin)
FACE_CENTROID_HEAD_MM = FACE_MODEL.mean(axis=0) * 10.0


def orthonormalize(m: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to m (SVD); MediaPipe's rotation is only approximately orthonormal."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def virtual_focal_length_px(image_height: int, vertical_fov_deg: float = MEDIAPIPE_VERTICAL_FOV_DEG) -> float:
    return (image_height / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)


def mediapipe_translation_to_real_camera(t_mp_mm: Sequence[float], image_size: Tuple[int, int],
                                         camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                         rotation_cam: np.ndarray,
                                         pivot_head_mm: Optional[np.ndarray] = None,
                                         vertical_fov_deg: float = MEDIAPIPE_VERTICAL_FOV_DEG) -> np.ndarray:
    """Re-express MediaPipe's head translation in the calibrated camera.

    MediaPipe estimates the pose as if the image came from a virtual pinhole camera (63 deg
    vertical field of view, principal point at the image centre). The pivot point of the rigid
    head (the face centroid by default) keeps its image position and apparent size: its depth is
    scaled by f_real / f_virtual and its lateral position is back-projected with the real
    intrinsics, including lens distortion. Returns the head-model origin in the real camera, mm.
    """
    pivot = FACE_CENTROID_HEAD_MM if pivot_head_mm is None else np.asarray(pivot_head_mm, float)
    offset = np.asarray(rotation_cam, float) @ pivot
    width, height = image_size
    f_virtual = virtual_focal_length_px(height, vertical_fov_deg)
    pivot_mp = np.asarray(t_mp_mm, float) + offset
    if pivot_mp[2] <= 0:
        raise ValueError("MediaPipe pose has non-positive depth")
    uv = f_virtual * pivot_mp[:2] / pivot_mp[2] + np.array([width / 2.0, height / 2.0])
    z_real = pivot_mp[2] * camera_matrix[1, 1] / f_virtual
    ray = cv2.undistortPoints(uv.reshape(1, 1, 2), camera_matrix, dist_coeffs).reshape(2)
    return np.array([ray[0] * z_real, ray[1] * z_real, z_real]) - offset


def head_pose_from_mediapipe(transformation_matrix: np.ndarray, image_size: Tuple[int, int],
                             camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """MediaPipe facial transformation matrix -> (euler_zyx_rad, rotvec_rad, translation_mm, R_cam)."""
    m = np.asarray(transformation_matrix, float)
    rotation_cv = m[:3, :3] * CORRECTION_MATRIX
    euler_zyx = Rotation.from_matrix(rotation_cv).as_euler("zyx")       # [roll, yaw, pitch]
    r_cam = orthonormalize(rotation_cv)
    rotvec = Rotation.from_matrix(r_cam).as_rotvec()
    t_mp = m[:3, 3] * CORRECTION_VECTOR
    t_real = mediapipe_translation_to_real_camera(t_mp, image_size, camera_matrix, dist_coeffs, r_cam)
    return euler_zyx, rotvec, t_real, r_cam


def eye_midpoint_cam(rotation_cam: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    """Midpoint between the eyes (canonical eye corners) in the camera frame, mm."""
    midpoint_head = FACE_MODEL[list(EYE_CORNERS)].mean(axis=0) * 10.0
    return np.asarray(rotation_cam, float) @ midpoint_head + np.asarray(translation_mm, float)


def screen_target_mm(pog_px: Sequence[float], setup: dict) -> np.ndarray:
    """Screen target in physical pixels (origin top-left) -> camera frame, mm.

    Uses the screen size and the camera pose relative to the screen from setup.json
    (screen orientation 0, as in ReCalib).
    """
    scr, cam = setup["screen"], setup["camera"]
    if scr.get("orientation", 0) != 0:
        raise NotImplementedError("only screen orientation 0 is supported")
    p_screen = np.array([pog_px[0] / scr["width_px"] * scr["width_mm"],
                         pog_px[1] / scr["height_px"] * scr["height_mm"], 0.0])
    pos, rot = cam["position_mm"], cam["rotation_rad"]
    t = np.array([pos["x"] + scr["width_mm"] / 2.0, -pos["y"], pos.get("z", 0.0)])
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [rot["x"], rot["y"], rot["z"]]).as_matrix()
    m[:3, 3] = t
    return (np.linalg.inv(m) @ np.append(p_screen, 1.0))[:3]
