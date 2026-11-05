"""Geometric gaze labelling of one ReCalib image: image + setup + screen target -> release-format JSON.

Reproduces the labelling step of the ReCalib pipeline (begibox ff8b084) without its quality
filtering: MediaPipe FaceLandmarker landmarks and head pose, the head pose re-expressed in the
calibrated camera, the gaze origin at the midpoint between the eyes, and the gaze vector towards
the known screen target. Quality-assurance metrics (landmark statistics, eye aspect ratio) are
computed as descriptive values; no sample is filtered, except that an image without a usable
face cannot be labelled (discard code 2, as in the release).
"""
import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2
import mediapipe as mp
import numpy as np

from .geometry import eye_midpoint_cam, head_pose_from_mediapipe, screen_target_mm
from .model import ensure_model

KEY_LANDMARKS = (4, 33, 61, 129, 133, 168, 263, 308, 358, 362, 469, 470, 471, 472, 474, 475, 476, 477)
REQUIRED_LANDMARKS = (33, 133, 263, 362, 469)
EAR_LEFT = (33, 160, 158, 133, 153, 144)     # outer, up-outer, up-inner, inner, down-inner, down-outer
EAR_RIGHT = (362, 385, 387, 263, 373, 380)   # inner, up-inner, up-outer, outer, down-outer, down-inner
HPE_MODEL_ERROR = 2


def _r(values, digits):
    return [round(float(v), digits) + 0.0 for v in values]


def load_setup(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Labeller:
    """Holds the MediaPipe FaceLandmarker and the acquisition setup; label() one image at a time."""

    def __init__(self, setup: Union[dict, str, Path], model_path=None):
        self.setup = load_setup(setup) if not isinstance(setup, dict) else setup
        cam = self.setup["camera"]
        self.camera_matrix = np.asarray(cam["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.asarray(cam["dist_coeffs_k1_k2_p1_p2_k3"], dtype=np.float64)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model(model_path)),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_facial_transformation_matrixes=True)
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def label(self, image: Union[str, Path, np.ndarray], pog_px: Sequence[float],
              image_name: Optional[str] = None, ids: Optional[dict] = None, depth_scale: float = 1.0) -> dict:
        """Label one image.

        image:       path or BGR array (as read by cv2.imread).
        pog_px:      screen target (x, y) in physical pixels of the screen described in setup.json.
        ids:         optional identifiers copied to the output (user_id, session_id, task_id,
                     sample_id, task_type).
        depth_scale: per-user face-scale factor (see face_scale.py); scales the head position and
                     gaze origin along their camera rays. 1.0 = MediaPipe's canonical face size.
        """
        if not isinstance(image, np.ndarray):
            image_name = image_name or Path(image).name
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(image)
        else:
            bgr = image
        h, w = bgr.shape[:2]
        doc = {"image": image_name}
        for key in ("user_id", "session_id", "task_id", "sample_id", "task_type"):
            if ids and key in ids:
                doc[key] = ids[key]
        doc["pog_px"] = {"x": float(pog_px[0]), "y": float(pog_px[1])}

        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                  data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        lms = result.face_landmarks[0] if result.face_landmarks else None
        key_px = self._key_landmarks_px(lms, w, h) if lms else {}
        if (lms is None or not result.facial_transformation_matrixes
                or any(i not in key_px for i in REQUIRED_LANDMARKS)):
            doc.update(gaze=None, head_pose=None, quality_assurance_metrics=None,
                       discard_info={"discarded": True, "discarding_reason": HPE_MODEL_ERROR,
                                     "discarding_reason_name": "HPE_MODEL_ERROR"})
            return doc

        euler, rotvec, t_real, r_cam = head_pose_from_mediapipe(
            result.facial_transformation_matrixes[0], (w, h), self.camera_matrix, self.dist_coeffs)
        # a face s times larger at s times the distance projects identically: scale both together
        origin = np.round(eye_midpoint_cam(r_cam, t_real) * depth_scale, 4)
        t_real = t_real * depth_scale
        pog_mm = screen_target_mm(pog_px, self.setup)
        vector = (pog_mm - origin) / np.linalg.norm(pog_mm - origin)

        doc["gaze"] = {"vector": _r(vector, 6), "origin_mm": _r(origin, 4), "pog_mm": _r(pog_mm, 4)}
        doc["head_pose"] = {"rotation_rad": _r(euler, 4), "rotvec_rad": _r(rotvec, 6),
                            "translation_mm": _r(t_real, 4),
                            "mediapipe_face_mesh_2d": {str(i): list(key_px[i]) for i in KEY_LANDMARKS if i in key_px}}
        doc["quality_assurance_metrics"] = self._qa_metrics(lms, w, h)
        doc["discard_info"] = None
        return doc

    @staticmethod
    def _key_landmarks_px(lms, w, h) -> dict:
        """Selected landmarks in integer pixels; landmarks outside the image are omitted."""
        def inside(v):
            return (v > 0 or math.isclose(0, v)) and (v < 1 or math.isclose(1, v))
        out = {}
        for i in KEY_LANDMARKS:
            p = lms[i]
            if inside(p.x) and inside(p.y):
                out[i] = (min(round(p.x * w), w - 1), min(round(p.y * h), h - 1))
        return out

    @staticmethod
    def _qa_metrics(lms, w, h) -> dict:
        xs = np.array([p.x for p in lms])
        ys = np.array([p.y for p in lms])
        max_x, min_x, max_y, min_y = (round(float(v), 4) for v in (xs.max(), xs.min(), ys.max(), ys.min()))
        px = {i: np.array([lms[i].x * w, lms[i].y * h]) for i in EAR_LEFT + EAR_RIGHT}

        def ear(a, b, c, d, e, f):     # (|b-f| + |c-e|) / (2 |a-d|)
            return (np.linalg.norm(px[b] - px[f]) + np.linalg.norm(px[c] - px[e])) / (2 * np.linalg.norm(px[a] - px[d]))
        left, right = ear(*EAR_LEFT), ear(*EAR_RIGHT)
        return {
            "landmark_statistics_2d": {
                "max_x": max_x, "min_x": min_x, "max_y": max_y, "min_y": min_y,
                "mean_x": round(float(xs.mean()), 4), "mean_y": round(float(ys.mean()), 4),
                "head_bounding_box_surface": round((max_x - min_x) * (max_y - min_y), 4)},
            "eye_aspect_ratio": {"left": round(float(left), 4), "right": round(float(right), 4),
                                 "mean": round(float((left + right) / 2), 4)},
        }
