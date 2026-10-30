import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from eth_xGaze_inf import (  # noqa: E402
    MEDIAPIPE_ETH_LANDMARKS, denormalize_gaze, landmarks_from_json, load_camera_npz,
    load_face_model, prepare_frame,
)
from eth_xgaze_baseline import frame_metrics  # noqa: E402
from data_normalization import gaze_vector_to_pitchyaw  # noqa: E402


def unit(v):
    return np.asarray(v, dtype=np.float64) / np.linalg.norm(v)


class EthBaselineTests(unittest.TestCase):
    def test_denormalization_inverts_the_label_normalization(self):
        # data_normalization.py: g_norm = R @ g_cam -> (pitch, yaw); the baseline must undo exactly that.
        face_model = load_face_model(str(REPO_ROOT / "evaluation" / "face_model.txt"))
        rng = np.random.default_rng(0)
        landmarks = np.array([[726, 652], [823, 650], [922, 640], [1008, 625], [824, 774], [957, 758]],
                             dtype=np.float64).reshape(6, 1, 2)
        image = np.zeros((1080, 1920, 3), np.uint8)
        camera, distortion = load_camera_npz()
        _, R, _ = prepare_frame(image, landmarks, face_model, camera, distortion)
        for _ in range(20):
            g_cam = unit(rng.normal(size=3) * [1, 1, 0.3] + [0, 0, -1])
            pitchyaw = gaze_vector_to_pitchyaw(R @ g_cam)
            np.testing.assert_allclose(denormalize_gaze(pitchyaw, R), g_cam, atol=1e-5)

    def test_landmark_order_runs_image_left_to_right(self):
        mesh = {str(i): [x, 0] for i, x in zip((33, 133, 362, 263, 129, 358), (726, 823, 922, 1008, 824, 957))}
        lm = landmarks_from_json({"head_pose": {"mediapipe_face_mesh_2d": mesh}})
        self.assertEqual(MEDIAPIPE_ETH_LANDMARKS, (33, 133, 362, 263, 129, 358))
        xs = lm.reshape(6, 2)[:, 0]
        self.assertTrue(np.all(np.diff(xs[:4]) > 0))   # eye corners: outer-L, inner-L, inner-R, outer-R
        self.assertLess(xs[4], xs[5])                    # nose corners: left before right
        self.assertIsNone(landmarks_from_json({"head_pose": {"mediapipe_face_mesh_2d": {"33": [0, 0]}}}))

    def test_screen_intersection_and_errors(self):
        face_center = np.array([10.0, 50.0, 400.0])
        pog = np.array([30.0, 20.0, 0.0])
        label_eth = unit(pog - face_center)
        # Predicting exactly the ETH-origin label (identity normalization) must give zero error.
        pitch, yaw = gaze_vector_to_pitchyaw(label_eth)
        m = frame_metrics(np.array([pitch, yaw]), np.eye(3), face_center, pog)
        self.assertAlmostEqual(m["ang_err_eth"], 0.0, places=2)
        self.assertAlmostEqual(m["pog_err"], 0.0, places=2)
        np.testing.assert_allclose(m["pog_pred"], pog, atol=1e-2)

    def test_gaze_pointing_away_from_screen_has_no_intersection(self):
        m = frame_metrics(np.array([0.0, np.pi]), np.eye(3), np.array([0.0, 0.0, 400.0]),
                          np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.isnan(m["pog_err"]))


if __name__ == "__main__":
    unittest.main()
