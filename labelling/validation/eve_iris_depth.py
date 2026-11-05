"""Iris-based depth on EVE: can the iris size replace MediaPipe's fixed face scale?

    python -m validation.eve_iris_depth --frames-root EVE/frames --h5-root EVE/webcam_h5 \
        --out eve_eval/eve_iris.csv [--stride 50]

Same frames, target and reference origin as eve_compare.py. For each frame it also stores the iris
diameters in pixels from the MediaPipe iris landmarks (ring 469-472 and ring 474-477; opposite points
469-471 / 470-472 and 474-476 / 475-477), so the depth implied by a fixed iris diameter can be compared
with our origin and with EVE's.
"""
import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np

from recalib_labelling.geometry import eye_midpoint_cam, head_pose_from_mediapipe
from recalib_labelling.labeller import REQUIRED_LANDMARKS, Labeller

IRIS_PAIRS = {"a1": (469, 471), "a2": (470, 472), "b1": (474, 476), "b2": (475, 477)}
COLUMNS = (["recording", "frame", "fx", "gt_x", "gt_y", "gt_z", "ours_x", "ours_y", "ours_z",
            "target_x", "target_y", "target_z"] + [f"iris_{k}_px" for k in IRIS_PAIRS])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-root", required=True, type=Path)
    ap.add_argument("--h5-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--model")
    args = ap.parse_args(argv)

    recs = sorted(p.parent.relative_to(args.frames_root).as_posix() for p in args.frames_root.glob("*/*/webcam_c.zip"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = {"camera": {"camera_matrix": np.eye(3).tolist(), "dist_coeffs_k1_k2_p1_p2_k3": [0] * 5}}
    n_ok = n_noface = 0
    start = time.time()
    with Labeller(dummy, model_path=args.model) as labeller, open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for rec in recs:
            with h5py.File(args.h5_root / rec / "webcam_c.h5", "r") as h:
                K = h["camera_matrix"][()].astype(np.float64)
                T = h["camera_transformation"][()].astype(np.float64)
                mpp = h["millimeters_per_pixel"][()].astype(np.float64)
                pog, lo, ro = (h[k + "/data"][()].astype(np.float64) for k in ("face_PoG_tobii", "left_o", "right_o"))
                valid = (h["face_PoG_tobii/validity"][()] & h["left_o/validity"][()] & h["right_o/validity"][()]
                         & h["face_o/validity"][()] & h["face_g_tobii/validity"][()])
            zf = zipfile.ZipFile(args.frames_root / rec / "webcam_c.zip")
            names = set(zf.namelist())
            for i in range(0, len(pog), args.stride):
                name = f"frame_{i}.jpg"
                if not valid[i] or name not in names:
                    continue
                bgr = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                hh, ww = bgr.shape[:2]
                res = labeller._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                if (not res.face_landmarks or not res.facial_transformation_matrixes
                        or any(j not in labeller._key_landmarks_px(res.face_landmarks[0], ww, hh) for j in REQUIRED_LANDMARKS)):
                    n_noface += 1
                    continue
                lms = res.face_landmarks[0]
                px = lambda j: np.array([lms[j].x * ww, lms[j].y * hh])
                _, _, t_real, r_cam = head_pose_from_mediapipe(res.facial_transformation_matrixes[0], (ww, hh), K, np.zeros(5))
                ours = eye_midpoint_cam(r_cam, t_real)
                target = (T @ np.array([pog[i, 0] * mpp[0], pog[i, 1] * mpp[1], 0.0, 1.0]))[:3]
                gt = (lo[i] + ro[i]) / 2
                iris = [float(np.linalg.norm(px(a) - px(b))) for a, b in IRIS_PAIRS.values()]
                w.writerow([rec, i, round(K[0, 0], 2), *np.round(gt, 2), *np.round(ours, 2), *np.round(target, 2),
                            *np.round(iris, 3)])
                n_ok += 1
            print(f"{rec}: {n_ok} frames so far ({(time.time() - start) / max(n_ok + n_noface, 1) * 1000:.0f} ms/frame)", flush=True)
    print(f"done: {n_ok} frames, {n_noface} without a usable face -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
