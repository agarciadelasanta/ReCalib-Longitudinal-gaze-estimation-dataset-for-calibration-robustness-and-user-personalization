"""Compare the ReCalib labelling with the EVE ground truth (webcam_c).

    python -m validation.eve_compare --frames-root H:/EVE/Mamu_unzipped --h5-root H:/EVE/Extracted_webcam_h5 \
        --out H:/EVE/relabel_eval/eve_compare.csv [--stride 10] [--recordings train01/step040_...]

For each frame frame_N.jpg (row N of webcam_c.h5) with valid EVE eye positions and gaze target:
  - the target is EVE's face_PoG_tobii converted to camera mm with camera_transformation;
  - the ground-truth gaze origin is the midpoint of EVE's left_o and right_o;
  - our gaze origin comes from the ReCalib labelling (MediaPipe pose re-expressed with EVE's camera_matrix);
  - both gaze vectors point from their origin to the same target, so the angle between them only
    reflects the origin difference.
EVE's webcam_c.h5 gives no distortion coefficients, so none are applied.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import h5py
import mediapipe as mp
import numpy as np

from recalib_labelling.geometry import eye_midpoint_cam, head_pose_from_mediapipe
from recalib_labelling.labeller import REQUIRED_LANDMARKS, Labeller

COLUMNS = ["recording", "frame", "fx", "fy",
           "gt_x", "gt_y", "gt_z", "ours_x", "ours_y", "ours_z", "target_x", "target_y", "target_z",
           "origin_err_mm", "depth_err_mm", "lateral_err_mm", "angle_err_deg", "angle_vs_eve_face_g_deg"]


def unit(v):
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def our_origin(labeller, bgr, K):
    """Gaze origin (eye midpoint, camera mm) from the ReCalib labelling, or None without a usable face."""
    h, w = bgr.shape[:2]
    res = labeller._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks or not res.facial_transformation_matrixes:
        return None
    if any(i not in labeller._key_landmarks_px(res.face_landmarks[0], w, h) for i in REQUIRED_LANDMARKS):
        return None
    _, _, t_real, r_cam = head_pose_from_mediapipe(res.facial_transformation_matrixes[0], (w, h), K, np.zeros(5))
    return eye_midpoint_cam(r_cam, t_real)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-root", required=True, type=Path)
    ap.add_argument("--h5-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=10, help="use every Nth frame")
    ap.add_argument("--recordings", nargs="*", help="e.g. train01/step040_wikipedia_wikipedia-random (default: all)")
    ap.add_argument("--model")
    args = ap.parse_args(argv)

    recs = args.recordings or sorted(str(p.parent.relative_to(args.frames_root).as_posix())
                                     for p in args.frames_root.glob("*/*/webcam_c"))
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
                pog, lo, ro, fo, g, Rf = (h[k + "/data"][()].astype(np.float64) for k in
                                          ("face_PoG_tobii", "left_o", "right_o", "face_o", "face_g_tobii", "face_R"))
                valid = (h["face_PoG_tobii/validity"][()] & h["left_o/validity"][()] & h["right_o/validity"][()]
                         & h["face_o/validity"][()] & h["face_g_tobii/validity"][()])
            for i in range(0, len(pog), args.stride):
                img = args.frames_root / rec / "webcam_c" / f"frame_{i}.jpg"
                if not valid[i] or not img.exists():
                    continue
                bgr = cv2.imread(str(img))
                ours = our_origin(labeller, bgr, K)
                if ours is None:
                    n_noface += 1
                    continue
                target = (T @ np.array([pog[i, 0] * mpp[0], pog[i, 1] * mpp[1], 0.0, 1.0]))[:3]
                gt = (lo[i] + ro[i]) / 2
                pitch, yaw = g[i]
                eve_g = Rf[i].T @ -np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
                d = ours - gt
                ray = unit(gt)                                  # depth measured along the camera ray to the eyes
                depth = float(d @ ray)
                w.writerow([rec, i, round(K[0, 0], 2), round(K[1, 1], 2), *np.round(gt, 2), *np.round(ours, 2),
                            *np.round(target, 2), round(float(np.linalg.norm(d)), 2), round(depth, 2),
                            round(float(np.linalg.norm(d - depth * ray)), 2),
                            round(angle_deg(target - ours, target - gt), 4),
                            round(angle_deg(target - ours, eve_g), 4)])
                n_ok += 1
            print(f"{rec}: {n_ok} frames compared so far, {n_noface} without a face "
                  f"({(time.time() - start) / max(n_ok + n_noface, 1) * 1000:.0f} ms/frame)", flush=True)
    print(f"done: {n_ok} frames, {n_noface} without a usable face -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
