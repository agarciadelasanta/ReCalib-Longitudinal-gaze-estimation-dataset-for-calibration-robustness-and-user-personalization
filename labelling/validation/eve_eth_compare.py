"""Compare the ETH-XGaze eye centre with the EVE ground truth, on the frames of eve_compare.py.

    python labelling/validation/eve_eth_compare.py --compare H:/EVE/relabel_eval/eve_compare_all.csv \
        --frames-root H:/EVE/Mamu_untagged --h5-root H:/EVE/Extracted_webcam_h5 \
        --out H:/EVE/relabel_eval/eve_eth_compare_all.csv [--workers 12]

The counterpart of eve_compare.py for the ETH-XGaze head pose (evaluation/eth_xGaze_inf.py): dlib
landmarks on the frame, the generic 6-point ETH face model and solvePnP with EVE's camera_matrix (no
distortion, as in eve_compare.py). Its eye centre (`two_eye_center`, the mean of the four eye-corner
model points) has the same definition as the ReCalib gaze origin and EVE's eye position (midpoint of
left_o and right_o). For every frame of --compare (EVE ground truth and target taken from it), the same
quantities are written: depth error along the camera ray to the eyes, lateral error, and the angle
between the gaze vectors from the ETH and EVE origins to the same target.
"""
import argparse
import csv
import sys
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

COLUMNS = ["recording", "frame", "gt_x", "gt_y", "gt_z", "eth_x", "eth_y", "eth_z", "target_x", "target_y", "target_z",
           "depth_err_mm", "depth_err_pct", "lateral_err_mm", "angle_err_deg"]


def unit(v):
    return v / np.linalg.norm(v)


def run_recording(job):
    rec, frames, frames_root, h5_root, shape_predictor = job
    import dlib
    from eth_xGaze_inf import dlib_eth_landmarks, estimateHeadPose, load_face_model
    cv2.setNumThreads(1)
    face_model = load_face_model(str(EVAL_DIR / "face_model.txt"))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    with h5py.File(h5_root / rec / "webcam_c.h5", "r") as h:
        K = h["camera_matrix"][()].astype(np.float64)
    zpath = frames_root / rec / "webcam_c.zip"
    if zpath.exists():
        zf = zipfile.ZipFile(zpath)
        read = lambda i: cv2.imdecode(np.frombuffer(zf.read(f"frame_{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    else:
        read = lambda i: cv2.imread(str(frames_root / rec / "webcam_c" / f"frame_{i}.jpg"))
    out, n_noface = [], 0
    for r in frames:
        lm = dlib_eth_landmarks(detector, predictor, read(int(r["frame"])))
        if lm is None:
            n_noface += 1
            continue
        hr, ht = estimateHeadPose(lm, face_model.reshape(6, 1, 3), K, np.zeros(5))
        fc = cv2.Rodrigues(hr)[0] @ face_model.T + ht.reshape(3, 1)
        eth = fc[:, 0:4].mean(axis=1)
        gt = np.array([float(r[f"gt_{a}"]) for a in "xyz"])
        target = np.array([float(r[f"target_{a}"]) for a in "xyz"])
        d = eth - gt
        ray = unit(gt)
        depth = float(d @ ray)
        angle = float(np.degrees(np.arccos(np.clip(unit(target - eth) @ unit(target - gt), -1, 1))))
        out.append([rec, r["frame"], *np.round(gt, 2), *np.round(eth, 2), *np.round(target, 2), round(depth, 2),
                    round(100 * depth / np.linalg.norm(gt), 3), round(float(np.linalg.norm(d - depth * ray)), 2), round(angle, 4)])
    return rec, out, n_noface


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", required=True, type=Path, help="eve_compare.py output (frames, ground truth, targets)")
    ap.add_argument("--frames-root", required=True, type=Path)
    ap.add_argument("--h5-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shape-predictor", help="dlib 68-point model (default: evaluation/dlib_model.py, downloaded on first use)")
    args = ap.parse_args()
    from dlib_model import ensure_shape_predictor
    shape_predictor = ensure_shape_predictor(args.shape_predictor)

    by_rec = defaultdict(list)
    with open(args.compare, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_rec[r["recording"]].append(r)
    jobs = [(rec, rows, args.frames_root, args.h5_root, shape_predictor) for rec, rows in sorted(by_rec.items())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_noface = 0
    start = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as fh, Pool(args.workers) as pool:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for k, (rec, rows, noface) in enumerate(pool.imap_unordered(run_recording, jobs), 1):
            w.writerows(rows)
            fh.flush()
            n_ok += len(rows)
            n_noface += noface
            print(f"{k}/{len(jobs)} recordings, {n_ok} frames compared, {n_noface} without a dlib face, "
                  f"{time.time() - start:.0f} s", flush=True)
    print(f"done: {n_ok} frames, {n_noface} without a dlib face -> {args.out}")


if __name__ == "__main__":
    main()
