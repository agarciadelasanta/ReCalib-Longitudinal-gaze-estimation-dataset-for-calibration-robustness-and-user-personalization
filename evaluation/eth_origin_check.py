"""Compare the ReCalib gaze origin with an independent estimate from the ETH-XGaze pipeline.

The ReCalib gaze origin (gaze.origin_mm) is the midpoint of the four eye corners of MediaPipe's face
mesh, placed in the calibrated camera. The ETH-XGaze pipeline places the same point (the mean of its
four eye-corner model points, `two_eye_center`) independently: dlib landmarks on the image, a generic
6-point face model and solvePnP with the camera intrinsics. Its face centre, used as gaze origin by
ETH-XGaze, lies halfway between the eyes and the nose, so the eye centre is compared here instead.

    py evaluation/eth_origin_check.py <release root> --out eth_origin_check.csv [--n 5000] [--workers 8]

Samples random frames kept after quality filtering, writes one row per frame and prints the
differences ETH - ReCalib (mm, and depth in %), overall and per user.
"""
import argparse
import csv
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from eth_xGaze_inf import dlib_eth_landmarks, estimateHeadPose, load_camera_npz, load_face_model

EVAL_DIR = Path(__file__).resolve().parent
_STATE = {}


def _init(face_model_path, shape_predictor, camera_npz):
    import dlib
    cv2.setNumThreads(1)
    camera, distortion = load_camera_npz(camera_npz)
    _STATE.update(face_model=load_face_model(str(face_model_path)), camera=camera, distortion=distortion,
                  detector=dlib.get_frontal_face_detector(), predictor=dlib.shape_predictor(str(shape_predictor)))


def eth_points(image):
    """ETH eye centre and face centre (camera frame, mm), or None if dlib finds no face."""
    s = _STATE
    lm = dlib_eth_landmarks(s["detector"], s["predictor"], image)
    if lm is None:
        return None
    hr, ht = estimateHeadPose(lm, s["face_model"].reshape(6, 1, 3), s["camera"], s["distortion"])
    fc = cv2.Rodrigues(hr)[0] @ s["face_model"].T + ht.reshape(3, 1)
    eye, nose = fc[:, 0:4].mean(axis=1), fc[:, 4:6].mean(axis=1)
    return eye, (eye + nose) / 2


def process(args):
    root, row = args
    out = dict(image=row["image"], user=row["user_id"], session=row["session_id"], task=row["task_id"], ok=0)
    image = cv2.imread(str(root / row["image"]))
    pts = eth_points(image) if image is not None else None
    if pts is None:
        return out
    ours = json.loads((root / row["image"]).with_suffix(".json").read_text(encoding="utf-8"))["gaze"]["origin_mm"]
    out.update(ok=1, **{f"ours_{a}": v for a, v in zip("xyz", ours)},
               **{f"eth_eye_{a}": float(v) for a, v in zip("xyz", pts[0])},
               **{f"eth_face_center_{a}": float(v) for a, v in zip("xyz", pts[1])})
    return out


def summarize(rows):
    ours = np.array([[r[f"ours_{a}"] for a in "xyz"] for r in rows])
    eth = np.array([[r[f"eth_eye_{a}"] for a in "xyz"] for r in rows])
    d = eth - ours
    depth_pct = 100 * d[:, 2] / ours[:, 2]
    q = lambda v: f"{np.median(v):+7.1f} [{np.percentile(v, 5):+7.1f}, {np.percentile(v, 95):+7.1f}]"
    return f"x {q(d[:, 0])}  y {q(d[:, 1])}  z {q(d[:, 2])} mm  depth {q(depth_pct)} %  n={len(rows)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--face-model", type=Path, default=EVAL_DIR / "face_model.txt")
    ap.add_argument("--shape-predictor", type=Path, default=EVAL_DIR / "modules" / "shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--camera-npz", type=Path, default=EVAL_DIR.parent / "docs" / "camera_intrinsics.npz")
    args = ap.parse_args()

    with open(args.release_root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["discard_code"] == "0"]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    columns = ["image", "user", "session", "task", "ok"] + [f"{p}_{a}" for p in ("ours", "eth_eye", "eth_face_center") for a in "xyz"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results, start = [], time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as fh, \
            Pool(args.workers, _init, (args.face_model, args.shape_predictor, args.camera_npz)) as pool:
        w = csv.DictWriter(fh, columns, restval="")
        w.writeheader()
        # rows are written as they finish, so a stopped run keeps what it has done
        for n, res in enumerate(pool.imap_unordered(process, [(args.release_root, r) for r in rows], chunksize=4), 1):
            results.append(res)
            w.writerow(res)
            if n % 100 == 0 or n == len(rows):
                fh.flush()
                print(f"{n}/{len(rows)} frames, {time.time() - start:.0f} s", flush=True)

    ok = [r for r in results if r["ok"]]
    print(f"{len(ok)}/{len(results)} frames with a dlib face; ETH eye centre - ReCalib origin, median [p5, p95]:")
    print("all  ", summarize(ok))
    for user in sorted({r["user"] for r in ok}):
        print(f"user {user}", summarize([r for r in ok if r["user"] == user]))


if __name__ == "__main__":
    main()
