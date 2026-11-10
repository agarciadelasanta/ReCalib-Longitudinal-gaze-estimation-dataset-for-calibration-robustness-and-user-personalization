"""Per-user face scale from the iris size (labelling with several images of the same user).

A single image is labelled with MediaPipe's canonical face, whose size is fixed, so the depth of the
head and gaze origin is off by a per-user factor: larger faces are placed too close, smaller ones too
far. With several images of the same user this factor can be estimated, because the visible iris
diameter varies little between adults (about 11.7 mm):

    factor = median over the user's images of  (f * 11.7 mm / iris diameter in px) / origin depth

computed on labels made with factor 1. Labelling with Labeller.label(..., depth_scale=factor) then
places the head and gaze origin at the corrected depth (a face s times larger at s times the
distance projects identically, so landmarks and rotation are unchanged).

    python -m recalib_reconstruction.face_scale --labels labels.csv --setup setup.json --out user_depth_scale.csv

--labels is a CSV in the recalib_index.csv format, or a folder of JSON labels, made with factor 1.
The factors used for ReCalib are in 3d_reconstruction/user_depth_scale.csv.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional

import cv2
import numpy as np

from .labeller import load_setup

IRIS_DIAMETER_MM = 11.7
IRIS_PAIRS = ((469, 471), (470, 472), (474, 476), (475, 477))   # opposite points of both iris rings
RECALIB_FACTORS = Path(__file__).resolve().parents[1] / "user_depth_scale.csv"


def iris_diameter_px(landmarks: Mapping, camera_matrix=None, dist_coeffs=None) -> Optional[float]:
    """Mean iris diameter in pixels from the 8 iris ring landmarks (keys int or str, values (x, y)).

    With camera_matrix and dist_coeffs the points are undistorted first. None if a landmark is missing.
    """
    try:
        pts = {i: np.asarray(landmarks.get(i, landmarks.get(str(i))), dtype=np.float64)
               for pair in IRIS_PAIRS for i in pair}
    except (TypeError, ValueError):
        return None
    if any(p.shape != (2,) for p in pts.values()):
        return None
    if camera_matrix is not None:
        K = np.asarray(camera_matrix, dtype=np.float64)
        ids = list(pts)
        und = cv2.undistortPoints(np.array([pts[i] for i in ids]).reshape(-1, 1, 2), K,
                                  np.asarray(dist_coeffs, dtype=np.float64), P=K).reshape(-1, 2)
        pts = dict(zip(ids, und))
    return float(np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in IRIS_PAIRS]))


def iris_depth_mm(diameter_px: float, camera_matrix, iris_mm: float = IRIS_DIAMETER_MM) -> float:
    """Depth (mm) at which an iris of iris_mm appears diameter_px wide."""
    K = np.asarray(camera_matrix, dtype=np.float64)
    return float((K[0, 0] + K[1, 1]) / 2 * iris_mm / diameter_px)


def user_factors(records: Iterable[dict], camera_matrix, dist_coeffs) -> dict:
    """Per-user factor from labels made with factor 1.

    records: dicts with user_id, origin_z (mm), landmarks ({id: (x, y)}) and kept (bool).
    Returns {user_id: (factor, n_images, iris_px_median)}.
    """
    ratios, diam = defaultdict(list), defaultdict(list)
    for r in records:
        if not r["kept"]:
            continue
        d_px = iris_diameter_px(r["landmarks"], camera_matrix, dist_coeffs)
        if d_px is None:
            continue
        diam[r["user_id"]].append(d_px)
        ratios[r["user_id"]].append(iris_depth_mm(d_px, camera_matrix) / r["origin_z"])
    return {u: (float(np.median(v)), len(v), float(np.median(diam[u]))) for u, v in ratios.items()}


def load_factors(path=RECALIB_FACTORS) -> dict:
    """{user_id: factor} from a user_depth_scale.csv file."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["user_id"]: float(r["depth_scale"]) for r in csv.DictReader(fh)}


def _records_from_index(path):
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["gaze_origin_mm_z"] in ("", None):
                continue
            yield {"user_id": r["user_id"], "origin_z": float(r["gaze_origin_mm_z"]),
                   "kept": r["discard_code"] == "0",
                   "landmarks": {i: (float(r[f"lm_{i}_x"]), float(r[f"lm_{i}_y"])) for pair in IRIS_PAIRS
                                 for i in pair if r.get(f"lm_{i}_x") not in ("", None)}}


def _records_from_jsons(folder):
    for p in sorted(Path(folder).rglob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not doc.get("gaze") or "user_id" not in doc:
            continue
        yield {"user_id": doc["user_id"], "origin_z": doc["gaze"]["origin_mm"][2],
               "kept": not (doc.get("discard_info") or {}).get("discarded", False),
               "landmarks": doc["head_pose"]["mediapipe_face_mesh_2d"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path, help="index CSV or folder of JSON labels (factor 1)")
    ap.add_argument("--setup", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    cam = load_setup(args.setup)["camera"]
    records = _records_from_jsons(args.labels) if args.labels.is_dir() else _records_from_index(args.labels)
    factors = user_factors(records, cam["camera_matrix"], cam["dist_coeffs_k1_k2_p1_p2_k3"])
    if factors and all(abs(f - 1) < 0.002 for f, _, _ in factors.values()):
        print("warning: every factor is 1. The labels already include a face scale (e.g. the released "
              "recalib_index.csv); --labels needs labels made with the one-image method (factor 1)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "depth_scale", "n_images", "iris_px_median"])
        for u in sorted(factors):
            f, n, d = factors[u]
            w.writerow([u, round(f, 5), n, round(d, 3)])
    print(f"{len(factors)} users -> {args.out}: " + ", ".join(f"{u} {factors[u][0]:.4f}" for u in sorted(factors)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
