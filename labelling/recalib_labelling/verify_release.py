"""Relabel release samples and compare with the released JSON files.

    python -m recalib_labelling.verify_release --release-root <ReCalib> [--n 200] [--seed 0]

Picks random samples from recalib_index.csv, labels each image with this package and reports
the largest differences per field (angles in degrees, positions in mm, landmarks in pixels).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

from .labeller import Labeller


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model")
    args = ap.parse_args(argv)

    root = args.release_root
    with open(root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["discard_code"] in ("0", "10")]   # samples with labels
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    diffs = {k: [] for k in ("gaze.vector (deg)", "gaze.origin_mm (mm)", "gaze.pog_mm (mm)",
                             "head_pose.translation_mm (mm)", "head_pose.rotvec (deg)",
                             "head_pose.rotation_rad (max abs, rad)", "landmarks (max abs, px)",
                             "QA metrics (max abs)")}
    with Labeller(root / "setup.json", model_path=args.model) as labeller:
        for r in rows:
            ref = json.loads((root / r["image"]).with_suffix(".json").read_text(encoding="utf-8"))
            new = labeller.label(root / r["image"], (float(r["pog_px_x"]), float(r["pog_px_y"])))
            if new["gaze"] is None:
                print(f"no face found in {r['image']} (released with labels)")
                continue
            g0, g1, h0, h1 = ref["gaze"], new["gaze"], ref["head_pose"], new["head_pose"]
            diffs["gaze.vector (deg)"].append(angle_deg(g0["vector"], g1["vector"]))
            diffs["gaze.origin_mm (mm)"].append(float(np.linalg.norm(np.subtract(g0["origin_mm"], g1["origin_mm"]))))
            diffs["gaze.pog_mm (mm)"].append(float(np.linalg.norm(np.subtract(g0["pog_mm"], g1["pog_mm"]))))
            diffs["head_pose.translation_mm (mm)"].append(float(np.linalg.norm(np.subtract(h0["translation_mm"], h1["translation_mm"]))))
            diffs["head_pose.rotvec (deg)"].append(float(np.degrees(np.linalg.norm(np.subtract(h0["rotvec_rad"], h1["rotvec_rad"])))))
            diffs["head_pose.rotation_rad (max abs, rad)"].append(float(np.abs(np.subtract(h0["rotation_rad"], h1["rotation_rad"])).max()))
            m0, m1 = h0["mediapipe_face_mesh_2d"], h1["mediapipe_face_mesh_2d"]
            diffs["landmarks (max abs, px)"].append(max(abs(a - b) for k in m0 if k in m1 for a, b in zip(m0[k], m1[k])))
            q0, q1 = ref["quality_assurance_metrics"], new["quality_assurance_metrics"]
            qd = [abs(q0["landmark_statistics_2d"][k] - q1["landmark_statistics_2d"][k]) for k in q0["landmark_statistics_2d"]]
            qd += [abs(q0["eye_aspect_ratio"][k] - q1["eye_aspect_ratio"][k]) for k in q0["eye_aspect_ratio"]]
            diffs["QA metrics (max abs)"].append(max(qd))

    n = len(diffs["gaze.vector (deg)"])
    print(f"\ncompared {n} released samples")
    print(f"{'field':40s} {'median':>10s} {'p99':>10s} {'max':>10s}")
    for k, v in diffs.items():
        v = np.array(v)
        print(f"{k:40s} {np.median(v):10.4g} {np.percentile(v, 99):10.4g} {v.max():10.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
