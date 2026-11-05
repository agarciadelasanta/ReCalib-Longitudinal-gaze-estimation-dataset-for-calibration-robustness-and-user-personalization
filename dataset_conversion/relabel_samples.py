"""Relabel released samples in place with labelling/, keeping their quality flags.

    python dataset_conversion/relabel_samples.py --release-root ReCalib --codes 10 \
        --backup backup_relabel_code10 [--write]

For every sample of recalib_index.csv whose discard code is in --codes, the gaze, head_pose and
quality_assurance_metrics blocks of its JSON are regenerated with labelling/recalib_labelling (with the
user's face-scale factor from labelling/user_depth_scale.csv), as for the rest of the release.
discard_info, identifiers and target are kept. The matching recalib_index.csv rows are updated.

Used for the 249 low-EAR samples (code 10), whose QA block came from an earlier landmark run (110 of
them had none). Without --write the differences are only reported. With --write the original JSON
files and recalib_index.csv are first copied to --backup.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labelling"))
from recalib_labelling import Labeller  # noqa: E402
from recalib_labelling.face_scale import RECALIB_FACTORS, load_factors  # noqa: E402

BLOCKS = ("gaze", "head_pose", "quality_assurance_metrics")


def index_values(doc):
    """recalib_index.csv columns that come from the labels."""
    g, h, q = doc["gaze"], doc["head_pose"], doc["quality_assurance_metrics"]
    out = {}
    for prefix, values in (("gaze_vector", g["vector"]), ("gaze_origin_mm", g["origin_mm"]), ("gaze_pog_mm", g["pog_mm"]),
                           ("head_rotation_rad", h["rotation_rad"]), ("head_rotvec_rad", h["rotvec_rad"]),
                           ("head_translation_mm", h["translation_mm"])):
        out.update({f"{prefix}_{a}": v for a, v in zip("xyz", values)})
    s, e = q["landmark_statistics_2d"], q["eye_aspect_ratio"]
    out.update({f"qa_{k}": s[k] for k in ("max_x", "min_x", "max_y", "min_y", "mean_x", "mean_y", "head_bounding_box_surface")})
    out.update(qa_ear_left=e["left"], qa_ear_right=e["right"], qa_ear_mean=e["mean"])
    for k, (x, y) in h["mediapipe_face_mesh_2d"].items():
        out[f"lm_{k}_x"], out[f"lm_{k}_y"] = x, y
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--codes", default="10", help="discard codes of the samples to relabel")
    ap.add_argument("--backup", required=True, type=Path, help="folder for the original JSON files and index (outside the release)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    root = args.release_root
    index = root / "recalib_index.csv"
    with open(index, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    selected = [r for r in rows if r["discard_code"] in args.codes.split(",")]
    factors = load_factors(RECALIB_FACTORS)

    if args.write:
        args.backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(index, args.backup / "recalib_index.csv")

    origin_d, ear_old, ear_new, no_face, changed_lm = [], [], [], [], 0
    with Labeller(root / "setup.json") as labeller:
        for row in selected:
            path = (root / row["image"]).with_suffix(".json")
            doc = json.loads(path.read_text(encoding="utf-8"))
            new = labeller.label(root / row["image"], (float(row["pog_px_x"]), float(row["pog_px_y"])),
                                 depth_scale=factors[row["user_id"]])
            if new["gaze"] is None:
                no_face.append(row["image"])
                continue
            if doc.get("gaze"):
                origin_d.append(float(np.linalg.norm(np.subtract(doc["gaze"]["origin_mm"], new["gaze"]["origin_mm"]))))
            if doc["head_pose"]["mediapipe_face_mesh_2d"] != new["head_pose"]["mediapipe_face_mesh_2d"]:
                changed_lm += 1
            ear_old.append(None if doc["quality_assurance_metrics"] is None else doc["quality_assurance_metrics"]["eye_aspect_ratio"]["mean"])
            ear_new.append(new["quality_assurance_metrics"]["eye_aspect_ratio"]["mean"])
            if args.write:
                dest = args.backup / row["image"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest.with_suffix(".json"))
                for k in BLOCKS:
                    doc[k] = new[k]
                with open(path, "w", encoding="utf-8") as fh:      # CRLF and indent 4, as the release
                    json.dump(doc, fh, indent=4)
                row.update(index_values(doc))

    if args.write:
        with open(index, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    n = len(selected) - len(no_face)
    print(f"{len(selected)} samples with codes {args.codes}; relabelled {n}; no face found in {len(no_face)} {no_face[:3]}")
    print(f"landmarks changed in {changed_lm}")
    if origin_d:
        print(f"gaze origin change: median {np.median(origin_d):.4f} mm, max {max(origin_d):.4f} mm")
    had = [(o, e) for o, e in zip(ear_old, ear_new) if o is not None]
    print(f"QA block missing before: {sum(o is None for o in ear_old)}; mean EAR now {min(ear_new):.4f}-{max(ear_new):.4f}, "
          f"below 0.05: {sum(e < 0.05 for e in ear_new)}/{len(ear_new)}; stored EAR before (where present) "
          f"{min(o for o, _ in had):.4f}-{max(o for o, _ in had):.4f}")
    print("written (originals in %s)" % args.backup if args.write else "dry run, nothing written")


if __name__ == "__main__":
    main()
