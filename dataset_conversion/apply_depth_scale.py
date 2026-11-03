"""Apply the per-user face scale to the released labels (gaze origin, gaze vector, head translation).

    python dataset_conversion/apply_depth_scale.py --release-root F:/recalib_release/ReCalib \
        --factors labelling/user_depth_scale.csv --canonical F:/recalib_release/recalib_index_canonical_face.csv

The first run copies recalib_index.csv (labels with MediaPipe's canonical face size) to --canonical,
outside the release. Every run starts from that copy, so running it again does not rescale twice.
For each labelled sample with factor s of its user:
    origin_mm      = s * canonical origin_mm
    translation_mm = s * canonical translation_mm
    vector         = unit(pog_mm - origin_mm)
Everything else (landmarks, rotation, target, QA metrics, discard info) is unchanged. The JSON files
and recalib_index.csv are rewritten in place.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labelling"))
from recalib_labelling.face_scale import load_factors  # noqa: E402


def r(values, digits):
    return [round(float(v), digits) + 0.0 for v in values]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--factors", required=True, type=Path)
    ap.add_argument("--canonical", required=True, type=Path, help="copy of the canonical-face index (outside the release)")
    args = ap.parse_args(argv)

    index = args.release_root / "recalib_index.csv"
    if not args.canonical.exists():
        shutil.copy2(index, args.canonical)
        print(f"canonical-face labels saved to {args.canonical}")
    factors = load_factors(args.factors)
    with open(args.canonical, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)

    n_scaled = 0
    start = time.time()
    for k, row in enumerate(rows, 1):
        if row["gaze_origin_mm_z"] in ("", None):
            continue
        s = factors[row["user_id"]]
        origin = np.array([float(row[f"gaze_origin_mm_{a}"]) for a in "xyz"]) * s
        pog = np.array([float(row[f"gaze_pog_mm_{a}"]) for a in "xyz"])
        vector = (pog - origin) / np.linalg.norm(pog - origin)
        trans = np.array([float(row[f"head_translation_mm_{a}"]) for a in "xyz"]) * s
        origin, vector, trans = r(origin, 4), r(vector, 6), r(trans, 4)
        for i, a in enumerate("xyz"):
            row[f"gaze_origin_mm_{a}"], row[f"gaze_vector_{a}"], row[f"head_translation_mm_{a}"] = origin[i], vector[i], trans[i]

        path = (args.release_root / row["image"]).with_suffix(".json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["gaze"]["origin_mm"], doc["gaze"]["vector"] = origin, vector
        doc["head_pose"]["translation_mm"] = trans
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=4)
        n_scaled += 1
        if k % 10000 == 0:
            print(f"{k}/{len(rows)} ({(time.time() - start) / k * 1000:.1f} ms/row)", flush=True)

    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)       # CRLF, as the release index
        w.writeheader()
        w.writerows(rows)
    print(f"rescaled {n_scaled} of {len(rows)} samples; JSON files and {index} updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
