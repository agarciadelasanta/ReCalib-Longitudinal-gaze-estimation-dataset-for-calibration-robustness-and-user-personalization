"""Add the ETH-XGaze gaze origin and gaze vector to every released sample (field gaze_eth_xgaze).

    python dataset_conversion/add_eth_xgaze_origin.py --release-root ReCalib \
        --eth-csv temp/eth_origin_check/eth_origin_check_all.csv --backup recalib_index_before_eth.csv

--eth-csv is the per-frame output of evaluation/eth_origin_check.py (--n 0 --discard-codes 0,10): the
official ETH-XGaze preprocessing (dlib landmarks, generic 6-point face model, solvePnP with the
ReCalib intrinsics). Its gaze origin is the face centre, the origin ETH-XGaze uses for its own gaze
vectors (halfway between the eye centre and the nose). For each sample:
    gaze_eth_xgaze.origin_mm = ETH face centre (camera frame, mm)
    gaze_eth_xgaze.vector    = unit(gaze.pog_mm - origin_mm)
gaze_eth_xgaze is null when the sample has no ReCalib gaze (discard code 2), when dlib finds no face,
or when the face centre is more than --max-depth mm from the camera (false detection, as in the
baseline). The field is inserted after "gaze" in each JSON; the index gets the columns
gaze_eth_xgaze_vector_* and gaze_eth_xgaze_origin_mm_* after gaze_pog_mm_z (empty when null).
The first run copies recalib_index.csv to --backup, outside the release; running it again rewrites
the same values.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

NEW_COLS = [f"gaze_eth_xgaze_vector_{a}" for a in "xyz"] + [f"gaze_eth_xgaze_origin_mm_{a}" for a in "xyz"]


def r(values, digits):
    return [round(float(v), digits) + 0.0 for v in values]


def load_eth(path, max_depth):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["ok"] != "1":
                continue
            c = [float(row[f"eth_face_center_{a}"]) for a in "xyz"]
            if c[2] <= max_depth:
                out[row["image"].replace("\\", "/")] = np.array(c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--eth-csv", required=True, type=Path)
    ap.add_argument("--backup", required=True, type=Path, help="copy of recalib_index.csv before the change (outside the release)")
    ap.add_argument("--max-depth", type=float, default=900.0)
    args = ap.parse_args(argv)

    index = args.release_root / "recalib_index.csv"
    if not args.backup.exists():
        shutil.copy2(index, args.backup)
        print(f"index saved to {args.backup}")
    eth = load_eth(args.eth_csv, args.max_depth)
    with open(index, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames), list(reader)
    if NEW_COLS[0] not in fields:
        at = fields.index("gaze_pog_mm_z") + 1
        fields[at:at] = NEW_COLS

    n_set = 0
    start = time.time()
    for k, row in enumerate(rows, 1):
        block = None
        origin = eth.get(row["image"].replace("\\", "/"))
        if origin is not None and row["gaze_origin_mm_z"] not in ("", None):
            pog = np.array([float(row[f"gaze_pog_mm_{a}"]) for a in "xyz"])
            vector = (pog - origin) / np.linalg.norm(pog - origin)
            block = {"vector": r(vector, 6), "origin_mm": r(origin, 4)}
            n_set += 1
        for i, a in enumerate("xyz"):
            row[f"gaze_eth_xgaze_vector_{a}"] = block["vector"][i] if block else ""
            row[f"gaze_eth_xgaze_origin_mm_{a}"] = block["origin_mm"][i] if block else ""

        path = (args.release_root / row["image"]).with_suffix(".json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        new = {}
        for key, value in doc.items():
            if key == "gaze_eth_xgaze":
                continue
            new[key] = value
            if key == "gaze":
                new["gaze_eth_xgaze"] = block
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(new, fh, indent=4)
        if k % 10000 == 0:
            print(f"{k}/{len(rows)} ({(time.time() - start) / k * 1000:.1f} ms/row)", flush=True)

    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)       # CRLF, as the release index
        w.writeheader()
        w.writerows(rows)
    print(f"gaze_eth_xgaze set for {n_set} of {len(rows)} samples (null for the rest); JSON files and {index} updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
