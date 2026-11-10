"""Label many images listed in a CSV.

    python -m recalib_reconstruction.label_batch --csv targets.csv --images-root DIR --setup setup.json --out-dir OUT

The CSV needs the columns image (path relative to --images-root), pog_px_x and pog_px_y;
optional columns user_id, session_id, task_id, sample_id and task_type are copied to the JSON.
The release index (recalib_index.csv) has all of them, so the whole dataset can be relabelled
with --csv recalib_index.csv --images-root <release root>.

One JSON per image is written to --out-dir, mirroring the relative image path.

--depth-scale user_depth_scale.csv applies each user's face scale (see face_scale.py); the ReCalib
factors are in 3d_reconstruction/user_depth_scale.csv. Without it the canonical face size is used (factor 1).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from .face_scale import load_factors
from .labeller import Labeller

ID_COLUMNS = ("user_id", "session_id", "task_id", "sample_id", "task_type")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--setup", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", help="face_landmarker.task to use instead of the cached download")
    ap.add_argument("--limit", type=int, help="label only the first N rows")
    ap.add_argument("--depth-scale", type=Path, help="CSV with user_id,depth_scale (default: factor 1 for all)")
    args = ap.parse_args(argv)

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    factors = load_factors(args.depth_scale) if args.depth_scale else {}
    labelled = unlabelled = 0
    start = time.time()
    with Labeller(args.setup, model_path=args.model) as labeller:
        for n, row in enumerate(rows, 1):
            rel = Path(row["image"])
            ids = {k: row[k].zfill(4 if k == "sample_id" else 2) if k != "task_type" else row[k]
                   for k in ID_COLUMNS if row.get(k)}
            doc = labeller.label(args.images_root / rel, (float(row["pog_px_x"]), float(row["pog_px_y"])),
                                 image_name=rel.name, ids=ids, depth_scale=factors.get(ids.get("user_id"), 1.0))
            out = (args.out_dir / rel).with_suffix(".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=4), encoding="utf-8")
            if doc["gaze"] is None:
                unlabelled += 1
            else:
                labelled += 1
            if n % 500 == 0:
                print(f"{n}/{len(rows)} images ({(time.time() - start) / n * 1000:.0f} ms/image)")
    print(f"done: {labelled} labelled, {unlabelled} without a usable face, JSON in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
