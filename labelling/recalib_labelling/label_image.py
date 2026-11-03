"""Label one image.

    python -m recalib_labelling.label_image IMAGE --setup setup.json --pog-px X Y [--out label.json]
        [--user-id 00 --session-id 00 --task-id 01 --sample-id 0001 --task-type test] [--depth-scale F]

With --user-id of a ReCalib user, the user's face scale from labelling/user_depth_scale.csv is used
unless --depth-scale is given (1 = canonical face, single-image labelling).

Prints the JSON to stdout unless --out is given.
"""
import argparse
import json
import sys

from .face_scale import RECALIB_FACTORS, load_factors
from .labeller import Labeller


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--setup", required=True, help="setup.json of the dataset (screen, camera calibration)")
    ap.add_argument("--pog-px", required=True, nargs=2, type=float, metavar=("X", "Y"),
                    help="screen target in physical screen pixels (origin top-left)")
    ap.add_argument("--out", help="output JSON file (default: stdout)")
    ap.add_argument("--model", help="face_landmarker.task to use instead of the cached download")
    for key in ("user-id", "session-id", "task-id", "sample-id", "task-type"):
        ap.add_argument(f"--{key}")
    ap.add_argument("--depth-scale", type=float, help="per-user face scale (default: ReCalib factor of --user-id, else 1)")
    args = ap.parse_args(argv)

    ids = {k: getattr(args, k) for k in ("user_id", "session_id", "task_id", "sample_id", "task_type")
           if getattr(args, k) is not None}
    with Labeller(args.setup, model_path=args.model) as labeller:
        scale = args.depth_scale if args.depth_scale is not None else load_factors(RECALIB_FACTORS).get(args.user_id, 1.0)
        doc = labeller.label(args.image, args.pog_px, ids=ids, depth_scale=scale)
    text = json.dumps(doc, indent=4)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
