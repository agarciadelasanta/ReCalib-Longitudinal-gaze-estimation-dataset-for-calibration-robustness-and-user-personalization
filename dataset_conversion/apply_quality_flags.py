"""Set the quality flags (discard_info) of labelled samples, as in the released dataset.

    python dataset_conversion/apply_quality_flags.py --labels relabelled --setup ReCalib/setup.json --write
    python dataset_conversion/apply_quality_flags.py --labels relabelled --setup ReCalib/setup.json \
        --compare-index ReCalib/recalib_index.csv

--labels is a folder of JSON annotations in the release format, e.g. the output of
3d_reconstruction/recalib_reconstruction/label_batch.py. The rules, applied in this order (first match wins):

  2  HPE_MODEL_ERROR        no face or no usable landmarks (gaze/head_pose are null; set by the labelling)
  5  LANDMARKS_OUT_OF_RANGE any of the 12 eye and iris landmarks (33, 133, 263, 362, 469-472, 474-477)
                            within 4% of the left/right or 5% of the top/bottom image border, fewer than
                            12 of them available, or their bounding box covering < 0.1% or > 80% of the image
 10  LOW_EYE_ASPECT_RATIO   mean eye aspect ratio of both eyes below 0.05 (closed eyes)

Samples matching none of them get discard_info = null. Without --write nothing is changed; the counts
are printed and, with --compare-index, compared with the discard_code column of an index CSV
(rows matched by image name). --report writes one row per sample (image, code, mean EAR).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

EYE_IRIS_LANDMARKS = ("33", "133", "263", "362", "469", "470", "471", "472", "474", "475", "476", "477")
MARGIN_X, MARGIN_Y = 0.04, 0.05          # fraction of the image width / height
SURFACE_MIN, SURFACE_MAX = 0.001, 0.8     # fraction of the image area
EAR_MIN = 0.05

NAMES = {2: "HPE_MODEL_ERROR", 5: "LANDMARKS_OUT_OF_RANGE", 10: "LOW_EYE_ASPECT_RATIO"}


def landmarks_out_of_range(landmarks: dict, width: int, height: int) -> bool:
    pts = [landmarks[k] for k in EYE_IRIS_LANDMARKS if k in landmarks]
    if len(pts) < len(EYE_IRIS_LANDMARKS):
        return True
    xs = [p[0] / width for p in pts]
    ys = [p[1] / height for p in pts]
    if min(xs) < MARGIN_X or max(xs) > 1 - MARGIN_X or min(ys) < MARGIN_Y or max(ys) > 1 - MARGIN_Y:
        return True
    surface = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return not SURFACE_MIN <= surface <= SURFACE_MAX


def mean_ear(doc: dict):
    qa = doc.get("quality_assurance_metrics") or {}
    return (qa.get("eye_aspect_ratio") or {}).get("mean")


def quality_code(doc: dict, width: int, height: int):
    """Discard code of one annotation (0 = kept), or None if its quality metrics are missing."""
    if doc.get("gaze") is None or doc.get("head_pose") is None:
        return 2
    if landmarks_out_of_range(doc["head_pose"].get("mediapipe_face_mesh_2d") or {}, width, height):
        return 5
    ear = mean_ear(doc)
    if ear is None:
        return None
    return 10 if ear < EAR_MIN else 0


def discard_info(code: int):
    if code == 0:
        return None
    return {"discarded": True, "discarding_reason": code, "discarding_reason_name": NAMES[code]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path, help="folder of JSON annotations (searched recursively)")
    ap.add_argument("--setup", required=True, type=Path, help="setup.json of the release (image size)")
    ap.add_argument("--write", action="store_true", help="write discard_info into the JSON files")
    ap.add_argument("--compare-index", type=Path, help="index CSV with image and discard_code columns")
    ap.add_argument("--report", type=Path, help="CSV with one row per sample")
    args = ap.parse_args(argv)

    res = json.loads(args.setup.read_text(encoding="utf-8"))["image_resolution_px"]
    width, height = res["width"], res["height"]

    reference = {}
    if args.compare_index:
        with open(args.compare_index, newline="", encoding="utf-8") as fh:
            reference = {Path(r["image"]).name: int(r["discard_code"]) for r in csv.DictReader(fh)}

    counts, missing_qa, agree = Counter(), [], Counter()
    rows = []
    for path in sorted(args.labels.rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "pog_px" not in doc:          # not a sample annotation (e.g. setup.json)
            continue
        code = quality_code(doc, width, height)
        rows.append((doc.get("image", path.with_suffix(".png").name), code, mean_ear(doc)))
        if code is None:
            missing_qa.append(path)
            continue
        counts[code] += 1
        if reference:
            ref = reference.get(doc.get("image", path.with_suffix(".png").name))
            agree[(ref, code)] += 1
        if args.write and doc.get("discard_info") != discard_info(code):
            doc["discard_info"] = discard_info(code)
            path.write_text(json.dumps(doc, indent=4), encoding="utf-8")

    total = sum(counts.values()) + len(missing_qa)
    print(f"{total} annotations: " + ", ".join(f"code {c} ({NAMES.get(c, 'kept')}): {n}" for c, n in sorted(counts.items())))
    if missing_qa:
        print(f"{len(missing_qa)} labelled annotations without quality metrics (not flagged), e.g. {missing_qa[0]}")
    if reference:
        same = sum(n for (ref, code), n in agree.items() if ref == code)
        print(f"compared with {args.compare_index.name}: {same} of {sum(agree.values())} codes identical")
        for (ref, code), n in sorted(agree.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            if ref != code:
                print(f"  index code {ref} -> computed code {code}: {n}")
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "discard_code", "ear_mean"])
            w.writerows(rows)
    if args.write:
        print("discard_info written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
