"""Write dataset_summary CSV (one row per sample) from the released ReCalib dataset.

Same columns and order as statistical_analysis/create_csv_summary.py, so the analysis and
visualization tools keep working, adapted to the release schema written by
standardize_release.py:
  * eye aspect ratio is read from quality_assurance_metrics.eye_aspect_ratio
  * constants (rotCam, image resolution, screen size) come from setup.json; screen size is
    given in physical pixels (2736x1824), the space of pog_px
  * attr_* columns are left empty: participant attributes are not part of the release
  * discarded samples are included (discarded / discard_code / discard_reason columns);
    filter on discarded == False for the recommended subset

Usage:
    python create_dataset_summary.py --root F:/recalib_release/ReCalib --out dataset_summary_recalib.csv
"""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LANDMARK_IDS = [4, 33, 61, 129, 133, 168, 263, 308, 358, 362, 469, 470, 471, 472, 474, 475, 476, 477]
DISCARD_REASONS_MAP = {
    0: "NO_DISCARDED", 1: "IMAGE_CANNOT_BE_LOADED", 2: "HPE_MODEL_ERROR", 3: "JSON_CANNOT_BE_LOADED",
    4: "IMAGE_HAS_NOT_JSON", 5: "LANDMARKS_OUT_OF_RANGE", 6: "OUTLIER", 7: "BLINK",
    8: "BLINK_DETECTION_MODEL_ERROR", 9: "POOR_IMAGE_QUALITY", 10: "LOW_EYE_ASPECT_RATIO",
    11: "BAD_DETECTION", 12: "HIRU_COORDINATES_OUT_OF_RANGE", 13: "NOT_FIXATION", 14: "DUPLICATE_FRAME",
}
FIELDNAMES = [
    "image", "sample_id", "task_id", "session_id", "user_id", "task_type",
    "pog_px_x", "pog_px_y",
    "gaze_vector_x", "gaze_vector_y", "gaze_vector_z",
    "gaze_origin_x", "gaze_origin_y", "gaze_origin_z",
    "gaze_pog_x", "gaze_pog_y", "gaze_pog_z",
    "head_rot_x", "head_rot_y", "head_rot_z",
    "head_trans_x", "head_trans_y", "head_trans_z",
    "rot_cam_x", "rot_cam_y", "rot_cam_z",
    "qa_max_x", "qa_min_x", "qa_max_y", "qa_min_y", "qa_mean_x", "qa_mean_y",
    "qa_head_bbox_surface", "qa_ear_left", "qa_ear_right",
    "img_res_width", "img_res_height", "screen_width_px", "screen_height_px",
    "attr_gender", "attr_age", "attr_distance", "attr_contacts",
] + [f"lm_{i}_{a}" for i in LANDMARK_IDS for a in "xy"] + ["discarded", "discard_code", "discard_reason"]


def xyz(values):
    return list(values) if values else [None, None, None]


def parse_sample(json_path, setup):
    try:
        with open(json_path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:  # report and skip unreadable files
        print(f"skipped {json_path}: {exc}")
        return None
    pog = d.get("pog_px") or {}
    gz = d.get("gaze") or {}
    hp = d.get("head_pose") or {}
    qa = d.get("quality_assurance_metrics") or {}
    st = qa.get("landmark_statistics_2d") or {}
    ear = qa.get("eye_aspect_ratio") or st.get("eye_aspect_ratio") or {}
    info = d.get("discard_info")
    code = int(info.get("discarding_reason", 0)) if info else 0
    cam, img, scr = setup["camera"]["rotation_rad"], setup["image_resolution_px"], setup["screen"]

    row = dict.fromkeys(FIELDNAMES)
    row.update({
        "image": d.get("image"), "sample_id": d.get("sample_id"), "task_id": d.get("task_id"),
        "session_id": d.get("session_id"), "user_id": d.get("user_id"), "task_type": d.get("task_type"),
        "pog_px_x": int(pog["x"]) if pog.get("x") is not None else None,
        "pog_px_y": int(pog["y"]) if pog.get("y") is not None else None,
        "rot_cam_x": cam["x"], "rot_cam_y": cam["y"], "rot_cam_z": cam["z"],
        "qa_max_x": st.get("max_x"), "qa_min_x": st.get("min_x"), "qa_max_y": st.get("max_y"),
        "qa_min_y": st.get("min_y"), "qa_mean_x": st.get("mean_x"), "qa_mean_y": st.get("mean_y"),
        "qa_head_bbox_surface": st.get("head_bounding_box_surface"),
        "qa_ear_left": ear.get("left"), "qa_ear_right": ear.get("right"),
        "img_res_width": img["width"], "img_res_height": img["height"],
        "screen_width_px": scr["width_px"], "screen_height_px": scr["height_px"],
        "discarded": bool(info and info.get("discarded")), "discard_code": code,
        "discard_reason": DISCARD_REASONS_MAP.get(code, f"UNKNOWN_CODE_{code}"),
    })
    for prefix, values in (("gaze_vector", gz.get("vector")), ("gaze_origin", gz.get("origin_mm")),
                           ("gaze_pog", gz.get("pog_mm")), ("head_rot", hp.get("rotation_rad")),
                           ("head_trans", hp.get("translation_mm"))):
        for axis, v in zip("xyz", xyz(values)):
            row[f"{prefix}_{axis}"] = v
    mesh = hp.get("mediapipe_face_mesh_2d") or {}
    for lm in LANDMARK_IDS:
        x, y = mesh.get(str(lm)) or (None, None)
        row[f"lm_{lm}_x"] = int(x) if x is not None else None
        row[f"lm_{lm}_y"] = int(y) if y is not None else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="release root (contains setup.json and user_XX)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4) * 2)
    args = ap.parse_args()

    with open(args.root / "setup.json", encoding="utf-8") as fh:
        setup = json.load(fh)
    files = sorted(p for p in args.root.glob("user_*/session_*/task_*/*.json"))
    print(f"found {len(files)} samples under {args.root}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(lambda p: parse_sample(p, setup), files, chunksize=256) if r]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    kept = sum(not r["discarded"] for r in rows)
    print(f"wrote {len(rows)} rows ({kept} kept, {len(rows) - kept} discarded) to {args.out}")


if __name__ == "__main__":
    main()
