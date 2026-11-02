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


ETH_COLUMNS = {  # eth_xgaze_labels.csv column -> summary column
    "ok": "eth_ok", "fail_reason": "eth_fail_reason", "pred_pitch": "eth_pred_pitch", "pred_yaw": "eth_pred_yaw",
    "pred_vec_x": "eth_pred_vec_x", "pred_vec_y": "eth_pred_vec_y", "pred_vec_z": "eth_pred_vec_z",
    "face_center_x": "eth_face_center_x", "face_center_y": "eth_face_center_y", "face_center_z": "eth_face_center_z",
    "label_eth_x": "eth_label_x", "label_eth_y": "eth_label_y", "label_eth_z": "eth_label_z",
    "ang_err_eth_deg": "eth_ang_err_deg", "pog_pred_x_mm": "eth_pog_pred_x_mm", "pog_pred_y_mm": "eth_pog_pred_y_mm",
    "pog_err_mm": "eth_pog_err_mm",
}


def add_eth_labels(rows, eth_csv, setup, tol_mm=0.5):
    """Append ETH-XGaze baseline columns, matched on (user, session, task, image index).

    The baseline uses only the image, the intrinsics and the stimulus point, so its outputs do
    not depend on the head-pose/gaze labels. Each match is checked by comparing the baseline's
    stimulus point (pog_gt_*_mm) with the one derived from the sample's pog_px.
    """
    scr, pos = setup["screen"], setup["camera"]["position_mm"]
    eth = {}
    with open(eth_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # identifiers from the image name (UU_SS_TT_img-N.png): always present, unlike the
            # user/session/task columns, which are empty for some failed baseline rows
            ids, n = r["image"].rsplit("_img-", 1)
            key = (*ids.split("_"), int(n.split(".")[0]))
            if key in eth:
                raise ValueError(f"{eth_csv.name}: sample {key} appears twice")
            eth[key] = r
    def task_key(row):
        return row["user_id"], row["session_id"], row["task_id"]

    def assign(row, r):
        for src, col in ETH_COLUMNS.items():
            row[col] = r[src] if r[src] != "" else None
        row["eth_available"] = True

    # pass 1: rows with a stimulus point are checked frame by frame
    matched = mismatched = 0
    task_ok, task_bad, pending = set(), set(), []
    for row in rows:
        for col in ETH_COLUMNS.values():
            row[col] = None
        row["eth_available"] = False
        r = eth.get((*task_key(row), int(row["sample_id"])))
        if r is None:
            continue
        if r["pog_gt_x_mm"] == "":          # baseline failure (no landmarks): nothing to check
            pending.append((row, r))
            continue
        # stimulus point from pog_px (screen px, origin top-left) in camera mm, as standardize_release
        x_mm = scr["width_mm"] / 2 - row["pog_px_x"] * scr["width_mm"] / scr["width_px"] + pos["x"]
        y_mm = pos["y"] + row["pog_px_y"] * scr["height_mm"] / scr["height_px"]
        if abs(float(r["pog_gt_x_mm"]) - x_mm) > tol_mm or abs(float(r["pog_gt_y_mm"]) - y_mm) > tol_mm:
            mismatched += 1
            task_bad.add(task_key(row))
            continue
        assign(row, r)
        task_ok.add(task_key(row))
        matched += 1

    # pass 2: failure rows carry no stimulus point to check; they are accepted on the image-name
    # match (verified frame by frame for every row that has one), except in tasks with a mismatch
    failures = rejected_failures = 0
    for row, r in pending:
        if mismatched == 0 or task_key(row) not in task_bad:
            assign(row, r)
            failures += 1
        else:
            rejected_failures += 1
    print(f"ETH-XGaze labels: {len(eth)} rows in {eth_csv.name}; matched {matched} samples (checked "
          f"frame by frame) + {failures} baseline failures (matched by image name); rejected {mismatched} "
          f"(stimulus point differs > {tol_mm} mm) + {rejected_failures} failures in tasks with a mismatch; "
          f"{len(rows) - matched - failures} samples without baseline output")
    return mismatched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="release root (contains setup.json and user_XX)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--eth-csv", type=Path, help="ETH-XGaze baseline labels (eth_xgaze_labels.csv) to append")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4) * 2)
    args = ap.parse_args()

    with open(args.root / "setup.json", encoding="utf-8") as fh:
        setup = json.load(fh)
    files = sorted(p for p in args.root.glob("user_*/session_*/task_*/*.json"))
    print(f"found {len(files)} samples under {args.root}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(lambda p: parse_sample(p, setup), files, chunksize=256) if r]

    fieldnames = list(FIELDNAMES)
    if args.eth_csv:
        add_eth_labels(rows, args.eth_csv, setup)
        fieldnames += ["eth_available"] + list(ETH_COLUMNS.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    kept = sum(not r["discarded"] for r in rows)
    print(f"wrote {len(rows)} rows ({kept} kept, {len(rows) - kept} discarded) to {args.out}")


if __name__ == "__main__":
    main()
