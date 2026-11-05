"""Convert the labelled ReCalib acquisition tree into the released dataset layout.

Input (after the begibox labelling pipeline):
    RAW/<user hash>/<session start>/                 calibration task (images + JSONs)
    RAW/<user hash>/<session start>/<task start>/    test tasks
    DISCARDED/...                                    same layout, frames discarded by the pipeline

Output (release):
    OUT/user_XX/session_XX_YY/task_XX_YY_ZZ/XX_YY_ZZ_img-NNNN.{png,json}
    OUT/setup.json          acquisition constants, camera calibration, conventions
    OUT/sessions.csv        one row per task, with its start date-time
    OUT/recalib_index.csv   one row per sample, including discarded samples
Private (kept out of the release, they contain the original pseudonymous hashes):
    PRIVATE/user_mapping.csv, PRIVATE/manifest.csv, PRIVATE/conversion_report.json

Rules
    users     sorted original hashes -> user_00 .. (same order as previous releases)
    sessions  chronological per user -> session_XX_YY
    tasks     task 00 = calibration (session folder), 01.. = test tasks (sub-folders, chronological)
    samples   sample id = position of the frame in its task (4 digits, from 0001), in acquisition order
    discards  every sample is released; discarded ones carry discard_info.

Dry run by default (nothing is moved); add --execute to write the release. Execution is
resumable: samples already moved are skipped.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}$")
IMG_RE = re.compile(r"_img-(\d+)\.png$")

DISCARD_REASONS = {
    1: "IMAGE_CANNOT_BE_LOADED", 2: "HPE_MODEL_ERROR", 3: "JSON_CANNOT_BE_LOADED", 4: "IMAGE_HAS_NOT_JSON",
    5: "LANDMARKS_OUT_OF_RANGE", 6: "OUTLIER", 7: "BLINK", 8: "BLINK_DETECTION_MODEL_ERROR",
    9: "POOR_IMAGE_QUALITY", 10: "LOW_EYE_ASPECT_RATIO", 11: "BAD_DETECTION",
    12: "HIRU_COORDINATES_OUT_OF_RANGE", 13: "NOT_FIXATION",
}
SETUP_KEYS = ("posCam_mm", "rotCam", "screen_mm", "screen_pixels", "screen_zoom", "img_res", "screen_orientation")
LANDMARK_IDS = ("4", "33", "61", "129", "133", "168", "263", "308", "358", "362",
                "469", "470", "471", "472", "474", "475", "476", "477")


def rnd(values, digits=4):
    if values is None:
        return None
    return [round(float(v), digits) + 0.0 for v in values]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ts_dirs(path):
    return sorted(p for p in path.iterdir() if p.is_dir() and TS_RE.match(p.name))


def frames_in(task_rel, raw, discarded):
    """{image index: (png path, json path, discarded_by_pipeline)} for one task folder."""
    frames = {}
    for root, flagged in ((raw, False), (discarded, True)):
        folder = root / task_rel
        if not folder.is_dir():
            continue
        for png in folder.glob("*.png"):
            m = IMG_RE.search(png.name)
            if not m or png.name.endswith("_landmarks.png"):
                continue
            idx = int(m.group(1))
            if idx in frames:
                raise RuntimeError(f"image index {idx} present twice for {task_rel}")
            frames[idx] = (png, png.with_suffix(".json"), flagged)
    return frames


def plan(raw, discarded, workers):
    """Walk the tree and return (users, tasks, samples) without touching any file."""
    users = sorted(p for p in raw.iterdir() if p.is_dir())
    tasks, samples = [], []
    for u_idx, user in enumerate(users):
        u = f"{u_idx:02d}"
        stray = [p.name for p in user.iterdir() if p.is_dir() and not TS_RE.match(p.name)]
        if stray:
            print(f"  user {u}: ignoring non-session folders {stray}")
        for s_idx, session in enumerate(ts_dirs(user)):
            s = f"{s_idx:02d}"
            task_dirs = [session] + ts_dirs(session)
            for t_idx, task_dir in enumerate(task_dirs):
                t = f"{t_idx:02d}"
                rel = task_dir.relative_to(raw)
                frames = frames_in(rel, raw, discarded)
                task = dict(user_id=u, session_id=s, task_id=t,
                            task_type="calibration" if t_idx == 0 else "test",
                            start=datetime.strptime(task_dir.name, "%Y-%m-%dT%H_%M_%S").isoformat(),
                            raw_rel=str(rel), n_frames=len(frames))
                tasks.append(task)
                for pos, idx in enumerate(sorted(frames), 1):
                    png, js, flagged = frames[idx]
                    sa = f"{pos:04d}"
                    base = f"{u}_{s}_{t}_img-{sa}"
                    out_dir = Path(f"user_{u}", f"session_{u}_{s}", f"task_{u}_{s}_{t}")
                    samples.append(dict(task=task, sample_id=sa, raw_png=png, raw_json=js, flagged=flagged,
                                        out_png=out_dir / f"{base}.png", out_json=out_dir / f"{base}.json"))
    print(f"  hashing {len(samples)} images ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(lambda s: md5(s["raw_png"]), samples, chunksize=64))
    for s, h in zip(samples, hashes):
        s["md5"] = h
    return users, tasks, samples


def release_json(src, sample):
    task = sample["task"]
    hp, gz = src.get("head_pose"), src.get("gaze")
    qa = (src.get("QualityAssuranceMetrics") or {}).get("2DLandmarksInfo")
    expected = "calibration" if task["task_type"] == "calibration" else "test_with_targets"
    if src.get("session_type") != expected:
        raise RuntimeError(f"{sample['raw_json']}: session_type {src.get('session_type')!r}, expected {expected!r}")

    info = src.get("discard_info")
    code = int(info["discarding_reason"]) if info else 0
    return {
        "image": sample["out_png"].name,
        "user_id": task["user_id"], "session_id": task["session_id"],
        "task_id": task["task_id"], "sample_id": sample["sample_id"],
        "task_type": task["task_type"],
        "pog_px": src.get("pog_px", src.get("pos")),
        "gaze": None if not gz else {
            "vector": rnd(gz["vector"], 6), "origin_mm": rnd(gz["origin_mm"]), "pog_mm": rnd(gz["pog_mm"])},
        "head_pose": None if not hp else {
            "rotation_rad": rnd(hp["rotation_rad"]), "rotvec_rad": rnd(hp.get("rotvec_rad"), 6),
            "translation_mm": rnd(hp["translation_mm"]),
            "mediapipe_face_mesh_2d": hp.get("mediapipe_face_mesh_2d")},
        "quality_assurance_metrics": None if not qa else {
            "landmark_statistics_2d": {
                "max_x": qa["max_x"], "min_x": qa["min_x"], "max_y": qa["max_y"], "min_y": qa["min_y"],
                "mean_x": qa["meanX"], "mean_y": qa["meanY"],
                "head_bounding_box_surface": qa["head_Bounding_Box_surface"]},
            "eye_aspect_ratio": {
                "left": qa["eye_aspect_ratio"]["left_EAR"], "right": qa["eye_aspect_ratio"]["right_EAR"],
                "mean": qa["eye_aspect_ratio"]["cumulative_EAR"]}},
        "discard_info": None if code == 0 else {
            "discarded": True, "discarding_reason": code, "discarding_reason_name": DISCARD_REASONS[code]},
    }


def index_row(rel_png, d):
    row = {"image": str(rel_png).replace("\\", "/"), "user_id": d["user_id"], "session_id": d["session_id"],
           "task_id": d["task_id"], "sample_id": d["sample_id"], "task_type": d["task_type"],
           "pog_px_x": d["pog_px"]["x"], "pog_px_y": d["pog_px"]["y"]}
    gz, hp, qa = d["gaze"] or {}, d["head_pose"] or {}, d["quality_assurance_metrics"] or {}
    for name, vals in (("gaze_vector", gz.get("vector")), ("gaze_origin_mm", gz.get("origin_mm")),
                       ("gaze_pog_mm", gz.get("pog_mm")), ("head_rotation_rad", hp.get("rotation_rad")),
                       ("head_rotvec_rad", hp.get("rotvec_rad")), ("head_translation_mm", hp.get("translation_mm"))):
        for axis, v in zip("xyz", vals or (None, None, None)):
            row[f"{name}_{axis}"] = v
    for k, v in (qa.get("landmark_statistics_2d") or {}).items():
        row[f"qa_{k}"] = v
    for k, v in (qa.get("eye_aspect_ratio") or {}).items():
        row[f"qa_ear_{k}"] = v
    mesh = hp.get("mediapipe_face_mesh_2d") or {}
    for lid in LANDMARK_IDS:
        xy = mesh.get(lid) or (None, None)
        row[f"lm_{lid}_x"], row[f"lm_{lid}_y"] = xy[0], xy[1]
    info = d["discard_info"]
    row.update(discarded=bool(info), discard_code=info["discarding_reason"] if info else 0,
               discard_reason=info["discarding_reason_name"] if info else "")
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def git_version(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--discarded", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="release root")
    ap.add_argument("--private", required=True, type=Path, help="folder for mapping/manifest (not released)")
    ap.add_argument("--camera-npz", required=True, type=Path)
    ap.add_argument("--labelling-repo", type=Path, help="begibox checkout, to record its version")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--execute", action="store_true", help="write the release (default: dry run)")
    args = ap.parse_args()

    print(f"planning from {args.raw} (+ {args.discarded}) ...")
    users, tasks, samples = plan(args.raw, args.discarded, args.workers)

    # read all JSONs, check constants, build release documents
    setup_values, docs = {}, []
    for s in samples:
        with open(s["raw_json"], encoding="utf-8") as fh:
            src = json.load(fh)
        for k in SETUP_KEYS:
            setup_values.setdefault(k, set()).add(json.dumps(src.get(k), sort_keys=True))
        docs.append(release_json(src, s))
    varying = {k: v for k, v in setup_values.items() if len(v) != 1}
    if varying:
        raise RuntimeError(f"setup fields are not constant across the dataset: {sorted(varying)}")
    setup = {k: json.loads(next(iter(v))) for k, v in setup_values.items()}

    cal = np.load(args.camera_npz)
    counts = {}
    for d in docs:
        code = d["discard_info"]["discarding_reason"] if d["discard_info"] else 0
        counts[code] = counts.get(code, 0) + 1
    report = {
        "users": len(users), "sessions": len({(t["user_id"], t["session_id"]) for t in tasks}),
        "tasks": len(tasks), "samples": len(samples),
        "kept": counts.get(0, 0), "discarded": len(samples) - counts.get(0, 0),
        "discard_counts": {DISCARD_REASONS.get(c, "KEPT") if c else "KEPT": n for c, n in sorted(counts.items())},
        "labelling_code_version": git_version(args.labelling_repo) if args.labelling_repo else "not recorded",
    }
    print(json.dumps(report, indent=2))

    setup_json = {
        "device": "Microsoft Surface Pro 7+ (integrated front camera)",
        "image_resolution_px": setup["img_res"],
        "screen": {"width_mm": setup["screen_mm"]["width"], "height_mm": setup["screen_mm"]["height"],
                   "width_px": int(setup["screen_pixels"]["width"] * setup["screen_zoom"]),
                   "height_px": int(setup["screen_pixels"]["height"] * setup["screen_zoom"]),
                   "logical_width_px": setup["screen_pixels"]["width"],
                   "logical_height_px": setup["screen_pixels"]["height"],
                   "display_zoom": setup["screen_zoom"], "orientation": setup["screen_orientation"]},
        "camera": {"position_mm": setup["posCam_mm"], "rotation_rad": setup["rotCam"],
                   "camera_matrix": np.asarray(cal["mtx"]).round(4).tolist(),
                   "dist_coeffs_k1_k2_p1_p2_k3": np.asarray(cal["dist"]).ravel().round(6).tolist()},
        "conventions": {
            "pog_px": "screen target in physical pixels of the native panel (origin top-left)",
            "3d_frame": "camera frame, mm: +X image right, +Y image down, +Z away from the camera",
            "gaze": "origin_mm = midpoint between the eyes; pog_mm = target on the screen plane; "
                    "vector = unit(pog_mm - origin_mm)",
            "head_pose": "MediaPipe FaceLandmarker pose re-expressed in the calibrated camera; "
                         "rotation_rad = extrinsic Euler 'zyx' angles [roll, yaw, pitch]; rotvec_rad = "
                         "same rotation as axis-angle; translation_mm = origin of MediaPipe's canonical face",
            "discard_info": "null for kept samples; otherwise discarding_reason (code) and its name"},
        "labelling_code_version": report["labelling_code_version"],
    }
    sessions_rows = [{k: t[k] for k in ("user_id", "session_id", "task_id", "task_type", "start", "n_frames")}
                     for t in tasks]
    index_rows = [index_row(s["out_png"], d) for s, d in zip(samples, docs)]
    mapping_rows = [{"original_hash": u.name, "user_id": f"user_{i:02d}"} for i, u in enumerate(users)]
    manifest_rows = [{"raw_png": str(s["raw_png"]), "release_png": str(s["out_png"]), "md5": s["md5"],
                      "discard_code": d["discard_info"]["discarding_reason"] if d["discard_info"] else 0}
                     for s, d in zip(samples, docs)]

    args.private.mkdir(parents=True, exist_ok=True)
    write_csv(args.private / "user_mapping.csv", mapping_rows)
    write_csv(args.private / "manifest.csv", manifest_rows)
    with open(args.private / "conversion_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"private files written to {args.private}")

    if not args.execute:
        print("dry run: nothing moved. Re-run with --execute to write the release.")
        return 0

    moved = skipped = 0
    for s, d in zip(samples, docs):
        out_png, out_json = args.out / s["out_png"], args.out / s["out_json"]
        if out_png.exists() and not s["raw_png"].exists():
            skipped += 1
            continue
        out_png.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=4)
        shutil.move(str(s["raw_png"]), str(out_png))
        if s["raw_json"].exists():
            s["raw_json"].unlink()
        moved += 1
    with open(args.out / "setup.json", "w", encoding="utf-8") as fh:
        json.dump(setup_json, fh, indent=4)
    write_csv(args.out / "sessions.csv", sessions_rows)
    write_csv(args.out / "recalib_index.csv", index_rows)
    print(f"release written to {args.out}: {moved} samples moved, {skipped} already done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
