"""Extract every per-frame JSON label of the raw ReCalib recordings into one CSV.

The raw acquisition JSONs use different key names than the released dataset
(``pos`` instead of ``pog_px``, ``hpe.6d`` instead of ``head_pose``,
``gaze.destiny`` instead of ``gaze.pog_mm``, ...). This script flattens each raw
JSON and renames the fields to the column names of ``dataset_summary_recalib.csv``.

Output columns, in order:
  1. the released-CSV columns (same names and order as ``dataset_summary_recalib.csv``),
  2. the remaining raw labels (attributes, device, eye patch, ...) under new flat names,
  3. recording dates parsed from the file names (``session_date``, ``session_datetime``,
     ``task_date``, ``task_datetime``; ISO format, local time as recorded),
  4. provenance columns (``user_hash``, ``session_folder``, ``task_folder``,
     ``image_raw``, ``source_json``, ``session_type``).

Raw folder layout (``<root>/<user_hash>/<session>/<task>/``):
  * JSONs directly in ``<session>/``  -> calibration, 9-point,  task_id 0
  * JSONs in the ``<task>/`` subfolders -> test_with_targets, 16-point, task_id 1, 2, 3
    (subfolders sorted by timestamp)
``user_id`` is not stored in the raw data (folders are hashes) so it is taken from
``--user-map`` (JSON ``{user_hash: "00"}``); without it ``user_id`` and ``image`` are empty.
``session_id`` is the 0-based chronological index of the session folder of that user.
Every file name starts with the timestamp of its recording block (``2024-12-12T09_53_53_img-1.json``):
the session start for calibration frames, the task start for the 16-point frames. So
``session_datetime`` is the session folder timestamp and ``task_datetime`` the timestamp in
the file name (equal to ``session_datetime`` for task_id 0).

Nothing is filtered: discarded frames are kept and flagged in ``discarded`` /
``discard_code`` / ``discard_reason``. Frames whose JSON has no ``hpe``/``gaze``
(discarded stubs) get empty values in those columns.

Same name does not mean same value. Compared with ``dataset_summary_recalib.csv`` for
the same frames, ``pog_px_*``, ``gaze_pog_*``, ``rot_cam_*``, ``qa_*`` and ``img_res_*``
are identical, but ``gaze_origin_*``, ``gaze_vector_*``, ``head_rot_*``, ``head_trans_*``
and the ``lm_*`` pixels differ slightly in the raw JSONs (the release was re-estimated),
and the released ``attr_*`` columns are zeroed while the raw JSONs hold the real values.
The raw attributes and device fields are personal data: keep the output out of the
public repository.

Example:
    py -I evaluation/extract_raw_labels.py --raw-root recalib_raw --output recalib_raw_labels.csv
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LANDMARK_IDS = [4, 33, 61, 129, 133, 168, 263, 308, 358, 362,
                469, 470, 471, 472, 474, 475, 476, 477]
TASK_TYPE = {"calibration": "9-point", "test_with_targets": "16-point"}
IMG_RE = re.compile(r"_img-(\d+)\.")
STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})")

# (raw path, column) for scalar fields; order = column order of dataset_summary_recalib.csv
RELEASED_SCALARS = [
    ("pos.x", "pog_px_x"), ("pos.y", "pog_px_y"),
]
RELEASED_LISTS = [
    ("gaze.vector", ["gaze_vector_x", "gaze_vector_y", "gaze_vector_z"]),
    ("gaze.origin", ["gaze_origin_x", "gaze_origin_y", "gaze_origin_z"]),
    ("gaze.destiny", ["gaze_pog_x", "gaze_pog_y", "gaze_pog_z"]),
    ("hpe.6d", ["head_rot_x", "head_rot_y", "head_rot_z",
                "head_trans_x", "head_trans_y", "head_trans_z"]),
]
RELEASED_SCALARS_2 = [
    ("rotCam.x", "rot_cam_x"), ("rotCam.y", "rot_cam_y"), ("rotCam.z", "rot_cam_z"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.max_x", "qa_max_x"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.min_x", "qa_min_x"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.max_y", "qa_max_y"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.min_y", "qa_min_y"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.meanX", "qa_mean_x"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.meanY", "qa_mean_y"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.head_Bounding_Box_surface", "qa_head_bbox_surface"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.eye_aspect_ratio.left_EAR", "qa_ear_left"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.eye_aspect_ratio.right_EAR", "qa_ear_right"),
    ("img_res.width", "img_res_width"), ("img_res.height", "img_res_height"),
    ("screen_pixels.width", "screen_width_px"), ("screen_pixels.height", "screen_height_px"),
    ("attributes.gender", "attr_gender"), ("attributes.age", "attr_age"),
    ("attributes.distance", "attr_distance"), ("attributes.contacts", "attr_contacts"),
]
# raw labels that the released CSV does not carry
EXTRA_SCALARS = [
    ("QualityAssuranceMetrics.2DLandmarksInfo.diff_x", "qa_diff_x"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.diff_y", "qa_diff_y"),
    ("QualityAssuranceMetrics.2DLandmarksInfo.eye_aspect_ratio.cumulative_EAR", "qa_ear_cumulative"),
    ("attributes.eyesColor", "attr_eyes_color"), ("attributes.eyesShape", "attr_eyes_shape"),
    ("attributes.facialHair", "attr_facial_hair"), ("attributes.mask", "attr_mask"),
    ("attributes.place", "attr_place"), ("attributes.chinrest", "attr_chinrest"),
    ("attributes.annotations", "attr_annotations"), ("attributes.test", "attr_test"),
    ("attributes.wrongPositioning", "attr_wrong_positioning"),
    ("attributes.glasses.using", "attr_glasses_using"),
    ("attributes.glasses.progressiveGlasses", "attr_glasses_progressive"),
    ("attributes.glasses.myopia.left", "attr_glasses_myopia_left"),
    ("attributes.glasses.myopia.right", "attr_glasses_myopia_right"),
    ("attributes.glasses.hyperopia.left", "attr_glasses_hyperopia_left"),
    ("attributes.glasses.hyperopia.right", "attr_glasses_hyperopia_right"),
    ("attributes.glasses.astigmatism.left", "attr_glasses_astigmatism_left"),
    ("attributes.glasses.astigmatism.right", "attr_glasses_astigmatism_right"),
    ("attributes.glasses.presbyopia.left", "attr_glasses_presbyopia_left"),
    ("attributes.glasses.presbyopia.right", "attr_glasses_presbyopia_right"),
    ("outdoor.valid", "outdoor_valid"),
    ("screen_mm.width", "screen_width_mm"), ("screen_mm.height", "screen_height_mm"),
    ("screen_orientation", "screen_orientation"), ("screen_zoom", "screen_zoom"),
    ("posCam_mm.x", "cam_pos_x_mm"), ("posCam_mm.y", "cam_pos_y_mm"), ("posCam_mm.z", "cam_pos_z_mm"),
    ("hpe.method", "hpe_method"),
    ("deviceSpecifications.System", "device_system"),
    ("deviceSpecifications.Node", "device_node"),
    ("deviceSpecifications.Release", "device_release"),
    ("deviceSpecifications.Version", "device_version"),
    ("deviceSpecifications.Machine", "device_machine"),
    ("deviceSpecifications.Processor", "device_processor"),
    ("deviceSpecifications.manufacturer", "device_manufacturer"),
    ("deviceSpecifications.model", "device_model"),
    ("deviceSpecifications.name_device", "device_name"),
    ("deviceSpecifications.num_processors", "device_num_processors"),
    ("deviceSpecifications.system_type", "device_system_type"),
    ("deviceSpecifications.system_family", "device_system_family"),
    ("session_name", "session_name_raw"),
]
EXTRA_LISTS = [("eyePatch.patchSize", ["eye_patch_width", "eye_patch_height"])]
for _side, _name in (("leftEye", "eye_left"), ("rightEye", "eye_right")):
    EXTRA_SCALARS += [(f"eyePatch.{_side}.angle", f"{_name}_angle"),
                      (f"eyePatch.{_side}.openEye", f"{_name}_open")]
    EXTRA_LISTS += [
        (f"eyePatch.{_side}.inner", [f"{_name}_inner_x", f"{_name}_inner_y"]),
        (f"eyePatch.{_side}.outer", [f"{_name}_outer_x", f"{_name}_outer_y"]),
        (f"eyePatch.{_side}.gaze", [f"{_name}_gaze_x", f"{_name}_gaze_y", f"{_name}_gaze_z"]),
        (f"eyePatch.{_side}.gazeRaw", [f"{_name}_gaze_raw_x", f"{_name}_gaze_raw_y", f"{_name}_gaze_raw_z"]),
        (f"eyePatch.{_side}.location", [f"{_name}_location_x", f"{_name}_location_y", f"{_name}_location_z"]),
    ]
EXTRA_MATRICES = [(f"eyePatch.{s}.Mrot", n) for s, n in (("leftEye", "eye_left"), ("rightEye", "eye_right"))]

LM_PREFIX = "hpe.facialLandmarks2D."
HANDLED_EXTRA = {"image", "session_type", "discard_info.discarded", "discard_info.discarding_reason"}


def _released_columns():
    cols = ["image", "sample_id", "task_id", "session_id", "user_id", "task_type"]
    cols += [c for _, c in RELEASED_SCALARS]
    for _, names in RELEASED_LISTS:
        cols += names
    # dataset_summary_recalib.csv puts head pose before rot_cam, then qa/img/screen/attr, then landmarks
    cols += [c for _, c in RELEASED_SCALARS_2]
    cols += [f"lm_{i}_{a}" for i in LANDMARK_IDS for a in "xy"]
    cols += ["discarded", "discard_code", "discard_reason"]
    return cols


RELEASED_COLUMNS = _released_columns()
EXTRA_COLUMNS = ([c for _, c in EXTRA_SCALARS]
                 + [c for _, names in EXTRA_LISTS for c in names]
                 + [f"{n}_mrot_r{r}c{c}" for _, n in EXTRA_MATRICES for r in range(2) for c in range(3)])
DATE_COLUMNS = ["session_date", "session_datetime", "task_date", "task_datetime"]
PROVENANCE_COLUMNS = ["session_type", "user_hash", "session_folder", "task_folder",
                      "image_raw", "source_json"]
COLUMNS = RELEASED_COLUMNS + EXTRA_COLUMNS + DATE_COLUMNS + PROVENANCE_COLUMNS

KNOWN_PATHS = (
    {p for p, _ in RELEASED_SCALARS + RELEASED_SCALARS_2 + EXTRA_SCALARS}
    | {p for p, _ in RELEASED_LISTS + EXTRA_LISTS}
    | {p for p, _ in EXTRA_MATRICES}
    | HANDLED_EXTRA
)


def leaf_paths(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaf_paths(value, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix


def get_path(obj, dotted):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def load_discard_names():
    path = Path(__file__).resolve().parent.parent / "docs" / "discarding_criteria.py"
    try:
        spec = importlib.util.spec_from_file_location("discarding_criteria", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {member.value: member.name for member in module.DiscardingReasons}
    except Exception:  # noqa: BLE001 - names are a convenience, the code is the source of truth
        return {}


DISCARD_NAMES = load_discard_names()


def flatten(data, ctx):
    """Flatten one raw JSON dict into {column: value}; returns (row, unknown_paths)."""
    row = dict.fromkeys(COLUMNS)
    for path, col in RELEASED_SCALARS + RELEASED_SCALARS_2 + EXTRA_SCALARS:
        row[col] = get_path(data, path)
    for path, cols in RELEASED_LISTS + EXTRA_LISTS:
        values = get_path(data, path)
        if isinstance(values, list):
            for col, value in zip(cols, values):
                row[col] = value
    for path, name in EXTRA_MATRICES:
        matrix = get_path(data, path)
        if isinstance(matrix, list):
            for r, line in enumerate(matrix):
                for c, value in enumerate(line):
                    row[f"{name}_mrot_r{r}c{c}"] = value

    landmarks = get_path(data, "hpe.facialLandmarks2D") or {}
    for key, xy in landmarks.items():
        if int(key) in LANDMARK_IDS:
            row[f"lm_{key}_x"], row[f"lm_{key}_y"] = xy
    unknown = {p for p in leaf_paths(data)
               if p not in KNOWN_PATHS and not (p.startswith(LM_PREFIX) and int(p[len(LM_PREFIX):]) in LANDMARK_IDS)}

    info = data.get("discard_info")
    if isinstance(info, dict):
        code = info.get("discarding_reason")
        row["discarded"] = info.get("discarded")
        row["discard_code"] = code
        row["discard_reason"] = DISCARD_NAMES.get(code, "" if code is None else f"UNKNOWN_{code}")
    else:
        row["discarded"], row["discard_code"], row["discard_reason"] = False, 0, DISCARD_NAMES.get(0, "NO_DISCARDED")

    session_type = data.get("session_type")
    row["session_type"] = session_type
    row["task_type"] = TASK_TYPE.get(session_type, "")
    row["image_raw"] = data.get("image")
    row.update(ctx)
    return row, unknown


def collect_jobs(raw_root, user_map):
    """One job per JSON: (path, context dict). Deterministic order = user hash, session, task, frame."""
    jobs = []
    for user_hash in sorted(p.name for p in raw_root.iterdir() if p.is_dir()):
        user_dir = raw_root / user_hash
        user_id = user_map.get(user_hash, "")
        for session_idx, session in enumerate(sorted(p.name for p in user_dir.iterdir() if p.is_dir())):
            session_dir = user_dir / session
            session_date, session_datetime = parse_stamp(session)
            groups = [(0, "", session_dir)]
            tasks = sorted(p.name for p in session_dir.iterdir() if p.is_dir())
            groups += [(i + 1, t, session_dir / t) for i, t in enumerate(tasks)]
            for task_id, task_folder, folder in groups:
                for name in sorted(os.listdir(folder), key=natural_key):
                    if not name.endswith(".json"):
                        continue
                    match = IMG_RE.search(name)
                    frame = int(match.group(1)) if match else None
                    image = ""
                    if user_id != "" and frame is not None:
                        image = f"{int(user_id):02d}_{session_idx:02d}_{task_id:02d}_img-{frame:03d}.png"
                    task_date, task_datetime = parse_stamp(name)
                    jobs.append((folder / name, {
                        "session_date": session_date, "session_datetime": session_datetime,
                        "task_date": task_date, "task_datetime": task_datetime,
                        "image": image, "user_id": user_id, "session_id": session_idx, "task_id": task_id,
                        "user_hash": user_hash, "session_folder": session, "task_folder": task_folder,
                        "source_json": str((folder / name).relative_to(raw_root)).replace("\\", "/"),
                    }))
    return jobs


def parse_stamp(text):
    """'2024-12-12T09_53_53...' -> ('2024-12-12', '2024-12-12T09:53:53'); ('', '') if not a timestamp."""
    match = STAMP_RE.match(text)
    if not match:
        return "", ""
    date, hh, mm, ss = match.groups()
    return date, f"{date}T{hh}:{mm}:{ss}"


def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def process(job):
    path, ctx = job
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return flatten(data, ctx)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder with one subfolder per user hash.")
    parser.add_argument("--output", type=Path, default=Path("recalib_raw_labels.csv"))
    parser.add_argument("--user-map", type=Path, default=None,
                        help='JSON {"<user_hash>": "00", ...}; fills user_id and the repo-style image name.')
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.raw_root.is_dir():
        sys.exit(f"raw root not found: {args.raw_root}")
    user_map = {}
    if args.user_map is not None:
        user_map = {str(k): str(v) for k, v in json.loads(args.user_map.read_text(encoding="utf-8")).items()}

    jobs = collect_jobs(args.raw_root, user_map)
    print(f"{len(jobs)} JSON files found", flush=True)
    unknown_counts, failures = Counter(), []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", newline="", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        results = pool.map(lambda job: _safe(job), jobs, chunksize=64)
        for job, result in zip(jobs, results):
            if isinstance(result, Exception):
                failures.append((job[0], repr(result)))
                continue
            row, unknown = result
            unknown_counts.update(unknown)
            writer.writerow(row)
            written += 1
            if written % 20000 == 0:
                print(f"  {written}", flush=True)
    print(f"wrote {written} rows x {len(COLUMNS)} columns -> {args.output}")
    if user_map:
        missing = sorted({j[1]["user_hash"] for j in jobs} - set(user_map))
        if missing:
            print(f"WARNING: no user_id for hashes {missing}")
    if unknown_counts:
        print("WARNING: JSON keys not mapped to any column (their values are NOT in the CSV):")
        for path, count in unknown_counts.most_common():
            print(f"  {count:8d}  {path}")
    if failures:
        print(f"WARNING: {len(failures)} files failed to parse, first 10:")
        for path, error in failures[:10]:
            print(f"  {path}: {error}")
        sys.exit(1)


def _safe(job):
    try:
        return process(job)
    except Exception as error:  # noqa: BLE001 - reported per file, not fatal to the whole export
        return error


if __name__ == "__main__":
    main()
