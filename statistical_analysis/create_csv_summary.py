"""Export sample JSON labels using the project's dataset summary CSV headers.

The column names and order match dataset_summary_recalib.csv and
dataset_summary_targets.csv, consumed by statistical_analysis and visualization.
Set DATASET_ROOT and OUTPUT_CSV_PATH below before running this script.
"""

import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIGURATION =================
DATASET_ROOT = r"D:\Mamu\Mamu"
OUTPUT_CSV_PATH = "dataset_summary_targets.csv"
MAX_WORKERS = os.cpu_count() * 2  # High performance multi-threading
# =================================================

# Standard MediaPipe landmark IDs used in your dataset
LANDMARK_IDS = [4, 33, 61, 129, 133, 168, 263, 308, 358, 362, 469, 470, 471, 472, 474, 475, 476, 477]

# Human-readable mapping for discarding reason codes
DISCARD_REASONS_MAP = {
    0: "NO_DISCARDED",
    1: "IMAGE_CANNOT_BE_LOADED",
    2: "HPE_MODEL_ERROR",
    3: "JSON_CANNOT_BE_LOADED",
    4: "IMAGE_HAS_NOT_JSON",
    5: "LANDMARKS_OUT_OF_RANGE",
    6: "OUTLIER",
    7: "BLINK",
    8: "BLINK_DETECTION_MODEL_ERROR",
    9: "POOR_IMAGE_QUALITY",
    10: "LOW_EYE_ASPECT_RATIO",
    11: "BAD_DETECTION",
    12: "HIRU_COORDINATES_OUT_OF_RANGE",
    13: "NOT_FIXATION"
}

def parse_complete_sample(json_path):
    """Parses a standardized JSON file and extracts all parameters, mapping raw discard codes to text labels."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    # Base structures
    pog_px = data.get("pog_px") or {}
    gaze = data.get("gaze") or {}
    head_pose = data.get("head_pose") or {}
    mesh_2d = head_pose.get("mediapipe_face_mesh_2d") or {}
    qa = data.get("quality_assurance_metrics") or {}
    stats_2d = qa.get("landmark_statistics_2d") or {}
    ear = stats_2d.get("eye_aspect_ratio") or {}
    rot_cam = data.get("rotCam") or {}
    img_res = data.get("img_res") or {}
    attrs = data.get("attributes") or {}
    screen_pixels = data.get("screen_pixels") or {}

    # Extract vector lists safely
    g_vector = gaze.get("vector") or [None, None, None]
    g_origin = gaze.get("origin_mm") or [None, None, None]
    g_pog = gaze.get("pog_mm") or [None, None, None]
    
    hp_rot = head_pose.get("rotation_rad") or [None, None, None]
    hp_trans = head_pose.get("translation_mm") or [None, None, None]
    
    # Handle discard info, tracking BOTH code and textual reason
    discard_info = data.get("discard_info")
    is_discarded = False
    raw_reason_code = 0
    discard_reason_text = "NO_DISCARDED"
    
    if isinstance(discard_info, dict):
        is_discarded = discard_info.get("discarded", False)
        raw_reason_code = discard_info.get("discarding_reason", 0)
        discard_reason_text = DISCARD_REASONS_MAP.get(raw_reason_code, f"UNKNOWN_CODE_{raw_reason_code}")
    elif discard_info is None:
        is_discarded = False
        raw_reason_code = 0
        discard_reason_text = "NO_DISCARDED"

    # 1. Map core, gaze, camera setup and head pose data
    row = {
        "image": data.get("image"),
        "sample_id": data.get("sample_id"),
        "task_id": data.get("task_id"),
        "session_id": data.get("session_id"),
        "user_id": data.get("user_id"),
        "task_type": data.get("task_type"),
        
        "pog_px_x": int(pog_px.get("x")) if pog_px.get("x") is not None else None,
        "pog_px_y": int(pog_px.get("y")) if pog_px.get("y") is not None else None,
        
        "gaze_vector_x": g_vector[0] if len(g_vector) > 0 else None,
        "gaze_vector_y": g_vector[1] if len(g_vector) > 1 else None,
        "gaze_vector_z": g_vector[2] if len(g_vector) > 2 else None,
        
        "gaze_origin_x": g_origin[0] if len(g_origin) > 0 else None,
        "gaze_origin_y": g_origin[1] if len(g_origin) > 1 else None,
        "gaze_origin_z": g_origin[2] if len(g_origin) > 2 else None,
        
        "gaze_pog_x": g_pog[0] if len(g_pog) > 0 else None,
        "gaze_pog_y": g_pog[1] if len(g_pog) > 1 else None,
        "gaze_pog_z": g_pog[2] if len(g_pog) > 2 else None,
        
        "head_rot_x": hp_rot[0] if len(hp_rot) > 0 else None,
        "head_rot_y": hp_rot[1] if len(hp_rot) > 1 else None,
        "head_rot_z": hp_rot[2] if len(hp_rot) > 2 else None,
        
        "head_trans_x": hp_trans[0] if len(hp_trans) > 0 else None,
        "head_trans_y": hp_trans[1] if len(hp_trans) > 1 else None,
        "head_trans_z": hp_trans[2] if len(hp_trans) > 2 else None,
        
        # Camera Extrinsics Matrix (rotCam profile)
        "rot_cam_x": rot_cam.get("x"),
        "rot_cam_y": rot_cam.get("y"),
        "rot_cam_z": rot_cam.get("z"),
        
        # Quality Metrics
        "qa_max_x": stats_2d.get("max_x"),
        "qa_min_x": stats_2d.get("min_x"),
        "qa_max_y": stats_2d.get("max_y"),
        "qa_min_y": stats_2d.get("min_y"),
        "qa_mean_x": stats_2d.get("mean_x"),
        "qa_mean_y": stats_2d.get("mean_y"),
        "qa_head_bbox_surface": stats_2d.get("head_bounding_box_surface"),
        "qa_ear_left": ear.get("left"),
        "qa_ear_right": ear.get("right"),
        
        # Image Properties & Metadata
        "img_res_width": img_res.get("width"),
        "img_res_height": img_res.get("height"),
        "screen_width_px": screen_pixels.get("width"),
        "screen_height_px": screen_pixels.get("height"),
        "attr_gender": attrs.get("gender"),
        "attr_age": attrs.get("age"),
        "attr_distance": attrs.get("distance"),
        "attr_contacts": attrs.get("contacts"),
        
        # Status Evaluation Parameters
        "discarded": is_discarded,
        "discard_code": raw_reason_code,
        "discard_reason": discard_reason_text
    }

    # 2. Dynamically unpack all 18 facial mesh landmark coordinates
    for lm_id in LANDMARK_IDS:
        lm_coords = mesh_2d.get(str(lm_id)) or [None, None]
        row[f"lm_{lm_id}_x"] = int(lm_coords[0]) if lm_coords[0] is not None else None
        row[f"lm_{lm_id}_y"] = int(lm_coords[1]) if lm_coords[1] is not None else None

    return row

def generate_complete_labels_csv():
    base_dir = Path(DATASET_ROOT)
    if not base_dir.exists():
        print(f"❌ Error: Directory path {DATASET_ROOT} does not exist.")
        return

    print("🔍 Scanning directories for sample files...")
    json_files = list(base_dir.rglob("*.json"))
    total_files = len(json_files)
    print(f"Found {total_files} samples to process.")

    if total_files == 0:
        return

    # Define layout order for output columns
    fieldnames = [
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
        "attr_gender", "attr_age", "attr_distance", "attr_contacts"
    ]
    
    # Append the landmark coordinate headers dynamically
    for lm_id in LANDMARK_IDS:
        fieldnames.extend([f"lm_{lm_id}_x", f"lm_{lm_id}_y"])
        
    # Append final status flag evaluations at the end
    fieldnames.extend(["discarded", "discard_code", "discard_reason"])

    print(f"🚀 Processing labels across {MAX_WORKERS} concurrent threads...")
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(parse_complete_sample, json_files), total=total_files, desc="Parsing Dataset"))
        rows = [r for r in results if r is not None]

    print("💾 Saving comprehensive master spreadsheet...")
    out_path = Path(OUTPUT_CSV_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"🎉 Done! Complete master matrix spreadsheet written ({len(rows)} samples) to:\n   {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    generate_complete_labels_csv()
