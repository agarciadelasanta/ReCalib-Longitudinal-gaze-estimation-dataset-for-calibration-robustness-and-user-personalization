import json
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path(r'E:\user_personalization_tagged\Mamu')
DRY_RUN = False  # Set to False to apply changes
# ---------------------

def round_and_convert(obj):
    """Recursively rounds all floating point numbers to 4 decimal places."""
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: round_and_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_and_convert(i) for i in obj]
    return obj

def final_standardization(root_dir):
    mode_text = "DRY RUN" if DRY_RUN else "LIVE MODE"
    print(f"--- Standardizing Order, Keys, and Numeric Types: {mode_text} ---\n")

    json_files = list(root_dir.glob("user_*/session_*_*/task_*_*/*.json"))

    for json_path in json_files:
        # Extract indices from parent folders
        task_folder = json_path.parent
        session_folder = task_folder.parent
        user_folder = session_folder.parent

        try:
            u_idx = user_folder.name.split('_')[1]
            s_idx = session_folder.name.split('_')[2]
            t_idx = task_folder.name.split('_')[3]
        except (IndexError, AttributeError):
            continue

        try:
            with open(json_path, 'r') as f:
                old_data = json.load(f)

            # --- BUILD NEW ORDERED DICTIONARY ---
            new_data = {}
            
            # 1. image
            new_data["image"] = json_path.with_suffix('.png').name
            
            # 2. task
            new_data["task"] = t_idx
            
            # 3. session
            new_data["session"] = s_idx
            
            # 4. user
            new_data["user"] = u_idx
            
            # 5. task_type
            new_data["task_type"] = "9-point" if t_idx == "00" else "16-point"
            
            # 6. pos (Convert to Integers)
            if "pos" in old_data:
                # Ensuring x and y are integers
                pos_data = old_data["pos"]
                new_data["pos"] = {
                    "x": int(float(pos_data.get("x", 0))),
                    "y": int(float(pos_data.get("y", 0)))
                }
            
            # 7. gaze (intersection rename)
            if "gaze" in old_data:
                g_data = old_data["gaze"]
                if "destiny" in g_data:
                    g_data["intersection"] = g_data.pop("destiny")
                new_data["gaze"] = g_data
            
            # 8. hpe (facial_landmarks_2D rename)
            if "hpe" in old_data:
                h_data = old_data["hpe"]
                h_data.pop("method", None)
                if "facialLandmarks2D" in h_data:
                    h_data["facial_landmarks_2D"] = h_data.pop("facialLandmarks2D")
                new_data["hpe"] = h_data
                
            # # 9. eye_roi (Deep cleanup and 2d_point renames)
            # eye_source = old_data.get("eyePatch") or old_data.get("eye_roi") or old_data.get("eyeROI")
            # if eye_source:
            #     new_eye_roi = {}
            #     mapping = {"leftEye": "left_eye", "rightEye": "right_eye", 
            #                "left_eye": "left_eye", "right_eye": "right_eye"}
                
            #     for old_key, new_key in mapping.items():
            #         if old_key in eye_source:
            #             e_data = eye_source[old_key]
            #             if "inner" in e_data: e_data["2d_inner_point"] = e_data.pop("inner")
            #             if "outer" in e_data: e_data["2d_outer_point"] = e_data.pop("outer")
            #             # Remove specific unwanted fields
            #             for field in ["Mrot", "gaze", "gazeRaw", "angle", "location"]:
            #                 e_data.pop(field, None)
            #             new_eye_roi[new_key] = e_data
            #     new_data["eye_roi"] = new_eye_roi

            # 10. quality_assurance_metrics
            qa_source = old_data.get("QualityAssuranceMetrics") or old_data.get("quality_assurance_metrics")
            if qa_source:
                if "2DLandmarksInfo" in qa_source:
                    qa_source["2D_landmark_info"] = qa_source.pop("2DLandmarksInfo")
                new_data["quality_assurance_metrics"] = qa_source

            # 11. discard_info
            new_data["discard_info"] = old_data.get("discard_info", None)

            # Apply final rounding to everything else
            final_data = round_and_convert(new_data)

            if not DRY_RUN:
                with open(json_path, 'w') as f:
                    json.dump(final_data, f, indent=4)
            else:
                print(f"Processed: {json_path.name}")

        except Exception as e:
            print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    final_standardization(BASE_PATH)