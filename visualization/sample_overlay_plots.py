import json
import cv2
import numpy as np
from pathlib import Path

# --- Constants for Visualization ---
GAZE_VECTOR_SCALE = 450  # Length of the rendered gaze vector in pixels

# MediaPipe Landmark IDs for key points
LMK_RIGHT_EYE_OUTER = "33"
LMK_RIGHT_EYE_INNER = "133"
LMK_LEFT_EYE_OUTER = "263"
LMK_LEFT_EYE_INNER = "362"
LMK_NOSE_TIP = "4"
LMK_HEAD_CENTER = "168" # Approx. center of the head model


def overlay_visualization(image_path, json_path, pred_vector=None):
    """
    Visualizes 2D landmarks with index numbers, GT gaze, estimated vector, 
    and full metadata with large text.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # --- 1. Draw 2D Facial Landmarks and Index Numbers ---
    # Updated to the new 'head_pose' and 'mediapipe_face_mesh_2d' keys
    landmarks = data.get("head_pose", {}).get("mediapipe_face_mesh_2d", {})
    for lmk_id, coords in landmarks.items():
        x, y = int(coords[0]), int(coords[1])
        
        # Draw the landmark point
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        
        # Draw the index number (ID) next to the point
        # Offset the text slightly so it doesn't sit directly on the dot
        cv2.putText(img, str(lmk_id), (x + 8, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- 2. Identify Midpoint for Gaze Origin using Specific Landmark IDs ---
    try:
        # Use the head center landmark as the origin for the visualized vector
        start_point = tuple(map(int, landmarks[LMK_HEAD_CENTER]))
        
        # Highlight selected eye corners for visibility
        key_lmk_ids = [LMK_LEFT_EYE_INNER, LMK_RIGHT_EYE_INNER, LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER]
        for lmk_id in key_lmk_ids:
            pt = landmarks[lmk_id]
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (255, 255, 0), -1)
            
    except KeyError as e:
        print(f"Warning: Landmark {e} not found. Falling back to nose tip ({LMK_NOSE_TIP}).")
        nose = landmarks.get(LMK_NOSE_TIP, [img.shape[1]//2, img.shape[0]//2])
        start_point = (int(nose[0]), int(nose[1]))

    # --- 3. Draw Gaze Vectors ---
    gt_vector = data.get("gaze", {}).get("vector")

    if gt_vector:
        gt_end = (int(start_point[0] + gt_vector[0] * GAZE_VECTOR_SCALE),
                  int(start_point[1] + gt_vector[1] * GAZE_VECTOR_SCALE))
        cv2.arrowedLine(img, start_point, gt_end, (0, 0, 255), 5, tipLength=0.1)

    if pred_vector is not None:
        pred_end = (int(start_point[0] + pred_vector[0] * GAZE_VECTOR_SCALE),
                    int(start_point[1] + pred_vector[1] * GAZE_VECTOR_SCALE))
        cv2.arrowedLine(img, start_point, pred_end, (255, 0, 0), 5, tipLength=0.1)

    # --- 4. UI: Metadata (Top Left) ---
    # Updated identifiers to user_id, session_id, and task_id
    meta_text = [
        f"User: {data.get('user_id')}",
        f"Session: {data.get('session_id')}",
        f"Task: {data.get('task_id')} ({data.get('task_type')})"
    ]
    
    for i, text in enumerate(meta_text):
        cv2.putText(img, text, (30, 60 + (i * 50)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # --- 5. UI: Legend (Top Right) ---
    # Define legend box properties
    legend_w, legend_h = 430, 180
    legend_x, legend_y = img.shape[1] - legend_w - 20, 20
    
    # Create a semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Define text entries and their colors
    legend_items = [("Landmarks", (0, 255, 0)), ("GT Gaze", (0, 0, 255))]
    if pred_vector is not None:
        legend_items.append(("Pred Gaze", (255, 0, 0)))
    
    # Draw legend title and items
    cv2.putText(img, "LEGEND", (legend_x + 15, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    for i, (text, color) in enumerate(legend_items):
        cv2.putText(img, text, (legend_x + 15, legend_y + 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # --- Rendering ---
    cv2.namedWindow("Gaze & Landmark Visualizer", cv2.WINDOW_NORMAL)
    cv2.imshow("Gaze & Landmark Visualizer", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()