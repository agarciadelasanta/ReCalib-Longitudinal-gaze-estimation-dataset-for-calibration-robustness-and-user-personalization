import json
import cv2
import numpy as np
from pathlib import Path

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
    landmarks = data.get("hpe", {}).get("facial_landmarks_2D", {})
    for lmk_id, coords in landmarks.items():
        x, y = int(coords[0]), int(coords[1])
        
        # Draw the landmark point
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        
        # Draw the index number (ID) next to the point
        # Offset the text slightly so it doesn't sit directly on the dot
        # cv2.putText(img, str(lmk_id), (x + 8, y - 8), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- 2. Identify Midpoint for Gaze Origin using Specific Landmark IDs ---
    # MediaPipe mapping: 362/133 (inner corners), 263/33 (outer corners)
    try:
        l_inner = landmarks["362"]
        r_inner = landmarks["133"]
        l_outer = landmarks["263"]
        r_outer = landmarks["33"]
        middle = landmarks["168"]

        start_point = middle
        
        # Highlight selected eye corners for visibility
        for pt in [l_inner, r_inner, l_outer, r_outer]:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (255, 255, 0), -1)
            
    except KeyError as e:
        print(f"Warning: Landmark {e} not found. Falling back to nose tip (ID 4).")
        nose = landmarks.get("4", [img.shape[1]//2, img.shape[0]//2])
        start_point = (int(nose[0]), int(nose[1]))

    # --- 3. Draw Gaze Vectors ---
    scale = 450 
    gt_vector = data.get("gaze", {}).get("vector")

    if gt_vector:
        gt_end = (int(start_point[0] + gt_vector[0] * scale),
                  int(start_point[1] + gt_vector[1] * scale))
        cv2.arrowedLine(img, start_point, gt_end, (0, 0, 255), 5, tipLength=0.1)

    if pred_vector is not None:
        pred_end = (int(start_point[0] + pred_vector[0] * scale),
                    int(start_point[1] + pred_vector[1] * scale))
        cv2.arrowedLine(img, start_point, pred_end, (255, 0, 0), 5, tipLength=0.1)

    # --- 4. UI: Metadata (Top Left) ---
    meta_text = [
        f"User: {data.get('user')}",
        f"Session: {data.get('session')}",
        f"Task: {data.get('task')} ({data.get('task_type')})"
    ]
    
    for i, text in enumerate(meta_text):
        cv2.putText(img, text, (30, 60 + (i * 50)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # --- 5. UI: Legend (Top Right) ---
    w = img.shape[1]
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 450, 20), (w - 20, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, "LEGEND", (w - 420, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img, "Landmarks", (w - 420, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, "GT Gaze", (w - 420, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    if pred_vector is not None:
        cv2.putText(img, "Pred Gaze", (w - 420, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # --- Rendering ---
    cv2.namedWindow("Gaze & Landmark Visualizer", cv2.WINDOW_NORMAL)
    cv2.imshow("Gaze & Landmark Visualizer", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()