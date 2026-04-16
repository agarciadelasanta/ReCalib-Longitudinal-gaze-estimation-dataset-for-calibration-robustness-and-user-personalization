import json
from pathlib import Path
import cv2
import dlib
import h5py
import numpy as np
from tqdm import tqdm
from imutils import face_utils

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

INPUT_ROOT = Path(r'E:\user_personalization_tagged\Mamu\PRUEBAS')
OUTPUT_H5 = ROOT_DIR / 'temp' / 'full_dataset.h5'

# Make sure these paths point to your actual local files
CAMERA_NPZ = str(ROOT_DIR / 'docs' / 'camera_intrinsics.npz')
SHAPE_PREDICTOR = str(SCRIPT_DIR / 'modules' / 'shape_predictor_68_face_landmarks.dat')
FACE_MODEL = str(SCRIPT_DIR / 'face_model.txt')
# ---------------------

def gaze_vector_to_pitchyaw(v):
    v = np.asarray(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    pitch = -np.arcsin(v[1])
    yaw = np.arctan2(-v[0], -v[2])
    return np.array([pitch, yaw], dtype=np.float32)

def load_camera_npz(npz_path):
    data = np.load(npz_path)
    return np.asarray(data["mtx"], dtype=np.float64), np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1)

def estimateHeadPose(landmarks_2d_6, facePts_3d_6, K, dist):
    _, rvec, tvec = cv2.solvePnP(facePts_3d_6, landmarks_2d_6, K, dist, flags=cv2.SOLVEPNP_EPNP)
    _, rvec, tvec = cv2.solvePnP(facePts_3d_6, landmarks_2d_6, K, dist, rvec, tvec, True)
    return rvec, tvec

def normalizeData_face(img, face_model_6pts, landmarks_2d_6, hr, ht, K, out_size=224):
    focal_norm, distance_norm = 960, 600
    roiSize = (out_size, out_size)
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = (hR @ face_model_6pts.T) + ht
    
    # Calculate face center based on eye and nose points
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
    
    z_scale = distance_norm / (np.linalg.norm(face_center) + 1e-9)
    cam_norm = np.array([[focal_norm, 0, out_size/2], [0, focal_norm, out_size/2], [0, 0, 1.0]])
    S = np.diag([1.0, 1.0, z_scale])
    
    forward = (face_center / (np.linalg.norm(face_center) + 1e-9)).reshape(3)
    down = np.cross(forward, hR[:, 0]); down /= (np.linalg.norm(down) + 1e-9)
    right = np.cross(down, forward); right /= (np.linalg.norm(right) + 1e-9)
    R = np.c_[right, down, forward].T
    
    W = cam_norm @ S @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(img, W, roiSize), R

def process_dataset():
    # Initialize skip counters
    n_skip = 0
    
    K, dist = load_camera_npz(CAMERA_NPZ)
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    
    # face model 3D (6 points) compatible with ETH-XGaze
    face_model_load = np.loadtxt(FACE_MODEL)
    landmark_use = [20, 23, 26, 29, 15, 19] 
    face_model_6pts = face_model_load[landmark_use, :].astype(np.float64)
    facePts_3d_6 = face_model_6pts.reshape(6, 1, 3)

    print("Scanning directory structure...")
    items = []
    for img_p in INPUT_ROOT.rglob("*.png"):
        js_p = img_p.with_suffix(".json")
        if js_p.exists():
            items.append((img_p, js_p))
    
    print(f"Found {len(items)} samples. Initializing H5 file...")
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)

    total_samples = len(items)

    with h5py.File(str(OUTPUT_H5), "w") as h5:
        face_patch = h5.create_dataset("face_patch", shape=(total_samples, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype=np.uint8, chunks=(1, 224, 224, 3), compression="lzf")
        face_gaze = h5.create_dataset("face_gaze", shape=(total_samples, 2), maxshape=(None, 2), dtype=np.float32, chunks=(1024, 2))
        
        dt = h5py.string_dtype(encoding="utf-8")
        m_user = h5.create_dataset("meta/user", (total_samples,), maxshape=(None,), dtype=dt)
        m_session = h5.create_dataset("meta/session", (total_samples,), maxshape=(None,), dtype=dt)
        m_task = h5.create_dataset("meta/task", (total_samples,), maxshape=(None,), dtype=dt)
        m_img_name = h5.create_dataset("meta/image_name", (total_samples,), maxshape=(None,), dtype=dt)

        curr_idx = 0

        for img_path, json_path in tqdm(items):
            try:
                with open(json_path, "r") as f:
                    j = json.load(f)

                # --- DISCARD CHECK ---
                # Skip the sample if discard_info is NOT null
                if j.get("discard_info") is not None:
                    print(f"Discarding sample {img_path} due to discard_info: {j['discard_info']}")
                    n_skip += 1
                    continue

                img = cv2.imread(str(img_path))
                if img is None: continue

                # --- LANDMARK EXTRACTION ---
                hpe = j.get("head_pose", {})
                lmk_dict = hpe.get("mediapipe_face_mesh_2d", None)
                landmarks_2d_6 = None

                if lmk_dict is not None:
                    try:
                        # Map MediaPipe indices to 6 points: Left/Right eyes and nose
                        points = [
                            lmk_dict["263"], # Left eye outer
                            lmk_dict["362"], # Left eye inner
                            lmk_dict["133"], # Right eye inner
                            lmk_dict["33"],  # Right eye outer
                            lmk_dict["129"], # Nose corner left
                            lmk_dict["358"]  # Nose corner right
                        ]
                        landmarks_2d_6 = np.array(points, dtype=np.float64).reshape(6, 1, 2)
                    except KeyError:
                        pass # Fall through to dlib if dictionary keys are missing

                if landmarks_2d_6 is None:
                    # Fallback to dlib detection
                    dets = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
                    if len(dets) == 0:
                        n_skip += 1
                        continue
                    shape = predictor(img, dets[0])
                    shape68 = face_utils.shape_to_np(shape).astype(np.float64)
                    landmarks_2d_6 = shape68[[36, 39, 42, 45, 31, 35], :].reshape(6, 1, 2)

                # --- POSE ESTIMATION & NORMALIZATION ---
                hr, ht = estimateHeadPose(landmarks_2d_6, facePts_3d_6, K, dist)
                img_norm, R = normalizeData_face(img, face_model_6pts, landmarks_2d_6, hr, ht, K)

                # --- GAZE VECTOR NORMALIZATION ---
                # Retrieve the original gaze vector
                gaze_vec = np.array(j["gaze"]["vector"], dtype=np.float64).reshape(3, 1)
                # Rotate the gaze vector into the normalized camera space
                gaze_norm = (R @ gaze_vec).reshape(3)
                # Convert the normalized gaze to pitch/yaw
                gaze_pitchyaw = gaze_vector_to_pitchyaw(gaze_norm)

                # --- APPEND TO H5 ---
                face_patch[curr_idx] = img_norm
                face_gaze[curr_idx] = gaze_pitchyaw
                m_user[curr_idx] = str(j["user_id"])
                m_session[curr_idx] = str(j["session_id"])
                m_task[curr_idx] = str(j["task_id"])
                m_img_name[curr_idx] = img_path.name
                
                curr_idx += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # Resize datasets down to the actual number of generated valid samples
        for ds in [face_patch, face_gaze, m_user, m_session, m_task, m_img_name]:
            ds.resize((curr_idx,) + ds.shape[1:])
    
    print(f"Processing finished. Total skipped: {n_skip}")

if __name__ == "__main__":
    process_dataset()