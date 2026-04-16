import os
import sys
import random
from pathlib import Path

# --- Local Imports ---
from read_gaze_data import Read_gaze_data
from utils import checkIfisAValidPNGPair
from sample_overlay_plots import overlay_visualization

# Add evaluation module to path for ETH-XGaze
sys.path.append('evaluation')
from eth_xGaze_inf import ETHXGazeEstimator

# ============================================================
# ⚙️ CONFIGURATION & PATHS
# ============================================================

TEST_IMAGE_PATH = "./example/07_00_02_img-040.png"
SETUP_CONFIG    = "./docs/setup_config.json"

# Model weights and parameters
SHAPE_PREDICTOR = "./evaluation/modules/shape_predictor_68_face_landmarks.dat"
FACE_MODEL      = "./visualization/face_model.txt"
CHECKPOINT      = "./evaluation/ckpt/epoch_24_ckpt.pth.tar"
CAMERA_INTRIN   = "./docs/camera_intrinsics.npz"


def get_random_png(folder_path):
    """Grabs a random .png file from a folder and its subfolders."""
    png_files = list(Path(folder_path).rglob("*.png"))
    return random.choice(png_files) if png_files else None


def main():
    # 1. Validate the input file pair (PNG + JSON)
    is_valid, png_path, json_path = checkIfisAValidPNGPair(TEST_IMAGE_PATH)
    
    if not is_valid:
        print(f"[Error] Invalid image or missing JSON for: {TEST_IMAGE_PATH}")
        return

    print(f"Processing: {TEST_IMAGE_PATH}")

    # 2. Initialize the ETH-XGaze Estimator
    estimator = ETHXGazeEstimator(
        shape_predictor_path=SHAPE_PREDICTOR,
        face_model_path=FACE_MODEL,
        ckpt_path=CHECKPOINT,
        camera_npz_path=CAMERA_INTRIN,
        camera_xml_path=None,
        device="auto",
    )

    # 3. Load Ground Truth Data & Reconstruct 3D Scene
    gaze_reader = Read_gaze_data(png_path, json_path)
    gaze_reader.loadSetupSpecs(SETUP_CONFIG)
    gaze_reader.sceneReconstruction()

    # 4. Predict Gaze & Inject into Reader
    gaze_prediction = estimator.predict_gaze_vector(TEST_IMAGE_PATH)
    gaze_reader.addGazePrediction(gaze_prediction)
    
    # 5. Generate Visualizations
    overlay_visualization(TEST_IMAGE_PATH, json_path)
    gaze_reader.plot3D()
    gaze_reader.plot2D()
    
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()