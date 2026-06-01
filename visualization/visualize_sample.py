import os
import sys
import random
import argparse
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

DEFAULT_TEST_IMAGE = "./example/07_00_02_img-040.png"
SETUP_CONFIG    = "./docs/setup_config.json"

# Model weights and parameters
SHAPE_PREDICTOR = "./evaluation/modules/shape_predictor_68_face_landmarks.dat"
FACE_MODEL      = "./visualization/face_model.txt"
CHECKPOINT      = "./evaluation/ckpt/epoch_24_ckpt.pth.tar"
CAMERA_INTRIN   = "./docs/camera_intrinsics.npz"


def main():
    parser = argparse.ArgumentParser(description="Visualize a sample from the ReCalib dataset with gaze prediction.")
    parser.add_argument("image_path", nargs='?', default=DEFAULT_TEST_IMAGE,
                        help=f"Path to the input PNG image. Defaults to: {DEFAULT_TEST_IMAGE}")
    args = parser.parse_args()

    # 1. Validate the input file pair (PNG + JSON)
    is_valid, png_path, json_path = checkIfisAValidPNGPair(args.image_path)
    
    if not is_valid:
        print(f"[Error] Invalid image or missing JSON for: {args.image_path}")
        return

    print(f"Processing: {png_path}")

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
    # Note: Read_gaze_data __init__ calls sceneReconstruction once.
    # It's called again below after loading setup specs. This is inefficient
    # but ensures the final state uses the global setup config.
    gaze_reader = Read_gaze_data(png_path, json_path)
    gaze_reader.loadSetupSpecs(SETUP_CONFIG)
    gaze_reader.sceneReconstruction()

    # 4. Predict Gaze & Inject into Reader
    gaze_prediction = estimator.predict_gaze_vector(png_path)
    gaze_reader.addGazePrediction(gaze_prediction)
    
    # 5. Generate Visualizations
    overlay_visualization(png_path, json_path, pred_vector=gaze_prediction)
    gaze_reader.plot3D()
    gaze_reader.plot2D()
    
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()