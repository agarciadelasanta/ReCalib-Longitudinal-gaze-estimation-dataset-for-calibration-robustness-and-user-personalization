from pathlib import Path
import os
import random
import sys
from read_gaze_data import Read_gaze_data

from utils import checkIfisAValidPNGPair
from sample_overlay_plots import overlay_visualization

sys.path.append('evaluation')
from eth_xGaze_inf import ETHXGazeEstimator

def get_random_png(folder_path):
    """
    Grab a random .png file from a folder and its subfolders.

    :param folder_path: Path to the root folder.
    :return: Path to the random .png file or None if no .png file is found.
    """
    png_files = []

    # Walk through the directory and collect all .png files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))

    # Return a random .png file if the list is not empty
    if png_files:
        return random.choice(png_files)
    else:
        return None


auxFileName = "./example/00_00_01_img-001.png"

validPair, auxFilePngName, auxFileJsonName = checkIfisAValidPNGPair(auxFileName)


est = ETHXGazeEstimator(
    shape_predictor_path="./evaluation/modules/shape_predictor_68_face_landmarks.dat",
    face_model_path="./visualization/face_model.txt",
    ckpt_path="./evaluation/ckpt/epoch_24_ckpt.pth.tar",
    camera_npz_path="./docs/camera_intrinsics.npz",
    camera_xml_path=None,
    device="auto",
)

if validPair:
    print(auxFileName)
    
    irisbondPatchJsonReader = Read_gaze_data(auxFilePngName, auxFileJsonName)
    irisbondPatchJsonReader.loadSetupSpecs("./docs/setup_config.json")
    irisbondPatchJsonReader.sceneReconstruction()
    eye3DPredictionCenter_cam = irisbondPatchJsonReader.eye3DPredictionCenter_cam
    pogPx_screen = irisbondPatchJsonReader.pogPx_screen
    pog_cm_cam = irisbondPatchJsonReader.pog_cm_cam

    gazePrediction = est.predict_gaze_vector(auxFileName)
    irisbondPatchJsonReader.addGazePrediction(gazePrediction)
    
    overlay_visualization(auxFileName, auxFileJsonName)
    
    irisbondPatchJsonReader.plot3D()
    irisbondPatchJsonReader.plot2D()
    input("Press Enter to continue...")
    #irisbondPatchJsonReader.plot2D()
else:
    print("Invalid pair")
    