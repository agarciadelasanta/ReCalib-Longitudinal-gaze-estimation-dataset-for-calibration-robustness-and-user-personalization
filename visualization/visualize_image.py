from pathlib import Path
import os
import random

from read_gaze_data import Read_gaze_data

from scene_operations import Screen
from utils import checkIfisAValidPNGPair


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


folder_path = R"example\input"
auxFileName = get_random_png(folder_path)
auxFileName = r"C:\Projects\ReCalib-A-multi-session-gaze-dataset-for-calibration-robustness-and-user-adaptation\example\input\sample.png"

validPair, auxFilePngName, auxFileJsonName = checkIfisAValidPNGPair(auxFileName)


if validPair:
    print(auxFileName)
    irisbondPatchJsonReader = Read_gaze_data(auxFilePngName, auxFileJsonName)

    irisbondPatchJsonReader.sceneReconstruction()

    irisbondPatchJsonReader.plot3D()
    irisbondPatchJsonReader.plot2D()
    input("Press Enter to continue...")
    #irisbondPatchJsonReader.plot2D()
    