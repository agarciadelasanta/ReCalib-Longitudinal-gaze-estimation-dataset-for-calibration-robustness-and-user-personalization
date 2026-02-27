from pathlib import Path
import sys

if not __package__:
    levels_to_project_dir = 1
    current_file_path = Path(__file__).resolve()
    top_package_containing_folder = current_file_path.parents[levels_to_project_dir + 1]
    sys.path.insert(0, str(top_package_containing_folder))
    package_name_parts = current_file_path.parts[-2 - levels_to_project_dir:-1]
    __package__ = '.'.join(package_name_parts)

import os
from tensorflow import norm, reduce_sum, acos, abs, train
from tensorflow.keras import optimizers
import keras_tuner as kt
import collections.abc
import numpy as np
import tensorflow as tf
import shutil
import math
from os.path import exists, splitext, islink, join, isdir



def checkIfisAValidPNGPair(pngFile):
    if not (pngFile.endswith('.png') or pngFile.endswith('.jpg')) :
        return False, "", ""

    auxFileJsonName = splitext(pngFile)[0]+'.json'

    if not (exists(auxFileJsonName)):
        return False, "", ""
    
    return True, pngFile, auxFileJsonName
