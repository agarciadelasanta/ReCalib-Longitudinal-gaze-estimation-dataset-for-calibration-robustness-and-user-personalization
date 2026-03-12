import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Standard normalization used in your previous data_loader.py
TRANS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # Scales pixels from [0, 255] to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class GazeH5Dataset(Dataset):
    """
    Dataset class that reads face patches and gaze labels from a single H5 file.
    Uses indices provided by the logic-filtering functions (Leave-One-Out).
    """
    def __init__(self, h5_path, indices, transform=TRANS, is_load_label=True):
        self.h5_path = h5_path
        self.indices = indices
        self.transform = transform
        self.is_load_label = is_load_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the local DataLoader index to the global index in the H5 file
        global_idx = self.indices[idx]
        
        # Opening in SWMR (Single Writer Multiple Reader) mode for faster access
        with h5py.File(self.h5_path, 'r', swmr=True) as hdf:
            # 1. Load the pre-normalized face patch (uint8)
            image = hdf['face_patch'][global_idx, :]
            
            # 2. Convert from BGR (OpenCV default) to RGB (PyTorch/PIL default)
            image = image[:, :, [2, 1, 0]]
            
            # 3. Apply transformations (To Tensor & Normalize)
            if self.transform:
                image = self.transform(image)

            # 4. Load Gaze Labels (Pitch and Yaw in Radians)
            if self.is_load_label:
                gaze_label = hdf['face_gaze'][global_idx, :].astype('float32')
                return image, gaze_label
            
            return image