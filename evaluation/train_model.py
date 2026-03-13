import argparse
import numpy as np
import torch
import h5py
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from pathlib import Path

# Import your provided components
from trainer import Trainer
from data_loader_h5 import GazeH5Dataset  # The H5 Dataset class we built
from utils import angular_error

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

H5_PATH = ROOT_DIR / 'temp' / 'full_dataset.h5'

class TrainingConfig:
    """Configuration object compatible with the Trainer class."""
    def __init__(self, **kwargs):
        self.is_train = kwargs.get("is_train", True)
        self.batch_size = kwargs.get("batch_size", 32)
        self.epochs = kwargs.get("epochs", 20)
        self.init_lr = kwargs.get("init_lr", 0.001)
        self.lr_patience = kwargs.get("lr_patience", 5)
        self.lr_decay_factor = kwargs.get("lr_decay_factor", 0.1)
        self.ckpt_dir = kwargs.get("ckpt_dir", "./checkpoints")
        self.print_freq = kwargs.get("print_freq", 10)
        self.use_gpu = torch.cuda.is_available()
        self.use_amp = True
        self.resume_ckpt = kwargs.get("resume_ckpt", None)

def get_logic_indices(target_user, target_session=None, session_calibration=False):
    """Filters the H5 metadata to find indices for your 3 core requirements."""
    with h5py.File(H5_PATH, 'r') as h5:
        users = h5['meta/user'][:].astype(str)
        sessions = h5['meta/session'][:].astype(str)
        tasks = h5['meta/task'][:].astype(str)
        
        if session_calibration:
            train_idx = np.where((users == target_user) & 
                           (sessions == target_session) & 
                           (tasks == "00"))[0]
            test_idx = np.where((users == target_user) & 
                           (sessions == target_session) & 
                           (tasks != "00"))[0]
            return train_idx, test_idx
        elif target_session:
            test_idx = np.where((users == target_user) & (sessions == target_session))[0]
            train_idx = np.where((users != target_user) | 
                                 ((users == target_user) & (sessions != target_session)))[0]
            return train_idx, test_idx
        else:
            test_idx = np.where(users == target_user)[0]
            train_idx = np.where(users != target_user)[0]
            
            return train_idx, test_idx
def split_train_val(indices, val_size=0.15, seed=42):
    """
    Performs a standard random split. 
    Images from the same user/session will likely appear in both sets.
    """
    print(f"[INFO] Performing random split with leakage (val_size={val_size}).")
    
    # train_test_split shuffles the data by default
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_size, 
        random_state=seed, 
        shuffle=True
    )
    
    return train_idx, val_idx

def split_train_val_no_leakage(indices, split_by='user', val_size=0.15):
    """Splits training indices ensuring no leakage between groups."""
    with h5py.File(H5_PATH, 'r') as h5:
        if split_by == 'user':
            groups = h5['meta/user'][indices].astype(str)
        else:
            u = h5['meta/user'][indices].astype(str)
            s = h5['meta/session'][indices].astype(str)
            groups = np.array([f"{user}_{sess}" for user, sess in zip(u, s)])

    unique_groups = np.unique(groups)
    
    if len(unique_groups) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        try:
            train_idx_in, val_idx_in = next(gss.split(indices, groups=groups))
            return indices[train_idx_in], indices[val_idx_in]
        except ValueError:
            pass

    print(f"[WARNING] Only {len(unique_groups)} group(s) found. Falling back to random split.")
    return train_test_split(indices, test_size=val_size, random_state=42)

def main():
    # --- HARDCODE YOUR TARGET HERE ---
    CONFIG_VARS = {
        "target_user": "01",
        "target_session": "00",  # Set to None for Leave-One-User-Out
        "session_calibration": True,  # Set to True for calibration-only subset
        "batch_size": 32,
        "epochs": 2,
        "ckpt_dir": "./checkpoints",
    }

    # 1. Logic Filtering
    train_idx, test_idx = get_logic_indices(
        CONFIG_VARS["target_user"], 
        CONFIG_VARS["target_session"], 
        CONFIG_VARS["session_calibration"]
    )

    # 2. Split for Validation (No Leakage)
    if not CONFIG_VARS["session_calibration"]:
        final_train_idx, val_idx = split_train_val_no_leakage(train_idx, split_by='user')
    else:
        final_train_idx, val_idx = split_train_val(train_idx, val_size=0.15, seed=42)
    
    print(f"Final Train Samples: {len(final_train_idx)}, Val Samples: {len(val_idx) if val_idx is not None else 0}, Test Samples: {len(test_idx)}")

    # 3. Create Loaders
    train_loader = DataLoader(GazeH5Dataset(H5_PATH, final_train_idx), 
                              batch_size=CONFIG_VARS["batch_size"], shuffle=True, num_workers=4)
    
    val_loader = None
    if val_idx is not None:
        val_loader = DataLoader(GazeH5Dataset(H5_PATH, val_idx), 
                                batch_size=CONFIG_VARS["batch_size"], shuffle=False, num_workers=4)
    
    test_loader = DataLoader(GazeH5Dataset(H5_PATH, test_idx, is_load_label=False),  
                             batch_size=CONFIG_VARS["batch_size"], shuffle=False, num_workers=4)

    # 4. Initialize Trainer
    # Pass the config object the Trainer expects
    config = TrainingConfig(
        is_train=True, 
        batch_size=CONFIG_VARS["batch_size"],
        epochs=CONFIG_VARS["epochs"],
        ckpt_dir=CONFIG_VARS["ckpt_dir"]
    )

    # Note: We provide val_loader for model selection (best.pth.tar)
    trainer = Trainer(config, train_loader, val_loader=val_loader)

    # 5. Start Training
    trainer.train()

    # Set the pre-trained model path to the best checkpoint for testing
    trainer.pre_trained_model_path = os.path.join(config.ckpt_dir, "best_ckpt.pth.tar")

    # 6. Final Evaluation on Test Set (Leave-One-Out set)
    # We swap the trainer's test loader manually for the final score
    trainer.test_loader = test_loader
    trainer.num_test = len(test_idx)
    print("\n[*] Evaluating on Leave-One-Out Test Set...")
    trainer.test()

if __name__ == "__main__":
    main()