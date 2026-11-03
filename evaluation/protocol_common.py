"""Shared pieces of the ReCalib evaluation protocol: scenarios, gaze conventions, H5 access, metrics.

Gaze angles follow the ETH-XGaze convention used by eth_xGaze_inf.py:
    vector = (-cos(pitch) sin(yaw), -sin(pitch), -cos(pitch) cos(yaw))
"""
import h5py
import numpy as np
import torch

CALIBRATION_TASK = "00"

# scenario -> hyperparameter group in protocol_config.json
SCENARIO_GROUPS = {
    "zeroshot": None,                    # (i) pretrained model, no training
    "louo": "domain",                    # (ii) domain adaptation, leave-one-user-out
    "personal": "personalization",       # (iii) user personalization, held-out session
    "calib": "calibration",              # (iv) session calibration from the pretrained model
    "personal_calib": "calibration",     # (iii) + (iv)
    "louo_calib": "calibration",         # (ii) + (iv)
}
# learning-curve scenarios personal_prev<k> belong to the personalization group


def scenario_group(scenario):
    if scenario.startswith("personal_prev"):
        return "personalization"
    return SCENARIO_GROUPS[scenario]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def vector_to_pitchyaw(v):
    """(N, 3) gaze vectors -> (N, 2) pitch, yaw in radians."""
    v = unit(v)
    return np.stack([-np.arcsin(np.clip(v[:, 1], -1, 1)), np.arctan2(-v[:, 0], -v[:, 2])], axis=1)


def pitchyaw_to_vector(py):
    """(N, 2) pitch, yaw in radians -> (N, 3) unit gaze vectors."""
    py = np.asarray(py, dtype=np.float64)
    p, y = py[:, 0], py[:, 1]
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], axis=1)


def angle_deg(a, b):
    """Row-wise angle between (N, 3) vectors in degrees."""
    c = np.sum(unit(a) * unit(b), axis=1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def screen_hit(origin, direction):
    """Intersection of the rays origin + t * direction with the screen plane z = 0 (camera frame)."""
    t = -origin[:, 2] / direction[:, 2]
    hit = origin + t[:, None] * direction
    hit[direction[:, 2] >= 0] = np.nan          # ray pointing away from the screen
    return hit


def load_meta(h5_path):
    """Per-row identifiers as numpy string arrays plus the validity flag."""
    with h5py.File(h5_path, "r") as h5:
        meta = {k: h5[f"meta/{k}"][:].astype(str) for k in ("user", "session", "task", "sample", "image")}
        meta["valid"] = h5["meta/valid"][:].astype(bool)
    return meta


def select(meta, clauses):
    """Rows matching any clause {"users": [...], "sessions": [...] | None, "tasks": calibration|test|all}."""
    mask = np.zeros(len(meta["user"]), dtype=bool)
    for c in clauses:
        m = np.isin(meta["user"], c["users"])
        if c.get("sessions") is not None:
            m &= np.isin(meta["session"], c["sessions"])
        if c["tasks"] == "calibration":
            m &= meta["task"] == CALIBRATION_TASK
        elif c["tasks"] == "test":
            m &= meta["task"] != CALIBRATION_TASK
        mask |= m
    return np.flatnonzero(mask & meta["valid"])


class H5Faces(torch.utils.data.Dataset):
    """Normalized face patches (stored BGR uint8) and their pitch/yaw label from the protocol H5."""

    def __init__(self, h5_path, indices, label_key=None):
        self.h5_path, self.indices, self.label_key = str(h5_path), np.asarray(indices), label_key
        self.h5 = None  # opened lazily, once per worker

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        row = int(self.indices[i])
        rgb = self.h5["face_patch"][row][:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = torch.from_numpy((rgb - IMAGENET_MEAN) / IMAGENET_STD)
        y = torch.from_numpy(self.h5[self.label_key][row].astype(np.float32)) if self.label_key else torch.zeros(2)
        return x, y, row
