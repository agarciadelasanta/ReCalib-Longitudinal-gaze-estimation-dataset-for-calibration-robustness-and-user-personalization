"""Normalize every released frame once and store it in one H5 file for the evaluation protocol.

Each frame is normalized as in ETH-XGaze (6 landmarks -> solvePnP with the ETH face model ->
224x224 face patch at 600 mm, rotation R and face centre in the camera frame). Landmarks come from
dlib on the image by default, so the inputs are the image and the camera intrinsics only (as in the
zero-shot baseline, eth_xgaze_baseline.py).

Two training labels are stored, both as normalized (pitch, yaw):
  gaze_eth       unit(pog_mm - ETH face centre)        stimulus point only (default label)
  gaze_released  the released gaze.vector (MediaPipe origin -> pog_mm)
plus what is needed to score predictions in the camera frame (R, face_center, pog_mm,
origin_released). Frames whose normalization fails are kept with meta/valid = 0 and listed in
<out>_failures.csv. That includes frames whose ETH face centre lies outside --depth-range (default
300-900 mm from the camera): a wrong dlib detection makes solvePnP place the face metres away
(about 1% of frames in a test sample, at 2.2-7.6 m, against 0.37-0.83 m for all others), which
would corrupt both the label and the score.

    python evaluation/prepare_protocol_h5.py <release_root> --out protocol.h5 [--landmarks dlib]
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from eth_xGaze_inf import (MEDIAPIPE_ETH_LANDMARKS, dlib_eth_landmarks, load_camera_npz,
                           load_face_model, prepare_frame)
from protocol_common import unit, vector_to_pitchyaw

EVAL_DIR = Path(__file__).resolve().parent
META = ("user", "session", "task", "sample", "image", "task_type")


class ReleaseFrames(torch.utils.data.Dataset):
    def __init__(self, root, rows, face_model, camera, distortion, source, shape_predictor, depth_range):
        self.root, self.rows, self.depth_range = root, rows, depth_range
        self.face_model, self.camera, self.distortion = face_model, camera, distortion
        self.source, self.shape_predictor = source, shape_predictor
        self._dlib = None

    def __len__(self):
        return len(self.rows)

    def _landmarks(self, row, image):
        lm = None
        if self.source in ("dlib", "dlib_then_json"):
            if self._dlib is None:
                import dlib
                from dlib_model import ensure_shape_predictor
                self._dlib = (dlib.get_frontal_face_detector(), dlib.shape_predictor(ensure_shape_predictor(self.shape_predictor)))
            lm = dlib_eth_landmarks(*self._dlib, image)
        if lm is None and self.source in ("json", "dlib_then_json"):
            try:
                lm = np.array([[float(row[f"lm_{i}_x"]), float(row[f"lm_{i}_y"])] for i in MEDIAPIPE_ETH_LANDMARKS])
                lm = lm.reshape(6, 1, 2)
            except (KeyError, ValueError):
                lm = None
        return lm

    def __getitem__(self, i):
        row = self.rows[i]
        out = dict(i=i, ok=0, reason="", patch=np.zeros((224, 224, 3), np.uint8),
                   R=np.eye(3), face_center=np.full(3, np.nan))
        try:
            image = cv2.imread(str(self.root / row["image"]))
            if image is None:
                out["reason"] = "imread_failed"
                return out
            lm = self._landmarks(row, image)
            if lm is None:
                out["reason"] = "no_landmarks"
                return out
            patch, R, face_center = prepare_frame(image, lm, self.face_model, self.camera, self.distortion)
            if not self.depth_range[0] <= face_center[2] <= self.depth_range[1]:
                out["reason"] = f"implausible_face_depth: {face_center[2]:.0f} mm"
                return out
            out.update(ok=1, patch=patch, R=R, face_center=face_center)
        except Exception as exc:  # one bad frame must not stop the run
            out["reason"] = f"{type(exc).__name__}: {exc}"
        return out


def _worker_init(_):
    cv2.setNumThreads(1)  # many workers: avoid cv2 oversubscribing the CPU


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release_root", type=Path, help="folder with recalib_index.csv and user_XX/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discard-codes", default="0", help="comma-separated discard codes to include (0 = kept)")
    ap.add_argument("--landmarks", choices=("dlib", "json", "dlib_then_json"), default="dlib")
    ap.add_argument("--face-model", type=Path, default=EVAL_DIR / "face_model.txt")
    ap.add_argument("--camera-npz", type=Path)
    ap.add_argument("--shape-predictor", type=Path,
                    default=EVAL_DIR / "modules" / "shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--depth-range", type=float, nargs=2, default=(300.0, 900.0), metavar=("MIN", "MAX"),
                    help="valid ETH face-centre depth in mm (camera z)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame (smoke tests)")
    ap.add_argument("--limit", type=int, help="first N frames only (smoke tests)")
    args = ap.parse_args()

    codes = set(args.discard_codes.split(","))
    with open(args.release_root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["discard_code"] in codes][:: args.every]
    rows = rows[: args.limit] if args.limit else rows
    n = len(rows)
    camera, distortion = load_camera_npz(args.camera_npz)
    print(f"{n} frames (discard codes {sorted(codes)}), landmarks={args.landmarks}")

    pog = np.array([[float(r[f"gaze_pog_mm_{a}"]) for a in "xyz"] for r in rows])
    vec = np.array([[float(r[f"gaze_vector_{a}"]) for a in "xyz"] for r in rows])
    origin = np.array([[float(r[f"gaze_origin_mm_{a}"]) for a in "xyz"] for r in rows])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(
        ReleaseFrames(args.release_root, rows, load_face_model(str(args.face_model)), camera, distortion,
                      args.landmarks, str(args.shape_predictor), args.depth_range),
        batch_size=32, num_workers=args.num_workers, shuffle=False,
        worker_init_fn=_worker_init)
    failures = []
    start = time.time()
    with h5py.File(args.out, "w") as h5:
        patches = h5.create_dataset("face_patch", (n, 224, 224, 3), np.uint8, chunks=(1, 224, 224, 3),
                                    compression="lzf")
        R_all = np.tile(np.eye(3), (n, 1, 1))
        fc_all = np.full((n, 3), np.nan)
        valid = np.zeros(n, np.uint8)
        for batch in tqdm(loader, total=len(loader)):
            for k, i in enumerate(batch["i"].tolist()):
                if batch["ok"][k]:
                    patches[i] = batch["patch"][k].numpy()
                    R_all[i], fc_all[i], valid[i] = batch["R"][k].numpy(), batch["face_center"][k].numpy(), 1
                else:
                    failures.append((rows[i]["image"], batch["reason"][k]))

        # labels in the normalized space of each frame
        label_eth = unit(pog - fc_all)
        h5["gaze_eth"] = vector_to_pitchyaw(np.einsum("nij,nj->ni", R_all, label_eth)).astype(np.float32)
        h5["gaze_released"] = vector_to_pitchyaw(np.einsum("nij,nj->ni", R_all, vec)).astype(np.float32)
        h5["R"], h5["face_center"] = R_all.astype(np.float32), fc_all.astype(np.float32)
        h5["pog_mm"], h5["origin_released"] = pog.astype(np.float32), origin.astype(np.float32)
        dt = h5py.string_dtype("utf-8")
        cols = dict(user="user_id", session="session_id", task="task_id", sample="sample_id",
                    image="image", task_type="task_type")
        for k in META:
            h5.create_dataset(f"meta/{k}", data=[r[cols[k]] for r in rows], dtype=dt)
        h5["meta/discard_code"] = np.array([int(r["discard_code"]) for r in rows], np.int16)
        h5["meta/valid"] = valid
        h5.attrs.update(release_root=str(args.release_root), landmarks=args.landmarks,
                        discard_codes=args.discard_codes, depth_range=args.depth_range, fx=camera[0, 0], fy=camera[1, 1],
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"), n_failed=len(failures))

    fail_csv = args.out.with_name(args.out.stem + "_failures.csv")
    with open(fail_csv, "w", newline="") as fh:
        csv.writer(fh).writerows([("image", "reason"), *failures])
    print(f"wrote {args.out}: {n - len(failures)} normalized, {len(failures)} failed ({fail_csv.name}), "
          f"{(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
