"""Zero-shot ETH-XGaze baseline over a ReCalib tree (PNG + JSON pairs) -> CSV.

Inputs are the image and the camera intrinsics only: landmarks come from dlib on the image (default,
as in the official ETH-XGaze demo), and the face centre / depth come from solvePnP. The only label
used is pog_mm (the stimulus point). gaze.vector / gaze.origin_mm (MediaPipe estimates) are NOT used.
--landmarks json uses the dataset's MediaPipe landmarks instead (corrected ETH order), which is not
image-only.
The prediction is de-normalized (R^T) into the camera frame and scored two ways:
  * angular error against unit(pog_mm - ETH face centre);
  * on-screen error: predicted ray from the ETH face centre intersected with the screen plane
    z = 0, compared with pog_mm (mm).

Run from anywhere:
    py evaluation/eth_xgaze_baseline.py <dataset_root> --out baseline.csv --device auto
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from eth_xGaze_inf import (
    denormalize_gaze, dlib_eth_landmarks, landmarks_from_json, load_camera_npz,
    load_face_model, prepare_frame,
)
from model import gaze_network

EVAL_DIR = Path(__file__).resolve().parent
NAN3 = np.full(3, np.nan)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

COLUMNS = (
    ["user", "session", "task", "task_type", "image", "discarded", "ok", "fail_reason",
     "pred_pitch", "pred_yaw"]
    + [f"{n}_{a}" for n in ("pred_vec", "face_center", "label_eth") for a in "xyz"]
    + ["ang_err_eth_deg", "pog_pred_x_mm", "pog_pred_y_mm",
       "pog_gt_x_mm", "pog_gt_y_mm", "pog_err_mm"]
)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


def frame_metrics(pitchyaw, R, face_center, pog_mm):
    """Camera-frame prediction, label from the ETH face centre and the error figures for one frame."""
    pred = denormalize_gaze(pitchyaw, R).astype(np.float64)
    label_eth = unit(pog_mm - face_center)
    # Ray face_center + t * pred meets the screen plane z = 0 (screen lies at z = 0, camera frame).
    pog_pred = np.full(3, np.nan)
    if pred[2] < 0:
        pog_pred = face_center + (-face_center[2] / pred[2]) * pred
    return dict(
        pred=pred, label_eth=label_eth, pog_pred=pog_pred,
        ang_err_eth=angle_deg(pred, label_eth),
        pog_err=float(np.linalg.norm(pog_pred[:2] - pog_mm[:2])),
    )


class ReCalibFrames(torch.utils.data.Dataset):
    """Loads and normalizes one frame; a failure yields a zero patch and ok=False, never an exception."""

    def __init__(self, pairs, face_model, camera, distortion, source="dlib", shape_predictor=None):
        self.pairs, self.face_model, self.camera, self.distortion = pairs, face_model, camera, distortion
        self.source, self.shape_predictor = source, shape_predictor
        self._dlib = None  # built lazily, once per worker

    def __len__(self):
        return len(self.pairs)

    def _dlib_landmarks(self, image):
        if self._dlib is None:
            import dlib
            self._dlib = (dlib.get_frontal_face_detector(), dlib.shape_predictor(self.shape_predictor))
        return dlib_eth_landmarks(*self._dlib, image)

    def _landmarks(self, data, image):
        if self.source == "json":
            return landmarks_from_json(data)
        lm = self._dlib_landmarks(image)
        if lm is None and self.source == "dlib_then_json":
            lm = landmarks_from_json(data)
        return lm

    def __getitem__(self, i):
        png, js = self.pairs[i]
        item = dict(x=np.zeros((3, 224, 224), np.float32), R=np.eye(3), face_center=NAN3,
                    pog_mm=NAN3, ok=0, reason="", discarded=0, idx=i,
                    user="", session="", task="", task_type="", image=png.name)
        try:
            data = json.loads(js.read_text())
            item.update(user=str(data.get("user_id", "")), session=str(data.get("session_id", "")),
                        task=str(data.get("task_id", "")), task_type=str(data.get("task_type", "")),
                        discarded=int(data.get("discard_info") is not None),
                        pog_mm=np.asarray(data["gaze"]["pog_mm"], np.float64))
            image = cv2.imread(str(png))
            if image is None:
                item["reason"] = "imread_failed"
                return item
            lm = self._landmarks(data, image)
            if lm is None:
                item["reason"] = "no_landmarks"
                return item
            img_norm, R, face_center = prepare_frame(image, lm, self.face_model, self.camera, self.distortion)
            rgb = img_norm[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            item.update(x=(rgb - IMAGENET_MEAN) / IMAGENET_STD, R=R, face_center=face_center, ok=1)
        except Exception as exc:  # one bad frame must not stop a 150k-frame run
            item["reason"] = f"{type(exc).__name__}: {exc}"
        return item


def _worker_init(_):
    cv2.setNumThreads(1)  # many workers: avoid cv2 oversubscribing the CPU


def find_pairs(root, limit=None, every=1):
    pairs = [(p, p.with_suffix(".json")) for p in sorted(root.rglob("*.png")) if p.with_suffix(".json").exists()]
    pairs = pairs[::every]
    return pairs[:limit] if limit else pairs


def csv_row(batch, k, pitchyaw):
    row = {c: "" for c in COLUMNS}
    for c in ("user", "session", "task", "task_type", "image"):
        row[c] = batch[c][k]
    row.update(discarded=int(batch["discarded"][k]), ok=int(batch["ok"][k]), fail_reason=batch["reason"][k])
    if not batch["ok"][k]:
        return row
    m = frame_metrics(pitchyaw, batch["R"][k].numpy(), batch["face_center"][k].numpy(),
                      batch["pog_mm"][k].numpy())
    row.update(pred_pitch=pitchyaw[0], pred_yaw=pitchyaw[1])
    for name, v in (("pred_vec", m["pred"]), ("face_center", batch["face_center"][k].numpy()),
                    ("label_eth", m["label_eth"])):
        row.update({f"{name}_{a}": float(x) for a, x in zip("xyz", v)})
    row.update(ang_err_eth_deg=m["ang_err_eth"],
               pog_pred_x_mm=m["pog_pred"][0], pog_pred_y_mm=m["pog_pred"][1],
               pog_gt_x_mm=float(batch["pog_mm"][k][0]), pog_gt_y_mm=float(batch["pog_mm"][k][1]),
               pog_err_mm=m["pog_err"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset root (searched recursively for PNG + JSON pairs)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    ap.add_argument("--checkpoint", type=Path, default=EVAL_DIR / "ckpt" / "epoch_24_ckpt.pth.tar")
    ap.add_argument("--face-model", type=Path, default=EVAL_DIR / "face_model.txt")
    ap.add_argument("--camera-npz", type=Path, help="override docs/camera_intrinsics.npz")
    ap.add_argument("--landmarks", choices=("dlib", "json", "dlib_then_json"), default="dlib",
                    help="landmark source: dlib on the image (default, image-only), the JSON MediaPipe "
                         "landmarks, or dlib with the JSON as fallback when no face is found")
    ap.add_argument("--shape-predictor", type=Path,
                    default=EVAL_DIR / "modules" / "shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping frames already in it")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth pair (quick sample across the tree)")
    ap.add_argument("--limit", type=int, help="process only the first N pairs (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    camera, distortion = load_camera_npz(args.camera_npz)

    model = gaze_network(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state"])
    model.eval()

    pairs = find_pairs(args.root, args.limit, args.every)
    resumed = args.resume and args.out.exists()
    if resumed:
        # Keep only complete rows (a crash can leave a truncated last line), then skip what is done.
        with open(args.out, newline="") as f:
            rows = [r for r in csv.reader(f) if len(r) == len(COLUMNS)]
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        done = {r[COLUMNS.index("image")] for r in rows[1:]}
        pairs = [p for p in pairs if p[0].name not in done]
        print(f"resuming: {len(done)} frames already in {args.out}")
    print(f"{len(pairs)} PNG+JSON pairs, device={device}, fx={camera[0, 0]:.2f} fy={camera[1, 1]:.2f}")
    loader = torch.utils.data.DataLoader(
        ReCalibFrames(pairs, load_face_model(str(args.face_model)), camera, distortion,
                      args.landmarks, str(args.shape_predictor)),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    with open(args.out, "a" if resumed else "w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not resumed:
            writer.writeheader()
        for batch in tqdm(loader):
            pitchyaw = model(batch["x"].float().to(device)).cpu().numpy()
            for k in range(len(pitchyaw)):
                row = csv_row(batch, k, pitchyaw[k])
                n_fail += not row["ok"]
                writer.writerow(row)
            f.flush()
    print(f"Wrote {args.out} ({len(pairs)} rows, {n_fail} failures)")


if __name__ == "__main__":
    main()
