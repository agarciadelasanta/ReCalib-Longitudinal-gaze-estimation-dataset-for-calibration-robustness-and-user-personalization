"""Detect closed eyes with the open-eye-detector classifier, one row per labelled sample.

    python dataset_conversion/detect_blinks.py <release root> --model weights.149-0.01.hdf5 --out blinks.csv [--workers 6] [--resume]

Classifier: open-eye-detector (https://github.com/Chris10M/open-eye-detector, MIT licence, commit
64c301b3210856af2d1a5b94737d55b8f4198e17), models/weights.149-0.01.hdf5 (md5 269704d9773e5165b230cfe84232817c),
the model BegiBox used for blink detection. Each eye is classified separately from a grey-scale crop:
the bounding box of the eye's six contour landmarks, resized to 20 x 10 px and scaled to [-1, 1]
(keras mobilenet.preprocess_input), as in open-eye-detector's main.py. A sample is a blink when both
eyes are classified closed.

The eyes are located with MediaPipe FaceLandmarker, the model and settings used for the labels
(labelling/recalib_labelling), instead of dlib: the six contour landmarks of each eye are the ones of
the eye aspect ratio (subject's right eye 33, 160, 158, 133, 153, 144; left eye 362, 385, 387, 263,
373, 380), the MediaPipe counterparts of dlib's eye landmarks 36-41 and 42-47. As in BegiBox, the image
is first rotated about the eye centre so that the eye corners 33 and 263 are level.

As a check, the largest distance between the eye corners found here and those in the released JSON is
written (corner_diff_px, 0 when the detection is the same). Samples without labels (discard code 2) are
skipped. Nothing in the release is changed.
"""
import argparse
import csv
import hashlib
import math
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labelling"))
from recalib_labelling.model import ensure_model  # noqa: E402

MODEL_MD5 = "269704d9773e5165b230cfe84232817c"
RIGHT_EYE = (33, 160, 158, 133, 153, 144)     # subject's right eye (left in the image)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
CORNERS = ("33", "133", "362", "263")
COLUMNS = ["image", "user", "session", "task", "discard_code", "ok", "right_open_p", "left_open_p",
           "right_open", "left_open", "blink", "mean_ear", "corner_diff_px", "error"]
_STATE = {}


def _init(cnn_path, landmarker_path):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import mediapipe as mp
    import tensorflow as tf
    from keras.models import load_model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    cv2.setNumThreads(1)
    options = mp.tasks.vision.FaceLandmarkerOptions(        # as labelling/recalib_labelling/labeller.py
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_facial_transformation_matrixes=True)
    _STATE.update(mp=mp, landmarker=mp.tasks.vision.FaceLandmarker.create_from_options(options),
                  cnn=load_model(str(cnn_path), compile=False))


def ear(p, idx):
    a, b, c, d, e, f = (p[i] for i in idx)
    return (np.linalg.norm(b - f) + np.linalg.norm(c - e)) / (2 * np.linalg.norm(a - d))


def eye_patches(bgr, pts):
    """20 x 10 grey crops of the subject's right and left eye, image levelled on the eye corners."""
    angle = math.degrees(math.atan2(pts[263, 1] - pts[33, 1], pts[263, 0] - pts[33, 0]))
    centre = tuple(float(v) for v in (pts[33] + pts[263]) / 2)
    rot = cv2.getRotationMatrix2D(centre, angle, 1.0)
    gray = cv2.warpAffine(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), rot, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_LINEAR)
    levelled = pts @ rot[:, :2].T + rot[:, 2]
    patches = []
    for eye in (RIGHT_EYE, LEFT_EYE):
        x, y, w, h = cv2.boundingRect(np.round(levelled[list(eye)]).astype(np.int32))
        patch = gray[y:y + h, x:x + w]
        if patch.size == 0:
            raise ValueError("empty eye crop")
        patches.append(cv2.resize(patch, (20, 10)).astype(np.float32).reshape(10, 20, 1) / 127.5 - 1.0)
    return np.stack(patches)


def process(args):
    root, row = args
    out = dict(image=row["image"], user=row["user_id"], session=row["session_id"], task=row["task_id"],
               discard_code=row["discard_code"], ok=0)
    try:
        mp = _STATE["mp"]
        bgr = cv2.imread(str(root / row["image"]))
        h, w = bgr.shape[:2]
        res = _STATE["landmarker"].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            out["error"] = "no face"
            return out
        pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]])
        probs = _STATE["cnn"](eye_patches(bgr, pts), training=False).numpy()     # softmax, column 1 = open
        mesh = json.loads((root / row["image"]).with_suffix(".json").read_text(encoding="utf-8"))["head_pose"]["mediapipe_face_mesh_2d"]
        diff = max(np.hypot(*(pts[int(k)] - mesh[k])) for k in CORNERS if k in mesh)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    right, left = float(probs[0, 1]), float(probs[1, 1])
    out.update(ok=1, right_open_p=round(right, 4), left_open_p=round(left, 4), right_open=int(right > 0.5),
               left_open=int(left > 0.5), blink=int(right <= 0.5 and left <= 0.5),
               mean_ear=round(float((ear(pts, RIGHT_EYE) + ear(pts, LEFT_EYE)) / 2), 4), corner_diff_px=round(float(diff), 2))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release_root", type=Path)
    ap.add_argument("--model", type=Path, required=True, help="open-eye-detector models/weights.149-0.01.hdf5")
    ap.add_argument("--landmarker", help="face_landmarker.task to use instead of the cached download")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--codes", default="0,10", help="discard codes of the samples to classify")
    ap.add_argument("--limit", type=int, help="first N samples only (test)")
    ap.add_argument("--resume", action="store_true", help="keep the rows already in --out, process the rest")
    args = ap.parse_args()

    if md5(args.model) != MODEL_MD5:
        raise SystemExit(f"{args.model} is not the open-eye-detector model (md5 {MODEL_MD5})")
    landmarker = ensure_model(args.landmarker)
    with open(args.release_root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["discard_code"] in args.codes.split(",")]
    if args.limit:
        rows = rows[: args.limit]
    results = []
    if args.resume and args.out.exists():
        with open(args.out, newline="", encoding="utf-8") as fh:
            results = list(csv.DictReader(fh))
        done = {r["image"] for r in results}
        rows = [r for r in rows if r["image"] not in done]
        print(f"resuming: {len(results)} samples already in {args.out}, {len(rows)} to do", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as fh, Pool(args.workers, _init, (args.model, landmarker)) as pool:
        w = csv.DictWriter(fh, COLUMNS, restval="")
        w.writeheader()
        w.writerows(results)
        for n, res in enumerate(pool.imap_unordered(process, [(args.release_root, r) for r in rows], chunksize=8), 1):
            results.append(res)
            w.writerow(res)
            if n % 1000 == 0 or n == len(rows):
                fh.flush()
                print(f"{n}/{len(rows)} samples, {time.time() - start:.0f} s", flush=True)

    ok = [r for r in results if int(r["ok"])]
    print(f"\n{len(ok)}/{len(results)} samples classified")
    for title, codes in (("kept (code 0)", {"0"}), ("low EAR (code 10)", {"10"}), ("all labelled", {"0", "10"})):
        sel = [r for r in ok if r["discard_code"] in codes]
        if sel:
            blinks = sum(int(r["blink"]) for r in sel)
            one = sum(int(r["right_open"]) != int(r["left_open"]) for r in sel)
            print(f"{title:18s} n={len(sel):7d}  blink (both closed) {blinks:6d} ({100 * blinks / len(sel):.2f} %)  one eye closed {one:6d}")


if __name__ == "__main__":
    main()
