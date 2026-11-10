"""MediaPipe FaceLandmarker model used for the ReCalib labels: download and integrity check."""
import hashlib
import os
import urllib.request
from pathlib import Path

# Official Google model, version 1 (float16). Byte-identical to the file used to label ReCalib.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_CACHE = Path(os.environ.get("RECALIB_RECONSTRUCTION_CACHE", Path.home() / ".cache" / "recalib_reconstruction"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path=None) -> Path:
    """Return the path of a verified face_landmarker.task, downloading it if needed.

    path: explicit model file to use (verified, never downloaded). By default the model is
    cached in ~/.cache/recalib_reconstruction (override with RECALIB_RECONSTRUCTION_CACHE).
    """
    if path is not None:
        model = Path(path)
        if not model.is_file():
            raise FileNotFoundError(model)
    else:
        model = DEFAULT_CACHE / "face_landmarker.task"
        if not model.is_file():
            model.parent.mkdir(parents=True, exist_ok=True)
            tmp = model.with_suffix(".part")
            print(f"downloading {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(model)
    digest = sha256(model)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"{model}: SHA-256 {digest} does not match the ReCalib model {MODEL_SHA256}")
    return model
