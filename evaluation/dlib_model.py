"""dlib 68-point shape predictor used by the ETH-XGaze scripts: downloaded on first use.

The model is not stored in this repository. On first use it is downloaded from its official
source (dlib.net), checked, and cached outside the repository in ~/.cache/recalib_evaluation
(override with RECALIB_EVALUATION_CACHE); later runs use the cached copy. It is trained on the
iBUG 300-W dataset, whose licence excludes commercial use; the user is responsible for complying
with that licence.
"""
import bz2
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_PREDICTOR_SHA256 = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"
DEFAULT_CACHE = Path(os.environ.get("RECALIB_EVALUATION_CACHE", Path.home() / ".cache" / "recalib_evaluation"))
DEFAULT_SHAPE_PREDICTOR = DEFAULT_CACHE / "shape_predictor_68_face_landmarks.dat"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_shape_predictor(path=None, url=SHAPE_PREDICTOR_URL) -> str:
    """Return the path of the shape predictor.

    path: a local dlib model, used as it is (so another dlib model can be passed explicitly).
    None: the cached model, downloaded from dlib.net and checked (SHA-256) if it is not cached yet.
    """
    if path is not None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"dlib shape predictor not found: {path}")
        return str(path)
    model = DEFAULT_SHAPE_PREDICTOR
    if model.is_file():
        return str(model)
    model.parent.mkdir(parents=True, exist_ok=True)
    compressed = model.with_name(model.name + ".bz2.part")
    tmp = model.with_name(model.name + ".part")
    print(f"downloading {url} (dlib 68-point model; iBUG 300-W licence, non-commercial use only) to {model.parent}")
    try:
        urllib.request.urlretrieve(url, compressed)
        with bz2.open(compressed, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        digest = sha256(tmp)
        if digest != SHAPE_PREDICTOR_SHA256:
            raise RuntimeError(f"{url}: SHA-256 {digest} does not match the expected model {SHAPE_PREDICTOR_SHA256}")
        tmp.replace(model)
    finally:
        compressed.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return str(model)
