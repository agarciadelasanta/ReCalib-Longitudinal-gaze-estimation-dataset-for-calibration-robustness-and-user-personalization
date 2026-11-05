"""dlib 68-point shape predictor used by the ETH-XGaze scripts: download on first use.

The model is not stored in this repository. It is trained on the iBUG 300-W dataset, whose
licence excludes commercial use, so it is downloaded from its official source (dlib.net) and
the user is responsible for complying with that licence.
"""
import bz2
import hashlib
import shutil
import urllib.request
from pathlib import Path

SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_PREDICTOR_SHA256 = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"
DEFAULT_SHAPE_PREDICTOR = Path(__file__).resolve().parent / "modules" / "shape_predictor_68_face_landmarks.dat"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_shape_predictor(path=None, url=SHAPE_PREDICTOR_URL) -> str:
    """Return the path of the shape predictor, downloading and checking it if the file is missing.

    path: where the model is (or should be) stored; default evaluation/modules/.
    An existing file is used as it is, so another dlib model can be passed explicitly.
    """
    model = Path(path) if path is not None else DEFAULT_SHAPE_PREDICTOR
    if model.is_file():
        return str(model)
    model.parent.mkdir(parents=True, exist_ok=True)
    compressed = model.with_name(model.name + ".bz2.part")
    tmp = model.with_name(model.name + ".part")
    print(f"downloading {url} (dlib 68-point model; iBUG 300-W licence, non-commercial use only)")
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
