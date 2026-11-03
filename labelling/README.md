# ReCalib pseudo-labelling

Stand-alone code that produces the ReCalib per-sample labels from a single image, the acquisition setup and the screen target. It reproduces the labelling step of the pipeline used for the released dataset, without its quality filtering.

```
image (PNG/JPG) + setup.json + pog_px  ->  JSON annotation (release schema)
```

## Installation

```bash
pip install -r requirements.txt
```

The MediaPipe FaceLandmarker model (`face_landmarker.task`, Apache-2.0) is downloaded from Google on first use and cached in `~/.cache/recalib_labelling` (override with `RECALIB_LABELLING_CACHE`). Its SHA-256 is checked against the file used to label ReCalib (`64184e22…0bc9ff`, model version 1, float16). The labels were produced with MediaPipe 0.10.21.

Run the commands below from this folder (`labelling/`).

## Usage

**One image**

```bash
python -m recalib_labelling.label_image path/to/image.png --setup ReCalib/setup.json --pog-px 915 10 --out label.json
```

Identifiers can be added with `--user-id`, `--session-id`, `--task-id`, `--sample-id` and `--task-type`.

**Many images**: a CSV with the columns `image`, `pog_px_x` and `pog_px_y`, plus the optional identifier columns. The release index has all of them, so the whole dataset can be relabelled:

```bash
python -m recalib_labelling.label_batch --csv ReCalib/recalib_index.csv --images-root ReCalib --setup ReCalib/setup.json --out-dir relabelled
```

**From Python**

```python
from recalib_labelling import Labeller, label_image

doc = label_image("image.png", (915, 10), "ReCalib/setup.json")      # one image

with Labeller("ReCalib/setup.json") as labeller:                     # many images
    doc = labeller.label("image.png", (915, 10), ids={"user_id": "00"})
```

**Check against the release**: relabels random released samples and reports the differences per field.

```bash
python -m recalib_labelling.verify_release --release-root ReCalib --n 200
```

On 200 random samples the labels are reproduced exactly. Landmarks, head rotation, screen target and quality metrics are identical; gaze origin and head translation differ by at most 0.0002 mm (rounding of the intrinsics stored in `setup.json`).

## What is computed

| Output field | How |
|---|---|
| `head_pose.mediapipe_face_mesh_2d` | 18 FaceLandmarker landmarks (eye corners, irises, nose, mouth corners) in integer pixels; landmarks outside the image are omitted |
| `head_pose.rotation_rad` | MediaPipe facial transformation matrix converted to OpenCV axes; extrinsic Euler angles `zyx` = [roll, yaw, pitch] |
| `head_pose.rotvec_rad` | same rotation (orthonormalized) as a rotation vector |
| `head_pose.translation_mm` | origin of MediaPipe's canonical face, **re-expressed in the calibrated camera**. MediaPipe assumes a virtual camera (63° vertical field of view); the face centroid keeps its image position and apparent size, its depth is scaled by f_real / f_virtual and its lateral position is back-projected with the real intrinsics and distortion |
| `gaze.origin_mm` | midpoint between the eyes (canonical eye corners 33, 133, 263, 362) placed with that pose |
| `gaze.pog_mm` | screen target converted from physical screen pixels to the camera frame with the screen size and camera position of `setup.json` |
| `gaze.vector` | unit vector from `gaze.origin_mm` to `gaze.pog_mm` |
| `quality_assurance_metrics` | extreme and mean normalized coordinates of the 478 landmarks, the relative area of their bounding box, and the eye aspect ratio of each eye (Soukupová & Čech, 2016) and their mean |
| `discard_info` | `null`; only an image without a usable face (no face, or a required eye landmark outside the image) cannot be labelled and gets code 2 (`HPE_MODEL_ERROR`), as in the release |

The coordinate frame is the camera frame in mm: +x towards the right of the image, +y downwards, +z away from the camera. Quality filtering (eye-aspect-ratio threshold, landmark plausibility) and duplicate-frame removal are part of the dataset preparation and are not applied here.

## Tests

```bash
python -m pytest tests
```
