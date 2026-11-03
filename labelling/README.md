# ReCalib pseudo-labelling

Stand-alone code that produces the ReCalib per-sample labels from the images, the acquisition setup and the screen targets. It reproduces the labelling step of the pipeline used for the released dataset, without its quality filtering.

```
image (PNG/JPG) + setup.json + pog_px  ->  JSON annotation (release schema)
```

There are two methods, which differ only in the depth of the head and gaze origin:

1. **One image.** The face is assumed to have the size of MediaPipe's canonical face.
2. **Several images of the same user.** The user's face size is estimated from the iris size over all of their images, and applied as a per-user factor (near 1) to the depth. This is how the ReCalib labels were made; the factors are in [`user_depth_scale.csv`](user_depth_scale.csv).

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

Identifiers can be added with `--user-id`, `--session-id`, `--task-id`, `--sample-id` and `--task-type`. With the `--user-id` of a ReCalib user, that user's factor from `user_depth_scale.csv` is applied; `--depth-scale` sets a factor explicitly (1 = one-image method).

**Many images**: a CSV with the columns `image`, `pog_px_x` and `pog_px_y`, plus the optional identifier columns. The release index has all of them, so the whole dataset can be relabelled:

```bash
python -m recalib_labelling.label_batch --csv ReCalib/recalib_index.csv --images-root ReCalib --setup ReCalib/setup.json --out-dir relabelled --depth-scale user_depth_scale.csv
```

Without `--depth-scale` every image is labelled with the one-image method.

**From Python**

```python
from recalib_labelling import Labeller, label_image

doc = label_image("image.png", (915, 10), "ReCalib/setup.json")      # one image

with Labeller("ReCalib/setup.json") as labeller:                     # many images, user 00's face scale
    doc = labeller.label("image.png", (915, 10), ids={"user_id": "00"}, depth_scale=0.99843)
```

**Check against the release**: relabels random released samples (with the factors of `user_depth_scale.csv`) and reports the differences per field.

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
| `head_pose.translation_mm` | origin of MediaPipe's canonical face, **re-expressed in the calibrated camera**. MediaPipe assumes a virtual camera (63° vertical field of view); the face centroid keeps its image position and apparent size, its depth is scaled by f_real / f_virtual and its lateral position is back-projected with the real intrinsics and distortion. Then multiplied by the user's face-scale factor |
| `gaze.origin_mm` | midpoint between the eyes (canonical eye corners 33, 133, 263, 362) placed with that pose, multiplied by the user's face-scale factor |
| `gaze.pog_mm` | screen target converted from physical screen pixels to the camera frame with the screen size and camera position of `setup.json` |
| `gaze.vector` | unit vector from `gaze.origin_mm` to `gaze.pog_mm` |
| `quality_assurance_metrics` | extreme and mean normalized coordinates of the 478 landmarks, the relative area of their bounding box, and the eye aspect ratio of each eye (Soukupová & Čech, 2016) and their mean |
| `discard_info` | `null`; only an image without a usable face (no face, or a required eye landmark outside the image) cannot be labelled and gets code 2 (`HPE_MODEL_ERROR`), as in the release |

The coordinate frame is the camera frame in mm: +x towards the right of the image, +y downwards, +z away from the camera. Quality filtering (eye-aspect-ratio threshold, landmark plausibility) and duplicate-frame removal are part of the dataset preparation and are not applied here.

## Face scale from several images of the same user

MediaPipe fits a canonical face of fixed size, so with one image the depth of the head and gaze origin is off by a factor that depends on the person: larger faces are placed too close, smaller ones too far. The visible iris diameter varies little between adults (about 11.7 mm). With several images of the same user, the factor is

```
factor = median over the user's images of  (f × 11.7 mm / iris diameter in px) / origin depth (one-image method)
```

The iris diameter is the mean of the four diameters of the two iris rings (landmarks 469–472 and 474–477), after lens undistortion. The factor multiplies the head translation and the gaze origin: a face s times larger at s times the distance gives the same image, so landmarks and rotation are unchanged. The gaze vector then points from the scaled origin to the same target.

```bash
python -m recalib_labelling.face_scale --labels labels_one_image.csv --setup ReCalib/setup.json --out user_depth_scale.csv
```

`--labels` takes labels made with the one-image method: a CSV in the `recalib_index.csv` format or a folder of JSON labels. The ReCalib factors (`user_depth_scale.csv`) were computed this way from the samples kept after quality filtering:

| User | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|---|
| Factor | 0.998 | 1.040 | 0.997 | 0.976 | 1.017 | 1.013 | 1.032 | 0.984 | 1.027 |

**Validation on EVE** (`validation/`): see `Review answer/eve_origin_validation.md` in the paper repository.

## Tests

```bash
python -m pytest tests
```
