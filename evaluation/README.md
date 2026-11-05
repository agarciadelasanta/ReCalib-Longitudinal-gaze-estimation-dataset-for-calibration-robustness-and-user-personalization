# Evaluation

Scripts to evaluate and fine-tune a third-party gaze estimation model, [ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) (Zhang et al., ECCV 2020), on ReCalib under the reuse scenarios of the paper. They use the data; they do not produce any part of it.

```bash
python -m pip install -r requirements.txt
```

The pretrained checkpoint is not stored in Git; `ckpt/readme.txt` says where to download it.

Face landmarks for the ETH-XGaze normalization come from dlib's 68-point shape predictor. It is not stored in Git either: `dlib_model.py` downloads it from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) into `evaluation/modules/` the first time it is needed and checks its SHA-256. The model is trained on the iBUG 300-W dataset, whose licence excludes commercial use; check that licence before using it.

| Script | What it does |
|---|---|
| `eth_xgaze_baseline.py` | Zero-shot ETH-XGaze over a ReCalib tree → CSV with the angular and on-screen error of every sample. Uses only the image, the camera intrinsics and the target (`pog_mm`) |
| `eth_xGaze_inf.py` | ETH-XGaze inference on one image (used by `visualization/visualize_sample.py`; runs on the bundled example) |
| `data_normalization.py` | Converts images and annotations into normalized HDF5 files (ETH-XGaze normalization). Set `INPUT_ROOT` to the dataset folder |
| `train_model.py` | Fine-tuning and evaluation, configured through its `CONFIG_VARS` dictionary (target participant, target session, calibration-only adaptation, …) |
| `extract_raw_labels.py` | Flattens the raw acquisition JSONs into one CSV with the column names of the release |

Switching between scenarios in `train_model.py`:

```python
CONFIG_VARS = {
    "target_user": "01",         # participant evaluated
    "target_session": "00",      # None for leave-one-user-out
    "session_calibration": True, # True: adapt on the 9-point calibration task only
    "batch_size": 32,
    "epochs": 20,
    "ckpt_dir": "./checkpoints",
}
```

Zero-shot baseline on the dataset:

```bash
python evaluation/eth_xgaze_baseline.py path/to/ReCalib --out baseline.csv --device auto
```
