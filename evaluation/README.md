# Evaluation

Scripts to evaluate and fine-tune a third-party gaze estimation model, [ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) (Zhang et al., ECCV 2020), on ReCalib under the reuse scenarios of the paper. They use the data; they do not produce any part of it.

```bash
python -m pip install -r requirements.txt
```

The pretrained checkpoint is not stored in Git; `ckpt/readme.txt` says where to download it.

Face landmarks for the ETH-XGaze normalization come from dlib's 68-point shape predictor. It is not stored in Git: the first time it is needed, `dlib_model.py` downloads it from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), checks its SHA-256 and caches it outside the repository in `~/.cache/recalib_evaluation` (override with `RECALIB_EVALUATION_CACHE`; `--shape-predictor` uses another local copy). The model is trained on the iBUG 300-W dataset, whose licence excludes commercial use; check that licence before using it.

The evaluation protocol of the paper (zero-shot, session calibration, personalization and leave-one-user-out, each scored on the test tasks of held-out sessions; Table 6) is described step by step in [`PROTOCOL.md`](PROTOCOL.md).

| Script | What it does |
|---|---|
| `prepare_protocol_h5.py` | ETH-XGaze normalization of the released frames into one HDF5 file used by the protocol |
| `make_protocol_folds.py` | Training, validation and test sets of every fold of the protocol (JSON manifest) |
| `run_protocol_fold.py` | Fine-tunes and scores one or more folds of the manifest |
| `summarize_protocol.py` | Per-scenario and per-user errors, paired contrasts between scenarios |
| `cross_session_eval.py`, `summarize_cross_session.py` | A session-calibrated model scored on the user's other sessions |
| `eth_origin_check.py` | Compares the released gaze origin with the ETH-XGaze eye centre (per frame) |
| `plot_label_consistency.py` | Screen intersections of the released gaze vectors and the ETH-XGaze predictions |
| `eth_xgaze_baseline.py` | Zero-shot ETH-XGaze over a ReCalib tree → CSV with the angular and on-screen error of every sample. Uses only the image, the camera intrinsics and the target (`pog_mm`) |
| `eth_xGaze_inf.py` | ETH-XGaze inference on one image (used by `visualization/visualize_sample.py`; runs on the bundled example) |
| `data_normalization.py` | Converts images and annotations into normalized HDF5 files (ETH-XGaze normalization). Set `INPUT_ROOT` to the dataset folder |
| `train_model.py` | Fine-tuning and evaluation, configured through its `CONFIG_VARS` dictionary (target participant, target session, calibration-only adaptation, …) |
| `extract_raw_labels.py` | Flattens the raw acquisition JSONs (before conversion to the release; not public) into one CSV with the column names of the earlier summary `dataset_summary_recalib.csv`, not those of `recalib_index.csv` |

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
