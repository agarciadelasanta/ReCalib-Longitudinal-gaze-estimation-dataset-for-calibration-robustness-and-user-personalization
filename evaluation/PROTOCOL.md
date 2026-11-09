# Evaluation protocol: personalization and session calibration

Runs the reuse scenarios of the Usage Notes (Figure 5) with the public ETH-XGaze model and reports, per held-out session, how much each level of adaptation helps.

| | No session calibration | Session calibration (task 00 of the session) |
|---|---|---|
| No personalization (ETH-XGaze) | `zeroshot` | `calib` |
| Personalization (the user's other sessions) | `personal` | `personal_calib` |
| Domain control (the other users, run last) | `louo` | `louo_calib` |

- Personalization is needed if `zeroshot → personal` and `calib → personal_calib` improve.
- Session calibration is needed if `personal → personal_calib` improves.
- Session-to-session drift is the spread of a user's session errors.

Test sets are always the 16-point test tasks of the held-out session. Everything follows the official ETH-XGaze repository (github.com/xucong-zhang/ETH-XGaze): face detection with dlib's HOG detector (`get_frontal_face_detector`, upsampling 1), the 6 landmarks 36, 39, 42, 45, 31, 35, EPnP + iterative solvePnP with the ETH face model, normalization to 224×224 at 600 mm (focal 960), ImageNet input normalization, and fine-tuning with the defaults of its training code (Adam, lr 1e-4, batch 50, 25 epochs, StepLR every 10 epochs × 0.1, L1 loss on normalized pitch/yaw, whole network, BatchNorm in train mode; `protocol_config.json`). Only the camera intrinsics are ReCalib's, and training stops early: after 3 epochs without improvement of the validation error, keeping the best weights (at most the official 25 epochs). Validation never comes from the held-out session. For personalization it is one held-out training session of the user (its test tasks). For leave-one-user-out it is one held-out training user. For calibration it is 2 of the 9 calibration targets; the model is then retrained from its start weights on all 9 targets for the selected number of epochs. The one addition is the face-depth check below, which excludes frames instead of changing the method.

## Inputs and labels

- Inputs: the image and the camera intrinsics (`docs/camera_intrinsics.npz`). Six landmarks from dlib give the head pose (solvePnP with the ETH face model) and the ETH normalization (224×224 face, 600 mm).
- Label (default `gaze_eth`): unit(`pog_mm` − ETH face centre). It uses only the stimulus point, as in the zero-shot baseline (`eth_xgaze_baseline.py`). `--label gaze_released` uses the released `gaze.vector` instead.
- Frames: kept frames only (discard code 0). Frames where dlib finds no face, or where the ETH face centre is outside 300–900 mm from the camera (a wrong detection), are marked invalid and listed in `<h5>_failures.csv`. In a 1,000-frame sample these were 4.4% and about 1%.

## Steps

Requirements: `pip install -r requirements.txt`, plus `evaluation/ckpt/epoch_24_ckpt.pth.tar`. The dlib 68-point shape predictor is downloaded from dlib.net into `evaluation/modules/` on first use (see `evaluation/README.md`). Run from the repo root.

**1. Normalize the release once** (CPU, dlib on the full 1920×1080 image; about 2 s per frame per worker on a laptop CPU):

```bash
python evaluation/prepare_protocol_h5.py /data/ReCalib --out /data/protocol.h5 --num-workers 16
```

**2. Write the folds** (seconds). `--test-sessions` sets how many held-out sessions are used in total, spread over the users and over time. The same sessions are used in every scenario. Training sets always use all of the user's other sessions.

```bash
python evaluation/make_protocol_folds.py --h5 /data/protocol.h5 --out runs/folds.json --test-sessions 20
```

Options: `--scenarios zeroshot,calib,personal,personal_calib,louo,louo_calib` adds the leave-one-user-out control, `--curve 1,2,4,8` adds the learning curve (`personal_prev<k>`: the k sessions recorded just before), and `--personal-order chronological` trains on earlier sessions only.

**3. (Optional) check the fine-tuning settings before scoring any test task.** `protocol_config.json` uses the official ETH-XGaze defaults. To check, for example, the number of epochs without touching test data, write a tuning manifest with validation sessions held out of training, train without testing, and read `val_err_deg` per epoch in each `log.json`:

```bash
python evaluation/make_protocol_folds.py --h5 /data/protocol.h5 --out runs/tune.json --test-sessions 9 --val-sessions 2 --scenarios personal
python evaluation/run_protocol_fold.py --manifest runs/tune.json --out-dir runs/tune --scenario personal --no-test
```

Then set `protocol_config.json` and leave it unchanged for the real runs.

**4. Run the scenarios in this order.** `personal_calib` and `louo_calib` start from the saved `personal` and `louo` models.

```bash
python evaluation/run_protocol_fold.py --manifest runs/folds.json --list
python evaluation/run_protocol_fold.py --manifest runs/folds.json --out-dir runs/out --scenario zeroshot
python evaluation/run_protocol_fold.py --manifest runs/folds.json --out-dir runs/out --scenario calib
python evaluation/run_protocol_fold.py --manifest runs/folds.json --out-dir runs/out --scenario personal
python evaluation/run_protocol_fold.py --manifest runs/folds.json --out-dir runs/out --scenario personal_calib
```

On a cluster, run one fold per job with `--scenario personal --index $SLURM_ARRAY_TASK_ID`. Folds that already have a `log.json` are skipped, so an interrupted run can simply be restarted.

**5. Summarize:**

```bash
python evaluation/summarize_protocol.py --runs runs/out --out runs/summary
```

This writes `scenarios.csv`, `users.csv`, `contrasts.csv` (paired gains, % of sessions improved, Wilcoxon p over sessions and over users, 95% CI from resampling users and then sessions), `sessions.csv`, `curve.csv` and `report.md`.

## Cross-session check

Calibrates the model on one session (the `calib` fold of that session, same training) and scores it on the test tasks of every session of the same user. This gives one row of the user's session × session error matrix.

```bash
python evaluation/cross_session_eval.py --manifest runs/folds_all.json --out-dir runs/cross --list
python evaluation/cross_session_eval.py --manifest runs/folds_all.json --out-dir runs/cross --index 0
python evaluation/summarize_cross_session.py --runs runs/cross --sessions-csv ReCalib/sessions.csv --out runs/cross/summary
```

Run one `--index` per process; a session whose `pairs/u<user>_s<session>.csv` exists is skipped. The calibrated model is deleted after scoring unless `--keep-model`. The summary compares the same-session error with the mean error on the user's other sessions, using the same statistics as `summarize_protocol.py`. It also groups the error by days between the two sessions (from `sessions.csv`) and gives per-user correlations of error with days apart. About 1.5 min per session on one GPU, 4.5 h for all 186.

## Cost per fold (about 800 frames per session)

| Scenario | Training frames | Folds |
|---|---|---|
| `calib`, `personal_calib`, `louo_calib` | about 126 (the calibration task) | 1 per held-out session |
| `personal` | about 16,000 (the user's other ~20 sessions) | 1 per held-out session |
| `louo` | about 135,000 (8 users) | 1 per user |

`personal` dominates: 20 held-out sessions × 16k frames × 25 epochs ≈ 8 M training images. The number of epochs is the main cost lever.
