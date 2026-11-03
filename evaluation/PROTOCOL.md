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

Test sets are always the 16-point test tasks of the held-out session. Every scenario fine-tunes the whole network in the same way (L1 loss on normalized pitch/yaw, Adam), with the settings of its group in `protocol_config.json`.

## Inputs and labels

- Inputs: the image and the camera intrinsics (`docs/camera_intrinsics.npz`). Six landmarks from dlib give the head pose (solvePnP with the ETH face model) and the ETH normalization (224×224 face, 600 mm).
- Label (default `gaze_eth`): unit(`pog_mm` − ETH face centre). It uses only the stimulus point, as in the zero-shot baseline (`eth_xgaze_baseline.py`). `--label gaze_released` uses the released `gaze.vector` instead.
- Frames: kept frames only (discard code 0). Frames where dlib finds no face, or where the ETH face centre is outside 300–900 mm from the camera (a wrong detection), are marked invalid and listed in `<h5>_failures.csv`. In a 1,000-frame sample these were 4.4% and about 1%.

## Steps

Requirements: `pip install -r requirements.txt`, plus `evaluation/ckpt/epoch_24_ckpt.pth.tar` and `evaluation/modules/shape_predictor_68_face_landmarks.dat`. Run from the repo root.

**1. Normalize the release once** (CPU, dlib on the full 1920×1080 image; about 2 s per frame per worker on a laptop CPU):

```bash
python evaluation/prepare_protocol_h5.py /data/ReCalib --out /data/protocol.h5 --num-workers 16
```

**2. Write the folds** (seconds). `--test-sessions` sets how many held-out sessions are used in total, spread over the users and over time. The same sessions are used in every scenario. Training sets always use all of the user's other sessions.

```bash
python evaluation/make_protocol_folds.py --h5 /data/protocol.h5 --out runs/folds.json --test-sessions 20
```

Options: `--scenarios zeroshot,calib,personal,personal_calib,louo,louo_calib` adds the leave-one-user-out control, `--curve 1,2,4,8` adds the learning curve (`personal_prev<k>`: the k sessions recorded just before), and `--personal-order chronological` trains on earlier sessions only.

**3. Fix the fine-tuning settings before scoring any test task.** Write a tuning manifest with validation sessions held out of training, train without testing, and read `val_err_deg` per epoch in each `log.json`:

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

## Cost per fold (about 800 frames per session)

| Scenario | Training frames | Folds |
|---|---|---|
| `calib`, `personal_calib`, `louo_calib` | about 126 (the calibration task) | 1 per held-out session |
| `personal` | about 16,000 (the user's other ~20 sessions) | 1 per held-out session |
| `louo` | about 135,000 (8 users) | 1 per user |

`personal` dominates: 20 held-out sessions × 16k frames × 5 epochs ≈ 1.6 M training images.
