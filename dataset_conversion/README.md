# Dataset conversion

Scripts that turned the labelled acquisition output into the released dataset, plus the quality flags. Together with [`../3d_reconstruction/`](../3d_reconstruction/README.md) they cover every step from the recorded images to the released files; only the acquisition software is not included.

## How the release was built

| Step | Script | What it does |
|---|---|---|
| 1 | `standardize_release.py` | Converts the labelled acquisition tree into the released layout (`user_XX/session_XX_YY/task_XX_YY_ZZ/XX_YY_ZZ_img-NNNN.{png,json}`), with `setup.json`, `sessions.csv` and `recalib_index.csv`. Discarded frames are kept and flagged. The table that links the pseudonymous user codes to `user_XX` stays private |
| 2 | `apply_depth_scale.py` | Applies the per-participant face-scale factors (`3d_reconstruction/user_depth_scale.csv`) to the gaze origin, gaze vector and head translation |
| 3 | `apply_quality_flags.py` | Sets `discard_info` with the rules of the quality pipeline used for the release: code 2 (no face or landmarks), 5 (eye landmarks out of range or implausible face size) and 10 (mean eye aspect ratio below 0.05) |
| 3b | `add_eth_xgaze_origin.py` | Adds `gaze_eth_xgaze` (gaze origin = ETH-XGaze face centre, and the unit vector from it to the same target) to every JSON and to the index, from the per-frame output of `evaluation/eth_origin_check.py --n 0 --discard-codes 0,10`; null when dlib finds no face or places it more than 900 mm away |
| 4 | `create_dataset_summary.py` | Writes the one-row-per-sample summary CSV used by the visualization and analysis tools |

## Reproducing the annotations from the released images

With the dataset unpacked in `ReCalib/` at the repository root:

```bash
cd 3d_reconstruction
python -m recalib_reconstruction.label_batch --csv ../ReCalib/recalib_index.csv --images-root ../ReCalib \
    --setup ../ReCalib/setup.json --out-dir relabelled --depth-scale user_depth_scale.csv
cd ..
python dataset_conversion/apply_quality_flags.py --labels 3d_reconstruction/relabelled --setup ReCalib/setup.json \
    --write --compare-index ReCalib/recalib_index.csv
```

The first command recomputes landmarks, head pose, gaze and quality metrics; the second sets the flags and compares them with the released ones. `gaze_eth_xgaze` is reproduced separately, with `evaluation/eth_origin_check.py` and `add_eth_xgaze_origin.py`. Each script documents its options in `--help`.
