# Visualization

Tools to inspect single samples and the distributions of the dataset. Run them from the repository root; they find their bundled data and configuration files on their own.

```bash
python -m pip install -r requirements-visualization.txt
```

## `explore_dataset.py`: dataset overview

Builds a self-contained interactive HTML report from the bundled `dataset_summary.csv`, so it works without the restricted images. It has cascading participant, session and task filters, exact sample counts, longitudinal session coverage and a per-session breakdown (Overview tab), and the screen-target, head-rotation and head-position distributions per task (Sample distributions tab). The screen-target panel acts as a cross-filter for the rest of the report.

```bash
python visualization/explore_dataset.py                 # writes temp/recalib_dataset_explorer.html
python visualization/explore_dataset.py --open          # and opens it
python visualization/explore_dataset.py --id-user 7 --id-session 13 --output results/explorer.html
```

Head-pose plots use a deterministic subsample to keep the report responsive; counts and screen-target coverage are exact. See `--help` for all options.

## `visualize_sample.py`: one sample

Shows the labels of one image: the target and gaze intersection on the screen, the landmarks, eye regions and gaze vector over the image, and an interactive 3D view of the camera, screen, head pose and gaze ray. By default it opens the bundled example (`example/07_00_02_img-0040.png`); any other `.png`/`.jpg` with a same-named `.json` can be given.

```bash
python visualization/visualize_sample.py
python visualization/visualize_sample.py path/to/sample.png
python visualization/visualize_sample.py --no-display        # non-interactive check
```

Optionally it overlays the prediction of an ETH-XGaze model and reports its angular error. This needs the full `requirements.txt` and the checkpoint linked in `evaluation/ckpt/readme.txt`:

```bash
python visualization/visualize_sample.py --checkpoint evaluation/ckpt/epoch_24_ckpt.pth.tar
```

| 2D screen view | Image overlay | Interactive 3D view |
| :---: | :---: | :---: |
| ![2D screen view](../docs/assets/2d_sample_visualization.png) | ![Image overlay](../docs/assets/sample_overlay_visualization.png) | ![3D view](../docs/assets/3d_sample_visualization.png) |

## `main_gaze_visualization.py`: distributions of a participant, session or task

Histograms, KDE heat maps and scatter plots of head pose, gaze vectors and gaze intersections, comparing calibration and test tasks, the sessions of a participant, or the tasks of a session. Reads `dataset_summary.csv` (participant 7 by default) and exports the figures and a filtered CSV to `temp/`.

```bash
python visualization/main_gaze_visualization.py
python visualization/main_gaze_visualization.py --id-user 7 --id-session 13 --id-task 0
python visualization/main_gaze_visualization.py --all-users --no-plots --no-export
```

| Task: `hpe_t_z` vs `hpe_t_x` | Session: `hpe_t_z` vs `hpe_t_x` | Participant: normalized PoG |
| :---: | :---: | :---: |
| ![Task](../docs/assets/07_13_hpe_t_z_vs_hpe_t_x_task.png) | ![Session](../docs/assets/07_hpe_t_z_vs_hpe_t_x_task_type.png) | ![PoG](../docs/assets/07_pog_intersect_norm.png) |

## `explore_statistical_analysis.py`

Interactive explorer of the report written by `statistical_analysis` (see [`../statistical_analysis/README.md`](../statistical_analysis/README.md)).
