"""Panels of the paper figure 'Distribution of gaze and head pose parameters for a representative participant'.

    python visualization/dataset_distribution_figure.py --index ReCalib/recalib_index.csv \
        --user 05 --out <paper>/assets/figures

Kept samples of one participant, calibration vs. test tasks (panels a-f) and calibration tasks per
session (g, h). Angles in degrees, positions in mm in the camera frame (x to the image right, y down,
z away from the camera). Writes one PNG per panel, with the file names used by main.tex:
  a  gaze direction (yaw, pitch)             smooth_coverage_map_gt_gaze_x_vs_gt_gaze_y.png
  b  head rotation, yaw vs pitch             smooth_coverage_map_hpe_r_x_vs_hpe_r_y.png
  c  head position, x vs y                   smooth_coverage_map_hpe_t_x_vs_hpe_t_y.png
  d  head position, x vs z                   smooth_coverage_map_hpe_t_x_vs_hpe_t_z.png
  e  gaze rays from the median gaze origin intersected with the screen plane
                                             smooth_coverage_map_pog_intersect_norm_x_vs_pog_intersect_norm_y.png
  f  head rotation, yaw vs roll              smooth_coverage_map_hpe_r_x_vs_hpe_r_z.png
  g  head position x vs y, calibration tasks per session   smooth_coverage_map_hpe_t_x_vs_hpe_t_y_session.png
  h  head position x vs z, calibration tasks per session   smooth_coverage_map_hpe_t_x_vs_hpe_t_z_session.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

CALIB, TEST = "#2a78d6", "#eb6834"              # categorical: calibration (blue), test (orange)
INK, MUTED = "#0b0b0b", "#52514e"
MAX_POINTS = 5000                               # per KDE, for speed
plt.rcParams.update({"font.size": 20, "axes.labelsize": 26, "xtick.labelsize": 20, "ytick.labelsize": 20,
                     "legend.fontsize": 22, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def load(index, user):
    cols = ["user_id", "session_id", "task_type", "discard_code", "gaze_vector_x", "gaze_vector_y", "gaze_vector_z",
            "gaze_origin_mm_x", "gaze_origin_mm_y", "gaze_origin_mm_z", "head_rotation_rad_x", "head_rotation_rad_y",
            "head_rotation_rad_z", "head_translation_mm_x", "head_translation_mm_y", "head_translation_mm_z"]
    d = pd.read_csv(index, usecols=cols, dtype={"user_id": str, "session_id": str})
    d = d[(d.user_id == user) & (d.discard_code == 0)].copy()
    v = d[["gaze_vector_x", "gaze_vector_y", "gaze_vector_z"]].to_numpy()
    d["gaze_yaw"] = np.degrees(np.arctan2(v[:, 0], -v[:, 2]))      # + towards the image right
    d["gaze_pitch"] = np.degrees(np.arcsin(-v[:, 1]))              # + up
    d["roll"], d["yaw"], d["pitch"] = (np.degrees(d[f"head_rotation_rad_{a}"]) for a in "xyz")   # zyx = [roll, yaw, pitch]
    d["tx"], d["ty"], d["tz"] = (d[f"head_translation_mm_{a}"] for a in "xyz")
    o = np.median(d[["gaze_origin_mm_x", "gaze_origin_mm_y", "gaze_origin_mm_z"]].to_numpy(), axis=0)
    t = -o[2] / v[:, 2]                                             # ray from the median origin to the plane z = 0
    d["pog_x"], d["pog_y"] = o[0] + t * v[:, 0], o[1] + t * v[:, 1]
    return d


def sample(df):
    return df.sample(min(len(df), MAX_POINTS), random_state=0)


def kde(ax, df, x, y, color, label=None):
    sns.kdeplot(data=sample(df), x=x, y=y, fill=True, levels=8, thresh=0.05, alpha=0.55, color=color, ax=ax)
    sns.kdeplot(data=sample(df), x=x, y=y, fill=False, levels=8, thresh=0.05, linewidths=0.6, color=color, ax=ax)
    if label:
        ax.plot([], [], color=color, lw=10, alpha=0.7, label=label)


def finish(ax, xlabel, ylabel, path, legend=True, invert_y=False, equal=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2, columnspacing=1.2)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=200, facecolor="white")
    plt.close(ax.figure)


def two_tasks(d, x, y, xlabel, ylabel, path, **kw):
    fig, ax = plt.subplots(figsize=(6, 5))
    kde(ax, d[d.task_type == "test"], x, y, TEST, "test")
    kde(ax, d[d.task_type == "calibration"], x, y, CALIB, "calibration")
    finish(ax, xlabel, ylabel, path, **kw)


def per_session(d, x, y, xlabel, ylabel, path, **kw):
    cal = d[d.task_type == "calibration"]
    sessions = sorted(cal.session_id.unique())
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, s in enumerate(sessions):                                # chronological: light (first) -> dark (last)
        color = cmap(0.35 + 0.6 * k / max(len(sessions) - 1, 1))
        sns.kdeplot(data=cal[cal.session_id == s], x=x, y=y, fill=True, levels=5, thresh=0.05, alpha=0.6, color=color, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", [cmap(0.35), cmap(0.95)]),
                               norm=plt.Normalize(1, len(sessions)))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, ticks=[1, len(sessions)])
    cb.set_label("session", fontsize=22)
    cb.ax.tick_params(labelsize=18)
    finish(ax, xlabel, ylabel, path, legend=False, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--user", default="05")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    d = load(args.index, args.user)
    n = lambda f: args.out / f"smooth_coverage_map_{f}.png"

    two_tasks(d, "gaze_yaw", "gaze_pitch", "gaze yaw (°)", "gaze pitch (°)", n("gt_gaze_x_vs_gt_gaze_y"))
    two_tasks(d, "yaw", "pitch", "head yaw (°)", "head pitch (°)", n("hpe_r_x_vs_hpe_r_y"))
    two_tasks(d, "tx", "ty", "head x (mm)", "head y (mm)", n("hpe_t_x_vs_hpe_t_y"), invert_y=True)
    two_tasks(d, "tx", "tz", "head x (mm)", "head z (mm)", n("hpe_t_x_vs_hpe_t_z"))
    # (e): gaze rays from the median origin on the screen plane, screen outline from setup.json
    setup = json.loads((args.index.parent / "setup.json").read_text())
    scr, cam = setup["screen"], setup["camera"]["position_mm"]
    fig, ax = plt.subplots(figsize=(6, 5))
    kde(ax, d[d.task_type == "test"], "pog_x", "pog_y", TEST, "test")
    kde(ax, d[d.task_type == "calibration"], "pog_x", "pog_y", CALIB, "calibration")
    ax.add_patch(Rectangle((-scr["width_mm"] / 2 + cam["x"], cam["y"]), scr["width_mm"], scr["height_mm"],
                           fill=False, ec=INK, lw=1.5, ls="--"))
    ax.plot([0], [0], marker="s", color=INK, ms=7)                  # camera
    finish(ax, "screen x (mm)", "screen y (mm)", n("pog_intersect_norm_x_vs_pog_intersect_norm_y"), invert_y=True, equal=True)
    two_tasks(d, "yaw", "roll", "head yaw (°)", "head roll (°)", n("hpe_r_x_vs_hpe_r_z"))
    per_session(d, "tx", "ty", "head x (mm)", "head y (mm)", n("hpe_t_x_vs_hpe_t_y_session"), invert_y=True)
    per_session(d, "tx", "tz", "head x (mm)", "head z (mm)", n("hpe_t_x_vs_hpe_t_z_session"))
    print(f"user {args.user}: {len(d)} kept samples, {d.session_id.nunique()} sessions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
