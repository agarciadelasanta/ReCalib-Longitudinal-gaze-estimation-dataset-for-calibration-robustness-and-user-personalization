"""Screen intersections of the ReCalib gaze vectors and the ETH-XGaze predictions, cast from one fixed point.

    py evaluation/plot_label_consistency.py <release root> --eth-pred eth_xgaze_labels.csv --out figure.pdf [--origin task | X Y Z]

Both gaze vectors of each sample (the released gaze.vector and the zero-shot ETH-XGaze model
prediction, --eth-pred: the eth_xgaze_labels.csv written by the baseline evaluation, which is not part
of the release) are cast from the same point and intersected with the screen plane
(z = 0). Casting both from one point removes the differences between the two gaze origins, so only the
directions are compared. The point is either fixed (--origin X Y Z, mm in the camera frame) or, with
--origin task (default), the median ReCalib gaze origin of each calibration task, i.e. where the
participant's eyes were. A fixed point far from the eyes shifts and scales the whole grid of
intersections, for both sets of vectors alike.
Within a calibration point the head barely moves, so the spread of each cluster shows how consistent
each set of vectors is from frame to frame.

The figure shows the calibration task (9 points) of example sessions; the statistics are printed for
all calibration tasks: per point, the spread of each cluster (median distance to the cluster median)
and the distance of the cluster median to the target.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OURS, ETH, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b29", "#8a8a85"


def eth_vectors(path):
    """{4-digit image name: predicted unit gaze vector} for the frames ETH-XGaze could process."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["ok"] != "1":
                continue
            stem, num = r["image"].rsplit("-", 1)
            out[f"{stem}-{int(num.split('.')[0]):04d}.png"] = np.array([float(r[f"pred_vec_{a}"]) for a in "xyz"])
    return out


def hit(origin, v):
    """Intersection of the ray origin + t v with the screen plane z = 0 (x, y in mm)."""
    t = -origin[2] / v[2]
    return (origin + t * v)[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release_root", type=Path)
    ap.add_argument("--eth-pred", type=Path, required=True,
                    help="ETH-XGaze zero-shot predictions (eth_xgaze_labels.csv: image, ok, pred_vec_x/y/z)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--origin", nargs="+", default=["task"], help="'task' or X Y Z in mm")
    ap.add_argument("--sessions", nargs="*", help="sessions to plot as UU_SS (default: one per user, first sessions)")
    ap.add_argument("--panels", type=int, default=4)
    args = ap.parse_args()

    fixed = None if args.origin == ["task"] else np.array([float(v) for v in args.origin])
    eth = eth_vectors(args.eth_pred)
    samples = defaultdict(list)                      # (user, session) -> [(target, ours vector, eth vector, ours origin)]
    with open(args.release_root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["discard_code"] != "0" or r["task_type"] != "calibration":
                continue
            v_eth = eth.get(Path(r["image"]).name)
            if v_eth is None:
                continue
            samples[(r["user_id"], r["session_id"])].append((
                (float(r["gaze_pog_mm_x"]), float(r["gaze_pog_mm_y"])),
                np.array([float(r[f"gaze_vector_{a}"]) for a in "xyz"]), v_eth,
                np.array([float(r[f"gaze_origin_mm_{a}"]) for a in "xyz"])))
    tasks = defaultdict(lambda: defaultdict(list))   # (user, session) -> target -> [(ours_xy, eth_xy)]
    origins = {}
    for key, rows_ in samples.items():
        origin = fixed if fixed is not None else np.median([s[3] for s in rows_], axis=0)
        origins[key] = origin
        for target, v_ours, v_eth, _ in rows_:
            tasks[key][target].append((hit(origin, v_ours), hit(origin, v_eth)))

    # statistics over all calibration tasks
    spread = {"ours": [], "eth": []}
    bias = {"ours": [], "eth": []}
    for points in tasks.values():
        for target, pairs in points.items():
            for k, col in (("ours", 0), ("eth", 1)):
                xy = np.array([p[col] for p in pairs])
                centre = np.median(xy, axis=0)
                spread[k].append(np.median(np.linalg.norm(xy - centre, axis=1)))
                bias[k].append(np.linalg.norm(centre - np.array(target)))
    print(f"{len(tasks)} calibration tasks, {sum(len(p) for p in tasks.values())} points, origin "
          + (f"{fixed.tolist()} mm" if fixed is not None else "median ReCalib gaze origin of each task"))
    for k in ("ours", "eth"):
        s, b = np.array(spread[k]), np.array(bias[k])
        print(f"{k:4s}: cluster spread median {np.median(s):5.1f} mm (p95 {np.percentile(s, 95):5.1f}); "
              f"cluster centre to target median {np.median(b):5.1f} mm (p95 {np.percentile(b, 95):5.1f})")

    # figure: calibration task of example sessions
    keys = [tuple(s.split("_")) for s in args.sessions] if args.sessions else \
        [min(k for k in tasks if k[0] == u) for u in sorted({k[0] for k in tasks})][: args.panels]
    cols = min(len(keys), 2)
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.9 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        ax.add_patch(plt.Rectangle((-130, 10), 260, 173, fill=False, ec=MUTED, lw=0.8))
        for target, pairs in tasks[key].items():
            o = np.array([p[0] for p in pairs])
            e = np.array([p[1] for p in pairs])
            ax.plot(*target, marker="o", ms=9, mfc="none", mew=1.0, c=INK, zorder=1)
            ax.scatter(e[:, 0], e[:, 1], s=9, c=ETH, alpha=0.75, lw=0, zorder=2)
            ax.scatter(o[:, 0], o[:, 1], s=10, c=OURS, alpha=0.9, lw=0, zorder=3)
        o_ = origins[key]
        ax.set_title(f"User {key[0]}, session {key[1]}, calibration task\n"
                     f"rays cast from ({o_[0]:.0f}, {o_[1]:.0f}, {o_[2]:.0f}) mm", fontsize=8, color=INK)
        ax.set_xlim(-260, 260)
        ax.set_ylim(330, -70)          # screen y grows downwards
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ax.spines.values():
            s.set_color(MUTED)
            s.set_linewidth(0.6)
        ax.set_xlabel("x (mm)", fontsize=7, color=MUTED)
        ax.set_ylabel("y (mm)", fontsize=7, color=MUTED)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, c=OURS, label="ReCalib gaze vector"),
               plt.Line2D([], [], ls="", marker="o", ms=4, c=ETH, label="ETH-XGaze prediction"),
               plt.Line2D([], [], ls="", marker="o", ms=8, mfc="none", mew=1.0, c=INK, label="Target"),
               plt.Line2D([], [], c=MUTED, lw=0.8, label="Screen")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    if args.out.suffix != ".png":
        fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
