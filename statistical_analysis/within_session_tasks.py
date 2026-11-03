"""Within-session, between-task analysis of head pose (calibration vs test tasks).

For every session (kept samples only) the calibration task (task 00) is compared with the
three test tasks (01-03):
  * stability:   head-rotation spread within each task (median deviation from the task median)
  * shift:       distance between the calibration and test task medians (rotation and position)
  * protocol:    shift of test task 1 relative to test tasks 2-3 (participants were asked to
                 keep the first test task static, and to move freely in the following ones)
  * overlap:     share of test frames whose head pose stays close to the calibration pose
  * test-test:   distance between the medians of the three test tasks
Head rotation uses the released Euler angles (rotation_rad = [roll, yaw, pitch]) in degrees;
position uses the gaze origin (midpoint between the eyes, mm).

Outputs (in --out): per_session_metrics.csv, per_user_metrics.csv, statistical_tests.csv,
within_session_tasks.png/.pdf

Usage:
    python within_session_tasks.py --index F:/recalib_release/ReCalib/recalib_index.csv --out temp/within_session
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch
from scipy.stats import friedmanchisquare, wilcoxon

ROT, POS = ["yaw", "pitch", "roll"], ["px", "py", "pz"]
NEAR_ROT_DEG, NEAR_POS_MM = 5.0, 25.0

# reference sequential ramp (blue 100 -> 650), text tokens, accent for the per-column maximum
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, ACCENT = "#0b0b0b", "#52514e", "#fcfcfb", "#eb6834"
NEUTRAL = "#f0efec"

COLUMNS = [  # key, header, group, format
    ("sessions", "sessions", "volume", "{:.0f}"),
    ("frames_k", "kept frames\n(thousands)", "volume", "{:.1f}"),
    ("cal_rot_spread", "rotation spread\ncalibration (°)", "head stability within task", "{:.1f}"),
    ("test_rot_spread", "rotation spread\ntest (°)", "head stability within task", "{:.1f}"),
    ("shift_rot", "rotation shift\ncalib.→test (°)", "calibration → test", "{:.1f}"),
    ("shift_pos", "position shift\ncalib.→test (mm)", "calibration → test", "{:.0f}"),
    ("t1_ratio", "test-1 shift /\ntests 2–3 shift", "calibration → test", "{:.2f}"),
    ("near_calib", f"test frames near\ncalib. pose (%)", "overlap & test-to-test", "{:.0f}"),
    ("test_test_rot", "rotation change\nbetween tests (°)", "overlap & test-to-test", "{:.1f}"),
]


def spread(g, cols):
    d = g[cols].to_numpy() - g[cols].median().to_numpy()
    return float(np.median(np.linalg.norm(d, axis=1)))


def session_metrics(k):
    rows = []
    for (u, s), ses in k.groupby(["user_id", "session_id"]):
        cal = ses[ses.task_id == "00"]
        tests = {t: g for t, g in ses.groupby("task_id") if t != "00"}
        if len(cal) < 20 or len(tests) < 3 or min(len(g) for g in tests.values()) < 20:
            continue  # incomplete sessions (see incomplete_tasks.csv) are excluded
        ts = sorted(tests)
        c_rot, c_pos = cal[ROT].median().to_numpy(), cal[POS].median().to_numpy()
        med_rot = {t: tests[t][ROT].median().to_numpy() for t in ts}
        med_pos = {t: tests[t][POS].median().to_numpy() for t in ts}
        shift_rot = {t: np.linalg.norm(med_rot[t] - c_rot) for t in ts}
        shift_pos = {t: np.linalg.norm(med_pos[t] - c_pos) for t in ts}
        allt = pd.concat(tests.values())
        near = ((np.linalg.norm(allt[ROT].to_numpy() - c_rot, axis=1) <= NEAR_ROT_DEG)
                & (np.linalg.norm(allt[POS].to_numpy() - c_pos, axis=1) <= NEAR_POS_MM)).mean()
        pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
        rows.append({
            "user_id": u, "session_id": s, "frames": len(ses),
            "cal_rot_spread": spread(cal, ROT),
            "test_rot_spread": np.mean([spread(tests[t], ROT) for t in ts]),
            "test1_rot_spread": spread(tests[ts[0]], ROT),
            "shift_rot": np.mean(list(shift_rot.values())), "shift_pos": np.mean(list(shift_pos.values())),
            "shift_rot_t1": shift_rot[ts[0]], "shift_rot_t23": np.mean([shift_rot[t] for t in ts[1:]]),
            "shift_pos_t1": shift_pos[ts[0]], "shift_pos_t23": np.mean([shift_pos[t] for t in ts[1:]]),
            **{f"shift_rot_{t}": shift_rot[t] for t in ts},
            "near_calib": 100 * near,
            "test_test_rot": np.mean([np.linalg.norm(med_rot[ts[i]] - med_rot[ts[j]]) for i, j in pairs]),
            "test_test_pos": np.mean([np.linalg.norm(med_pos[ts[i]] - med_pos[ts[j]]) for i, j in pairs]),
        })
    s = pd.DataFrame(rows)
    s["t1_ratio"] = s.shift_rot_t1 / s.shift_rot_t23
    return s


def tests_table(S):
    out = []

    def add(scope, name, x, y, alternative):
        x, y = np.asarray(x), np.asarray(y)
        stat, p = wilcoxon(x, y, alternative=alternative)
        out.append({"scope": scope, "test": name, "n_sessions": len(x),
                    "median_a": np.median(x), "median_b": np.median(y), "statistic": stat, "p_value": p})

    for scope, g in [("all", S)] + [(f"user {u}", g) for u, g in S.groupby("user_id")]:
        add(scope, "rotation spread: test > calibration (Wilcoxon, one-sided)", g.test_rot_spread, g.cal_rot_spread, "greater")
        add(scope, "rotation shift: test 1 < tests 2-3 (Wilcoxon, one-sided)", g.shift_rot_t1, g.shift_rot_t23, "less")
        add(scope, "position shift: test 1 < tests 2-3 (Wilcoxon, one-sided)", g.shift_pos_t1, g.shift_pos_t23, "less")
    fr = friedmanchisquare(S.shift_rot_01, S.shift_rot_02, S.shift_rot_03)
    out.append({"scope": "all", "test": "rotation shift differs between test tasks 1-3 (Friedman)",
                "n_sessions": len(S), "median_a": np.nan, "median_b": np.nan,
                "statistic": fr.statistic, "p_value": fr.pvalue})
    return pd.DataFrame(out)


def heatmap(U, all_row, path_png):
    data = pd.concat([U, all_row])
    labels = [f"user {u}" for u in U.index] + ["all sessions"]
    n_rows, n_cols = len(data), len(COLUMNS)
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    fig, ax = plt.subplots(figsize=(1.32 * n_cols + 1.3, 0.52 * n_rows + 1.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    gap = 0.04
    for j, (key, header, _, fmt) in enumerate(COLUMNS):
        col = U[key]
        lo, hi = col.min(), col.max()
        for i, v in enumerate(data[key]):
            y = (n_rows - 1 - i) + (0.0 if i < len(U) else -0.25)  # separate the summary row
            t = 0.0 if hi == lo else np.clip((v - lo) / (hi - lo), 0, 1)
            face = cmap(0.08 + 0.92 * t) if i < len(U) else NEUTRAL  # summary row is not on the scale
            ax.add_patch(plt.Rectangle((j + gap, y + gap), 1 - 2 * gap, 1 - 2 * gap, color=face, lw=0))
            r, g, b = to_rgb(face)
            ink = "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45 else TEXT_PRIMARY
            bold = i < len(U) and v == lo
            ax.text(j + 0.5, y + 0.5, fmt.format(v), ha="center", va="center", fontsize=10.5,
                    color=ink, fontweight="bold" if bold else "normal")
            if i < len(U) and v == hi:
                ax.add_patch(FancyBboxPatch((j + gap + 0.03, y + gap + 0.03), 1 - 2 * gap - 0.06, 1 - 2 * gap - 0.06,
                                            boxstyle="round,pad=0,rounding_size=0.12", fill=False,
                                            ec=ACCENT, lw=2.2))
        ax.text(j + 0.5, n_rows + 0.12, header, ha="center", va="bottom", fontsize=9.5, color=TEXT_SECONDARY)
    for i, lab in enumerate(labels):
        y = (n_rows - 1 - i) + (0.0 if i < len(U) else -0.25)
        ax.text(-0.12, y + 0.5, lab, ha="right", va="center", fontsize=10.5, color=TEXT_PRIMARY,
                fontweight="bold" if i == len(U) else "normal")
    # group labels under the table
    groups = []
    for j, (_, _, grp, _) in enumerate(COLUMNS):
        if groups and groups[-1][0] == grp:
            groups[-1][2] = j
        else:
            groups.append([grp, j, j])
    for grp, a, b in groups:
        ax.plot([a + 0.1, b + 0.9], [-0.55, -0.55], color=TEXT_SECONDARY, lw=1)
        ax.text((a + b + 1) / 2, -0.75, grp, ha="center", va="top", fontsize=10.5, color=TEXT_PRIMARY,
                fontweight="bold")
    ax.set_xlim(-1.5, n_cols + 0.05)
    ax.set_ylim(-1.15, n_rows + 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200, facecolor=SURFACE)
    fig.savefig(path_png.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, type=Path, help="recalib_index.csv of the release")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(args.index, dtype={"user_id": str, "session_id": str, "task_id": str}, low_memory=False)
    k = idx[~idx.discarded].copy()
    k["roll"], k["yaw"], k["pitch"] = [np.degrees(k[f"head_rotation_rad_{a}"]) for a in "xyz"]
    k["px"], k["py"], k["pz"] = k.gaze_origin_mm_x, k.gaze_origin_mm_y, k.gaze_origin_mm_z

    S = session_metrics(k)
    S.to_csv(args.out / "per_session_metrics.csv", index=False)

    U = S.groupby("user_id").median(numeric_only=True)
    U["sessions"] = S.groupby("user_id").size()
    U["frames_k"] = k.groupby("user_id").size() / 1000
    all_row = S.median(numeric_only=True).to_frame().T
    all_row["sessions"] = len(S)
    all_row["frames_k"] = len(k) / 1000
    all_row.index = ["all"]
    keys = [c[0] for c in COLUMNS]
    pd.concat([U, all_row])[keys + ["shift_rot_t1", "shift_rot_t23", "shift_pos_t1", "shift_pos_t23",
                                    "test_test_pos", "test1_rot_spread"]].to_csv(args.out / "per_user_metrics.csv")

    T = tests_table(S)
    T.to_csv(args.out / "statistical_tests.csv", index=False)
    heatmap(U, all_row, args.out / "within_session_tasks.png")

    pd.set_option("display.width", 220)
    print(pd.concat([U, all_row])[keys].round(2).to_string())
    print()
    print(T[T.scope == "all"].round(4).to_string(index=False))
    print("\nper-user p-values (one-sided Wilcoxon):")
    pu = T[T.scope != "all"].pivot(index="scope", columns="test", values="p_value")
    pu.columns = ["pos t1<t23", "rot t1<t23", "spread test>calib"]
    print(pu.round(4).to_string())
    print(f"\nexcluded incomplete sessions: {k.groupby(['user_id', 'session_id']).ngroups - len(S)}")


if __name__ == "__main__":
    main()
