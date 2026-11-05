"""Summarize the protocol runs: error per scenario, paired gains between scenarios, session drift.

The session (one participant, one sitting) is the unit of analysis: frame errors are averaged per
held-out session first, and every comparison pairs the same sessions in two scenarios.

  scenarios.csv   per scenario: sessions, frames, mean / median / IQR of the session means
                  (deg and mm), and the mean over users of their session means
  users.csv       per scenario and user: mean, SD and range over that user's sessions
  contrasts.csv   per pair of scenarios (A -> B): mean and median gain (A - B, positive = B better),
                  % of sessions that improve, Wilcoxon signed-rank p over sessions and over user
                  means, and a 95% CI of the mean gain from a two-level bootstrap (users, then
                  sessions within users)
  sessions.csv    the session means of every scenario side by side (for figures)
  curve.csv       personal_prev<k> learning curve on the sessions that have every k (if run)
  report.md       the tables above in Markdown

    python evaluation/summarize_protocol.py --runs runs --out runs/summary
    python evaluation/summarize_protocol.py --runs runs --out runs/summary_sub --sessions keep.csv

--sessions restricts every table to the sessions listed in a CSV with columns user, session.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CONTRASTS = [  # (from, to, question)
    ("zeroshot", "personal", "personalization, no session calibration"),
    ("calib", "personal_calib", "personalization, with session calibration"),
    ("personal", "personal_calib", "session calibration after personalization"),
    ("zeroshot", "calib", "session calibration without personalization"),
    ("louo", "personal", "own sessions vs other users (domain control)"),
    ("louo_calib", "personal_calib", "own sessions vs other users, with session calibration"),
    ("louo", "louo_calib", "session calibration after domain adaptation"),
    ("zeroshot", "louo", "domain adaptation"),
]


def load_runs(runs):
    frames = []
    for csv_path in sorted(runs.glob("*/*/predictions.csv")):
        df = pd.read_csv(csv_path, dtype={"user": str, "session": str, "task": str, "sample": str})
        df["scenario"] = csv_path.parent.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"no predictions.csv under {runs}")
    return pd.concat(frames, ignore_index=True)


def two_level_bootstrap(diff, n_boot, rng):
    """95% CI of the mean session gain, resampling users and then sessions within each user."""
    groups = [g.to_numpy() for _, g in diff.groupby(level="user")]
    means = np.empty(n_boot)
    for b in range(n_boot):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        means[b] = np.mean(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in picked]))
    return np.percentile(means, [2.5, 97.5])


def p_wilcoxon(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if len(x) >= 2 and np.any(x != 0) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="--out-dir of run_protocol_fold.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", default="ang_err_deg", help="per-frame error column (deg)")
    ap.add_argument("--sessions", type=Path, help="CSV with columns user, session: analyse only these sessions")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_runs(args.runs)
    if args.sessions:
        keep = pd.read_csv(args.sessions, dtype=str)[["user", "session"]]
        keep = keep.assign(user=keep.user.str.zfill(2), session=keep.session.str.zfill(2))
        df = df.merge(keep, on=["user", "session"])
        print(f"restricted to {len(keep)} sessions from {args.sessions}")
    sess = (df.groupby(["scenario", "user", "session"])
              .agg(err_deg=(args.metric, "mean"), err_mm=("pog_err_mm", "mean"), frames=(args.metric, "size"))
              .reset_index())
    sess.pivot_table(index=["user", "session"], columns="scenario", values="err_deg").to_csv(args.out / "sessions.csv")

    rows = []
    for sc, g in sess.groupby("scenario"):
        q1, med, q3 = np.percentile(g.err_deg, [25, 50, 75])
        rows.append(dict(scenario=sc, sessions=len(g), users=g.user.nunique(), frames=int(g.frames.sum()),
                         mean_deg=g.err_deg.mean(), median_deg=med, q1_deg=q1, q3_deg=q3,
                         user_mean_deg=g.groupby("user").err_deg.mean().mean(),
                         mean_mm=g.err_mm.mean(), median_mm=g.err_mm.median()))
    scen = pd.DataFrame(rows)
    scen.to_csv(args.out / "scenarios.csv", index=False)

    users = (sess.groupby(["scenario", "user"]).err_deg
                 .agg(sessions="size", mean="mean", sd="std", min="min", max="max").reset_index())
    users["range"] = users["max"] - users["min"]
    users.to_csv(args.out / "users.csv", index=False)

    wide = sess.pivot_table(index=["user", "session"], columns="scenario", values="err_deg")
    crow = []
    for a, b, question in CONTRASTS:
        if a not in wide or b not in wide:
            continue
        diff = (wide[a] - wide[b]).dropna()
        if diff.empty:
            continue
        per_user = diff.groupby(level="user").mean()
        lo, hi = two_level_bootstrap(diff, args.bootstrap, rng)
        crow.append(dict(contrast=f"{a} -> {b}", question=question, sessions=len(diff), users=len(per_user),
                         mean_gain_deg=diff.mean(), median_gain_deg=diff.median(), ci95_low=lo, ci95_high=hi,
                         pct_sessions_improved=100 * (diff > 0).mean(),
                         users_improved=int((per_user > 0).sum()),
                         p_wilcoxon_sessions=p_wilcoxon(diff), p_wilcoxon_users=p_wilcoxon(per_user)))
    contrasts = pd.DataFrame(crow)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)

    curve = pd.DataFrame()
    prev = sorted((c for c in wide.columns if c.startswith("personal_prev")), key=lambda c: int(c[13:]))
    if prev:
        common = wide[prev].dropna()
        curve = pd.DataFrame(dict(previous_sessions=[int(c[13:]) for c in prev], sessions=len(common),
                                  mean_deg=[common[c].mean() for c in prev],
                                  median_deg=[common[c].median() for c in prev]))
        curve.to_csv(args.out / "curve.csv", index=False)

    with open(args.out / "report.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Protocol summary ({args.metric})\n\nUnit: held-out session (frame errors averaged per "
                 "session first).\n\n## Scenarios\n\n" + scen.round(3).to_markdown(index=False))
        fh.write("\n\n## Paired contrasts (gain = from - to, positive = better)\n\n"
                 + (contrasts.round(4).to_markdown(index=False) if len(contrasts) else "none"))
        fh.write("\n\n## Session-to-session spread per user\n\n" + users.round(3).to_markdown(index=False))
        if len(curve):
            fh.write("\n\n## Learning curve\n\n" + curve.round(3).to_markdown(index=False))
        fh.write("\n")
    print(scen.round(3).to_string(index=False))
    if len(contrasts):
        print()
        print(contrasts[["contrast", "sessions", "mean_gain_deg", "ci95_low", "ci95_high",
                         "pct_sessions_improved", "p_wilcoxon_sessions", "p_wilcoxon_users"]].round(4).to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
