"""Summarize the cross-session check (cross_session_eval.py).

A model calibrated on session s of a user is scored on every session of that user. Per calibrated
session, its error on the same session is compared with its error on the user's other sessions, and
errors are grouped by the number of days between the two sessions (recording dates from sessions.csv).

  pairs.csv        all (user, train session, test session) rows with the days between them
  summary.csv      same-session vs other-session error: means, paired difference with a two-level
                   bootstrap CI (users, then sessions), % of sessions worse elsewhere, Wilcoxon p
  users.csv        the same per user, plus the Spearman correlation of error with days apart
  days.csv         error by days between calibration and test session
  report.md        the tables above in Markdown

    python evaluation/summarize_cross_session.py --runs runs/cross --sessions-csv ReCalib/sessions.csv --out runs/cross/summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from summarize_protocol import p_wilcoxon, two_level_bootstrap

DAY_BINS = [(-1, 0, "same day"), (0, 7, "1-7 days"), (7, 14, "8-14 days"), (14, 30, "15-30 days"),
            (30, 10_000, "> 30 days")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="--out-dir of cross_session_eval.py")
    ap.add_argument("--sessions-csv", type=Path, required=True, help="sessions.csv of the release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ids = {"user": str, "train_session": str, "test_session": str}
    pairs = pd.concat([pd.read_csv(f, dtype=ids) for f in sorted((args.runs / "pairs").glob("*.csv"))],
                      ignore_index=True)
    ses = pd.read_csv(args.sessions_csv, dtype={"user_id": str, "session_id": str, "task_id": str})
    start = (ses.assign(start=pd.to_datetime(ses.start)).groupby(["user_id", "session_id"]).start.min())
    day = {k: v.normalize() for k, v in start.items()}
    pairs["days_apart"] = [abs((day[(u, b)] - day[(u, a)]).days)
                           for u, a, b in zip(pairs.user, pairs.train_session, pairs.test_session)]
    pairs["same"] = pairs.train_session == pairs.test_session
    pairs.to_csv(args.out / "pairs.csv", index=False)

    own = pairs[pairs.same].set_index(["user", "train_session"])
    other = pairs[~pairs.same].groupby(["user", "train_session"])[["mean_deg", "mean_mm"]].mean()
    per = own[["mean_deg", "mean_mm"]].join(other, lsuffix="_same", rsuffix="_other").dropna()
    per.index.names = ["user", "session"]
    diff = per.mean_deg_other - per.mean_deg_same
    lo, hi = two_level_bootstrap(diff, args.bootstrap, rng)
    per_user_diff = diff.groupby(level="user").mean()
    summary = pd.DataFrame([dict(
        sessions=len(per), users=per.index.get_level_values("user").nunique(),
        same_session_deg=per.mean_deg_same.mean(), other_sessions_deg=per.mean_deg_other.mean(),
        same_session_mm=per.mean_mm_same.mean(), other_sessions_mm=per.mean_mm_other.mean(),
        increase_deg=diff.mean(), median_increase_deg=diff.median(), ci95_low=lo, ci95_high=hi,
        pct_sessions_worse_elsewhere=100 * (diff > 0).mean(), users_worse_elsewhere=int((per_user_diff > 0).sum()),
        p_wilcoxon_sessions=p_wilcoxon(diff), p_wilcoxon_users=p_wilcoxon(per_user_diff))])
    summary.to_csv(args.out / "summary.csv", index=False)

    urows = []
    for u, g in per.groupby(level="user"):
        o = pairs[(pairs.user == u) & ~pairs.same]
        rho = spearmanr(o.days_apart, o.mean_deg).statistic if o.days_apart.nunique() > 1 else np.nan
        urows.append(dict(user=u, sessions=len(g), same_session_deg=g.mean_deg_same.mean(),
                          other_sessions_deg=g.mean_deg_other.mean(),
                          increase_deg=(g.mean_deg_other - g.mean_deg_same).mean(),
                          spearman_days_vs_error=rho, max_days_apart=int(o.days_apart.max())))
    users = pd.DataFrame(urows)
    users.to_csv(args.out / "users.csv", index=False)

    rel = pairs.merge(own.mean_deg.rename("own_deg").reset_index(), on=["user", "train_session"])
    rel["increase_deg"] = rel.mean_deg - rel.own_deg
    drows = [dict(group="same session", pairs=int(pairs.same.sum()), mean_deg=pairs[pairs.same].mean_deg.mean(),
                  increase_over_same_session_deg=0.0)]
    for lo_d, hi_d, name in DAY_BINS:
        g = rel[~rel.same & (rel.days_apart > lo_d) & (rel.days_apart <= hi_d)]
        if len(g):
            drows.append(dict(group=f"other session, {name}", pairs=len(g), mean_deg=g.mean_deg.mean(),
                              increase_over_same_session_deg=g.increase_deg.mean()))
    days = pd.DataFrame(drows)
    days.to_csv(args.out / "days.csv", index=False)

    with open(args.out / "report.md", "w", encoding="utf-8") as fh:
        fh.write("# Cross-session check\n\nModel calibrated on the 9-point task of one session, scored on the 16-point "
                 "test tasks of every session of the same user (session means).\n\n## Same vs other sessions\n\n"
                 + summary.round(4).to_markdown(index=False) + "\n\n## By days between calibration and test\n\n"
                 + days.round(3).to_markdown(index=False) + "\n\n## Per user\n\n" + users.round(3).to_markdown(index=False)
                 + "\n")
    print(summary.round(3).T.to_string(header=False))
    print()
    print(days.round(3).to_string(index=False))
    print()
    print(users.round(3).to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
