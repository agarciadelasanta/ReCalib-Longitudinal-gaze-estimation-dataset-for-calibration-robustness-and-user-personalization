"""Summarize validation_session_eval.py or leave_two_out_eval.py, next to the cross-session check of the calibration-only model.

For each held-out session s, the personalized model (with and without calibration on s) is scored on
s and on another session v of the same user that it was not trained on (the validation session for
validation_session_eval.py; the second left-out session for leave_two_out_eval.py; both called
"validation session" below). With
--cross, the calibration-only model of s (cross_session_eval.py) is added, scored on s, on v (the same
session as the personalized models) and on all the user's other sessions.

  table.csv      one row per (model, test session): mean, median, P5-P95 over session means, in degrees
                 and mm on the screen (the format of the paper's Table tab:session_check)
  contrasts.csv  other-session minus same-session error per model, and between models on session v:
                 mean, two-level bootstrap CI (users, then sessions), % of sessions, Wilcoxon p
  report.md      both tables

    python evaluation/summarize_validation_session.py --runs runs/valsess --cross runs/cross --sessions runs/sessions_all9_calibration_targets.csv --out runs/valsess/summary_all9
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_protocol import p_wilcoxon, two_level_bootstrap

IDS = {"user": str, "session": str, "val_session": str, "train_session": str, "test_session": str}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="--out-dir of validation_session_eval.py or leave_two_out_eval.py")
    ap.add_argument("--cross", type=Path, help="--out-dir of cross_session_eval.py (adds the calibration-only model)")
    ap.add_argument("--sessions", type=Path, help="CSV with user,session columns: keep only these held-out sessions")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    v = pd.concat([pd.read_csv(f, dtype=IDS) for f in sorted((args.runs / "pairs").glob("*.csv"))], ignore_index=True)
    if "other_session" in v:  # leave_two_out_eval.py: the unseen session is other_session, not val_session
        v = v.drop(columns="val_session").rename(columns={"other_session": "val_session"})
    if args.sessions:
        keep = pd.read_csv(args.sessions, dtype=str)
        keep = set(zip(keep.user, keep.session))
        v = v[[(u, s) in keep for u, s in zip(v.user, v.session)]]
    val_of = v.drop_duplicates(["user", "session"]).set_index(["user", "session"]).val_session

    # per held-out session: one value per (model, test) column
    cols = {}
    for (model, test_is), g in v.groupby(["model", "test_is"]):
        cols[(model, "same session" if test_is == "same" else "validation session")] = g.set_index(["user", "session"])
    if args.cross:
        c = pd.read_csv(args.cross / "summary" / "pairs.csv", dtype=IDS).rename(columns={"train_session": "session"})
        c = c[[k in val_of.index for k in zip(c.user, c.session)]]
        cols[("calib", "same session")] = c[c.test_session == c.session].set_index(["user", "session"])
        on_v = c[[val_of[(u, s)] == t for u, s, t in zip(c.user, c.session, c.test_session)]]
        cols[("calib", "validation session")] = on_v.set_index(["user", "session"])
        other = c[c.test_session != c.session].groupby(["user", "session"])[["mean_deg", "mean_mm"]].mean()
        cols[("calib", "all other sessions")] = other

    rows = []
    for (model, test), g in sorted(cols.items()):
        d, mm = g.mean_deg, g.mean_mm
        rows.append(dict(model=model, test=test, sessions=len(g), users=g.index.get_level_values("user").nunique(),
                         mean_deg=d.mean(), median_deg=d.median(), p5_deg=d.quantile(0.05), p95_deg=d.quantile(0.95),
                         mean_mm=mm.mean(), median_mm=mm.median()))
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "table.csv", index=False)

    def contrast(name, a, b):
        diff = (cols[b].mean_deg - cols[a].mean_deg).dropna()
        lo, hi = two_level_bootstrap(diff, args.bootstrap, rng)
        per_user = diff.groupby(level="user").mean()
        return dict(contrast=name, sessions=len(diff), mean_deg=diff.mean(), ci95_low=lo, ci95_high=hi,
                    pct_sessions_positive=100 * (diff > 0).mean(), users_positive=int((per_user > 0).sum()),
                    p_wilcoxon_sessions=p_wilcoxon(diff), p_wilcoxon_users=p_wilcoxon(per_user))

    models = sorted({m for m, _ in cols})
    crows = [contrast(f"{m}: validation session - same session", (m, "same session"), (m, "validation session"))
             for m in models]
    pairs_v = [("calib", "personal_calib"), ("calib", "personal"), ("personal", "personal_calib")]
    crows += [contrast(f"validation session: {a} - {b}", (b, "validation session"), (a, "validation session"))
              for a, b in pairs_v if a in models and b in models]
    contrasts = pd.DataFrame(crows)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)

    with open(args.out / "report.md", "w", encoding="utf-8") as fh:
        fh.write("# Personalized models on an unseen session of the same user (validation session)\n\n"
                 + table.round(3).to_markdown(index=False) + "\n\n## Contrasts (positive = first term larger)\n\n"
                 + contrasts.round(4).to_markdown(index=False) + "\n")
    print(table.round(2).to_string(index=False))
    print()
    print(contrasts.round(3).to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
