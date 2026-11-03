"""Write the fold manifest of the evaluation protocol (Usage Notes, Figure 5) from the protocol H5.

Every fold is a training set, an optional validation set and a test set, each written as clauses
over user / session / task (resolved against the H5 at run time). Test sets always contain only the
16-point test tasks of the held-out session(s). Scenarios:

  zeroshot        (i)  ETH-XGaze checkpoint, no training                          1 fold
  louo            (ii) fine-tune on the other users (all tasks)                   1 fold per user
  personal        (iii) fine-tune on the same user's other sessions (all tasks)   1 fold per session
  calib           (iv) fine-tune on the session's calibration task only           1 fold per session
  personal_calib  personal model of that session + its calibration task          1 fold per session
  louo_calib      louo model of that user + the session's calibration task       1 fold per session
  personal_prev<k> (optional, --curve) the k sessions recorded just before       1 fold per session

    python evaluation/make_protocol_folds.py --h5 protocol.h5 --out folds.json

Choices recorded in the manifest (identical for all folds):
  --label          gaze_eth (default; unit(pog_mm - ETH face centre), image + intrinsics only) or
                   gaze_released (the released MediaPipe-based gaze.vector)
  --personal-pool  user (default; only the held-out user's other sessions, scenario iii) or
                   all (other users + the user's other sessions)
  --personal-order loso (default; all other sessions) or chronological (earlier sessions only)
  --test-sessions  N held-out sessions in total instead of all of them (spread evenly over the users
                   and, within a user, over the recording period); every session-level scenario uses
                   the same held-out sessions, so they can be compared pairwise. Training sets still
                   use all other sessions.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from protocol_common import load_meta, select

ALL = ("zeroshot", "louo", "personal", "calib", "personal_calib", "louo_calib")
# leave-one-user-out is the domain control, run last: --scenarios with louo,louo_calib added
DEFAULT = ("zeroshot", "calib", "personal", "personal_calib")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", default=",".join(DEFAULT), help=f"comma list from {ALL}")
    ap.add_argument("--label", choices=("gaze_eth", "gaze_released"), default="gaze_eth")
    ap.add_argument("--personal-pool", choices=("user", "all"), default="user")
    ap.add_argument("--personal-order", choices=("loso", "chronological"), default="loso")
    ap.add_argument("--val-sessions", type=int, default=0,
                    help="personal*: hold out this many training sessions as validation (their test tasks); "
                         "chronological -> the most recent ones, loso -> random")
    ap.add_argument("--val-users", type=int, default=0, help="louo: hold out this many training users as validation")
    ap.add_argument("--test-sessions", type=int, help="held-out sessions in total (default: all)")
    ap.add_argument("--curve", default="", help="learning curve, e.g. 1,2,4,8 -> scenarios personal_prev<k>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = load_meta(args.h5)
    rng = np.random.default_rng(args.seed)
    valid = meta["valid"]
    users = sorted(set(meta["user"][valid]))
    sessions = {u: sorted(set(meta["session"][valid & (meta["user"] == u)])) for u in users}
    held_out = {u: list(v) for u, v in sessions.items()}
    if args.test_sessions:
        # spread N over the users (the remainder goes to the users with most sessions), then pick
        # evenly spaced sessions in time within each user
        per_user = {u: args.test_sessions // len(users) for u in users}
        for u in sorted(users, key=lambda u: -len(sessions[u]))[: args.test_sessions % len(users)]:
            per_user[u] += 1
        for u in users:
            k = min(per_user[u], len(sessions[u]))
            pick = np.unique(np.round(np.linspace(0, len(sessions[u]) - 1, k)).astype(int)) if k else []
            held_out[u] = [sessions[u][i] for i in pick]
    scenarios = args.scenarios.split(",") + [f"personal_prev{k}" for k in args.curve.split(",") if k]

    def clause(us, ss, tasks):
        return {"users": list(us), "sessions": None if ss is None else list(ss), "tasks": tasks}

    def personal_sets(u, s):
        others = [x for x in sessions[u] if x != s]
        if args.personal_order == "chronological":
            others = [x for x in others if x < s]
        val = []
        if args.val_sessions and len(others) > args.val_sessions:
            val = (others[-args.val_sessions:] if args.personal_order == "chronological"
                   else sorted(rng.choice(others, args.val_sessions, replace=False).tolist()))
            others = [x for x in others if x not in val]
        train = [clause([u], others, "all")] if others else []
        if args.personal_pool == "all":
            train.append(clause([x for x in users if x != u], None, "all"))
        return train, [clause([u], val, "test")] if val else []

    folds = []
    for sc in scenarios:
        if sc == "zeroshot":
            folds.append(dict(id="zeroshot/all", scenario=sc, init="eth", train=[], val=[],
                              test=[clause(users, None, "test")]))
        elif sc == "louo":
            for u in users:
                others = [x for x in users if x != u]
                val = sorted(rng.choice(others, args.val_users, replace=False).tolist()) if args.val_users else []
                folds.append(dict(id=f"louo/u{u}", scenario=sc, init="eth",
                                  train=[clause([x for x in others if x not in val], None, "all")],
                                  val=[clause(val, None, "test")] if val else [],
                                  test=[clause([u], held_out[u], "test")]))
        else:
            for u in users:
                for s in held_out[u]:
                    test = [clause([u], [s], "test")]
                    calib = [clause([u], [s], "calibration")]
                    fid = f"{sc}/u{u}_s{s}"
                    if sc == "personal":
                        train, val = personal_sets(u, s)
                        folds.append(dict(id=fid, scenario=sc, init="eth", train=train, val=val, test=test))
                    elif sc == "calib":
                        folds.append(dict(id=fid, scenario=sc, init="eth", train=calib, val=[], test=test))
                    elif sc == "personal_calib":
                        folds.append(dict(id=fid, scenario=sc, init=f"personal/u{u}_s{s}", train=calib, val=[],
                                          test=test))
                    elif sc == "louo_calib":
                        folds.append(dict(id=fid, scenario=sc, init=f"louo/u{u}", train=calib, val=[], test=test))
                    elif sc.startswith("personal_prev"):
                        k = int(sc[len("personal_prev"):])
                        before = [x for x in sessions[u] if x < s][-k:]
                        if len(before) == k:
                            folds.append(dict(id=fid, scenario=sc, init="eth",
                                              train=[clause([u], before, "all")], val=[], test=test))
                    else:
                        raise SystemExit(f"unknown scenario {sc}")

    # sizes, and drop folds that cannot be trained or scored
    kept, skipped = [], []
    for f in folds:
        f.update({f"n_{k}": int(len(select(meta, f[k]))) for k in ("train", "val", "test")})
        if f["n_test"] == 0 or (f["scenario"] != "zeroshot" and f["n_train"] == 0):
            skipped.append(f["id"])
        else:
            kept.append(f)
    ids = {f["id"] for f in kept}
    for f in kept:
        if f["init"] != "eth" and f["init"] not in ids:
            raise SystemExit(f"{f['id']} starts from {f['init']}, which is not in the manifest")
    inits = {f["init"] for f in kept}
    for f in kept:
        f["save_model"] = f["id"] in inits

    manifest = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), h5=str(args.h5), label=args.label,
                    personal_pool=args.personal_pool, personal_order=args.personal_order,
                    val_sessions=args.val_sessions, val_users=args.val_users, seed=args.seed,
                    test_sessions={u: v for u, v in held_out.items()},
                    scenarios=scenarios, n_users=len(users), n_sessions=sum(map(len, sessions.values())),
                    skipped=skipped, folds=kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(kept)} folds written to {args.out} ({len(skipped)} skipped: no training or test data)")
    for sc in scenarios:
        fs = [f for f in kept if f["scenario"] == sc]
        if fs:
            print(f"  {sc:18s} {len(fs):4d} folds, train median {int(np.median([f['n_train'] for f in fs])):6d}"
                  f" frames, test median {int(np.median([f['n_test'] for f in fs])):6d}")


if __name__ == "__main__":
    main()
