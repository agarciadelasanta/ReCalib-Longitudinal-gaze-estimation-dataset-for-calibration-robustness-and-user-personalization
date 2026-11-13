"""Leave two sessions out: personalized models scored on the held-out session and on another unseen session.

For held-out session s of user u (one `personal_calib` fold of the manifest), one more session t of the
same user is drawn at random (seeded by the fold id) and removed from the personalization data. Then

  1. a `personal` model is trained exactly as in run_protocol_fold.py on the user's other sessions minus
     t, with the same validation session v for early stopping (v is not t);
  2. it is scored on the test tasks of s and of t;
  3. it is calibrated on the calibration task of s exactly as in the personal_calib scenario, and scored
     on the test tasks of s and of t.

Session t is never used for training or for model selection, so the error on t is the error of the
personalized (and session-calibrated) model on another session of the same user. Rows are written to
<out-dir>/pairs/u<u>_s<s>.csv; models are deleted afterwards unless --keep-model.

    python evaluation/leave_two_out_eval.py --manifest runs/folds_all.json --out-dir runs/l2o --list
    python evaluation/leave_two_out_eval.py --manifest runs/folds_all.json --out-dir runs/l2o --index 0

Run one fold per process (e.g. a shell loop over --index); a fold whose pairs file exists is skipped, and
a personal model already trained for it is reused.
"""
import argparse
import csv
import json
import zlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from protocol_common import load_meta, select
from run_protocol_fold import load_model, loader, predict, run_fold, score

EVAL_DIR = Path(__file__).resolve().parent
COLUMNS = ["user", "session", "other_session", "val_session", "model", "test_session", "test_is", "frames",
           "mean_deg", "median_deg", "mean_mm", "median_mm"]


def score_sessions(model, h5_path, label, meta, user, sessions, batch_size, workers, device):
    idx = select(meta, [{"users": [user], "sessions": list(sessions), "tasks": "test"}])
    rows, pitchyaw, _ = predict(model, loader(h5_path, idx, None, batch_size, False, workers), device)
    out = {}
    for r in score(h5_path, label, rows, pitchyaw, meta):
        out.setdefault(r["session"], []).append((r["ang_err_deg"], r["pog_err_mm"]))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest with personal and personal_calib folds")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--index", type=int, help="position of the fold among the personal_calib folds")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--h5", help="override the H5 path stored in the manifest")
    ap.add_argument("--config", type=Path, default=EVAL_DIR / "protocol_config.json")
    ap.add_argument("--eth-checkpoint", type=Path, default=EVAL_DIR / "ckpt" / "epoch_24_ckpt.pth.tar")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-model", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in manifest["folds"]}
    folds = [f for f in manifest["folds"] if f["scenario"] == "personal_calib"]
    if args.list:
        print(f"personal_calib folds: {len(folds)} (--index 0..{len(folds) - 1})")
        return
    if args.index is None:
        ap.error("give --index or --list")
    calib_fold = folds[args.index]
    base = by_id[calib_fold["init"]]
    user, session = base["test"][0]["users"][0], base["test"][0]["sessions"][0]
    val_session = base["val"][0]["sessions"][0]
    pairs_csv = args.out_dir / "pairs" / f"u{user}_s{session}.csv"
    if pairs_csv.exists():
        print(f"[skip] {pairs_csv.name} already done")
        return

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    h5_path = args.h5 or manifest["h5"]
    meta = load_meta(h5_path)
    cfg_all = json.loads(args.config.read_text(encoding="utf-8"))
    bs, label = cfg_all["eval_batch_size"], manifest["label"]

    # the second left-out session: a training session of the personal fold that has test frames
    (train_clause,) = base["train"]
    candidates = [s for s in train_clause["sessions"]
                  if len(select(meta, [{"users": [user], "sessions": [s], "tasks": "test"}]))]
    rng = np.random.default_rng(zlib.crc32(base["id"].encode()) + args.seed)
    other = str(rng.choice(sorted(candidates)))
    train = [dict(train_clause, sessions=[s for s in train_clause["sessions"] if s != other])]
    models_dir = (args.out_dir / "models").resolve()

    # 1. personal model without sessions s and t
    personal = dict(base, id=f"personal_l2o/u{user}_s{session}", train=train, save_model=True,
                    n_train=int(len(select(meta, train))), other_session=other)
    personal_model = models_dir / personal["id"] / "model.pth"
    run_args = Namespace(out_dir=models_dir, h5=args.h5, eth_checkpoint=args.eth_checkpoint, seed=args.seed,
                         num_workers=args.num_workers, no_test=True, force=not personal_model.exists())
    run_fold(personal, manifest, meta, run_args, cfg_all, device)
    results = {"personal": score_sessions(load_model(personal_model, device), h5_path, label, meta, user,
                                          [session, other], bs, args.num_workers, device)}

    # 2. calibrate it on session s
    calib = dict(calib_fold, id=f"personal_calib_l2o/u{user}_s{session}", init=str(personal_model.parent),
                 save_model=True)
    run_fold(calib, manifest, meta, Namespace(**dict(vars(run_args), force=True)), cfg_all, device)
    calib_model = models_dir / calib["id"] / "model.pth"
    results["personal_calib"] = score_sessions(load_model(calib_model, device), h5_path, label, meta, user,
                                               [session, other], bs, args.num_workers, device)

    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = pairs_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for name, per_session in results.items():
            for test_session, e in sorted(per_session.items()):
                w.writerow(dict(user=user, session=session, other_session=other, val_session=val_session,
                                model=name, test_session=test_session,
                                test_is="same" if test_session == session else "other", frames=len(e),
                                mean_deg=e[:, 0].mean(), median_deg=np.median(e[:, 0]),
                                mean_mm=np.nanmean(e[:, 1]), median_mm=np.nanmedian(e[:, 1])))
    tmp.replace(pairs_csv)
    for name, per_session in results.items():
        print(f"  user {user} session {session} {name}: same {per_session[session][:, 0].mean():.2f} deg, "
              f"other session {other} {per_session[other][:, 0].mean():.2f} deg")
    if not args.keep_model:
        for p in (personal_model, calib_model):
            p.unlink()


if __name__ == "__main__":
    main()
