"""Score the personalized models on another, unseen session of the same user: the validation session.

Each `personal` fold (held-out session s of user u) trains on the user's other sessions except one, the
validation session v, whose test tasks are used only for early stopping. Session v is therefore a
session of the same user that the model was never trained on. For every `personal_calib` fold this
script

  1. scores the saved `personal` model (no calibration) on the test tasks of s and of v;
  2. calibrates that model on the calibration task of s exactly as in run_protocol_fold.py, and scores
     the result on the test tasks of s and of v.

This gives, for "personalization" and "personalization + session calibration", the error on the
calibrated session and on another session of the same user. Rows are written per (model, test session)
to <out-dir>/pairs/u<u>_s<s>.csv; the calibrated model is deleted afterwards unless --keep-model.
Caveat: the number of training epochs of the personal model was chosen on session v.

    python evaluation/validation_session_eval.py --manifest runs/folds_all.json --models runs/out --out-dir runs/valsess --list
    python evaluation/validation_session_eval.py --manifest runs/folds_all.json --models runs/out --out-dir runs/valsess --index 0

Run one fold per process (e.g. a shell loop over --index); a fold whose pairs file exists is skipped.
"""
import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from protocol_common import load_meta, select
from run_protocol_fold import load_model, loader, predict, run_fold, score

EVAL_DIR = Path(__file__).resolve().parent
COLUMNS = ["user", "session", "val_session", "model", "test_session", "test_is", "frames",
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
    ap.add_argument("--models", type=Path, required=True, help="--out-dir of run_protocol_fold.py (holds personal/<fold>/model.pth)")
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
    base = by_id[folds[args.index]["init"]]
    user, session = base["test"][0]["users"][0], base["test"][0]["sessions"][0]
    val_session = base["val"][0]["sessions"][0]
    pairs_csv = args.out_dir / "pairs" / f"u{user}_s{session}.csv"
    if pairs_csv.exists():
        print(f"[skip] {pairs_csv.name} already done")
        return
    personal_model = (args.models / base["id"] / "model.pth").resolve()
    if not personal_model.exists():
        raise SystemExit(f"{personal_model} not found")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    h5_path = args.h5 or manifest["h5"]
    meta = load_meta(h5_path)
    cfg_all = json.loads(args.config.read_text(encoding="utf-8"))
    bs, label = cfg_all["eval_batch_size"], manifest["label"]
    results = {}

    # 1. personalized model, no calibration
    model = load_model(personal_model, device)
    results["personal"] = score_sessions(model, h5_path, label, meta, user, [session, val_session], bs,
                                         args.num_workers, device)

    # 2. calibrate it on session s (same procedure as the personal_calib scenario)
    fold = dict(folds[args.index], init=str(personal_model.parent), save_model=True)
    run_args = Namespace(out_dir=args.out_dir / "models", h5=args.h5, eth_checkpoint=args.eth_checkpoint,
                         seed=args.seed, num_workers=args.num_workers, no_test=True, force=True)
    run_fold(fold, manifest, meta, run_args, cfg_all, device)
    model_path = args.out_dir / "models" / fold["id"] / "model.pth"
    model = load_model(model_path, device)
    results["personal_calib"] = score_sessions(model, h5_path, label, meta, user, [session, val_session], bs,
                                               args.num_workers, device)

    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = pairs_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for name, per_session in results.items():
            for test_session, e in sorted(per_session.items()):
                w.writerow(dict(user=user, session=session, val_session=val_session, model=name,
                                test_session=test_session, test_is="same" if test_session == session else "other",
                                frames=len(e), mean_deg=e[:, 0].mean(), median_deg=np.median(e[:, 0]),
                                mean_mm=np.nanmean(e[:, 1]), median_mm=np.nanmedian(e[:, 1])))
    tmp.replace(pairs_csv)
    for name, per_session in results.items():
        print(f"  user {user} session {session} {name}: same {per_session[session][:, 0].mean():.2f} deg, "
              f"session {val_session} {per_session[val_session][:, 0].mean():.2f} deg")
    if not args.keep_model:
        model_path.unlink()


if __name__ == "__main__":
    main()
