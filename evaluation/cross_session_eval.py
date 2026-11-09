"""Cross-session check: adapt the model on one session, test it on every session of the same user.

For one fold of the `calib` scenario (session s of user u), the ETH-XGaze model is fine-tuned on the
calibration task of session s exactly as in run_protocol_fold.py, and then scored on the 16-point test
tasks of every session of user u, session s included. The result is one row of the user's
session x session error matrix: same-session error on the diagonal, error on other sessions (other
days) elsewhere. Rows are written as aggregates per test session to <out-dir>/pairs/u<u>_s<s>.csv;
the fine-tuned model is deleted afterwards unless --keep-model.

    python evaluation/cross_session_eval.py --manifest runs/folds_all.json --out-dir runs/cross --list
    python evaluation/cross_session_eval.py --manifest runs/folds_all.json --out-dir runs/cross --index 5

Run one fold per process (e.g. a shell loop over --index); a fold whose pairs file exists is skipped.
Summarize with summarize_cross_session.py.
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
COLUMNS = ["user", "train_session", "test_session", "frames", "mean_deg", "median_deg", "mean_mm", "median_mm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest with calib folds (make_protocol_folds.py)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--index", type=int, help="position of the fold among the calib folds")
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
    folds = [f for f in manifest["folds"] if f["scenario"] == "calib"]
    if args.list:
        print(f"calib folds: {len(folds)} (--index 0..{len(folds) - 1})")
        return
    if args.index is None:
        ap.error("give --index or --list")
    fold = dict(folds[args.index], save_model=True)
    user, session = fold["test"][0]["users"][0], fold["test"][0]["sessions"][0]
    pairs_csv = args.out_dir / "pairs" / f"u{user}_s{session}.csv"
    if pairs_csv.exists():
        print(f"[skip] {pairs_csv.name} already done")
        return

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    h5_path = args.h5 or manifest["h5"]
    meta = load_meta(h5_path)
    cfg_all = json.loads(args.config.read_text(encoding="utf-8"))

    # 1. calibrate on this session's 9-point task (same procedure as the calib scenario)
    run_args = Namespace(out_dir=args.out_dir / "models", h5=args.h5, eth_checkpoint=args.eth_checkpoint,
                         seed=args.seed, num_workers=args.num_workers, no_test=True, force=True)
    run_fold(fold, manifest, meta, run_args, cfg_all, device)
    model_path = args.out_dir / "models" / fold["id"] / "model.pth"
    model = load_model(model_path, device)

    # 2. score it on the test tasks of every session of the same user
    idx = select(meta, [{"users": [user], "sessions": None, "tasks": "test"}])
    rows, pitchyaw, _ = predict(model, loader(h5_path, idx, None, cfg_all["eval_batch_size"], False,
                                              args.num_workers), device)
    result = score(h5_path, manifest["label"], rows, pitchyaw, meta)
    by_session = {}
    for r in result:
        by_session.setdefault(r["session"], []).append((r["ang_err_deg"], r["pog_err_mm"]))
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = pairs_csv.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for test_session in sorted(by_session):
            e = np.array(by_session[test_session], dtype=float)
            w.writerow(dict(user=user, train_session=session, test_session=test_session, frames=len(e),
                            mean_deg=e[:, 0].mean(), median_deg=np.median(e[:, 0]),
                            mean_mm=np.nanmean(e[:, 1]), median_mm=np.nanmedian(e[:, 1])))
    tmp.replace(pairs_csv)
    own = np.mean([x[0] for x in by_session[session]])
    other = np.mean([np.mean([x[0] for x in v]) for k, v in by_session.items() if k != session])
    print(f"  user {user} session {session}: same session {own:.2f} deg, other sessions {other:.2f} deg "
          f"({len(by_session) - 1} sessions)")
    if not args.keep_model:
        model_path.unlink()


if __name__ == "__main__":
    main()
