"""Train and score folds of the evaluation protocol (see make_protocol_folds.py).

Each fold starts from the ETH-XGaze checkpoint, or from the saved model of the fold named in its
"init" field, fine-tunes the whole network on its training set with the settings of its group in
protocol_config.json (L1 loss on normalized pitch/yaw, Adam, as in the ETH-XGaze training code) and
scores its test frames in the camera frame:

  ang_err_deg        angle to the manifest label (gaze_eth or gaze_released), in degrees
  ang_err_eth_deg    angle to unit(pog_mm - ETH face centre)
  ang_err_released_deg  angle to the released gaze.vector
  pog_err_mm         distance on the screen plane between pog_mm and the predicted ray
                     (from the ETH face centre, or the released origin for gaze_released)

Outputs go to <out-dir>/<fold id>/: predictions.csv, log.json and, when another fold starts from
it, model.pth. A fold with log.json is skipped unless --force.

    python evaluation/run_protocol_fold.py --manifest folds.json --out-dir runs --fold personal/u00_s03
    python evaluation/run_protocol_fold.py --manifest folds.json --out-dir runs --scenario personal --index 5
    python evaluation/run_protocol_fold.py --manifest folds.json --out-dir runs --scenario personal   # all
    python evaluation/run_protocol_fold.py --manifest folds.json --list

Run the scenarios in dependency order: zeroshot, calib, personal, louo, then personal_calib,
louo_calib (they start from the personal / louo models). --index fits job arrays
(e.g. --index $SLURM_ARRAY_TASK_ID).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from model import gaze_network
from protocol_common import (H5Faces, angle_deg, load_meta, pitchyaw_to_vector, scenario_group,
                             screen_hit, select, unit)

EVAL_DIR = Path(__file__).resolve().parent
COLUMNS = ["user", "session", "task", "sample", "image", "pred_pitch", "pred_yaw",
           "pred_x", "pred_y", "pred_z", "ang_err_deg", "ang_err_eth_deg", "ang_err_released_deg",
           "pog_pred_x_mm", "pog_pred_y_mm", "pog_gt_x_mm", "pog_gt_y_mm", "pog_err_mm"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path, device):
    model = gaze_network(pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model_state", state)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model.to(device)


def loader(h5_path, indices, label, batch_size, shuffle, workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(H5Faces(h5_path, indices, label), batch_size=batch_size, shuffle=shuffle,
                                       num_workers=workers, generator=g, persistent_workers=workers > 0,
                                       pin_memory=torch.cuda.is_available())


def predict(model, data, device):
    model.eval()
    rows, preds, labels = [], [], []
    with torch.no_grad():
        for x, y, idx in data:
            preds.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
            rows.append(idx.numpy())
            labels.append(y.numpy())
    return np.concatenate(rows), np.concatenate(preds), np.concatenate(labels)


def train(model, train_data, val_data, cfg, device, log):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg["freeze_bn"]:  # few frames: keep the pretrained BatchNorm statistics
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        losses, errs, n = 0.0, 0.0, 0
        for x, y, _ in train_data:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                pred = model(x)
                loss = F.l1_loss(pred.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            b = len(x)
            losses += float(loss) * b
            errs += float(angle_deg(pitchyaw_to_vector(pred.detach().float().cpu().numpy()),
                                    pitchyaw_to_vector(y.cpu().numpy())).sum())
            n += b
        sched.step()
        entry = dict(epoch=epoch + 1, train_loss=losses / n, train_err_deg=errs / n)
        if val_data is not None:
            _, vp, vy = predict(model, val_data, device)
            entry["val_err_deg"] = float(angle_deg(pitchyaw_to_vector(vp), pitchyaw_to_vector(vy)).mean())
        log["epochs"].append(entry)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))


def score(h5_path, label, rows, pitchyaw, meta):
    order = np.argsort(rows)
    rows, pitchyaw = rows[order], pitchyaw[order]
    with h5py.File(h5_path, "r") as h5:
        R, fc = h5["R"][rows].astype(np.float64), h5["face_center"][rows].astype(np.float64)
        pog, origin = h5["pog_mm"][rows].astype(np.float64), h5["origin_released"][rows].astype(np.float64)
        released_norm = h5["gaze_released"][rows]
    pred = np.einsum("nji,nj->ni", R, pitchyaw_to_vector(pitchyaw))          # R^T: back to the camera frame
    label_eth = unit(pog - fc)
    label_released = np.einsum("nji,nj->ni", R, pitchyaw_to_vector(released_norm))
    err_eth, err_rel = angle_deg(pred, label_eth), angle_deg(pred, label_released)
    hit = screen_hit(origin if label == "gaze_released" else fc, pred)
    out = []
    for k, r in enumerate(rows):
        out.append(dict(
            user=meta["user"][r], session=meta["session"][r], task=meta["task"][r], sample=meta["sample"][r],
            image=meta["image"][r], pred_pitch=pitchyaw[k, 0], pred_yaw=pitchyaw[k, 1],
            pred_x=pred[k, 0], pred_y=pred[k, 1], pred_z=pred[k, 2],
            ang_err_deg=err_rel[k] if label == "gaze_released" else err_eth[k],
            ang_err_eth_deg=err_eth[k], ang_err_released_deg=err_rel[k],
            pog_pred_x_mm=hit[k, 0], pog_pred_y_mm=hit[k, 1], pog_gt_x_mm=pog[k, 0], pog_gt_y_mm=pog[k, 1],
            pog_err_mm=float(np.linalg.norm(hit[k, :2] - pog[k, :2]))))
    return out


def run_fold(fold, manifest, meta, args, cfg_all, device):
    out = args.out_dir / fold["id"]
    if (out / "log.json").exists() and not args.force:
        print(f"[skip] {fold['id']} already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    h5_path, label = args.h5 or manifest["h5"], manifest["label"]
    group = scenario_group(fold["scenario"])
    cfg = cfg_all[group] if group else {}
    seed_everything(args.seed)
    init = args.eth_checkpoint if fold["init"] == "eth" else args.out_dir / fold["init"] / "model.pth"
    if not Path(init).exists():
        raise SystemExit(f"{fold['id']}: start model {init} not found (run {fold['init']} first)")
    print(f"[{fold['id']}] init={fold['init']} train={fold['n_train']} val={fold['n_val']} test={fold['n_test']}")
    model = load_model(init, device)
    log = dict(fold=fold["id"], scenario=fold["scenario"], init=fold["init"], label=label, config=cfg,
               seed=args.seed, n_train=fold["n_train"], n_val=fold["n_val"], n_test=fold["n_test"],
               device=str(device), epochs=[], started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()
    if fold["train"]:
        tr = loader(h5_path, select(meta, fold["train"]), label, cfg["batch_size"], True, args.num_workers, args.seed)
        va = (loader(h5_path, select(meta, fold["val"]), label, cfg_all["eval_batch_size"], False, args.num_workers)
              if fold["val"] else None)
        train(model, tr, va, cfg, device, log)
    if fold["save_model"]:
        torch.save({"model_state": model.state_dict()}, out / "model.pth")
    if not args.no_test:
        te = loader(h5_path, select(meta, fold["test"]), None, cfg_all["eval_batch_size"], False, args.num_workers)
        rows, py, _ = predict(model, te, device)
        result = score(h5_path, label, rows, py, meta)
        with open(out / "predictions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(result)
        log["test_mean_err_deg"] = float(np.mean([r["ang_err_deg"] for r in result]))
        log["test_mean_err_mm"] = float(np.nanmean([r["pog_err_mm"] for r in result]))
        print(f"  test: {log['test_mean_err_deg']:.3f} deg, {log['test_mean_err_mm']:.1f} mm")
    log["seconds"] = round(time.time() - t0, 1)
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--h5", help="override the H5 path stored in the manifest")
    ap.add_argument("--fold", help="fold id, e.g. personal/u00_s03")
    ap.add_argument("--scenario", help="run the folds of one scenario (all of them, or --index)")
    ap.add_argument("--index", type=int, help="position of the fold within --scenario (job arrays)")
    ap.add_argument("--list", action="store_true", help="print the folds and exit")
    ap.add_argument("--config", type=Path, default=EVAL_DIR / "protocol_config.json")
    ap.add_argument("--eth-checkpoint", type=Path, default=EVAL_DIR / "ckpt" / "epoch_24_ckpt.pth.tar")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-test", action="store_true", help="train only (hyperparameter tuning on validation)")
    ap.add_argument("--force", action="store_true", help="rerun folds that already have a log.json")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = manifest["folds"]
    if args.list:
        for sc in manifest["scenarios"]:
            fs = [f for f in folds if f["scenario"] == sc]
            print(f"{sc}: {len(fs)} folds (--index 0..{len(fs) - 1})")
        return
    if args.out_dir is None:
        ap.error("--out-dir is required")
    if args.fold:
        todo = [f for f in folds if f["id"] == args.fold]
    elif args.scenario:
        todo = [f for f in folds if f["scenario"] == args.scenario]
        todo = [todo[args.index]] if args.index is not None else todo
    else:
        ap.error("give --fold or --scenario")
    if not todo:
        raise SystemExit("no matching fold")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    meta = load_meta(args.h5 or manifest["h5"])
    cfg_all = json.loads(args.config.read_text(encoding="utf-8"))
    for f in todo:
        run_fold(f, manifest, meta, args, cfg_all, device)


if __name__ == "__main__":
    main()
