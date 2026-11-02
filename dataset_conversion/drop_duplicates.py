"""Remove duplicate frames from a release written by standardize_release.py.

A frame is a duplicate when its image is byte-identical (MD5) to the previous frame of the
same task: the capture loop read the same buffered camera image twice, so it carries no new
information. The first frame of each run of identical images is kept.

Duplicates are moved (image + JSON) out of the release into QUARANTINE, mirroring the release
layout, and listed in QUARANTINE/duplicates_removed.csv. recalib_index.csv, sessions.csv
(n_frames, n_duplicates_removed) and PRIVATE/conversion_report.json are updated. Sample
indices are not renumbered, so indices within a task are not always consecutive.

Safe to re-run: already moved duplicates are skipped.

Usage:
    python drop_duplicates.py --release F:/recalib_release/ReCalib \
        --private F:/recalib_relabel/release_private --quarantine F:/recalib_relabel/release_private/duplicates
"""
import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

IDX_RE = re.compile(r"_img-(\d+)\.png$")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def find_duplicates(manifest):
    """Release-relative PNG paths whose MD5 equals that of the previous frame in the same task."""
    by_task = defaultdict(list)
    for r in manifest:
        rel = Path(r["release_png"])
        by_task[rel.parent].append((int(IDX_RE.search(rel.name).group(1)), rel, r["md5"]))
    dups = []
    for frames in by_task.values():
        frames.sort()
        for (_, _, prev_md5), (_, rel, md5) in zip(frames, frames[1:]):
            if md5 == prev_md5:
                dups.append((rel, md5))
    return dups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--private", required=True, type=Path, help="folder with manifest.csv and conversion_report.json")
    ap.add_argument("--quarantine", required=True, type=Path)
    args = ap.parse_args()

    dups = find_duplicates(read_csv(args.private / "manifest.csv"))
    print(f"{len(dups)} duplicate frames found")

    moved = already = 0
    removed_rows = []
    for rel, md5 in dups:
        for src in (args.release / rel, args.release / rel.with_suffix(".json")):
            dst = args.quarantine / src.relative_to(args.release)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += src.suffix == ".png"
            elif dst.exists():
                already += src.suffix == ".png"
            else:
                raise FileNotFoundError(f"{src} is neither in the release nor in quarantine")
        removed_rows.append({"image": str(rel).replace("\\", "/"), "md5": md5})
    args.quarantine.mkdir(parents=True, exist_ok=True)
    write_csv(args.quarantine / "duplicates_removed.csv", removed_rows, ["image", "md5"])
    print(f"moved {moved} duplicates to {args.quarantine} ({already} already there)")

    # index: drop removed samples
    dup_images = {r["image"] for r in removed_rows}
    index = read_csv(args.release / "recalib_index.csv")
    kept_index = [r for r in index if r["image"] not in dup_images]
    write_csv(args.release / "recalib_index.csv", kept_index, list(index[0].keys()))

    # sessions: released frame counts and removed duplicates per task
    per_task_dups = defaultdict(int)
    for r in removed_rows:
        u, s, t = Path(r["image"]).name.split("_img-")[0].split("_")
        per_task_dups[(u, s, t)] += 1
    per_task_frames = defaultdict(int)
    for r in kept_index:
        per_task_frames[(r["user_id"].zfill(2), r["session_id"].zfill(2), r["task_id"].zfill(2))] += 1
    sessions = read_csv(args.release / "sessions.csv")
    for row in sessions:
        key = (row["user_id"].zfill(2), row["session_id"].zfill(2), row["task_id"].zfill(2))
        row["n_frames"] = per_task_frames[key]
        row["n_duplicates_removed"] = per_task_dups[key]
    fields = [f for f in sessions[0].keys()]
    write_csv(args.release / "sessions.csv", sessions, fields)

    # report
    rep_path = args.private / "conversion_report.json"
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    discarded = sum(r["discarded"] == "True" for r in kept_index)
    report.update(duplicates_removed=len(removed_rows), samples_released=len(kept_index),
                  kept_after_duplicate_removal=len(kept_index) - discarded,
                  discarded_after_duplicate_removal=discarded)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"release now has {len(kept_index)} samples ({len(kept_index) - discarded} kept, {discarded} flagged as discarded)")


if __name__ == "__main__":
    main()
