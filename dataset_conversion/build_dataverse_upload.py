"""Build the folder with every file uploaded to the Dataverse record, from the release folder.

    python dataset_conversion/build_dataverse_upload.py --release-root F:/recalib_release/ReCalib \
        --out H:/recalib_dataverse_upload --sample-session 07_00

Writes to --out:
  * the root files of the release, copied as they are: README.md, the DUA PDF, setup.json,
    camera_intrinsics.npz, recalib_index.csv, sessions.csv, incomplete_tasks.csv;
  * user_XX.zip, one per participant (restricted on Dataverse): user_XX/session_*/task_*/*.{png,json},
    stored without compression (the PNG files are already compressed). Unzipping all of them into one
    folder together with the root files gives the release layout;
  * recalib_sample.tar (open): one complete session, user_UU/session_UU_SS/..., plus README.md,
    setup.json, camera_intrinsics.npz and recalib_index.csv restricted to that session, so the code
    repository can be used on it directly;
  * with --wrap-zips-in-tar: user_XX.zip.tar, an uncompressed tar holding only user_XX.zip, so that
    Dataverse stores it as one file (it tries to unpack every uploaded .zip and refuses those with more
    than 1000 files). Each zip is deleted once its tar is written and checked;
  * checksums_sha256.txt: SHA-256 of every other file in --out, computed by reading them back (for
    the tars, while they are written).
Every archive is checked against recalib_index.csv: each indexed sample must be present with its
PNG and JSON. Archives already complete in --out are skipped, so the script can be resumed.
"""
import argparse
import csv
import hashlib
import io
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT_FILES = ["README.md", "Data Usage Agreement (DUA) of ReCalib Dataset.pdf", "setup.json", "camera_intrinsics.npz",
              "recalib_index.csv", "sessions.csv", "incomplete_tasks.csv"]
SAMPLE_EXTRA = ["README.md", "setup.json", "camera_intrinsics.npz"]


def sample_files(index, root):
    """{user: [relative paths of the PNG and JSON of every indexed sample]}"""
    out = defaultdict(list)
    with open(index, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            png = r["image"].replace("\\", "/")
            out[r["user_id"]] += [png, png[:-4] + ".json"]
    return out


def build_zip(root, user, files, dst):
    tmp = dst.with_suffix(".zip.part")
    start = time.time()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, rel in enumerate(sorted(files), 1):
            z.write(root / rel, rel)
            if k % 20000 == 0:
                print(f"  user_{user}: {k}/{len(files)} files ({time.time() - start:.0f} s)", flush=True)
    with zipfile.ZipFile(tmp) as z:                       # completeness check against the index
        names = set(z.namelist())
    missing = set(files) - names
    if missing or len(names) != len(files):
        raise RuntimeError(f"user_{user}.zip: {len(missing)} missing, {len(names)} entries for {len(files)} files")
    tmp.replace(dst)
    print(f"user_{user}.zip: {len(files)} files, {dst.stat().st_size / 1e9:.1f} GB, {time.time() - start:.0f} s", flush=True)


def build_sample(root, session, dst):
    user = session.split("_")[0]
    prefix = f"user_{user}/session_{session}/"
    with open(root / "recalib_index.csv", newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)
    rows = [lines[0]] + [l for l in lines[1:] if l.replace("\\", "/").startswith(prefix)]
    files = []
    for l in rows[1:]:
        png = l.split(",", 1)[0].replace("\\", "/")
        files += [png, png[:-4] + ".json"]
    if not files:
        raise RuntimeError(f"no samples for session {session}")
    tmp = dst.with_suffix(".tar.part")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as t:
        for name in SAMPLE_EXTRA:
            t.add(root / name, name)
        data = "".join(rows).encode("utf-8")
        info = tarfile.TarInfo("recalib_index.csv")
        info.size, info.mtime = len(data), int((root / "recalib_index.csv").stat().st_mtime)
        t.addfile(info, io.BytesIO(data))
        for rel in sorted(files):
            t.add(root / rel, rel)
    with tarfile.open(tmp) as t:
        names = set(t.getnames())
    missing = set(files) - names
    if missing:
        raise RuntimeError(f"recalib_sample.tar: {len(missing)} files missing")
    tmp.replace(dst)
    print(f"recalib_sample.tar: session {session}, {len(rows) - 1} samples, {dst.stat().st_size / 1e9:.2f} GB", flush=True)


class HashingWriter:
    """File object that writes to fh and updates a SHA-256 of everything written."""
    def __init__(self, fh):
        self.fh, self.h, self.pos = fh, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.pos += len(b)
        return self.fh.write(b)

    def tell(self):
        return self.pos


def wrap_zip(src, dst):
    """Write dst = tar holding only src (as its base name); return the SHA-256 of dst."""
    tmp = dst.with_name(dst.name + ".part")
    start = time.time()
    with open(tmp, "wb") as raw:
        w = HashingWriter(raw)
        with tarfile.open(fileobj=w, mode="w", format=tarfile.PAX_FORMAT) as t:
            t.add(src, src.name)
    with tarfile.open(tmp) as t:                          # one member, same size as the zip
        members = t.getmembers()
    if len(members) != 1 or members[0].name != src.name or members[0].size != src.stat().st_size:
        raise RuntimeError(f"{dst.name}: unexpected content {[(m.name, m.size) for m in members]}")
    if tmp.stat().st_size != w.pos:
        raise RuntimeError(f"{dst.name}: {tmp.stat().st_size} bytes on disk, {w.pos} written")
    tmp.replace(dst)
    print(f"{dst.name}: {dst.stat().st_size / 1e9:.1f} GB, {time.time() - start:.0f} s", flush=True)
    return w.h.hexdigest()


def sha256(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample-session", required=True, help="UU_SS, e.g. 07_00")
    ap.add_argument("--wrap-zips-in-tar", action="store_true", help="upload user_XX.zip.tar instead of user_XX.zip")
    args = ap.parse_args(argv)
    root, out = args.release_root, args.out
    out.mkdir(parents=True, exist_ok=True)

    for name in ROOT_FILES:
        shutil.copy2(root / name, out / name)
    print(f"root files copied: {', '.join(ROOT_FILES)}", flush=True)

    if not (out / "recalib_sample.tar").exists():
        build_sample(root, args.sample_session, out / "recalib_sample.tar")
    for user, files in sorted(sample_files(root / "recalib_index.csv", root).items()):
        dst = out / f"user_{user}.zip"
        if dst.exists() or dst.with_name(dst.name + ".tar").exists():
            print(f"user_{user}.zip exists, skipped", flush=True)
            continue
        build_zip(root, user, files, dst)

    known = {}
    if args.wrap_zips_in_tar:
        for src in sorted(out.glob("user_*.zip")):
            dst = src.with_name(src.name + ".tar")
            if not dst.exists():
                known[dst.name] = wrap_zip(src, dst)
            src.unlink()                                  # the tar holds it, checked in wrap_zip
            print(f"{src.name} deleted", flush=True)

    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "checksums_sha256.txt" and not p.name.endswith(".part"))
    with open(out / "checksums_sha256.txt", "w", encoding="utf-8", newline="\n") as fh:
        for name in names:
            fh.write(f"{known.get(name) or sha256(out / name)}  {name}\n")
            print(f"sha256 {name}", flush=True)
    print(f"done: {len(names)} files + checksums_sha256.txt in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
