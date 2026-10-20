"""Download the authors' preprocessed AVA validation tracks (27.4 GB) and mark them complete.

    python tools/download_ava_val.py                  # stream-download + extract to data/ava_val/
    python tools/download_ava_val.py --mark-complete  # you extracted the archive yourself (tar exited OK)

The archive is streamed straight into the extractor, so no 27 GB .tar.gz is kept on disk. A marker file
data/ava_val.COMPLETE (with the track count) is written ONLY after the stream ended cleanly - eval_ava.py
refuses to run without it. That is the fix for LART issue #29: an interrupted download silently produced a
low mAP because LART cached the incomplete file list.

Each .pkl is one person tracklet produced by PHALP on AVA (see docs/01 and DATASET.md upstream).
Other splits (ava_train 4 parts, ~93 GB; kinetics_train 13 parts, ~210 GB) are only needed for training.
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.dropbox.com/s/gqjfd6f4e9lvl76/ava_val.tar.gz"


class _Progress:
    def __init__(self, raw, total):
        self.raw, self.total, self.done, self.t0 = raw, total, 0, time.time()

    def read(self, n=-1):
        b = self.raw.read(n)
        self.done += len(b)
        if self.done // (256 << 20) != (self.done - len(b)) // (256 << 20):
            mbps = self.done / 1e6 / max(1e-6, time.time() - self.t0)
            print(f"\r  {self.done / 1e9:6.1f} / {self.total / 1e9:.1f} GB  ({mbps:.0f} MB/s)", end="", flush=True)
        return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--mark-complete", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    split = out / "ava_val"
    marker = out / "ava_val.COMPLETE"

    if args.mark_complete:
        n = len([f for f in os.listdir(split) if f.endswith(".pkl")])
        marker.write_text(f"{n} tracks\n")
        print(f"marked {split} complete: {n} tracks")
        return

    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("content-length", 0))
        print(f"streaming {total / 1e9:.1f} GB from {URL}")
        n = 0
        with tarfile.open(fileobj=_Progress(r, total), mode="r|gz") as tar:
            for member in tar:
                tar.extract(member, out)
                n += member.isfile() and member.name.endswith(".pkl")
    print(f"\nextracted {n} tracks")
    marker.write_text(f"{n} tracks\n")


if __name__ == "__main__":
    sys.exit(main())
