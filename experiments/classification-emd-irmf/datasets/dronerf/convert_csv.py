"""Convert the extracted DroneRF CSV segments into a float32 .npy cache.

Each DroneRF segment file is one comma-separated line of 10^7 integer-valued
samples written as floats ("15.000000,-8.000000,..."). Parsing it takes about
2 s, so we parse every file once and keep a float32 copy (exact for these
integers) in `<DATA_ROOT>/dronerf/raw/npy/<BUI><band>_<index>.npy`.

The extracted folder is untrusted data: this script only reads text from it.
Run with `python -I`:

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/dronerf/convert_csv.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
NAME_RE = re.compile(r"^(?P<bui>[01]{5})(?P<band>[LH])_(?P<idx>\d+)\.csv$")


def convert_one(args: tuple[str, str]) -> dict:
    src, dst = args
    t0 = time.perf_counter()
    x = np.fromfile(src, sep=",", dtype=np.float64)
    is_int = bool(np.all(x == np.round(x)))
    out = x.astype(np.float32)
    tmp = dst + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, dst)
    return {
        "src": src,
        "dst": dst,
        "n": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "all_integer": is_int,
        "n_zero": int((x == 0).sum()),
        "seconds": time.perf_counter() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DATA_ROOT / "dronerf/raw/extracted/rar")
    p.add_argument("--dst", type=Path, default=DATA_ROOT / "dronerf/raw/npy")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    a.dst.mkdir(parents=True, exist_ok=True)
    jobs = []
    seen = {}
    for csv in sorted(a.src.rglob("*.csv")):
        m = NAME_RE.match(csv.name)
        if m is None:
            raise SystemExit(f"unexpected file name: {csv}")
        if csv.name in seen:
            raise SystemExit(f"duplicate segment file: {csv} and {seen[csv.name]}")
        seen[csv.name] = csv
        dst = a.dst / (csv.stem + ".npy")
        if dst.exists() and not a.overwrite:
            continue
        jobs.append((str(csv), str(dst)))
    print(f"{len(seen)} CSV files found, {len(jobs)} to convert with {a.workers} workers", flush=True)

    t0 = time.perf_counter()
    log = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(convert_one, jobs, chunksize=1)):
            log.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    elapsed = time.perf_counter() - t0

    log_path = a.dst / "conversion_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {"files": []}
    by_dst = {r["dst"]: r for r in old["files"]}
    by_dst.update({r["dst"]: r for r in log})
    log_path.write_text(json.dumps({"files": sorted(by_dst.values(), key=lambda r: r["dst"]),
                                    "last_run_seconds": elapsed}, indent=1))
    print(f"converted {len(jobs)} files in {elapsed:.0f} s; log at {log_path}")


if __name__ == "__main__":
    main()
