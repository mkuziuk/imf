"""Per-segment envelope and spectrum statistics for DroneRF (label-free).

For every cached segment file `<BUI><band>_<idx>.npy` (10^7 samples) this
computes

- the mean energy (mean of x^2 after removing the segment mean) in
  non-overlapping windows of `--win` samples, stored as float32 log10 values;
- a Welch power spectral density (Hann, `--nfft` points, one-sided, since the
  samples are real-valued).

Results go to `<DATA_ROOT>/dronerf/work/segment_stats.npz`. `diagnostics.py`
uses them for the L/H lag table and the spectra figure; `prepare.py` does not
need them. Run with `python -I`.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import welch

DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
FS = 40e6
NAME_RE = re.compile(r"^(?P<bui>[01]{5})(?P<band>[LH])_(?P<idx>\d+)\.npy$")


def one(args):
    path, win, nfft = args
    x = np.load(path, mmap_mode="r").astype(np.float64)
    x = x - x.mean()
    n = (x.size // win) * win
    e = (x[:n].reshape(-1, win) ** 2).mean(axis=1)
    _, pxx = welch(x, fs=FS, window="hann", nperseg=nfft, noverlap=nfft // 2, detrend=False)
    return np.log10(e + 1e-12).astype(np.float32), pxx.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", type=Path, default=DATA_ROOT / "dronerf/raw/npy")
    p.add_argument("--out", type=Path, default=DATA_ROOT / "dronerf/work/segment_stats.npz")
    p.add_argument("--win", type=int, default=256)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    files = sorted(f for f in a.npy.glob("*.npy") if NAME_RE.match(f.name))
    keys = [NAME_RE.match(f.name).groupdict() for f in files]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(str(f), a.win, a.nfft) for f in files], chunksize=2))
    print(f"{len(files)} files in {time.perf_counter() - t0:.0f} s")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.out,
        bui=np.array([k["bui"] for k in keys]),
        band=np.array([k["band"] for k in keys]),
        idx=np.array([int(k["idx"]) for k in keys]),
        log_energy=np.stack([r[0] for r in res]),
        psd=np.stack([r[1] for r in res]),
        freqs=np.fft.rfftfreq(a.nfft, 1 / FS),
        win=a.win,
        nfft=a.nfft,
    )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
