#!/usr/bin/env python3
"""Cache the shared reader's exact downsampled tensors, without fitting data."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import scipy

from emdirmf.paths import results_dir
from emdirmf.stacks import StackReader


class CachedStackReader(StackReader):
    """Read the same float32 arrays as StackReader, with its FIR applied once."""

    def __init__(self, dataset, method, downsample=1):
        super().__init__(dataset, method, downsample)
        files = [Path(self.frames.filename)]
        if self.source is not None:
            files.append(Path(self.source.filename))
        signature = {"method": self.method, "downsample": self.downsample,
                     "scipy": scipy.__version__, "shape": self.shape,
                     "inputs": [{"path": str(p), "bytes": p.stat().st_size,
                                 "mtime_ns": p.stat().st_mtime_ns} for p in files],
                     "code": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
                         "reader": STUDY / "emdirmf/stacks.py",
                         "normalization": STUDY / "emdirmf/decompose.py",
                         "cache": Path(__file__),
                     }.items()}}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        directory = results_dir(dataset) / "representations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"stack_{self.method}_d{self.downsample}_{key}.npy"
        started = time.perf_counter()
        created = not path.exists()
        if created:
            temporary = path.with_suffix(f".{os.getpid()}.tmp.npy")
            values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=self.shape)
            for start in range(0, self.shape[0], 32):
                indices = np.arange(start, min(start + 32, self.shape[0]))
                values[indices] = super().take(indices)
            values.flush()
            del values
            temporary.replace(path)
            path.with_suffix(".json").write_text(json.dumps(signature, indent=2) + "\n")
        self.cached_values = np.load(path, mmap_mode="r")
        assert self.cached_values.shape == self.shape and self.cached_values.dtype == np.float32
        # DataLoader originally reads one frame at a time. Check those exact
        # calls against the cache, including median/MAD normalization for raw.
        indices = np.unique(np.r_[np.arange(4), np.random.default_rng(20261021).choice(self.shape[0], 12, replace=False)])
        for index in indices:
            expected = super().take([index])
            np.testing.assert_array_equal(self.take([index]), expected)
        self.cache_description = {"path": str(path), "created": created,
                                  "setup_seconds": time.perf_counter() - started,
                                  "verified_frames": indices.tolist(), "exact_input_equality": True}

    def take(self, indices):
        return np.ascontiguousarray(self.cached_values[np.asarray(indices, dtype=np.int64)], dtype=np.float32)

    def description(self):
        return {**super().description(), "reader_cache": self.cache_description}


def prepare_one(arguments):
    dataset, method, downsample = arguments
    reader = CachedStackReader(dataset, method, downsample)
    return {"method": method, "shape": reader.shape, **reader.cache_description}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dronerf")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    jobs = [(args.dataset, method, args.downsample) for method in args.methods]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(prepare_one, jobs))
    result = {"seconds": time.perf_counter() - started, "workers": args.workers, "caches": records}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
