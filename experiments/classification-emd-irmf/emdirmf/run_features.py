"""Feature tables per method: python -m emdirmf.run_features --dataset NAME --method TAG [TAG ...]

Writes <DATA_ROOT>/NAME/features/TAG.parquet with columns
  frame_id, decomp_ok, ch{c}_log_scale, ch{c}_c{k}_{feature} ...
and TAG.json (feature list, slot labels, source decomposition, timing). `raw`
uses the normalized frame as the only component. Frames whose decomposition
failed keep NaN features and decomp_ok = False.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import decompose as dec
from .features import FEATURE_NAMES, feature_names, frame_features, log_scale_names, raw_components
from .io import load_frames
from .parallel import check_workers, limit_threads, single_thread_env
from .paths import decomp_dir, features_dir, prepared_dir
from .run_decompose import decomp_paths, load_decomposition

TARGET_BLOCK_VALUES = 1 << 22


def _block_frames(n_slots, n_channels, length):
    return max(1, TARGET_BLOCK_VALUES // (n_slots * n_channels * length))


def _features_block(dataset, tag, start, stop):
    frames = np.load(prepared_dir(dataset) / "frames.npy", mmap_mode="r")
    block = np.asarray(frames[start:stop], dtype=np.float64)
    raw, log_scale = raw_components(block)
    if tag == "raw":
        components = raw
    else:
        components = np.load(decomp_dir(dataset) / f"{tag}.npy", mmap_mode="r")[start:stop]
    return start, log_scale, frame_features(components)


def run_features(dataset, tag, *, workers=1, verbose=True):
    spec = dec.parse_method(tag)
    tag = spec.tag
    frames = load_frames(dataset)
    n_frames, n_channels, length = frames.shape
    ok = np.ones(n_frames, dtype=bool)
    source = None
    if spec.is_decomposition:
        components, meta, summary = load_decomposition(dataset, tag)
        if summary.get("partial") or components.shape[0] != n_frames:
            raise RuntimeError(f"{tag} decomposition covers {components.shape[0]} of {n_frames} frames")
        n_slots = components.shape[2]
        ok = meta.groupby("frame_id")["ok"].all().reindex(np.arange(n_frames), fill_value=False).to_numpy(dtype=bool)
        source = {"decomposition": str(decomp_paths(dataset, tag)["json"]),
                  "frames_fingerprint": summary.get("frames_fingerprint")}
    else:
        n_slots = 1
    step = _block_frames(n_slots, n_channels, length)
    ranges = [(start, min(start + step, n_frames)) for start in range(0, n_frames, step)]
    started = time.perf_counter()
    values = np.empty((n_frames, n_channels * n_slots * len(FEATURE_NAMES)))
    log_scale = np.empty((n_frames, n_channels))
    workers = check_workers(workers)
    if workers == 1:
        limit_threads()
        results = (_features_block(dataset, tag, start, stop) for start, stop in ranges)
        for start, block_log_scale, block_values in results:
            values[start : start + len(block_values)] = block_values
            log_scale[start : start + len(block_values)] = block_log_scale
    else:
        single_thread_env()
        with ProcessPoolExecutor(max_workers=workers, initializer=limit_threads) as pool:
            futures = [pool.submit(_features_block, dataset, tag, start, stop) for start, stop in ranges]
            for future in futures:
                start, block_log_scale, block_values = future.result()
                values[start : start + len(block_values)] = block_values
                log_scale[start : start + len(block_values)] = block_log_scale
    values[~ok] = np.nan
    elapsed = time.perf_counter() - started

    names = feature_names(n_channels, n_slots)
    table = pd.concat(
        [
            pd.DataFrame({"frame_id": np.arange(n_frames), "decomp_ok": ok}),
            pd.DataFrame(log_scale, columns=log_scale_names(n_channels)),
            pd.DataFrame(values.astype(np.float32), columns=names),
        ],
        axis=1,
    )
    out_dir = features_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"{tag}.parquet", index=False)
    n_nonfinite_ok = int((~np.isfinite(values[ok])).any(axis=1).sum())
    info = {
        "dataset": dataset,
        "tag": tag,
        "n_frames": int(n_frames),
        "n_channels": int(n_channels),
        "n_slots": int(n_slots),
        "slot_labels": dec.slot_labels(spec, length),
        "features_per_component": list(FEATURE_NAMES),
        "n_features": len(names),
        "log_scale_columns": log_scale_names(n_channels),
        "n_failed_frames": int((~ok).sum()),
        "n_nonfinite_rows_among_ok": n_nonfinite_ok,
        "seconds": elapsed,
        "source": source,
    }
    with open(out_dir / f"{tag}.json", "w") as handle:
        json.dump(info, handle, indent=2)
    if verbose:
        print(f"{dataset}/{tag}: {len(names)} features x {n_frames} frames in {elapsed:.1f} s; "
              f"{info['n_failed_frames']} failed frames, {n_nonfinite_ok} non-finite rows among ok frames")
    return table


def load_features(dataset, tag):
    return pd.read_parquet(features_dir(dataset) / f"{dec.canonical_tag(tag)}.parquet")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute per-component feature tables")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    for tag in args.method:
        run_features(args.dataset, tag, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
