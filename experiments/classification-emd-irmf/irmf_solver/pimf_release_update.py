"""Rerun converged IRMF with a new pimf release on every frame, without touching
the study caches.

Run with an environment that holds the new release (not the study venv, which
stays on pimf 0.4.0 while other experiments use it):

    python irmf_solver/pimf_release_update.py decompose --dataset dronerf --workers 4
    python irmf_solver/pimf_release_update.py summarize
    python irmf_solver/pimf_release_update.py versus --base 0.5.0

``decompose`` writes, under ALT_ROOT/<dataset>/ (default
/Users/mikhail/Datasets/imf-classification-pimf<version>/):

- features/irmf_h2_a1.41_conv.parquet and .json, in the study's feature-table
  format, from pimf's ``solver="newton"`` with the study normalization, H rule and
  schedule;
- symlinks to the study's prepared/ data, split files and the raw, EMD and
  linear-IMF feature tables, so the unchanged study runners can classify the
  new features when pointed at ALT_ROOT through IMF_CLS_DATA;
- pimf_release/channels.parquet: per channel, the difference from the cached
  prototype decomposition, Newton diagnostics and CPU time.

``versus`` checks a later release against an earlier one's full-data outputs:
identical feature tables mean identical classification results, because the
runners, splits and seeds are the same.

The script also checks that linear IMF from the new release reproduces the
cached linear IMF decomposition, because the symlinked linear features assume it.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pimf  # noqa: E402

from check_pimf_release import newton_release_components  # noqa: E402
from emdirmf import decompose as dec  # noqa: E402
from emdirmf.features import FEATURE_NAMES, feature_names, frame_features, log_scale_names, raw_components  # noqa: E402
from emdirmf.parallel import check_workers, limit_threads, single_thread_env  # noqa: E402

STUDY_ROOT = Path("/Users/mikhail/Datasets/imf-classification")
DATASETS = ["synthetic", "dronerf", "birdclef2023", "esc50_animals"]
TAG = "irmf_h2_a1.41_conv"
LINEAR_TAG = "imf_linear_a1.41"
LINKED_FEATURES = ["raw", "emd", LINEAR_TAG]
BLOCK = 16


def alt_root():
    return Path(os.environ.get("PIMF_RELEASE_ROOT", f"/Users/mikhail/Datasets/imf-classification-pimf{pimf.__version__}"))


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.resolve() != target.resolve():
            raise SystemExit(f"{path} exists and does not point to {target}")
        return
    path.symlink_to(target)


def setup_root(dataset):
    """Link the study inputs that the new release does not change."""
    source, dest = STUDY_ROOT / dataset, alt_root() / dataset
    link(source / "prepared", dest / "prepared")
    for split in sorted(source.glob("splits_*.json")):
        link(split, dest / split.name)
    for tag in LINKED_FEATURES:
        for suffix in (".parquet", ".json"):
            link(source / "features" / f"{tag}{suffix}", dest / "features" / f"{tag}{suffix}")


def _block(dataset, start, stop):
    limit_threads()
    frames = np.load(STUDY_ROOT / dataset / "prepared" / "frames.npy", mmap_mode="r")
    proto = np.load(STUDY_ROOT / dataset / "decomp" / f"{TAG}.npy", mmap_mode="r")
    linear = np.load(STUDY_ROOT / dataset / "decomp" / f"{LINEAR_TAG}.npy", mmap_mode="r")
    block = np.asarray(frames[start:stop], dtype=np.float64)
    _, log_scale = raw_components(block)
    stacks, rows = [], []
    linear_mismatches = 0
    for offset, frame in enumerate(block):
        frame_id = start + offset
        channels = []
        for channel, signal in enumerate(frame):
            y, _ = dec.normalize_frame(signal)
            started = time.process_time()
            stack, info = newton_release_components(y)
            seconds = time.process_time() - started
            channels.append(stack)
            reference = np.asarray(proto[frame_id, channel], dtype=np.float64)
            delta = stack - reference
            linear_new = dec.imf_linear_components(y)[0].astype(np.float32)
            linear_mismatches += int(np.count_nonzero(linear_new != linear[frame_id, channel]))
            rows.append({
                "frame_id": frame_id,
                "channel": channel,
                "rel_l2_vs_prototype": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-300)),
                "max_abs_vs_prototype": float(np.abs(delta).max()),
                "float32_identical": bool(np.array_equal(stack.astype(np.float32), proto[frame_id, channel])),
                "cpu_seconds": seconds,
                **info,
            })
        stacks.append(np.stack(channels))
    values = frame_features(np.stack(stacks))
    return start, log_scale, values, rows, linear_mismatches


def decompose(dataset, workers):
    setup_root(dataset)
    frames = np.load(STUDY_ROOT / dataset / "prepared" / "frames.npy", mmap_mode="r")
    n_frames, n_channels, length = frames.shape
    n_slots = dec.n_slots(dec.parse_method(TAG), length)
    ranges = [(start, min(start + BLOCK, n_frames)) for start in range(0, n_frames, BLOCK)]
    values = np.empty((n_frames, n_channels * n_slots * len(FEATURE_NAMES)))
    log_scale = np.empty((n_frames, n_channels))
    rows, linear_mismatches = [], 0
    started = time.perf_counter()
    single_thread_env()
    with ProcessPoolExecutor(max_workers=check_workers(workers), initializer=limit_threads) as pool:
        futures = [pool.submit(_block, dataset, start, stop) for start, stop in ranges]
        for done, future in enumerate(futures, start=1):
            start, block_log, block_values, block_rows, mismatches = future.result()
            values[start : start + len(block_values)] = block_values
            log_scale[start : start + len(block_values)] = block_log
            rows.extend(block_rows)
            linear_mismatches += mismatches
            if done % 50 == 0 or done == len(futures):
                print(f"{dataset}: {done}/{len(futures)} blocks, {time.perf_counter() - started:.0f} s", flush=True)
    elapsed = time.perf_counter() - started
    channels = pd.DataFrame(rows)
    ok = channels.groupby("frame_id")["converged"].all().reindex(np.arange(n_frames), fill_value=False).to_numpy()
    values[~ok] = np.nan

    dest = alt_root() / dataset
    (dest / "pimf_release").mkdir(parents=True, exist_ok=True)
    channels.to_parquet(dest / "pimf_release" / "channels.parquet", index=False)
    names = feature_names(n_channels, n_slots)
    table = pd.concat([
        pd.DataFrame({"frame_id": np.arange(n_frames), "decomp_ok": ok}),
        pd.DataFrame(log_scale, columns=log_scale_names(n_channels)),
        pd.DataFrame(values.astype(np.float32), columns=names),
    ], axis=1)
    (dest / "features").mkdir(parents=True, exist_ok=True)
    table.to_parquet(dest / "features" / f"{TAG}.parquet", index=False)
    info = {
        "dataset": dataset,
        "tag": TAG,
        "meaning": f"converged IRMF from pimf {pimf.__version__} IMF(SmoothAbs(H, lookup=True), solver='newton'), "
                   "study normalization, H = 2 sigma_hat, h1 = 0.25, a = sqrt 2, h_min = 1.5 / L",
        "pimf_version": pimf.__version__,
        "pimf_file": pimf.__file__,
        "n_frames": int(n_frames),
        "n_channels": int(n_channels),
        "n_slots": int(n_slots),
        "slot_labels": dec.slot_labels(dec.parse_method(TAG), length),
        "features_per_component": list(FEATURE_NAMES),
        "n_features": len(names),
        "log_scale_columns": log_scale_names(n_channels),
        "n_failed_frames": int((~ok).sum()),
        "linear_imf_float32_mismatches": int(linear_mismatches),
        "wall_seconds": elapsed,
        "workers": workers,
    }
    (dest / "features" / f"{TAG}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("dataset", "n_failed_frames", "linear_imf_float32_mismatches", "wall_seconds")}))


def summarize(output):
    rows = []
    for dataset in DATASETS:
        path = alt_root() / dataset / "pimf_release" / "channels.parquet"
        if not path.exists():
            continue
        channels = pd.read_parquet(path)
        info = json.loads((alt_root() / dataset / "features" / f"{TAG}.json").read_text())
        new = pd.read_parquet(alt_root() / dataset / "features" / f"{TAG}.parquet")
        old = pd.read_parquet(STUDY_ROOT / dataset / "features" / f"{TAG}.parquet")
        columns = [c for c in new.columns if c.startswith("ch") and not c.endswith("_log_scale")]
        feature_diff = (new[columns].to_numpy(np.float64) - old[columns].to_numpy(np.float64))
        frame_changed = np.nanmax(np.abs(feature_diff), axis=1) > 1e-4 * (1 + np.nanmax(np.abs(old[columns].to_numpy(np.float64)), axis=1))
        per_frame = channels.groupby("frame_id")
        rows.append({
            "dataset": dataset,
            "frames": int(new.shape[0]),
            "channels": int(channels.shape[0]),
            "all_converged": bool(channels["converged"].all()),
            "unconverged_windows": int(channels["unconverged_windows"].sum()),
            "channels_float32_identical": int(channels["float32_identical"].sum()),
            "channels_rel_l2_above_1e-6": int((channels["rel_l2_vs_prototype"] > 1e-6).sum()),
            "channels_rel_l2_above_1e-3": int((channels["rel_l2_vs_prototype"] > 1e-3).sum()),
            "rel_l2_max": float(channels["rel_l2_vs_prototype"].max()),
            "frames_with_changed_features": int(frame_changed.sum()),
            "linear_imf_float32_mismatches": info["linear_imf_float32_mismatches"],
            "newton_iterations_max": int(channels["iterations_max"].max()),
            "bisection_fallbacks_per_channel": float(channels["bisection_fallbacks"].mean()),
            "max_abs_score": float(channels["max_abs_score"].max()),
            "cpu_s_per_frame": float(per_frame["cpu_seconds"].sum().mean()),
            "wall_seconds": info["wall_seconds"],
        })
    table = pd.DataFrame(rows)
    out = Path(output) / pimf.__version__
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "full_data_comparison.csv", index=False)
    print(table.to_string(index=False))


def versus(base, output):
    base_root = alt_root().parent / f"imf-classification-pimf{base}"
    keys = ["frame_id", "channel"]
    rows = []
    for dataset in DATASETS:
        new_dir, old_dir = alt_root() / dataset, base_root / dataset
        if not (new_dir / "pimf_release" / "channels.parquet").exists():
            continue
        new_ch = pd.read_parquet(new_dir / "pimf_release" / "channels.parquet").sort_values(keys, ignore_index=True)
        old_ch = pd.read_parquet(old_dir / "pimf_release" / "channels.parquet").sort_values(keys, ignore_index=True)
        if not new_ch[keys].equals(old_ch[keys]):
            raise SystemExit(f"{dataset}: channel lists differ")
        # Everything except timing depends only on the decomposition.
        columns = [c for c in new_ch.columns if c not in keys + ["cpu_seconds"]]
        mismatches = {c: int((new_ch[c].ne(old_ch[c]) & ~(new_ch[c].isna() & old_ch[c].isna())).sum()) for c in columns}
        new_f = pd.read_parquet(new_dir / "features" / f"{TAG}.parquet")
        old_f = pd.read_parquet(old_dir / "features" / f"{TAG}.parquet")
        bitwise = list(new_f.columns) == list(old_f.columns) and all(
            new_f[c].dtype == old_f[c].dtype and new_f[c].to_numpy().tobytes() == old_f[c].to_numpy().tobytes()
            for c in new_f.columns)
        new_info = json.loads((new_dir / "features" / f"{TAG}.json").read_text())
        old_info = json.loads((old_dir / "features" / f"{TAG}.json").read_text())
        new_cpu = new_ch.groupby("frame_id")["cpu_seconds"].sum()
        old_cpu = old_ch.groupby("frame_id")["cpu_seconds"].sum()
        rows.append({
            "dataset": dataset,
            "frames": int(new_f.shape[0]),
            "channels": int(new_ch.shape[0]),
            "feature_table_bitwise_identical": bool(bitwise),
            "channel_diagnostics_identical": not any(mismatches.values()),
            "diagnostic_mismatches": json.dumps({c: n for c, n in mismatches.items() if n}),
            "all_converged": bool(new_ch["converged"].all()),
            "linear_imf_float32_mismatches": new_info["linear_imf_float32_mismatches"],
            f"cpu_s_per_frame_{base}": float(old_cpu.mean()),
            f"cpu_s_per_frame_{pimf.__version__}": float(new_cpu.mean()),
            f"wall_seconds_{base}": old_info["wall_seconds"],
            f"wall_seconds_{pimf.__version__}": new_info["wall_seconds"],
        })
    table = pd.DataFrame(rows)
    out = Path(output) / pimf.__version__
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"full_data_vs_{base}.csv", index=False)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("decompose")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--workers", type=int, default=4)
    report = sub.add_parser("summarize")
    report.add_argument("--output", default=str(HERE.parents[2] / "output/classification-emd-irmf/pimf_release"))
    against = sub.add_parser("versus")
    against.add_argument("--base", required=True, help="earlier release whose ALT_ROOT outputs to compare with")
    against.add_argument("--output", default=str(HERE.parents[2] / "output/classification-emd-irmf/pimf_release"))
    args = parser.parse_args()
    if pimf.__version__ == "0.4.0":
        raise SystemExit("run this with the new pimf release, not the study venv")
    if args.command == "decompose":
        decompose(args.dataset, args.workers)
    elif args.command == "versus":
        versus(args.base, args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
