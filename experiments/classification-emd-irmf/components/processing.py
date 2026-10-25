"""Resumable float32 native decompositions and shared component features."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from components.construction import (
    BASE_TAGS, DEFAULTS, all_tags, fold_emd, native_components, parse_tag,
    relative_error, truncate_imf,
)
from components.integrity import DATA_ROOT, DATASETS, check_pimf, record_entry
from components.pilot import initialize
from components.storage import check_budget, code_hash, run_dir, write_json, write_npz
from emdirmf.decompose import normalize_frame
from emdirmf.features import feature_names, frame_features, log_scale_names
from emdirmf.parallel import limit_threads, single_thread_env

BLOCK_SIZE = 32


def source_tag(spec):
    if spec.construction == "fold":
        return "emd"
    if spec.family == "gd60":
        return "irmf_h2_a1.41"
    return DEFAULTS[spec.family]


def block_job(dataset, tag, start, stop):
    check_pimf()
    directory = run_dir(dataset) / "blocks" / tag
    checkpoint = directory / f"{start:06d}.npz"
    metadata_path = directory / f"{start:06d}.json"
    if checkpoint.exists() and metadata_path.exists():
        return tag, start, "reused"
    if checkpoint.exists() or metadata_path.exists():
        raise RuntimeError(f"incomplete immutable checkpoint: {checkpoint}")
    spec = parse_tag(tag)
    frames = np.load(DATA_ROOT / dataset / "prepared/frames.npy", mmap_mode="r", allow_pickle=False)
    normalized, log_scale = normalize_frame(frames[start:stop])
    cpu, wall = time.process_time(), time.perf_counter()
    diagnostics = []
    if spec.construction == "native":
        stacks = np.empty((stop - start, frames.shape[1], spec.k + 1, frames.shape[2]), dtype=np.float32)
        for i, frame in enumerate(normalized):
            for channel, y in enumerate(frame):
                stack, info = native_components(y, spec.family, spec.k)
                stacks[i, channel] = stack
                diagnostics.append({"frame_id": start + i, "channel": channel, **info})
    else:
        source = source_tag(spec)
        cached = np.load(DATA_ROOT / dataset / "decomp" / f"{source}.npy", mmap_mode="r", allow_pickle=False)
        source_block = cached[start:stop]
        stacks = (fold_emd(source_block, spec.k) if spec.construction == "fold" else
                  truncate_imf(source_block, spec.k, source)).astype(np.float32)
    decomposition_cpu = time.process_time() - cpu
    reconstruction = relative_error(stacks.sum(axis=-2, dtype=np.float64), normalized)
    assert reconstruction < 2e-7, (dataset, tag, start, reconstruction)
    feature_cpu = time.process_time()
    values = frame_features(stacks).astype(np.float32)
    assert np.all(np.isfinite(values))
    energy = np.mean(stacks.astype(np.float64) ** 2, axis=-1)
    relative_energy = np.divide(energy, energy.sum(axis=-1, keepdims=True),
                                out=np.zeros_like(energy), where=energy.sum(axis=-1, keepdims=True) > 0)
    arrays = {"features": values, "log_scale": log_scale, "relative_energy": relative_energy,
              "frame_id": np.arange(start, stop)}
    if spec.construction == "native":
        arrays["components"] = stacks
    # Maximum allocation was reserved by the parent before launching this batch.
    check_pimf()
    write_npz(checkpoint, **arrays)
    write_json(metadata_path, {"dataset": dataset, "tag": tag, "start": start, "stop": stop,
                              "decomposition_cpu_seconds": decomposition_cpu,
                              "feature_cpu_seconds": time.process_time() - feature_cpu,
                              "wall_seconds": time.perf_counter() - wall,
                              "reconstruction_relative_error": reconstruction,
                              "diagnostics": diagnostics})
    return tag, start, "computed"


def assemble_features(dataset, tag, n_frames, n_channels):
    target = run_dir(dataset) / "features" / f"{tag}.parquet"
    if target.exists():
        return
    values, scales, indices = [], [], []
    for start in range(0, n_frames, BLOCK_SIZE):
        with np.load(run_dir(dataset) / "blocks" / tag / f"{start:06d}.npz", allow_pickle=False) as block:
            values.append(block["features"])
            scales.append(block["log_scale"])
            indices.extend(block["frame_id"])
    assert np.array_equal(indices, np.arange(n_frames))
    values, scales = np.concatenate(values), np.concatenate(scales)
    slots = values.shape[1] // (n_channels * 20)
    table = pd.concat([pd.DataFrame({"frame_id": np.arange(n_frames), "decomp_ok": True}),
                       pd.DataFrame(scales, columns=log_scale_names(n_channels)),
                       pd.DataFrame(values, columns=feature_names(n_channels, slots))], axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False)


def run(dataset, workers):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be 1..6")
    record_entry(dataset, "processing")
    single_thread_env()
    limit_threads()
    frames = np.load(DATA_ROOT / dataset / "prepared/frames.npy", mmap_mode="r", allow_pickle=False)
    n, channels, length = frames.shape
    tags = [tag for tag in all_tags() if tag not in BASE_TAGS]
    manifest_path = run_dir(dataset) / "processing_manifest.json"
    manifest = {"shape": list(frames.shape), "tags": tags, "block_size": BLOCK_SIZE,
                "code": code_hash([__file__, __import__('components.construction', fromlist=['']).__file__,
                                    __import__('emdirmf.features', fromlist=['']).__file__])}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("processing manifest changed; do not reuse stale checkpoints")
    else:
        write_json(manifest_path, manifest)
    additional = sum(n * channels * (parse_tag(tag).k + 1) * length * 4
                     for tag in tags if parse_tag(tag).construction == "native")
    # Reserve the full worst case, including 2 GB for feature/checkpoint overhead.
    existing_native = sum(path.stat().st_size for path in (run_dir(dataset) / "blocks").rglob("*.npz"))
    check_budget(max(0, additional - existing_native) + 2_000_000_000)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for tag in tags:
            jobs = [pool.submit(block_job, dataset, tag, start, min(start + BLOCK_SIZE, n))
                    for start in range(0, n, BLOCK_SIZE)]
            for future in as_completed(jobs):
                future.result()
            assemble_features(dataset, tag, n, channels)
            check_budget()
            check_pimf()
            print(f"{dataset}/{tag}: processing complete, elapsed {time.perf_counter()-started:.1f}s", flush=True)
    write_json(run_dir(dataset) / f"processing_completed_{time.time_ns()}.json",
               {"dataset": dataset, "wall_seconds": time.perf_counter() - started, "workers": workers})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset, args.workers)


if __name__ == "__main__":
    main()
