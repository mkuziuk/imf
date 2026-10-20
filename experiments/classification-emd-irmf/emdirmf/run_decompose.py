"""Batch decomposition of a prepared dataset into a resumable float32 memmap cache.

python -m emdirmf.run_decompose --dataset NAME --method TAG [--workers N]

Writes <DATA_ROOT>/NAME/decomp/:
  TAG.npy           float32 (N, C, K + 1, L), canonical slot order (see decompose.py)
  TAG.meta.parquet  one row per (frame_id, channel): ok/error, counts, solver diagnostics,
                    log_scale, seconds (wall) and cpu_seconds
  TAG.json          parameters, slot labels, window sizes, versions, timing summary
  TAG.parts/        manifest.json plus one parquet per finished chunk (resume state)

Chunks are written to the memmap by the parent process, flushed, and only then
marked done, so an interrupted run resumes at the first unfinished chunk. A frame
whose decomposition raises is stored as NaN with ok=False and the error text.
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import decompose as dec
from .io import frames_fingerprint, load_frames
from .parallel import check_workers, limit_threads, single_thread_env
from .paths import decomp_dir, prepared_dir

DEFAULT_CHUNK_SIZE = 64
_FRAMES_CACHE = {}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def versions():
    import PyEMD
    import pimf
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pimf": pimf.__version__,
        "PyEMD": getattr(PyEMD, "__version__", "unknown"),
        "platform": platform.platform(),
    }


def decomp_paths(dataset, tag):
    directory = decomp_dir(dataset)
    return {
        "dir": directory,
        "npy": directory / f"{tag}.npy",
        "meta": directory / f"{tag}.meta.parquet",
        "json": directory / f"{tag}.json",
        "parts": directory / f"{tag}.parts",
    }


def _worker_frames(frames_path):
    frames = _FRAMES_CACHE.get(frames_path)
    if frames is None:
        frames = np.load(frames_path, mmap_mode="r")
        _FRAMES_CACHE[frames_path] = frames
    return frames


def decompose_chunk(frames_path, tag, start, stop):
    """Decompose frames[start:stop]; returns (float32 array, list of meta rows)."""
    frames = _worker_frames(frames_path)
    spec = dec.parse_method(tag)
    block = np.asarray(frames[start:stop], dtype=np.float64)
    count, n_channels, length = block.shape
    output = np.empty((count, n_channels, dec.n_slots(spec, length), length), dtype=np.float32)
    rows = []
    for offset in range(count):
        components, infos = dec.decompose_frame(block[offset], spec, catch=True)
        output[offset] = components
        for info in infos:
            row = {"frame_id": start + offset}
            row.update(info)
            if "stage_converged" in row:
                row["n_unconverged"] = int(len(row["stage_converged"]) - sum(row["stage_converged"]))
                row["total_iterations"] = int(sum(row["stage_iterations"]))
            rows.append(row)
    return output, rows


def _write_part(parts_dir, index, rows):
    path = parts_dir / f"chunk_{index:06d}.parquet"
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _done_chunks(parts_dir):
    return {int(path.stem.split("_")[1]) for path in parts_dir.glob("chunk_*.parquet")}


def _remove_outputs(paths):
    for key in ("npy", "meta", "json"):
        if paths[key].exists():
            paths[key].unlink()
    if paths["parts"].exists():
        shutil.rmtree(paths["parts"])


def run_decompose(
    dataset, tag, *, workers=1, chunk_size=DEFAULT_CHUNK_SIZE, overwrite=False, limit=None, verbose=True
):
    """Decompose every frame of a prepared dataset with one method; resumable.

    limit restricts the run to the first `limit` frames (for timing checks); the
    output then has N = limit rows and is tagged as partial in TAG.json.
    """
    spec = dec.parse_method(tag)
    if not spec.is_decomposition:
        raise ValueError("raw needs no decomposition; run_features handles it directly")
    tag = spec.tag
    workers = check_workers(workers)
    frames_path = str(prepared_dir(dataset) / "frames.npy")
    frames = load_frames(dataset)
    n_total, n_channels, length = frames.shape
    n_frames = n_total if limit is None else min(int(limit), n_total)
    shape = (n_frames, n_channels, dec.n_slots(spec, length), length)
    paths = decomp_paths(dataset, tag)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if overwrite:
        _remove_outputs(paths)

    manifest = {
        "dataset": dataset,
        "tag": tag,
        "shape": list(shape),
        "chunk_size": int(chunk_size),
        "frames_fingerprint": frames_fingerprint(frames),
        "params": spec.params(),
        "limit": None if limit is None else n_frames,
    }
    manifest_path = paths["parts"] / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as handle:
            previous = json.load(handle)
        if previous.get("limit") is not None and previous.get("shape") != manifest["shape"]:
            if verbose:
                print(f"discarding a previous --limit {previous['limit']} cache")
            _remove_outputs(paths)
    if manifest_path.exists():
        keys = ("shape", "chunk_size", "frames_fingerprint", "params")
        if any(previous.get(key) != manifest[key] for key in keys):
            raise RuntimeError(
                f"existing cache {paths['parts']} was made with different inputs or settings; "
                "rerun with --overwrite"
            )
        manifest["sessions"] = previous.get("sessions", [])
        output = np.lib.format.open_memmap(paths["npy"], mode="r+")
        if output.shape != shape or output.dtype != np.float32:
            raise RuntimeError(f"{paths['npy']} has shape {output.shape}; rerun with --overwrite")
    else:
        if paths["parts"].exists() or paths["npy"].exists():
            _remove_outputs(paths)
        paths["parts"].mkdir(parents=True)
        output = np.lib.format.open_memmap(paths["npy"], mode="w+", dtype=np.float32, shape=shape)
        manifest["sessions"] = []
        manifest["created"] = _now()
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

    chunks = [(index, start, min(start + chunk_size, n_frames))
              for index, start in enumerate(range(0, n_frames, chunk_size))]
    done = _done_chunks(paths["parts"])
    todo = [chunk for chunk in chunks if chunk[0] not in done]
    if verbose:
        print(f"{dataset}/{tag}: shape {shape}, {len(chunks)} chunks, {len(todo)} to do, workers={workers}")

    session_start, wall_start = _now(), time.perf_counter()
    finished = 0

    def store(index, start, stop, block, rows):
        nonlocal finished
        output[start:stop] = block
        output.flush()
        _write_part(paths["parts"], index, rows)
        finished += 1
        if verbose and (finished == len(todo) or finished % max(1, len(todo) // 20) == 0):
            elapsed = time.perf_counter() - wall_start
            rate = elapsed / finished
            print(f"  {finished}/{len(todo)} chunks, {elapsed:.0f} s elapsed, "
                  f"~{rate * (len(todo) - finished):.0f} s left", flush=True)

    if todo and workers == 1:
        limit_threads()
        if spec.converged:
            from .irmf_newton import warm_up

            warm_up()
        for index, start, stop in todo:
            store(index, start, stop, *decompose_chunk(frames_path, tag, start, stop))
    elif todo:
        single_thread_env()
        initializer = limit_threads
        if spec.converged:
            from .irmf_newton import initialize_worker

            initializer = initialize_worker
        with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
            futures = {
                pool.submit(decompose_chunk, frames_path, tag, start, stop): (index, start, stop)
                for index, start, stop in todo
            }
            for future in as_completed(futures):
                index, start, stop = futures[future]
                store(index, start, stop, *future.result())

    manifest["sessions"].append(
        {"start": session_start, "end": _now(), "wall_seconds": time.perf_counter() - wall_start,
         "workers": workers, "chunks": len(todo)}
    )
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    del output
    return finalize(dataset, tag, manifest, verbose=verbose)


def finalize(dataset, tag, manifest, verbose=True):
    """Merge chunk metadata into TAG.meta.parquet and write the TAG.json summary."""
    paths = decomp_paths(dataset, tag)
    parts = sorted(paths["parts"].glob("chunk_*.parquet"))
    meta = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    meta = meta.sort_values(["frame_id", "channel"]).reset_index(drop=True)
    n_frames, n_channels, n_slot, length = manifest["shape"]
    if len(meta) != n_frames * n_channels:
        raise RuntimeError(f"meta has {len(meta)} rows, expected {n_frames * n_channels}")
    meta.to_parquet(paths["meta"], index=False)

    spec = dec.parse_method(tag)
    per_frame = meta.groupby("frame_id")[["seconds", "cpu_seconds"]].sum()
    summary = {
        "dataset": dataset,
        "tag": tag,
        "family": spec.family,
        "params": spec.params(),
        "shape": manifest["shape"],
        "partial": n_frames < np.load(prepared_dir(dataset) / "frames.npy", mmap_mode="r").shape[0],
        "slot_labels": dec.slot_labels(spec, length),
        "slot_window_sizes": dec.slot_window_sizes(spec, length),
        "scale_order": dec.scale_order(spec, length),
        "frames_fingerprint": manifest["frames_fingerprint"],
        "n_failed": int((~meta["ok"]).sum()),
        "failed_examples": meta.loc[~meta["ok"], ["frame_id", "channel", "error"]].head(10).to_dict("records"),
        "seconds_per_frame": {
            "wall_mean": float(per_frame["seconds"].mean()),
            "wall_median": float(per_frame["seconds"].median()),
            "cpu_mean": float(per_frame["cpu_seconds"].mean()),
            "cpu_median": float(per_frame["cpu_seconds"].median()),
            "cpu_total": float(per_frame["cpu_seconds"].sum()),
        },
        "sessions": manifest["sessions"],
        "wall_seconds_total": float(sum(session["wall_seconds"] for session in manifest["sessions"])),
        "scale_source_counts": _counts(meta["scale_source"]) if "scale_source" in meta else {},
        "versions": versions(),
        "finished": _now(),
    }
    if spec.family in ("imf_linear", "irmf"):
        windows, bandwidths = dec.imf_schedule(length, spec.a)
        summary["schedule"] = {
            "h1": dec.H1,
            "a": spec.a,
            "h_min": dec.imf_h_min(length),
            "window_sizes_pimf_order": list(windows),
            "bandwidths_pimf_order": list(bandwidths),
            "n_stages": len(windows),
        }
    if spec.family == "emd" and "n_imfs" in meta:
        ok = meta[meta["ok"]]
        summary["emd_n_imfs_counts"] = _counts(ok["n_imfs"])
        summary["emd_frames_padded"] = int((ok["n_imfs"] < dec.K_EMD).sum())
        summary["emd_frames_folded"] = int((ok["n_imfs"] > dec.K_EMD).sum())
    if spec.family == "irmf" and "stage_converged" in meta:
        ok = meta[meta["ok"]]
        converged = np.stack(ok["stage_converged"].to_numpy()) if len(ok) else np.zeros((0, 0))
        iterations = np.stack(ok["stage_iterations"].to_numpy()) if len(ok) else np.zeros((0, 0))
        summary["irmf"] = {
            "channels_all_converged": int(converged.all(axis=1).sum()) if len(ok) else 0,
            "unconverged_fraction_per_slot": (1 - converged.mean(axis=0)).round(4).tolist(),
            "mean_iterations_per_slot": iterations.mean(axis=0).round(2).tolist(),
            "sigma_source_counts": _counts(ok["sigma_source"]),
            "H_quantiles": ok["H"].quantile([0, 0.05, 0.5, 0.95, 1]).round(5).to_dict(),
        }
        if spec.converged and len(ok):
            fallbacks = np.stack(ok["stage_bisection_fallbacks"].to_numpy())
            scores = np.stack(ok["stage_max_score_residual"].to_numpy())
            summary["irmf"].update(
                max_iterations_per_slot=iterations.max(axis=0).tolist(),
                bisection_fallbacks_per_slot=fallbacks.sum(axis=0).tolist(),
                max_score_residual_per_slot=scores.max(axis=0).tolist(),
            )
    with open(paths["json"], "w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    if verbose:
        spf = summary["seconds_per_frame"]
        print(f"done {dataset}/{tag}: {summary['n_failed']} failed, cpu {spf['cpu_mean']:.4f} s/frame "
              f"(wall {spf['wall_mean']:.4f}), total wall {summary['wall_seconds_total']:.0f} s")
    return summary


def _counts(series):
    series = series.dropna().astype(int)
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_decomposition(dataset, tag, mmap_mode="r"):
    """Return (components memmap (N, C, K + 1, L), meta DataFrame, summary dict)."""
    tag = dec.canonical_tag(tag)
    paths = decomp_paths(dataset, tag)
    if not paths["json"].exists():
        raise FileNotFoundError(f"{paths['json']} missing; run emdirmf.run_decompose first")
    with open(paths["json"]) as handle:
        summary = json.load(handle)
    return np.load(paths["npy"], mmap_mode=mmap_mode), pd.read_parquet(paths["meta"]), summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, nargs="+", help="one or more method tags")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    parser.add_argument("--overwrite", action="store_true", help="discard an existing cache")
    args = parser.parse_args(argv)
    for tag in args.method:
        run_decompose(args.dataset, tag, workers=args.workers, chunk_size=args.chunk_size,
                      overwrite=args.overwrite, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
