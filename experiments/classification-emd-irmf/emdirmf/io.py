"""Loading and validating prepared datasets (PROTOCOL.md, "Prepared dataset format")."""

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from .paths import prepared_dir

REQUIRED_INFO_KEYS = ("name", "fs", "frame_length", "channels", "tasks", "primary_task", "n_frames")
DOCUMENTARY_INFO_KEYS = ("n_groups", "license", "citation", "notes")
SOURCE_KEYS = ("source_url", "source", "url", "source_urls")
CLASS_COUNT_KEYS = ("class_counts", "classes", "counts")
SEED_KEYS = ("seed", "rng_seed", "random_seed")
PREP_KEYS = ("preparation", "preparation_params", "params", "parameters", "prep")
MIN_GROUPS_PER_CLASS = 5


def load_info(name):
    with open(prepared_dir(name) / "dataset.json") as handle:
        return json.load(handle)


def load_meta(name):
    return pd.read_parquet(prepared_dir(name) / "meta.parquet")


def load_frames(name, mmap_mode="r"):
    """frames.npy as a read-only memmap of shape (N, C, L)."""
    return np.load(prepared_dir(name) / "frames.npy", mmap_mode=mmap_mode)


def load_prepared(name, mmap_mode="r"):
    """Return (frames memmap (N, C, L), meta DataFrame, dataset.json dict)."""
    return load_frames(name, mmap_mode), load_meta(name), load_info(name)


def label_column(info, task=None):
    """Meta column holding the labels of a task (primary task by default)."""
    task = info["primary_task"] if task is None else task
    tasks = info["tasks"]
    if task not in tasks:
        raise KeyError(f"task {task!r} not in dataset.json tasks {sorted(tasks)}")
    return tasks[task]


def labels(meta, info, task=None):
    return meta[label_column(info, task)].astype(str).to_numpy()


def meta_fingerprint(meta, columns):
    """Content hash of selected meta columns, used to detect stale splits."""
    digest = hashlib.blake2b(digest_size=16)
    for column in columns:
        digest.update(column.encode())
        digest.update(pd.util.hash_pandas_object(meta[column], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def frames_fingerprint(frames, n_samples=256):
    """Cheap content hash: shape, dtype and up to n_samples evenly spaced frames."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str((frames.shape, frames.dtype.str)).encode())
    count = frames.shape[0]
    for index in np.unique(np.linspace(0, count - 1, min(count, n_samples)).astype(int)):
        digest.update(np.ascontiguousarray(frames[index]).tobytes())
    return digest.hexdigest()


def _find_key(mapping, keys):
    for key in keys:
        if key in mapping:
            return key
    return None


def _find_seed(info):
    key = _find_key(info, SEED_KEYS)
    if key is not None:
        return info[key]
    for prep_key in PREP_KEYS:
        section = info.get(prep_key)
        if isinstance(section, dict):
            key = _find_key(section, SEED_KEYS)
            if key is not None:
                return section[key]
    return None


def validate_prepared(name, *, verbose=True, check_values=True, chunk_frames=512):
    """Check a prepared dataset against PROTOCOL.md and print a summary.

    Returns a dict with "errors", "warnings" and "summary". Errors make the dataset
    unusable for the shared pipeline; warnings flag missing documentation or
    choices that weaken grouped cross-validation.
    """
    errors, warnings, summary = [], [], {}
    directory = prepared_dir(name)
    for filename in ("frames.npy", "meta.parquet", "dataset.json"):
        if not (directory / filename).exists():
            errors.append(f"missing {directory / filename}")
    if errors:
        return _finish(name, errors, warnings, summary, verbose)

    frames, meta, info = load_prepared(name)

    missing = [key for key in REQUIRED_INFO_KEYS if key not in info]
    if missing:
        errors.append(f"dataset.json lacks required keys {missing}")
    for key in DOCUMENTARY_INFO_KEYS:
        if key not in info:
            warnings.append(f"dataset.json lacks {key!r}")
    if _find_key(info, SOURCE_KEYS) is None:
        warnings.append(f"dataset.json lacks a source URL (one of {SOURCE_KEYS})")
    if _find_seed(info) is None:
        warnings.append("dataset.json does not record the preparation RNG seed")
    if info.get("name", name) != name:
        warnings.append(f"dataset.json name {info.get('name')!r} differs from folder {name!r}")

    if frames.dtype != np.float32:
        errors.append(f"frames dtype {frames.dtype}, expected float32")
    if frames.ndim != 3:
        errors.append(f"frames shape {frames.shape}, expected (N, C, L)")
        return _finish(name, errors, warnings, summary, verbose)
    n_frames, n_channels, length = frames.shape
    summary.update(n_frames=n_frames, n_channels=n_channels, frame_length=length)
    if info.get("n_frames") not in (None, n_frames):
        errors.append(f"dataset.json n_frames={info.get('n_frames')} but frames has {n_frames}")
    if info.get("frame_length") not in (None, length):
        errors.append(f"dataset.json frame_length={info.get('frame_length')} but frames has {length}")
    channels = info.get("channels")
    if channels is not None and len(channels) != n_channels:
        errors.append(f"dataset.json lists {len(channels)} channels, frames have {n_channels}")
    if length > 4096:
        warnings.append(f"L={length} exceeds the 4096-sample budget")
    if n_frames > 20000:
        warnings.append(f"N={n_frames} exceeds the ~20,000-frame budget")

    if len(meta) != n_frames:
        errors.append(f"meta has {len(meta)} rows, frames has {n_frames}")
    if "frame_id" not in meta:
        errors.append("meta lacks frame_id")
    elif not np.array_equal(meta["frame_id"].to_numpy(), np.arange(len(meta))):
        errors.append("meta frame_id is not 0..N-1 in row order")
    if "group" not in meta:
        errors.append("meta lacks group")
    elif meta["group"].isna().any():
        errors.append("meta group has missing values")
    for column in ("source_file", "start_sample"):
        if column not in meta:
            warnings.append(f"meta lacks provenance column {column!r}")

    tasks = info.get("tasks", {})
    if info.get("primary_task") not in tasks:
        errors.append(f"primary_task {info.get('primary_task')!r} not in tasks")
    counts_key = _find_key(info, CLASS_COUNT_KEYS)
    if counts_key is None:
        warnings.append("dataset.json lacks per-task class counts")
    if "group" in meta:
        n_groups = int(meta["group"].nunique())
        summary["n_groups"] = n_groups
        if info.get("n_groups") not in (None, n_groups):
            errors.append(f"dataset.json n_groups={info.get('n_groups')} but meta has {n_groups}")
        summary["frames_per_group"] = meta.groupby("group").size().describe().round(1).to_dict()
    summary["tasks"] = {}
    for task, column in tasks.items():
        if not column.startswith("y_"):
            warnings.append(f"task {task!r} label column {column!r} does not follow y_<task>")
        if column not in meta:
            errors.append(f"meta lacks label column {column!r} for task {task!r}")
            continue
        values = meta[column]
        if values.isna().any():
            errors.append(f"label column {column!r} has missing values")
        if not (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)):
            errors.append(f"label column {column!r} has dtype {values.dtype}, expected strings")
        counts = values.astype(str).value_counts().sort_index()
        task_summary = {"column": column, "class_counts": counts.to_dict()}
        if counts_key is not None and isinstance(info[counts_key], dict):
            declared = info[counts_key].get(task)
            if declared is not None and {str(k): int(v) for k, v in declared.items()} != {
                str(k): int(v) for k, v in counts.items()
            }:
                errors.append(f"declared class counts for {task!r} differ from meta")
        if "group" in meta:
            groups_per_class = meta.groupby(values.astype(str))["group"].nunique()
            task_summary["groups_per_class"] = groups_per_class.to_dict()
            thin = groups_per_class[groups_per_class < MIN_GROUPS_PER_CLASS]
            if len(thin):
                warnings.append(
                    f"task {task!r}: classes with < {MIN_GROUPS_PER_CLASS} groups "
                    f"(grouped 5-fold CV cannot place them in every fold): {thin.to_dict()}"
                )
            mixed = int((meta.groupby("group")[column].nunique() > 1).sum())
            task_summary["groups_with_several_labels"] = mixed
        summary["tasks"][task] = task_summary

    if check_values and frames.dtype == np.float32 and frames.ndim == 3:
        n_nonfinite, n_constant, n_zero_mad = 0, 0, 0
        for start in range(0, n_frames, chunk_frames):
            block = np.asarray(frames[start : start + chunk_frames], dtype=np.float64)
            n_nonfinite += int((~np.isfinite(block)).any(axis=(1, 2)).sum())
            finite_block = np.where(np.isfinite(block), block, 0.0)
            n_constant += int((np.ptp(finite_block, axis=-1) == 0).any(axis=1).sum())
            median = np.median(finite_block, axis=-1, keepdims=True)
            mad = np.median(np.abs(finite_block - median), axis=-1)
            n_zero_mad += int((mad == 0).any(axis=1).sum())
        if n_nonfinite:
            errors.append(f"{n_nonfinite} frames contain non-finite values")
        if n_constant:
            warnings.append(f"{n_constant} frames have a constant channel")
        if n_zero_mad:
            warnings.append(f"{n_zero_mad} frames have a channel with zero MAD (std fallback used)")
        summary.update(
            frames_nonfinite=n_nonfinite, frames_constant=n_constant, frames_zero_mad=n_zero_mad
        )
    summary["fs"] = info.get("fs")
    if info.get("fs"):
        summary["frame_seconds"] = round(length / float(info["fs"]), 6)
    summary["frames_bytes"] = int(frames.nbytes)
    return _finish(name, errors, warnings, summary, verbose)


def _finish(name, errors, warnings, summary, verbose):
    report = {"dataset": name, "ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}
    if verbose:
        print(format_report(report))
    return report


def format_report(report):
    lines = [f"dataset {report['dataset']}: {'OK' if report['ok'] else 'INVALID'}"]
    summary = report["summary"]
    if "n_frames" in summary:
        lines.append(
            f"  frames N={summary['n_frames']} C={summary['n_channels']} L={summary['frame_length']}"
            f" fs={summary.get('fs')} ({summary.get('frame_seconds')} s/frame),"
            f" {summary.get('frames_bytes', 0) / 2**20:.1f} MiB"
        )
    if "n_groups" in summary:
        lines.append(f"  groups {summary['n_groups']}; frames/group {summary['frames_per_group']}")
    for task, task_summary in summary.get("tasks", {}).items():
        lines.append(f"  task {task} ({task_summary['column']}):")
        groups = task_summary.get("groups_per_class", {})
        for label, count in task_summary["class_counts"].items():
            lines.append(f"    {label:<30} frames={count:<7} groups={groups.get(label, '?')}")
        if task_summary.get("groups_with_several_labels"):
            lines.append(f"    groups with several labels: {task_summary['groups_with_several_labels']}")
    for key in ("frames_nonfinite", "frames_constant", "frames_zero_mad"):
        if key in summary:
            lines.append(f"  {key}: {summary[key]}")
    lines.extend(f"  ERROR: {message}" for message in report["errors"])
    lines.extend(f"  warning: {message}" for message in report["warnings"])
    return "\n".join(lines)


def class_counts(values):
    return dict(sorted(Counter(map(str, values)).items()))
