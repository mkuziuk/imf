"""Prepare moving-target, slow-time I/Q from converted numeric radar cubes.

Default: audited GitHub crops, (1000, 20, 30) complex64, one file per
recording. No raw pickle is opened here. Run with Python -I.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("/Users/mikhail/Datasets/imf-classification")
SEED = 0
N_CELLS = 16
FRAME_LENGTH = 1000
PATTERN = re.compile(
    r"^(?:cut_)?(?P<target>bird|mavik|p3p)_(?P<pose>.*?)_"
    r"(?P<state>still|rotation|5pct|50pct|100pct)"
    r"(?P<variant>_no_absorber|_old_absorber_configuration)?$"
)


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_recording(stem):
    match = PATTERN.fullmatch(stem)
    if match is None:
        return None
    row = match.groupdict()
    row["target"] = "mavic" if row["target"] == "mavik" else row["target"]
    row["variant"] = (row["variant"] or "_standard").lstrip("_")
    return row


def load_public_cube(path):
    cube = np.load(path, mmap_mode="r", allow_pickle=False)
    if cube.shape != (1000, 20, 30) or cube.dtype != np.complex64:
        raise ValueError(f"unexpected public cube {path}: {cube.shape} {cube.dtype}")
    return cube, 60, 60


# FULL DATAPORT RELEASE: UNTESTED, deliberately separate from the public path.
# Expected inputs: Data.zip manually extracted outside the repo, then scanned
# and converted with convert_pickles.py into converted_dataport/. One .npy per
# recording, original names retained, complex cubes (T>=1000, 128, 168).
# No zip extraction, downloaded Python execution, or pickle reading occurs here.
def load_dataport_cube_untested(path):
    cube = np.load(path, mmap_mode="r", allow_pickle=False)
    if cube.ndim != 3 or cube.shape[0] < FRAME_LENGTH or cube.shape[1:] != (128, 168):
        raise ValueError(f"unverified DataPort layout {path}: {cube.shape}")
    if cube.dtype.hasobject or not np.issubdtype(cube.dtype, np.complexfloating):
        raise ValueError(f"expected complex numeric DataPort cube: {path}")
    # Match the authors' crop first; verify against the public export before
    # interpreting a full-release result. Longer records use nonoverlapping windows.
    return cube[:, 60:80, 60:90], 60, 60


def prepare(base, source="public", n_cells=N_CELLS):
    converted = base / ("converted" if source == "public" else "converted_dataport")
    manifest_path = converted / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_rows = {row["output_file"]: row for row in manifest["files"]}
    loader = load_public_cube if source == "public" else load_dataport_cube_untested
    if not 1 <= n_cells <= 600:
        raise ValueError("n_cells must be between 1 and 600")
    frames, rows, excluded = [], [], []
    for name, entry in sorted(manifest_rows.items()):
        path = converted / name
        recording = parse_recording(path.stem)
        if recording is None or recording["state"] == "still":
            excluded.append(name)
            continue
        if path.suffix != ".npy" or sha256_file(path) != entry["output_sha256"]:
            raise ValueError(f"converted-file integrity check failed: {path}")
        cube, angle_offset, range_offset = loader(path)
        if not np.isfinite(cube).all():
            raise ValueError(f"nonfinite samples: {path}")
        centered = np.asarray(cube, dtype=np.complex128)
        centered -= centered.mean(axis=0, keepdims=True)
        variance = np.mean(np.abs(centered) ** 2, axis=0)
        # Stable sort makes ties resolve by flattened angle-major cell index.
        cells = np.argsort(-variance.ravel(), kind="stable")[:n_cells]
        target, pose = recording["target"], recording["pose"]
        for rank, cell in enumerate(cells):
            angle, distance = np.unravel_index(int(cell), variance.shape)
            for start in range(0, len(centered) - FRAME_LENGTH + 1, FRAME_LENGTH):
                signal = centered[start:start + FRAME_LENGTH, angle, distance]
                frames.append(np.stack([signal.real, signal.imag]).astype(np.float32))
                rows.append({
                    "frame_id": len(rows), "group": f"{target}_{pose}",
                    "group_recording": path.stem, "group_pose": f"{target}_{pose}",
                    "campaign": "2021-05-19_indoor", "pose": pose,
                    "mount": "tripod" if pose.startswith("tripod_") else "turntable",
                    "target": target, "state": recording["state"],
                    "absorber_variant": recording["variant"],
                    "y_bird_drone": "bird" if target == "bird" else "uas",
                    "y_target3": target, "source_file": entry["source_file"],
                    "converted_file": str(path), "source_sha256": entry["source_sha256"],
                    "start_sample": start, "cell_rank": rank,
                    "angle_bin_crop": int(angle), "range_bin_crop": int(distance),
                    "angle_bin_original": int(angle + angle_offset),
                    "range_bin_original": int(distance + range_offset),
                    "dynamic_variance": float(variance[angle, distance]),
                })
    if not frames:
        raise ValueError("no moving target recordings found")
    array, meta = np.stack(frames), pd.DataFrame(rows)
    tasks = {"bird_drone": "y_bird_drone", "target3": "y_target3"}
    info = {
        "name": "uas_birds_60ghz", "fs": None, "fs_assumed": 1000,
        "frame_length": FRAME_LENGTH, "channels": ["I_mean_removed", "Q_mean_removed"],
        "tasks": tasks, "primary_task": "bird_drone", "n_frames": len(meta),
        "n_groups": int(meta["group"].nunique()),
        "n_recordings": int(meta["group_recording"].nunique()),
        "class_counts": {task: meta[col].value_counts().sort_index().to_dict()
                         for task, col in tasks.items()},
        "groups_per_class": {task: meta.groupby(col)["group"].nunique().to_dict()
                             for task, col in tasks.items()},
        "recordings_per_class": {task: meta.groupby(col)["group_recording"].nunique().to_dict()
                                 for task, col in tasks.items()},
        "preparation": {"seed": SEED, "source": source, "n_cells": n_cells,
                        "cell_selection": "top dynamic complex variance, stable flat-index ties",
                        "clutter_removal": "subtract full-recording complex mean per cell",
                        "window": "nonoverlapping 1000 samples, discard short tail",
                        "grouping": "target plus pose, merging settings and absorber variants",
                        "excluded_files": excluded,
                        "conversion_manifest_sha256": sha256_file(manifest_path)},
        "source_url": "https://github.com/sk8053/drone_bird_detection_with_radar/tree/0e991a1b99f8eac2acab420254f656dc434d7f38",
        "dataport_url": "https://doi.org/10.21227/c8r5-6h79",
        "license": "No GitHub license; research use only; no redistribution",
        "citation": "Kang et al., Millimeter Wave Radar Measurements: Distinguishing UAS and Birds Based on 60 GHz micro-Doppler Signatures, IEEE VTC2024-Fall, doi:10.1109/VTC2024-Fall63153.2024.10757735",
        "notes": [
            "One indoor campaign, mounted targets and a bionic toy bird, not wild birds or free flight.",
            "No independent flight/session IDs. Conservative target-pose groups merge repeated settings.",
            "PRF undocumented. fs is null; fs_assumed=1000 is only a plot-based hypothesis.",
            "Features use cycles/sample. Physical Doppler and velocity require measured PRF.",
            "16 spatial cells are correlated views, not independent recordings.",
            "Moving-only tasks exclude static targets and pole. No random sampling or balancing.",
            "Full DataPort path untested; must verify layout and crop equality after authorized conversion.",
        ],
    }
    destination = base / "prepared"
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "frames.npy", array, allow_pickle=False)
    meta.to_parquet(destination / "meta.parquet", index=False)
    (destination / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ["n_frames", "n_groups", "n_recordings", "class_counts", "groups_per_class"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["public", "dataport"], default="public")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("IMF_CLS_DATA", DEFAULT_ROOT)))
    parser.add_argument("--cells", type=int, default=N_CELLS)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("use Python -I")
    prepare(args.data_root / "uas_birds_60ghz", args.source, args.cells)


if __name__ == "__main__":
    main()
