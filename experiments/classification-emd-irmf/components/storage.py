"""Append-only checkpoints, source fingerprints, and a 60 GB allocation cap."""

import hashlib
import json
from pathlib import Path

import numpy as np

from components.integrity import DATA_ROOT, DATASETS, HERE

STUDY = HERE.parent
OUTPUT = STUDY.parents[1] / "output/classification-emd-irmf/components"
BUDGET_BYTES = 60_000_000_000


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hash(paths):
    return {str(Path(path).relative_to(STUDY)): file_hash(path) for path in paths}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
        raise TypeError("object arrays are forbidden")
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def allocated_bytes():
    return sum(path.stat().st_size for dataset in DATASETS
               for path in (DATA_ROOT / dataset / "components").rglob("*") if path.is_file())


def check_budget(additional=0):
    size = allocated_bytes()
    if size + additional > BUDGET_BYTES:
        raise RuntimeError(f"STOP: component experiment disk budget exceeded: {size}+{additional}")
    return size


def run_dir(dataset):
    return DATA_ROOT / dataset / "components"
