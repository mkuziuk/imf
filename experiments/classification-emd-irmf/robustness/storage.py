"""Numeric-only immutable checkpoints under each dataset's robustness directory."""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from emdirmf.paths import data_root
from robustness.contamination import DATASETS


def root(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    return data_root()/dataset/'robustness'


def publish(path, writer):
    """Publish a new file atomically; never replace an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()  # Only our unpublished scratch inode; never an existing output.


def save_json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False).encode()
    publish(path, lambda f: f.write(data))


def save_npz(path, **arrays):
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError('object arrays would require pickle')
    publish(path, lambda f: np.savez_compressed(f, **arrays))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def disk_bytes():
    # Do not traverse sibling directories, especially raw.
    return sum(p.stat().st_size for d in DATASETS if root(d).exists()
               for p in root(d).rglob('*') if p.is_file())


def check_disk():
    size = disk_bytes()
    if size > 60_000_000_000:
        raise RuntimeError(f'60 GB robustness budget exceeded: {size}')
    return size
