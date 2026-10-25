"""Immutable numeric checkpoints under each dataset's robustness_laplace directory."""
from emdirmf.paths import data_root
from robustness.storage import digest, publish, save_json, save_npz
from robustness_laplace.contamination import DATASETS

__all__ = ['check_disk', 'digest', 'disk_bytes', 'publish', 'root', 'save_json', 'save_npz']
LIMIT_BYTES = 30_000_000_000


def root(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    return data_root()/dataset/'robustness_laplace'


def disk_bytes():
    return sum(p.stat().st_size for d in DATASETS if root(d).exists()
               for p in root(d).rglob('*') if p.is_file())


def check_disk():
    size = disk_bytes()
    if size > LIMIT_BYTES:
        raise RuntimeError(f'30 GB robustness_laplace budget exceeded: {size}')
    return size
