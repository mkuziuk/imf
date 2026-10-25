"""Filesystem layout shared by every phase of the study (see PROTOCOL.md)."""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/Users/mikhail/Datasets/imf-classification")
STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "classification-emd-irmf"


def data_root():
    """DATA_ROOT, read from IMF_CLS_DATA at call time so tests can redirect it."""
    return Path(os.environ.get("IMF_CLS_DATA", str(DEFAULT_DATA_ROOT)))


def dataset_dir(name):
    return data_root() / name


def prepared_dir(name):
    return dataset_dir(name) / "prepared"


def decomp_dir(name):
    return dataset_dir(name) / "decomp"


def features_dir(name):
    return dataset_dir(name) / "features"


def results_dir(name):
    """Bulky per-dataset results (out-of-fold predictions) that stay out of the repo."""
    return dataset_dir(name) / "results"


def splits_path(name, task):
    return dataset_dir(name) / f"splits_{task}.json"


def output_dir(name, *parts):
    """Small tables and figures that may be committed: output/classification-emd-irmf/<name>/."""
    return OUTPUT_ROOT.joinpath(name, *parts)
