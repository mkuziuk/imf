"""Run the unchanged classical driver with resumable probability-recording jobs.

All large caches stay outside the repository. Each fit uses one CPU thread.
Native probability classifiers also save their held-out class probabilities.
Linear SVM probabilities, if needed for a selected model, require a separate
training-only calibration and are deliberately not invented from hard labels.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["IMF_CLS_MAX_WORKERS"] = "3"

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV

REPO = Path("/Users/mikhail/.t3/worktrees/imf/t3-44ed3594")
STUDY = REPO / "experiments/classification-emd-irmf"
sys.path.insert(0, str(STUDY / "baselines"))
sys.path.insert(0, str(STUDY))

import run_classical as runner
from emdirmf.cv import split_metrics

DATA = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification")) / "esc50_animals"
OUT = REPO / "output/classification-emd-irmf/esc50_animals/classical"
MAIN = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
ABLATIONS = ["irmf_h1_a1.41", "irmf_h4_a1.41", "irmf_h2_a2", "imf_linear_a2"]
SCRATCH = DATA / "results/phase2_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
runner.tempfile.tempdir = str(SCRATCH)


def probability_job(key, estimator, X, y, split, groups, labels, *, task, batch):
    tag, feature_set, name = key
    feature_path = DATA / "features" / f"{tag}.parquet"
    signature = hashlib.sha256((repr(estimator.get_params()) + str(feature_path.stat().st_mtime_ns)
                               + repr(split.test.tolist())).encode()).hexdigest()[:12]
    cache = DATA / "results" / f"phase2_{batch}" / task / "jobs"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{tag}__{feature_set}__{name}__{split.split_id}__{signature}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return key, json.loads(str(saved["row"])), saved["y_pred"]
    np.random.seed(0)
    model = clone(estimator)
    fit_params = {"groups": groups[split.train]} if isinstance(model, BaseSearchCV) else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model.fit(X[split.train], y[split.train], **fit_params)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = model.predict(X[split.test])
        predict_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(X[split.test]) if hasattr(model, "predict_proba") else np.empty((0, 0))
    row = {"split_id": split.split_id, "seed": split.seed, "fold": split.fold,
           "n_train": len(split.train), "n_test": len(split.test),
           "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
           "best_params": json.dumps(getattr(model, "best_params_", None), default=str)}
    row.update(split_metrics(y[split.test], y_pred, labels))
    messages = sorted(set(str(warning.message) for warning in caught))
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row=json.dumps(row), y_pred=y_pred.astype(str),
                            frame_id=split.test, probabilities=probabilities,
                            classes=np.asarray(model.classes_, dtype=str), warnings=json.dumps(messages))
    temporary.replace(path)
    print(f"done {task} {tag} {feature_set} {name} split={split.split_id} "
          f"BA={row['balanced_accuracy']:.4f} fit={fit_seconds:.1f}s", flush=True)
    return key, row, np.asarray(y_pred)


def run_batch(batch, methods=None):
    from functools import partial

    allowed = MAIN if batch == "main" else ABLATIONS
    methods = allowed if methods is None else methods
    if any(tag not in allowed for tag in methods):
        raise ValueError(f"Unsupported {batch} method selection: {methods}")
    output = OUT if batch == "main" else OUT / "ablations"
    # The upstream writer uses one predictions filename per task. Isolate batches
    # so the ablation run cannot overwrite the main-method OOF predictions.
    runner.results_dir = lambda dataset: DATA / "results" / f"phase2_{batch}"
    for task in ("animal", "bird"):
        runner._job = partial(probability_job, task=task, batch=batch)
        runner.run("esc50_animals", task=task, methods=methods, workers=3, out=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=("main", "ablations"), default="main")
    parser.add_argument("--methods", nargs="+", default=None)
    arguments = parser.parse_args()
    run_batch(arguments.batch, arguments.methods)
