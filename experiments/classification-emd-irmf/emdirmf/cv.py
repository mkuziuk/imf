"""Grouped, stratified, repeated cross-validation shared by every method (PROTOCOL.md, "Evaluation").

make_splits(name, task) creates StratifiedGroupKFold(5) x seeds (0, 1, 2) once and
stores test indices in <DATA_ROOT>/NAME/splits_<task>.json; later calls reload it
and refuse to proceed if the frames, groups or labels changed.
"""

import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._search import BaseSearchCV

from .io import label_column, load_info, load_meta, meta_fingerprint
from .paths import splits_path

N_SPLITS = 5
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Split:
    split_id: int
    seed: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def _check_split_set(splits, groups, n_frames):
    seeds = sorted({split.seed for split in splits})
    for split in splits:
        overlap = set(groups[split.train]) & set(groups[split.test])
        if overlap:
            raise AssertionError(f"split {split.split_id}: groups in train and test: {sorted(overlap)[:5]}")
        if len(np.intersect1d(split.train, split.test)):
            raise AssertionError(f"split {split.split_id}: frames in train and test")
    for seed in seeds:
        tests = np.concatenate([split.test for split in splits if split.seed == seed])
        if len(tests) != n_frames or not np.array_equal(np.sort(tests), np.arange(n_frames)):
            raise AssertionError(f"seed {seed}: test folds do not partition the frames")


def _splits_from_records(records, n_frames):
    every = np.arange(n_frames)
    splits = []
    for record in records:
        test = np.asarray(record["test"], dtype=np.int64)
        splits.append(Split(record["split_id"], record["seed"], record["fold"], np.setdiff1d(every, test), test))
    return splits


def load_splits(name, task=None):
    """Reload stored splits as a list of Split (train indices are the complement of test)."""
    info, meta = load_info(name), load_meta(name)
    task = info["primary_task"] if task is None else task
    with open(splits_path(name, task)) as handle:
        stored = json.load(handle)
    column = label_column(info, task)
    fingerprint = meta_fingerprint(meta, ["frame_id", "group", column])
    if stored["meta_fingerprint"] != fingerprint or stored["n_frames"] != len(meta):
        raise RuntimeError(
            f"{splits_path(name, task)} no longer matches meta.parquet (frames, groups or labels "
            "changed); recreate with make_splits(..., overwrite=True) and rerun every method"
        )
    splits = _splits_from_records(stored["splits"], len(meta))
    _check_split_set(splits, meta["group"].astype(str).to_numpy(), len(meta))
    return splits


def make_splits(name, task=None, *, n_splits=N_SPLITS, seeds=SEEDS, overwrite=False, verbose=True):
    """Create (once) or reload the shared splits for a dataset and task."""
    info, meta = load_info(name), load_meta(name)
    task = info["primary_task"] if task is None else task
    path = splits_path(name, task)
    if path.exists() and not overwrite:
        return load_splits(name, task)
    column = label_column(info, task)
    y = meta[column].astype(str).to_numpy()
    groups = meta["group"].astype(str).to_numpy()
    n_frames = len(meta)
    records = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            folds = list(splitter.split(np.zeros(n_frames), y, groups))
        for fold, (_, test) in enumerate(folds):
            records.append({"split_id": len(records), "seed": int(seed), "fold": fold, "test": test.tolist()})
    splits = _splits_from_records(records, n_frames)
    _check_split_set(splits, groups, n_frames)
    classes = sorted(set(y))
    fold_counts = []
    for split in splits:
        train_classes, test_classes = set(y[split.train]), set(y[split.test])
        fold_counts.append({
            "split_id": split.split_id,
            "n_train": len(split.train),
            "n_test": len(split.test),
            "test_class_counts": pd.Series(y[split.test]).value_counts().sort_index().to_dict(),
        })
        if len(train_classes) < len(classes) or len(test_classes) < len(classes):
            missing = sorted(set(classes) - (train_classes & test_classes))
            print(f"warning: split {split.split_id} lacks classes {missing} in train or test")
    payload = {
        "dataset": name,
        "task": task,
        "label_column": column,
        "n_frames": n_frames,
        "n_groups": int(len(set(groups))),
        "n_splits": n_splits,
        "seeds": [int(seed) for seed in seeds],
        "splitter": "sklearn.model_selection.StratifiedGroupKFold(shuffle=True, random_state=seed)",
        "meta_fingerprint": meta_fingerprint(meta, ["frame_id", "group", column]),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folds": fold_counts,
        "splits": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)
    if verbose:
        sizes = [len(split.test) for split in splits]
        print(f"wrote {path}: {len(splits)} splits, test sizes {min(sizes)}-{max(sizes)}")
    return splits


def split_metrics(y_true, y_pred, labels):
    """Balanced accuracy, macro-F1, accuracy and per-class recall (NaN if absent from y_true)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }
    for label in labels:
        mask = y_true == label
        out[f"recall_{label}"] = float(np.mean(y_pred[mask] == label)) if mask.any() else np.nan
    return out


def fit_predict_split(estimator, X, y, split, *, groups=None, labels=None,
                      return_probabilities=False):
    """Return metrics and predictions from a fresh training-fold fit.

    String labels use Unicode arrays; numeric labels retain their dtype. With
    return_probabilities, also return native probabilities and their class order.
    Estimators without predict_proba return an (n_test, 0) array. The default
    two-item return value is unchanged.
    """
    y = np.asarray(y)
    if y.dtype.kind == "O" and all(isinstance(label, str) for label in y):
        y = y.astype(str)
    labels = sorted(set(y)) if labels is None else labels
    model = clone(estimator)
    fit_params = {}
    if groups is not None and isinstance(model, BaseSearchCV):
        fit_params["groups"] = groups[split.train]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        model.fit(X[split.train], y[split.train], **fit_params)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = np.asarray(model.predict(X[split.test]))
        if y.dtype.kind == "U":
            y_pred = y_pred.astype(str)
        predict_seconds = time.perf_counter() - started
        if return_probabilities:
            probabilities = model.predict_proba(X[split.test]) if hasattr(model, "predict_proba") \
                else np.empty((len(split.test), 0))
            classes = np.asarray(model.classes_, dtype=y.dtype) if probabilities.shape[1] else np.array([], dtype=y.dtype)
    row = {
        "split_id": split.split_id,
        "seed": split.seed,
        "fold": split.fold,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "best_params": json.dumps(getattr(model, "best_params_", None), default=str),
    }
    row.update(split_metrics(y[split.test], y_pred, labels))
    if return_probabilities:
        return row, y_pred, probabilities, classes
    return row, y_pred


def evaluate(pipeline_factory, X, y, splits, *, groups=None, n_jobs=1, return_predictions=False):
    """Per-split metrics for an estimator factory (or estimator) over the shared splits.

    Fitting, scaling and tuning happen on each training fold only. Pass groups so
    that inner searches (GridSearchCV with a group-aware cv) never split a group.
    Returns a DataFrame (and an out-of-fold predictions DataFrame if requested).
    """
    from joblib import Parallel, delayed, parallel_config

    estimator = pipeline_factory() if callable(pipeline_factory) and not hasattr(pipeline_factory, "fit") \
        else pipeline_factory
    X = np.asarray(X)
    y = np.asarray(y).astype(str)
    labels = sorted(set(y))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(fit_predict_split)(estimator, X, y, split, groups=groups, labels=labels) for split in splits
        )
    table = pd.DataFrame([row for row, _ in outputs])
    if not return_predictions:
        return table
    predictions = pd.concat(
        [pd.DataFrame({"split_id": split.split_id, "seed": split.seed, "frame_id": split.test,
                       "y_true": y[split.test], "y_pred": y_pred})
         for split, (_, y_pred) in zip(splits, outputs)],
        ignore_index=True,
    )
    return table, predictions
