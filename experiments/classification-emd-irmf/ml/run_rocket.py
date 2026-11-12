#!/usr/bin/env python3
"""Batch-streamed aeon MiniRocket/MultiRocket with a standardized logistic head."""

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from ml.common import feature_sets, finish, inner_split, parser_for, record, setup

import json
import tempfile
import time

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from emdirmf.stacks import StackReader


def transformer(name, budget, workers, seed):
    if name == "minirocket":
        return MiniRocket(n_kernels=budget, n_jobs=workers, random_state=seed)
    # MultiRocket produces four pooling features on both signal and differences.
    return MultiRocket(n_kernels=max(84, budget // 8), n_features_per_kernel=4,
                       normalise=False, n_jobs=workers, random_state=seed)


def transform_to_disk(reader, model, fit_indices, all_indices, path, args, seed):
    sample = np.random.default_rng(seed).choice(fit_indices, min(len(fit_indices), args.fit_samples), replace=False)
    model.fit(reader.take(sample))
    values = None
    for indices, batch in reader.batches(all_indices, args.batch_size):
        transformed = np.asarray(model.transform(batch), dtype=np.float32)
        if values is None:
            values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                               shape=(reader.shape[0], transformed.shape[1]))
        values[indices] = transformed
    values.flush()
    return values


def blocks(indices, batch_size, seed=None):
    indices = np.asarray(indices, dtype=np.int64).copy()
    if seed is not None:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def input_features(values, indices, scales):
    block = np.asarray(values[indices], dtype=np.float32)
    return np.concatenate([block, scales[indices]], axis=1) if scales is not None else block


def fit_head(values, scales, train, y, alpha, args, seed):
    scaler = StandardScaler()
    for indices in blocks(train, args.head_batch_size):
        scaler.partial_fit(input_features(values, indices, scales))
    model = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed,
                          learning_rate="optimal", average=True, shuffle=False)
    weights = np.zeros(len(y))
    weights[train] = compute_sample_weight("balanced", y[train])
    classes = np.unique(y[train])
    for epoch in range(args.head_epochs):
        for indices in blocks(train, args.head_batch_size, seed + epoch):
            X = scaler.transform(input_features(values, indices, scales))
            model.partial_fit(X, y[indices], classes=classes, sample_weight=weights[indices])
    return scaler, model


def predict_head(values, scales, indices, head, args):
    scaler, model = head
    return np.concatenate([model.predict(scaler.transform(input_features(values, batch, scales)))
                           for batch in blocks(indices, args.head_batch_size)])


def run(args):
    context, _, y, groups, splits, methods, out = setup(args, "rocket")
    if args.device == "mps":
        raise ValueError("aeon ROCKET runs on CPU; use --device cpu or auto")
    import numba

    numba.set_num_threads(min(args.workers, numba.config.NUMBA_NUM_THREADS))
    rows, predictions, descriptions = [], [], {}
    for tag in methods:
        reader = StackReader(args.dataset, tag, args.downsample)
        descriptions[tag] = reader.description()
        scales = np.empty((len(y), reader.frames.shape[1]), dtype=np.float32)
        for indices in blocks(np.arange(len(y)), args.batch_size):
            scales[indices] = reader.log_scales(indices)
        for name in args.classifiers:
            for split in splits:
                seed = split.seed * 100 + split.fold
                fit, valid = inner_split(split.train, y, groups, split.seed)
                # Two transforms are necessary: inner validation must never inform
                # the learned ROCKET biases used to select regularization.
                with tempfile.TemporaryDirectory(prefix="emdirmf_rocket_") as scratch:
                    started = time.perf_counter()
                    model = transformer(name, args.features, args.workers, seed)
                    inner = transform_to_disk(reader, model, fit, split.train,
                                              Path(scratch) / "inner.npy", args, seed)
                    inner_transform_seconds = time.perf_counter() - started
                    choices, tuning_times = {}, {}
                    for feature_set in feature_sets(args):
                        started = time.perf_counter()
                        side = scales if feature_set.endswith("log_scale") else None
                        scores = []
                        for alpha in (1e-4, 1e-3, 1e-2):
                            head = fit_head(inner, side, fit, y, alpha, args, seed)
                            scores.append((balanced_accuracy_score(y[valid], predict_head(inner, side, valid, head, args)), alpha))
                        choices[feature_set] = max(scores, key=lambda item: item[0])[1]
                        tuning_times[feature_set] = time.perf_counter() - started
                    del inner, model
                    (Path(scratch) / "inner.npy").unlink()
                    started = time.perf_counter()
                    model = transformer(name, args.features, args.workers, seed)
                    # Only outer train is transformed before the final head fit.
                    values = transform_to_disk(reader, model, split.train, split.train,
                                               Path(scratch) / "outer.npy", args, seed)
                    outer_transform_seconds = time.perf_counter() - started
                    heads, times = {}, {}
                    for feature_set in feature_sets(args):
                        started = time.perf_counter()
                        side = scales if feature_set.endswith("log_scale") else None
                        heads[feature_set] = fit_head(values, side, split.train, y, choices[feature_set], args, seed)
                        times[feature_set] = inner_transform_seconds + tuning_times[feature_set] + outer_transform_seconds + time.perf_counter() - started
                    started = time.perf_counter()
                    for indices, batch in reader.batches(split.test, args.batch_size):
                        values[indices] = model.transform(batch)
                    test_transform_seconds = time.perf_counter() - started
                    for feature_set in feature_sets(args):
                        started = time.perf_counter()
                        side = scales if feature_set.endswith("log_scale") else None
                        pred = predict_head(values, side, split.test, heads[feature_set], args)
                        predict_seconds = test_transform_seconds + time.perf_counter() - started
                        record(rows, predictions, split, y, pred, tag, feature_set, name,
                               times[feature_set], predict_seconds,
                               transform_fit_seconds=inner_transform_seconds + outer_transform_seconds,
                               transform_predict_seconds=test_transform_seconds,
                               n_features=values.shape[1],
                               best_params=json.dumps({"alpha": choices[feature_set], "seed": seed}))
                    del values, model, heads
    return finish(context, out, rows, predictions, notes=[
        "MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. "
        "Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.",
        "The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. "
        "Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.",
        "Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in "
        "temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.",
        "The same total target feature budget, fit sample count and downsampling apply to every method. "
        "aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.",
    ], extra={"device": "cpu", "stacks": descriptions, "feature_sets": list(feature_sets(args)),
              "timings_share_transforms": True, "head": {"type": "SGDClassifier", "loss": "log_loss",
                  "average": True, "alpha_grid": [1e-4, 1e-3, 1e-2], "class_weights": "balanced"}})


def main(argv=None):
    parser = parser_for("Phase-3 streaming MiniRocket and MultiRocket")
    parser.add_argument("--classifiers", nargs="+", choices=("minirocket", "multirocket"), default=["minirocket"])
    parser.add_argument("--features", type=int, help="target transformed features, same budget for both models")
    parser.add_argument("--fit-samples", type=int, default=256, help="training frames to fit transform biases")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args(argv)
    args.features = args.features if args.features is not None else (672 if args.quick else 10080)
    args.head_epochs = args.head_epochs if args.head_epochs is not None else (5 if args.quick else 15)
    if args.features < 84 or min(args.batch_size, args.head_batch_size, args.head_epochs, args.fit_samples) < 1:
        parser.error("features must be >=84 and batch sizes, epochs and fit samples positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
