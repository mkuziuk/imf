#!/usr/bin/env python3
"""Grouped boosted-tree and ExtraTrees baselines on the shared feature tables."""

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from ml.common import feature_sets, finish, inner_split, parser_for, record, setup

import json
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from emdirmf.decompose import scale_order, slot_labels
from emdirmf.features import feature_columns
from emdirmf.io import load_info
from emdirmf.paths import features_dir


def estimator(name, args, seed, n_estimators, min_leaf=2, early=False, n_classes=2):
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15,
                              min_child_samples=min_leaf, reg_lambda=1.0, verbosity=-1,
                              n_jobs=args.workers, random_state=seed, deterministic=True,
                              force_col_wise=True)
    if name == "xgboost":
        return XGBClassifier(n_estimators=n_estimators, learning_rate=0.05, max_depth=4,
                             min_child_weight=min_leaf, reg_lambda=1.0, tree_method="hist",
                             eval_metric="mlogloss" if n_classes > 2 else "logloss",
                             n_jobs=args.workers, random_state=seed,
                             **({"early_stopping_rounds": 12} if early else {}))
    return ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_leaf,
                                max_features="sqrt", n_jobs=args.workers, random_state=seed)


def fit_tree(name, args, X, y, groups, split):
    fit, valid = inner_split(split.train, y, groups, split.seed)
    seed = split.seed * 100 + split.fold
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_fit = imputer.fit_transform(X[fit])
    x_valid = imputer.transform(X[valid])
    weights = compute_sample_weight("balanced", y[fit])
    rounds = 60 if args.quick else 500
    n_classes = len(np.unique(y[split.train]))
    if name == "extratrees":
        candidates = []
        for leaf in (1, 3):
            model = estimator(name, args, seed, 64 if args.quick else 300, leaf)
            model.fit(x_fit, y[fit], sample_weight=weights)
            candidates.append((balanced_accuracy_score(y[valid], model.predict(x_valid)), leaf))
        leaf = max(candidates, key=lambda value: value[0])[1]
        rounds = 64 if args.quick else 300
    else:
        leaf = 10 if name == "lightgbm" else 2
        model = estimator(name, args, seed, rounds, leaf, early=True, n_classes=n_classes)
        options = {"eval_sample_weight" if name == "lightgbm" else "sample_weight_eval_set":
                   [compute_sample_weight("balanced", y[valid])]}
        if name == "lightgbm":
            options.update(eval_X=x_valid, eval_y=y[valid])
            options["callbacks"] = [early_stopping(12, verbose=False)]
        else:
            options["eval_set"] = [(x_valid, y[valid])]
            options["verbose"] = False
        model.fit(x_fit, y[fit], sample_weight=weights, **options)
        rounds = (model.best_iteration_ if name == "lightgbm" else model.best_iteration + 1) or rounds
    # Refit the selected budget and imputer on all outer training rows.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(X[split.train])
    final = estimator(name, args, seed, rounds, leaf, n_classes=n_classes)
    final.fit(x_train, y[split.train], sample_weight=compute_sample_weight("balanced", y[split.train]))
    params = {"n_estimators": int(rounds), "min_leaf": leaf, "inner_fit": len(fit),
              "inner_valid": len(valid), "seed": seed}
    return final, imputer, params


def importance_rows(model, columns, tag, length, feature_set, name, split_id):
    """Aggregate normalized model importance over features and input channels."""
    values = np.asarray(model.feature_importances_, dtype=float)
    values /= values.sum() if values.sum() else 1.0
    order = scale_order(tag, length)
    labels = slot_labels(tag, length)
    totals = {}
    for column, value in zip(columns, values):
        match = re.match(r"ch\d+_c(\d+)_", column)
        slot = int(match.group(1)) if match else -1
        totals[slot] = totals.get(slot, 0.0) + value
    return [{"method": tag, "feature_set": feature_set, "classifier": name,
             "split_id": split_id, "storage_slot": slot,
             "scale_rank": order.index(slot) if slot >= 0 else -1,
             "slot_label": labels[slot] if slot >= 0 else "log_scale",
             "importance": value,
             "importance_type": "gain" if name == "xgboost" else (
                 "split_count" if name == "lightgbm" else "impurity")}
            for slot, value in totals.items()]


def run(args):
    context, meta, y, groups, splits, methods, out = setup(args, "trees")
    if args.device == "mps":
        raise ValueError("tree implementations use CPU; use --device cpu or auto")
    encoder = LabelEncoder().fit(y)
    encoded = encoder.transform(y)
    rows, predictions, importance = [], [], []
    length = load_info(args.dataset)["frame_length"]
    for tag in methods:
        table = pd.read_parquet(features_dir(args.dataset) / f"{tag}.parquet")
        if not np.array_equal(table.frame_id.to_numpy(), meta.frame_id.to_numpy()):
            raise ValueError(f"{tag}: features do not align with prepared frames")
        if not table.decomp_ok.all():
            raise ValueError(f"{tag}: failed decompositions; repair before paired ML")
        for feature_set in feature_sets(args):
            columns = feature_columns(table.columns, include_log_scale=feature_set.endswith("log_scale"))
            X = table[columns].to_numpy(dtype=np.float32, copy=True)
            X[~np.isfinite(X)] = np.nan
            for name in args.classifiers:
                for split in splits:
                    started = time.perf_counter()
                    model, imputer, params = fit_tree(name, args, X, encoded, groups, split)
                    fit_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    pred = encoder.inverse_transform(model.predict(imputer.transform(X[split.test])).astype(int))
                    predict_seconds = time.perf_counter() - started
                    record(rows, predictions, split, y, pred, tag, feature_set, name,
                           fit_seconds, predict_seconds, best_params=json.dumps(params))
                    importance.extend(importance_rows(model, columns, tag, length, feature_set, name, split.split_id))
    detail = pd.DataFrame(importance).sort_values(["method", "classifier", "feature_set", "split_id", "scale_rank"])
    detail.to_csv(out / f"importance_folds_{context['task']}.csv", index=False)
    keys = ["method", "classifier", "feature_set", "storage_slot", "scale_rank", "slot_label", "importance_type"]
    detail.groupby(keys).importance.agg(["mean", "std"]).reset_index().sort_values(
        ["method", "classifier", "feature_set", "scale_rank"]).to_csv(
            out / f"importance_scales_{context['task']}.csv", index=False)
    return finish(context, out, rows, predictions, notes=[
        "LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.",
        "ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. "
        "Median imputation and class weights are fitted inside each training partition.",
        "Importance is normalized per fitted model and summed over features and input channels by storage slot; "
        "scale_rank=0 is finest. These model-specific importances are descriptive, not causal.",
    ], extra={"device": "cpu", "feature_sets": list(feature_sets(args)), "classifier_params": {
        "lightgbm": {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10,
                     "reg_lambda": 1.0, "early_stopping_rounds": 12},
        "xgboost": {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 2,
                    "reg_lambda": 1.0, "tree_method": "hist", "early_stopping_rounds": 12},
        "extratrees": {"max_features": "sqrt", "leaf_grid": [1, 3]},
    }})


def main(argv=None):
    parser = parser_for("Phase-3 LightGBM, XGBoost and ExtraTrees")
    parser.add_argument("--classifiers", nargs="+", choices=("lightgbm", "xgboost", "extratrees"),
                        default=["lightgbm", "xgboost", "extratrees"])
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
