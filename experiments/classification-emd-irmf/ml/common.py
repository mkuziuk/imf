"""Training-fold-only validation and common phase-3 output conventions."""

import argparse
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before numerical imports when entered from a runner. Runtime limits below
# also cover imports made earlier by a notebook or pytest.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from emdirmf.cv import load_splits, split_metrics
from emdirmf.decompose import canonical_tag
from emdirmf.io import label_column, load_info, load_meta
from emdirmf.paths import decomp_dir, output_dir, results_dir
from emdirmf.stats import compare_methods

DEFAULT_METHODS = ("raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41")
FEATURE_SETS = ("comp", "comp+log_scale")


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--quick", action="store_true", help="first seed and reduced model budget")
    parser.add_argument("--workers", type=int, default=1, help="CPU threads; folds run sequentially")
    parser.add_argument("--folds", help="first N shared splits, or explicit comma-separated split IDs")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--log-scale", choices=("both", "without", "with"), default="both")
    parser.add_argument("--out", help="override output/.../DATASET/ml/RUNNER")
    return parser


def feature_sets(args):
    return FEATURE_SETS if args.log_scale == "both" else (
        "comp+log_scale" if args.log_scale == "with" else "comp",
    )


def select_splits(splits, folds=None, quick=False):
    if folds:
        if "," in str(folds):
            ids = [int(value) for value in str(folds).split(",")]
            if len(ids) != len(set(ids)) or not set(ids) <= {s.split_id for s in splits}:
                raise ValueError("fold IDs must be distinct and present in the shared splits")
            chosen = [s for s in splits if s.split_id in ids]
        else:
            count = int(folds)
            if count < 1 or count > len(splits):
                raise ValueError(f"fold count must be 1..{len(splits)}")
            chosen = splits[:count]
    elif quick:
        chosen = [s for s in splits if s.seed == splits[0].seed]
    else:
        chosen = splits
    return chosen


def inner_split(train, y, groups, seed):
    """Absolute fit/validation indices, derived from outer training rows only.

    Require every outer-training class on both sides. Fail explicitly when too
    few recording groups make a meaningful validation split impossible.
    """
    train = np.asarray(train, dtype=np.int64)
    classes = set(y[train])
    n_splits = min(3, len(np.unique(groups[train])))
    if n_splits < 2:
        raise ValueError("need at least two training groups for inner validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit, valid in splitter.split(np.zeros(len(train)), y[train], groups[train]):
        fit, valid = train[fit], train[valid]
        if set(y[fit]) == classes and set(y[valid]) == classes:
            if set(groups[fit]) & set(groups[valid]):
                raise AssertionError("inner group leakage")
            return fit, valid
    raise ValueError("inner grouped validation cannot cover all training classes")


def setup(args, runner):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    threadpool_limits(args.workers)
    info, meta = load_info(args.dataset), load_meta(args.dataset)
    task = info["primary_task"] if args.task is None else args.task
    y = meta[label_column(info, task)].astype(str).to_numpy()
    groups = meta["group"].astype(str).to_numpy()
    splits = select_splits(load_splits(args.dataset, task), args.folds, args.quick)
    for split in splits:
        if set(y[split.train]) != set(y):
            raise ValueError(f"split {split.split_id}: outer training fold is missing classes")
    methods = list(dict.fromkeys(canonical_tag(tag) for tag in args.methods))
    out = Path(args.out) if args.out else output_dir(args.dataset, "ml", runner)
    out.mkdir(parents=True, exist_ok=True)
    context = {"dataset": args.dataset, "task": task, "runner": runner, "methods": methods,
               "n_frames": len(meta), "n_groups": len(set(groups)), "n_classes": len(set(y)),
               "split_ids": [s.split_id for s in splits], "workers": args.workers,
               "created": datetime.now(timezone.utc).isoformat(), "params": vars(args),
               "inner_validation": "first class-complete StratifiedGroupKFold(3) split of outer train",
               "versions": versions(), "started": time.perf_counter()}
    return context, meta, y, groups, splits, methods, out


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "xgboost", "torch", "aeon"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def record(rows, predictions, split, y, pred, method, feature_set, classifier,
           fit_seconds, predict_seconds, **extra):
    row = {"method": method, "feature_set": feature_set, "classifier": classifier,
           "split_id": split.split_id, "seed": split.seed, "fold": split.fold,
           "n_train": len(split.train), "n_test": len(split.test),
           "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, **extra}
    row.update(split_metrics(y[split.test], pred, sorted(set(y))))
    rows.append(row)
    predictions.append(pd.DataFrame({
        **{key: row[key] for key in ("method", "feature_set", "classifier", "split_id", "seed", "fold")},
        "frame_id": split.test, "y_true": y[split.test], "y_pred": pred,
    }))
    print(f"{method} {feature_set} {classifier} split={split.split_id}: "
          f"BA={row['balanced_accuracy']:.3f}, fit={fit_seconds:.2f}s, predict={predict_seconds:.2f}s",
          flush=True)


def finish(context, out, rows, predictions, *, notes=(), extra=None):
    task = context["task"]
    folds = pd.DataFrame(rows)
    folds.insert(0, "task", task)
    folds.insert(0, "dataset", context["dataset"])
    folds.to_csv(out / f"results_folds_{task}.csv", index=False)
    ids = ["dataset", "task", "method", "feature_set", "classifier", "split_id", "seed", "fold",
           "n_train", "n_test"]
    metrics = ["balanced_accuracy", "macro_f1", "accuracy", "fit_seconds", "predict_seconds"]
    metrics += [c for c in folds if c.startswith("recall_")]
    long = folds.melt(id_vars=ids, value_vars=metrics, var_name="metric", value_name="value")
    long.to_csv(out / f"results_long_{task}.csv", index=False)
    pairs = [(a, b) for i, a in enumerate(context["methods"]) for b in context["methods"][i + 1:]]
    tables = []
    for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
        table = compare_methods(long, metric=metric, pairs=pairs, by=("feature_set", "classifier"))
        if len(table):
            tables.append(table)
    comparisons = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=[
        "metric", "feature_set", "classifier", "method_a", "method_b", "n_splits",
        "mean_diff", "ci_low", "ci_high", "p_nb", "p_nb_holm", "p_wilcoxon", "p_wilcoxon_holm",
    ])
    comparisons.to_csv(out / f"comparisons_{task}.csv", index=False)
    pred_dir = results_dir(context["dataset"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{context['runner']}_{task}_predictions.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(pred_path, index=False)
    context["seconds"] = time.perf_counter() - context.pop("started")
    context["predictions"] = str(pred_path)
    context["fold_timings"] = folds[ids + ["fit_seconds", "predict_seconds"]].to_dict("records")
    parameter_columns = [column for column in ("best_params", "best_epoch", "n_features", "n_parameters")
                         if column in folds]
    context["fold_parameters"] = folds[ids + parameter_columns].to_dict("records")
    context["notes"] = list(notes)
    context["decomposition_costs"] = []
    for tag in context["methods"]:
        path = decomp_dir(context["dataset"]) / f"{tag}.json"
        if tag != "raw" and path.exists():
            info = json.loads(path.read_text())
            context["decomposition_costs"].append({"method": tag,
                "seconds_per_frame": info.get("seconds_per_frame"), "n_failed": info.get("n_failed"),
                "irmf": info.get("irmf")})
    context.update(extra or {})
    (out / f"run_{task}.json").write_text(json.dumps(context, indent=2, default=str) + "\n")
    scores = folds.groupby(["method", "feature_set", "classifier"])["balanced_accuracy"].agg(["mean", "std"])
    lines = [f"# {context['runner']}: {context['dataset']} / {task}", "",
             f"{len(folds)} fits on shared split IDs {context['split_ids']}; "
             f"wall time {context['seconds']:.1f} s with {context['workers']} CPU threads.", "",
             "Balanced accuracy, mean and SD across the selected paired folds:", "",
             scores.to_markdown(floatfmt=".4f"), "", *notes, "",
             "Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, "
             "Holm across each metric's table. Fewer than two splits produce no inference.", ""]
    if len(comparisons):
        columns = ["feature_set", "classifier", "method_a", "method_b", "mean_diff", "ci_low",
                   "ci_high", "p_nb_holm", "p_wilcoxon_holm"]
        lines += [comparisons[comparisons.metric == "balanced_accuracy"][columns].to_markdown(
            index=False, floatfmt=".4f"), ""]
    (out / f"summary_{task}.md").write_text("\n".join(lines))
    print(f"wrote {out}, total {context['seconds']:.1f}s", flush=True)
    return folds, comparisons
