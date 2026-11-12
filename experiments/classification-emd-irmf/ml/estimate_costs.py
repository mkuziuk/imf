#!/usr/bin/env python3
"""Extrapolate synthetic fold timings without opening any real-data arrays.

This is a planning model, not a benchmark of the target datasets. Its formulas,
budgets, CPU scaling and broad uncertainty interval are saved with the tables.
"""

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from ml.common import versions

import argparse
import json
import time

import numpy as np
import pandas as pd

from emdirmf.decompose import n_slots
from emdirmf.io import load_info
from emdirmf.paths import output_dir

TARGETS = {"dronerf": (12000, 2, 2048, 10), "birdclef2023": (12000, 1, 2048, 16),
           "esc50_animals": (5100, 1, 2048, 10)}


def calibrate_head(source, task, directory):
    """Separate minibatch overhead from feature-count cost on synthetic size only."""
    from types import SimpleNamespace
    from threadpoolctl import threadpool_limits
    from emdirmf.cv import load_splits
    from emdirmf.io import label_column, load_meta
    from ml.common import inner_split
    from ml.run_rocket import fit_head, predict_head

    threadpool_limits(2)
    info, meta = load_info(source), load_meta(source)
    y = meta[label_column(info, task)].astype(str).to_numpy()
    split = load_splits(source, task)[0]
    fit, valid = inner_split(split.train, y, meta.group.to_numpy(), 0)
    rng = np.random.default_rng(0)
    # These are a timing workload, not an additional classification experiment.
    # Dense unit-Gaussian features exercise the same fixed-pass head code.
    X = rng.standard_normal((len(y), 10080), dtype=np.float32)
    measurements = []
    for features, epochs in ((672, 5), (10080, 15)):
        args = SimpleNamespace(head_batch_size=128, head_epochs=epochs)
        started = time.perf_counter()
        for alpha in (1e-4, 1e-3, 1e-2):
            head = fit_head(X[:, :features], None, fit, y, alpha, args, 0)
            predict_head(X[:, :features], None, valid, head, args)
        head = fit_head(X[:, :features], None, split.train, y, 0.001, args, 0)
        predict_head(X[:, :features], None, split.test, head, args)
        measurements.append({"features": features, "epochs": epochs,
                             "seconds": time.perf_counter() - started})
    calibration = {"dataset": source, "task": task, "n_frames": len(y), "seed": 0,
        "workload": "unit-Gaussian float32 features, three inner heads and one full-train head",
        "measurements": measurements, "full_over_quick": measurements[1]["seconds"] / measurements[0]["seconds"],
        "versions": versions()}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "head_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def estimate(source="synthetic", task="signal", cores=16, efficiency=0.5):
    root = output_dir(source, "ml")
    source_info = load_info(source)
    calibration_path = root / "costs" / "head_calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    records, measured = [], []
    for runner in ("trees", "rocket", "cnn"):
        directory = root / runner
        context = json.loads((directory / f"run_{task}.json").read_text())
        folds = pd.read_csv(directory / f"results_folds_{task}.csv")
        n0, classes0 = context["n_frames"], context["n_classes"]
        params = context["params"]
        # Both CPU-bound transforms and single-threaded head work improve less
        # than ideally with 16 cores. MPS gets no assumed core-count speedup.
        speedup = max(1.0, cores / context["workers"] * efficiency)
        curve = pd.read_csv(directory / f"curves_{task}.csv") if runner == "cnn" else None
        for (method, feature_set, classifier), cell in folds.groupby(["method", "feature_set", "classifier"]):
            median_fit = float(cell.fit_seconds.median())
            median_predict = float(cell.predict_seconds.median())
            measured.append({"runner": runner, "method": method, "feature_set": feature_set,
                             "classifier": classifier, "n_splits": len(cell),
                             "median_fit_seconds": median_fit, "median_predict_seconds": median_predict})
            if runner == "cnn":
                description = context["stacks"][method]
                length0 = description["shape"][-1]
                channels0 = description["shape"][1]
                epochs = curve[(curve.method == method) & (curve.feature_set == feature_set) & (curve.stage == "refit")]
                epoch_seconds = float(epochs.seconds.median())
            elif runner == "rocket":
                description = context["stacks"][method]
                length0 = description["shape"][-1]
                channels0 = description["shape"][1]
            else:
                length0 = source_info["frame_length"]
                channels0 = len(source_info["channels"]) * n_slots(method, length0)
            for dataset, (n, channels, length, classes) in TARGETS.items():
                target_channels = channels * n_slots(method, length)
                n_ratio = n / n0
                class_ratio = classes / classes0
                channel_ratio = target_channels / channels0
                length_ratio = length / length0
                if runner == "trees":
                    budgets = [json.loads(value)["n_estimators"] for value in cell.best_params]
                    target_budget = 300 if classifier == "extratrees" else 500
                    # Boosters scale with class count; ExtraTrees splits share a tree.
                    factor = n_ratio * channel_ratio * (class_ratio if classifier != "extratrees" else 1)
                    if classifier == "extratrees":
                        budget_factor = target_budget / np.median(budgets)
                    else:
                        source_cap = 60 if params["quick"] else 500
                        source_work = np.median([rounds + min(source_cap, rounds + 12) for rounds in budgets])
                        budget_factor = 2 * target_budget / source_work
                    factor *= budget_factor / speedup
                    seconds = (median_fit + median_predict) * factor
                    pilot_seconds = (median_fit + median_predict) * n_ratio * channel_ratio * (
                        class_ratio if classifier != "extratrees" else 1) / speedup
                    target_folds = 15
                elif runner == "rocket":
                    budget_ratio = 10080 / float(cell.n_features.median())
                    transform = float((cell.transform_fit_seconds + cell.transform_predict_seconds).median())
                    head = max(0, median_fit + median_predict - transform)
                    # Half of shared transform work is allocated to each scale
                    # variant, so summing the table does not double-count it.
                    share = 0.5 if len(folds.feature_set.unique()) == 2 else 1.0
                    transform_factor = n_ratio * length_ratio * np.sqrt(channel_ratio) * budget_ratio
                    calibrated = calibration is not None and params["head_epochs"] == 5 and float(cell.n_features.median()) == 672
                    head_budget_factor = calibration["full_over_quick"] if calibrated else budget_ratio * 15 / params["head_epochs"]
                    head_factor = n_ratio * class_ratio * head_budget_factor
                    # Tiny partial_fit updates are slower with per-class thread
                    # pools. The streaming head deliberately uses serial updates.
                    seconds = share * transform * transform_factor / speedup + head * head_factor
                    pilot_seconds = share * transform * n_ratio * length_ratio * np.sqrt(channel_ratio) / speedup + head * n_ratio * class_ratio
                    target_folds = 15
                else:
                    # Selection: 2/3 train and 1/3 forward-only validation;
                    # refit: all train. Budget 30 epochs each; forward = 1/3 train.
                    equivalents = 30 * (1 + 2 / 3 + 1 / 9)
                    factor = n_ratio * length_ratio * (0.9 + 0.1 * channel_ratio)
                    seconds = epoch_seconds * equivalents * factor + median_predict * n_ratio * length_ratio
                    pilot_seconds = epoch_seconds * equivalents * 6 / 30 * factor + median_predict * n_ratio * length_ratio
                    target_folds = 5
                records.append({"dataset": dataset, "runner": runner, "method": method,
                                "feature_set": feature_set, "classifier": classifier,
                                "target_folds": target_folds, "seconds_per_fold": float(seconds),
                                "pilot_5fold_minutes": float(pilot_seconds * 5 / 60),
                                "total_minutes": float(seconds * target_folds / 60),
                                "low_minutes": float(seconds * target_folds / 120),
                                "high_minutes": float(seconds * target_folds / 30)})
    out = root / "costs"
    out.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(records)
    pd.DataFrame(measured).to_csv(out / "synthetic_fold_timings.csv", index=False)
    detail.to_csv(out / "estimates_detail.csv", index=False)
    # MiniRocket is the standard run. MultiRocket is shown separately as optional.
    totals = detail.groupby(["dataset", "runner", "classifier", "target_folds"])[
        ["total_minutes", "low_minutes", "high_minutes", "pilot_5fold_minutes"]].sum().reset_index()
    totals.to_csv(out / "estimates.csv", index=False)
    assumptions = [
        "Planning estimates only; no target dataset arrays or feature tables were read.",
        "All four main tags including raw, both log_scale variants; one primary task per dataset.",
        "pilot_5fold_minutes scales the validated quick budgets and first-seed five folds. "
        "CNN pilot budgets six selection plus six refit epochs; tree early-stop patterns are assumed to match synthetic.",
        "Trees/ROCKET: 15 folds. CNN: 5 folds, 30 selection plus 30 refit epochs as a budget ceiling. "
        "15-fold CNN costs three times the shown amount; early stopping can reduce it substantially.",
        "Trees extrapolate to 500 boosting rounds / 300 ExtraTrees trees, including inner selection/refit, "
        "with linear N, channel/slot and booster class-count scaling. Synthetic observed rounds are the reference.",
        f"CPU transforms/trees use {cores} cores at {efficiency:.0%} relative scaling efficiency; "
        "serial logistic-head updates get no core-count speedup. MPS gets no assumed speedup from more CPU cores.",
        "ROCKET uses 10080 features, 256 bias-fit samples and 15 SGD epochs; transform scales with N, L "
        "and sqrt(channel count), head with N and class count. Feature/epoch scaling uses the saved synthetic-size "
        "head calibration when available, otherwise linear scaling. Shared transforms are counted once.",
        "CNN uses observed median refit epoch time, scales with N*L, and allocates 10% of time to input-channel "
        "scaling. Validation forward cost is approximated as one-third of training forward/backward cost.",
        "Ranges are 0.5x-2x the model estimate, not statistical confidence intervals. Numba compilation, "
        "thermal load, disk bandwidth, class count and different early stopping can change actual times.",
        "Decomposition and feature-cache generation are excluded. Existing decomposition jobs must finish first.",
    ]
    (out / "assumptions.json").write_text(json.dumps({"targets": TARGETS, "cores": cores,
        "cpu_efficiency": efficiency, "source": source, "task": task, "assumptions": assumptions,
        "versions": versions(), "head_calibration": calibration}, indent=2) + "\n")
    (out / "summary.md").write_text("# ML cost estimates\n\n" + totals.to_markdown(index=False, floatfmt=".1f") +
        "\n\n" + "\n\n".join(assumptions) + "\n")
    print(totals.to_string(index=False))
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--task", default="signal")
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--cpu-efficiency", type=float, default=0.5)
    parser.add_argument("--calibrate-head", action="store_true", help="small synthetic-size CPU benchmark before estimating")
    args = parser.parse_args(argv)
    if args.cores < 1 or not 0 < args.cpu_efficiency <= 1:
        parser.error("cores must be positive and efficiency in (0,1]")
    if args.calibrate_head:
        calibrate_head(args.source, args.task, output_dir(args.source, "ml", "costs"))
    estimate(args.source, args.task, args.cores, args.cpu_efficiency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
