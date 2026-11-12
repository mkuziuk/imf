#!/usr/bin/env python3
"""Audit saved UAS/birds radar ML outputs and make paired, clip and cost tables."""

import itertools
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

STUDY = Path(__file__).resolve().parents[2]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from emdirmf.cv import load_splits
from emdirmf.io import load_meta
from emdirmf.paths import data_root
from emdirmf.stats import compare_methods
from scipy import stats
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import ndtr, gammaln
from functools import lru_cache

MAIN_METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
METHODS = MAIN_METHODS + ["irmf_h2_a2_conv", "imf_linear_a2", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv"]
TASKS = ["bird_drone", "target3"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
OUT = ROOT / "output/classification-emd-irmf/uas_birds_60ghz/ml"
DATA = data_root() / "uas_birds_60ghz"


def audit_predictions(folds, predictions, meta, task, runner, *, n_splits=None, methods=None):
    keys = ["method", "feature_set", "classifier", "split_id"]
    assert not folds.duplicated(keys).any()
    assert not predictions.duplicated(keys + ["frame_id"]).any()
    split_map = {split.split_id: split for split in load_splits("uas_birds_60ghz", task)}
    classes = sorted(meta[f"y_{task}"].unique())
    probability_columns = [f"prob_{label}" for label in classes]
    ties = 0
    for key, group in predictions.groupby(keys):
        method, feature_set, classifier, split_id = key
        group = group.sort_values("frame_id")
        split = split_map[split_id]
        np.testing.assert_array_equal(group.frame_id, split.test)
        np.testing.assert_array_equal(group.y_true, meta.iloc[split.test][f"y_{task}"])
        assert not set(meta.iloc[split.train].group) & set(meta.iloc[split.test].group)
        p = group[probability_columns].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-5)
        indices = np.searchsorted(classes, group.y_pred.to_numpy())
        np.testing.assert_allclose(p[np.arange(len(p)), indices], p.max(axis=1), atol=2e-6)
        ties += int((np.asarray(classes)[p.argmax(axis=1)] != group.y_pred.to_numpy()).sum())
        match = folds
        for column, value in zip(keys, key):
            match = match[match[column] == value]
        assert len(match) == 1
        assert match.n_train.iloc[0] == len(split.train)
        assert match.n_test.iloc[0] == len(split.test)
        assert match.seed.iloc[0] == split.seed and match.fold.iloc[0] == split.fold
        observed = [balanced_accuracy_score(group.y_true, group.y_pred),
                    f1_score(group.y_true, group.y_pred, average="macro", zero_division=0),
                    accuracy_score(group.y_true, group.y_pred)]
        np.testing.assert_allclose(match[METRICS].iloc[0].to_numpy(dtype=float), observed, atol=1e-12)
        for label in classes:
            mask = group.y_true.to_numpy() == label
            recall = np.mean(group.y_pred.to_numpy()[mask] == label) if mask.any() else np.nan
            np.testing.assert_allclose(match[f"recall_{label}"].iloc[0], recall, atol=1e-12)
    expected = set(range(n_splits if n_splits is not None else 15))
    expected_methods = METHODS if methods is None else methods
    expected_classifiers = {"trees": ["lightgbm", "xgboost", "extratrees"],
        "rocket": ["minirocket", "multirocket"], "cnn": ["compact_resnet"],
        "multirocket": ["multirocket"], "spectrogram": ["log_stft_cnn2d"]}.get(runner, ["lightgbm"])
    observed_cells = set(folds[["method", "feature_set", "classifier"]].itertuples(index=False, name=None))
    assert observed_cells == set(itertools.product(expected_methods, ["comp", "comp+log_scale"],
                                                    expected_classifiers))
    assert len(folds) == len(expected_methods) * 2 * len(expected_classifiers) * len(expected)
    for _, group in folds.groupby(["method", "feature_set", "classifier"]):
        assert set(group.split_id) == expected
    assert set(folds.method) == set(expected_methods)
    return {"runner": runner, "task": task, "fits": len(folds),
            "prediction_rows": len(predictions), "probability_tied_argmax_labels": ties,
            "split_ids": sorted(expected)}


CELL = ["task", "method", "feature_set", "classifier"]
SENSITIVITY_PAIRS = [
    ("irmf_h2_a2_conv", "emd"), ("imf_linear_a2", "emd"),
    ("irmf_h2_a2_conv", "imf_linear_a2"),
    ("irmf_h2_a2_conv", "irmf_h2_a1.41_conv"),
    ("imf_linear_a2", "imf_linear_a1.41"),
    ("irmf_h1_a1.41_conv", "irmf_h2_a1.41_conv"),
    ("irmf_h4_a1.41_conv", "irmf_h2_a1.41_conv"),
]


def metrics(target, predicted):
    return dict(balanced_accuracy=balanced_accuracy_score(target, predicted),
                macro_f1=f1_score(target, predicted, average="macro", zero_division=0),
                accuracy=accuracy_score(target, predicted))


def summarise(table, keys):
    summary = table.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    return summary


def aggregate_recordings(predictions, meta, task):
    """Aggregate each of 49 held-out recordings within its unchanged pose split."""
    values = predictions.merge(meta[["frame_id", "group_recording", "group"]],
                               on="frame_id", validate="many_to_one")
    values["wrong"] = values.y_true != values.y_pred
    keys = ["method", "feature_set", "classifier", "split_id", "seed", "fold"]
    probabilities = sorted(c for c in values if c.startswith("prob_"))
    classes = np.asarray([c[5:] for c in probabilities])
    records = []
    for key, part in values.groupby(keys + ["group_recording", "group"], sort=False):
        assert part.y_true.nunique() == 1 and len(part) == 16
        means = part[probabilities].mean().to_numpy()
        votes = part.y_pred.value_counts().reindex(classes, fill_value=0).to_numpy()
        row = dict(zip(keys + ["group_recording", "group"], key))
        row.update(task=task, y_true=part.y_true.iloc[0],
                   y_prob=classes[means.argmax()], y_vote=classes[votes.argmax()],
                   probability_tie=int((means == means.max()).sum() > 1),
                   vote_tie=int((votes == votes.max()).sum() > 1),
                   frame_errors=int(part.wrong.sum()), n_frames=len(part))
        records.append(row)
    recording = pd.DataFrame(records)
    for _, cell in recording.groupby(["method", "feature_set", "classifier", "seed"]):
        assert len(cell) == 49 and cell.group_recording.nunique() == 49
    return values, recording


def infer(folds, level):
    tables = []
    for task, part in folds.groupby("task"):
        for family, pairs in [("primary", list(itertools.combinations(MAIN_METHODS, 2))),
                              ("sensitivity", SENSITIVITY_PAIRS)]:
            long = part.melt(id_vars=["method", "feature_set", "classifier", "split_id",
                                      "n_train_groups", "n_test_groups"],
                             value_vars=METRICS, var_name="metric", value_name="value")
            # Use held-out pose units for the correction, as in the phase-2 report.
            long = long.rename(columns={"n_train_groups": "n_train", "n_test_groups": "n_test"})
            for metric in METRICS:
                table = compare_methods(long, metric=metric, pairs=pairs,
                                        by=("feature_set", "classifier"))
                table.insert(0, "task", task)
                table["level"] = level
                table["inference_table"] = family
                table["holm_family_size"] = len(table)
                assert len(table) == len(pairs) * 2 * 6
                tables.append(table)
    return pd.concat(tables, ignore_index=True)


def orient(table, numerator, denominator):
    direct = table[(table.method_a == numerator) & (table.method_b == denominator)].copy()
    reverse = table[(table.method_b == numerator) & (table.method_a == denominator)].copy()
    reverse["method_a"], reverse["method_b"] = numerator, denominator
    for column in ("mean_diff", "t"):
        reverse[column] *= -1
    low, high = reverse.ci_low.copy(), reverse.ci_high.copy()
    reverse["ci_low"], reverse["ci_high"] = -high, -low
    for a, b in [("mean_a", "mean_b"), ("wins_a", "wins_b")]:
        reverse[a], reverse[b] = reverse[b].copy(), reverse[a].copy()
    return pd.concat([direct, reverse], ignore_index=True)


@lru_cache(maxsize=None)
def power_noncentrality(df, alpha):
    """Integrate conditional normal tails to avoid tiny-tail nct CDF NaNs."""
    critical = stats.t.ppf(1 - alpha / 2, df)
    log_normalizer = df / 2 * np.log(2) + gammaln(df / 2)

    def detection(noncentrality):
        def integrand(v):
            scale = critical * np.sqrt(v / df)
            density = np.exp((df / 2 - 1) * np.log(v) - v / 2 - log_normalizer)
            return (ndtr(noncentrality - scale) + ndtr(-noncentrality - scale)) * density
        return quad(integrand, 0, np.inf, epsabs=1e-10)[0]

    np.testing.assert_allclose(detection(0), alpha, atol=1e-9)
    ncp = brentq(lambda x: detection(x) - .8, 0, 100)
    np.testing.assert_allclose(detection(ncp), .8, atol=1e-9)
    return ncp


def power_table(paired):
    """Plug-in, two-sided noncentral-t planning sensitivity, not post-hoc power."""
    rows = []
    for item in paired[paired.metric == "balanced_accuracy"].itertuples():
        se = item.se_corrected
        row = item._asdict()
        row["positive_headroom"] = 1 - item.mean_b
        for name, alpha in [("nominal", .05), ("holm_first_step", .05 / item.holm_family_size)]:
            critical = stats.t.ppf(1 - alpha / 2, item.df)
            row[f"threshold_{name}"] = critical * se
            if se == 0:
                # A zero fold variance is not evidence that arbitrary effects are detectable.
                row[f"mde80_{name}"] = np.nan
            else:
                row[f"mde80_{name}"] = power_noncentrality(item.df, alpha) * se
        row["zero_variance"] = se == 0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    threadpool_limits(1)
    meta = load_meta("uas_birds_60ghz")
    assert len(meta) == 784 and meta.group.nunique() == 45 and meta.group_recording.nunique() == 49
    scale_columns = ["frame_id", "ch0_log_scale", "ch1_log_scale"]
    raw = pd.read_parquet(DATA / "features/raw.parquet", columns=scale_columns)
    for method in METHODS:
        table = pd.read_parquet(DATA / "features" / f"{method}.parquet",
                                columns=scale_columns + ["decomp_ok"])
        np.testing.assert_array_equal(table[scale_columns], raw)
        assert table.decomp_ok.all()
    split_counts = {}
    split_audit = []
    all_folds, recordings, frame_errors, checks, costs, importance = [], [], [], [], [], []
    for task in TASKS:
        split_map = {s.split_id: s for s in load_splits("uas_birds_60ghz", task)}
        for split_id, split in split_map.items():
            train_groups = set(meta.iloc[split.train].group)
            test_groups = set(meta.iloc[split.test].group)
            assert not train_groups & test_groups
            assert not set(meta.iloc[split.train].group_recording) & set(meta.iloc[split.test].group_recording)
            split_counts[task, split_id] = len(train_groups), len(test_groups)
            target = f"y_{task}"
            assert set(meta.iloc[split.train][target]) == set(meta[target])
            assert set(meta.iloc[split.test][target]) == set(meta[target])
            split_audit.append({"task": task, "split_id": split_id, "seed": split.seed,
                                "n_train_frames": len(split.train), "n_test_frames": len(split.test),
                                "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
                                "n_train_recordings": meta.iloc[split.train].group_recording.nunique(),
                                "n_test_recordings": meta.iloc[split.test].group_recording.nunique(),
                                "train_class_counts": json.dumps(meta.iloc[split.train][target].value_counts().to_dict()),
                                "test_class_counts": json.dumps(meta.iloc[split.test][target].value_counts().to_dict()),
                                "group_overlap": 0, "recording_overlap": 0})
        for runner in ("trees", "rocket", "cnn"):
            directory = OUT / runner
            folds = pd.read_csv(directory / f"results_folds_{task}.csv")
            folds["runner"] = runner
            for name, side in (("n_train_groups", "train"), ("n_test_groups", "test")):
                folds[name] = folds.split_id.map(lambda i: meta.iloc[getattr(split_map[i], side)].group.nunique())
            assert (folds.n_train_groups + folds.n_test_groups == 45).all()
            run = json.loads((directory / f"run_{task}.json").read_text())
            assert run["workers"] == 1 and run["split_ids"] == list(range(15))
            if runner == "trees":
                for item in folds.itertuples():
                    slots = 1 if item.method == "raw" else (9 if item.method in {"emd", "irmf_h2_a2_conv", "imf_linear_a2"} else 16)
                    assert item.n_input_features == 40 * slots + (2 if item.feature_set == "comp+log_scale" else 0)
                for item in folds.best_params.map(json.loads):
                    assert 0 < item["n_estimators"] <= 500
                imp = pd.read_csv(directory / f"importance_scales_{task}.csv")
                imp["task"] = task
                importance.append(imp)
            elif runner == "rocket":
                assert set(folds.loc[folds.classifier == "minirocket", "n_features"]) == {1008}
                assert set(folds.loc[folds.classifier == "multirocket", "n_features"]) == {672}
            else:
                assert folds.best_epoch.between(1, 20).all()
                matched = folds[folds.method.isin(["emd", "irmf_h2_a2_conv", "imf_linear_a2"])]
                assert (matched.groupby("feature_set").n_parameters.nunique() == 1).all()
            predictions = pd.read_parquet(run["predictions"])
            checks.append(audit_predictions(folds, predictions, meta, task, runner))
            frames, records = aggregate_recordings(predictions, meta, task)
            frames["task"] = task
            frame_errors.append(frames[CELL + ["frame_id", "split_id", "seed", "y_true", "y_pred",
                                              "group_recording", "group", "wrong"]])
            recordings.append(records)
            all_folds.append(folds)
            costs.append({"task": task, "runner": runner, "wall_seconds": run["seconds"],
                          "n_fits": len(folds), "workers": run["workers"],
                          "model_fit_seconds": float(folds.fit_seconds.sum()),
                          "predict_seconds": float(folds.predict_seconds.sum()),
                          "timings_share_transforms": run.get("timings_share_transforms", False)})
    pd.DataFrame(split_audit).to_csv(OUT / "split_audit.csv", index=False)
    folds = pd.concat(all_folds, ignore_index=True)
    records = pd.concat(recordings, ignore_index=True)
    frames = pd.concat(frame_errors, ignore_index=True)
    assert len(folds) == 3240 and len(frames) == 508032
    assert (frames.groupby(CELL + ["frame_id"]).size() == 3).all()
    assert not frames.duplicated(CELL + ["frame_id", "seed"]).any()
    folds.to_csv(OUT / "all_folds.csv", index=False)
    scores = summarise(folds, CELL)
    scores.to_csv(OUT / "scores.csv", index=False)
    paired = infer(folds, "frame")
    paired.to_csv(OUT / "paired_all_metrics.csv", index=False)
    requested_pairs = [("irmf_h2_a1.41", "emd"), ("irmf_h2_a1.41", "imf_linear_a1.41"),
                       ("irmf_h2_a1.41", "raw"), ("irmf_h2_a1.41_conv", "emd"),
                       ("irmf_h2_a1.41_conv", "irmf_h2_a1.41"),
                       ("imf_linear_a1.41", "emd")] + SENSITIVITY_PAIRS
    requested = pd.concat([orient(paired, *pair) for pair in requested_pairs], ignore_index=True)
    requested.to_csv(OUT / "paired_requested.csv", index=False)
    power = power_table(requested)
    power.to_csv(OUT / "power.csv", index=False)
    rec_folds, rec_seeds, rec_error_rows, confusion = [], [], [], []
    for aggregation, column in (("probability_mean", "y_prob"), ("majority_vote", "y_vote")):
        records[f"wrong_{aggregation}"] = records.y_true != records[column]
        for key, part in records.groupby(CELL + ["split_id", "seed", "fold"]):
            row = dict(zip(CELL + ["split_id", "seed", "fold"], key))
            n_train, n_test = split_counts[row["task"], row["split_id"]]
            row.update(aggregation=aggregation, n_train_groups=n_train, n_test_groups=n_test,
                       n_recordings=len(part), **metrics(part.y_true, part[column]))
            rec_folds.append(row)
        for key, part in records.groupby(CELL + ["seed"]):
            row = dict(zip(CELL + ["seed"], key))
            row.update(aggregation=aggregation, n_recordings=len(part),
                       **metrics(part.y_true, part[column]))
            rec_seeds.append(row)
        for key, part in records.groupby(CELL + ["group_recording", "group", "y_true"]):
            row = dict(zip(CELL + ["group_recording", "group", "class"], key))
            row.update(aggregation=aggregation, errors=int((part.y_true != part[column]).sum()),
                       n_evaluations=len(part), any_error=bool((part.y_true != part[column]).any()))
            rec_error_rows.append(row)
        part = records.rename(columns={column: "predicted", "y_true": "class"})
        tmp = part.groupby(CELL + ["class", "predicted"]).size().rename("n").reset_index()
        tmp["level"] = aggregation
        confusion.append(tmp)
    rec_folds = pd.DataFrame(rec_folds)
    rec_seeds = pd.DataFrame(rec_seeds)
    rec_folds.to_csv(OUT / "recording_folds.csv", index=False)
    rec_seeds.to_csv(OUT / "recording_metrics_by_seed.csv", index=False)
    summarise(rec_seeds, CELL + ["aggregation"]).to_csv(OUT / "recording_scores.csv", index=False)
    rec_errors = pd.DataFrame(rec_error_rows)
    rec_errors.to_csv(OUT / "recording_errors.csv", index=False)
    rec_pairs = pd.concat([infer(p, level) for level, p in rec_folds.groupby("aggregation")], ignore_index=True)
    rec_pairs.to_csv(OUT / "paired_recording.csv", index=False)
    records.to_parquet(DATA / "results/uas_phase3_recording_predictions.parquet", index=False)
    per_recording = frames.groupby(CELL + ["group_recording", "group", "y_true"]).agg(
        n_frame_errors=("wrong", "sum"), n_frame_evaluations=("wrong", "size"),
        n_distinct_frames_wrong=("frame_id", lambda x: x[frames.loc[x.index, "wrong"]].nunique())).reset_index()
    per_recording.to_csv(OUT / "frame_errors_by_recording.csv", index=False)
    group_errors = frames.groupby(CELL + ["group"]).agg(
        n_frame_errors=("wrong", "sum"), n_frame_evaluations=("wrong", "size")).reset_index()
    group_errors.to_csv(OUT / "group_errors.csv", index=False)
    error_counts = []
    for key, part in frames.groupby(CELL):
        wrong = part[part.wrong]
        row = dict(zip(CELL, key))
        row.update(frame_errors=len(wrong), frame_evaluations=len(part),
                   distinct_frames_ever_wrong=wrong.frame_id.nunique(),
                   recordings_ever_frame_wrong=wrong.group_recording.nunique(),
                   groups_ever_frame_wrong=wrong.group.nunique())
        selected = records
        for column, value in zip(CELL, key):
            selected = selected[selected[column] == value]
        for aggregation in ("probability_mean", "majority_vote"):
            bad = selected[selected[f"wrong_{aggregation}"]]
            row.update({f"{aggregation}_errors": len(bad),
                        f"recordings_ever_{aggregation}_wrong": bad.group_recording.nunique(),
                        f"groups_ever_{aggregation}_wrong": bad.group.nunique()})
        error_counts.append(row)
    pd.DataFrame(error_counts).to_csv(OUT / "error_counts.csv", index=False)
    tmp = frames.rename(columns={"y_true": "class", "y_pred": "predicted"}).groupby(
        CELL + ["class", "predicted"]).size().rename("n").reset_index()
    tmp["level"] = "frame"
    confusion.append(tmp)
    confusion = pd.concat(confusion, ignore_index=True)
    confusion.to_csv(OUT / "confusion.csv", index=False)
    confusion["errors"] = np.where(confusion["class"] != confusion.predicted, confusion.n, 0)
    class_errors = confusion.groupby(CELL + ["level", "class"]).agg(
        errors=("errors", "sum"), n_evaluations=("n", "sum")).reset_index()
    class_errors["recall"] = 1 - class_errors.errors / class_errors.n_evaluations
    class_errors.to_csv(OUT / "per_class_errors.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "runner_costs.csv", index=False)
    importance = pd.concat(importance, ignore_index=True)
    importance.to_csv(OUT / "importance_by_scale.csv", index=False)
    decomposition = []
    for method in METHODS:
        if method == "raw":
            decomposition.append({"method": method, "cpu_seconds_per_frame": 0, "n_slots": 1})
            continue
        info = json.loads((DATA / "decomp" / f"{method}.json").read_text())
        if method.endswith("_conv"):
            assert info["irmf"]["channels_all_converged"] == 1568
        timing = info["seconds_per_frame"]
        decomposition.append({"method": method, "cpu_seconds_per_frame": timing.get("cpu_mean"),
                              "wall_seconds_per_frame": timing.get("wall_mean"),
                              "n_slots": info["shape"][2], "n_failed": info.get("n_failed")})
    pd.DataFrame(decomposition).to_csv(OUT / "decomposition_costs.csv", index=False)
    (OUT / "validation.json").write_text(json.dumps({"checks": checks,
        "fits": len(folds), "prediction_rows": len(frames), "all_checks_passed": True,
        "shared_log_scales_identical_both_channels": True,
        "recording_probability_ties": int(records.probability_tie.sum()),
        "recording_majority_ties": int(records.vote_tie.sum()),
        "correction_units": "observed pose groups per split: 35-37 train and 8-10 test",
        "holm": "120 primary or 84 sensitivity tests per task, metric and level",
        "recording": "49 recordings per seed; mean probabilities and majority; alphabetical tie breaks",
        "power": "noncentral t, 80% power, plug-in paired variance, nominal and conservative Holm first step; zero variance undefined"}, indent=2) + "\n")
    write_tables(scores, requested, power, rec_seeds, importance)
    print(f"Audited {len(folds)} fits and {len(frames)} predictions.")


def write_tables(scores, requested, power, recording, importance):
    tables = ["# Phase-3 balanced accuracy", "", "Outer-fold means, percent. All 15 folds for every cell.", ""]
    detail = ["# All frame metrics", "", "Balanced accuracy / macro-F1 / accuracy, percent.", ""]
    effects = ["# Paired effects", "", "A minus B, percentage points, individual corrected 95% intervals.",
               "Holm covers 120 primary or 84 sensitivity tests per task/metric. Correction uses the mean observed pose-group test/train ratio.", ""]
    for task in TASKS:
        for feature_set in ("comp", "comp+log_scale"):
            cell = scores[(scores.task == task) & (scores.feature_set == feature_set)].copy()
            title = f"## {task}, {feature_set}"
            tables += [title, "", cell.pivot(index="method", columns="classifier", values="balanced_accuracy_mean").mul(100).to_markdown(floatfmt=".2f"), ""]
            cell["BA/F1/accuracy"] = cell.apply(lambda r: "/".join(f"{100*r[f'{m}_mean']:.2f}" for m in METRICS), axis=1)
            detail += [title, "", cell.pivot(index="method", columns="classifier", values="BA/F1/accuracy").to_markdown(), ""]
            for classifier in sorted(cell.classifier.unique()):
                part = requested[(requested.task == task) & (requested.feature_set == feature_set) &
                                 (requested.classifier == classifier) & (requested.metric == "balanced_accuracy")].copy()
                part["A - B"] = part.method_a + " - " + part.method_b
                part["difference [95% CI]"] = part.apply(lambda r: f"{100*r.mean_diff:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]", axis=1)
                effects += [f"### {task}, {feature_set}, {classifier}", "", part[["A - B", "inference_table", "difference [95% CI]", "p_nb", "p_nb_holm", "p_wilcoxon", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""]
    (OUT / "TABLES.md").write_text("\n".join(tables))
    (OUT / "SCORES_ALL_METRICS.md").write_text("\n".join(detail))
    (OUT / "PAIRS.md").write_text("\n".join(effects))
    lines = ["# Tree importance by scale", "", "Normalized per model, then averaged over folds. Descriptive, not causal. Finest IMF residual has rank zero. Split counts, gain and impurity differ.", ""]
    selected = []
    for key, part in importance.groupby(["task", "classifier", "method", "feature_set", "importance_type"]):
        row = dict(zip(["task", "classifier", "method", "feature_set", "importance_type"], key))
        row["top scales"] = ", ".join(f"{r.slot_label} {100*r.mean:.1f}%" for r in part.nlargest(5, "mean").itertuples())
        selected.append(row)
    lines.append(pd.DataFrame(selected).to_markdown(index=False))
    (OUT / "IMPORTANCE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
