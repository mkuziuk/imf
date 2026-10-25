#!/usr/bin/env python3
"""Audit saved ESC-50 ML outputs and make paired, clip and cost tables."""

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

METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
OUT = ROOT / "output/classification-emd-irmf/esc50_animals/ml"
DATA = data_root() / "esc50_animals"


def audit_predictions(folds, predictions, meta, task, runner, *, n_splits=None, methods=None):
    keys = ["method", "feature_set", "classifier", "split_id"]
    assert not folds.duplicated(keys).any()
    assert not predictions.duplicated(keys + ["frame_id"]).any()
    split_map = {split.split_id: split for split in load_splits("esc50_animals", task)}
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
        observed = [balanced_accuracy_score(group.y_true, group.y_pred),
                    f1_score(group.y_true, group.y_pred, average="macro", zero_division=0),
                    accuracy_score(group.y_true, group.y_pred)]
        np.testing.assert_allclose(match[METRICS].iloc[0].to_numpy(dtype=float), observed, atol=1e-12)
        for label in classes:
            mask = group.y_true.to_numpy() == label
            recall = np.mean(group.y_pred.to_numpy()[mask] == label) if mask.any() else np.nan
            np.testing.assert_allclose(match[f"recall_{label}"].iloc[0], recall, atol=1e-12)
    expected = set(range(n_splits if n_splits is not None else (5 if runner == "cnn" else 15)))
    expected_methods = METHODS if methods is None else methods
    expected_classifiers = {"trees": ["lightgbm", "xgboost", "extratrees"],
        "rocket": ["minirocket"], "cnn": ["compact_resnet"],
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


def clip_scores(predictions, meta, task, runner):
    predictions = predictions.merge(meta[["frame_id", "recording_id", "group"]],
                                    on="frame_id", validate="many_to_one")
    keys = ["method", "feature_set", "classifier", "seed"]
    probabilities = sorted(column for column in predictions if column.startswith("prob_"))
    labels = np.asarray([column[5:] for column in probabilities])
    rows = []
    for key, group in predictions.groupby(keys):
        assert group.groupby("recording_id").split_id.nunique().max() == 1
        assert group.groupby("recording_id").y_true.nunique().max() == 1
        means = group.groupby("recording_id")[probabilities].mean()
        target = group.groupby("recording_id").y_true.first().loc[means.index]
        pred = labels[means.to_numpy().argmax(axis=1)]
        assert len(means) == meta.recording_id.nunique() == 400
        row = dict(zip(keys, key))
        row.update(task=task, runner=runner, n_clips=len(means),
                   balanced_accuracy=balanced_accuracy_score(target, pred),
                   macro_f1=f1_score(target, pred, average="macro", zero_division=0),
                   accuracy=accuracy_score(target, pred))
        rows.append(row)
    return rows


def main():
    threadpool_limits(1)
    meta = load_meta("esc50_animals")
    raw_scales = pd.read_parquet(DATA / "features/raw.parquet", columns=["frame_id", "ch0_log_scale"])
    for method in METHODS:
        values = pd.read_parquet(DATA / "features" / f"{method}.parquet",
                                 columns=["frame_id", "ch0_log_scale", "decomp_ok"])
        np.testing.assert_array_equal(values.frame_id, meta.frame_id)
        np.testing.assert_array_equal(values.ch0_log_scale, raw_scales.ch0_log_scale)
        assert values.decomp_ok.all()
    all_folds, all_pairs, clips, checks, costs, importance = [], [], [], [], [], []
    supplemental_pairs = []
    pairs = list(itertools.combinations(METHODS, 2))
    for task in ("animal", "bird"):
        task_folds = []
        for runner in ("trees", "rocket", "cnn"):
            directory = OUT / runner
            folds = pd.read_csv(directory / f"results_folds_{task}.csv")
            folds["runner"] = runner
            run = json.loads((directory / f"run_{task}.json").read_text())
            if runner == "trees":
                parameters = folds.best_params.map(json.loads)
                for classifier, cap in (("lightgbm", 150), ("xgboost", 60), ("extratrees", 64)):
                    selected = parameters[folds.classifier == classifier]
                    assert all(0 < value["n_estimators"] <= cap for value in selected)
            if runner == "cnn":
                assert folds.best_epoch.between(1, 20).all()
            if runner == "rocket":
                assert set(folds.n_features) == {1008}
            predictions = pd.read_parquet(run["predictions"])
            checks.append(audit_predictions(folds, predictions, meta, task, runner))
            clips.extend(clip_scores(predictions, meta, task, runner))
            task_folds.append(folds)
            costs.append({"task": task, "runner": runner, "wall_seconds": run["seconds"],
                          "n_fits": len(folds), "workers": run["workers"],
                          "model_fit_seconds": float(folds.fit_seconds.sum()),
                          "predict_seconds": float(folds.predict_seconds.sum()),
                          "timings_share_transforms": run.get("timings_share_transforms", False)})
            if runner == "trees":
                imp = pd.read_csv(directory / f"importance_scales_{task}.csv")
                imp["task"] = task
                importance.append(imp)
        combined = pd.concat(task_folds, ignore_index=True)
        all_folds.append(combined)
        long = combined.melt(id_vars=["method", "feature_set", "classifier", "split_id",
                                     "n_train", "n_test"], value_vars=METRICS,
                             var_name="metric", value_name="value")
        for metric in METRICS:
            table = compare_methods(long, metric=metric, pairs=pairs,
                                    by=("feature_set", "classifier"))
            table.insert(0, "task", task)
            table["holm_family_size"] = len(table)
            table["inference_table"] = "primary"
            all_pairs.append(table)
        for runner, expected_methods in (("multirocket", METHODS), ("spectrogram", ["raw"])):
            directory = OUT / runner
            if not (directory / f"run_{task}.json").exists():
                continue
            extra_folds = pd.read_csv(directory / f"results_folds_{task}.csv")
            extra_folds["runner"] = runner
            if runner == "multirocket":
                assert set(extra_folds.n_features) == {672}
            run = json.loads((directory / f"run_{task}.json").read_text())
            predictions = pd.read_parquet(run["predictions"])
            checks.append(audit_predictions(extra_folds, predictions, meta, task, runner,
                                            n_splits=5, methods=expected_methods))
            clips.extend(clip_scores(predictions, meta, task, runner))
            all_folds.append(extra_folds)
            costs.append({"task": task, "runner": runner, "wall_seconds": run["seconds"],
                          "n_fits": len(extra_folds), "workers": run["workers"],
                          "model_fit_seconds": float(extra_folds.fit_seconds.sum()),
                          "predict_seconds": float(extra_folds.predict_seconds.sum()),
                          "timings_share_transforms": run.get("timings_share_transforms", False)})
            if runner == "multirocket":
                extra_long = extra_folds.melt(id_vars=["method", "feature_set", "classifier", "split_id",
                                                       "n_train", "n_test"], value_vars=METRICS,
                                              var_name="metric", value_name="value")
                for metric in METRICS:
                    table = compare_methods(extra_long, metric=metric, pairs=pairs,
                                            by=("feature_set", "classifier"))
                    table.insert(0, "task", task)
                    table["holm_family_size"] = len(table)
                    table["inference_table"] = "supplemental_multirocket"
                    supplemental_pairs.append(table)
    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(OUT / "all_folds.csv", index=False)
    paired = pd.concat(all_pairs, ignore_index=True)
    paired.to_csv(OUT / "paired_all_metrics.csv", index=False)
    for task, table in paired.groupby("task"):
        table.to_csv(OUT / f"paired_{task}.csv", index=False)
    if supplemental_pairs:
        supplemental = pd.concat(supplemental_pairs, ignore_index=True)
        supplemental.to_csv(OUT / "paired_multirocket.csv", index=False)
        paired = pd.concat([paired, supplemental], ignore_index=True)
    # Reverse selected contrasts so positive always means the named numerator wins.
    wanted = [("irmf_h2_a1.41", "emd"), ("irmf_h2_a1.41", "imf_linear_a1.41"),
              ("irmf_h2_a1.41", "raw"), ("imf_linear_a1.41", "raw"), ("emd", "raw"),
              ("irmf_h2_a1.41_conv", "irmf_h2_a1.41"), ("irmf_h2_a1.41_conv", "emd")]
    wanted += [("imf_linear_a1.41", "emd"), ("irmf_h2_a1.41_conv", "imf_linear_a1.41"),
               ("irmf_h2_a1.41_conv", "raw")]
    selected = []
    for numerator, denominator in wanted:
        table = paired[(paired.method_a == numerator) & (paired.method_b == denominator)].copy()
        reverse = paired[(paired.method_b == numerator) & (paired.method_a == denominator)].copy()
        reverse["method_a"], reverse["method_b"] = numerator, denominator
        for column in ("mean_diff", "t"):
            reverse[column] *= -1
        lower, upper = reverse.ci_low.copy(), reverse.ci_high.copy()
        reverse["ci_low"], reverse["ci_high"] = -upper, -lower
        for a, b in (("mean_a", "mean_b"), ("wins_a", "wins_b")):
            first, second = reverse[a].copy(), reverse[b].copy()
            reverse[a], reverse[b] = second, first
        selected.extend([table, reverse])
    requested = pd.concat(selected, ignore_index=True)
    requested.to_csv(OUT / "paired_requested.csv", index=False)
    significance = requested[requested.metric == "balanced_accuracy"].copy()
    significance["favorable_significant"] = (significance.p_nb_holm < .05) & (significance.mean_diff > 0)
    significance["opposite_significant"] = (significance.p_nb_holm < .05) & (significance.mean_diff < 0)
    significance.groupby(["task", "inference_table", "feature_set", "method_a", "method_b"]).agg(
        n_model_cells=("classifier", "size"), favorable_significant=("favorable_significant", "sum"),
        opposite_significant=("opposite_significant", "sum")).reset_index().to_csv(
            OUT / "significance_counts.csv", index=False)
    keys = ["task", "method", "feature_set", "classifier"]
    scores = folds.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    scores.columns = ["_".join(column).rstrip("_") for column in scores.columns]
    scores.to_csv(OUT / "scores.csv", index=False)
    recall_columns = [column for column in folds if column.startswith("recall_")]
    recall = folds.melt(id_vars=keys, value_vars=recall_columns,
                        var_name="class", value_name="recall").dropna(subset=["recall"])
    recall["class"] = recall["class"].str.removeprefix("recall_")
    recall.groupby(keys + ["class"]).recall.agg(["mean", "std"]).reset_index().to_csv(
        OUT / "per_class_recall.csv", index=False)
    clip = pd.DataFrame(clips)
    clip.to_csv(OUT / "clip_metrics_by_seed.csv", index=False)
    clip_summary = clip.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    clip_summary.columns = ["_".join(column).rstrip("_") for column in clip_summary.columns]
    clip_summary = clip_summary.merge(clip.groupby(keys).seed.nunique().rename("n_seeds").reset_index(), on=keys)
    clip_summary.to_csv(OUT / "clip_scores.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "runner_costs.csv", index=False)
    imp = pd.concat(importance, ignore_index=True)
    imp.to_csv(OUT / "importance_by_scale.csv", index=False)
    importance_lines = ["# Tree importance by scale", "",
        "Normalized within each fitted model, then averaged across folds. Each component sums its 20 statistics.",
        "Scale rank 0 is finest; residual_fine is the finest IMF/IRMF content. Importance is descriptive, not causal.",
        "Split count, gain and impurity are different measures and should not be compared between model families.", ""]
    importance_summary = []
    for key, group in imp.groupby(["task", "classifier", "method", "feature_set", "importance_type"]):
        row = dict(zip(["task", "classifier", "method", "feature_set", "importance_type"], key))
        row["top five scales"] = ", ".join(f"{item.slot_label} {100 * item.mean:.1f}%"
                                            for item in group.nlargest(5, "mean").itertuples())
        importance_summary.append(row)
    importance_lines.append(pd.DataFrame(importance_summary).to_markdown(index=False))
    (OUT / "IMPORTANCE.md").write_text("\n".join(importance_lines) + "\n")
    decomposition = []
    for method in METHODS:
        if method == "raw":
            decomposition.append({"method": method, "seconds_per_frame": 0, "n_slots": 1})
            continue
        info = json.loads((DATA / "decomp" / f"{method}.json").read_text())
        if method.endswith("_conv"):
            assert info["irmf"]["channels_all_converged"] == info["shape"][0] * info["shape"][1]
        timing = info.get("seconds_per_frame", {})
        decomposition.append({"method": method, "seconds_per_frame": timing.get("cpu_mean"),
                              "wall_seconds_per_frame": timing.get("wall_mean"),
                              "n_slots": info["shape"][2], "n_failed": info.get("n_failed"),
                              "solver": info.get("params", {}).get("solver", "pimf/PyEMD")})
    pd.DataFrame(decomposition).to_csv(OUT / "decomposition_costs.csv", index=False)
    (OUT / "validation.json").write_text(json.dumps({"checks": checks,
        "holm": "primary: all 10 pairs x 2 feature sets x 5 classifiers = 100 tests per task/metric; MultiRocket separately 20",
        "clip": "mean native probabilities within a held-out clip; pool 400 clips per seed",
        "shared_log_scales_identical": True, "all_checks_passed": True}, indent=2) + "\n")
    lines = ["# Phase-3 score tables", "", "Percentages, outer-fold means. SDs and every metric are in scores.csv.", ""]
    for task in ("animal", "bird"):
        for feature_set in ("comp", "comp+log_scale"):
            lines.extend([f"## {task}, {feature_set}", "", scores[(scores.task == task) &
                (scores.feature_set == feature_set) & (scores.classifier != "log_stft_cnn2d")].pivot(index="method", columns="classifier",
                values="balanced_accuracy_mean").mul(100).to_markdown(floatfmt=".2f"), ""])
            reference = scores[(scores.task == task) & (scores.feature_set == feature_set) &
                               (scores.classifier == "log_stft_cnn2d")]
            if len(reference):
                lines.extend([f"Raw log-STFT 2-D reference: {100 * reference.balanced_accuracy_mean.iloc[0]:.2f}% BA.", ""])
    (OUT / "TABLES.md").write_text("\n".join(lines))
    detailed = ["# All frame metrics", "", "Each cell is balanced accuracy / macro-F1 / accuracy, percent.", ""]
    effects = ["# Requested paired effects", "", "Positive differences favor method A. Percentage points and corrected 95% CI.",
               "Primary Holm covers 100 tests per task/metric, supplemental MultiRocket 20. Wilcoxon is secondary; all metrics are in paired_requested.csv.", ""]
    for task in ("animal", "bird"):
        for feature_set in ("comp", "comp+log_scale"):
            cell = scores[(scores.task == task) & (scores.feature_set == feature_set)].copy()
            cell["BA/F1/accuracy"] = cell.apply(lambda row: "/".join(
                f"{100 * row[f'{metric}_mean']:.2f}" for metric in METRICS), axis=1)
            detailed.extend([f"## {task}, {feature_set}", "", cell.pivot(index="method", columns="classifier",
                             values="BA/F1/accuracy").fillna("-").to_markdown(), ""])
            for classifier in cell.classifier.unique():
                table = requested[(requested.task == task) & (requested.feature_set == feature_set) &
                                  (requested.classifier == classifier) &
                                  (requested.metric == "balanced_accuracy")].copy()
                if not len(table):
                    continue
                table["A - B"] = table.method_a + " - " + table.method_b
                table["difference [95% CI]"] = table.apply(lambda row:
                    f"{100 * row.mean_diff:+.2f} [{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}]", axis=1)
                effects.extend([f"## {task}, {feature_set}, {classifier}", "",
                    table[["A - B", "n_splits", "difference [95% CI]", "p_nb", "p_nb_holm",
                           "p_wilcoxon", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""])
    (OUT / "SCORES_ALL_METRICS.md").write_text("\n".join(detailed))
    (OUT / "PAIRS.md").write_text("\n".join(effects))
    print(scores.to_string(index=False))
    print("All prediction, metric, split and probability checks passed.")


if __name__ == "__main__":
    main()
