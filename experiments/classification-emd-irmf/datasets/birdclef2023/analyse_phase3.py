#!/usr/bin/env python3
"""Audit BirdCLEF phase-3 outputs and aggregate native frame probabilities."""

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd

from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import load_meta
from emdirmf.paths import decomp_dir, features_dir, output_dir, results_dir
from emdirmf.stats import compare_methods

OUT = output_dir("birdclef2023", "ml")
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
NAMES = {"raw": "Raw", "emd": "EMD", "imf_linear_a1.41": "Linear IMF",
         "irmf_h2_a1.41": "IRMF", "irmf_h2_a1.41_conv": "Converged IRMF"}
MODELS = {"lightgbm": "LGBM", "xgboost": "XGB", "extratrees": "ExtraTrees",
          "minirocket": "MiniRocket", "multirocket": "MultiRocket", "compact_resnet": "CNN"}


def comparisons(folds):
    identifiers = [c for c in folds if c not in METRICS]
    long = folds.melt(id_vars=identifiers, value_vars=METRICS,
                      var_name="metric", value_name="value")
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
    # One Holm family per task/metric, covering all models, log variants and pairs.
    return pd.concat([
        compare_methods(long, metric=metric, pairs=pairs,
                        by=("classifier", "feature_set"))
        for metric in METRICS
    ], ignore_index=True)


def analyse(task):
    meta = load_meta("birdclef2023")
    y = meta[f"y_{task}"].astype(str).to_numpy()
    labels = np.unique(y)
    splits = {s.split_id: s for s in load_splits("birdclef2023", task)}
    shared_scale = None
    for method in METHODS:
        values = pd.read_parquet(
            features_dir("birdclef2023") / f"{method}.parquet",
            columns=["ch0_log_scale"]).to_numpy()
        if shared_scale is None:
            shared_scale = values
        else:
            np.testing.assert_array_equal(shared_scale, values)
    all_folds, all_recordings, recording_predictions, costs, audit = [], [], [], [], []
    importance_tables = []
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir() or directory.name == "pilot":
            continue
        path = directory / f"results_folds_{task}.csv"
        context_path = directory / f"run_{task}.json"
        if not path.exists() or not context_path.exists():
            continue
        try:
            context = json.loads(context_path.read_text())
        except json.JSONDecodeError:
            # Another runner can still be writing its completion metadata.
            continue
        folds = pd.read_csv(path)
        if context.get("reference_only"):
            continue
        runner = context["runner"]
        predictions = pd.read_parquet(results_dir("birdclef2023") / f"{runner}_{task}_predictions.parquet")
        probability_columns = [f"prob_{label}" for label in labels]
        assert all(c in predictions for c in probability_columns)
        keys = ["method", "classifier", "feature_set", "split_id"]
        assert not folds.duplicated(keys).any()
        assert not predictions.duplicated(keys + ["frame_id"]).any()
        expected_methods = set(context["methods"])
        assert set(folds.method) == expected_methods == set(METHODS)
        for _, cell in folds.groupby(["method", "classifier", "feature_set"]):
            assert set(cell.split_id) == set(context["split_ids"])
        recordings = []
        for key, cell in predictions.groupby(keys, sort=False):
            method, classifier, feature_set, split_id = key
            split = splits[split_id]
            cell = cell.sort_values("frame_id")
            np.testing.assert_array_equal(cell.frame_id, np.sort(split.test))
            np.testing.assert_array_equal(cell.y_true, y[cell.frame_id])
            assert not set(meta.iloc[split.train].group) & set(meta.iloc[split.test].group)
            probabilities = cell[probability_columns].to_numpy()
            assert np.isfinite(probabilities).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-5)
            np.testing.assert_array_equal(labels[probabilities.argmax(axis=1)], cell.y_pred)
            selected = folds
            for column, value in zip(keys, key):
                selected = selected[selected[column] == value]
            assert len(selected) == 1
            assert selected.iloc[0].n_train == len(split.train)
            assert selected.iloc[0].n_test == len(split.test)
            expected = split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels)
            for metric in METRICS:
                np.testing.assert_allclose(selected.iloc[0][metric], expected[metric], atol=1e-12)
            cell = cell.copy()
            cell["recording_id"] = meta.iloc[cell.frame_id.to_numpy()].recording_id.to_numpy()
            assert cell.groupby("recording_id").y_true.nunique().max() == 1
            averaged = cell.groupby("recording_id")[probability_columns].mean()
            averaged["y_true"] = cell.groupby("recording_id").y_true.first()
            averaged["y_pred"] = labels[averaged[probability_columns].to_numpy().argmax(axis=1)]
            row = dict(zip(keys, key))
            row.update(seed=split.seed, fold=split.fold,
                       n_train=meta.iloc[split.train].recording_id.nunique(), n_test=len(averaged))
            row.update(split_metrics(averaged.y_true.to_numpy(), averaged.y_pred.to_numpy(), labels))
            all_recordings.append(row)
            for column, value in row.items():
                if column in keys or column in ("seed", "fold"):
                    averaged[column] = value
            recordings.append(averaged.reset_index())
        rec = pd.concat(recordings, ignore_index=True)
        assert not rec.duplicated(["method", "classifier", "feature_set", "seed", "recording_id"]).any()
        recording_predictions.append(rec)
        rec.to_parquet(results_dir("birdclef2023") / f"{runner}_{task}_recording_predictions.parquet", index=False)
        all_folds.append(folds)
        costs.append({"runner": runner, "wall_seconds": context["seconds"],
                      "workers": context["workers"], "n_fits": len(folds),
                      "split_ids": json.dumps(context["split_ids"]), "device": context["device"]})
        audit.append({"runner": runner, "n_fit_rows": len(folds), "n_predictions": len(predictions),
                      "n_recording_predictions": len(rec), "n_methods": len(expected_methods),
                      "split_ids": context["split_ids"], "checks": "passed"})
        importance_path = directory / f"importance_folds_{task}.csv"
        if importance_path.exists():
            importance_tables.append(pd.read_csv(importance_path))
        del predictions
    folds = pd.concat(all_folds, ignore_index=True)
    assert not folds.duplicated(["method", "classifier", "feature_set", "split_id"]).any()
    for _, cell in folds.groupby("classifier"):
        memberships = cell.groupby(["method", "feature_set"]).split_id.agg(lambda x: tuple(sorted(x)))
        assert len(memberships) == 10 and len(set(memberships)) == 1
    summary = folds.groupby(["method", "classifier", "feature_set"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / f"summary_metrics_{task}.csv", index=False)
    models = [model for model in MODELS if model in set(folds.classifier)]
    key_table = []
    for method in METHODS:
        row = {"Method": NAMES[method]}
        for model in models:
            row[MODELS[model]] = "/".join(f"{100 * summary.loc[(method, model, side), 'balanced_accuracy_mean']:.1f}"
                                         for side in ("comp", "comp+log_scale"))
        key_table.append(row)
    (OUT / f"key_table_{task}.md").write_text(pd.DataFrame(key_table).to_markdown(index=False) + "\n")
    (OUT / f"full_metrics_{task}.md").write_text(
        "Mean and SD over the matched outer folds. All values are proportions.\n\n" +
        summary.to_markdown(floatfmt=".4f") + "\n")
    folds.to_csv(OUT / f"all_folds_{task}.csv", index=False)
    paired = comparisons(folds)
    paired.to_csv(OUT / f"paired_all_metrics_{task}.csv", index=False)
    key_pairs = []
    requested = [("irmf_h2_a1.41", "emd"), ("irmf_h2_a1.41", "imf_linear_a1.41"),
                 ("irmf_h2_a1.41", "raw"), ("imf_linear_a1.41", "raw"), ("emd", "raw")]
    short = {"irmf_h2_a1.41": "I", "emd": "E", "imf_linear_a1.41": "L", "raw": "R"}
    for model in models:
        for a, b in requested:
            row = {"Model/pair": MODELS[model] + " " + short[a] + "−" + short[b]}
            for side, label in (("comp", "−log"), ("comp+log_scale", "+log")):
                cell = paired[(paired.classifier == model) & (paired.metric == "balanced_accuracy") &
                              (paired.feature_set == side) &
                              (((paired.method_a == a) & (paired.method_b == b)) |
                               ((paired.method_a == b) & (paired.method_b == a)))]
                if cell.empty:
                    row[label] = "n/a"
                    continue
                result = cell.iloc[0]
                sign = 1 if result.method_a == a else -1
                delta = sign * result.mean_diff
                low, high = (result.ci_low, result.ci_high) if sign == 1 else (-result.ci_high, -result.ci_low)
                row[label] = f"{100*delta:+.2f}[{100*low:.2f},{100*high:.2f}];{result.p_nb_holm:.2g}"
            key_pairs.append(row)
    (OUT / f"key_pairs_{task}.md").write_text(pd.DataFrame(key_pairs).to_markdown(index=False) + "\n")
    (OUT / f"paired_all_metrics_{task}.md").write_text(
        "Nadeau-Bengio corrected resampled t-test and corrected 95% CI; Wilcoxon secondary. "
        "Holm covers every model, log variant and method pair separately per task/metric.\n\n" +
        paired[["metric", "classifier", "feature_set", "method_a", "method_b", "n_splits", "mean_diff",
                "ci_low", "ci_high", "p_nb", "p_nb_holm", "p_wilcoxon", "p_wilcoxon_holm"]].to_markdown(
                    index=False, floatfmt=".5g") + "\n")
    recording_folds = pd.DataFrame(all_recordings)
    recording_folds.to_csv(OUT / f"recording_folds_{task}.csv", index=False)
    comparisons(recording_folds).to_csv(OUT / f"recording_comparisons_{task}.csv", index=False)
    # Pool after combining shards: a seed can span multiple runner exports.
    all_recording_predictions = pd.concat(recording_predictions, ignore_index=True)
    record_keys = ["method", "classifier", "feature_set", "seed", "recording_id"]
    assert not all_recording_predictions.duplicated(record_keys).any()
    pooled_recordings = []
    for key, cell in all_recording_predictions.groupby(record_keys[:-1]):
        row = dict(zip(record_keys[:-1], key))
        complete = cell.fold.nunique() == 5
        if complete:
            assert len(cell) == meta.recording_id.nunique()
        row.update(n_recordings=len(cell), complete_seed=complete,
                   **split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels))
        pooled_recordings.append(row)
    pd.DataFrame(pooled_recordings).to_csv(OUT / f"recording_pooled_seeds_{task}.csv", index=False)
    all_recording_predictions.to_parquet(
        results_dir("birdclef2023") / f"ml_{task}_recording_predictions.parquet", index=False)
    pd.DataFrame(costs).to_csv(OUT / f"runner_costs_{task}.csv", index=False)
    folds.assign(total_seconds=folds.fit_seconds + folds.predict_seconds).groupby(
        ["method", "classifier", "feature_set"]).total_seconds.agg(["mean", "median", "sum"]).reset_index().to_csv(
            OUT / f"fit_costs_{task}.csv", index=False)
    (OUT / f"audit_{task}.json").write_text(json.dumps(audit, indent=2) + "\n")
    decomposition_costs = []
    for method in METHODS:
        row = {"method": method, "cpu_seconds_per_frame": 0.0, "wall_seconds_per_frame": 0.0}
        if method != "raw":
            info = json.loads((decomp_dir("birdclef2023") / f"{method}.json").read_text())
            row.update(cpu_seconds_per_frame=info["seconds_per_frame"]["cpu_mean"],
                       wall_seconds_per_frame=info["seconds_per_frame"]["wall_mean"],
                       channels_all_converged=info.get("irmf", {}).get("channels_all_converged"),
                       source_solver=info["params"].get("solver", "pimf/PyEMD default"))
        decomposition_costs.append(row)
    pd.DataFrame(decomposition_costs).to_csv(OUT / "decomposition_costs.csv", index=False)
    importance_folds = pd.concat(importance_tables, ignore_index=True)
    importance = importance_folds.groupby([
        "method", "classifier", "feature_set", "storage_slot", "scale_rank", "slot_label", "importance_type"
    ]).importance.agg(["mean", "std"]).reset_index()
    importance.to_csv(OUT / f"scale_importance_{task}.csv", index=False)
    importance.sort_values("mean", ascending=False).groupby(
        ["method", "classifier", "feature_set"], sort=False).head(5).to_csv(
            OUT / f"top_scale_importance_{task}.csv", index=False)
    print(summary.to_string(), flush=True)
    print("All metrics, fold membership, group exclusion and probabilities audited.", flush=True)


if __name__ == "__main__":
    analyse(sys.argv[1] if len(sys.argv) > 1 else "species")
