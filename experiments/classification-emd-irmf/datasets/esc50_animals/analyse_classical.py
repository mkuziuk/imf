"""Descriptive diagnostics and paired inference for the ESC-50 phase-2 run.

Run after run_phase2.py. Large OOF tables remain in DATA_ROOT; small derived
tables and three publication-style figures go to the requested output folder.
No feature rankings below are used to fit or select model inputs.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

REPO = Path("/Users/mikhail/.t3/worktrees/imf/t3-44ed3594")
STUDY = REPO / "experiments/classification-emd-irmf"
sys.path[:0] = [str(STUDY), str(STUDY / "baselines")]

import run_classical as runner
from emdirmf import decompose
from emdirmf.cv import fit_predict_split, load_splits, split_metrics
from emdirmf.features import feature_columns
from emdirmf.stats import compare_methods, default_pairs, holm, paired_comparison

DATA = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification")) / "esc50_animals"
OUT = REPO / "output/classification-emd-irmf/esc50_animals/classical"
FIGURES = OUT / "figures"
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
NAMES = dict(zip(METHODS, ["Raw", "EMD", "Linear IMF", "IRMF"]))
SETS = ["comp", "comp+log_scale"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def write_csv(table, name):
    table.to_csv(OUT / name, index=False)


def paired_tables(folds):
    id_columns = ["task", "method", "feature_set", "classifier", "split_id", "n_train", "n_test"]
    long = folds.melt(id_vars=id_columns, value_vars=METRICS, var_name="metric", value_name="value")
    results = []
    for task, cell in long.groupby("task", sort=False):
        for metric in METRICS:
            table = compare_methods(cell, metric=metric, pairs=default_pairs(METHODS),
                                    by=["feature_set", "classifier"])
            table.insert(0, "task", task)
            results.append(table)
    table = pd.concat(results, ignore_index=True)
    write_csv(table, "paired_all_metrics.csv")
    write_csv(table[table["metric"] == "balanced_accuracy"], "paired_balanced_accuracy.csv")
    return table


def feature_scales(meta):
    records = []
    for task in ("animal", "bird"):
        disc = pd.read_csv(OUT / f"discriminability_{task}.csv")
        for tag in METHODS:
            features = pd.read_parquet(DATA / "features" / f"{tag}.parquet")
            order = decompose.scale_order(tag, 2048)
            labels = decompose.slot_labels(tag, 2048)
            windows = decompose.slot_window_sizes(tag, 2048)
            for rank, slot in enumerate(order):
                part = disc[(disc["method"] == tag) & (disc["slot"] == slot)]
                present = features[f"ch0_c{slot}_std"].to_numpy() >= 1e-6
                hz = features.loc[present, f"ch0_c{slot}_spec_centroid"].to_numpy() * 22050
                record = {"task": task, "method": tag, "scale_rank": rank, "slot": slot,
                          "slot_label": labels[slot], "n_present": int(present.sum()),
                          "centroid_hz_median": np.median(hz) if len(hz) else np.nan,
                          "centroid_hz_p10": np.quantile(hz, 0.1) if len(hz) else np.nan,
                          "centroid_hz_p90": np.quantile(hz, 0.9) if len(hz) else np.nan,
                          "window_samples": windows[slot] if windows and slot < len(windows) else np.nan,
                          "mean_epsilon2": part["epsilon2"].mean(), "max_epsilon2": part["epsilon2"].max(),
                          "mean_mi_nats": part["mi"].mean(), "max_mi_nats": part["mi"].max(),
                          "best_epsilon2_feature": part.sort_values("epsilon2", ascending=False).iloc[0]["feat"],
                          "best_mi_feature": part.sort_values("mi", ascending=False).iloc[0]["feat"]}
                record["inverse_window_hz"] = 22050 / record["window_samples"]
                records.append(record)
    table = pd.DataFrame(records)
    write_csv(table, "discriminability_by_scale.csv")
    log_scale = pd.concat([pd.read_csv(OUT / f"discriminability_{task}.csv").query("slot == -1").assign(task=task)
                           for task in ("animal", "bird")], ignore_index=True)
    write_csv(log_scale, "log_scale_discriminability.csv")
    return table


def convergence_tables(meta, folds, best):
    diagnostic = pd.read_parquet(DATA / "decomp/irmf_h2_a1.41.meta.parquet").sort_values("frame_id")
    assert diagnostic["frame_id"].tolist() == meta["frame_id"].tolist()
    converged = np.stack(diagnostic["stage_converged"].to_numpy())
    iterations = np.stack(diagnostic["stage_iterations"].to_numpy())
    all_converged = converged.all(axis=1)
    stage_table = pd.DataFrame({"slot": np.arange(converged.shape[1]),
                                "slot_label": decompose.slot_labels("irmf", 2048)[:-1],
                                "fraction_converged": converged.mean(axis=0),
                                "mean_iterations": iterations.mean(axis=0),
                                "fraction_at_iteration_cap": (iterations == 60).mean(axis=0)})
    write_csv(stage_table, "convergence_by_stage.csv")
    by_class = meta.groupby("y_animal").apply(
        lambda rows: pd.Series({"n_frames": len(rows), "fraction_all_converged": all_converged[rows.index].mean()}),
        include_groups=False).reset_index()
    write_csv(by_class, "convergence_by_class.csv")
    rows = []
    class_rows = []
    for task in ("animal", "bird"):
        labels = sorted(meta[f"y_{task}"].unique())
        predictions = pd.read_parquet(DATA / "results/phase2_main" / f"classical_{task}_predictions.parquet")
        predictions = predictions[predictions["method"] == "irmf_h2_a1.41"].copy()
        predictions["converged"] = all_converged[predictions["frame_id"].to_numpy()]
        for keys, part in predictions.groupby(["feature_set", "classifier", "split_id", "converged"], observed=True):
            feature_set, classifier, split_id, status = keys
            row = {"task": task, "feature_set": feature_set, "classifier": classifier,
                   "split_id": split_id, "all_stages_converged": status, "n_frames": len(part),
                   "n_classes_present": part["y_true"].nunique()}
            row.update(split_metrics(part["y_true"].astype(str).to_numpy(), part["y_pred"].astype(str).to_numpy(), labels))
            rows.append(row)
        for keys, part in predictions.groupby(["feature_set", "classifier", "y_true", "converged"], observed=True):
            feature_set, classifier, label, status = keys
            class_rows.append({"task": task, "feature_set": feature_set, "classifier": classifier,
                               "label": label, "all_stages_converged": status, "n_predictions": len(part),
                               "accuracy": (part["y_true"] == part["y_pred"]).mean()})
    subsets = pd.DataFrame(rows)
    write_csv(subsets, "convergence_subset_folds.csv")
    write_csv(pd.DataFrame(class_rows), "convergence_class_accuracy.csv")
    summary = subsets.groupby(["task", "feature_set", "classifier", "all_stages_converged"], observed=True)[METRICS].mean().reset_index()
    write_csv(summary, "convergence_subset_summary.csv")
    paired = []
    for keys, part in subsets.groupby(["task", "feature_set", "classifier"]):
        part = part.pivot(index="split_id", columns="all_stages_converged", values="accuracy").dropna()
        matched = folds[(folds["task"] == keys[0]) & (folds["method"] == "irmf_h2_a1.41")
                        & (folds["feature_set"] == keys[1]) & (folds["classifier"] == keys[2])].set_index("split_id").loc[part.index]
        row = dict(zip(["task", "feature_set", "classifier"], keys))
        row.update(paired_comparison(part[True], part[False], matched["n_train"], matched["n_test"]))
        paired.append(row)
    comparisons = pd.DataFrame(paired)
    comparisons["p_nb_holm"] = holm(comparisons["p_nb"])
    comparisons["p_wilcoxon_holm"] = holm(comparisons["p_wilcoxon"])
    write_csv(comparisons, "convergence_accuracy_comparisons.csv")
    return {"all_converged": int(all_converged.sum()), "n_frames": len(meta),
            "fraction_stages_unconverged": float(1 - converged.mean()),
            "fraction_stages_at_cap": float((iterations == 60).mean()),
            "mean_iterations": float(iterations.mean())}


def uploader_balanced_metrics(meta):
    records = []
    for task in ("animal", "bird"):
        predictions = pd.read_parquet(DATA / "results/phase2_main" / f"classical_{task}_predictions.parquet")
        predictions["group"] = meta.iloc[predictions["frame_id"].to_numpy()]["group"].to_numpy()
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
        keys = ["method", "feature_set", "classifier", "split_id", "seed", "fold"]
        group_recall = predictions.groupby(keys + ["y_true", "group"], observed=True)["correct"].mean()
        class_recall = group_recall.groupby(keys + ["y_true"], observed=True).mean()
        table = class_recall.groupby(keys, observed=True).mean().rename("uploader_balanced_accuracy").reset_index()
        table.insert(0, "task", task)
        records.append(table)
    write_csv(pd.concat(records, ignore_index=True), "uploader_balanced_metrics.csv")


def audit_selected_svm(path_string):
    """Separate a warning in a rejected grid candidate from the chosen fit."""
    import warnings

    path = Path(path_string)
    task = path.parent.parent.name
    tag, feature_set, _, split_id, _ = path.stem.split("__")
    out = path.parent.parent / "svm_refit_audit" / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    table = pd.read_parquet(DATA / "features" / f"{tag}.parquet")
    columns = feature_columns(table.columns, include_log_scale=feature_set == "comp+log_scale")
    X = table[columns].to_numpy(dtype=float)
    y = meta[f"y_{task}"].to_numpy(dtype=str)
    split = next(item for item in load_splits("esc50_animals", task) if item.split_id == int(split_id))
    with np.load(path, allow_pickle=False) as saved:
        original = json.loads(str(saved["row"]))
        original_prediction = saved["y_pred"]
    model = clone(runner.classifiers()["linsvm"].estimator).set_params(**json.loads(original["best_params"]))
    np.random.seed(0)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[split.train], y[split.train])
    prediction = model.predict(X[split.test])
    iterations = int(np.max(model.named_steps["clf"].n_iter_))
    record = {"task": task, "method": tag, "feature_set": feature_set, "split_id": int(split_id),
              "best_params": original["best_params"], "n_iter": iterations,
              "selected_reached_cap": iterations >= model.named_steps["clf"].max_iter,
              "heldout_label_agreement": float(np.mean(prediction == original_prediction)),
              "balanced_accuracy": balanced_accuracy_score(y[split.test], prediction),
              "balanced_accuracy_difference": balanced_accuracy_score(y[split.test], prediction) - original["balanced_accuracy"],
              "warnings": sorted(set(str(item.message) for item in caught)),
              "seconds": time.perf_counter() - started}
    if record["selected_reached_cap"]:
        sensitivity = clone(model).set_params(clf__max_iter=20000)
        np.random.seed(0)
        with warnings.catch_warnings(record=True) as sensitivity_warnings:
            warnings.simplefilter("always")
            sensitivity.fit(X[split.train], y[split.train])
        sensitivity_prediction = sensitivity.predict(X[split.test])
        record.update(sensitivity_max_iter=20000,
                      sensitivity_n_iter=int(np.max(sensitivity.named_steps["clf"].n_iter_)),
                      sensitivity_ba_difference=balanced_accuracy_score(y[split.test], sensitivity_prediction) - original["balanced_accuracy"],
                      sensitivity_warnings=sorted(set(str(item.message) for item in sensitivity_warnings)))
    record["seconds"] = time.perf_counter() - started
    out.write_text(json.dumps(record, indent=2))
    return record


def validate_run(meta):
    """Check the wrapper against one direct upstream fit and audit cached jobs."""
    task, tag = "animal", "raw"
    split = load_splits("esc50_animals", task)[0]
    table = pd.read_parquet(DATA / "features/raw.parquet")
    X = table[feature_columns(table.columns)].to_numpy(dtype=float)
    y = meta["y_animal"].to_numpy(dtype=str)
    groups = meta["group"].to_numpy(dtype=str)
    _, prediction = fit_predict_split(runner.classifiers()["lda"], X, y, split, groups=groups)
    saved_path = next((DATA / "results/phase2_main/animal/jobs").glob("raw__comp__lda__0__*.npz"))
    with np.load(saved_path, allow_pickle=False) as saved:
        assert np.array_equal(prediction, saved["y_pred"])
    reference = table["ch0_log_scale"].to_numpy()
    for tag in METHODS:
        features = pd.read_parquet(DATA / "features" / f"{tag}.parquet")
        assert np.array_equal(features["frame_id"].to_numpy(), meta["frame_id"].to_numpy())
        assert np.array_equal(features["ch0_log_scale"].to_numpy(), reference)
        assert features["decomp_ok"].all()
        assert np.isfinite(features[feature_columns(features.columns, include_log_scale=True)].to_numpy()).all()
    warnings_found = []
    warned_svms = []
    counts = {}
    for task in ("animal", "bird"):
        paths = list((DATA / "results/phase2_main" / task / "jobs").glob("*.npz"))
        assert len(paths) == 720
        counts[task] = len(paths)
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                warnings = json.loads(str(saved["warnings"]))
                if saved["probabilities"].size:
                    probability = saved["probabilities"]
                    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
                    assert np.array_equal(saved["classes"][probability.argmax(axis=1)], saved["y_pred"])
                if warnings:
                    tag, feature_set, classifier, split_id, _ = path.stem.split("__")
                    if classifier == "linsvm":
                        warned_svms.append(str(path))
                    for message in warnings:
                        warnings_found.append({"task": task, "method": tag, "feature_set": feature_set,
                                               "classifier": classifier, "split_id": int(split_id), "warning": message})
    warning_table = pd.DataFrame(warnings_found, columns=["task", "method", "feature_set", "classifier", "split_id", "warning"])
    write_csv(warning_table, "fit_warnings.csv")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        audits = Parallel(n_jobs=3)(delayed(audit_selected_svm)(path) for path in warned_svms)
    write_csv(pd.DataFrame(audits), "svm_selected_fit_audit.csv")
    rng = np.random.default_rng(0)
    toy_y = np.repeat(np.arange(3), 20)
    toy_X = rng.normal(size=(60, 4))
    toy_X[:, 0] = toy_y + 1e-3 * rng.normal(size=60)
    auto_failed = False
    try:
        runner.QuadraticDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(toy_X, toy_y)
    except np.linalg.LinAlgError:
        auto_failed = True
    runner.QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.05).fit(toy_X, toy_y)
    assert auto_failed
    validation = {"direct_upstream_raw_lda_predictions_match": True, "shared_log_scale_identical": True,
                  "cached_fits": counts, "native_probabilities_finite_normalized_and_match_predictions": True,
                  "warning_records": len(warnings_found),
                  "joblib_worker_restart_warnings": (DATA / "results/phase2_main/run.log").read_text().count("A worker stopped"),
                  "qda_minimal_reproduction": {"auto_failed_on_nearly_class_constant_feature": auto_failed,
                                                "fixed_shrinkage_0.05_succeeded": True},
                  "svm_selected_refit_audit": {"n_warned_grid_fits": len(audits),
                                               "n_selected_refits_at_cap": sum(row["selected_reached_cap"] for row in audits),
                                               "minimum_label_agreement": min((row["heldout_label_agreement"] for row in audits), default=1.0),
                                               "maximum_absolute_ba_difference": max((abs(row["balanced_accuracy_difference"]) for row in audits), default=0.0)},
                  "versions": {package: version(package) for package in ["numpy", "scipy", "pandas", "scikit-learn", "pimf", "EMD-signal"]},
                  "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [STUDY / "PROTOCOL.md", STUDY / "baselines/run_classical.py",
                                                 STUDY / "emdirmf/cv.py", STUDY / "emdirmf/decompose.py"]}}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    return validation


def per_class(folds, best):
    records = []
    differences = []
    for task, task_rows in folds.groupby("task", sort=False):
        recall_columns = [column for column in task_rows if column.startswith("recall_") and task_rows[column].notna().any()]
        for keys, part in task_rows.groupby(["feature_set", "method", "classifier"]):
            for column in recall_columns:
                records.append({"task": task, "feature_set": keys[0], "method": keys[1],
                                "classifier": keys[2], "label": column[7:], "recall_mean": part[column].mean(),
                                "recall_sd": part[column].std(), "n_present_folds": part[column].notna().sum()})
        for keys, part in task_rows.groupby(["feature_set", "classifier"]):
            for method_a, method_b in default_pairs(METHODS):
                a = part[part["method"] == method_a].set_index("split_id")
                b = part[part["method"] == method_b].set_index("split_id")
                for column in recall_columns:
                    keep = a[column].notna() & b[column].notna()
                    row = {"task": task, "feature_set": keys[0], "classifier": keys[1],
                           "method_a": method_a, "method_b": method_b, "label": column[7:]}
                    row.update(paired_comparison(a.loc[keep, column], b.loc[keep, column],
                                                a.loc[keep, "n_train"], a.loc[keep, "n_test"]))
                    differences.append(row)
    table = pd.DataFrame(records)
    write_csv(table, "per_class_recall.csv")
    write_csv(table.merge(best[["task", "feature_set", "method", "classifier"]],
                          on=["task", "feature_set", "method", "classifier"]), "best_per_class_recall.csv")
    comparisons = pd.DataFrame(differences)
    for task, indices in comparisons.groupby("task").groups.items():
        comparisons.loc[indices, "p_nb_holm"] = holm(comparisons.loc[indices, "p_nb"])
        comparisons.loc[indices, "p_wilcoxon_holm"] = holm(comparisons.loc[indices, "p_wilcoxon"])
    write_csv(comparisons, "per_class_recall_differences.csv")


def clip_aggregation(meta, folds, best):
    # Fix the same native-probability classifier before examining clip scores.
    # LinearSVC has no predict_proba, so no artificial probabilities are made.
    selected_models = folds.query("classifier == 'logreg'")[["task", "method", "feature_set", "classifier"]].drop_duplicates()
    records = []
    selected_probabilities = []
    clip_probabilities = []
    for selected in selected_models.itertuples():
        task, tag, feature_set, classifier = selected.task, selected.method, selected.feature_set, selected.classifier
        labels = sorted(meta[f"y_{task}"].unique())
        for split in load_splits("esc50_animals", task):
            directory = DATA / "results/phase2_main" / task
            paths = list((directory / "jobs").glob(f"{tag}__{feature_set}__{classifier}__{split.split_id}__*.npz"))
            assert len(paths) == 1, paths
            path = paths[0]
            with np.load(path, allow_pickle=False) as saved:
                classes = saved["classes"].astype(str)
                probabilities = saved["probabilities"]
                frame_id = saved["frame_id"]
            assert np.array_equal(frame_id, split.test)
            assert probabilities.shape == (len(split.test), len(classes))
            assert np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1)
            heldout = meta.iloc[frame_id]
            part = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])
            part["recording_id"] = heldout["recording_id"].to_numpy()
            part["y_true"] = heldout[f"y_{task}"].to_numpy()
            part["frame_id"] = frame_id
            part["task"], part["method"], part["feature_set"], part["classifier"] = task, tag, feature_set, classifier
            part["split_id"], part["seed"], part["fold"] = split.split_id, split.seed, split.fold
            selected_probabilities.append(part)
            assert part.groupby("recording_id")["y_true"].nunique().max() == 1
            averaged = part.groupby("recording_id")[[f"prob_{label}" for label in classes]].mean()
            truth = part.groupby("recording_id")["y_true"].first().loc[averaged.index].to_numpy()
            prediction = classes[averaged.to_numpy().argmax(axis=1)]
            clip_table = averaged.reset_index()
            clip_table["y_true"], clip_table["y_pred"] = truth, prediction
            for column, value in [("task", task), ("method", tag), ("feature_set", feature_set),
                                  ("classifier", classifier), ("split_id", split.split_id),
                                  ("seed", split.seed), ("fold", split.fold)]:
                clip_table[column] = value
            clip_probabilities.append(clip_table)
            row = {"task": task, "method": tag, "feature_set": feature_set, "classifier": classifier,
                   "split_id": split.split_id, "seed": split.seed, "fold": split.fold,
                   "n_clips": len(averaged), "probability_kind": "native"}
            row.update(split_metrics(truth, prediction, labels))
            records.append(row)
    probabilities = pd.concat(selected_probabilities, ignore_index=True)
    probabilities.to_parquet(DATA / "results/phase2_main/selected_oof_probabilities.parquet", index=False)
    all_clips = pd.concat(clip_probabilities, ignore_index=True)
    all_clips.to_parquet(DATA / "results/phase2_main/selected_clip_probabilities.parquet", index=False)
    clip = pd.DataFrame(records)
    write_csv(clip, "clip_metrics_folds.csv")
    fold_summary = clip.groupby(["task", "method", "feature_set", "classifier"])[METRICS].agg(["mean", "std"]).reset_index()
    fold_summary.columns = ["_".join(filter(None, col)) if isinstance(col, tuple) else col for col in fold_summary.columns]
    write_csv(fold_summary, "clip_metrics_fold_summary.csv")
    seed_rows = []
    for keys, part in all_clips.groupby(["task", "method", "feature_set", "classifier", "seed"]):
        assert len(part) == 400 and part["recording_id"].nunique() == 400
        row = dict(zip(["task", "method", "feature_set", "classifier", "seed"], keys))
        row["n_clips"] = len(part)
        row.update(split_metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy(), sorted(part["y_true"].unique())))
        seed_rows.append(row)
    seed_metrics = pd.DataFrame(seed_rows)
    write_csv(seed_metrics, "clip_metrics_seeds.csv")
    summary = seed_metrics.groupby(["task", "method", "feature_set", "classifier"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, col)) if isinstance(col, tuple) else col for col in summary.columns]
    write_csv(summary, "clip_metrics_summary.csv")
    return summary


def plot_results(folds, paired, scales):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.constrained_layout.use": True})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, task in zip(axes, ["animal", "bird"]):
        means = folds[folds["task"] == task].groupby(["method", "feature_set", "classifier"])["balanced_accuracy"].mean()
        rows = [(tag, feature_set) for tag in METHODS for feature_set in SETS]
        classifiers = ["lda", "qda", "gnb", "logreg", "knn", "linsvm"]
        values = np.array([[means.loc[tag, feature_set, name] for name in classifiers] for tag, feature_set in rows])
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.2 if task == "animal" else 0.5, vmax=0.8 if task == "animal" else 1.0)
        ax.set_xticks(np.arange(len(classifiers)), classifiers, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(rows)), [f"{NAMES[tag]} {'+ level' if feature_set != 'comp' else 'no level'}" for tag, feature_set in rows])
        ax.set_title(f"{task}: mean balanced accuracy")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                threshold = 0.56 if task == "animal" else 0.81
                ax.text(j, i, f"{100 * values[i,j]:.1f}", ha="center", va="center", color="white" if values[i,j] > threshold else "black", fontsize=9)
        figure.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    figure.savefig(FIGURES / "balanced_accuracy.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, task in enumerate(["animal", "bird"]):
        for tag in METHODS[1:]:
            part = scales[(scales["task"] == task) & (scales["method"] == tag)].sort_values("scale_rank")
            for ax, metric in [(axes[0,column], "mean_epsilon2"), (axes[1,column], "mean_mi_nats")]:
                ax.plot(part["centroid_hz_median"], part[metric], "o-", label=NAMES[tag], markersize=4)
                ax.set_xscale("log")
                ax.grid(alpha=0.2)
        axes[0,column].set_title(task)
        axes[0,column].set_ylabel("Mean Kruskal-Wallis epsilon²")
        axes[1,column].set_ylabel("Mean MI, nats")
        axes[1,column].set_xlabel("Observed median component centroid, Hz")
        axes[0,column].legend(frameon=False)
    figure.savefig(FIGURES / "discriminability_scales.png")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    figure.suptitle("Pointwise 95% intervals; Holm-adjusted p-values are in the paired tables", fontsize=10)
    for ax, task in zip(axes, ["animal", "bird"]):
        part = paired[(paired["task"] == task) & (paired["metric"] == "balanced_accuracy")
                      & (paired["method_a"] == "irmf_h2_a1.41") & (paired["method_b"] == "emd")]
        for offset, feature_set, color in [(-0.12, "comp", "#187b91"), (0.12, "comp+log_scale", "#ba5931")]:
            cell = part[part["feature_set"] == feature_set].set_index("classifier").loc[classifiers]
            y = np.arange(6) + offset
            ax.errorbar(cell["mean_diff"] * 100, y, xerr=np.array([cell["mean_diff"] - cell["ci_low"], cell["ci_high"] - cell["mean_diff"]]) * 100,
                        fmt="o", color=color, capsize=2, label="No log_scale" if feature_set == "comp" else "With log_scale")
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_yticks(np.arange(6), classifiers)
        ax.set_title(task)
        ax.set_xlabel("IRMF minus EMD balanced accuracy, pp\n95% corrected resampled CI")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False)
    figure.savefig(FIGURES / "irmf_emd_paired.png")
    plt.close(figure)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    assert len(meta) == 5100 and meta["recording_id"].nunique() == 400
    assert meta.groupby("recording_id")["group"].nunique().max() == 1
    folds = pd.concat([pd.read_csv(OUT / f"results_folds_{task}.csv") for task in ("animal", "bird")], ignore_index=True)
    assert len(folds) == 1440 and not folds[METRICS].isna().any().any()
    assert not folds.duplicated(["task", "method", "feature_set", "classifier", "split_id"]).any()
    assert folds.groupby(["task", "method", "feature_set", "classifier"]).size().eq(15).all()
    validation = validate_run(meta)
    means = folds.groupby(["task", "method", "feature_set", "classifier"], sort=False)[METRICS].mean().reset_index()
    best = means.sort_values("balanced_accuracy", ascending=False, kind="stable").drop_duplicates(["task", "method", "feature_set"])
    write_csv(means, "classifier_metrics.csv")
    write_csv(best, "best_metrics.csv")
    paired = paired_tables(folds)
    scales = feature_scales(meta)
    convergence = convergence_tables(meta, folds, best)
    per_class(folds, best)
    uploader_balanced_metrics(meta)
    clip = clip_aggregation(meta, folds, best)
    logreg = means.query("classifier == 'logreg'")
    write_csv(logreg.merge(clip, on=["task", "method", "feature_set", "classifier"]), "logreg_frame_clip_metrics.csv")
    costs = []
    for tag in METHODS:
        if tag == "raw":
            costs.append({"method": tag, "cpu_seconds_per_frame": 0.0, "wall_seconds_per_frame": 0.0, "n_failed": 0})
        else:
            saved = json.loads((DATA / "decomp" / f"{tag}.json").read_text())
            costs.append({"method": tag, "cpu_seconds_per_frame": saved["seconds_per_frame"]["cpu_mean"],
                          "wall_seconds_per_frame": saved["seconds_per_frame"]["wall_mean"], "n_failed": saved["n_failed"]})
    write_csv(pd.DataFrame(costs), "decomposition_costs.csv")
    confounds = meta.groupby("y_animal").agg(n_frames=("frame_id", "size"), n_groups=("group", "nunique"),
                                              median_rms_dbfs=("frame_rms_dbfs", "median"))
    confounds["largest_uploader_fraction"] = meta.groupby("y_animal")["group"].apply(lambda x: x.value_counts().max() / len(x))
    write_csv(confounds.reset_index(), "class_confounds.csv")
    plot_results(folds, paired, scales)
    ablations = ["irmf_h1_a1.41", "irmf_h4_a1.41", "irmf_h2_a2", "imf_linear_a2"]
    readiness = []
    for tag in ablations:
        path = DATA / "decomp" / f"{tag}.json"
        record = {"tag": tag, "decomp_json": path.exists(),
                  "features_parquet": (DATA / "features" / f"{tag}.parquet").exists(),
                  "features_json": (DATA / "features" / f"{tag}.json").exists()}
        saved = json.loads(path.read_text()) if path.exists() else {}
        record["ready"] = bool(record["decomp_json"] and record["features_parquet"] and record["features_json"]
                               and not saved.get("partial", True) and saved.get("shape", [0])[0] == 5100)
        readiness.append(record)
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(), "convergence": convergence,
               "analysis_seconds": time.perf_counter() - started, "validation": validation,
               "ablation_readiness": readiness, "all_ablations_ready": all(row["ready"] for row in readiness),
               "ready_ablations": [row["tag"] for row in readiness if row["ready"]],
               "inference": "Holm separately for each task/metric table: six pairs x six classifiers x two feature sets = 72 tests.",
               "feature_statistics": "Descriptive all-frame H/(N-1) epsilon2 and 16-quantile-bin Miller-Madow-corrected MI, nats.",
               "frequency": "Hz are observed median centroids of nonzero components; inverse-window Hz is descriptive, not a passband edge.",
               "clip_aggregation": "Logistic regression fixed for every method. Within each outer fold, average its native probabilities over a clip's frames, then argmax. Headline clip scores pool the 400 held-out clips per seed, then average three seeds. The 15-fold mean is also saved. No artificial SVM probabilities or extra calibration fits.",
               "selection": "Best classifier selected by mean outer-CV balanced accuracy. Its headline score is exploratory; paired inference always matches classifier and split. Clip classifier is fixed logistic regression."}
    (OUT / "analysis_context.json").write_text(json.dumps(summary, indent=2))
    print(best.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def analyse_ablations():
    """Use the locked main-analysis readiness snapshot, not later arrivals."""
    context = json.loads((OUT / "analysis_context.json").read_text())
    selected = context["ready_ablations"]
    if not selected:
        raise RuntimeError("No ablation caches were complete at the main-analysis deadline")
    main_folds = pd.concat([pd.read_csv(OUT / f"results_folds_{task}.csv") for task in ("animal", "bird")])
    ablation_folds = pd.concat([pd.read_csv(OUT / "ablations" / f"results_folds_{task}.csv") for task in ("animal", "bird")])
    assert set(ablation_folds["method"]) == set(selected)
    assert len(ablation_folds) == len(selected) * 2 * 2 * 6 * 15
    assert ablation_folds.groupby(["task", "method", "feature_set", "classifier"]).size().eq(15).all()
    combined = pd.concat([main_folds, ablation_folds], ignore_index=True)
    means = combined.groupby(["task", "method", "feature_set", "classifier"])[METRICS].mean().reset_index()
    best = means.sort_values("balanced_accuracy", ascending=False, kind="stable").drop_duplicates(["task", "method", "feature_set"])
    write_csv(means, "ablation_classifier_metrics.csv")
    write_csv(best, "ablation_best_metrics.csv")
    pairs = [("irmf_h1_a1.41", "irmf_h2_a1.41"), ("irmf_h4_a1.41", "irmf_h2_a1.41"),
             ("irmf_h1_a1.41", "irmf_h4_a1.41"), ("irmf_h2_a2", "irmf_h2_a1.41"),
             ("imf_linear_a2", "imf_linear_a1.41"), ("irmf_h2_a2", "imf_linear_a2")]
    present = set(combined["method"])
    pairs = [(a, b) for a, b in pairs if a in present and b in present]
    long = combined.melt(id_vars=["task", "method", "feature_set", "classifier", "split_id", "n_train", "n_test"],
                         value_vars=METRICS, var_name="metric", value_name="value")
    comparisons = []
    for task, part in long.groupby("task"):
        for metric in METRICS:
            table = compare_methods(part, metric=metric, pairs=pairs, by=["feature_set", "classifier"])
            table.insert(0, "task", task)
            comparisons.append(table)
    write_csv(pd.concat(comparisons, ignore_index=True), "ablation_paired_comparisons.csv")
    (OUT / "ablation_analysis_context.json").write_text(json.dumps(
        {"readiness_checked_at": context["checked_at"], "methods_run": selected,
         "methods_skipped": [row["tag"] for row in context["ablation_readiness"] if not row["ready"]],
         "pairs": pairs, "holm_tests_per_task_metric": len(pairs) * 12}, indent=2))
    print(best[best["method"].isin(selected)].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations-only", action="store_true")
    arguments = parser.parse_args()
    analyse_ablations() if arguments.ablations_only else main()
