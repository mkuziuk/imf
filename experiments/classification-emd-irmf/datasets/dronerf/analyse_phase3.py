#!/usr/bin/env python3
"""Audit saved DroneRF ML outputs and make paired, segment and cost tables."""

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
from emdirmf.decompose import scale_order
from emdirmf.features import FEATURE_NAMES
from emdirmf.io import load_meta
from emdirmf.paths import data_root
from emdirmf.stats import compare_methods

METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
OUT = ROOT / "output/classification-emd-irmf/dronerf/ml"
DATA = data_root() / "dronerf"


def audit_predictions(folds, predictions, meta, task, runner, *, n_splits=None, methods=None, classifiers=None, split_ids=None):
    keys = ["method", "feature_set", "classifier", "split_id"]
    assert not folds.duplicated(keys).any()
    assert not predictions.duplicated(keys + ["frame_id"]).any()
    split_map = {split.split_id: split for split in load_splits("dronerf", task)}
    classes = sorted(meta[f"y_{task}"].unique())
    probability_columns = [f"prob_{label}" for label in classes]
    ties = 0
    for key, group in predictions.groupby(keys):
        method, feature_set, classifier, split_id = key
        group = group.sort_values("frame_id")
        split = split_map[split_id]
        assert set(meta.iloc[split.train][f"y_{task}"]) == set(classes)
        assert set(meta.iloc[split.test][f"y_{task}"]) == set(classes)
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
    expected = set(split_ids) if split_ids is not None else set(range(
        n_splits if n_splits is not None else (5 if runner == "cnn" else 15)))
    expected_methods = METHODS if methods is None else methods
    expected_classifiers = {"trees": ["lightgbm", "xgboost", "extratrees"],
        "rocket": ["minirocket"], "cnn": ["compact_resnet"],
        "multirocket": ["multirocket"], "spectrogram": ["log_stft_cnn2d"]}.get(runner, ["lightgbm"])
    if classifiers is not None:
        expected_classifiers = classifiers
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


def segment_scores(predictions, meta, task, runner):
    """Average native held-out probabilities per segment, then pool per seed."""
    predictions = predictions.merge(meta[["frame_id", "group"]],
                                    on="frame_id", validate="many_to_one")
    keys = ["method", "feature_set", "classifier", "seed"]
    probabilities = sorted(c for c in predictions if c.startswith("prob_"))
    labels = np.asarray([c[5:] for c in probabilities])
    rows, fold_rows = [], []
    split_map = {s.split_id: s for s in load_splits("dronerf", task)}
    for key, group in predictions.groupby(keys):
        assert group.groupby("group").split_id.nunique().max() == 1
        assert group.groupby("group").y_true.nunique().max() == 1
        means = group.groupby("group")[probabilities].mean()
        target = group.groupby("group").y_true.first().loc[means.index]
        pred = labels[means.to_numpy().argmax(axis=1)]
        assert len(means) == meta.group.nunique() == 227
        row = dict(zip(keys, key))
        row.update(task=task, runner=runner, n_segments=len(means),
                   balanced_accuracy=balanced_accuracy_score(target, pred),
                   macro_f1=f1_score(target, pred, average="macro", zero_division=0),
                   accuracy=accuracy_score(target, pred))
        rows.append(row)
        for split_id, part in group.groupby("split_id"):
            part_means = part.groupby("group")[probabilities].mean()
            truth = part.groupby("group").y_true.first().loc[part_means.index]
            guess = labels[part_means.to_numpy().argmax(axis=1)]
            split = split_map[split_id]
            item = dict(zip(keys, key))
            item.update(task=task, runner=runner, split_id=split_id,
                        n_train=meta.iloc[split.train].group.nunique(), n_test=len(truth),
                        balanced_accuracy=balanced_accuracy_score(truth, guess),
                        macro_f1=f1_score(truth, guess, average="macro", zero_division=0),
                        accuracy=accuracy_score(truth, guess))
            fold_rows.append(item)
    return rows, fold_rows


def summarize(frame, keys, metrics=METRICS):
    result = frame.groupby(keys)[metrics].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(c).rstrip("_") for c in result.columns]
    return result


def convergence_strata(predictions, task, runner, status):
    """Descriptive matched populations; no independence claim for strata."""
    if task != "10class" or "lonly" in runner:
        return []
    selected = predictions[predictions.method.isin(METHODS)].merge(status, on="frame_id", validate="many_to_one")
    keys = ["method", "feature_set", "classifier", "seed", "gd60_all_converged"]
    rows = []
    for key, group in selected.groupby(keys):
        row = dict(zip(keys, key))
        row.update(task=task, runner=runner, n_frames=len(group), n_classes=group.y_true.nunique(),
                   balanced_accuracy=balanced_accuracy_score(group.y_true, group.y_pred),
                   macro_f1=f1_score(group.y_true, group.y_pred, average="macro", zero_division=0),
                   accuracy=accuracy_score(group.y_true, group.y_pred))
        rows.append(row)
    return rows


def paired_tables(folds, *, level="frame"):
    """Separate predeclared main and matched-component Holm families."""
    output = []
    families = {
        "primary": METHODS,
        "matched_capacity": ["emd", "imf_linear_a2", "irmf_h2_a2_conv"],
        "h_sensitivity": ["irmf_h2_a1.41_conv", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv"],
    }
    for task, task_folds in folds.groupby("task"):
        for family, methods in families.items():
            selected = task_folds[task_folds.method.isin(methods) & ~task_folds.runner.str.contains("lonly")]
            if selected.empty:
                continue
            long = selected.melt(id_vars=["method", "feature_set", "classifier", "split_id",
                                          "n_train", "n_test"], value_vars=METRICS,
                                 var_name="metric", value_name="value")
            for metric in METRICS:
                table = compare_methods(long, metric=metric,
                                        pairs=list(itertools.combinations(methods, 2)),
                                        by=("feature_set", "classifier"))
                if table.empty:
                    continue
                table.insert(0, "task", task)
                table["inference_table"] = family
                table["level"] = level
                table["holm_family_size"] = len(table)
                output.append(table)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def orient_requested(paired):
    wanted = [(a, b) for a, b in itertools.permutations(METHODS, 2)
              if a in {"irmf_h2_a1.41", "irmf_h2_a1.41_conv"} or b in {"emd", "raw"}]
    wanted += [("irmf_h2_a2_conv", "emd"), ("imf_linear_a2", "emd"),
               ("irmf_h2_a2_conv", "imf_linear_a2")]
    rows = []
    for numerator, denominator in wanted:
        direct = paired[(paired.method_a == numerator) & (paired.method_b == denominator)].copy()
        reverse = paired[(paired.method_b == numerator) & (paired.method_a == denominator)].copy()
        reverse["method_a"], reverse["method_b"] = numerator, denominator
        for column in ("mean_diff", "t"):
            reverse[column] *= -1
        lower, upper = reverse.ci_low.copy(), reverse.ci_high.copy()
        reverse["ci_low"], reverse["ci_high"] = -upper, -lower
        for a, b in (("mean_a", "mean_b"), ("wins_a", "wins_b")):
            reverse[a], reverse[b] = reverse[b].copy(), reverse[a].copy()
        rows.extend([direct, reverse])
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        ["task", "metric", "feature_set", "classifier", "method_a", "method_b", "inference_table", "level"])


def snapshot_inputs():
    import hashlib
    from datetime import datetime, timezone

    jobs = json.loads((OUT / "jobs.json").read_text())["jobs"]
    methods = sorted({m for job in jobs for m in job["methods"]})
    files = [DATA / "prepared/frames.npy", DATA / "prepared/meta.parquet", DATA / "prepared/dataset.json"]
    files.extend(DATA / f"splits_{task}.json" for task in ("10class", "4class", "2class"))
    for tag in methods:
        files.extend([DATA / f"features/{tag}.parquet", DATA / f"features/{tag}.json"])
        if tag != "raw":
            files.extend(DATA / f"decomp/{tag}{suffix}" for suffix in (".npy", ".json", ".meta.parquet"))
    files.extend(STUDY / name for name in ("ml/common.py", "ml/run_trees.py", "ml/run_rocket.py", "ml/run_cnn.py",
                                         "emdirmf/stacks.py", "emdirmf/stats.py", "emdirmf/decompose.py", "emdirmf/features.py"))
    entries = []
    for path in files:
        stat = path.stat()
        entry = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix != ".npy":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        entries.append(entry)
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(), "files": entries,
              "note": "Large npy caches use size and mtime; other inputs and shared modeling sources use SHA-256."}
    (OUT / "input_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Captured {len(entries)} input/source files")


def main():
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="analyse completed jobs while queues continue")
    parser.add_argument("--snapshot-inputs", action="store_true", help="capture the input manifest before starting a new study run")
    args = parser.parse_args()
    threadpool_limits(1)
    if args.snapshot_inputs:
        snapshot_inputs()
        return
    snapshot = json.loads((OUT / "input_manifest.json").read_text())
    for entry in snapshot["files"]:
        path = Path(entry["path"])
        current = path.stat()
        assert current.st_size == entry["bytes"] and current.st_mtime_ns == entry["mtime_ns"], path
        if "sha256" in entry:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
            assert digest.hexdigest() == entry["sha256"], path
    meta = load_meta("dronerf")
    manifest = json.loads((OUT / "jobs.json").read_text())
    all_folds, segments, segment_folds, checks, costs, importance = [], [], [], [], [], []
    strata = []
    diagnostic = pd.read_parquet(DATA / "decomp/irmf_h2_a1.41.meta.parquet", columns=["frame_id", "n_unconverged"])
    status = diagnostic.groupby("frame_id").n_unconverged.sum().eq(0).rename("gd60_all_converged").reset_index()
    complete, pending = [], []
    for job in manifest["jobs"]:
        task, runner = job["task"], job["runner"]
        directory = OUT / job["directory"]
        path = directory / f"run_{task}.json"
        if not path.exists():
            pending.append(job)
            continue
        run = json.loads(path.read_text())
        folds = pd.read_csv(directory / f"results_folds_{task}.csv")
        logical_runner = job.get("analysis_runner", job["name"])
        folds["runner"] = logical_runner
        folds["task"] = task
        predictions = pd.read_parquet(run["predictions"])
        assert run["methods"] == job["methods"]
        expected_ids = job.get("split_ids", list(range(job["folds"])))
        assert run["split_ids"] == expected_ids
        checks.append(audit_predictions(folds, predictions, meta, task, runner,
                                       n_splits=job["folds"], methods=job["methods"],
                                       classifiers=job["classifiers"], split_ids=expected_ids))
        checks[-1]["job"] = job["name"]
        if runner == "trees":
            parameters = folds.best_params.map(json.loads)
            for classifier, cap in job["caps"].items():
                assert all(0 < value["n_estimators"] <= cap
                           for value in parameters[folds.classifier == classifier])
            channels = 1 if "lonly" in job["name"] else 2
            for (method, feature_set), cell in folds.groupby(["method", "feature_set"]):
                slots = 1 if method == "raw" else json.loads(
                    (DATA / "decomp" / f"{method}.json").read_text())["shape"][2]
                expected = channels * (slots * len(FEATURE_NAMES) + feature_set.endswith("log_scale"))
                assert cell.n_input_features.eq(expected).all()
            importance_path = directory / f"importance_scales_{task}.csv"
            if importance_path.exists():
                imp = pd.read_csv(importance_path)
                imp["task"], imp["runner"] = task, job["name"]
                importance.append(imp)
        elif runner == "cnn":
            assert folds.best_epoch.between(1, job["epochs"]).all()
        elif runner == "rocket":
            assert set(folds.n_features) == {job["features"]}
        if runner in {"cnn", "rocket"}:
            for method, description in run["stacks"].items():
                shape = description["shape"]
                if len(shape) == 4:
                    assert shape == [12000, 2, 129, 33] and description["fs"] == 40_000_000
                else:
                    order = scale_order(method, 2048)
                    assert shape == [12000, 2 * len(order), 2048 // run["params"]["downsample"]]
                    assert description["storage_slots_in_scale_order"] == order
                    if job.get("cache_stacks", False):
                        assert description["reader_cache"]["exact_input_equality"]
        pooled, by_fold = segment_scores(predictions, meta, task, logical_runner)
        strata.extend(convergence_strata(predictions, task, logical_runner, status))
        segments.extend(pooled)
        segment_folds.extend(by_fold)
        all_folds.append(folds)
        costs.append({"task": task, "runner": job["name"], "wall_seconds": run["seconds"],
                      "n_fits": len(folds), "workers": run["workers"],
                      "model_fit_seconds": float(folds.fit_seconds.sum()),
                      "predict_seconds": float(folds.predict_seconds.sum()),
                      "timings_share_transforms": run.get("timings_share_transforms", False)})
        complete.append(job)
    if pending and not args.partial:
        raise RuntimeError(f"{len(pending)} registered jobs have not finished")
    if not all_folds:
        raise RuntimeError("no completed jobs")
    used_methods = sorted({m for job in complete for m in job["methods"]})
    log_columns = ["ch0_log_scale", "ch1_log_scale"]
    raw_scales = pd.read_parquet(DATA / "features/raw.parquet", columns=["frame_id", *log_columns])
    decomposition = [{"method": "raw", "seconds_per_frame": 0, "n_slots": 1}]
    for method in used_methods:
        values = pd.read_parquet(DATA / "features" / f"{method}.parquet",
                                 columns=["frame_id", *log_columns, "decomp_ok"])
        np.testing.assert_array_equal(values.frame_id, meta.frame_id)
        np.testing.assert_array_equal(values[log_columns], raw_scales[log_columns])
        assert values.decomp_ok.all()
        if method == "raw":
            continue
        info = json.loads((DATA / "decomp" / f"{method}.json").read_text())
        if method.endswith("_conv"):
            assert info["irmf"]["channels_all_converged"] == info["shape"][0] * info["shape"][1]
        decomposition.append({"method": method, "seconds_per_frame": info["seconds_per_frame"]["cpu_mean"],
                              "n_slots": info["shape"][2], "n_failed": info["n_failed"],
                              "solver": info.get("params", {}).get("solver", "pimf/PyEMD")})
    folds = pd.concat(all_folds, ignore_index=True)
    assert not folds.duplicated(["task", "runner", "method", "feature_set", "classifier", "split_id"]).any()
    folds.to_csv(OUT / "all_folds.csv", index=False)
    keys = ["task", "runner", "method", "feature_set", "classifier"]
    scores = summarize(folds, keys)
    scores.to_csv(OUT / "scores.csv", index=False)
    segment = pd.DataFrame(segments)
    assert not segment.duplicated(keys + ["seed"]).any()
    segment.to_csv(OUT / "segment_metrics_by_seed.csv", index=False)
    segment_summary = summarize(segment, keys).merge(
        segment.groupby(keys).seed.nunique().rename("n_seeds").reset_index(), on=keys)
    segment_summary.to_csv(OUT / "segment_scores.csv", index=False)
    segment_fold = pd.DataFrame(segment_folds)
    segment_fold.to_csv(OUT / "segment_metrics_by_fold.csv", index=False)
    paired = paired_tables(folds)
    paired.to_csv(OUT / "paired_all_metrics.csv", index=False)
    requested = orient_requested(paired)
    requested.to_csv(OUT / "paired_requested.csv", index=False)
    segment_pairs = paired_tables(segment_fold, level="segment")
    segment_pairs.to_csv(OUT / "paired_segment.csv", index=False)
    # Same-method L-only minus both-view contrast, with Holm across all
    # methods and feature sets in each task/metric table.
    band_tables = []
    for task, selected in folds.groupby("task"):
        l_only = selected[selected.runner.str.contains("lonly")].copy()
        if l_only.empty:
            continue
        reference = selected[~selected.runner.str.contains("lonly") &
                             selected.classifier.isin(l_only.classifier)].copy()
        methods = list(l_only.method.unique())
        l_only["method"] += "__L"
        reference["method"] += "__LH"
        long = pd.concat([l_only, reference]).melt(
            id_vars=["method", "feature_set", "classifier", "split_id", "n_train", "n_test"],
            value_vars=METRICS, var_name="metric", value_name="value")
        for metric in METRICS:
            table = compare_methods(long, metric=metric, pairs=[(m + "__L", m + "__LH") for m in methods],
                                    by=("feature_set", "classifier"))
            table["task"], table["holm_family_size"] = task, len(table)
            band_tables.append(table)
    if band_tables:
        pd.concat(band_tables).to_csv(OUT / "paired_bands.csv", index=False)
    recall_columns = [c for c in folds if c.startswith("recall_")]
    recall = folds.melt(id_vars=keys, value_vars=recall_columns, var_name="class", value_name="recall").dropna()
    recall["class"] = recall["class"].str.removeprefix("recall_")
    recall.groupby(keys + ["class"]).recall.agg(["mean", "std"]).reset_index().to_csv(
        OUT / "per_class_recall.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "runner_costs.csv", index=False)
    pd.DataFrame(decomposition).to_csv(OUT / "decomposition_costs.csv", index=False)
    if strata:
        pd.DataFrame(strata).to_csv(OUT / "convergence_strata.csv", index=False)
    if importance:
        pd.concat(importance).to_csv(OUT / "importance_by_scale.csv", index=False)
    validation = {"checks": checks, "all_checks_passed": True, "completed_jobs": len(complete),
                  "pending_jobs": len(pending), "shared_log_scales_identical": True,
                  "input_and_shared_model_sources_unchanged_since": snapshot["captured_utc"],
                  "inference": "Nadeau-Bengio primary; Wilcoxon secondary; Holm per task/metric/family across model and scale variants",
                  "segment": "mean native OOF probabilities within each held-out segment; pool 227 segments per seed"}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    detailed = ["# All frame metrics", "", "Each cell is balanced accuracy / macro-F1 / accuracy, percent.", ""]
    effects = ["# Requested paired effects", "", "Positive differences favor A. Individual Nadeau-Bengio 95% CIs; Holm within each declared table.", ""]
    for (task, runner, feature_set), cell in scores.groupby(["task", "runner", "feature_set"], sort=False):
        cell = cell.copy()
        cell["BA/F1/accuracy"] = cell.apply(lambda r: "/".join(f"{100 * r[f'{m}_mean']:.2f}" for m in METRICS), axis=1)
        detailed.extend([f"## {task}, {runner}, {feature_set}", "", cell.pivot(
            index="method", columns="classifier", values="BA/F1/accuracy").to_markdown(), ""])
    for key, cell in requested[requested.metric == "balanced_accuracy"].groupby(
            ["task", "inference_table", "classifier", "feature_set"], sort=False):
        cell = cell.copy()
        cell["A - B"] = cell.method_a + " - " + cell.method_b
        cell["difference [95% CI]"] = cell.apply(lambda r:
            f"{100 * r.mean_diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]", axis=1)
        effects.extend(["## " + ", ".join(key), "", cell[["A - B", "n_splits", "holm_family_size",
            "difference [95% CI]", "p_nb", "p_nb_holm", "p_wilcoxon", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""])
    (OUT / "SCORES_ALL_METRICS.md").write_text("\n".join(detailed))
    (OUT / "PAIRS.md").write_text("\n".join(effects))
    names = {"raw": "Raw", "emd": "EMD", "imf_linear_a1.41": "Linear IMF",
             "irmf_h2_a1.41": "IRMF GD60", "irmf_h2_a1.41_conv": "IRMF converged",
             "irmf_h2_a2_conv": "IRMF converged a=2", "imf_linear_a2": "Linear IMF a=2"}
    model_names = {"lightgbm": "LGBM", "xgboost": "XGB", "extratrees": "ET",
                   "minirocket": "MiniRocket", "compact_resnet": "CNN"}
    compact = ["# Balanced accuracy", "", "Percent, without / with log_scale. Frame-fold means.", ""]
    for task in ("10class", "4class", "2class"):
        chosen = scores[(scores.task == task) & ~scores.runner.str.contains("lonly")]
        families = [c for c in model_names if c in set(chosen.classifier)]
        if chosen.empty:
            continue
        rows = []
        for method, label in names.items():
            row = {"Method": label}
            for classifier in families:
                values = []
                for feature_set in ("comp", "comp+log_scale"):
                    cell = chosen[(chosen.method == method) & (chosen.classifier == classifier) &
                                  (chosen.feature_set == feature_set)]
                    values.append(f"{100 * cell.balanced_accuracy_mean.iloc[0]:.2f}" if len(cell) else "-")
                row[model_names[classifier]] = "/".join(values)
            rows.append(row)
        compact.extend([f"## {task}", "", pd.DataFrame(rows).to_markdown(index=False), ""])
    (OUT / "TABLES.md").write_text("\n".join(compact))
    print(f"Audited {len(complete)} jobs, {len(folds)} fits, {sum(c['prediction_rows'] for c in checks)} predictions; {len(pending)} pending.")


if __name__ == "__main__":
    main()
