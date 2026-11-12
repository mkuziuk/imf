"""DroneRF-only phase-2 analyses. Data and predictions stay under DATA_ROOT.

Run with --bands after the three full run_classical.py tasks, then --report.
No shared package or runner is modified. Inner CV and grids come from the runner.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

STUDY_DIR = Path(__file__).resolve().parents[2]
ROOT = STUDY_DIR.parents[1]
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR / "baselines"))
import run_classical as baseline
from emdirmf.cv import fit_predict_split, load_splits, split_metrics
from emdirmf.decompose import scale_order, slot_labels
from emdirmf.features import feature_columns
from emdirmf.stats import compare_methods, corrected_resampled_ttest, holm, paired_comparison

DATA = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification")) / "dronerf"
OUT = ROOT / "output/classification-emd-irmf/dronerf/classical"
CACHE = DATA / "work/classical"
TAGS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
NAMES = dict(zip(TAGS, ["raw", "EMD", "linear IMF", "IRMF"]))
TASKS = ["2class", "4class", "10class"]
FEATURE_SETS = ["comp", "comp+log_scale"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
COLORS = ["#64748b", "#2563eb", "#d97706", "#059669"]


def load_folds():
    return pd.concat([pd.read_csv(OUT / f"results_folds_{task}.csv") for task in TASKS],
                     ignore_index=True)


def choose_classifiers(folds):
    """One classifier per task: the highest-scoring method/classifier pair, averaged over both feature sets.

    This outer-score choice is exploratory and must not be presented as nested
    model-family selection. All-classifier comparisons remain the primary record.
    """
    return (folds.groupby(["task", "method", "classifier"])["balanced_accuracy"].mean()
            .reset_index().sort_values("balanced_accuracy", ascending=False)
            .drop_duplicates("task").set_index("task")["classifier"].to_dict())


def models_for_task(task):
    models = baseline.classifiers()
    path = OUT / f"run_{task}.json"
    context = json.loads(path.read_text()) if path.exists() else {}
    if "grids" in context:
        grids = context["grids"]
        for name, grid in grids.items():
            if grid is not None:
                models[name].param_grid = grid
    for name, parameters in context.get("fixed_parameters", {}).items():
        models[name] = models[name].estimator.set_params(**parameters)
    return models


def run_bands(tasks=("10class",)):
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    groups = meta["group"].astype(str).to_numpy()
    chosen = choose_classifiers(load_folds())
    pd.Series({task: chosen[task] for task in tasks}).rename("classifier").rename_axis("task").to_csv(OUT / "band_classifiers.csv")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(CACHE / "tmp")
    all_rows = []
    for task in tasks:
        path = CACHE / f"band_folds_{task}.csv"
        y = meta[f"y_{task}"].astype(str).to_numpy()
        labels = sorted(set(y))
        splits = load_splits("dronerf", task)
        classifier = chosen[task]
        model = models_for_task(task)[classifier]
        jobs = []
        arrays = []
        for tag in TAGS:
            table = pd.read_parquet(DATA / f"features/{tag}.parquet")
            for feature_set in FEATURE_SETS:
                columns = feature_columns(table.columns,
                                          include_log_scale=feature_set.endswith("log_scale"))
                columns = [name for name in columns if name.startswith("ch0_")]
                x_path = CACHE / f"band_X_{tag}_{feature_set}.npy"
                np.save(x_path, table[columns].to_numpy(dtype=np.float64))
                x = np.load(x_path, mmap_mode="r")
                arrays.append(x)
                for split in splits:
                    jobs.append(((tag, feature_set, "L"), x, split))
        table = pd.read_parquet(DATA / "features/raw.parquet")
        x_path = CACHE / "log_scale_only_X.npy"
        np.save(x_path, table[["ch0_log_scale", "ch1_log_scale"]].to_numpy(dtype=np.float64))
        x = np.load(x_path, mmap_mode="r")
        arrays.append(x)
        jobs.extend([(("raw", "log_scale_only", "L+H"), x, split) for split in splits])
        print(f"{task}: {len(jobs)} band/control fits, {classifier}", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            outputs = Parallel(n_jobs=3)(
                delayed(baseline._job)(key, model, x, y, split, groups, labels)
                for key, x, split in jobs
            )
        rows, predictions = [], []
        for (tag, feature_set, band), row, pred in outputs:
            row.update(task=task, method=tag, feature_set=feature_set, band=band,
                       classifier=classifier)
            rows.append(row)
            split = splits[row["split_id"]]
            predictions.append(pd.DataFrame({
                "task": task, "method": tag, "feature_set": feature_set, "band": band,
                "classifier": classifier, "split_id": split.split_id,
                "seed": split.seed, "fold": split.fold, "frame_id": split.test,
                "y_true": y[split.test], "y_pred": pred,
            }))
        pd.DataFrame(rows).to_csv(path, index=False)
        pd.concat(predictions).to_parquet(CACHE / f"band_predictions_{task}.parquet", index=False)
        all_rows.append(pd.DataFrame(rows))
        print(f"{task} band/control finished", flush=True)
    pd.concat(all_rows).to_csv(OUT / "band_folds.csv", index=False)
    (OUT / "band_run.json").write_text(json.dumps({
        "seconds": time.perf_counter() - started, "workers": 3, "tasks": list(tasks),
        "classifiers": {task: chosen[task] for task in tasks},
        "fixed_parameters": {task: json.loads((OUT / f"run_{task}.json").read_text()).get("fixed_parameters", {})
                             for task in tasks},
        "selection": "exploratory best method/classifier pair averaged over both feature sets per task",
        "grids": {task: {name: getattr(model, "param_grid", None)
                           for name, model in models_for_task(task).items()} for task in tasks},
        "log_scale_control": "two original channel log scales only; same selected classifier",
    }, indent=2))


def static_diagnostics():
    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    irmf = pd.read_parquet(DATA / "decomp/irmf_h2_a1.41.meta.parquet")
    assert len(irmf) == 2 * len(meta) and irmf["ok"].all()
    with open(DATA / "decomp/irmf_h2_a1.41.json") as handle:
        spec = json.load(handle)
    # stage_* arrays are in canonical storage order: fine to coarse, unlike the
    # schedule.window_sizes_pimf_order field in the JSON.
    converged = np.stack(irmf["stage_converged"].to_numpy()).astype(bool)
    iterations = np.stack(irmf["stage_iterations"].to_numpy())
    assert (iterations <= 60).all()
    assert np.array_equal((~converged).sum(axis=1), irmf["n_unconverged"].to_numpy())
    assert np.allclose((~converged).mean(axis=0),
                       spec["irmf"]["unconverged_fraction_per_slot"], atol=0.000051)
    merged = irmf.merge(meta[["frame_id", "y_10class", "drone"]], on="frame_id", validate="many_to_one")
    records = []
    for channel in ("all", 0, 1):
        for label in ["all"] + sorted(meta["y_10class"].unique()):
            mask = np.ones(len(merged), dtype=bool)
            if channel != "all":
                mask &= merged["channel"].to_numpy() == channel
            if label != "all":
                mask &= merged["y_10class"].to_numpy() == label
            for slot, name in enumerate(slot_labels("irmf_h2_a1.41", 2048)[:-1]):
                records.append({"channel": channel, "class": label, "slot": slot,
                                "stage": name, "n_channels": int(mask.sum()),
                                "unconverged_fraction": float((~converged[mask, slot]).mean()),
                                "mean_iterations": float(iterations[mask, slot].mean()),
                                "at_cap_fraction": float((iterations[mask, slot] == 60).mean())})
    pd.DataFrame(records).to_csv(OUT / "convergence_stages.csv", index=False)
    frame_status = irmf.groupby("frame_id")["n_unconverged"].max().gt(0).rename("any_unconverged")
    frame_status.to_frame().to_parquet(CACHE / "frame_convergence.parquet")
    class_rows = []
    for label, cell in merged.groupby("y_10class", sort=True):
        status = frame_status.loc[meta.loc[meta["y_10class"] == label, "frame_id"]]
        class_rows.append({"class": label, "n_frames": len(status),
                           "frame_any_unconverged_fraction": status.mean(),
                           "channel_any_unconverged_fraction": cell["n_unconverged"].gt(0).mean(),
                           "stage_unconverged_fraction": np.mean(~converged[cell.index])})
    pd.DataFrame(class_rows).to_csv(OUT / "convergence_classes.csv", index=False)
    costs = []
    for tag in TAGS:
        if tag == "raw":
            costs.append({"method": tag, "cpu_seconds_per_two_band_frame": 0.0,
                          "decomposition_wall_seconds": 0.0, "failed": 0})
            continue
        summary = json.loads((DATA / f"decomp/{tag}.json").read_text())
        timing = pd.read_parquet(DATA / f"decomp/{tag}.meta.parquet")
        row = {"method": tag, "cpu_seconds_per_two_band_frame": summary["seconds_per_frame"]["cpu_mean"],
               "wall_seconds_per_two_band_frame": summary["seconds_per_frame"]["wall_mean"],
               "decomposition_wall_seconds": summary["wall_seconds_total"],
               "cpu_seconds_total": summary["seconds_per_frame"]["cpu_total"],
               "failed": summary["n_failed"]}
        for channel in [0, 1]:
            row[f"channel_{channel}_cpu_seconds"] = timing.loc[timing.channel == channel, "cpu_seconds"].mean()
        costs.append(row)
    pd.DataFrame(costs).to_csv(OUT / "decomposition_costs.csv", index=False)
    raw = pd.read_parquet(DATA / "features/raw.parquet")
    confounds = meta[["y_10class", "activity_db", "activity_db_L", "activity_db_H", "rms_L", "rms_H"]].copy()
    confounds["log_scale_L"] = raw["ch0_log_scale"]
    confounds["log_scale_H"] = raw["ch1_log_scale"]
    confounds.groupby("y_10class").median().to_csv(OUT / "power_snr_by_class.csv")
    (OUT / "convergence_overall.json").write_text(json.dumps({
        "n_frames": len(meta), "unconverged_frames": int(frame_status.sum()),
        "unconverged_frame_fraction": float(frame_status.mean()),
        "unconverged_channel_fraction": float(irmf.n_unconverged.gt(0).mean()),
        "unconverged_stage_fraction": float((~converged).mean()),
        "unconverged_not_at_cap": int(((~converged) & (iterations != 60)).sum()),
    }, indent=2))
    print("static diagnostics saved", flush=True)


def summarize(folds):
    summaries = []
    for key, cell in folds.groupby(["task", "feature_set", "method", "classifier"], sort=False):
        row = dict(zip(["task", "feature_set", "method", "classifier"], key))
        for metric in METRICS:
            result = corrected_resampled_ttest(cell[metric], cell.n_train, cell.n_test)
            row[metric] = result["mean_diff"]
            row[f"{metric}_ci_low"] = result["ci_low"]
            row[f"{metric}_ci_high"] = result["ci_high"]
        row["fit_seconds_total"] = cell.fit_seconds.sum()
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "all_classifier_metrics.csv", index=False)
    best = (summary.sort_values("balanced_accuracy", ascending=False)
            .drop_duplicates(["task", "feature_set", "method"])
            .sort_values(["task", "method", "feature_set"]))
    best.to_csv(OUT / "best_metrics.csv", index=False)
    comparisons = []
    for task, cell in folds.groupby("task", sort=False):
        long = cell.melt(id_vars=["method", "feature_set", "classifier", "split_id", "n_train", "n_test"],
                         value_vars=METRICS, var_name="metric", value_name="value")
        for metric in METRICS:
            table = compare_methods(long, metric=metric, by=("feature_set", "classifier"))
            table.insert(0, "task", task)
            comparisons.append(table)
    comparisons = pd.concat(comparisons, ignore_index=True)
    comparisons.to_csv(OUT / "paired_all_metrics.csv", index=False)
    chosen = choose_classifiers(folds)
    primary = comparisons[[row.classifier == chosen[row.task] for row in comparisons.itertuples()]]
    primary.to_csv(OUT / "paired_selected_classifier.csv", index=False)
    recall_rows = []
    for task, task_folds in folds.groupby("task", sort=False):
        recalls = [name for name in task_folds if name.startswith("recall_") and task_folds[name].notna().any()]
        for (feature_set, classifier), cell in task_folds.groupby(["feature_set", "classifier"]):
            table_rows = []
            for tag_a, tag_b in baseline.default_pairs(TAGS):
                a = cell[cell.method == tag_a].set_index("split_id")
                b = cell[cell.method == tag_b].set_index("split_id").loc[a.index]
                for name in recalls:
                    result = paired_comparison(a[name], b[name], a.n_train, a.n_test)
                    table_rows.append({"task": task, "feature_set": feature_set, "classifier": classifier,
                                       "class": name[7:], "method_a": tag_a, "method_b": tag_b, **result})
            table = pd.DataFrame(table_rows)
            recall_rows.append(table)
    recall_table = pd.concat(recall_rows, ignore_index=True)
    recall_table["p_nb_holm"] = recall_table.groupby("task")["p_nb"].transform(lambda x: holm(x))
    recall_table["p_wilcoxon_holm"] = recall_table.groupby("task")["p_wilcoxon"].transform(lambda x: holm(x))
    recall_table.to_csv(OUT / "paired_class_recall.csv", index=False)
    return summary, best, comparisons, chosen


def aggregate_scales():
    all_scales = []
    for task in TASKS:
        disc = pd.read_csv(OUT / f"discriminability_{task}.csv")
        disc["task"] = task
        disc["scale_rank"] = [(-1 if row.slot < 0 else scale_order(row.method, 2048).index(row.slot))
                              for row in disc.itertuples()]
        all_scales.append(disc)
    disc = pd.concat(all_scales, ignore_index=True)
    agg = disc.groupby(["task", "method", "channel", "slot", "slot_label", "scale_rank"]).agg(
        epsilon2_mean=("epsilon2", "mean"), epsilon2_median=("epsilon2", "median"),
        epsilon2_max=("epsilon2", "max"), mi_mean=("mi", "mean"),
        mi_median=("mi", "median"), mi_max=("mi", "max"), n_features=("feature", "size"),
    ).reset_index().sort_values(["task", "method", "channel", "scale_rank"])
    agg.to_csv(OUT / "discriminability_scales.csv", index=False)
    disc.sort_values(["task", "method", "mi"], ascending=[True, True, False]).groupby(
        ["task", "method"]).head(20).to_csv(OUT / "top_discriminative_features.csv", index=False)
    return agg


def band_comparisons(folds, chosen):
    bands = pd.read_csv(OUT / "band_folds.csv")
    rows, summaries = [], []
    for (task, tag, feature_set), cell in bands[bands.band == "L"].groupby(["task", "method", "feature_set"]):
        both = folds[(folds.task == task) & (folds.method == tag)
                     & (folds.feature_set == feature_set) & (folds.classifier == chosen[task])].set_index("split_id")
        left = cell.set_index("split_id").loc[both.index]
        for metric in METRICS:
            rows.append({"task": task, "method": tag, "feature_set": feature_set,
                         "classifier": chosen[task], "metric": metric,
                         **paired_comparison(left[metric], both[metric], both.n_train, both.n_test)})
        summaries.append({"task": task, "method": tag, "feature_set": feature_set,
                          "classifier": chosen[task],
                          **{f"L_{metric}": left[metric].mean() for metric in METRICS},
                          **{f"both_{metric}": both[metric].mean() for metric in METRICS}})
    table = pd.DataFrame(rows)
    table["p_nb_holm"] = table.groupby(["task", "metric"])["p_nb"].transform(lambda x: holm(x))
    table["p_wilcoxon_holm"] = table.groupby(["task", "metric"])["p_wilcoxon"].transform(lambda x: holm(x))
    table.to_csv(OUT / "paired_band_comparisons.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "band_metrics.csv", index=False)
    bands[bands.feature_set == "log_scale_only"].groupby(["task", "classifier"])[METRICS].mean().to_csv(
        OUT / "log_scale_only_metrics.csv")
    return table


def convergence_accuracy(chosen):
    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    status = pd.read_parquet(CACHE / "frame_convergence.parquet")
    all_rows, class_rows = [], []
    for task in TASKS:
        pred = pd.read_parquet(DATA / f"results/classical_{task}_predictions.parquet")
        pred = pred[pred.classifier == chosen[task]].merge(status, on="frame_id", validate="many_to_one")
        for key, cell in pred.groupby(["method", "feature_set", "split_id"], observed=True):
            labels = sorted(set(cell.y_true.astype(str)))
            subcells = {level: cell[cell.any_unconverged == level] for level in [False, True]}
            common = sorted(set(subcells[False].y_true.astype(str)) & set(subcells[True].y_true.astype(str)))
            for level, sub in subcells.items():
                y, yp = sub.y_true.astype(str).to_numpy(), sub.y_pred.astype(str).to_numpy()
                metrics = split_metrics(y, yp, labels)
                common_ba = np.mean([metrics[f"recall_{label}"] for label in common])
                all_rows.append({"task": task, "method": key[0], "feature_set": key[1],
                                 "split_id": key[2], "classifier": chosen[task],
                                 "any_unconverged": level, "n_test": len(sub),
                                 "n_present_classes": len(set(y)), "n_common_classes": len(common),
                                 "common_class_balanced_accuracy": common_ba,
                                 **{metric: metrics[metric] for metric in METRICS}})
                for label in labels:
                    class_rows.append({"task": task, "method": key[0], "feature_set": key[1],
                                       "split_id": key[2], "classifier": chosen[task],
                                       "any_unconverged": level, "class": label,
                                       "n_test": int(np.sum(y == label)), "recall": metrics[f"recall_{label}"]})
    pd.DataFrame(all_rows).to_csv(OUT / "convergence_accuracy_folds.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "convergence_class_recall.csv", index=False)


def make_figures(summary, scales, folds, chosen):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    classifiers = ["lda", "qda", "gnb", "logreg", "knn", "linsvm"]
    for row, task in enumerate(TASKS):
        for col, fs in enumerate(FEATURE_SETS):
            ax = axes[row, col]
            for j, tag in enumerate(TAGS):
                cell = summary[(summary.task == task) & (summary.feature_set == fs) & (summary.method == tag)]
                cell = cell.set_index("classifier").loc[classifiers]
                means = 100 * cell.balanced_accuracy.to_numpy()
                errs = np.vstack([means - 100 * cell.balanced_accuracy_ci_low.to_numpy(),
                                  100 * cell.balanced_accuracy_ci_high.to_numpy() - means])
                ax.errorbar(np.arange(6) + (j - 1.5) * 0.16, means, yerr=errs,
                            fmt="o", ms=4, capsize=2, lw=0.8, color=COLORS[j], label=NAMES[tag])
            ax.set_title(f"{task}: {'without' if col == 0 else 'with'} log_scale")
            ax.set_ylabel("Balanced accuracy (%)")
            ax.set_xticks(range(6), [name.upper() for name in classifiers])
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=min(102, ax.get_ylim()[1]))
    axes[0, 0].legend(ncol=4, loc="lower center", bbox_to_anchor=(1.03, 1.17), frameon=False)
    fig.suptitle("DroneRF: all 15 segment-held-out splits; corrected 95% mean intervals", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "figures/balanced_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey="row")
    for col, tag in enumerate(TAGS):
        for channel in [0, 1]:
            cell = scales[(scales.task == "10class") & (scales.method == tag)
                          & (scales.channel == channel) & (scales.slot >= 0)].sort_values("scale_rank")
            for row, metric in enumerate(["epsilon2_mean", "mi_mean"]):
                axes[row, col].plot(range(len(cell)), cell[metric], marker=".", lw=1.3,
                                    label=["L", "H"][channel])
                axes[row, col].set_xticks(range(len(cell)), cell.slot_label, rotation=75, ha="right", fontsize=7)
                axes[row, col].grid(axis="y", alpha=0.2)
        axes[0, col].set_title(NAMES[tag])
        axes[1, col].set_xlabel("Fine to coarse (method-specific scales)")
    axes[0, 0].set_ylabel("Mean Kruskal-Wallis epsilon²")
    axes[1, 0].set_ylabel("Mean mutual information (nats)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("10-class discriminability: mean over 20 features per scale, descriptive all-frame statistics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "figures/discriminability_scales.png", dpi=180)
    plt.close(fig)

    stage = pd.read_csv(OUT / "convergence_stages.csv", dtype={"class": str, "channel": str})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), sharey=True)
    classes = sorted(stage.loc[stage["class"] != "all", "class"].unique())
    order = slot_labels("irmf_h2_a1.41", 2048)[:-1]
    color_limit = max(5.0, 5 * np.ceil(100 * stage.unconverged_fraction.max() / 5))
    for channel, ax in enumerate(axes):
        cell = stage[(stage.channel == str(channel)) & (stage["class"] != "all")]
        matrix = cell.pivot(index="class", columns="stage", values="unconverged_fraction").reindex(
            index=classes, columns=order)
        im = ax.imshow(100 * matrix, aspect="auto", vmin=0, vmax=color_limit, cmap="magma")
        ax.set_xticks(range(len(order)), order, rotation=70)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_title(f"IRMF unconverged stages: {'L' if channel == 0 else 'H'} band")
        ax.set_xlabel("Fine to coarse extracted stages")
    axes[0].set_ylabel("BUI / recording session")
    fig.subplots_adjust(left=0.07, bottom=0.22, right=0.87, wspace=0.12, top=0.90)
    color_ax = fig.add_axes([0.89, 0.28, 0.015, 0.53])
    fig.colorbar(im, cax=color_ax, label="Unconverged (%)")
    fig.savefig(OUT / "figures/convergence.png", dpi=180)
    plt.close(fig)



def write_tables(summary, best, comparisons, scales, chosen):
    lines = ["# DroneRF numerical tables", "",
             "All scores are percentages unless marked otherwise. Differences are percentage points.",
             "Method names below abbreviate the four main tags. See METHODS.md for selection and Holm families.", ""]
    display = best[["task", "feature_set", "method", "classifier"] + METRICS].copy()
    display["method"] = display.method.map(NAMES)
    for metric in METRICS:
        display[metric] *= 100
    lines += ["## Best classifier per method, descriptive", "", display.to_markdown(index=False, floatfmt=".3f"), ""]
    selected = comparisons[[r.classifier == chosen[r.task] for r in comparisons.itertuples()]].copy()
    selected["pair"] = selected.method_a.map(NAMES) + " minus " + selected.method_b.map(NAMES)
    selected["difference, 95% CI"] = [f"{100*r.mean_diff:+.3f} [{100*r.ci_low:+.3f}, {100*r.ci_high:+.3f}]"
                                          for r in selected.itertuples()]
    for task in TASKS:
        lines += [f"## {task} paired comparisons, common classifier {chosen[task]}", "",
                  "The p-values retain the 72-test task/metric Holm family across every classifier and both feature sets.", ""]
        for metric in METRICS:
            c = selected[(selected.task == task) & (selected.metric == metric)]
            lines += [f"### {metric}", "", c[["feature_set", "pair", "difference, 95% CI", "p_nb", "p_nb_holm",
                                                          "p_wilcoxon", "p_wilcoxon_holm"]].to_markdown(
                                                              index=False, floatfmt=".5g"), ""]
    cell = comparisons[(comparisons.method_a == TAGS[3]) & (comparisons.method_b == "emd")
                       & (comparisons.metric == "balanced_accuracy")].copy()
    cell["mean_diff_pp"] = 100 * cell.mean_diff
    lines += ["## IRMF minus EMD by classifier", "", cell[["task", "feature_set", "classifier", "mean_diff_pp",
                  "p_nb", "p_nb_holm", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""]
    recall = pd.read_csv(OUT / "paired_class_recall.csv", dtype={"class": str})
    recall = recall[(recall.task == "10class") & (recall.classifier == chosen["10class"])
                    & (recall.method_a == TAGS[3]) & (recall.method_b == "emd")].copy()
    recall["class"] = "BUI " + recall["class"]
    for col in ["mean_a", "mean_b", "mean_diff", "ci_low", "ci_high"]:
        recall[col] *= 100
    lines += ["## 10-class recall, IRMF versus EMD", "", recall[["feature_set", "class", "mean_a", "mean_b",
                  "mean_diff", "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm"]].to_markdown(
                      index=False, floatfmt=".4g"), ""]
    top = scales[(scales.task == "10class") & (scales.slot >= 0)].sort_values("mi_mean", ascending=False)
    top = top.groupby(["method", "channel"]).head(3).copy()
    top["method"] = top.method.map(NAMES)
    lines += ["## Most informative 10-class scales", "",
              "Sorted by mean MI across 20 features per slot; channel 0=L and 1=H. Descriptive, all frames.", "",
              top[["method", "channel", "slot_label", "epsilon2_mean", "mi_mean"]].to_markdown(
                  index=False, floatfmt=".4f"), ""]
    bands = pd.read_csv(OUT / "paired_band_comparisons.csv")
    bands = bands[bands.metric == "balanced_accuracy"].copy()
    bands["method"] = bands.method.map(NAMES)
    bands["L_BA"] = 100 * bands.mean_a
    bands["both_BA"] = 100 * bands.mean_b
    bands["difference, 95% CI"] = [f"{100*r.mean_diff:+.3f} [{100*r.ci_low:+.3f}, {100*r.ci_high:+.3f}]"
                                      for r in bands.itertuples()]
    lines += ["## L-only versus both views", "", bands[["task", "feature_set", "method", "classifier", "L_BA",
                  "both_BA", "difference, 95% CI", "p_nb", "p_nb_holm", "p_wilcoxon_holm"]].to_markdown(
                      index=False, floatfmt=".5g"), ""]
    control = pd.read_csv(OUT / "log_scale_only_metrics.csv")
    control[METRICS] *= 100
    lines += ["## Log-scale-only control", "", control.to_markdown(index=False, floatfmt=".3f"), ""]
    conv = pd.read_csv(OUT / "convergence_accuracy_folds.csv")
    conv = conv[conv.method == TAGS[3]].groupby(["task", "feature_set", "any_unconverged"])[
        METRICS + ["common_class_balanced_accuracy", "n_test", "n_common_classes"]].mean().reset_index()
    conv[METRICS + ["common_class_balanced_accuracy"]] *= 100
    lines += ["## IRMF accuracy by convergence status", "",
              "Means over splits. n_test is the mean subset size per fold; common-class BA uses classes present in both statuses.",
              "This association is descriptive, since status changes which frames are evaluated.", "",
              conv.to_markdown(index=False, floatfmt=".3f"), ""]
    costs = pd.read_csv(OUT / "decomposition_costs.csv")
    lines += ["## Cached decomposition cost", "", costs.to_markdown(index=False, floatfmt=".5g"), "",
              "Stage-by-class rates: convergence_stages.csv. Frame-level class rates: convergence_classes.csv.", "",
              "Complete classifier/metric comparisons: paired_all_metrics.csv. All per-class pairs: paired_class_recall.csv.", ""]
    (OUT / "tables.md").write_text("\n".join(lines))



def validate_outputs(folds, chosen):
    import hashlib
    from importlib.metadata import version
    import platform

    meta = pd.read_parquet(DATA / "prepared/meta.parquet")
    keys = ["method", "feature_set", "classifier", "split_id"]
    scale_columns = ["frame_id", "ch0_log_scale", "ch1_log_scale"]
    reference_scale = pd.read_parquet(DATA / "features/raw.parquet", columns=scale_columns)
    assert np.array_equal(reference_scale.frame_id, meta.frame_id)
    for tag in TAGS:
        scale = pd.read_parquet(DATA / f"features/{tag}.parquet", columns=scale_columns)
        np.testing.assert_array_equal(scale.to_numpy(), reference_scale.to_numpy())
    assert not folds.duplicated(["task"] + keys).any()
    assert np.isfinite(folds[METRICS].to_numpy()).all()
    assert ((folds[METRICS] >= 0) & (folds[METRICS] <= 1)).all().all()
    checks = []
    for task in TASKS:
        splits = load_splits("dronerf", task)  # Also checks disjoint groups and complete partitions.
        assert len(splits) == 15
        y_all = meta[f"y_{task}"].astype(str).to_numpy()
        expected_classes = set(y_all)
        assert all(set(y_all[split.train]) == expected_classes and set(y_all[split.test]) == expected_classes
                   for split in splits)
        pred = pd.read_parquet(DATA / f"results/classical_{task}_predictions.parquet")
        assert len(pred) == len(meta) * 3 * 4 * 2 * 6
        coverage = pred.groupby(["method", "feature_set", "classifier", "seed"], observed=True)["frame_id"].agg(
            ["size", "nunique"])
        assert (coverage == len(meta)).all().all()
        expected_y = meta[f"y_{task}"].astype(str).to_numpy()[pred.frame_id.to_numpy()]
        assert np.array_equal(expected_y, pred.y_true.astype(str).to_numpy())
        selected = pred[pred.classifier == chosen[task]]
        reference = folds[folds.task == task].set_index(keys)
        n_checked = 0
        for key, cell in selected.groupby(keys, observed=True, sort=False):
            actual = split_metrics(cell.y_true.astype(str).to_numpy(), cell.y_pred.astype(str).to_numpy(),
                                   sorted(meta[f"y_{task}"].unique()))
            np.testing.assert_allclose([actual[m] for m in METRICS], reference.loc[key, METRICS].to_numpy(dtype=float),
                                       atol=1e-12, rtol=0)
            n_checked += 1
        checks.append({"task": task, "prediction_rows": len(pred), "n_classifiers": 6,
                       "n_outer_rows": int((folds.task == task).sum()), "recomputed_metric_rows": n_checked,
                       "seed_frame_coverage_complete": True, "groups_disjoint": True})

    bands = pd.read_csv(OUT / "band_folds.csv")
    band_checks = []
    band_keys = ["method", "feature_set", "band", "classifier", "split_id"]
    assert not bands.duplicated(["task"] + band_keys).any()
    for task, reference in bands.groupby("task", sort=False):
        assert len(reference) == 135
        pred = pd.read_parquet(CACHE / f"band_predictions_{task}.parquet")
        assert len(pred) == len(meta) * 3 * 9
        coverage = pred.groupby(["method", "feature_set", "band", "seed"], observed=True)["frame_id"].agg(
            ["size", "nunique"])
        assert (coverage == len(meta)).all().all()
        expected_y = meta[f"y_{task}"].astype(str).to_numpy()[pred.frame_id.to_numpy()]
        assert np.array_equal(expected_y, pred.y_true.astype(str).to_numpy())
        reference = reference.set_index(band_keys)
        n_checked = 0
        for key, cell in pred.groupby(band_keys, observed=True, sort=False):
            actual = split_metrics(cell.y_true.astype(str).to_numpy(), cell.y_pred.astype(str).to_numpy(),
                                   sorted(meta[f"y_{task}"].unique()))
            np.testing.assert_allclose([actual[m] for m in METRICS], reference.loc[key, METRICS].to_numpy(dtype=float),
                                       atol=1e-12, rtol=0)
            n_checked += 1
        assert n_checked == 135
        band_checks.append({"task": task, "prediction_rows": len(pred),
                            "recomputed_metric_rows": n_checked, "seed_frame_coverage_complete": True})

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    files = [Path(__file__), STUDY_DIR / "baselines/run_classical.py",
             Path(__file__).with_name("run_classical_logged.py"),
             Path(__file__).with_name("benchmark_svm_solver.py"), DATA / "prepared/meta.parquet",
             DATA / "prepared/dataset.json"] + [DATA / f"features/{tag}.parquet" for tag in TAGS]
    (OUT / "validation.json").write_text(json.dumps({
        "checks": checks, "band_checks": band_checks, "python": platform.python_version(),
        "shared_log_scale_identical_across_methods": True,
        "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "pimf", "EMD-signal"]},
        "sha256": {str(path): digest(path) for path in files},
        "selection": chosen,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--bands", action="store_true")
    parser.add_argument("--band-tasks", nargs="+", choices=TASKS, default=["10class"],
                        help="band sensitivity task(s); primary 10class by default")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.static:
        static_diagnostics()
    if args.bands:
        run_bands(args.band_tasks)
    if args.report:
        report_started = time.perf_counter()
        folds = load_folds()
        assert len(folds) == 2160
        assert (folds.groupby(["task", "method", "feature_set", "classifier"]).size() == 15).all()
        summary, best, comparisons, chosen = summarize(folds)
        scales = aggregate_scales()
        band_comparisons(folds, chosen)
        convergence_accuracy(chosen)
        make_figures(summary, scales, folds, chosen)
        write_tables(summary, best, comparisons, scales, chosen)
        validate_outputs(folds, chosen)
        (OUT / "analysis_run.json").write_text(json.dumps({
            "seconds": time.perf_counter() - report_started, "outer_rows": len(folds),
            "n_figures": 3, "validation": "validation.json",
        }, indent=2))
        print("Analyses and three figures saved", flush=True)


if __name__ == "__main__":
    main()
