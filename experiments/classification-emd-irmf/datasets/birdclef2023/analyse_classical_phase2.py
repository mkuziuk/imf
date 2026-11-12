"""BirdCLEF-only phase-2 analysis. Numerical caches stay under DATA_ROOT.

Logistic regression is the common classifier for the compact paired verdict
and recording check. It was chosen before inspecting decomposition results.
Best-of-six results are descriptive, with no classifier-selection adjustment.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = STUDY_DIR.parents[1]
DATA_DIR = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification")) / "birdclef2023"
OUT_DIR = ROOT_DIR / "output" / "classification-emd-irmf" / "birdclef2023" / "classical"
sys.path.insert(0, str(STUDY_DIR))
from emdirmf.decompose import scale_order, slot_labels, slot_window_sizes
from emdirmf.stats import compare_methods, default_pairs, holm, paired_comparison
from emdirmf.cv import split_metrics

METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
SHORT = dict(zip(METHODS, ["Raw", "EMD", "Linear IMF", "IRMF"]))
FEATURE_SETS = ["comp", "comp+log_scale"]
CLASSIFIERS = ["lda", "qda", "gnb", "logreg", "knn", "linsvm"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
COLORS = ["#777777", "#3274a1", "#e29a36", "#3a9160"]
PAIRS = default_pairs(METHODS)
FS = 22050
LENGTH = 2048


def save_table(table, name):
    table.to_csv(OUT_DIR / f"{name}.csv", index=False)


def compare_all(folds):
    tables = []
    for task, cell in folds.groupby("task", sort=False):
        long = cell.melt(id_vars=["method", "feature_set", "classifier", "split_id", "n_train", "n_test"],
                         value_vars=METRICS, var_name="metric", value_name="value")
        for metric in METRICS:
            comparisons = compare_methods(long, metric=metric, pairs=PAIRS,
                                          by=("feature_set", "classifier"))
            comparisons.insert(0, "task", task)
            comparisons["holm_family_size"] = len(comparisons)
            tables.append(comparisons)
    comparisons = pd.concat(tables, ignore_index=True)
    save_table(comparisons, "paired_all_metrics")
    primary = comparisons[(comparisons["classifier"] == "logreg") &
                          (comparisons["metric"] == "balanced_accuracy")]
    save_table(primary, "paired_logreg_balanced_accuracy")
    return comparisons, primary


def summarize_folds(folds):
    summary = folds.groupby(["task", "feature_set", "method", "classifier"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).strip("_") for x in summary.columns]
    save_table(summary, "all_classifier_metrics")
    best = summary.sort_values("balanced_accuracy_mean", ascending=False).drop_duplicates(["task", "feature_set", "method"])
    save_table(best.sort_values(["task", "feature_set", "method"]), "best_classifier_metrics")
    ablations = []
    for (task, method, classifier), cell in folds.groupby(["task", "method", "classifier"]):
        a = cell[cell["feature_set"] == FEATURE_SETS[1]].set_index("split_id").sort_index()
        b = cell[cell["feature_set"] == FEATURE_SETS[0]].set_index("split_id").sort_index()
        for metric in METRICS:
            row = dict(task=task, method=method, classifier=classifier, metric=metric)
            row.update(paired_comparison(a[metric], b[metric], a["n_train"], a["n_test"]))
            ablations.append(row)
    ablations = pd.DataFrame(ablations)
    ablations["p_nb_holm"] = ablations.groupby(["task", "metric"])["p_nb"].transform(holm)
    ablations["p_wilcoxon_holm"] = ablations.groupby(["task", "metric"])["p_wilcoxon"].transform(holm)
    save_table(ablations, "log_scale_ablation")
    return summary, best


def per_class(folds):
    cell = folds[folds["task"] == "species"]
    labels = sorted(column[7:] for column in cell if column.startswith("recall_") and cell[column].notna().any())
    rows = []
    for (feature_set, classifier), table in cell.groupby(["feature_set", "classifier"]):
        for a, b in PAIRS:
            x = table[table["method"] == a].set_index("split_id").sort_index()
            z = table[table["method"] == b].set_index("split_id").sort_index()
            for label in labels:
                column = f"recall_{label}"
                keep = x[column].notna() & z[column].notna()
                row = dict(feature_set=feature_set, classifier=classifier, species=label,
                           method_a=a, method_b=b)
                row.update(paired_comparison(x.loc[keep, column], z.loc[keep, column],
                                            x.loc[keep, "n_train"], x.loc[keep, "n_test"]))
                rows.append(row)
    results = pd.DataFrame(rows)
    results["p_nb_holm"] = holm(results["p_nb"])
    results["p_wilcoxon_holm"] = holm(results["p_wilcoxon"])
    save_table(results, "per_species_recall_comparisons")
    return results


def scale_analysis(task):
    disc = pd.read_csv(OUT_DIR / f"discriminability_{task}.csv")
    rows = []
    for method in METHODS:
        table = pd.read_parquet(DATA_DIR / "features" / f"{method}.parquet")
        labels = slot_labels(method, LENGTH)
        cached_labels = json.loads((DATA_DIR / "features" / f"{method}.json").read_text())["slot_labels"]
        assert labels == cached_labels, f"Component labels changed: {method}"
        windows = slot_window_sizes(method, LENGTH)
        for rank, slot in enumerate(scale_order(method, LENGTH)):
            cell = disc[(disc["method"] == method) & (disc["slot"] == slot)]
            keep = table[f"ch0_c{slot}_energy_frac"].to_numpy() > 1e-12
            centroid = FS * table.loc[keep, f"ch0_c{slot}_spec_centroid"].to_numpy()
            peak = FS * table.loc[keep, f"ch0_c{slot}_spec_peak"].to_numpy()
            row = dict(task=task, method=method, scale_rank=rank, slot=slot, slot_label=labels[slot],
                       n_features=len(cell), n_nonzero=int(keep.sum()),
                       epsilon2_mean=cell["epsilon2"].mean(), epsilon2_median=cell["epsilon2"].median(),
                       epsilon2_max=cell["epsilon2"].max(), mi_mean=cell["mi"].mean(),
                       mi_median=cell["mi"].median(), mi_max=cell["mi"].max(),
                       centroid_hz_median=float(np.median(centroid)) if len(centroid) else np.nan,
                       centroid_hz_q10=float(np.quantile(centroid, 0.1)) if len(centroid) else np.nan,
                       centroid_hz_q90=float(np.quantile(centroid, 0.9)) if len(centroid) else np.nan,
                       peak_hz_median=float(np.median(peak)) if len(peak) else np.nan,
                       nominal_fs_over_window_hz=FS / windows[slot] if windows is not None and slot < len(windows) else np.nan)
            rows.append(row)
    results = pd.DataFrame(rows)
    save_table(results, f"scale_discriminability_{task}")
    return results


def convergence_analysis(meta, folds):
    diag = pd.read_parquet(DATA_DIR / "decomp" / "irmf_h2_a1.41.meta.parquet").sort_values("frame_id")
    assert len(diag) == len(meta) and diag["frame_id"].is_unique
    assert np.array_equal(diag["frame_id"], meta["frame_id"])
    stages = np.stack(diag["stage_converged"].to_numpy())
    iterations = np.stack(diag["stage_iterations"].to_numpy())
    frame_converged = stages.all(axis=1)
    assert np.array_equal((~stages).sum(axis=1), diag["n_unconverged"])
    windows = slot_window_sizes("irmf_h2_a1.41", LENGTH)
    stage_table = pd.DataFrame({
        "slot": np.arange(stages.shape[1]), "slot_label": slot_labels("irmf_h2_a1.41", LENGTH)[:-1],
        "window_samples": windows, "converged_fraction": stages.mean(axis=0),
        "iteration_cap_fraction": (iterations == 60).mean(axis=0),
        "mean_iterations": iterations.mean(axis=0), "median_iterations": np.median(iterations, axis=0),
    })
    save_table(stage_table, "irmf_stage_convergence")
    counts = []
    with_scale = pd.read_parquet(DATA_DIR / "features" / "raw.parquet", columns=["ch0_log_scale"])
    for label, cell in meta.assign(converged=frame_converged, log_scale=with_scale["ch0_log_scale"]).groupby(["y_species", "converged"]):
        counts.append(dict(species=label[0], converged=label[1], n_frames=len(cell),
                           median_rms_dbfs=cell["frame_rms_dbfs"].median(), median_log_scale=cell["log_scale"].median()))
    save_table(pd.DataFrame(counts), "convergence_species_composition")
    rows = []
    for task in ("species", "passerine"):
        pred = pd.read_parquet(DATA_DIR / "results" / f"classical_{task}_predictions.parquet")
        assert len(pred) == 1728000
        assert not pred.duplicated(["method", "feature_set", "classifier", "seed", "frame_id"]).any()
        assert pred.groupby(["method", "feature_set", "classifier", "seed"], observed=True)["frame_id"].nunique().eq(len(meta)).all()
        pred["converged"] = frame_converged[pred["frame_id"].to_numpy()]
        labels = sorted(meta[f"y_{task}"].unique())
        for key, cell in pred.groupby(["method", "feature_set", "classifier", "split_id", "seed", "fold", "converged"], observed=True):
            row = dict(zip(["method", "feature_set", "classifier", "split_id", "seed", "fold", "converged"], key))
            row.update(task=task, n_frames=len(cell), n_classes=cell["y_true"].nunique())
            row.update(split_metrics(cell["y_true"].to_numpy(), cell["y_pred"].to_numpy(), labels))
            rows.append(row)
    results = pd.DataFrame(rows)
    save_table(results, "convergence_subset_folds")
    grouped = results.groupby(["task", "method", "feature_set", "classifier", "converged"])[METRICS].mean().reset_index()
    save_table(grouped, "convergence_subset_metrics")
    converged_scores = grouped[grouped["converged"]].drop(columns="converged")
    unconverged_scores = grouped[~grouped["converged"]].drop(columns="converged")
    keys = ["task", "method", "feature_set", "classifier"]
    gaps = converged_scores.merge(unconverged_scores, on=keys, suffixes=("_converged", "_unconverged"))
    for metric in METRICS:
        gaps[f"{metric}_gap"] = gaps[f"{metric}_converged"] - gaps[f"{metric}_unconverged"]
    save_table(gaps, "convergence_subset_gaps")
    stats = dict(n_frames=len(diag), n_all_converged=int(frame_converged.sum()),
                 n_any_unconverged=int((~frame_converged).sum()),
                 unconverged_stage_fraction=float((~stages).mean()),
                 mean_unconverged_stages=float((~stages).sum(axis=1).mean()),
                 mean_total_iterations=float(iterations.sum(axis=1).mean()),
                 n_failed=int((~diag["ok"]).sum()))
    (OUT_DIR / "convergence.json").write_text(json.dumps(stats, indent=2))
    return grouped, stage_table, stats


def recording_analysis(meta):
    rows, pooled_rows = [], []
    cache_dir = DATA_DIR / "results" / "classical_phase2" / "narrow"
    recording_cache = DATA_DIR / "results" / "classical_phase2" / "recording_probabilities"
    recording_cache.mkdir(parents=True, exist_ok=True)
    for task in ("species", "passerine"):
        frames = []
        paths = sorted((cache_dir / task).glob("*.npz"))
        assert len(paths) == 720, (task, len(paths))
        for path in paths:
            method, feature_set, classifier, split_id = path.stem.split("__")
            if classifier == "linsvm":
                continue
            with np.load(path, allow_pickle=False) as cached:
                frame_ids = cached["frame_id"]
                probs = cached["probabilities"]
                classes = cached["classes"]
                row = json.loads(str(cached["row"]))
                assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1)
                assert np.array_equal(classes[np.argmax(probs, axis=1)], cached["y_pred"])
                cell = pd.DataFrame(probs, columns=classes)
                cell["recording_id"] = meta.iloc[frame_ids]["recording_id"].to_numpy()
                cell["y_true"] = cached["y_true"]
                assert cell.groupby("recording_id")["y_true"].nunique().max() == 1
                rec = cell.groupby("recording_id")[list(classes)].mean()
                rec["y_true"] = cell.groupby("recording_id")["y_true"].first()
                rec["y_pred"] = classes[np.argmax(rec[list(classes)].to_numpy(), axis=1)]
                rec["method"], rec["feature_set"], rec["classifier"] = method, feature_set, classifier
                rec["seed"], rec["fold"], rec["split_id"] = row["seed"], row["fold"], int(split_id)
                frames.append(rec.reset_index())
                result = dict(task=task, method=method, feature_set=feature_set, classifier=classifier,
                              split_id=int(split_id), seed=row["seed"], fold=row["fold"],
                              n_recordings=len(rec), frame_balanced_accuracy=row["balanced_accuracy"],
                              frame_macro_f1=row["macro_f1"], frame_accuracy=row["accuracy"])
                result.update(split_metrics(rec["y_true"].to_numpy(), rec["y_pred"].to_numpy(), classes))
                rows.append(result)
        recordings = pd.concat(frames, ignore_index=True)
        recordings.to_parquet(recording_cache / f"{task}.parquet", index=False)
        assert not recordings.duplicated(["method", "feature_set", "classifier", "seed", "recording_id"]).any()
        for key, cell in recordings.groupby(["method", "feature_set", "classifier", "seed"], observed=True):
            assert len(cell) == 1280
            row = dict(zip(["method", "feature_set", "classifier", "seed"], key))
            row.update(task=task, n_recordings=len(cell))
            row.update(split_metrics(cell["y_true"].to_numpy(), cell["y_pred"].to_numpy(), sorted(meta[f"y_{task}"].unique())))
            pooled_rows.append(row)
    folds = pd.DataFrame(rows)
    save_table(folds, "recording_results_folds")
    save_table(pd.DataFrame(pooled_rows), "recording_results_pooled_seeds")
    summary = folds.groupby(["task", "method", "feature_set", "classifier"])[METRICS + [f"frame_{m}" for m in METRICS]].mean().reset_index()
    save_table(summary, "recording_vs_frame_metrics")
    save_table(summary[summary["classifier"] == "logreg"], "recording_logreg_metrics")
    return summary


def cost_analysis():
    rows = []
    for method in METHODS:
        if method == "raw":
            rows.append(dict(method=method, cpu_s_per_frame=0.0, wall_s_per_frame=0.0,
                             cpu_total_s=0.0, decomposition_wall_s=0.0, n_failed=0, slots=1))
            continue
        info = json.loads((DATA_DIR / "decomp" / f"{method}.json").read_text())
        rows.append(dict(method=method, cpu_s_per_frame=info["seconds_per_frame"]["cpu_mean"],
                         wall_s_per_frame=info["seconds_per_frame"]["wall_mean"],
                         cpu_total_s=info["seconds_per_frame"]["cpu_total"],
                         decomposition_wall_s=info["wall_seconds_total"], n_failed=info["n_failed"],
                         slots=info["shape"][2]))
    costs = pd.DataFrame(rows)
    save_table(costs, "decomposition_costs")
    return costs


def figures(folds, recalls, scales, convergence, stages):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 180})
    folder = OUT_DIR / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, task in enumerate(("species", "passerine")):
        for column, feature_set in enumerate(FEATURE_SETS):
            ax = axes[row, column]
            cell = folds[(folds["task"] == task) & (folds["feature_set"] == feature_set)]
            for method, color in zip(METHODS, COLORS):
                mean = cell[cell["method"] == method].groupby("classifier")["balanced_accuracy"].mean().reindex(CLASSIFIERS)
                ax.plot(np.arange(6), mean * 100, "o-", color=color, label=SHORT[method], lw=1.7, ms=4)
            ax.set_xticks(np.arange(6), ["LDA", "QDA", "GNB", "LogReg", "kNN", "LinSVM"])
            ax.set_title(f"{task.title()}, {'with' if column else 'without'} log_scale")
            ax.set_ylabel("Balanced accuracy (%)")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle("BirdCLEF 2023: mean of 15 recordist-held-out folds")
    fig.savefig(folder / "classifier_balanced_accuracy.png")
    plt.close(fig)

    pair_names = [(METHODS[3], METHODS[1]), (METHODS[2], METHODS[1]), (METHODS[3], METHODS[2])]
    cell = recalls[recalls["classifier"] == "logreg"]
    species = sorted(cell["species"].unique())
    matrix, labels = [], []
    for feature_set in FEATURE_SETS:
        for a, b in pair_names:
            sub = cell[(cell["feature_set"] == feature_set) & (cell["method_a"] == a) & (cell["method_b"] == b)]
            matrix.append(sub.set_index("species")["mean_diff"].reindex(species).to_numpy() * 100)
            labels.append(f"{SHORT[a]} - {SHORT[b]}\n{'with' if feature_set.endswith('log_scale') else 'without'} log_scale")
    matrix = np.asarray(matrix).T
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    limit = max(1, np.abs(matrix).max())
    artist = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(species)), species)
    for i in range(len(species)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(matrix[i,j]) > 0.65 * limit else "black")
    fig.colorbar(artist, ax=ax, label="Recall difference (percentage points)", shrink=0.7)
    ax.set_title("Species recall, common logistic regression; descriptive fold means")
    fig.savefig(folder / "per_species_recall.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for method, color in zip(METHODS, COLORS):
        cell = scales[(scales["task"] == "species") & (scales["method"] == method)].sort_values("scale_rank")
        for ax, metric in zip(axes[0], ("epsilon2_mean", "mi_mean")):
            keep = cell["centroid_hz_median"] > 0
            ax.plot(cell.loc[keep, "centroid_hz_median"], cell.loc[keep, metric], "o-", ms=4, color=color, label=SHORT[method])
            ax.set_xscale("log")
            ax.set_xlabel("Observed median component spectral centroid (Hz)")
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean Kruskal-Wallis epsilon², 20 features")
    axes[0, 1].set_ylabel("Mean binned mutual information (nats)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title("Species discriminability, descriptive all-frame analysis")
    axes[1, 0].bar(np.arange(len(stages)), (1 - stages["converged_fraction"]) * 100, color=COLORS[3])
    axes[1, 0].set_xticks(np.arange(0, len(stages), 2), stages["slot_label"].iloc[::2], rotation=45)
    axes[1, 0].set_ylabel("Unconverged stages (%)")
    axes[1, 0].set_title("IRMF defaults: 60-iteration cap")
    cell = convergence[(convergence["task"] == "species") & (convergence["classifier"] == "logreg") & (convergence["method"] == METHODS[3])]
    for i, feature_set in enumerate(FEATURE_SETS):
        sub = cell[cell["feature_set"] == feature_set].set_index("converged")
        axes[1, 1].bar(np.arange(2) + i * 0.34, [100 * sub.loc[x, "balanced_accuracy"] for x in (True, False)],
                       width=0.32, label="with log_scale" if i else "without log_scale", color=["#87b9a0", COLORS[3]][i])
    axes[1, 1].set_xticks(np.arange(2) + 0.17, ["All stages converged", "Any stage unconverged"])
    axes[1, 1].set_ylabel("Species balanced accuracy (%)")
    axes[1, 1].set_title("LogReg OOF subsets; observational, composition differs")
    axes[1, 1].legend(fontsize=9)
    fig.savefig(folder / "scales_and_convergence.png")
    plt.close(fig)


def summary_html(best, comparisons, recordings):
    panels = {}
    for task in ("species", "passerine"):
        panels[task] = {}
        for feature_set in FEATURE_SETS:
            rows = []
            for method in METHODS:
                selected = best[(best["task"] == task) & (best["feature_set"] == feature_set) & (best["method"] == method)].iloc[0]
                common = recordings[(recordings["task"] == task) & (recordings["feature_set"] == feature_set) &
                                    (recordings["method"] == method) & (recordings["classifier"] == "logreg")].iloc[0]
                rows.append(dict(method=SHORT[method], best=float(selected["balanced_accuracy_mean"]),
                                 classifier=selected["classifier"], frame=float(common["frame_balanced_accuracy"]),
                                 recording=float(common["balanced_accuracy"])))
            tests = []
            for other in METHODS[1:3]:
                selected = comparisons[(comparisons["task"] == task) & (comparisons["feature_set"] == feature_set) &
                                       (comparisons["classifier"] == "logreg") & (comparisons["metric"] == "balanced_accuracy") &
                                       (comparisons["method_a"] == METHODS[3]) & (comparisons["method_b"] == other)].iloc[0]
                tests.append(dict(other=SHORT[other], delta=float(selected["mean_diff"]), low=float(selected["ci_low"]),
                                  high=float(selected["ci_high"]), q=float(selected["p_nb_holm"])))
            panels[task][feature_set] = dict(rows=rows, tests=tests)
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:var(--font-sans,system-ui);color:var(--foreground,#222);margin:0}button{font:inherit;cursor:pointer;border:1px solid var(--border,#ddd);background:transparent;color:inherit;border-radius:7px;padding:7px 12px}button[aria-pressed=true]{background:var(--accent-surface,#e7f2ea);color:var(--accent-surface-foreground,#205b37);border-color:var(--accent,#38805a)}.controls{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 12px}.caption{font-size:13px;color:var(--muted-foreground,#666);margin:12px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:11px 5px;border-bottom:1px solid var(--border,#ddd)}th:first-child,td:first-child{text-align:left}th{font-weight:500;font-size:12px;color:var(--muted-foreground,#666)}small{font-size:11px;color:var(--muted-foreground,#666)}.tests{margin-top:15px;padding:16px;background:var(--secondary,#f5f5f5);border-radius:var(--radius,9px);font-size:14px;line-height:1.7}.note{font-size:12px;color:var(--muted-foreground,#666);line-height:1.5;margin:12px 0 0}@media(max-width:450px){th,td{padding:9px 3px;font-size:12px}button{font-size:12px;padding:7px 9px}}</style></head><body>
<div class="controls"><button data-task="species" aria-pressed="true">16 species</button><button data-task="passerine" aria-pressed="false">Passerine</button></div>
<div class="controls"><button data-features="comp" aria-pressed="true">Without log_scale</button><button data-features="comp+log_scale" aria-pressed="false">With log_scale</button></div>
<div class="caption">Balanced accuracy (%), mean of 15 recordist-held-out folds</div>
<table><thead><tr><th>Method</th><th>Best frame<br><small>descriptive best of six</small></th><th>LogReg<br>frame</th><th>LogReg<br>recording</th></tr></thead><tbody id="rows"></tbody></table>
<div class="tests" id="tests"></div><p class="note">Tests use matched logistic-regression folds, Nadeau–Bengio 95% intervals, and Holm adjustment across 72 comparisons per task. Recording probabilities average the selected frames, retaining the recordist split. SVM C was limited to 0.001 and 0.01 for all methods; other tuning grids are unchanged.</p>
<p class="note">Decomposition CPU seconds/frame: raw 0, EMD 0.0116, linear IMF 0.0063, IRMF 1.3205.</p>
<script>const data=__DATA__;let task='species',features='comp';const pct=x=>(100*x).toFixed(1);const signed=x=>(x>=0?'+':'')+pct(x);const p=x=>x<.001?x.toExponential(1):x.toFixed(3);function draw(){const d=data[task][features];document.querySelector('#rows').innerHTML=d.rows.map(r=>`<tr><td>${r.method}</td><td>${pct(r.best)} <small>${r.classifier}</small></td><td>${pct(r.frame)}</td><td>${pct(r.recording)}</td></tr>`).join('');document.querySelector('#tests').innerHTML=d.tests.map(t=>`IRMF versus ${t.other}: <strong>${signed(t.delta)} pp</strong> [${signed(t.low)}, ${signed(t.high)}], q=${p(t.q)}`).join('<br>');document.querySelectorAll('[data-task]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.task===task));document.querySelectorAll('[data-features]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.features===features));}document.querySelectorAll('[data-task]').forEach(b=>b.onclick=()=>{task=b.dataset.task;draw()});document.querySelectorAll('[data-features]').forEach(b=>b.onclick=()=>{features=b.dataset.features;draw()});draw();console.log('Validated panels',Object.keys(data).length*2,'rows',document.querySelectorAll('#rows tr').length);</script></body></html>"""
    (OUT_DIR / "results_summary.html").write_text(html.replace("__DATA__", json.dumps(panels)))


def main():
    started = time.perf_counter()
    info = json.loads((DATA_DIR / "prepared" / "dataset.json").read_text())
    assert info["fs"] == FS and info["frame_length"] == LENGTH
    meta = pd.read_parquet(DATA_DIR / "prepared" / "meta.parquet").sort_values("frame_id").reset_index(drop=True)
    folds = pd.concat([pd.read_csv(OUT_DIR / f"results_folds_{task}.csv") for task in ("species", "passerine")], ignore_index=True)
    assert len(folds) == 1440
    assert not folds.duplicated(["task", "method", "feature_set", "classifier", "split_id"]).any()
    assert folds.groupby(["task", "method", "feature_set", "classifier"]).size().eq(15).all()
    assert np.isfinite(folds[METRICS].to_numpy()).all()
    reference_scale = pd.read_parquet(DATA_DIR / "features" / "raw.parquet", columns=["ch0_log_scale"]).to_numpy()
    for method in METHODS[1:]:
        values = pd.read_parquet(DATA_DIR / "features" / f"{method}.parquet", columns=["ch0_log_scale"]).to_numpy()
        assert np.array_equal(reference_scale, values), f"Shared log_scale differs: {method}"
    print("Paired statistics and per-class recall", flush=True)
    comparisons, _ = compare_all(folds)
    _, best = summarize_folds(folds)
    recalls = per_class(folds)
    print("Component scale analysis", flush=True)
    scales = pd.concat([scale_analysis(task) for task in ("species", "passerine")], ignore_index=True)
    print("Convergence subsets", flush=True)
    convergence, stages, stats = convergence_analysis(meta, folds)
    print("Recording probability aggregation", flush=True)
    recordings = recording_analysis(meta)
    costs = cost_analysis()
    print("Figures", flush=True)
    figures(folds, recalls, scales, convergence, stages)
    summary_html(best, comparisons, recordings)
    confounds = dict(n_multispecies_recordists=int(meta.groupby("group")["y_species"].nunique().gt(1).sum()),
                     species_median_rms_dbfs=meta.groupby("y_species")["frame_rms_dbfs"].median().to_dict(),
                     holm_families="72 comparisons per task and metric, spanning 2 feature sets, 6 classifiers, 6 method pairs; all 1152 species recall comparisons form one family",
                     recording_aggregation="Mean native predict_proba within recording in each outer fold, evaluated per fold and pooled per seed; linear SVM has no native probabilities and is excluded from this auxiliary table",
                     frequency_mapping="Observed median Hann-periodogram spectral centroid, with 10th/90th percentiles across nonzero components; not fixed filter passbands. IMF fs/window is a nominal scale only.")
    confounds["versions"] = {name: importlib.metadata.version(name) for name in
                             ["numpy", "scipy", "pandas", "scikit-learn", "pimf", "EMD-signal", "matplotlib", "joblib"]}
    source_paths = [STUDY_DIR / "baselines" / "run_classical.py", Path(__file__),
                    Path(__file__).with_name("run_classical_phase2.py")]
    source_paths += [STUDY_DIR / "emdirmf" / f"{name}.py" for name in ["cv", "stats", "features", "decompose"]]
    confounds["source_sha256"] = {str(path.relative_to(ROOT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    (OUT_DIR / "analysis_notes.json").write_text(json.dumps(confounds, indent=2))
    timing = {task: json.loads((OUT_DIR / f"run_{task}.json").read_text())["seconds"] for task in ("species", "passerine")}
    pilot_stat = (DATA_DIR / "results" / "classical_phase2" / "species_full.log").stat()
    timing["analysis_seconds"] = time.perf_counter() - started
    timing["elapsed_wall_seconds_including_pilot"] = time.time() - getattr(pilot_stat, "st_birthtime", pilot_stat.st_ctime)
    timing["reported_outer_fit_seconds_sum"] = float(folds["fit_seconds"].sum())
    (OUT_DIR / "execution_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps({"validated_outer_fits": len(folds), "convergence": stats,
                      "costs": costs.to_dict("records"), "output": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
