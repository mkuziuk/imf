"""Phase-2 classical baselines for one dataset and task.

  .venv/bin/python experiments/classification-emd-irmf/baselines/run_classical.py \
      --dataset NAME [--task TASK] [--methods raw emd imf_linear_a1.41 irmf_h2_a1.41] \
      [--workers 6] [--breakdown META_COLUMN] [--quick] \
      [--save-probabilities] [--results-tag RUN_NAME]

For every method tag (feature table from emdirmf.run_features) and both feature
sets (components only, components + log_scale) this runs

- per-feature discriminability on all frames (descriptive only): Kruskal-Wallis H,
  epsilon^2 = H / (n - 1), and mutual information in nats from 16 equal-frequency
  bins with the Miller-Madow bias correction (a kNN estimator is inflated by the
  near-duplicate frames inside each recording);
- LDA (Ledoit-Wolf shrinkage), QDA (per-class covariance shrinkage tuned over
  Ledoit-Wolf "auto" and fixed 0.05-0.8; sklearn 1.9 rejects reg_param with
  fewer samples than features per class), Gaussian NB, multinomial
  L2 logistic regression (C tuned), kNN (k tuned), linear SVM (C tuned), each as
  median imputer -> constant-feature filter -> StandardScaler -> classifier fitted
  on the training fold only; tuning uses a grouped inner 3-fold CV on that fold;

on the shared splits (emdirmf.cv.make_splits). Outputs go to
output/classification-emd-irmf/NAME/classical/ (long-format CSV, comparison
tables, summary.md); out-of-fold predictions go to DATA_ROOT/NAME/results/.
--results-tag isolates predictions under results/RUN_NAME/. --save-probabilities
adds native prob_<class> columns, with NaN for nonprobabilistic classifiers.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from joblib import Parallel, delayed, parallel_config  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402
from sklearn.discriminant_analysis import (  # noqa: E402
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from emdirmf import decompose as dec  # noqa: E402
from emdirmf.cv import fit_predict_split, make_splits, split_metrics  # noqa: E402
from emdirmf.features import feature_columns  # noqa: E402
from emdirmf.io import label_column, load_info, load_meta  # noqa: E402
from emdirmf.parallel import check_workers, single_thread_env  # noqa: E402
from emdirmf.paths import decomp_dir, features_dir, output_dir, results_dir  # noqa: E402
from emdirmf.stats import compare_methods, default_pairs  # noqa: E402

DEFAULT_METHODS = ("raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41")
FEATURE_SETS = ("comp", "comp+log_scale")
METRICS = ("balanced_accuracy", "macro_f1", "accuracy")
MI_BINS = 16
SEED = 0


def _pipeline(classifier):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("constant", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", classifier),
    ])


def _tuned(classifier, grid):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    return GridSearchCV(
        _pipeline(classifier),
        {f"clf__{key}": values for key, values in grid.items()},
        cv=inner,
        scoring="balanced_accuracy",
        n_jobs=1,
        error_score=0.0,
    )


def classifiers():
    """Name -> estimator; tuned ones are GridSearchCV over a grouped inner CV."""
    return {
        "lda": _pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "qda": _tuned(QuadraticDiscriminantAnalysis(solver="eigen"),
                      {"shrinkage": ["auto", 0.05, 0.2, 0.5, 0.8]}),
        "gnb": _pipeline(GaussianNB()),
        "logreg": _tuned(LogisticRegression(max_iter=3000), {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
        "knn": _tuned(KNeighborsClassifier(), {"n_neighbors": [1, 5, 15, 35]}),
        "linsvm": _tuned(LinearSVC(dual="auto", max_iter=5000), {"C": [0.001, 0.01, 0.1, 1.0]}),
    }


# ---------------------------------------------------------------------------
# Discriminability
# ---------------------------------------------------------------------------


def binned_mutual_information(ranks, y, n_bins=MI_BINS):
    """MI (nats) between each column's quantile bin and the class, Miller-Madow corrected.

    ranks: average ranks per column (ties share a bin, a constant column gives 0).
    """
    n, n_columns = ranks.shape
    classes, y_index = np.unique(y, return_inverse=True)
    k = len(classes)
    bins = np.minimum(((ranks - 1) * n_bins / n).astype(np.int64), n_bins - 1)
    codes = bins * k + y_index[:, None] + (np.arange(n_columns) * n_bins * k)[None, :]
    joint = np.bincount(codes.ravel(), minlength=n_columns * n_bins * k).reshape(n_columns, n_bins, k) / n
    px = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log(joint / (px * py)), 0.0)
    mi = terms.sum(axis=(1, 2))
    bias = ((joint > 0).sum(axis=(1, 2)) - (px > 0).sum(axis=(1, 2)) - (py > 0).sum(axis=(1, 2)) + 1) / (2 * n)
    return np.maximum(mi - bias, 0.0)


def kruskal_columns(X, y, ranks=None):
    """Vectorized Kruskal-Wallis H (tie-corrected), p-value and epsilon^2 per column."""
    n = X.shape[0]
    ranks = scipy_stats.rankdata(X, axis=0) if ranks is None else ranks
    classes = np.unique(y)
    H = np.zeros(X.shape[1])
    for label in classes:
        mask = y == label
        H += mask.sum() * (ranks[mask].mean(axis=0) - (n + 1) / 2) ** 2
    H *= 12.0 / (n * (n + 1))
    sorted_values = np.sort(X, axis=0)
    tie_sum = np.zeros(X.shape[1])
    for column in range(X.shape[1]):
        _, counts = np.unique(sorted_values[:, column], return_counts=True)
        tie_sum[column] = np.sum(counts**3 - counts)
    correction = 1 - tie_sum / (n**3 - n)
    H = np.divide(H, correction, out=np.zeros_like(H), where=correction > 0)
    p = scipy_stats.chi2.sf(H, len(classes) - 1)
    return H, p, H / (n - 1)


def discriminability(table, y, method, slot_labels):
    columns = feature_columns(table.columns, include_log_scale=True)
    X = table[columns].to_numpy(dtype=np.float64)
    keep = np.all(np.isfinite(X), axis=1)
    X, y_kept = X[keep], y[keep]
    ranks = scipy_stats.rankdata(X, axis=0)
    H, p, eps2 = kruskal_columns(X, y_kept, ranks)
    mi = binned_mutual_information(ranks, y_kept)
    records = []
    for name, h, pv, e, m in zip(columns, H, p, eps2, mi):
        if name.endswith("_log_scale"):
            channel, slot, feat, label = int(name[2:].split("_")[0]), -1, "log_scale", "frame"
        else:
            head, feat = name.split("_", 2)[:2], name.split("_", 2)[2]
            channel, slot = int(head[0][2:]), int(head[1][1:])
            label = slot_labels[slot]
        records.append({"method": method, "feature": name, "channel": channel, "slot": slot,
                        "slot_label": label, "feat": feat, "kw_H": h, "kw_p": pv, "epsilon2": e, "mi": m})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _job(key, estimator, X, y, split, groups, labels, save_probabilities=False):
    if save_probabilities:
        return (key, *fit_predict_split(estimator, X, y, split, groups=groups, labels=labels,
                                       return_probabilities=True))
    row, y_pred = fit_predict_split(estimator, X, y, split, groups=groups, labels=labels)
    return key, row, y_pred


def subset_rows(predictions, meta, column, labels):
    """Per-split metrics within each level of a meta column, from out-of-fold predictions."""
    merged = predictions.merge(meta[["frame_id", column]], on="frame_id")
    rows = []
    keys = ["method", "feature_set", "classifier", "split_id", "seed", "fold", "n_train", "n_test"]
    for (level, *values), cell in merged.groupby([column] + keys, sort=False):
        row = dict(zip(keys, values))
        row.update(subset=f"{column}={level}", n_test_subset=len(cell))
        row.update(split_metrics(cell["y_true"].to_numpy(), cell["y_pred"].to_numpy(), labels))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}"


def decomposition_costs(dataset, methods):
    rows = []
    for tag in methods:
        if tag == "raw":
            continue
        path = decomp_dir(dataset) / f"{tag}.json"
        if not path.exists():
            continue
        with open(path) as handle:
            summary = json.load(handle)
        row = {"method": tag, "slots": summary["shape"][2],
               "cpu_s_per_frame": summary["seconds_per_frame"]["cpu_mean"],
               "failed": summary["n_failed"]}
        if "irmf" in summary:
            n = summary["shape"][0] * summary["shape"][1]
            row["irmf_all_stages_converged"] = f"{summary['irmf']['channels_all_converged']}/{n}"
        if "emd_n_imfs_counts" in summary:
            row["emd_imf_counts"] = summary["emd_n_imfs_counts"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(path, context, folds, comparisons, disc, costs, breakdown):
    lines = [f"# Classical baselines: {context['dataset']} / {context['task']}", ""]
    lines.append(
        f"{context['n_frames']} frames, {context['n_groups']} groups, {context['n_classes']} classes; "
        f"{context['n_splits']} grouped splits (5 folds x {context['n_seeds']} seeds). "
        f"Runtime {context['seconds']:.0f} s with {context['workers']} workers."
    )
    lines.append("")
    if len(costs):
        lines += ["## Decomposition cost", "", costs.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## Balanced accuracy (mean ± sd over splits)", ""]
    for feature_set in FEATURE_SETS:
        cell = folds[folds["feature_set"] == feature_set]
        pivot = cell.groupby(["method", "classifier"])["balanced_accuracy"].agg(["mean", "std"])
        table = pivot.apply(lambda r: _fmt(r["mean"], r["std"]), axis=1).unstack("classifier")
        table = table.reindex(context["methods"])
        lines += [f"Feature set `{feature_set}`:", "", table.to_markdown(), ""]
    best = (folds.groupby(["feature_set", "method", "classifier"])["balanced_accuracy"].mean()
            .reset_index().sort_values("balanced_accuracy", ascending=False)
            .drop_duplicates(["feature_set", "method"]).sort_values(["feature_set", "method"]))
    lines += ["## Best classifier per method", "", best.to_markdown(index=False, floatfmt=".3f"), ""]
    if len(comparisons):
        show = comparisons[(comparisons["metric"] == "balanced_accuracy") & (comparisons["subset"] == "all")]
        show = show.assign(
            diff=show["mean_diff"].map("{:+.3f}".format),
            ci=show.apply(lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1),
            wins=show.apply(lambda r: f"{r['wins_a']}/{r['ties']}/{r['wins_b']}", axis=1),
        )
        columns = ["feature_set", "classifier", "method_a", "method_b", "diff", "ci", "p_nb_holm",
                   "p_wilcoxon_holm", "wins"]
        lines += ["## Paired comparisons, balanced accuracy (A - B; Holm within the table)", "",
                  "CI: 95% with the Nadeau-Bengio corrected standard error. wins: A better/tie/B better.", "",
                  show[columns].to_markdown(index=False, floatfmt=".4f"), ""]
    if breakdown is not None and len(breakdown):
        lines += [f"## Balanced accuracy by subset ({context['breakdown']})", ""]
        best_clf = best.set_index(["feature_set", "method"])["classifier"]
        for feature_set in FEATURE_SETS:
            cell = breakdown[breakdown["feature_set"] == feature_set]
            cell = cell[[best_clf.get((feature_set, m)) == c for m, c in zip(cell["method"], cell["classifier"])]]
            table = cell.groupby(["subset", "method"])["balanced_accuracy"].mean().unstack("method")
            table = table.reindex(columns=context["methods"])
            lines += [f"Feature set `{feature_set}`, best classifier per method:", "",
                      table.to_markdown(floatfmt=".3f"), ""]
    if len(disc):
        lines += ["## Most discriminative features (Kruskal-Wallis epsilon^2, all frames)", ""]
        top = disc.sort_values("epsilon2", ascending=False).groupby("method").head(5)
        top = top[["method", "feature", "slot_label", "epsilon2", "mi"]]
        summary = disc.groupby("method").agg(max_eps2=("epsilon2", "max"), max_mi=("mi", "max"),
                                             n_features=("feature", "size"))
        lines += [summary.reindex(context["methods"]).to_markdown(floatfmt=".3f"), "",
                  top.to_markdown(index=False, floatfmt=".3f"), ""]
    path.write_text("\n".join(lines))


def run(dataset, task=None, methods=DEFAULT_METHODS, workers=1, breakdown=None, quick=False,
        classifier_names=None, out=None, *, save_probabilities=False, results_tag=None):
    if results_tag is not None and (not results_tag or results_tag in (".", "..")
                                    or "/" in results_tag or "\\" in results_tag):
        raise ValueError("results_tag must be a nonempty directory name, not a path")
    started = time.perf_counter()
    workers = check_workers(workers)
    single_thread_env()
    info, meta = load_info(dataset), load_meta(dataset)
    task = info["primary_task"] if task is None else task
    y = meta[label_column(info, task)].to_numpy(dtype=str)
    groups = meta["group"].astype(str).to_numpy()
    labels = sorted(set(y))
    splits = make_splits(dataset, task)
    if quick:
        splits = [split for split in splits if split.seed == splits[0].seed]
    methods = [dec.canonical_tag(tag) for tag in methods]
    models = classifiers()
    if classifier_names:
        models = {name: models[name] for name in classifier_names}
    out = output_dir(dataset, "classical") if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tables, slot_label_map, disc = {}, {}, []
    for tag in methods:
        table = pd.read_parquet(features_dir(dataset) / f"{tag}.parquet")
        if not np.array_equal(table["frame_id"].to_numpy(), meta["frame_id"].to_numpy()):
            raise RuntimeError(f"{tag} features are not aligned with meta frame_id")
        with open(features_dir(dataset) / f"{tag}.json") as handle:
            slot_label_map[tag] = json.load(handle)["slot_labels"]
        tables[tag] = table
        disc.append(discriminability(table, y, tag, slot_label_map[tag]))
        print(f"{tag}: {len(feature_columns(table.columns))} component features, "
              f"{int((~table['decomp_ok']).sum())} failed frames", flush=True)
    disc = pd.concat(disc, ignore_index=True)
    disc.to_csv(out / f"discriminability_{task}.csv", index=False)

    # Feature matrices go to .npy memmaps so joblib passes file references to the
    # workers instead of hashing and pickling every matrix once per fit.
    with tempfile.TemporaryDirectory(prefix="run_classical_") as scratch:
        jobs = []
        for tag in methods:
            for feature_set in FEATURE_SETS:
                columns = feature_columns(tables[tag].columns, include_log_scale=feature_set.endswith("log_scale"))
                path = Path(scratch) / f"{tag}__{feature_set}.npy"
                np.save(path, tables[tag][columns].to_numpy(dtype=np.float64))
                X = np.load(path, mmap_mode="r")
                for name, model in models.items():
                    for split in splits:
                        jobs.append(((tag, feature_set, name), model, X, split))
        print(f"{len(jobs)} fits on {workers} workers", flush=True)
        # Do not pass a new keyword to existing dataset-local worker wrappers
        # unless probability export was explicitly requested.
        job_options = {"save_probabilities": True} if save_probabilities else {}
        with parallel_config(backend="loky", inner_max_num_threads=1):
            outputs = Parallel(n_jobs=workers, verbose=0)(
                delayed(_job)(key, model, X, y, split, groups, labels, **job_options)
                for key, model, X, split in jobs
            )
        del jobs, X
    rows, predictions = [], []
    split_by_id = {split.split_id: split for split in splits}
    for (tag, feature_set, name), row, y_pred, *probability_output in outputs:
        row.update(method=tag, feature_set=feature_set, classifier=name)
        rows.append(row)
        split = split_by_id[row["split_id"]]
        prediction = pd.DataFrame({
            "method": tag, "feature_set": feature_set, "classifier": name, "split_id": split.split_id,
            "seed": split.seed, "fold": split.fold, "n_train": len(split.train), "n_test": len(split.test),
            "frame_id": split.test, "y_true": y[split.test], "y_pred": np.asarray(y_pred, dtype=str),
        })
        if save_probabilities:
            probabilities, classes = probability_output
            # Column names carry the class order. Missing training classes get
            # zero; estimators without native probabilities get NaN, not scores.
            for label in labels:
                prediction[f"prob_{label}"] = 0.0 if len(classes) else np.nan
            for index, label in enumerate(classes):
                prediction[f"prob_{label}"] = probabilities[:, index]
        predictions.append(prediction)
    folds = pd.DataFrame(rows)
    folds.insert(0, "task", task)
    folds.insert(0, "dataset", dataset)
    folds["subset"] = "all"
    predictions = pd.concat(predictions, ignore_index=True)
    pred_dir = results_dir(dataset)
    if results_tag is not None:
        pred_dir = pred_dir / results_tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    categorical = ["method", "feature_set", "classifier", "y_true", "y_pred"]
    predictions.astype({column: "category" for column in categorical}).to_parquet(
        pred_dir / f"classical_{task}_predictions.parquet", index=False)

    breakdown_rows = None
    if breakdown:
        breakdown_rows = subset_rows(predictions, meta, breakdown, labels)
        breakdown_rows.insert(0, "task", task)
        breakdown_rows.insert(0, "dataset", dataset)
    folds.to_csv(out / f"results_folds_{task}.csv", index=False)
    id_columns = ["dataset", "task", "method", "feature_set", "classifier", "subset", "split_id", "seed",
                  "fold", "n_train", "n_test"]
    long = folds.melt(id_vars=id_columns, value_vars=list(METRICS), var_name="metric", value_name="value")
    if breakdown_rows is not None:
        long = pd.concat([long, breakdown_rows.melt(id_vars=id_columns, value_vars=list(METRICS),
                                                    var_name="metric", value_name="value")], ignore_index=True)
        breakdown_rows.to_csv(out / f"results_subsets_{task}.csv", index=False)
    long.to_csv(out / f"results_long_{task}.csv", index=False)

    pairs = default_pairs(methods)
    comparisons = []
    for metric in ("balanced_accuracy", "macro_f1"):
        for subset, cell in long.groupby("subset", sort=False):
            table = compare_methods(cell, metric=metric, pairs=pairs, by=("feature_set", "classifier"))
            if len(table):
                table.insert(0, "subset", subset)
                comparisons.append(table)
    comparisons = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    comparisons.to_csv(out / f"comparisons_{task}.csv", index=False)

    costs = decomposition_costs(dataset, methods)
    context = {
        "dataset": dataset, "task": task, "methods": methods, "n_frames": len(meta),
        "n_groups": int(len(set(groups))), "n_classes": len(labels), "n_splits": len(splits),
        "n_seeds": len({split.seed for split in splits}), "workers": workers,
        "seconds": time.perf_counter() - started, "breakdown": breakdown,
    }
    write_summary(out / f"summary_{task}.md", context, folds, comparisons, disc, costs, breakdown_rows)
    with open(out / f"run_{task}.json", "w") as handle:
        json.dump({**context, "classifiers": list(models), "feature_sets": list(FEATURE_SETS),
                   "pairs": pairs, "quick": quick, "results_tag": results_tag,
                   "save_probabilities": save_probabilities,
                   "predictions": str(pred_dir / f"classical_{task}_predictions.parquet")}, handle, indent=2)
    print(f"wrote {out} in {context['seconds']:.0f} s")
    return folds, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase-2 classical baselines")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--classifiers", nargs="+", default=None,
                        help="subset of lda qda gnb logreg knn linsvm")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--breakdown", default=None, help="meta column for per-subset metrics")
    parser.add_argument("--quick", action="store_true", help="first seed only (5 splits)")
    parser.add_argument("--out", default=None, help="output directory (default output/.../NAME/classical)")
    parser.add_argument("--results-tag", default=None,
                        help="prediction subdirectory under DATA_ROOT/NAME/results (default: legacy path)")
    parser.add_argument("--save-probabilities", action="store_true",
                        help="save native OOF prob_<class> columns; NaN for models without predict_proba")
    args = parser.parse_args(argv)
    run(args.dataset, args.task, args.methods, args.workers, args.breakdown, args.quick, args.classifiers,
        args.out, save_probabilities=args.save_probabilities, results_tag=args.results_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
