"""Paired phase-2b inference, concise reports and equal-dataset summary.

Nadeau-Bengio corrected fold uncertainty is primary. Holm families include both
feature sets and every contrast displayed in each table, separately by metric.
Pooled inference is conditional on the observed studies, not on new datasets.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from run_ablations import DATA, MAIN, METRICS, OUTPUT, REPO, STUDY
from emdirmf.stats import holm, paired_comparison, wilcoxon_signed_rank

DATASETS = ["uas_birds_60ghz", "esc50_animals", "birdclef2023", "dronerf"]
TASKS = {"uas_birds_60ghz": "bird_drone", "esc50_animals": "animal",
         "birdclef2023": "species", "dronerf": "10class"}
SHORT = {"raw": "raw", "emd": "EMD", "imf_linear_a1.41": "linear sqrt2",
         "irmf_h2_a1.41": "IRMF GD60 sqrt2", "imf_linear_a2": "linear a2",
         "irmf_h2_a1.41_conv": "IRMF H2 conv sqrt2", "irmf_h1_a1.41_conv": "IRMF H1 conv sqrt2",
         "irmf_h4_a1.41_conv": "IRMF H4 conv sqrt2", "irmf_h2_a2_conv": "IRMF H2 conv a2"}
CONTRASTS = [
    ("phase2 IRMF-EMD", "irmf_h2_a1.41", "emd"),
    ("phase2 linear-EMD", "imf_linear_a1.41", "emd"),
    ("phase2 IRMF-raw", "irmf_h2_a1.41", "raw"),
    ("Q1 conv-GD60", "irmf_h2_a1.41_conv", "irmf_h2_a1.41"),
    ("Q2 H1-H2", "irmf_h1_a1.41_conv", "irmf_h2_a1.41_conv"),
    ("Q2 H4-H2", "irmf_h4_a1.41_conv", "irmf_h2_a1.41_conv"),
    ("Q2 H1-H4", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv"),
    ("Q3 IRMF a2-EMD", "irmf_h2_a2_conv", "emd"),
    ("Q3 linear a2-EMD", "imf_linear_a2", "emd"),
    ("Q4 GD60-linear sqrt2", "irmf_h2_a1.41", "imf_linear_a1.41"),
    ("Q4 conv-linear sqrt2", "irmf_h2_a1.41_conv", "imf_linear_a1.41"),
    ("Q4 conv-linear a2", "irmf_h2_a2_conv", "imf_linear_a2"),
    ("conv sqrt2-EMD", "irmf_h2_a1.41_conv", "emd"),
    ("Q1 a2 conv-GD60", "irmf_h2_a2_conv", "irmf_h2_a2"),
]


def adjusted(frame):
    frame = frame.copy()
    frame["p_nb_holm"] = holm(frame.p_nb)
    frame["p_wilcoxon_holm"] = holm(frame.p_wilcoxon)
    frame["holm_family_size"] = len(frame)
    return frame


def comparisons(folds):
    rows = []
    for (dataset, task, classifier, feature_set), cell in folds.groupby(
            ["dataset", "task", "classifier", "feature_set"], observed=True):
        for contrast, a, b in CONTRASTS:
            fa = cell[cell.method == a].sort_values("split_id")
            fb = cell[cell.method == b].sort_values("split_id")
            if b == "irmf_h2_a2" and fb.empty:
                continue
            assert len(fa) == len(fb) == 15
            assert np.array_equal(fa.split_id, fb.split_id)
            assert np.array_equal(fa[["n_train", "n_test"]], fb[["n_train", "n_test"]])
            for metric in METRICS:
                row = dict(dataset=dataset, task=task, classifier=classifier, feature_set=feature_set,
                           contrast=contrast, method_a=a, method_b=b, metric=metric)
                row.update(paired_comparison(fa[metric], fb[metric], fa.n_train, fa.n_test))
                rows.append(row)
    return pd.DataFrame(rows)


def choose_classifier(folds):
    # Choose once from phase-2 controls, before looking at the ablation scores.
    ranks = (folds[folds.method.isin(MAIN)].groupby("classifier").balanced_accuracy.mean()
             .rename("phase2_mean_ba").reset_index()
             .sort_values(["phase2_mean_ba", "classifier"], ascending=[False, True]))
    return str(ranks.iloc[0].classifier), ranks


def interval(row):
    return f"{100*row.mean_diff:+.2f} [{100*row.ci_low:+.2f}, {100*row.ci_high:+.2f}]"


def probability(value):
    return f"{value:.3g}"


def comparison_table(frame, secondary=False):
    rows = []
    for contrast, _, _ in CONTRASTS:
        cells = frame[frame.contrast == contrast]
        if cells.empty:
            continue
        row = {"contrast": contrast}
        for fs, title in [("comp", "without log_scale"), ("comp+log_scale", "with log_scale")]:
            cell = cells[cells.feature_set == fs].iloc[0]
            value = f"{interval(cell)}; {probability(cell.p_nb_holm)}"
            if secondary:
                value += f" / {probability(cell.p_wilcoxon_holm)}"
            row[title] = value
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def verdicts(table):
    def sentence(prefix, label):
        cell = table[table.contrast.str.startswith(prefix)]
        low, high = (100 * cell.mean_diff).min(), (100 * cell.mean_diff).max()
        sig = cell[cell.p_nb_holm < .05]
        positive, negative = int((sig.mean_diff > 0).sum()), int((sig.mean_diff < 0).sum())
        return (f"{label}: {low:+.2f} to {high:+.2f} pp; corrected-t/Holm rejects "
                f"{positive} positive and {negative} negative contrasts of {len(cell)}.")
    return [sentence("Q1", "Convergence"), sentence("Q2", "H sensitivity"),
            sentence("Q3", "Matched-count contrasts against EMD"),
            sentence("Q4", "IRMF contrasts against linear IMF")]


def write_dataset(dataset):
    out = OUTPUT / dataset / "classical_ablations"
    primary = TASKS[dataset]
    paths = sorted(out.glob("results_folds_*.csv"))
    if not (out / f"results_folds_{primary}.csv").exists():
        return None
    reports = []
    primary_result = None
    for path in paths:
        task = path.stem.removeprefix("results_folds_")
        # Restore baseline scores directly from their original decimal strings.
        # Default CSV parsing can lose one ULP; repeated save/read cycles can
        # otherwise turn a tied score into a tiny nonzero paired difference.
        old = pd.read_csv(path, float_precision="round_trip")
        baseline = pd.read_csv(OUTPUT / dataset / "classical" / f"results_folds_{task}.csv",
                               float_precision="round_trip")
        baseline = baseline[baseline.classifier.isin(old.classifier.unique()) & baseline.method.isin(MAIN)].copy()
        baseline["source"] = "phase2_reused"
        new = pd.read_csv(out / f"ablation_folds_{task}.csv", float_precision="round_trip")
        folds = pd.concat([baseline, new], ignore_index=True)
        baseline.to_csv(out / f"reused_folds_{task}.csv", index=False)
        folds.to_csv(path, index=False)
        classifier, ranks = choose_classifier(folds)
        ranks.to_csv(out / f"classifier_selection_{task}.csv", index=False)
        raw = comparisons(folds)
        all_tables, fixed_tables = [], []
        for metric, cell in raw.groupby("metric", sort=False):
            all_tables.append(adjusted(cell))
            fixed_tables.append(adjusted(cell[cell.classifier == classifier]))
        all_tables = pd.concat(all_tables, ignore_index=True)
        fixed = pd.concat(fixed_tables, ignore_index=True)
        all_tables.to_csv(out / f"paired_all_metrics_{task}.csv", index=False)
        fixed.to_csv(out / f"paired_fixed_classifier_{task}.csv", index=False)
        means = (folds.groupby(["method", "feature_set", "classifier"], observed=True)[METRICS]
                 .agg(["mean", "std"]))
        means.columns = ["_".join(col) for col in means.columns]
        means.reset_index().to_csv(out / f"metric_summary_{task}.csv", index=False)
        audit = json.loads((out / f"audit_{task}.json").read_text())
        ba = fixed[fixed.metric == "balanced_accuracy"]
        reports += [f"## {task}, fixed {classifier}", ""]
        reports += verdicts(ba) + [""]
        fixed_folds = folds[folds.classifier == classifier]
        metric_rows = []
        for method, cell in fixed_folds.groupby("method", sort=False):
            row = {"method": SHORT.get(method, method)}
            for fs, title in [("comp", "without log_scale BA/F1/Acc"),
                              ("comp+log_scale", "with log_scale BA/F1/Acc")]:
                values = cell[cell.feature_set == fs][METRICS].mean()
                row[title] = "/".join(f"{100*x:.2f}" for x in values)
            metric_rows.append(row)
        reports += [pd.DataFrame(metric_rows).to_markdown(index=False), "",
                    "Scores are percentages. Paired differences below are percentage points, "
                    "95% Nadeau-Bengio CI; corrected-t Holm p / secondary Wilcoxon Holm p.", "",
                    comparison_table(ba, secondary=True), "",
                    f"Holm covers all {len(ba)} cells above, including both feature sets. "
                    f"The all-classifier table corrects {len(all_tables[all_tables.metric == 'balanced_accuracy'])} "
                    "cells per metric. CIs are marginal, not multiplicity-adjusted. "
                    "A nonsignificant result does not establish equivalence.", "",
                    f"[{task} macro-F1, accuracy and all classifiers](paired_all_metrics_{task}.csv). "
                    f"[Fixed-classifier tests for all three metrics](paired_fixed_classifier_{task}.csv). "
                    f"[Means and fold SDs](metric_summary_{task}.csv).", "",
                    f"Runtime {audit['seconds']/60:.1f} min; {audit['job_count']} new fits and "
                    f"{audit['reuse']['verified_fits']} reused phase-2 fits, four single-thread workers.", ""]
        warnings = folds[folds.source == "phase2b"]
        if "warning_count" in warnings:
            warnings = warnings[warnings.warning_count > 0]
            warnings.to_csv(out / f"fit_warnings_{task}.csv", index=False)
            reports += [f"{len(warnings)} new fits recorded warnings; "
                        f"[messages](fit_warnings_{task}.csv) are retained rather than suppressed. "
                        "These counts include inner-grid candidates when present.", ""]
        selected = folds[folds.source == "phase2b"]
        if {"selected_n_iter", "selected_max_iter"}.issubset(selected.columns):
            capped = selected[selected.selected_n_iter >= selected.selected_max_iter]
            if len(capped):
                reports += [f"{len(capped)} selected classifier fits reached their iteration cap. "
                            "Primary caps/settings match phase 2; these are classifier-optimization "
                            "caps, separate from the fully converged decomposition stages.", ""]
        log_path = DATA / dataset / "results/phase2b" / f"{task}.log"
        if log_path.exists():
            restart_warnings = log_path.read_text().count("A worker stopped while some jobs")
            if restart_warnings:
                reports += [f"The run log records {restart_warnings} worker-replacement warning(s). "
                            "Final coverage and unique checkpoint counts are checked separately; "
                            "no missing fit is silently dropped.", ""]
        if task == primary:
            primary_result = fixed.copy()
    audit = json.loads((out / f"audit_{primary}.json").read_text())
    available_tasks = set(json.loads((DATA / dataset / "prepared/dataset.json").read_text())["tasks"])
    completed_tasks = {p.stem.removeprefix("results_folds_") for p in paths}
    deferred_tasks = sorted(available_tasks - completed_tasks)
    ready = [r for r in audit["caches"] if r["ready"] and not r.get("skipped", False)]
    cost = pd.DataFrame(ready)[["tag", "slots", "cpu_seconds_per_frame"]]
    lines = [f"# {dataset}: classical ablations", ""]
    # Put primary task before any alphabetically earlier secondary task.
    starts = [i for i, s in enumerate(reports) if s.startswith("## ")]
    blocks = [reports[start:(starts[j+1] if j+1 < len(starts) else len(reports))]
              for j, start in enumerate(starts)]
    blocks.sort(key=lambda b: not b[0].startswith(f"## {primary},"))
    for block in blocks:
        lines.extend(block)
    lines += ["## Reproducibility and limits", "",
              "LDA, logistic regression and linear SVM retain their phase-2 estimators, grids, "
              "grouped tuning where used and all 15 saved splits. No baseline was refitted. "
              "The reporting classifier maximizes phase-2 mean BA across the four main methods and "
              "both feature sets, with alphabetical tie breaking, and stays fixed across all contrasts. "
              "This outer-score selection makes the inference exploratory. "
              "[The original six-classifier phase-2 report](../classical/REPORT.md) "
              "retains its original comparison families and verdicts.", "",
              cost.to_markdown(index=False, floatfmt=".5f"), "",
              "Slot counts include residuals and EMD's zero padding. EMD has 9 slots; a2 has 9 slots on UAS and 10 on "
              "the L=2048 datasets. Thus a2 is a near-matched, not identical, component count. "
              "No dimension-matched feature selection was performed. A2 also changes the bandwidth "
              "schedule, so this is not a pure feature-count intervention.", "",
              "All converged-tag channel-stages passed their stopping rule; all selected feature "
              "tables are finite, aligned and complete. Shared log_scale columns match raw exactly. "
              "The Newton tags solve pimf's lookup score to root tolerance 1e-12, with a 100-iteration "
              "cap; they preserve normalization, kernel, boundary and schedule. "
              "Reuse checks recompute BA/F1/accuracy from saved OOF predictions, match every "
              "held-out frame and label to the saved splits, and check fold sizes, seeds and IDs. "
              "Input hashes, estimator signatures, grids and readiness checks are in audit_*.json.", "",
              "The optional GD60 a2 tag requires completed decomposition and feature caches and "
              "explicit inclusion within the budget; its readiness/skip reason is recorded in the audit. "
              "Known phase-2 driver limitations are missing native OOF "
              "probabilities and prediction filenames shared across batches. The isolated wrapper "
              "records probabilities when supported and writes only phase2b checkpoints. Linear SVM "
              "has no native probability output. The new wrapper's early object-dtype y_true "
              "serialization was fixed; a one-time repair reconstructs Unicode labels from verified "
              "frame IDs and prepared metadata, requiring every stored metric to reproduce. "
              "[Repair audit](../../uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json). "
              "No new shared numerical bug was found.", "",
              "```bash",
              *[f"{REPO}/.venv/bin/python {STUDY}/baselines/ablations/run_ablations.py "
                f"--dataset {dataset} --task {completed_task}"
                + (" --allow-secondary" if completed_task != primary else "")
                for completed_task in sorted(completed_tasks, key=lambda value: (value != primary, value))],
              f"{REPO}/.venv/bin/python {STUDY}/baselines/ablations/analyse_ablations.py", "```", "",
              "Code changes are confined to baselines/ablations/. Large matrices, predictions and "
              "per-fit caches remain under DATA_ROOT/<dataset>/results/phase2b/. No commit was made.", ""]
    if deferred_tasks:
        lines += ["Secondary ablations deferred to preserve the four-dataset budget: "
                  + ", ".join(deferred_tasks) + ". Their phase-2 main results remain in the source report.", ""]
    if dataset == "dronerf":
        lines += ["The primary SVM fixes C=0.1 without redundant inner fits; logistic regression "
                  "uses C in {0.01, 0.1, 1}, exactly as in phase 2. Each BUI is one session, "
                  "so segment-held-out CV cannot establish unseen-session recognition. "
                  "L/H views are nonsimultaneous.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    return primary_result


def pooled_table(fixed):
    rows = []
    for (metric, feature_set, contrast), cell in fixed.groupby(["metric", "feature_set", "contrast"]):
        count = len(cell)
        if count < 2:
            continue
        # Independent datasets, equal weight. Every within-dataset variance is
        # the NB corrected variance. Welch-Satterthwaite accounts for df=14.
        variances = cell.se_corrected.to_numpy() ** 2
        variance = variances.sum() / count**2
        se = np.sqrt(variance)
        df = float(variances.sum()**2 / np.sum(variances**2 / cell.df)) if variances.sum() else 14.
        mean = cell.mean_diff.mean()
        half = stats.t.ppf(.975, df) * se
        p = float(2 * stats.t.sf(abs(mean / se), df)) if se else float(mean == 0)
        row = dict(dataset="pooled", task="primary tasks", classifier="dataset-fixed", metric=metric,
                   feature_set=feature_set, contrast=contrast, method_a=cell.method_a.iloc[0],
                   method_b=cell.method_b.iloc[0], mean_diff=mean, ci_low=mean-half, ci_high=mean+half,
                   se_corrected=se, df=df, p_nb=p, n_datasets=count,
                   min_dataset_difference=cell.mean_diff.min(), max_dataset_difference=cell.mean_diff.max())
        row.update(wilcoxon_signed_rank(cell.mean_diff))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--only", nargs="+", choices=DATASETS,
                        help="Recompute these datasets and reuse existing fixed tables for the others")
    arguments = parser.parse_args()
    tables, pending = [], []
    for dataset in DATASETS:
        cached = (OUTPUT / dataset / "classical_ablations"
                  / f"paired_fixed_classifier_{TASKS[dataset]}.csv")
        if arguments.only and dataset not in arguments.only:
            table = pd.read_csv(cached, float_precision="round_trip") if cached.exists() else None
        else:
            table = write_dataset(dataset)
        if table is None:
            pending.append(dataset)
        else:
            tables.append(table)
    if not tables:
        return
    fixed = pd.concat(tables, ignore_index=True)
    all_primary = pd.concat([
        pd.read_csv(OUTPUT / dataset / "classical_ablations"
                    / f"paired_all_metrics_{TASKS[dataset]}.csv", float_precision="round_trip")
        for dataset in fixed.dataset.unique()
    ], ignore_index=True)
    common_classifier_pools = []
    for classifier, cell in all_primary.groupby("classifier"):
        pooled_classifier = pooled_table(cell)
        if len(pooled_classifier):
            pooled_classifier["classifier"] = classifier
            common_classifier_pools.append(pooled_classifier)
    if common_classifier_pools:
        common = pd.concat(common_classifier_pools, ignore_index=True)
        common = pd.concat([adjusted(cell) for _, cell in common.groupby("metric")], ignore_index=True)
        common.to_csv(OUTPUT / "uas_birds_60ghz/classical_ablations/pooled_all_classifiers.csv", index=False)
    pooled = pooled_table(fixed)
    combined = pd.concat([fixed, pooled], ignore_index=True)
    # This is a new, single cross-dataset table; do not reuse narrower p values.
    combined = pd.concat([adjusted(cell) for _, cell in combined.groupby("metric")], ignore_index=True)
    combined.to_csv(OUTPUT / "uas_birds_60ghz/classical_ablations/cross_dataset_all_metrics.csv", index=False)
    ba = combined[combined.metric == "balanced_accuracy"]
    lines = ["# Classical classification summary", "",
             f"Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             "All differences are BA percentage points, A minus B, with 95% corrected intervals. "
             "Each dataset fixes the best phase-2 classifier across raw, EMD, linear IMF and GD60 IRMF, "
             "averaging both feature sets. The choice is exploratory and never changes by contrast.", "",
             "Phase-2 reports establish IRMF versus EMD gains on ESC-50 animals and BirdCLEF species, "
             "with and without log_scale. UAS is inconclusive after corrected-t/Holm testing and "
             "its recording votes reach a ceiling. DroneRF's primary SVM IRMF-EMD gain is "
             "+10.80/+10.67 pp without/with log_scale, while IRMF trails linear IMF by "
             "7.94/5.90 pp. The original reports establish no consistent robustness-specific "
             "IRMF gain over linear IMF; DroneRF gives clear evidence favoring linear IMF. "
             "BirdCLEF's original GD60 kNN is a classifier-specific positive exception, "
             "+2.53/+2.62 pp over linear IMF with original Holm significance; "
             "kNN convergence/schedule ablations are outside this budget. "
             "Source reports: [UAS](uas_birds_60ghz/classical/REPORT.md), "
             "[ESC-50](esc50_animals/classical/REPORT.md), "
             "[BirdCLEF](birdclef2023/classical/REPORT.md), "
             "[DroneRF](dronerf/classical/REPORT.md).", ""]
    for dataset, cell in ba.groupby("dataset", sort=False):
        lines += [f"{dataset}, {cell.classifier.iloc[0]}.", ""]
        lines += verdicts(cell) + [""]
    rows = []
    for dataset in DATASETS + ["pooled"]:
        cell = ba[ba.dataset == dataset]
        if cell.empty:
            rows.append({"dataset / fixed classifier": dataset, "contrast": "pending",
                         "without log_scale: delta [CI]; Holm p": "pending",
                         "with log_scale: delta [CI]; Holm p": "pending"})
            continue
        for contrast, _, _ in CONTRASTS:
            if not (cell.contrast == contrast).any():
                continue
            row = {"dataset / fixed classifier": f"{dataset} / {cell.classifier.iloc[0]}", "contrast": contrast}
            for fs, title in [("comp", "without log_scale: delta [CI]; Holm p"),
                              ("comp+log_scale", "with log_scale: delta [CI]; Holm p")]:
                r = cell[(cell.contrast == contrast) & (cell.feature_set == fs)].iloc[0]
                row[title] = f"{interval(r)}; {probability(r.p_nb_holm)}"
            rows.append(row)
    lines += [pd.DataFrame(rows).to_markdown(index=False), "",
              f"Holm correction spans all {len(ba)} numeric cells in this table, "
              "separately for corrected-t and Wilcoxon. The original phase-2 reports used their own "
              "comparison families; their published verdicts above retain that context. The CSV "
              "contains raw p values, secondary Wilcoxon and separate F1/accuracy families. "
              "Intervals are marginal 95% intervals.", "",
              "Pooled rows use the four primary tasks; secondary tasks remain dataset-specific. "
              "Pooled estimates weight each available dataset equally. Their SE propagates the "
              "independent dataset-specific Nadeau-Bengio corrected variances, with "
              "Welch-Satterthwaite degrees of freedom: SE_pool = sqrt(sum(SE_NB_i^2)) / D. "
              "This estimates the average across these "
              "observed studies, conditional on their chosen classifiers. It does not treat 60 "
              "overlapping folds as independent or establish an effect on unseen datasets. "
              "Pooled Wilcoxon uses one mean difference per dataset and has very low power. "
              "Between-dataset effect ranges are saved in the CSV. A separate "
              "[common-classifier pooled table]"
              "(uas_birds_60ghz/classical_ablations/pooled_all_classifiers.csv) "
              "holds LDA, logistic regression or linear SVM fixed across every dataset, "
              "with Holm correction across all three classifier tables per metric.", "",
              "Budget: at most four single-thread fit workers, three classifiers, no baseline "
              "refits. All 15 folds, both log_scale settings and five required tags are retained. "
              "Secondary tasks run when time permits. No PROTOCOL, shared package, baseline driver, "
              "ML source or pre-existing result was edited. No commit was made.", "",
              "Files: each dataset's classical_ablations/REPORT.md, reused and new fold CSVs, "
              "all-classifier/fixed-classifier paired CSVs, metric summaries, warning tables and "
              "audit JSONs. [Cross-dataset BA/F1/accuracy and secondary tests]"
              "(uas_birds_60ghz/classical_ablations/cross_dataset_all_metrics.csv). "
              "Code: experiments/classification-emd-irmf/baselines/ablations/.", "",
              "Bugs: the inherited driver omits OOF probabilities and shares prediction filenames "
              "between batches. Its default comparisons collapse tags to one representative per "
              "method family, so explicit pairs are required for H/convergence/schedule ablations. "
              "The new runner avoids these limitations without editing shared code. "
              "Its initial object-label serialization bug was fixed locally, with predictions, "
              "probabilities and timings preserved and metrics checked in the "
              "[repair audit](uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json).", ""]
    if pending:
        lines += ["Pending: " + ", ".join(pending) + ".", ""]
    (OUTPUT / "SUMMARY_classical.md").write_text("\n".join(lines))
    print("Wrote reports and summary for", sorted(fixed.dataset.unique()), "pending", pending)


if __name__ == "__main__":
    main()
