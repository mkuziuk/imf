"""Analyze completed full classical runs; data and predictions stay external.

Run from the study root after baselines/run_classical.py for both tasks.
Fixed-LDA paired tables are distinct from descriptive outer-CV winners.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

from baselines.run_classical import discriminability
from emdirmf.cv import load_splits, split_metrics
from emdirmf.decompose import normalize_frame, scale_order
from emdirmf.features import feature_columns
from emdirmf.io import load_frames, load_info, load_meta
from emdirmf.paths import dataset_dir, output_dir
from emdirmf.stats import compare_methods, default_pairs, holm, to_long

DATASET = "uas_birds_60ghz"
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
FEATURE_SETS = ["comp", "comp+log_scale"]
SHORT = dict(zip(METHODS, ["Raw", "EMD", "Linear IMF", "IRMF"]))
COLORS = dict(zip(METHODS, ["#6b7280", "#3876bf", "#dc9235", "#8c499f"]))
OUT = output_dir(DATASET, "classical")
BASE = dataset_dir(DATASET)


def paired_tables(folds, task, unit):
    tables, lda = [], []
    long = to_long(folds, metrics=METRICS)
    for metric in METRICS:
        table = compare_methods(long, metric=metric, pairs=default_pairs(METHODS),
                                by=("feature_set", "classifier"))
        tables.append(table)
        fixed = compare_methods(long[long["classifier"] == "lda"], metric=metric,
                                pairs=default_pairs(METHODS), by=("feature_set", "classifier"))
        lda.append(fixed)
    full, fixed = pd.concat(tables, ignore_index=True), pd.concat(lda, ignore_index=True)
    full.to_csv(OUT / f"paired_{unit}_all_metrics_{task}.csv", index=False)
    fixed.to_csv(OUT / f"paired_{unit}_lda_{task}.csv", index=False)
    return fixed


def recording_aggregation(task, meta):
    predictions = pd.read_parquet(BASE / "results" / f"classical_{task}_predictions.parquet")
    predictions = predictions.merge(meta[["frame_id", "group_recording"]], on="frame_id", validate="many_to_one")
    for column in ["method", "feature_set", "classifier", "y_true", "y_pred"]:
        predictions[column] = predictions[column].astype(str)
    keys = ["method", "feature_set", "classifier", "split_id", "seed", "fold", "group_recording"]
    counts = predictions.groupby(keys + ["y_pred"], observed=True).size().rename("votes").reset_index()
    # Alphabetical labels break a tied vote reproducibly; report the tie count.
    counts = counts.sort_values(keys + ["votes", "y_pred"], ascending=[True] * len(keys) + [False, True])
    votes = counts.drop_duplicates(keys).rename(columns={"y_pred": "recording_prediction"})
    truth = predictions.groupby(keys, observed=True)["y_true"].agg(["first", "nunique"]).reset_index()
    assert (truth["nunique"] == 1).all()
    votes = votes.merge(truth[keys + ["first"]], on=keys, validate="one_to_one")
    votes = votes.rename(columns={"first": "y_true", "recording_prediction": "y_pred"})
    maxima = counts.groupby(keys, observed=True)["votes"].transform("max")
    tied = counts[counts["votes"] == maxima].groupby(keys, observed=True).size()
    n_ties = int((tied > 1).sum())
    votes.to_parquet(BASE / "results" / f"classical_{task}_recording_predictions.parquet", index=False)
    splits = {s.split_id: s for s in load_splits(DATASET, task)}
    labels = sorted(meta[f"y_{task}"].unique())
    rows = []
    fold_keys = ["method", "feature_set", "classifier", "split_id", "seed", "fold"]
    for key, cell in votes.groupby(fold_keys, observed=True, sort=True):
        row = dict(zip(fold_keys, key))
        split = splits[row["split_id"]]
        # Ratios use the conservative pose groups, rather than correlated cells.
        row.update(task=task, n_train=meta.iloc[split.train]["group"].nunique(),
                   n_test=meta.iloc[split.test]["group"].nunique(),
                   n_recordings_test=len(cell))
        row.update(split_metrics(cell["y_true"].to_numpy(), cell["y_pred"].to_numpy(), labels))
        rows.append(row)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / f"recording_folds_{task}.csv", index=False)
    return folds, n_ties


def best_table(folds):
    grouped = folds.groupby(["feature_set", "method", "classifier"])[METRICS].agg(["mean", "std"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index().sort_values(
        ["balanced_accuracy_mean", "classifier"], ascending=[False, True], kind="stable")
    return grouped.drop_duplicates(["feature_set", "method"]).sort_values(["feature_set", "method"])


def group_discriminability(meta, info):
    tables, summaries = [], []
    for method in METHODS:
        features = pd.read_parquet(BASE / "features" / f"{method}.parquet")
        names = feature_columns(features.columns, include_log_scale=True)
        grouped = features[names].groupby(meta["group"], sort=True).median()
        group_labels = meta.groupby("group", sort=True)[list(info["tasks"].values())].first()
        labels = json.loads((BASE / "features" / f"{method}.json").read_text())["slot_labels"]
        for task, column in info["tasks"].items():
            disc = discriminability(grouped, group_labels[column].to_numpy(), method, labels)
            disc.insert(0, "task", task)
            tables.append(disc)
            for slot in scale_order(method, info["frame_length"]):
                cell = disc[disc["slot"] == slot]
                top = cell.sort_values(["epsilon2", "feature"], ascending=[False, True]).iloc[0]
                centroids = [f"ch{c}_c{slot}_spec_centroid" for c in range(2)]
                energies = [f"ch{c}_c{slot}_std" for c in range(2)]
                values = grouped[centroids].to_numpy()
                present = grouped[energies].to_numpy() >= 1e-6
                frequency = float(np.median(values[present])) if present.any() else np.nan
                summaries.append({"task": task, "method": method, "slot": slot,
                                  "scale_rank": scale_order(method, info["frame_length"]).index(slot),
                                  "slot_label": labels[slot], "top_feature": top["feature"],
                                  "max_rank_effect": top["epsilon2"], "max_mi": cell["mi"].max(),
                                  "centroid_cycles_per_sample": frequency,
                                  "conditional_centroid_hz_if_prf_1000": frequency * 1000})
    disc = pd.concat(tables, ignore_index=True)
    disc["kw_p_holm"] = disc.groupby("task")["kw_p"].transform(lambda p: holm(p.to_numpy()))
    disc.to_csv(OUT / "discriminability_pose_groups.csv", index=False)
    scales = pd.DataFrame(summaries)
    scales.to_csv(OUT / "component_scales.csv", index=False)
    return disc, scales


def decomposition_audit(frames):
    normalized = np.stack([normalize_frame(frame)[0] for frame in frames])
    rows, convergence = [], []
    for method in METHODS[1:]:
        summary = json.loads((BASE / "decomp" / f"{method}.json").read_text())
        components = np.load(BASE / "decomp" / f"{method}.npy", mmap_mode="r", allow_pickle=False)
        error = float(np.max(np.abs(components.sum(axis=2, dtype=np.float64) - normalized)))
        assert np.isfinite(components).all() and error < 2e-5
        row = {"method": method, "slots": components.shape[2], "failed_channels": summary["n_failed"],
               "cpu_seconds_per_frame": summary["seconds_per_frame"]["cpu_mean"],
               "run_wall_seconds": summary["wall_seconds_total"], "max_abs_reconstruction_error": error}
        if method.startswith("irmf"):
            diagnostics = pd.read_parquet(BASE / "decomp" / f"{method}.meta.parquet")
            converged = np.stack(diagnostics["stage_converged"])
            iterations = np.stack(diagnostics["stage_iterations"])
            row.update(all_stages_converged_channels=int(converged.all(axis=1).sum()),
                       total_channels=len(converged), unconverged_stages=int((~converged).sum()),
                       total_stages=converged.size,
                       frames_with_unconverged_channel=int((~converged.all(axis=1)).reshape(-1, 2).any(axis=1).sum()))
            for slot, label in enumerate(summary["slot_labels"][:-1]):
                convergence.append({"slot": slot, "slot_label": label,
                                    "unconverged": int((~converged[:, slot]).sum()),
                                    "fraction_unconverged": float((~converged[:, slot]).mean()),
                                    "mean_iterations": float(iterations[:, slot].mean())})
        rows.append(row)
    costs = pd.DataFrame(rows)
    convergence = pd.DataFrame(convergence)
    costs.to_csv(OUT / "decomposition_audit.csv", index=False)
    convergence.to_csv(OUT / "irmf_convergence.csv", index=False)
    return costs, convergence


def make_figures(frame_folds, recording_folds, scales, convergence):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharey=True, layout="constrained")
    for row, task in enumerate(frame_folds):
        for col, feature_set in enumerate(FEATURE_SETS):
            ax = axes[row, col]
            for offset, level, marker, folds in [(-0.08, "Cell frames", "o", frame_folds[task]),
                                                  (0.08, "Recording vote", "s", recording_folds[task])]:
                subset = folds[(folds["classifier"] == "lda") & (folds["feature_set"] == feature_set)]
                summary = subset.groupby("method")["balanced_accuracy"].agg(["mean", "std"]).reindex(METHODS)
                ax.errorbar(np.arange(4) + offset, summary["mean"], yerr=summary["std"],
                            fmt=marker, capsize=3, label=level, color="#275d91" if marker == "o" else "#b36336")
            ax.axhline(0.5 if task == "bird_drone" else 1 / 3, color="#aaa", linestyle=":")
            ax.set_xticks(range(4), [SHORT[m] for m in METHODS])
            ax.set_title(f"{'UAS vs bird' if task == 'bird_drone' else 'Three targets'} | "
                         f"{'without' if feature_set == 'comp' else 'with'} log_scale")
            ax.set_ylim(0.25, 1.03)
            ax.set_ylabel("Balanced accuracy")
            ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Fixed LDA, held-out target-plus-pose groups\nMean and SD of 15 overlapping CV folds; SD is not a confidence interval", fontsize=12)
    fig.savefig(OUT / "grouped_balanced_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax = axes[0]
    subset = scales[scales["task"] == "bird_drone"]
    for method in METHODS:
        cell = subset[subset["method"] == method].sort_values("centroid_cycles_per_sample")
        cell = cell[cell["centroid_cycles_per_sample"] > 0]
        ax.plot(cell["centroid_cycles_per_sample"], cell["max_rank_effect"], "o-",
                markersize=4, label=SHORT[method], color=COLORS[method], linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Median component centroid, cycles/sample\nMultiply by verified PRF for Hz")
    ax.set_ylabel("Maximum rank effect H/(G - 1), G=45")
    ax.set_title("Descriptive separation by realized scale")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    ax = axes[1]
    ax.bar(range(len(convergence)), convergence["fraction_unconverged"] * 100, color=COLORS[METHODS[-1]])
    ax.set_xticks(range(len(convergence)), convergence["slot_label"], rotation=60, ha="right")
    ax.set_ylabel("Channels with unconverged stage, %")
    ax.set_title("IRMF primary run, max_iter=60")
    ax.set_xlabel("Extraction windows, fine to coarse\nFinest solver residual is stored separately")
    ax.grid(axis="y", alpha=0.18)
    fig.savefig(OUT / "component_scales_and_convergence.png", dpi=180)
    plt.close(fig)


def compact_best(table):
    return table[["feature_set", "method", "classifier", "balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"]].rename(
        columns={"balanced_accuracy_mean": "balanced_accuracy", "macro_f1_mean": "macro_f1", "accuracy_mean": "accuracy"})


def comparison_display(table):
    table = table[table["metric"] == "balanced_accuracy"].copy()
    table["difference_95ci"] = table.apply(
        lambda r: f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
    return table[["feature_set", "method_a", "method_b", "difference_95ci", "p_nb_holm", "p_wilcoxon_holm"]]


def write_report(info, best, frame_folds, recording_folds, fixed_pairs, record_pairs, disc, scales, costs, convergence, ties):
    primary = fixed_pairs[info["primary_task"]]
    primary = primary[(primary.metric == "balanced_accuracy") & (primary.method_a == METHODS[-1])]
    wins = primary[(primary.mean_diff > 0) & (primary.p_nb_holm < 0.05)]
    opening = ("IRMF shows no established advantage on the primary UAS-versus-bird task: none of its "
               "six fixed-LDA comparisons passes the Holm-adjusted corrected t-test. "
               if wins.empty else
               f"IRMF passes {len(wins)} of six primary-task fixed-LDA superiority comparisons under "
               "the Holm-adjusted corrected t-test. ")
    lines = ["# UAS versus birds at 60 GHz: classical baselines", "",
             opening + "The comparison uses the public moving-target subset and holds out target-plus-pose groups. "
             "All six classifiers ran in full mode on 5 folds x seeds 0, 1 and 2, for both tasks, "
             "both feature sets and all four methods. The numerical evidence below concerns this indoor "
             "bionic-bird campaign; it does not measure new-session or free-flight performance.", "",
             "## Loader and representation", "",
             "The nonexecuting pickletools scan found exactly three NumPy globals in all 98 files. "
             "The authorized isolated converter allowed only ndarray, dtype and reconstruct, including the "
             "NumPy 2 module alias. It rejected all other globals, persistent/extension mechanisms, object "
             "arrays and unsupported top-level types. Negative checks for os.system, builtins.eval, "
             "object arrays and arbitrary objects passed. No downloaded code ran. Each source was converted "
             "once; source/output SHA-256, dtype and shape are in the external manifest. A second conversion "
             "run reused hash-verified exports. Downstream preparation loads only allow_pickle=False arrays.", "",
             "Each recording contributes its 16 cells with greatest within-recording complex dynamic variance. "
             "The cell's complex mean is removed; I and Q are the two channels. Stable cell-index tie breaking "
             "makes preparation deterministic. Labels play no role in cell selection. N=784, C=2, L=1000. "
             "The binary task has bird 320 and UAS 464 frames; the finer task has bird 320, Mavic 320, P3P 144. "
             "There are 49 recordings, 20/20/9 by target, and 45 CV groups, 18/18/9. Repeated bird settings "
             "and Mavic absorber variants at one pose share a group. Static targets, pole and the empty "
             "placeholder are excluded. Validation found no nonfinite, constant or zero-MAD channels.", "",
             "An independent preparation rerun produced byte-identical frames.npy, meta.parquet and "
             "dataset.json. Hashes are saved in preparation_checks.json; loader checks are in "
             "loader_checks.json.", "",
             "README.docx was read as ZIP/XML. It confirms sample and crop sizes but gives no PRF or flight IDs. "
             "The angle text files hold extracted signal phase, not flight metadata. The whole dataset is "
             "one indoor campaign. The sampling rate is unknown: fs=null, with the plot-based 1000 Hz "
             "hypothesis stored separately. Features remain in cycles/sample.", "",
             "## Best balanced accuracy per method", "",
             "Each row chooses the classifier with greatest mean outer-fold balanced accuracy, with exact "
             "ties resolved by alphabetical classifier name; macro-F1 "
             "and accuracy are for that same classifier. These winners are descriptive and optimistically "
             "selected from the outer results. They are not the basis for the paired superiority tests. "
             "comp excludes log_scale, comp+log_scale includes the shared I/Q amplitude-scale features. "
             "All imputing, scaling and tuning occurred inside training folds; tuning used grouped inner CV.", ""]
    lines += ["Raw has 40 component features, EMD 360, and each IMF method 640, plus two shared "
              "log_scale columns when requested. Feature dimension and classifier regularization can "
              "therefore contribute to differences. EMD has nine storage slots including padding/residue; "
              "the IMF methods have 16, including the finest solver residual.", ""]
    for task, table in best.items():
        lines += [f"### {task}", "", compact_best(table).to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## Paired method verdicts", "",
              "LDA is held fixed across methods and was chosen as the reporting classifier before viewing "
              "scores. Tables give A minus B, with a 95% Nadeau-Bengio corrected resampled interval. "
              "The corrected t-test is primary; Wilcoxon is secondary and optimistic on overlapping folds. "
              "Holm correction covers all 12 comparisons in each task/metric table, six pairs times two "
              "feature sets. The two-sided 0.05 threshold applies to p_nb_holm. The intervals are marginal, "
              "not simultaneous. Separate all-classifier CSVs apply Holm over 72 comparisons per task/metric. "
              "They include balanced accuracy, macro-F1 and accuracy, including comparisons omitted by the "
              "shared runner's default BA/F1-only export.", ""]
    verdict_rows = []
    for task, table in fixed_pairs.items():
        lines += [f"### {task}, frame-level balanced accuracy", "",
                  comparison_display(table).to_markdown(index=False, floatfmt=".4g"), ""]
        irmf = table[(table.metric == "balanced_accuracy") & (table.method_a == METHODS[-1])]
        for _, row in irmf.iterrows():
            conclusion = ("higher" if row.mean_diff > 0 else "lower") if row.p_nb_holm < 0.05 else "inconclusive"
            verdict_rows.append({"task": task, "feature_set": row.feature_set, "against": row.method_b,
                                 "mean_difference": row.mean_diff, "ci_low": row.ci_low, "ci_high": row.ci_high,
                                 "p_nb_holm": row.p_nb_holm, "verdict": conclusion})
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(OUT / "irmf_verdicts.csv", index=False)
    lines += ["### IRMF interpretation", "", verdict.to_markdown(index=False, floatfmt=".4f"), "",
              "A nonsignificant test does not establish equality. Differences that depend on classifier "
              "choice or log_scale are not a universal decomposition advantage. Some secondary Wilcoxon "
              "comparisons pass Holm, including IRMF versus EMD without log_scale on the primary task. "
              "They do not overturn the corrected t-test verdict because these repeated folds overlap.", "",
              "![Fixed-LDA balanced accuracy on held-out groups](grouped_balanced_accuracy.png)", "",
              "Figure 1. Fixed LDA, frame scores and recording-majority scores. Error bars show fold SD, "
              "not independent-sample uncertainty or confidence intervals.", "",
              "## Recording-level aggregation", "",
              "Within each held-out recording, the 16 cell predictions vote once. Ties use the "
              "alphabetically first label; this was deterministic and used no test truth. Recordings "
              "retain their original grouped split. A setting with three correlated recordings still "
              "belongs to one pose group. Corrected-test train/test ratios use pose-group counts. "
              "The following table holds LDA fixed; CSVs contain all classifiers, three metrics and paired "
              "tests. This is a prediction aggregation, not a separately trained recording model.", ""]
    lines += ["EMD, linear IMF and IRMF all reach 1.0 balanced accuracy after recording voting, on "
              "both tasks and both feature sets. This ceiling leaves no observed recording-level "
              "advantage for IRMF and limits what this subset can tell us about decomposition quality.", ""]
    recording_means = []
    for task, folds in recording_folds.items():
        table = folds[folds.classifier == "lda"].groupby(["feature_set", "method"])[METRICS].mean().reset_index()
        table.insert(0, "task", task)
        recording_means.append(table)
        lines += [f"### {task}, recording vote", "", table.drop(columns="task").to_markdown(index=False, floatfmt=".4f"), "",
                  comparison_display(record_pairs[task]).to_markdown(index=False, floatfmt=".4g"), "",
                  f"Across all classifiers and feature sets, {ties[task]} recording votes were tied out of "
                  f"{49 * 3 * 6 * 4 * 2} votes. Each recording appears once per seed and model.", ""]
    pd.concat(recording_means, ignore_index=True).to_csv(OUT / "recording_lda_means.csv", index=False)
    lines += ["## Discriminability by component scale", "",
              "The shared runner exports frame-level Kruskal-Wallis and mutual information, but its "
              "frame p-values treat correlated cells as independent. The extra analysis first takes "
              "the median feature across all cells and recordings within each of 45 pose groups. "
              "The table below gives the three strongest component slots per method for each task. "
              "Rank effect is H/(G-1), the quantity named epsilon2 in the shared code. Mutual information "
              "uses the shared 16-bin, bias-corrected estimator. These are all-data descriptive rankings, "
              "never training-time feature selectors. Group-feature KW p-values have Holm correction "
              "over every feature and method within a task in discriminability_pose_groups.csv.", ""]
    for task in info["tasks"]:
        top = (scales[scales.task == task].sort_values("max_rank_effect", ascending=False)
               .groupby("method", sort=False).head(3))
        lines += [f"### {task}, strongest component scales", "",
                  top[["method", "slot_label", "top_feature", "max_rank_effect", "max_mi",
                       "centroid_cycles_per_sample"]].to_markdown(index=False, floatfmt=".4f"), ""]
    scale_rows = disc[disc.slot == -1][["task", "method", "channel", "epsilon2", "mi", "kw_p_holm"]]
    lines += ["### Shared amplitude scale", "", scale_rows.to_markdown(index=False, floatfmt=".4g"), "",
              "The component centroid measures absolute slow-time spectral frequency, not signed "
              "radial motion, because I and Q are decomposed separately. Fine residuals and short "
              "windows describe rapid fluctuations, while long windows describe slower modulation. "
              "EMD's IMF order is data-adaptive; window size is not a Doppler passband. The realized "
              "centroids in component_scales.csv provide the comparison in cycles/sample. Only if PRF "
              "is verified as 1000 Hz can these centroids be multiplied by 1000 for Hz; at 60 GHz, "
              "v=0.0025 f_D m/s for monostatic Doppler. Aliasing and unknown PRF rule out identifying "
              "blade velocities. IMF residual_fine is the finest slot, not a coarse trend.", "",
              "Separation is spread across many component scales and is already strong in raw spectral "
              "features. Both IMF methods have strong three-target separation at w65, with realized "
              "centroids near 0.027-0.028 cycles/sample. There is no isolated IRMF-only Doppler band "
              "established by this descriptive analysis. log_scale itself separates the groups, so "
              "amplitude and acquisition conditions remain a plausible contributor to the scores.", "",
              "![Scale separation and IRMF convergence](component_scales_and_convergence.png)", "",
              "Figure 2. Left: descriptive group-level rank separation versus realized spectral scale "
              "on the primary task. Maximum-over-features values have selection optimism. Right: "
              "primary IRMF stage nonconvergence, over 1568 channels.", "",
              "## Compute, reconstruction and convergence", "",
              costs[["method", "slots", "failed_channels", "cpu_seconds_per_frame",
                     "run_wall_seconds", "max_abs_reconstruction_error"]].to_markdown(index=False, floatfmt=".6g"), "",
              "IRMF converged at every stage in 1374/1568 channels. There were 355 unconverged stages "
              "out of 23520, affecting 126/784 frames. The w7 stage missed convergence in 143 channels, "
              "9.12%, the highest rate.", "", convergence.to_markdown(index=False, floatfmt=".5f"), "",
              "CPU seconds per frame include both I and Q. Every channel was finite and reconstructed "
              "the normalized input within the displayed float32 error. Primary IRMF kept lookup=True, "
              "max_iter=60 and tol=1e-6. An unconverged stage still contributes a partial estimate and "
              "the residual preserves reconstruction; exact reconstruction does not establish solver "
              "convergence. A fully converged classification sensitivity run has not been performed, "
              "so the verdict is about the package-default feature extractor.", "",
              "## Comparison with published protocols", "",
              "The authors' [CNN code](https://github.com/sk8053/drone_bird_detection_with_radar/blob/"
              "0e991a1b99f8eac2acab420254f656dc434d7f38/drone_bird_classification_with_convnet.py) "
              "concatenates recordings, normalizes by each known class's maximum over the full data, "
              "randomly permutes time samples, then makes a 70/30 train/validation split. This shares "
              "recordings and fitted normalization information across partitions. The "
              "[ConvLSTM code](https://github.com/sk8053/drone_bird_detection_with_radar/blob/"
              "0e991a1b99f8eac2acab420254f656dc434d7f38/drone_bird_classification_with_conv_lstm.py) "
              "forms overlapping five-sample windows before permuting and splitting 30/70. Neighboring "
              "windows can share raw samples across partitions. Our inference from that code is that "
              "those protocols estimate within-recording interpolation. Their near-99% results described "
              "in the dataset notes do not establish performance on unseen recordings. We did not rerun "
              "the leaky neural protocols or treat their accuracy as a paired baseline.", "",
              "[Larrat and Sales, 2025](https://doi.org/10.3390/s25030721) report Transformer white-noise "
              "AUCs of 0.97/0.92/0.98 for bird/Mavic/P3P; those are one-vs-rest AUCs, not balanced "
              "accuracy. Their random validation protocol described in the dataset notes supplies no "
              "recording-held-out guarantee. Task subsets, signal representations, metrics and noise "
              "conditions differ, so our numbers cannot establish superiority over those publications.", "",
              "## Statistical power and limits", "",
              "There are at most 45 conservative pose units, and just nine P3P units, all from one "
              "campaign. Repeating five-fold CV three times does not produce 15 independent experiments "
              "or turn 784 correlated cells into independent samples. The corrected standard error is "
              "sqrt((1/15 + mean(n_test/n_train)) * variance_of_paired_differences), about 2.18 times "
              "the naive fold standard error when the ratio is 1/4. A typical test fold has about nine "
              "pose groups and only one or two P3P groups. Small improvements have low power; absence "
              "of significance is not equivalence. Every outer training and test partition contained "
              "all classes; the saved split audit verifies group and recording separation. "
              "No independent-session power claim is possible.", "",
              "Background, absorber configuration, target identity and mount remain potential confounds. "
              "Selection is label-free, but the strongest dynamic cells are not verified physical wing "
              "or rotor cells. A target-free background control and a max_iter convergence sensitivity "
              "would help separate mechanism from campaign artifacts. The full DataPort cubes may "
              "supply background context and PRF metadata, but may repeat the same recordings; only "
              "new verified recordings or sessions would add independent statistical power.", "",
              "## Reproduce and inspect", "",
              "Code is confined to datasets/uas_birds_60ghz/. Numeric data, converted cubes, feature "
              "caches, shared split files and out-of-fold predictions remain outside the repository. "
              "Small CSV/JSON tables, the two PNG figures and this report live here. No commit was made. "
              "The untested full-release input contract is documented in the dataset README.", "",
              "```bash", "cd /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf",
              "../../.venv/bin/python -m emdirmf.validate uas_birds_60ghz",
              "../../.venv/bin/python -m emdirmf.run_decompose --dataset uas_birds_60ghz --method emd imf_linear_a1.41 irmf_h2_a1.41 --workers 3",
              "../../.venv/bin/python -m emdirmf.run_features --dataset uas_birds_60ghz --method raw emd imf_linear_a1.41 irmf_h2_a1.41 --workers 3",
              "../../.venv/bin/python baselines/run_classical.py --dataset uas_birds_60ghz --task bird_drone --workers 3",
              "../../.venv/bin/python baselines/run_classical.py --dataset uas_birds_60ghz --task target3 --workers 3",
              "../../.venv/bin/python datasets/uas_birds_60ghz/analyze_classical.py", "```", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info, meta, frames = load_info(DATASET), load_meta(DATASET), load_frames(DATASET)
    if info["preparation"]["source"] != "public" or frames.shape != (784, 2, 1000):
        raise ValueError("This report describes the 16-cell public study; review it before changing inputs")
    best, frame_folds, recording_folds, fixed_pairs, record_pairs, ties = {}, {}, {}, {}, {}, {}
    split_audit = []
    for task, column in info["tasks"].items():
        for split in load_splits(DATASET, task):
            train, test = meta.iloc[split.train], meta.iloc[split.test]
            assert not (set(train["group"]) & set(test["group"]))
            assert not (set(train["group_recording"]) & set(test["group_recording"]))
            split_audit.append({"task": task, "split_id": split.split_id,
                                "train_groups": train.group.nunique(), "test_groups": test.group.nunique(),
                                "test_recordings": test.group_recording.nunique(),
                                "test_class_counts": test[column].value_counts().to_dict(),
                                "missing_test_classes": sorted(set(meta[column]) - set(test[column]))})
        folds = pd.read_csv(OUT / f"results_folds_{task}.csv")
        context = json.loads((OUT / f"run_{task}.json").read_text())
        assert not context["quick"] and context["n_splits"] == 15 and len(context["classifiers"]) == 6
        assert len(folds) == 4 * 2 * 6 * 15 and np.isfinite(folds[METRICS]).all().all()
        frame_folds[task] = folds
        best[task] = best_table(folds)
        best[task].to_csv(OUT / f"best_methods_{task}.csv", index=False)
        fixed_pairs[task] = paired_tables(folds, task, "frame")
        recording_folds[task], ties[task] = recording_aggregation(task, meta)
        record_pairs[task] = paired_tables(recording_folds[task], task, "recording")
    (OUT / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n")
    disc, scales = group_discriminability(meta, info)
    costs, convergence = decomposition_audit(frames)
    make_figures(frame_folds, recording_folds, scales, convergence)
    write_report(info, best, frame_folds, recording_folds, fixed_pairs, record_pairs,
                 disc, scales, costs, convergence, ties)
    print("Best frame classifiers:")
    for task, table in best.items():
        print(task, compact_best(table).to_string(index=False))
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
