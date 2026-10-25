"""Plot committed paired results only; never load data or refit a classifier."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[1]
OUTPUT = REPO_ROOT / "output/classification-emd-irmf"
DATASETS = {
    "dronerf": ("DroneRF: 10 classes", "10class", "linsvm"),
    "birdclef2023": ("BirdCLEF: species", "species", "logreg"),
    "esc50_animals": ("ESC-50: animals", "animal", "linsvm"),
    "uas_birds_60ghz": ("60 GHz: UAS vs bird", "bird_drone", "lda"),
    "synthetic": ("Synthetic: smoke test, separate x scale", "signal", "logreg"),
}
MODEL_NAMES = {
    "lda": "LDA", "linsvm": "Linear SVM", "logreg": "Logistic regression",
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "extratrees": "ExtraTrees",
    "minirocket": "MiniRocket", "multirocket": "MultiRocket", "compact_resnet": "CNN",
}
CONTRASTS = {
    "IRMF - EMD": ("irmf_h2_a1.41", "emd"),
    "IRMF - linear IMF": ("irmf_h2_a1.41", "imf_linear_a1.41"),
}


def read_comparisons(dataset):
    """Return source tables and their original per-metric Holm family sizes."""
    _, task, classifier = DATASETS[dataset]
    folder = OUTPUT / dataset
    if dataset == "uas_birds_60ghz":
        classical = folder / f"classical/paired_frame_lda_{task}.csv"
        family_size = 12
    else:
        classical = folder / (f"classical/comparisons_{task}.csv" if dataset == "synthetic"
                              else "classical/paired_all_metrics.csv")
        family_size = 72
    tables = [(classical, classifier, family_size)]
    if dataset == "synthetic":
        tables += [(folder / f"ml/{runner}/comparisons_{task}.csv", None, size)
                   for runner, size in (("trees", 36), ("rocket", 24), ("cnn", 12))]
    elif dataset == "birdclef2023":
        tables.append((folder / f"ml/paired_all_metrics_{task}.csv", None, 120))
    else:
        # ESC-50's requested table also includes supplemental MultiRocket.
        tables.append((folder / "ml/paired_requested.csv", None, None))
    loaded = []
    for path, fixed_classifier, size in tables:
        table = pd.read_csv(path)
        if "task" in table:
            table = table[table.task == task]
        if "level" in table:
            table = table[table.level == "frame"]
        if "subset" in table:
            table = table[table.subset == "all"]
        table = table[table.metric == "balanced_accuracy"].copy()
        if fixed_classifier:
            table = table[table.classifier == fixed_classifier]
        table["source"] = path.relative_to(OUTPUT).as_posix()
        if size is not None:
            table["holm_family_size"] = size
        loaded.append(table)
    return pd.concat(loaded, ignore_index=True)


def oriented_row(table, classifier, feature_set, method_a, method_b):
    """Select one stored contrast, reversing its sign and CI when needed."""
    cell = table[(table.classifier == classifier) & (table.feature_set == feature_set)]
    direct = cell[(cell.method_a == method_a) & (cell.method_b == method_b)]
    reverse = cell[(cell.method_a == method_b) & (cell.method_b == method_a)]
    assert len(direct) + len(reverse) == 1, (classifier, feature_set, method_a, method_b)
    row = (direct if len(direct) else reverse).iloc[0].to_dict()
    if not len(direct):
        row["mean_diff"] = -row["mean_diff"]
        row["ci_low"], row["ci_high"] = -row["ci_high"], -row["ci_low"]
    row["method_a"], row["method_b"] = method_a, method_b
    return row


def collect_values():
    rows = []
    for dataset, (_, task, fixed_classifier) in DATASETS.items():
        table = read_comparisons(dataset)
        for classifier in (fixed_classifier, *list(MODEL_NAMES)[3:]):
            if classifier not in set(table.classifier):
                continue
            for contrast, (method_a, method_b) in CONTRASTS.items():
                for feature_set in ("comp", "comp+log_scale"):
                    row = oriented_row(table, classifier, feature_set, method_a, method_b)
                    rows.append({
                        "dataset": dataset, "task": task, "classifier": classifier,
                        "feature_set": feature_set, "contrast": contrast,
                        "method_a": method_a, "method_b": method_b,
                        "mean_diff_pp": 100 * row["mean_diff"],
                        "ci_low_pp": 100 * row["ci_low"], "ci_high_pp": 100 * row["ci_high"],
                        "p_nb_holm": row["p_nb_holm"], "n_splits": int(row["n_splits"]),
                        "holm_family_size": int(row["holm_family_size"]), "source": row["source"],
                    })
    return pd.DataFrame(rows)


def main():
    values = collect_values()
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    values.to_csv(figures / "contrasts_forest.csv", index=False, float_format="%.12g")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 16.5))
    settings = [("comp", "#0072B2", -0.14), ("comp+log_scale", "#D55E00", 0.14)]
    for dataset_index, (dataset, (title, _, fixed_classifier)) in enumerate(DATASETS.items()):
        cell = values[values.dataset == dataset]
        classifiers = list(dict.fromkeys(cell.classifier))
        for contrast_index, contrast in enumerate(CONTRASTS):
            ax = axes[dataset_index, contrast_index]
            ax.axvline(0, color="0.4", linewidth=0.8)
            for model_index, classifier in enumerate(classifiers):
                for feature_set, color, offset in settings:
                    row = cell[(cell.contrast == contrast) & (cell.classifier == classifier)
                               & (cell.feature_set == feature_set)].iloc[0]
                    ax.errorbar(row.mean_diff_pp, model_index + offset,
                                xerr=np.array([[row.mean_diff_pp - row.ci_low_pp],
                                               [row.ci_high_pp - row.mean_diff_pp]]),
                                fmt="o", color=color, markersize=4, capsize=2, linewidth=1.1,
                                markerfacecolor=color if row.p_nb_holm < 0.05 else "white")
            labels = [f"Phase 2: {MODEL_NAMES[c]}" if c == fixed_classifier else MODEL_NAMES[c]
                      for c in classifiers]
            ax.set_yticks(range(len(labels)), labels if contrast_index == 0 else [])
            ax.set_ylim(len(labels) - 0.5, -0.7)
            ax.set_xlim((-50, 75) if dataset == "synthetic" else (-16, 20))
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
            ax.set_title(f"{title}\n{contrast}", loc="left", fontsize=11)
            ax.set_xlabel("Mean difference in balanced accuracy, percentage points")
    legend = [Line2D([], [], marker="o", color=color, label=label, linewidth=1)
              for (_, color, _), label in zip(settings, ("Without log_scale", "With log_scale"))]
    legend.append(Line2D([], [], marker="o", color="0.3", markerfacecolor="white",
                         linestyle="none", label="Hollow: Holm p ≥ 0.05"))
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle("GD60 IRMF contrasts on matched grouped folds", y=1.004, fontsize=15)
    fig.text(0.5, 0.006,
             "Marginal 95% Nadeau–Bengio corrected CIs; filled markers pass the source table's Holm correction.\n"
             "Synthetic ML uses five quick folds; real-data coverage varies by family. No pooled effect.\n"
             "Source paths, split counts and Holm family sizes are retained in contrasts_forest.csv.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965), h_pad=2)
    fig.savefig(figures / "contrasts_forest.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {len(values)} plotted contrasts to {figures}")


if __name__ == "__main__":
    main()
