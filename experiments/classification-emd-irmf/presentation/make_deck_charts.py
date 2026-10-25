"""Charts for the IRMF classification and robustness Keynote deck.

Every value is read from the study's committed result tables (fractions are
converted to percentage points). The Laplace charts read whatever datasets are
present in ``--laplace DIR`` and are stamped as preliminary unless all four
follow-up datasets are there. The dataset slide's example frames are read,
read-only, from ``--data`` (frame 0, channel 0 of each prepared dataset).

    .venv/bin/python experiments/classification-emd-irmf/presentation/make_deck_charts.py \
        --laplace output/classification-emd-irmf/robustness_laplace

The v3 charts (notebook-scale spikes, each method at its clean-selected K*) read
``--v3 DIR``, default output/classification-emd-irmf/robustness_v3.

PNGs are written at 2x slide resolution (200 px per figure inch, 100 slide
points per inch) to ``--out`` together with ``charts_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "output" / "classification-emd-irmf"
DEFAULT_OUT = Path.home() / "Documents" / "IRMF decks" / "charts"
DEFAULT_DATA = Path.home() / "Datasets" / "imf-classification"

COLORS = {"irmf": "#2a78d6", "linear": "#eb6834", "emd": "#1baf7a", "raw": "#898781"}
INPUT_LABELS = {"irmf": "IRMF", "linear": "Linear IMF", "emd": "EMD", "raw": "Raw"}
TEXT = "#484b4a"
AXIS = "#9b9b9b"
GRID = "#dcdcdc"
RED = "#c60010"
MINUS = "−"

DATASET_LABELS = {
    "dronerf": "DroneRF",
    "birdclef2023": "BirdCLEF 2023",
    "esc50_animals": "ESC-50 animals",
    "uas_birds_60ghz": "60 GHz radar",
    "synthetic": "Synthetic",
}
FOLLOW_UP_ORDER = ["dronerf", "birdclef2023", "esc50_animals", "synthetic"]
CHANCE = {"dronerf": 10.0, "birdclef2023": 100 / 16, "esc50_animals": 10.0, "synthetic": 20.0}
FIXED_CLASSICAL = {
    "dronerf": "linsvm",
    "birdclef2023": "logreg",
    "esc50_animals": "linsvm",
    "uas_birds_60ghz": "lda",
    "synthetic": "logreg",
}
CLASSIFIER_LABELS = {"linsvm": "linear SVM", "logreg": "logistic regression", "lda": "LDA"}
FULL_W, FULL_H = 9.0, 5.2  # inches; 1 inch = 100 slide points

plt.rcParams.update(
    {
        "font.family": "Helvetica Neue",
        "font.weight": "light",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.titleweight": "normal",
        "axes.labelsize": 14.5,
        "axes.labelweight": "light",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 13.5,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "savefig.transparent": True,
    }
)

MANIFEST: dict[str, dict] = {}


def pp(x):
    return 100.0 * np.asarray(x, dtype=float)


def fmt_pp(v, digits=1):
    s = f"{abs(v):.{digits}f}"
    return (MINUS if v < 0 else "+") + s


def signed_ticks(ax, axis="x"):
    """Use the true minus sign and explicit plus on signed axes."""
    def label(v, _):
        if abs(v) < 1e-9:
            return "0"
        digits = 0 if abs(v - round(v)) < 1e-9 else 1
        return fmt_pp(v, digits)

    fmt = matplotlib.ticker.FuncFormatter(label)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def save(fig, out: Path, name: str, meta: dict):
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, transparent=True)
    plt.close(fig)
    MANIFEST[name] = {"file": str(path), **meta}
    print(f"wrote {path}")


def marker_style(color, significant, size=7.5):
    """Filled with a light ring when significant, hollow otherwise."""
    if significant:
        return dict(marker="o", ms=size, mfc=color, mec="white", mew=1.1, color=color)
    return dict(marker="o", ms=size, mfc="white", mec=color, mew=1.6, color=color)


def ci_point(ax, x, y, lo, hi, color, significant, horizontal=True, marker="o", size=7.5):
    kw = marker_style(color, significant, size)
    kw["marker"] = marker
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot([x], [y], ls="none", zorder=3, **kw)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot([x], [y], ls="none", zorder=3, **kw)


def sig_legend_handles(color=TEXT):
    return [
        Line2D([], [], ls="none", **marker_style(color, True), label="filled: Holm p < .05"),
        Line2D([], [], ls="none", **marker_style(color, False), label="hollow: not significant"),
    ]


def zero_line(ax, horizontal=True):
    if horizontal:
        ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    else:
        ax.axhline(0, color=AXIS, lw=0.9, zorder=1)


# --------------------------------------------------------------------------
# Main study
# --------------------------------------------------------------------------

def pick_row(path: Path, *, classifier, method_a, method_b, feature_set="comp", task=None, _flipped=False):
    """One balanced-accuracy test row; a source stored as b - a is negated (CI bounds swap)."""
    df = pd.read_csv(path)
    q = (df["metric"] == "balanced_accuracy") & (df["feature_set"] == feature_set)
    q &= (df["classifier"] == classifier) & (df["method_a"] == method_a) & (df["method_b"] == method_b)
    if task is not None and "task" in df:
        q &= df["task"] == task
    if "subset" in df:
        q &= df["subset"] == "all"
    if "inference_table" in df:
        q &= df["inference_table"] == "primary"
    if "level" in df:
        q &= df["level"] == "frame"
    rows = df[q]
    if rows.empty and not _flipped:
        r = pick_row(path, classifier=classifier, method_a=method_b, method_b=method_a,
                     feature_set=feature_set, task=task, _flipped=True)
        return dict(d=-r["d"], lo=-r["hi"], hi=-r["lo"], p=r["p"])
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row for {classifier} {method_a}-{method_b}, got {len(rows)}")
    r = rows.iloc[0]
    return dict(d=100 * r.mean_diff, lo=100 * r.ci_low, hi=100 * r.ci_high, p=float(r.p_nb_holm))


MAIN_SOURCES = {
    # dataset: (task, classical csv, lightgbm csv)
    "dronerf": ("10class", "dronerf/classical/paired_all_metrics.csv", "dronerf/ml/paired_requested.csv"),
    "birdclef2023": ("species", "birdclef2023/classical/paired_all_metrics.csv",
                     "birdclef2023/ml/paired_all_metrics_species.csv"),
    "esc50_animals": ("animal", "esc50_animals/classical/paired_all_metrics.csv",
                      "esc50_animals/ml/paired_requested.csv"),
    "uas_birds_60ghz": ("bird_drone", "uas_birds_60ghz/classical/paired_frame_lda_bird_drone.csv",
                        "uas_birds_60ghz/ml/paired_requested.csv"),
    "synthetic": ("signal", "synthetic/classical/comparisons_signal.csv",
                  "synthetic/ml/trees/comparisons_signal.csv"),
}
MAIN_CONTRASTS = [
    ("IRMF " + MINUS + " EMD", "irmf_h2_a1.41", "emd", "irmf"),
    ("IRMF " + MINUS + " linear IMF", "irmf_h2_a1.41", "imf_linear_a1.41", "irmf"),
    ("Linear IMF " + MINUS + " EMD", "imf_linear_a1.41", "emd", "linear"),
]


def main_contrast_table(feature_set="comp"):
    rows = []
    for ds, (task, ccsv, mcsv) in MAIN_SOURCES.items():
        for model, csv, clf in (("classical", ccsv, FIXED_CLASSICAL[ds]), ("lightgbm", mcsv, "lightgbm")):
            for label, a, b, _ in MAIN_CONTRASTS:
                r = pick_row(OUT_ROOT / csv, classifier=clf, method_a=a, method_b=b,
                             feature_set=feature_set, task=task)
                rows.append(dict(dataset=ds, model=model, classifier=clf, contrast=label,
                                 feature_set=feature_set, source=csv, **r))
    return pd.DataFrame(rows)


def check_against_forest(tab: pd.DataFrame):
    """The two IRMF contrasts must equal figures/contrasts_forest.csv exactly."""
    forest = pd.read_csv(OUT_ROOT / "figures" / "contrasts_forest.csv")
    names = {"IRMF - EMD": MAIN_CONTRASTS[0][0], "IRMF - linear IMF": MAIN_CONTRASTS[1][0]}
    n = 0
    for _, f in forest.iterrows():
        if f.contrast not in names or f.feature_set != tab.feature_set.iloc[0]:
            continue
        m = tab[(tab.dataset == f.dataset) & (tab.classifier == f.classifier) & (tab.contrast == names[f.contrast])]
        if m.empty:
            continue
        r = m.iloc[0]
        assert abs(r.d - f.mean_diff_pp) < 1e-8 and abs(r.p - f.p_nb_holm) < 1e-12, (f.dataset, f.classifier)
        n += 1
    return n


def chart_main_contrasts(out: Path, tab: pd.DataFrame, datasets, name, xlim=None, title_note="", height=FULL_H):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, height), sharey=True)
    k = FULL_H / height
    fig.subplots_adjust(left=0.165, right=0.985, top=1 - 0.16 * k, bottom=0.14 * k, wspace=0.12)
    ylabels, ypos = [], []
    y = 0
    positions = {}
    for ds in datasets:
        for model in ("classical", "lightgbm"):
            positions[(ds, model)] = y
            y -= 1
        y -= 0.6
    for ax, (label, _, _, color_key) in zip(axes, MAIN_CONTRASTS):
        sub = tab[tab.contrast == label]
        for _, r in sub.iterrows():
            if r.dataset not in datasets:
                continue
            yy = positions[(r.dataset, r.model)]
            ci_point(ax, r.d, yy, r.lo, r.hi, COLORS[color_key], r.p < 0.05,
                     marker="o" if r.model == "classical" else "s", size=7 if r.model == "classical" else 6.5)
        zero_line(ax)
        ax.set_title(label, color=COLORS[color_key], fontsize=15.5, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax)
        if xlim:
            ax.set_xlim(*xlim)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    for ds in datasets:
        ypos.append(positions[(ds, "classical")] - 0.5)
        ylabels.append(DATASET_LABELS[ds])
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylabels, fontsize=14.5)
    for ax in axes:
        ax.set_ylim(y + 0.9, 0.8)
    fig.supxlabel("Difference in balanced accuracy, pp (95% corrected CI)", fontsize=14, y=0.025 * FULL_H / height)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec=TEXT, mew=1.4, label="fixed classical model"),
        Line2D([], [], ls="none", marker="s", ms=6.5, mfc="none", mec=TEXT, mew=1.4, label="LightGBM"),
        *sig_legend_handles(),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.0), handletextpad=0.3,
               columnspacing=1.2, fontsize=13)
    save(fig, out, name, {"datasets": list(datasets), "note": title_note})


def parse_best_table():
    """Best observed BA per input from SUMMARY.md (C = classical, M = ML)."""
    text = (OUT_ROOT / "SUMMARY.md").read_text()
    block = text.split("## Best observed frame balanced accuracy", 1)[1].split("## ", 1)[0]
    lines = [ln for ln in block.splitlines() if ln.startswith("| ") and "<br>" in ln]
    cols = ["raw", "emd", "linear", "irmf", "irmf_conv"]
    keymap = {"DroneRF": "dronerf", "BirdCLEF": "birdclef2023", "ESC-50 animals": "esc50_animals",
              "60 GHz radar": "uas_birds_60ghz", "Synthetic smoke test": "synthetic"}
    best = {}
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        ds = keymap[cells[0].split("<br>")[0].strip()]
        best[ds] = {}
        for col, cell in zip(cols, cells[1:]):
            vals = {}
            for part in cell.split("<br>"):
                m = re.match(r"([CM]): ([0-9.]+) (.+)", part.strip())
                if m:
                    vals[m.group(1)] = (float(m.group(2)), m.group(3))
            best[ds][col] = vals
    return best


def chart_best_ba(out: Path, best):
    datasets = ["dronerf", "birdclef2023", "esc50_animals", "uas_birds_60ghz", "synthetic"]
    inputs = ["raw", "emd", "linear", "irmf"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_H), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.83, bottom=0.12, wspace=0.10)
    bar_h = 0.19
    for ax, kind, title in zip(axes, ("C", "M"), ("Best classical model", "Best ML model")):
        for i, ds in enumerate(datasets):
            for j, key in enumerate(inputs):
                v = best[ds][key].get(kind)
                if v is None:
                    continue
                yy = i + (j - 1.5) * bar_h
                ax.barh(yy, v[0], height=bar_h * 0.86, color=COLORS[key], zorder=2)
                ax.text(v[0] + 0.8, yy, f"{v[0]:.1f}", va="center", ha="left", fontsize=11.5, color=TEXT)
        ax.set_title(title, fontsize=15, pad=6)
        ax.set_xlim(0, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(datasets)))
    axes[0].set_yticklabels([DATASET_LABELS[d] for d in datasets], fontsize=14.5)
    axes[0].set_ylim(len(datasets) - 0.45, -0.55)
    fig.supxlabel("Balanced accuracy, % (best model per input, outer CV)", fontsize=14, y=0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], label=INPUT_LABELS[k] + (" (pimf default)" if k == "irmf" else ""))
               for k in inputs]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.57, 1.0), handlelength=1.2)
    save(fig, out, "main_best_ba", {"source": "SUMMARY.md best observed table"})


# --------------------------------------------------------------------------
# Component count
# --------------------------------------------------------------------------

def chart_ba_vs_k(out: Path):
    s = pd.read_csv(OUT_ROOT / "components" / "score_summary.csv")
    s = s[s.feature_set == "comp"]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_H))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.14, wspace=0.42)
    defaults = {}
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        d = s[(s.dataset == ds) & (s.classifier == FIXED_CLASSICAL[ds])]
        get = lambda m: float(d.loc[d.method == m, "balanced_accuracy_mean"].iloc[0]) * 100  # noqa: E731
        k_default = 15 if ds == "synthetic" else 17
        native = [3, 4, 5, 6, 9, 13]
        series = {
            "irmf": ([*native, k_default], [*(get(f"irmf_native_k{k}") for k in native), get("irmf_h2_a1.41_conv")]),
            "linear": ([*native, k_default], [*(get(f"linear_native_k{k}") for k in native), get("imf_linear_a1.41")]),
            "emd": ([3, 4, 5, 6, 7, 8], [*(get(f"emd_k{k}") for k in [3, 4, 5, 6, 7]), get("emd")]),
        }
        defaults[ds] = {k: v[1][-1] for k, v in series.items()}
        for key in ("emd", "linear", "irmf"):
            ks, ba = series[key]
            ax.plot(ks, ba, color=COLORS[key], lw=1.6, zorder=2)
            ax.plot(ks[:-1], ba[:-1], ls="none", **marker_style(COLORS[key], True, 6.0), zorder=3)
            ax.plot(ks[-1:], ba[-1:], ls="none", marker="D", ms=6.5, mfc=COLORS[key], mec="white", mew=1.1, zorder=4)
        ax.set_title(DATASET_LABELS[ds], fontsize=14.5, pad=6)
        ax.set_xticks([3, 6, 9, 13, 17] if ds != "synthetic" else [3, 6, 9, 15])
        ax.set_xlim(2, 18)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("Balanced accuracy, %")
    fig.supxlabel("Number of extracted components K", fontsize=14, y=0.02)
    handles = [Line2D([], [], lw=1.6, **marker_style(COLORS[k], True, 6.0), label=INPUT_LABELS[k])
               for k in ("irmf", "linear", "emd")]
    handles.append(Line2D([], [], ls="none", marker="D", ms=6.5, mfc=TEXT, mec="white", label="study default"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0))
    save(fig, out, "comp_ba_vs_k", {"defaults": defaults})


def chart_equal_k(out: Path):
    fa = pd.read_csv(OUT_ROOT / "components" / "family_a.csv")
    fa = fa[(fa.variant == "classical/comp") & (fa.metric == "balanced_accuracy")]
    contrasts = [("IRMF - EMD", "irmf", "o", -0.22, "IRMF " + MINUS + " EMD"),
                 ("Linear IMF - EMD", "linear", "o", 0.0, "Linear IMF " + MINUS + " EMD"),
                 ("IRMF - Linear IMF", "irmf", "D", 0.22, "IRMF " + MINUS + " linear IMF")]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_H), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.76, bottom=0.14, wspace=0.12)
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        for cname, key, mk, off, _ in contrasts:
            sub = fa[(fa.dataset == ds) & (fa.contrast == cname)]
            for _, r in sub.iterrows():
                ci_point(ax, r.k + off, pp(r.mean_diff), pp(r.ci_low), pp(r.ci_high), COLORS[key],
                         r.p_nb_holm < 0.05, horizontal=False, marker=mk, size=6.5 if mk == "o" else 5.8)
        zero_line(ax, horizontal=False)
        ax.axvspan(4.5, 6.5, color="#000000", alpha=0.045, lw=0, zorder=0)
        ax.set_xticks([3, 4, 5, 6])
        ax.set_xlim(2.5, 6.5)
        ax.set_title(DATASET_LABELS[ds], fontsize=14.5, pad=6)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax, "y")
    axes[0].set_ylabel("Difference in balanced accuracy, pp")
    fig.supxlabel("Matched component count K (shaded: pre-registered K = 5, 6)", fontsize=14, y=0.02)
    handles = [Line2D([], [], ls="none", marker=mk, ms=6.5, mfc=COLORS[key], mec="white", label=lab)
               for _, key, mk, _, lab in contrasts]
    handles += sig_legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0), handletextpad=0.2,
               columnspacing=1.2, fontsize=13.5)
    save(fig, out, "comp_equal_k", {"source": "components/family_a.csv"})


def chart_kmax8(out: Path):
    fc = pd.read_csv(OUT_ROOT / "components" / "family_c.csv")
    fc = fc[(fc.variant == "classical/comp") & (fc.metric == "balanced_accuracy")]
    panels = [("GD60 k_max=8 - GD60 full", "pimf defaults " + MINUS + " same solver, all stages"),
              ("GD60 k_max=8 - EMD K=8", "pimf defaults " + MINUS + " EMD default")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 4.3), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.80, bottom=0.17, wspace=0.10)
    for ax, (cname, title) in zip(axes, panels):
        for i, ds in enumerate(FOLLOW_UP_ORDER):
            r = fc[(fc.dataset == ds) & (fc.contrast == cname)].iloc[0]
            d, lo, hi = pp(r.mean_diff), pp(r.ci_low), pp(r.ci_high)
            ci_point(ax, d, i, lo, hi, COLORS["irmf"], r.p_nb_holm < 0.05)
            ax.text(d, i - 0.2, fmt_pp(d), fontsize=12.5, va="bottom", ha="center", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="#f3f3f3", ec="none", alpha=0.9))
        zero_line(ax)
        ax.set_title(title, fontsize=14.5, pad=6)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_xlim(-14, 3)
    axes[1].set_xlim(-8, 36)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels([DATASET_LABELS[d] for d in FOLLOW_UP_ORDER], fontsize=14.5)
    axes[0].set_ylim(3.6, -0.6)
    fig.supxlabel("Difference in balanced accuracy, pp (95% corrected CI)", fontsize=14, y=0.02)
    fig.legend(handles=sig_legend_handles(COLORS["irmf"]), loc="upper center", ncol=2, bbox_to_anchor=(0.57, 1.0))
    save(fig, out, "comp_kmax8", {"source": "components/family_c.csv"})


# --------------------------------------------------------------------------
# Robustness, shared by the impulse and Laplace experiments
# --------------------------------------------------------------------------

def ba_curves(out: Path, scores: pd.DataFrame, datasets, conditions, xlabels, name, xlabel, stamp=None,
              size=(FULL_W, FULL_H)):
    s = scores[(scores.scenario == "shift") & (scores.feature_set == "comp")]
    method_key = {"raw": "raw", "emd": "emd", "imf_linear_a1.41": "linear", "irmf_h2_a1.41_conv": "irmf"}
    n = len(datasets)
    fig, axes = plt.subplots(1, max(n, 1), figsize=size, squeeze=False)
    axes = axes[0]
    narrow = size[0] < 7
    fig.subplots_adjust(left=0.075 if n > 1 else 0.12, right=0.99, top=0.76 if narrow else 0.80, bottom=0.14,
                        wspace=0.35)
    x = np.arange(len(conditions))
    values = {}
    for ax, ds in zip(axes, datasets):
        d = s[(s.dataset == ds) & (s.classifier == FIXED_CLASSICAL[ds])]
        for method in ("raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41_conv"):
            key = method_key[method]
            y = [float(d.loc[(d.method == method) & (d.condition == c), "balanced_accuracy_mean"].iloc[0]) * 100
                 for c in conditions]
            values.setdefault(ds, {})[key] = y
            ax.plot(x, y, color=COLORS[key], lw=1.6, zorder=3 if key == "irmf" else 2)
            ax.plot(x, y, ls="none", **marker_style(COLORS[key], True, 6.0), zorder=4 if key == "irmf" else 3)
        ax.axhline(CHANCE[ds], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(x[0] - 0.2, CHANCE[ds], "chance", fontsize=11.5, ha="left", va="bottom", color="#7a7a7a")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=12.5)
        ax.set_xlim(-0.3, len(conditions) - 0.7)
        ax.set_ylim(0, None)
        ax.set_title(DATASET_LABELS[ds] + (" (preliminary)" if stamp else ""), fontsize=14.5, pad=6,
                     color=RED if stamp else TEXT)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("Balanced accuracy, %")
    fig.supxlabel(xlabel, fontsize=14, y=0.02)
    handles = [Line2D([], [], lw=1.6, **marker_style(COLORS[k], True, 6.0), label=INPUT_LABELS[k])
               for k in ("irmf", "linear", "emd", "raw")]
    fig.legend(handles=handles, loc="upper center", ncol=2 if narrow else 4, bbox_to_anchor=(0.53, 1.0))
    save(fig, out, name, {"datasets": list(datasets), "values": values, "stamp": stamp})


def did_chart(out: Path, tests: pd.DataFrame, datasets, series, conditions, xlabels, name, xlabel, stamp=None,
              size=(FULL_W, FULL_H), ylabel="Change in IRMF gap, pp"):
    """series: list of (label, filter dict, condition prefix or None, marker, offset)."""
    n = len(datasets)
    fig, axes = plt.subplots(1, max(n, 1), figsize=size, sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.085 if n > 1 else 0.13, right=0.99, top=0.76 if size[0] < 7 else 0.80, bottom=0.14, wspace=0.12)
    x = np.arange(len(conditions))
    values = {}
    for ax, ds in zip(axes, datasets):
        for label, filt, prefix, mk, off, color in series:
            q = (tests.dataset == ds) & (tests.contrast == "DiD")
            for col, val in filt.items():
                q &= tests[col] == val
            q &= tests.classifier == FIXED_CLASSICAL[ds]
            for xi, cond in zip(x, conditions):
                c = f"{prefix}{cond}" if prefix else cond
                r = tests[q & (tests.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ci_point(ax, xi + off, pp(r.mean_diff), pp(r.ci_low), pp(r.ci_high), color,
                         r.p_nb_holm < 0.05, horizontal=False, marker=mk, size=6.5 if mk == "o" else 5.8)
                values.setdefault(ds, {}).setdefault(label, []).append(
                    dict(condition=c, did_pp=float(pp(r.mean_diff)), p_holm=float(r.p_nb_holm)))
        zero_line(ax, horizontal=False)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=12.5)
        ax.set_xlim(-0.5, len(conditions) - 0.5)
        ax.set_title(DATASET_LABELS[ds] + (" (preliminary)" if stamp else ""), fontsize=14.5, pad=6,
                     color=RED if stamp else TEXT)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax, "y")
    axes[0].set_ylabel(ylabel)
    fig.supxlabel(xlabel, fontsize=14, y=0.02)
    handles = [Line2D([], [], ls="none", marker=mk, ms=6.5, mfc=color, mec="white", label=label)
               for label, _, _, mk, _, color in series]
    handles += sig_legend_handles()
    narrow = size[0] < 7
    fig.legend(handles=handles, loc="upper center", ncol=2 if narrow else len(handles),
               bbox_to_anchor=(0.55 if narrow else 0.53, 1.0), handletextpad=0.2, columnspacing=1.1,
               fontsize=13 if narrow else 12.5)
    save(fig, out, name, {"datasets": list(datasets), "values": values, "stamp": stamp})


IMPULSE_RATES = ["0.002", "0.01", "0.04"]
IMPULSE_LABELS = ["0.2%", "1%", "4%"]


def impulse_charts(out: Path):
    root = OUT_ROOT / "robustness"
    scores = pd.read_csv(root / "score_summary.csv")
    ba_curves(out, scores, FOLLOW_UP_ORDER, ["clean"] + [f"impulse2_{r}" for r in IMPULSE_RATES],
              ["clean"] + IMPULSE_LABELS, "imp_ba_curves", "Share of test samples with two-sided spikes (A = 10)")
    tests = pd.read_csv(root / "paired_tests.csv")
    base = dict(family="Primary", scenario="shift", feature_set="comp", comparator="imf_linear_a1.41")
    did_chart(out, tests, FOLLOW_UP_ORDER,
              [("one-sided spikes", base, "impulse1_", "o", -0.12, COLORS["irmf"]),
               ("two-sided spikes", base, "impulse2_", "D", 0.12, COLORS["irmf"])],
              IMPULSE_RATES, IMPULSE_LABELS, "imp_did_linear", "Share of test samples with spikes",
              ylabel="Change in IRMF " + MINUS + " linear IMF, pp")
    s1 = dict(family="S1", scenario="shift", feature_set="comp", comparator="emd")
    did_chart(out, tests, FOLLOW_UP_ORDER,
              [("one-sided spikes", s1, "impulse1_", "o", -0.12, COLORS["irmf"]),
               ("two-sided spikes", s1, "impulse2_", "D", 0.12, COLORS["irmf"])],
              IMPULSE_RATES, IMPULSE_LABELS, "imp_did_emd", "Share of test samples with spikes",
              ylabel="Change in IRMF " + MINUS + " EMD, pp")


def laplace_charts(out: Path, laplace_dir: Path):
    scores = pd.read_csv(laplace_dir / "score_summary.csv")
    tests = pd.read_csv(laplace_dir / "paired_tests.csv")
    present = [d for d in FOLLOW_UP_ORDER if d in set(scores.dataset)]
    final = present == FOLLOW_UP_ORDER
    stamp = None if final else "Preliminary: " + ", ".join(DATASET_LABELS[d].lower() for d in present) + \
        " only, real datasets running"
    levels = ["0.1", "0.2", "0.3", "0.4"]
    labels = ["10%", "20%", "30%", "40%"]
    size = (FULL_W, 4.6) if final else (5.4, FULL_H)
    ba_curves(out, scores, present, ["clean"] + [f"laplace_{p}" for p in levels],
              ["clean"] + [lab.rstrip("%") for lab in labels], "lap_ba_curves",
              "Share of test samples with Laplace noise, % (b = 1 robust SD)", stamp=stamp, size=size)
    did_chart(out, tests, present,
              [("vs linear IMF (primary)", dict(family="Primary", scenario="shift", feature_set="comp",
                                                comparator="imf_linear_a1.41"), "laplace_", "D", -0.12,
                COLORS["irmf"]),
               ("vs EMD (S1)", dict(family="S1", scenario="shift", feature_set="comp", comparator="emd"),
                "laplace_", "o", 0.12, COLORS["irmf"])],
              levels, labels, "lap_did", "Share of test samples with Laplace noise", stamp=stamp, size=size,
              ylabel="Change in IRMF gap, pp")
    reading = pd.read_csv(laplace_dir / "prespecified_reading.csv")
    MANIFEST["laplace_status"] = {"dir": str(laplace_dir), "datasets": present, "final": final,
                                  "reading": reading.to_dict(orient="records")}
    if final:
        laplace_matched_chart(out, scores, present)
        spike_vs_laplace_chart(out, laplace_dir)


METHOD_KEY = {"irmf_h2_a1.41_conv": "irmf", "imf_linear_a1.41": "linear", "emd": "emd", "raw": "raw"}


def laplace_matched_chart(out: Path, scores: pd.DataFrame, datasets):
    """Clean, clean-trained at 40% and trained-and-tested at 40%, per input (bars in input colors)."""
    s = scores[scores.feature_set == "comp"]
    groups = [("shift", "clean", "clean"), ("shift", "laplace_0.4", "test\nat 40%"),
              ("matched", "laplace_0.4", "train + test\nat 40%")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(FULL_W, 4.6), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.35)
    bar_w = 0.2
    values = {}
    for ax, ds in zip(axes, datasets):
        d = s[(s.dataset == ds) & (s.classifier == FIXED_CLASSICAL[ds])]
        for j, (method, key) in enumerate(METHOD_KEY.items()):
            ys = []
            for scen, cond, _ in groups:
                r = d[(d.scenario == scen) & (d.condition == cond) & (d.method == method)]
                ys.append(100 * float(r.balanced_accuracy_mean.iloc[0]))
            values.setdefault(ds, {})[key] = dict(zip(["clean", "shift_40", "matched_40"], ys))
            ax.bar(np.arange(3) + (j - 1.5) * bar_w, ys, width=bar_w * 0.9, color=COLORS[key], zorder=2)
        ax.axhline(CHANCE[ds], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[2] for g in groups], fontsize=11)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, None)
        ax.set_title(DATASET_LABELS[ds], fontsize=14.5, pad=6)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("Balanced accuracy, %")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], label=INPUT_LABELS[k]) for k in METHOD_KEY.values()]
    handles.append(Line2D([], [], color=AXIS, lw=0.8, ls=(0, (2, 2)), label="chance"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), handlelength=1.2)
    save(fig, out, "lap_matched", {"values": values})


def spike_vs_laplace_chart(out: Path, laplace_dir: Path):
    """Primary gap changes (IRMF - linear IMF) of the spike and Laplace experiments, one panel per dataset."""
    spike = pd.read_csv(OUT_ROOT / "robustness" / "primary_tests.csv")
    lap = pd.read_csv(laplace_dir / "primary_tests.csv")
    series = [(spike, "impulse1_", IMPULSE_RATES, 0, -0.13, "o", "one-sided spikes"),
              (spike, "impulse2_", IMPULSE_RATES, 0, 0.13, "D", "two-sided spikes"),
              (lap, "laplace_", ["0.1", "0.2", "0.3", "0.4"], 4, 0.0, "s", "Laplace noise")]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_H), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.15, wspace=0.12)
    values = {}
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        for tab, prefix, levels, x0, off, mk, label in series:
            for i, lev in enumerate(levels):
                r = tab[(tab.dataset == ds) & (tab.condition == prefix + lev) & (tab.contrast == "DiD")]
                if len(r) != 1:
                    raise ValueError(f"{ds} {prefix}{lev}: {len(r)} rows")
                r = r.iloc[0]
                ci_point(ax, x0 + i + off, pp(r.mean_diff), pp(r.ci_low), pp(r.ci_high), COLORS["irmf"],
                         r.p_nb_holm < 0.05, horizontal=False, marker=mk, size=6.0 if mk == "o" else 5.5)
                values.setdefault(ds, {}).setdefault(label, []).append(
                    dict(condition=prefix + lev, did_pp=float(pp(r.mean_diff)), p_holm=float(r.p_nb_holm)))
        zero_line(ax, horizontal=False)
        ax.axvline(3, color=GRID, lw=0.9, zorder=0)
        ax.set_xticks([0, 1, 2, 4, 5, 6, 7])
        ax.set_xticklabels(["0.2", "1", "4", "10", "20", "30", "40"], fontsize=11)
        ax.set_xlim(-0.6, 7.5)
        ax.text(1, -0.1, "spikes, %", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12)
        ax.text(5.5, -0.1, "Laplace, %", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12)
        ax.set_title(DATASET_LABELS[ds], fontsize=14.5, pad=6)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax, "y")
    axes[0].set_ylabel("Change in IRMF " + MINUS + " linear IMF, pp")
    handles = [Line2D([], [], ls="none", marker=mk, ms=6.5, mfc=COLORS["irmf"], mec="white", label=label)
               for *_, mk, label in series]
    handles += sig_legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), handletextpad=0.1,
               columnspacing=0.9, fontsize=11.5)
    save(fig, out, "spike_vs_laplace_did", {"values": values})


# --------------------------------------------------------------------------
# Robustness v3: notebook-scale spikes, 31-sample stop, each method at its K*
# --------------------------------------------------------------------------

V3_LEVELS = ["0.1", "0.2", "0.3", "0.4"]
V3_LABELS = ["10%", "20%", "30%", "40%"]
V3_FAMILIES = ("irmf", "linear", "emd", "raw")


def v3_load(v3_dir: Path):
    """best_vs_best_ba, k_star and paired_tests, checked against each other.

    Every best-vs-best row must use the K* of k_star.csv, and every paired test the K* of both arms.
    """
    bvb = pd.read_csv(v3_dir / "best_vs_best_ba.csv")
    kstar = pd.read_csv(v3_dir / "k_star.csv")
    tests = pd.read_csv(v3_dir / "paired_tests.csv")
    kmap = {(r.dataset, r.classifier, r.method): int(r.k) for r in kstar.itertuples()}
    for r in bvb.itertuples():
        assert kmap[(r.dataset, r.classifier, r.family)] == int(r.k), r
    for r in tests.itertuples():
        assert kmap[(r.dataset, r.classifier, "irmf")] == int(r.irmf_k), r
        assert kmap[(r.dataset, r.classifier, r.comparator)] == int(r.comparator_k), r
    return bvb, kmap, tests


def v3_ba(bvb, ds, clf, family, cond, scenario="shift"):
    r = bvb[(bvb.dataset == ds) & (bvb.classifier == clf) & (bvb.family == family) & (bvb.condition == cond)
            & (bvb.scenario == scenario)]
    assert len(r) == 1, (ds, clf, family, cond, scenario)
    return 100 * float(r["mean"].iloc[0])


def v3_k_line(kmap, ds, clf):
    return (f"K: IRMF {kmap[(ds, clf, 'irmf')]}, linear {kmap[(ds, clf, 'linear')]}, "
            f"EMD {kmap[(ds, clf, 'emd')]}")


def v3_panel_title(ax, kmap, ds, clf, title=None):
    ax.set_title(title or DATASET_LABELS[ds], fontsize=14.5, pad=24)
    ax.text(0.5, 1.025, v3_k_line(kmap, ds, clf), transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, color="#6a6a6a")


def v3_curves(ax, bvb, ds, clf):
    x = np.arange(len(V3_LEVELS) + 1)
    conds = ["clean"] + [f"nb3_{p}" for p in V3_LEVELS]
    values = {}
    for key in ("raw", "emd", "linear", "irmf"):
        y = [v3_ba(bvb, ds, clf, key, c) for c in conds]
        values[key] = y
        ax.plot(x, y, color=COLORS[key], lw=1.6, zorder=3 if key == "irmf" else 2)
        ax.plot(x, y, ls="none", **marker_style(COLORS[key], True, 6.0), zorder=4 if key == "irmf" else 3)
    ax.axhline(CHANCE[ds], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(x[0] - 0.2, CHANCE[ds] - 0.6, "chance", fontsize=11.5, ha="left", va="top", color="#7a7a7a")
    ax.set_xticks(x)
    ax.set_xticklabels(["clean"] + [lab.rstrip("%") for lab in V3_LABELS], fontsize=12.5)
    ax.set_xlim(-0.3, len(conds) - 0.7)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    return values


def v3_ba_chart(out: Path, bvb, kmap):
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_H))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.76, bottom=0.14, wspace=0.35)
    values = {}
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        clf = FIXED_CLASSICAL[ds]
        values[ds] = v3_curves(ax, bvb, ds, clf)
        v3_panel_title(ax, kmap, ds, clf)
    axes[0].set_ylabel("Balanced accuracy, %")
    fig.supxlabel("Share of test samples with spikes, % (scale 3σ, about 1.5 H)", fontsize=14, y=0.02)
    handles = [Line2D([], [], lw=1.6, **marker_style(COLORS[k], True, 6.0), label=INPUT_LABELS[k])
               for k in ("irmf", "linear", "emd", "raw")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0))
    save(fig, out, "v3_ba_curves", {"values": values,
                                    "k": {ds: v3_k_line(kmap, ds, FIXED_CLASSICAL[ds]) for ds in FOLLOW_UP_ORDER}})


def v3_did_chart(out: Path, tests, kmap):
    """Change in the IRMF gap vs linear IMF (Primary) and vs EMD (S1), clean-trained classical model at K*."""
    series = [("vs linear IMF (primary)", "Primary", "linear", "D", -0.12),
              ("vs EMD (S1)", "S1", "emd", "o", 0.12)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_H), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.76, bottom=0.14, wspace=0.24)
    x = np.arange(len(V3_LEVELS))
    values = {}
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        clf = FIXED_CLASSICAL[ds]
        for label, family, comp, mk, off in series:
            for xi, lev in zip(x, V3_LEVELS):
                r = tests[(tests.dataset == ds) & (tests.family == family) & (tests.comparator == comp)
                          & (tests.classifier == clf) & (tests.scenario == "shift") & (tests.feature_set == "comp")
                          & (tests.condition == f"nb3_{lev}") & (tests.contrast == "DiD")]
                if len(r) != 1:
                    raise ValueError(f"{ds} {family} {lev}: {len(r)} rows")
                r = r.iloc[0]
                ci_point(ax, xi + off, pp(r.mean_diff), pp(r.ci_low), pp(r.ci_high), COLORS["irmf"],
                         r.p_nb_holm < 0.05, horizontal=False, marker=mk, size=6.5 if mk == "o" else 5.8)
                values.setdefault(ds, {}).setdefault(label, []).append(
                    dict(condition=f"nb3_{lev}", did_pp=float(pp(r.mean_diff)), p_holm=float(r.p_nb_holm),
                         irmf_k=int(r.irmf_k), comparator_k=int(r.comparator_k)))
        zero_line(ax, horizontal=False)
        ax.set_xticks(x)
        ax.set_xticklabels(V3_LABELS, fontsize=12.5)
        ax.set_xlim(-0.5, len(V3_LEVELS) - 0.5)
        v3_panel_title(ax, kmap, ds, clf)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        signed_ticks(ax, "y")
    axes[0].set_ylabel("Change in IRMF gap, pp")
    fig.supxlabel("Share of test samples with spikes", fontsize=14, y=0.02)
    handles = [Line2D([], [], ls="none", marker=mk, ms=6.5, mfc=COLORS["irmf"], mec="white", label=label)
               for label, _, _, mk, _ in series]
    handles += sig_legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), handletextpad=0.2,
               columnspacing=1.1, fontsize=12.5)
    save(fig, out, "v3_did", {"values": values})


def v3_lgbm_matched_chart(out: Path, bvb, kmap):
    """Left: ESC-50 LightGBM curves at its K*. Right: classical, trained and tested at 40%, real datasets."""
    fig = plt.figure(figsize=(FULL_W, 4.5))
    ax_l = fig.add_axes([0.085, 0.15, 0.33, 0.60])
    ax_r = fig.add_axes([0.53, 0.15, 0.46, 0.60])
    ds = "esc50_animals"
    lgbm = v3_curves(ax_l, bvb, ds, "lightgbm")
    v3_panel_title(ax_l, kmap, ds, "lightgbm", "ESC-50 animals, LightGBM")
    ax_l.set_ylabel("Balanced accuracy, %")
    ax_l.set_xlabel("Share of test samples with spikes, %", fontsize=13)
    real = ["dronerf", "birdclef2023", "esc50_animals"]
    bar_w = 0.2
    matched = {}
    for j, key in enumerate(V3_FAMILIES):
        for i, d in enumerate(real):
            clf = FIXED_CLASSICAL[d]
            m = v3_ba(bvb, d, clf, key, "nb3_0.4", "matched")
            sh = v3_ba(bvb, d, clf, key, "nb3_0.4")
            matched.setdefault(d, {})[key] = dict(matched_40=m, shift_40=sh, k=kmap[(d, clf, key)])
            xx = i + (j - 1.5) * bar_w
            ax_r.bar(xx, m, width=bar_w * 0.9, color=COLORS[key], zorder=2)
            ax_r.plot([xx - bar_w * 0.42, xx + bar_w * 0.42], [sh, sh], color="#2b2b2b", lw=2.0, zorder=3,
                      solid_capstyle="butt")
            if key != "raw":
                ax_r.text(xx, 1.2, str(kmap[(d, clf, key)]), ha="center", va="bottom", fontsize=10,
                          color="white", zorder=4)
    for i, d in enumerate(real):
        ax_r.plot([i - 0.45, i + 0.45], [CHANCE[d]] * 2, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax_r.set_xticks(range(len(real)))
    ax_r.set_xticklabels([DATASET_LABELS[d] for d in real], fontsize=12.5)
    ax_r.tick_params(axis="x", length=0)
    ax_r.set_xlim(-0.5, len(real) - 0.5)
    ax_r.set_ylim(0, 70)
    ax_r.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax_r.set_title("Classical model at 40%", fontsize=14.5, pad=24)
    ax_r.text(0.5, 1.025, "bars: trained and tested at 40%; dark line: clean-trained; K in bar", fontsize=11.5,
              ha="center", va="bottom", color="#6a6a6a", transform=ax_r.transAxes)
    handles = [Line2D([], [], lw=1.6, **marker_style(COLORS[k], True, 6.0), label=INPUT_LABELS[k])
               for k in ("irmf", "linear", "emd", "raw")]
    handles.append(Line2D([], [], color=AXIS, lw=0.8, ls=(0, (2, 2)), label="chance"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0))
    save(fig, out, "v3_lgbm_matched", {"lightgbm_esc50": lgbm, "matched": matched,
                                       "k_lightgbm": v3_k_line(kmap, ds, "lightgbm")})


def v3_charts(out: Path, v3_dir: Path):
    bvb, kmap, tests = v3_load(v3_dir)
    v3_ba_chart(out, bvb, kmap)
    v3_did_chart(out, tests, kmap)
    v3_lgbm_matched_chart(out, bvb, kmap)
    MANIFEST["v3_status"] = {"dir": str(v3_dir), "k_star": {"/".join(k): v for k, v in kmap.items()}}


# --------------------------------------------------------------------------
# Example frames for the dataset slide
# --------------------------------------------------------------------------

EXAMPLE_LABEL = {"dronerf": ("drone", "DroneRF, {}"),
                 "birdclef2023": ("species_common", "BirdCLEF, {}"),
                 "esc50_animals": ("y_animal", "ESC-50, {}"),
                 "synthetic": ("y_signal", "Synthetic, {}")}


def example_frames_chart(out: Path, data_root: Path, frame=0, channel=0):
    """Frame 0, channel 0 of each prepared dataset after the shared median / MAD normalization."""
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.9))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.17, wspace=0.18)
    meta_out = {}
    for ax, ds in zip(axes, FOLLOW_UP_ORDER):
        frames = np.load(data_root / ds / "prepared" / "frames.npy", mmap_mode="r", allow_pickle=False)
        meta = pd.read_parquet(data_root / ds / "prepared" / "meta.parquet")
        y = np.asarray(frames[frame, channel], dtype=float)
        y = (y - np.median(y)) / (1.4826 * np.median(np.abs(y - np.median(y))))
        col, fmt = EXAMPLE_LABEL[ds]
        label = str(meta.iloc[frame][col])
        ax.plot(np.arange(y.size), y, color=COLORS["raw"], lw=0.35)
        ax.set_title(fmt.format(label), fontsize=12, pad=4)
        ax.set_xlim(0, y.size)
        ax.set_xticks([0, y.size])
        ax.tick_params(labelsize=10)
        for tick, ha in zip(ax.get_xticklabels(), ("left", "right")):
            tick.set_horizontalalignment(ha)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        signed_ticks(ax, "y")
        meta_out[ds] = dict(frame=frame, channel=channel, label=label, length=int(y.size))
    save(fig, out, "dataset_frames", {"frames": meta_out})


def radar_frames_chart(out: Path, data_root: Path, channel=0):
    """First bird and first drone frame of the 60 GHz radar set, I channel, median / MAD normalized."""
    ds = "uas_birds_60ghz"
    frames = np.load(data_root / ds / "prepared" / "frames.npy", mmap_mode="r", allow_pickle=False)
    meta = pd.read_parquet(data_root / ds / "prepared" / "meta.parquet")
    picks = [int(meta.index[meta.y_bird_drone == label][0]) for label in ("bird", "uas")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.9))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.17, wspace=0.12)
    meta_out = {}
    for ax, frame in zip(axes, picks):
        row = meta.iloc[frame]
        y = np.asarray(frames[frame, channel], dtype=float)
        y = (y - np.median(y)) / (1.4826 * np.median(np.abs(y - np.median(y))))
        name = "toy bird" if row.target == "bird" else {"mavic": "DJI Mavic", "p3p": "DJI Phantom 3 Pro"}.get(
            row.target, str(row.target))
        ax.plot(np.arange(y.size), y, color=COLORS["raw"], lw=0.45)
        ax.set_title(f"Radar, {name}, {row['mount']}, pose {row.pose}, I channel", fontsize=12, pad=4)
        ax.set_xlim(0, y.size)
        ax.set_xticks([0, y.size])
        ax.tick_params(labelsize=10)
        for tick, ha in zip(ax.get_xticklabels(), ("left", "right")):
            tick.set_horizontalalignment(ha)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        signed_ticks(ax, "y")
        meta_out[str(frame)] = dict(target=str(row.target), mount=str(row["mount"]), pose=str(row.pose),
                                    label=str(row.y_bird_drone), channel=channel, length=int(y.size))
    save(fig, out, "radar_frames", {"frames": meta_out})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laplace", type=Path, default=OUT_ROOT / "robustness_laplace",
                    help="directory with the Laplace score_summary.csv / paired_tests.csv")
    ap.add_argument("--v3", type=Path, default=OUT_ROOT / "robustness_v3",
                    help="directory with the v3 best_vs_best_ba.csv / k_star.csv / paired_tests.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="prepared datasets, read only, for the example frames")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tab = main_contrast_table("comp")
    n = check_against_forest(tab)
    print(f"main contrasts: {n} IRMF rows match figures/contrasts_forest.csv")
    tab_log = main_contrast_table("comp+log_scale")
    check_against_forest(tab_log)
    pd.concat([tab, tab_log]).to_csv(args.out / "main_contrasts_values.csv", index=False)
    chart_main_contrasts(args.out, tab, ["dronerf", "birdclef2023", "esc50_animals", "uas_birds_60ghz"],
                         "main_contrasts", xlim=(-11, 22))
    chart_main_contrasts(args.out, tab, ["synthetic"], "main_contrasts_synthetic", height=3.0)
    chart_best_ba(args.out, parse_best_table())
    example_frames_chart(args.out, args.data)
    radar_frames_chart(args.out, args.data)
    chart_ba_vs_k(args.out)
    chart_equal_k(args.out)
    chart_kmax8(args.out)
    impulse_charts(args.out)
    if (args.laplace / "score_summary.csv").exists():
        laplace_charts(args.out, args.laplace)
    else:
        print(f"no Laplace results in {args.laplace}; Laplace charts skipped")
    if (args.v3 / "best_vs_best_ba.csv").exists():
        v3_charts(args.out, args.v3)
    else:
        print(f"no v3 results in {args.v3}; v3 charts skipped")
    (args.out / "charts_manifest.json").write_text(json.dumps(MANIFEST, indent=1, default=float))


if __name__ == "__main__":
    main()
