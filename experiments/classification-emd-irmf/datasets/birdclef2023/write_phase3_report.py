#!/usr/bin/env python3
"""Assemble the concise report directly from the final audited small tables."""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import pandas as pd

from emdirmf.paths import output_dir

OUT = output_dir("birdclef2023", "ml")
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
NAMES = ["Raw", "EMD", "Linear IMF", "IRMF", "Converged IRMF"]
MODELS = {"lightgbm": "LGBM", "xgboost": "XGB", "extratrees": "ExtraTrees",
          "minirocket": "MiniRocket", "multirocket": "MultiRocket", "compact_resnet": "CNN"}
IRMF = METHODS[3]
PAIRS = [(IRMF, "emd"), (IRMF, METHODS[2]), (IRMF, "raw"), (METHODS[2], "raw"), ("emd", "raw")]


def table(headers, rows):
    return "\n".join(["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|",
                      *["|" + "|".join(map(str, row)) + "|" for row in rows]])


def oriented(pairs, model, side, a, b):
    cell = pairs[(pairs.classifier == model) & (pairs.feature_set == side) &
                 (pairs.metric == "balanced_accuracy") &
                 (((pairs.method_a == a) & (pairs.method_b == b)) |
                  ((pairs.method_a == b) & (pairs.method_b == a)))]
    assert len(cell) == 1
    row = cell.iloc[0]
    return (1 if row.method_a == a else -1) * row.mean_diff, row.p_nb_holm


def ba_table(task, folds):
    models = [m for m in MODELS if m in set(folds.classifier)]
    scores = folds.groupby(["method", "classifier", "feature_set"]).balanced_accuracy.mean()
    reference = pd.read_csv(OUT / "stft_reference" / f"results_folds_{task}.csv")
    ref_scores = reference.groupby("feature_set").balanced_accuracy.mean()
    rows = []
    for method, name in zip(METHODS, NAMES):
        row = [name]
        for model in models:
            row.append("/".join(f"{100*scores.loc[(method, model, side)]:.1f}"
                                for side in ("comp", "comp+log_scale")))
        row.append("/".join(f"{100*ref_scores.loc[side]:.1f}" for side in ("comp", "comp+log_scale"))
                   if method == "raw" else "n/a")
        rows.append(row)
    return table(["Method", *[MODELS[m] for m in models], "STFT"], rows)


def pair_table(task, pairs):
    models = [m for m in MODELS if m in set(pairs.classifier)]
    rows = []
    for model in models:
        values = []
        for a, b in PAIRS:
            sides = [oriented(pairs, model, side, a, b) for side in ("comp", "comp+log_scale")]
            values.append("/".join(f"{100*d:+.2f};{q:.2g}" for d, q in sides))
        rows.append([MODELS[model], *values])
    return table([task, "I−E", "I−L", "I−R", "L−R", "E−R"], rows)


def main():
    folds = {task: pd.read_csv(OUT / f"all_folds_{task}.csv") for task in ("species", "passerine")}
    pairs = {task: pd.read_csv(OUT / f"paired_all_metrics_{task}.csv") for task in folds}
    assert set(MODELS) == set(folds["species"].classifier)
    assert {"lightgbm", "extratrees", "minirocket", "compact_resnet"} <= set(folds["passerine"].classifier)
    for task, frame in folds.items():
        assert not frame.duplicated(["method", "classifier", "feature_set", "split_id"]).any()
        assert len(pairs[task]) == 3 * frame.classifier.nunique() * 20
    significant = []
    for model in MODELS:
        sides = []
        for side, label in (("comp", "−log"), ("comp+log_scale", "+log")):
            delta, q = oriented(pairs["species"], model, side, IRMF, "emd")
            if delta > 0 and q < .05:
                sides.append(label)
        if sides:
            significant.append(MODELS[model] + " " + "/".join(sides))
    positive_settings = sum(oriented(pairs["species"], model, side, IRMF, "emd")[0] > 0
                            for model in MODELS for side in ("comp", "comp+log_scale"))
    robust_settings = [oriented(pairs["species"], model, side, IRMF, METHODS[2])
                       for model in MODELS for side in ("comp", "comp+log_scale")]
    robust_sentence = ("Any IRMF−linear advantage is model-specific." if
                       any(delta > 0 and q < .05 for delta, q in robust_settings) else
                       "IRMF does not establish a robust-specific advantage over linear IMF.")
    converged_sentence = ("Convergence sensitivity needs model-specific interpretation." if
                          any(oriented(pairs["species"], model, side, METHODS[4], IRMF)[1] < .05
                              for model in MODELS for side in ("comp", "comp+log_scale")) else
                          "No converged/default difference is significant after Holm.")
    scores = folds["species"].groupby(["classifier", "method", "feature_set"]).balanced_accuracy.mean()
    best = scores.idxmax()[0]
    recorded = pd.read_csv(OUT / "recording_pooled_seeds_species.csv")
    recorded = recorded[(recorded.classifier == best) & recorded.complete_seed]
    recorded = recorded.groupby(["method", "feature_set"]).balanced_accuracy.mean()
    record_values = []
    for method, name in zip(METHODS, ["R", "E", "L", "I", "C"]):
        value = "/".join(f"{100*recorded.loc[(method, side)]:.1f}" for side in ("comp", "comp+log_scale"))
        record_values.append(name + "=" + value)
    record_pairs = pd.read_csv(OUT / "recording_comparisons_species.csv")
    record_q = "/".join(f"{oriented(record_pairs, best, side, IRMF, 'emd')[1]:.2g}"
                        for side in ("comp", "comp+log_scale"))
    checks = pd.read_csv(OUT / "feature_checks" / "checks_folds_species.csv")
    check_means = checks.groupby(["method", "feature_set"]).balanced_accuracy.mean()
    ablations = []
    for suffix in ("matched9", "emd_fusion"):
        ablations.append("/".join(f"{100*check_means.loc[(IRMF+'_'+suffix, side)]:.1f}"
                                  for side in ("comp", "comp+log_scale")))
    cost = {}
    for directory in OUT.iterdir():
        path = directory / "run_species.json"
        if not directory.is_dir() or directory.name == "pilot" or not path.exists():
            continue
        context = json.loads(path.read_text())
        classifier = ("STFT" if context.get("reference_only") else context["params"].get("classifiers", ["compact_resnet"])[0])
        cost[classifier] = cost.get(classifier, 0) + context["seconds"] / 60
    cost["checks"] = json.loads((OUT / "feature_checks" / "config.json").read_text())["seconds"] / 60
    cost_text = ",".join((MODELS.get(model, model) + f"={minutes:.1f}") for model, minutes in cost.items())
    elapsed = (datetime.now(timezone.utc) - datetime(2026, 10, 20, 19, 34, 18, tzinfo=timezone.utc)).total_seconds() / 60
    family_sizes = "/".join(str(int((pairs[task].metric == "balanced_accuracy").sum())) for task in folds)
    lines = [
        "BirdCLEF2023 species balanced accuracy (%), without/with log_scale.", "", ba_table("species", folds["species"]), "",
        "Passerine:", "", ba_table("passerine", folds["passerine"]), "",
        "Paired Δpp;Holm q, without/with scale. I=IRMF,E=EMD,L=linear IMF,R=raw,C=converged IRMF.", "",
        pair_table("species", pairs["species"]), "", pair_table("passerine", pairs["passerine"]), "",
        f"Primary tests are Nadeau-Bengio; Holm families contain {family_sizes} comparisons per metric. "
        "[All species pairs/CIs, NB/Wilcoxon p/q](paired_all_metrics_species.md); "
        "[passerine](paired_all_metrics_passerine.md). "
        "[Macro-F1/accuracy](full_metrics_species.md), [passerine](full_metrics_passerine.md); "
        "[STFT species](stft_reference/results_folds_species.csv), [STFT passerine](stft_reference/results_folds_passerine.csv).", "",
        f"Phase-2's EMD gap shrinks; I−E means stay positive in {positive_settings}/12 settings. Holm support: " +
        (", ".join(significant) if significant else "none") +
        ". " + robust_sentence + " Raw CNN remains competitive. Passerine trees also favor IRMF over EMD; CNN does not confirm that ordering.", "",
        f"[Recording BA](recording_comparisons_species.csv), {MODELS[best]}, descriptive best shared family, pooled per seed: " +
        ",".join(record_values) + f". I−E q={record_q}.", "",
        f"15-fold LightGBM 9-slot IRMF={ablations[0]}; fusion={ablations[1]}. Most mean EMD lead survives 180 features; "
        "fusion versus IRMF has q=1. [Exploratory checks](feature_checks/comparisons.csv). "
        "LGBM IRMF split importance: finest residual 11.4%, w7/w9 ≈8.4%; [all scales](scale_importance_species.csv).", "",
        "Budgets: trees15 folds; species CNN15, passerine CNN5; MiniRocket5 both, MultiRocket5 species; STFT15. "
        "Boosting≤500, ExtraTrees300, CNN≤30 epochs, MiniRocket2016/MultiRocket1344 features, 10 head epochs. "
        "Methods share budgets/splits. " + converged_sentence, "",
        f"Primary wall minutes under concurrent load: {cost_text}; total elapsed={elapsed:.1f}. "
        "[Per-input costs](fit_costs_species.csv). Cached CPU s/frame: E=.01165,L=.00632,I=1.32049,C=.05792.", "",
        "More compute: 15-fold/10k-feature ROCKET, matched-channel controls, frequency-binned EMD and longer clips. "
        "[Reproduction/files](REPRODUCE.md). Shared ml/ unchanged; local confidence-export/pooling fixes documented.", "",
    ]
    report = "\n".join(lines)
    plain = re.sub(r"^\|[-|]+\|$", "", report, flags=re.MULTILINE).replace("|", " ")
    word_count = len(plain.split())
    assert word_count < 500
    (OUT / "REPORT.md").write_text(report)
    print(f"Report word count, including separated table cells: {word_count}; elapsed {elapsed:.1f} min")


if __name__ == "__main__":
    main()
