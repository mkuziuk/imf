"""Build output/classification-emd-irmf/results_new.html from the follow-up results.

Reads the component-count tables (output/.../components/), the pimf release check
(output/.../pimf_release/<version>/) and, once it exists, the robustness report,
then fills report_templates/results_new.template.html. Rerun it whenever one of
those result sets changes:

    .venv/bin/python experiments/classification-emd-irmf/make_results_new_html.py
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUT = REPO / "output" / "classification-emd-irmf"
COMP = OUT / "components"
RELEASE_VERSION = "0.5.0"
RELEASE = OUT / "pimf_release" / RELEASE_VERSION
NEXT_VERSION = "0.5.1"
NEXT = OUT / "pimf_release" / NEXT_VERSION
ROBUST = OUT / "robustness"
TEMPLATE = STUDY / "report_templates" / "results_new.template.html"
TARGET = OUT / "results_new.html"

DATASETS = ["dronerf", "birdclef2023", "esc50_animals", "synthetic"]
FIXED = {"synthetic": "logreg", "dronerf": "linsvm", "birdclef2023": "logreg", "esc50_animals": "linsvm"}
CLF_LABEL = {"lda": "LDA", "logreg": "Logistic regression", "linsvm": "Linear SVM", "lightgbm": "LightGBM"}
NAMES = {"synthetic": "synthetic", "dronerf": "DroneRF", "birdclef2023": "BirdCLEF", "esc50_animals": "ESC-50"}


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def default_k(dataset):
    return 15 if dataset == "synthetic" else 17


def curves(scores):
    out = {}
    families = {
        "irmf": [(f"irmf_native_k{k}", k) for k in (3, 4, 5, 6, 9, 13)] + [("irmf_h2_a1.41_conv", None)],
        "linear": [(f"linear_native_k{k}", k) for k in (3, 4, 5, 6, 9, 13)] + [("imf_linear_a1.41", None)],
        "emd": [(f"emd_k{k}", k) for k in (3, 4, 5, 6, 7)] + [("emd", 8)],
    }
    for dataset in DATASETS:
        for clf_key, classifier in (("classical", FIXED[dataset]), ("lightgbm", "lightgbm")):
            for fs in ("comp", "comp+log_scale"):
                rows = scores[(scores.dataset == dataset) & (scores.classifier == classifier) & (scores.feature_set == fs)]
                by_method = rows.set_index("method")
                series = {}
                for family, members in families.items():
                    points = []
                    for method, k in members:
                        if method not in by_method.index:
                            continue
                        row = by_method.loc[method]
                        is_default = method in ("irmf_h2_a1.41_conv", "imf_linear_a1.41", "emd")
                        k_value = default_k(dataset) if k is None else k
                        points.append({"k": k_value, "ba": 100 * float(row.balanced_accuracy_mean),
                                       "nf": int(row.n_features), "isDefault": is_default})
                    series[family] = sorted(points, key=lambda p: p["k"])
                out[f"{dataset}|{clf_key}|{fs}"] = {"series": series}
    return out


def components_data():
    scores = pd.read_csv(COMP / "score_summary.csv")
    family_a = pd.read_csv(COMP / "family_a.csv")
    family_c = pd.read_csv(COMP / "family_c.csv")
    family_c = family_c[family_c.variant == "classical/comp"]
    padding = pd.read_csv(COMP / "padding_fractions.csv")
    reading = pd.read_csv(COMP / "survival_reading.csv").set_index("dataset")

    def rows(table):
        return [{"dataset": r.dataset, "k": int(r.k), "contrast": r.contrast, "d": 100 * r.mean_diff,
                 "lo": 100 * r.ci_low, "hi": 100 * r.ci_high, "p": float(r.p_nb_holm)} for r in table.itertuples()]

    default_gap = {}
    for dataset in DATASETS:
        rows_fixed = scores[(scores.dataset == dataset) & (scores.classifier == FIXED[dataset]) & (scores.feature_set == "comp")]
        ba = rows_fixed.set_index("method").balanced_accuracy_mean
        default_gap[dataset] = 100 * float(ba["irmf_h2_a1.41_conv"] - ba["emd"])

    reading_text = {}
    for dataset, row in reading.iterrows():
        verdict = "Keeps the advantage" if row.survives_matched_counts else "Does not keep it"
        reading_text[dataset] = f"{verdict}: {row.positive_and_holm_significant} of {row.required_contrasts} required contrasts significant"
    kept = [NAMES[d] for d in DATASETS if reading.loc[d, "survives_matched_counts"]]
    partial = [NAMES[d] for d in DATASETS if not reading.loc[d, "survives_matched_counts"]
               and reading.loc[d, "positive_and_holm_significant"] >= 3]
    lost = [NAMES[d] for d in DATASETS if reading.loc[d, "positive_and_holm_significant"] == 0]
    summary = (f"At equal counts, the IMF methods keep their advantage over EMD on {' and '.join(kept)}. "
               f"{' and '.join(partial)} falls one test short of the pre-registered bar, and on {' and '.join(lost)} "
               "the advantage mostly disappears. EMD itself does best with only 3 or 4 IMFs, and pimf's default "
               "k_max = 8 costs about 10 points.")
    return {
        "curves": curves(scores),
        "familyA": rows(family_a),
        "familyC": rows(family_c),
        "padding": {d: {int(r.k): float(r.padding_fraction) for r in padding[padding.dataset == d].itertuples()} for d in DATASETS},
        "reading": reading_text,
        "readingText": summary,
        "defaultGap": default_gap,
    }


def paired_rows():
    frames = []
    for runner in ("classical", "trees"):
        path = RELEASE / f"paired_{runner}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def release_data():
    order = {d: i for i, d in enumerate(DATASETS)}
    full = pd.read_csv(RELEASE / "full_data_comparison.csv").sort_values("dataset", key=lambda c: c.map(order))
    sample = pd.read_csv(RELEASE / "comparison.csv").sort_values("dataset", key=lambda c: c.map(order))
    paired = paired_rows()

    def cell(dataset, classifier, fs, contrast):
        if paired.empty:
            return None
        hit = paired[(paired.dataset == dataset) & (paired.classifier == classifier)
                     & (paired.feature_set == fs) & (paired.contrast == contrast)]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return {"d": float(r.mean_diff), "lo": float(r.ci_low), "hi": float(r.ci_high), "p": float(r.p_nb_holm)}

    contrasts = ["release - prototype", "release IRMF - EMD", "prototype IRMF - EMD",
                 "release IRMF - linear IMF", "prototype IRMF - linear IMF"]
    class_main = []
    for dataset in DATASETS:
        class_main.append({"dataset": dataset, "model": CLF_LABEL[FIXED[dataset]],
                           "cells": [cell(dataset, FIXED[dataset], "comp", c) for c in contrasts]})
    # Trees were rerun for converged IRMF only, so the method contrasts stay empty there.
    trees_done = (RELEASE / "paired_trees.csv").exists()
    class_main.append({"dataset": "dronerf", "model": "LightGBM",
                       "cells": [cell("dronerf", "lightgbm", "comp", c) for c in contrasts],
                       "missing": "not rerun" if trees_done else "running"})
    class_all = []
    if not paired.empty:
        for r in paired.itertuples():
            class_all.append({"dataset": r.dataset, "model": CLF_LABEL.get(r.classifier, r.classifier), "fs": r.feature_set,
                              "contrast": r.contrast.replace(" - ", " − "), "d": float(r.mean_diff),
                              "lo": float(r.ci_low), "hi": float(r.ci_high), "p": float(r.p_nb_holm)})

    channels = int(full.channels.sum())
    matching = channels - int(full["channels_rel_l2_above_1e-6"].sum())
    drone = full.set_index("dataset").loc["dronerf"]
    drone_match = 1 - drone["channels_rel_l2_above_1e-6"] / drone.channels
    release_rows = paired[paired.contrast == "release - prototype"] if not paired.empty else paired
    fixed_rows = release_rows[[FIXED.get(d) == c for d, c in zip(release_rows.dataset, release_rows.classifier)]] if not release_rows.empty else release_rows
    if fixed_rows.empty:
        class_tile = {"label": "Classification change", "value": "running", "sub": "fixed classical models"}
    else:
        largest = float(fixed_rows.mean_diff.abs().max())
        any_sig = bool((release_rows.p_nb_holm < 0.05).any())
        class_tile = {"label": "Classification change", "value": f"{largest:.2f}", "unit": " pp",
                      "sub": "largest release − prototype change, fixed classical model",
                      "meta": "One or more differences pass Holm" if any_sig else "No difference passes Holm correction"}
    speed = [{"dataset": r.dataset, "newton": float(r.cpu_s_newton_release), "proto": float(r.cpu_s_newton_prototype),
              "gd": float(r.cpu_s_gd60_release)} for r in sample.itertuples()]
    ratios = [s["newton"] / s["proto"] for s in speed]
    nxt = next_release_data()
    if nxt["speed"]:
        up = [r["speedup"] for r in nxt["speed"]]
        down = [r["slowdown"] for r in nxt["speed"]]
        speed_tile = {"label": f"Newton speed, {NEXT_VERSION}", "value": f"{min(up):.1f}–{max(up):.1f}×", "unit": " faster",
                      "sub": f"than {RELEASE_VERSION} on a 40-frame sample per dataset",
                      "meta": f"Still {min(down):.1f}–{max(down):.1f}× slower than our Numba prototype"}
    else:
        speed_tile = {"label": "Newton speed", "value": f"{min(ratios):.0f}–{max(ratios):.0f}×", "unit": " slower",
                      "sub": "pimf's NumPy Newton vs our Numba prototype",
                      "meta": "About 2× faster than gradient descent on audio, slower on DroneRF"}
    tiles = [
        {"label": "Gradient descent", "value": f"{int(sample.gd60_bitwise_identical.sum())} of {len(sample)}",
         "sub": "datasets bit-for-bit identical to 0.4.0", "meta": "Stages that hit the cap now raise a warning"},
        {"label": "Newton vs our prototype", "value": f"{100 * matching / channels:.1f}", "unit": "%",
         "sub": "of all channels match to within 1e-6",
         "meta": f"DroneRF {100 * drone_match:.1f}%. The rest differ where the score is flat and both answers are valid"},
        class_tile,
        speed_tile,
    ]
    full_rows = [{"dataset": r["dataset"], "channels": int(r["channels"]),
                  "gt6": int(r["channels_rel_l2_above_1e-6"]), "gt3": int(r["channels_rel_l2_above_1e-3"]),
                  "relmax": float(r["rel_l2_max"]), "frames_changed": int(r["frames_with_changed_features"]),
                  "converged": bool(r["all_converged"])} for _, r in full.iterrows()]
    note = ("Both feature sets went through the study's unchanged classical runner on the same 15 grouped splits. "
            "Cells show paired differences in balanced-accuracy points with 95% intervals and Holm p, "
            "features without log_scale. 'Release' is pimf 0.5.0's Newton, 'prototype' our Numba solver. "
            "The runner tunes C for logistic regression and the linear SVM on every dataset. The main study fixed "
            "DroneRF's linear SVM at C = 0.1 to save time, so DroneRF's absolute scores here can differ from the main "
            "study. Release and prototype use the same tuning, so their difference does not depend on it.")
    return {"tiles": tiles, "full": full_rows, "classMain": class_main, "classAll": class_all,
            "classNote": note, "speed": speed, "next": nxt}


def next_release_data():
    """The later release against RELEASE_VERSION: full-data identity and interleaved Newton timing."""
    order = {d: i for i, d in enumerate(DATASETS)}
    out = {"version": NEXT_VERSION, "base": RELEASE_VERSION, "full": [], "speed": [], "note": ""}
    versus = NEXT / f"full_data_vs_{RELEASE_VERSION}.csv"
    if versus.exists():
        table = pd.read_csv(versus).sort_values("dataset", key=lambda c: c.map(order))
        out["full"] = [{"dataset": r["dataset"], "channels": int(r["channels"]),
                        "features": bool(r["feature_table_bitwise_identical"]),
                        "diagnostics": bool(r["channel_diagnostics_identical"]),
                        "converged": bool(r["all_converged"])} for _, r in table.iterrows()]
    done = {r["dataset"] for r in out["full"]}
    pending = [NAMES[d] for d in DATASETS if d not in done]
    identical = [r for r in out["full"] if r["features"] and r["diagnostics"]]
    if out["full"] and len(identical) == len(out["full"]):
        names = ([f"all {len(DATASETS)} datasets"] if len(out["full"]) == len(DATASETS)
                 else [NAMES[r["dataset"]] for r in out["full"]])
        channels = sum(r["channels"] for r in out["full"])
        out["note"] = (f"pimf {NEXT_VERSION} only speeds up the Newton solver and should return {RELEASE_VERSION}'s results bit for bit. "
                       f"We reran it on every frame and compared its feature tables with {RELEASE_VERSION}'s byte by byte. They match on "
                       f"{', '.join(names)} ({channels:,} channels), as do the per-channel iteration counts, fallbacks and "
                       "differences from our prototype. Same features with the same runner, splits and seeds give the same scores, "
                       f"so the classification table above holds for {NEXT_VERSION} as well.")
        if pending:
            out["note"] += f" Still running: {', '.join(pending)}."
    elif out["full"]:
        changed = [NAMES[r["dataset"]] for r in out["full"] if not (r["features"] and r["diagnostics"])]
        out["note"] = (f"pimf {NEXT_VERSION} differs from {RELEASE_VERSION} on {', '.join(changed)}, although it promises "
                       "bit-identical results. The classification table above does not carry over to those datasets.")
    else:
        out["note"] = f"pimf {NEXT_VERSION} is being rerun on every frame."
    timing = NEXT / "timing.csv"
    sample = NEXT / "comparison.csv"
    gd = {}
    if sample.exists():
        gd = pd.read_csv(sample).set_index("dataset")["cpu_s_gd60_release"].to_dict()
    if timing.exists():
        table = pd.read_csv(timing).sort_values("dataset", key=lambda c: c.map(order))
        out["speed"] = [{"dataset": r["dataset"], "proto": float(r["cpu_s_prototype"]),
                         "old": float(r[f"cpu_s_pimf_{RELEASE_VERSION}"]), "new": float(r[f"cpu_s_pimf_{NEXT_VERSION}"]),
                         "speedup": float(r[f"speedup_vs_{RELEASE_VERSION}"]), "slowdown": float(r["slowdown_vs_prototype"]),
                         "identical": bool(r[f"outputs_identical_to_{RELEASE_VERSION}"]),
                         "gd": float(gd[r["dataset"]]) if r["dataset"] in gd else None} for _, r in table.iterrows()]
    return out


ROBUST_METHODS = ["irmf_h2_a1.41_conv", "imf_linear_a1.41", "emd", "raw"]
ROBUST_KEYS = {"irmf_h2_a1.41_conv": "irmf", "imf_linear_a1.41": "linear", "emd": "emd", "raw": "raw"}


def robustness_scores(dataset):
    """Every completed fit's score rows, read-only. Unlike the agent's loader this
    accepts a dataset that is still running; callers check per-cell completeness."""
    from robustness.storage import root

    rows = []
    for path in sorted((root(dataset) / "classification").rglob("complete.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    return pd.DataFrame(rows)


def robustness_interim():
    sys.path.insert(0, str(STUDY))
    from emdirmf.stats import corrected_resampled_ttest, holm
    from robustness.analysis import did_vector, paired_tests, specs
    from robustness.classification import CLASSICAL
    from robustness.contamination import CONDITIONS
    from robustness.reporting import aggregate_representations

    datasets = []
    for dataset in DATASETS:
        scores = robustness_scores(dataset)
        if scores.empty:
            continue
        clf = CLASSICAL[dataset]
        cell = scores[(scores.scenario == "shift") & (scores.classifier == clf) & (scores.feature_set == "comp")]
        counts = cell.groupby(["method", "condition"]).split_id.nunique()
        complete = all(counts.get((m, c), 0) == 15 for m in ROBUST_METHODS for c in CONDITIONS)
        if not complete:
            continue
        curves = {}
        for method in ROBUST_METHODS:
            means = cell[cell.method == method].groupby("condition").balanced_accuracy.mean()
            curves[ROBUST_KEYS[method]] = {c: 100 * float(means[c]) for c in CONDITIONS}
        tests = []
        if dataset == "synthetic":
            table, _ = paired_tests(scores, ("synthetic",))
            table = table[table.family.isin(["Primary", "S1", "S2"]) & (table.contrast == "DiD")]
            for r in table.itertuples():
                tests.append({"condition": r.condition, "comparator": r.comparator, "d": 100 * r.mean_diff,
                              "lo": 100 * r.ci_low, "hi": 100 * r.ci_high, "p": float(r.p_nb_holm),
                              "pRaw": float(r.p_nb), "family": r.family, "corrected": True})
        else:
            for spec in specs(dataset):
                if spec["family"] not in ("Primary", "S1", "S2") or spec["contrast"] != "DiD":
                    continue
                args = {k: spec[k] for k in ("dataset", "classifier", "feature_set", "scenario", "condition", "comparator")}
                d, n_train, n_test = did_vector(scores, **args)
                stats = corrected_resampled_ttest(d, n_train, n_test)
                tests.append({"condition": spec["condition"], "comparator": spec["comparator"], "d": 100 * stats["mean_diff"],
                              "lo": 100 * stats["ci_low"], "hi": 100 * stats["ci_high"], "p": float(stats["p_nb"]),
                              "pRaw": float(stats["p_nb"]), "family": spec["family"], "corrected": False})
        stability = []
        try:
            frame, _, diagnostics, _ = aggregate_representations(dataset)
            drift = frame[(frame.measure == "feature_drift") & (frame.feature_set == "comp")]
            for r in drift.itertuples():
                stability.append({"condition": r.condition, "method": ROBUST_KEYS[r.method],
                                  "median": float(r.median), "p95": float(r.p95)})
            caps = int(diagnostics.newton_cap_channels.sum())
            h_ratio = {r.condition: float(r.h_ratio_median) for r in diagnostics.itertuples()}
        except Exception:  # stability tables are optional while a dataset is still processing
            caps, h_ratio = None, {}
        datasets.append({"dataset": dataset, "classifier": CLF_LABEL[clf], "curves": curves, "tests": tests,
                         "stability": stability, "newtonCaps": caps, "hRatio": h_ratio,
                         "chance": 100.0 / CLASSES[dataset], "final": dataset == "synthetic",
                         "tag": "" if dataset == "synthetic" else "interim"})
    # Once every real dataset has its main cells, the pre-registered Primary (vs linear IMF)
    # and S1 (vs EMD) families are complete: 18 spike tests each across the three datasets.
    # Holm applies within each family. S2 (Gaussian) also holds impulse-minus-Gaussian
    # contrasts that this page does not compute, so its rows stay uncorrected.
    real = [d for d in datasets if d["dataset"] != "synthetic"]
    holm_done = len(real) == len(REAL_DATASETS)
    if holm_done:
        for family in ("Primary", "S1"):
            members = [t for d in real for t in d["tests"] if t["family"] == family]
            if len(members) != PRIMARY_TESTS:
                raise SystemExit(f"{family}: {len(members)} tests instead of {PRIMARY_TESTS}")
            for t, adjusted in zip(members, holm([t["pRaw"] for t in members])):
                t["p"], t["corrected"] = float(adjusted), True
        for d in real:
            d["tag"] = "main tests complete"
    # The final report Holm-corrects S2 within its full 27-test family; take every
    # p value from it, after checking that this page's own estimates agree.
    report_tests = ROBUST / "paired_tests.csv"
    if holm_done and report_tests.exists():
        final = pd.read_csv(report_tests)
        final = final[(final.contrast == "DiD") & (final.scenario == "shift") & (final.feature_set == "comp")]
        for d in real:
            rows = final[(final.dataset == d["dataset"]) & (final.classifier == FIXED[d["dataset"]])]
            for t in d["tests"]:
                r = rows[(rows.condition == t["condition"]) & (rows.comparator == t["comparator"])]
                if len(r) != 1 or abs(100 * r.iloc[0].mean_diff - t["d"]) > 1e-9 or abs(r.iloc[0].p_nb - t["pRaw"]) > 1e-12:
                    raise SystemExit(f"{d['dataset']} {t['condition']}: page and final report disagree")
                t["p"], t["corrected"] = float(r.iloc[0].p_nb_holm), True
    for d in datasets:
        d["reading"] = robustness_reading(d["dataset"], d["tests"], d["curves"], d["chance"])
    return datasets, holm_done


CLASSES = {"synthetic": 5, "dronerf": 10, "birdclef2023": 16, "esc50_animals": 10}
REAL_DATASETS = ("dronerf", "birdclef2023", "esc50_animals")
PRIMARY_TESTS = 18  # Holm family size for the real datasets' primary tests


def signed(value):
    """One decimal with an explicit sign and a typographic minus, as on the rest of the page."""
    return f"{value:+.1f}".replace("-", "\u2212")


def robustness_reading(dataset, tests, curves, chance):
    """The pre-registered reading at p = 0.04, plus a floor-effect note."""
    def find(condition, comparator="imf_linear_a1.41"):
        return next(t for t in tests if t["condition"] == condition and t["comparator"] == comparator)

    def significant(t):
        # Holm-adjusted p never exceeds the family size times the raw p, so before
        # correction a raw p below 0.05 / 18 is significant whatever the other datasets show.
        return t["p"] < 0.05 if t["corrected"] else t["p"] * PRIMARY_TESTS < 0.05

    gauss = find("gauss_0.04")
    i1, i2 = find("impulse1_0.04"), find("impulse2_0.04")
    met = any(t["d"] > 0 and significant(t) and gauss["d"] < t["d"] for t in (i1, i2))
    at4 = [curves[m][f"{k}_0.04"] for m in ("irmf", "linear", "emd") for k in ("impulse1", "impulse2")]
    worst, lowest = max(at4), min(at4)
    mild = max(curves[m][f"{k}_0.002"] for m in ("irmf", "linear", "emd") for k in ("impulse1", "impulse2"))
    floor = worst < chance + 10
    collapse = (f"every method falls to {lowest:.0f}–{worst:.0f}% balanced accuracy at 4% spikes, against "
                f"{chance:.0f}% chance, and none stays above {mild:.0f}% even at 0.2%")
    lines = []
    if met:
        lines.append("Meets the pre-registered rule.")
        if floor:
            lines.append(f"But {collapse}. So the gap change mostly measures how much each method had to lose.")
        else:
            lines.append(f"At 4% spikes IRMF keeps {curves['irmf']['impulse2_0.04']:.0f}% against linear IMF's "
                         f"{curves['linear']['impulse2_0.04']:.0f}%.")
    elif i1["d"] > 0 and i2["d"] > 0:
        which = "Holm correction" if i1["corrected"] else "a correction for the 18 main tests"
        lines.append(f"Does not meet the pre-registered rule. At 4% spikes the gap grows by {i1['d']:.1f} and "
                     f"{i2['d']:.1f} points, one- and two-sided, but neither survives {which}.")
        lighter = [t for t in tests if t["comparator"] == "imf_linear_a1.41" and t["condition"].startswith("impulse")
                   and not t["condition"].endswith("_0.04")]
        if lighter and all(t["d"] > 0 and significant(t) for t in lighter):
            lines.append(f"At 0.2% and 1% spikes it grows by {min(t['d'] for t in lighter):.0f}–"
                         f"{max(t['d'] for t in lighter):.0f} points, significant after correction.")
        if floor:
            lines.append(f"Still, {collapse}.")
    else:
        lines.append(f"Does not meet the pre-registered rule. Linear IMF holds up at least as well: at 4% spikes "
                     f"the gap changes by {signed(i1['d'])} and {signed(i2['d'])} points, one- and two-sided.")
        if floor:
            lines.append(f"And {collapse}.")
    if gauss["d"] < 0 and gauss["p"] < 0.05:
        lines.append("Under Gaussian noise of the same power, IRMF loses significantly more than linear IMF.")
    return {"met": met, "text": " ".join(lines), "impulse1": i1["d"], "impulse2": i2["d"], "gauss": gauss["d"],
            "floor": floor, "mild": mild}


def robustness_data():
    report = ROBUST / "REPORT.md"
    interim, holm_done = robustness_interim()
    done = {d["dataset"] for d in interim}
    pending = [NAMES[d] for d in DATASETS if d not in done]
    if interim:
        names = [NAMES[d["dataset"]] for d in interim]
        if report.exists():
            status = "final"
        elif not pending:
            status = "main tests complete, full report pending"
        else:
            status = f"interim: {', '.join(names)} ready"
        paragraphs = [
            "Spikes were added to the held-out test signals only. Every model was trained on clean data, "
            "then scored on the same test fold clean and contaminated. The question is whether IRMF's robust fitting "
            "loses less accuracy than linear IMF when that happens.",
            "The pre-registered measure is the change in the IRMF − linear IMF gap from clean to contaminated test data. "
            "A positive value means contamination hurt linear IMF more than IRMF. Robustness counts as helping on a dataset "
            "if that change is positive and significant at the heaviest spike rate (4% of samples), and smaller under "
            "Gaussian noise of the same power.",
        ]
        if pending:
            paragraphs.append(
                f"Still running: {', '.join(pending)}. Synthetic results are final and Holm-corrected within synthetic's "
                "own family, which was added after pre-registration. Real-dataset p-values here are uncorrected, because the "
                "pre-registered Holm correction spans all 18 main tests across the three real datasets.")
            did_note = "Synthetic p-values are Holm-corrected within synthetic. Real-dataset p-values are uncorrected until all three real datasets finish."
        elif holm_done and not report.exists():
            paragraphs.append(
                "The main cells are complete on all four datasets, so the spike tests below are final. For the real datasets they are "
                "Holm-corrected within the two pre-registered families of 18 tests, one against linear IMF and one against EMD. "
                "Synthetic, added after pre-registration, has its own family. The agent is still running the secondary LightGBM, "
                "log_scale and matched-training tests for its full report.")
            did_note = ("Spike rows are Holm-corrected: within synthetic for synthetic, and across the three real datasets' 18 tests "
                        "per comparison for the others. Gaussian rows on the real datasets are uncorrected, because their family "
                        "also holds contrasts this page does not compute.")
        else:
            did_note = ("p-values come from the final report, Holm-corrected within its families: 18 spike tests per "
                        "comparison across the real datasets, 27 Gaussian tests, and synthetic's own family.")
        return {"status": status, "paragraphs": paragraphs, "datasets": interim, "didNote": did_note}
    return {
        "status": "still running",
        "paragraphs": [
            "This pre-registered test adds spikes to the real signals and asks whether IRMF's robust fitting then beats linear IMF. "
            "Models are trained on clean data and scored on contaminated held-out folds.",
            "Synthetic is finished. DroneRF, BirdCLEF and ESC-50 run one after another, and the agent checks the run every hour. "
            "This section fills in when the report is written.",
        ],
        "list": [
            "Contamination: one-sided spikes, two-sided spikes, and Gaussian noise of the same average power, each at about 4, 20 and 80 spikes per 2,048-sample frame.",
            "Main test: how much the IRMF − linear IMF gap grows from clean to contaminated data, 18 tests with Holm correction.",
            "Pre-registered rule: robustness counts as helping on a dataset if the gap grows significantly at the heaviest spike rate, and grows less under the matching Gaussian noise.",
        ],
    }


def main():
    head = git_head()
    data = {
        "meta": {
            "line": f"{date.today().strftime('%d %B %Y').lstrip('0')} · branch t3/compare-emd-irmf-classification at {head} · "
                    "built by experiments/classification-emd-irmf/make_results_new_html.py",
            "lead": None,
            "footer": f"Built from output/classification-emd-irmf/components/, pimf_release/{RELEASE_VERSION}/, "
                      f"pimf_release/{NEXT_VERSION}/ and robustness/ "
                      f"at commit {head}. Links are relative to output/classification-emd-irmf/.",
            "sources": [
                ["Component-count report", "components/REPORT.md"],
                ["Component-count protocol", "../../experiments/classification-emd-irmf/components/PROTOCOL_components.md"],
                ["pimf 0.5.0 full-data comparison", f"pimf_release/{RELEASE_VERSION}/full_data_comparison.csv"],
                ["pimf 0.5.0 paired classification", f"pimf_release/{RELEASE_VERSION}/paired_classical.csv"],
                ["pimf 0.5.1 vs 0.5.0, every frame", f"pimf_release/{NEXT_VERSION}/full_data_vs_{RELEASE_VERSION}.csv"],
                ["pimf 0.5.1 Newton timing", f"pimf_release/{NEXT_VERSION}/timing.csv"],
                ["Robustness protocol", "../../experiments/classification-emd-irmf/robustness/PROTOCOL_robustness.md"],
                ["Laplace-noise robustness page", "results_laplace.html"],
                ["Laplace-noise robustness report", "robustness_laplace/REPORT.md"],
                ["Notebook-scale spikes page (robustness v3)", "results_v3.html"],
                ["Notebook-scale spikes report (robustness v3)", "robustness_v3/REPORT.md"],
                ["Main study results page", "results.html"],
            ],
        },
        "components": components_data(),
        "release": release_data(),
        "robustness": robustness_data(),
    }
    data["meta"]["lead"] = lead_text(data["release"])
    # Drop links to result files that a running job has not written yet.
    data["meta"]["sources"] = [[label, path] for label, path in data["meta"]["sources"]
                               if path.startswith("../") or (OUT / path).exists()]
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    if "</" in text:
        raise SystemExit("data contains '</', which would break the script tag")
    html = TEMPLATE.read_text()
    if html.count("__DATA__") != 1:
        raise SystemExit("template must contain exactly one __DATA__ placeholder")
    TARGET.write_text(html.replace("__DATA__", text))
    print(f"wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")


LEAD = ("Three follow-ups to the main study. Matching the number of components shrinks the IMF advantage over EMD. "
        "It holds on DroneRF and synthetic, mostly holds on ESC-50 and largely disappears on BirdCLEF. "
        "pimf's default k_max = 8 costs about 10 points of balanced accuracy, enough to erase IRMF's lead over EMD. "
        "With spikes added to test signals, IRMF holds up far better than linear IMF on synthetic. On the real datasets, "
        "every method trained on clean data falls close to chance once spikes appear. IRMF loses less than linear IMF on "
        "DroneRF and, at lighter spike rates, on BirdCLEF, and no less on ESC-50. pimf 0.5.0 matches our converged "
        "decompositions apart from tie-breaks on flat windows.")


def lead_text(release):
    nxt = release["next"]
    if not nxt["full"]:
        return LEAD
    same = all(r["features"] and r["diagnostics"] for r in nxt["full"])
    speed = ""
    if nxt["speed"]:
        up = sorted(r["speedup"] for r in nxt["speed"])
        speed = f", with a Newton solver {up[0]:.1f}–{up[-1]:.1f} times faster"
    if same and len(nxt["full"]) == len(DATASETS):
        return LEAD + f" pimf {nxt['version']} returns exactly the same features on every frame{speed}."
    if same:
        return LEAD + (f" pimf {nxt['version']} returns exactly the same features on the {len(nxt['full'])} datasets "
                       f"checked so far{speed}.")
    return LEAD + f" pimf {nxt['version']} does not reproduce {nxt['base']} exactly; see its section."


if __name__ == "__main__":
    main()
