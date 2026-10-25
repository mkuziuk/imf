"""Build the self-contained robustness v3 page from the finished CSV results.

Run from the repository root with .venv/bin/python and this file's path.
Only the target HTML is written. Assertions guard the dated interpretation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import date
from pathlib import Path

STUDY = Path(__file__).resolve().parent
OUT = STUDY.parents[1] / "output" / "classification-emd-irmf"
TEMPLATE = STUDY / "report_templates" / "results_v3.template.html"
DATASETS = ("dronerf", "birdclef2023", "esc50_animals", "synthetic")
FIXED = dict(zip(DATASETS, ("linsvm", "logreg", "linsvm", "logreg")))
NAMES = dict(zip(DATASETS, ("DroneRF", "BirdCLEF 2023", "ESC-50 animals", "Synthetic")))
CLASSES = dict(zip(DATASETS, (10, 16, 10, 5)))
METHODS = ("irmf", "linear", "emd", "raw")
CONDITIONS = ("clean", "nb3_0.1", "nb3_0.2", "nb3_0.3", "nb3_0.4")
FAMILIES = {"Primary": 12, "S1": 12, "S3": 24, "S5": 6, "synthetic": 18}
WINDOWS = {
    2048: {3: [1025, 177, 31], 4: [1025, 317, 99, 31],
           6: [1025, 507, 251, 125, 63, 31], 8: [1025, 621, 375, 227, 139, 85, 51, 31],
           11: [1025, 721, 507, 357, 251, 177, 125, 89, 63, 45, 31]},
    1024: {3: [513, 125, 31], 4: [513, 201, 79, 31],
           6: [513, 293, 167, 95, 55, 31], 8: [513, 343, 229, 153, 103, 69, 47, 31],
           9: [513, 361, 253, 179, 125, 89, 63, 45, 31]},
}


def require(ok, message):
    """Keep interpretation checks active even with Python's -O flag."""
    if not ok:
        raise ValueError(f"Robustness v3 assertion failed: {message}")


def read_csv(source, name, columns):
    with (source / name).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(set(columns) <= set(reader.fieldnames or []), f"{name}: missing columns")
        rows = list(reader)
    require(bool(rows), f"{name}: empty results")
    return rows


def one(rows, **filters):
    found = [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]
    require(len(found) == 1, f"expected one row for {filters}, got {len(found)}")
    return found[0]


def close(actual, expected, label, tolerance=1e-10):
    require(math.isclose(float(actual), expected, abs_tol=tolerance, rel_tol=0),
            f"{label}: expected {expected}, got {actual}")


def signed(value):
    return f"{value:+.1f}".replace("-", "−")


def build_data(source, target):
    columns = {
        "paired_tests.csv": "dataset family classifier feature_set scenario comparator condition contrast irmf_k comparator_k n_splits holm_family mean_diff ci_low ci_high p_nb_holm holm_complete",
        "primary_tests.csv": "dataset family classifier scenario comparator condition mean_diff p_nb_holm",
        "best_vs_best_ba.csv": "dataset scenario classifier family variant k condition mean std",
        "ba_every_k_condition.csv": "dataset scenario classifier family variant k condition mean std size",
        "k_star.csv": "dataset classifier method variant k mean_clean_ba",
        "best_k_per_condition.csv": "dataset scenario classifier family k condition mean",
        "score_summary.csv": "dataset scenario classifier feature_set method condition balanced_accuracy_mean",
        "stability_metrics.csv": "dataset condition variant method k measure slot n median p95",
        "solver_diagnostics.csv": "dataset condition variant method k n_frames n_channels newton_cap_channels newton_cap_frames failed_frames h_median h_ratio_median h_ratio_p95 b_over_h_median b_over_h_p95",
        "prespecified_reading.csv": "dataset condition irmf_k linear_k did ci_low ci_high p_nb_holm verdict",
        "fit_costs.csv": "dataset scenario classifier method selection fit_seconds warning_count",
    }
    tables = {name: read_csv(source, name, fields.split()) for name, fields in columns.items()}
    tests, scores = tables["paired_tests.csv"], tables["best_vs_best_ba.csv"]
    all_k, stars = tables["ba_every_k_condition.csv"], tables["k_star.csv"]
    validation = json.loads((source / "validation.json").read_text())
    report = (source / "REPORT.md").read_text()
    protocol_path = STUDY / "robustness_v3" / "PROTOCOL_v3.md"
    protocol = protocol_path.read_text()
    require("## Amendment" in protocol and "fixed H per dataset" in protocol, "missing fixed-H amendment")
    require(validation["families"] == validation["expected_families"] == FAMILIES, "Holm family sizes")
    require(Counter(r["holm_family"] for r in tests) == Counter(FAMILIES), "paired-test family completeness")
    require(validation["all_prediction_metrics_recomputed"], "prediction validation incomplete")
    require({r["dataset"] for r in validation["datasets"]} == set(DATASETS), "validation dataset coverage")
    for r in validation["datasets"]:
        require(r["clean_exact_checks"] == r["clean_checks"] == 100 and r["pilot_exclusions"] == 100
                and r["grouped_splits_verified"] == 15 and r["inputs_verified"] == 5
                and r["fixed_h_recalibrated"] and r["matrices_recomputed"] and r["k_selection_recomputed"],
                f"{r['dataset']}: incomplete validation")
    require(len(stars) == 32 and len(scores) == 176 and len(all_k) == 656, "finished result dimensions")

    def star(d, method, clf=None):
        return one(stars, dataset=d, method=method, classifier=clf or FIXED[d])

    def ba(d, method, condition="clean", clf=None, scenario="shift"):
        return 100 * float(one(scores, dataset=d, family=method, classifier=clf or FIXED[d],
                               scenario=scenario, condition=condition)["mean"])

    def group(d, family, comp="linear"):
        return [r for r in tests if r["dataset"] == d and r["family"] == family and r["comparator"] == comp]

    # Cross-check every display source against the full score summary and K grid.
    for r in all_k:
        s = one(tables["score_summary.csv"], dataset=r["dataset"], classifier=r["classifier"],
                scenario=r["scenario"], method=r["variant"], feature_set="comp", condition=r["condition"])
        close(s["balanced_accuracy_mean"], float(r["mean"]), "score summary / all K")
        require(int(r["size"]) == 15, "all K split count")
    for r in stars:
        candidates = [s for s in all_k if s["dataset"] == r["dataset"] and s["classifier"] == r["classifier"]
                      and s["family"] == r["method"] and s["scenario"] == "shift" and s["condition"] == "clean"]
        expected_grid = {0} if r["method"] == "raw" else {2, 3, 4, 6, 8} if r["method"] == "emd" else set(WINDOWS[1024 if r["dataset"] == "synthetic" else 2048])
        require({int(s["k"]) for s in candidates} == expected_grid, "clean K grid")
        winner = min(candidates, key=lambda s: (-float(s["mean"]), int(s["k"])))
        require(winner["variant"] == r["variant"], "clean K* selection")
        close(r["mean_clean_ba"], float(winner["mean"]), "K* clean BA")
    for r in scores:
        require(r["k"] == star(r["dataset"], r["family"], r["classifier"])["k"], "K* must stay fixed")
        a = one(all_k, **{k: r[k] for k in ("dataset", "scenario", "classifier", "variant", "condition")})
        close(r["mean"], float(a["mean"]), "best-vs-best / all K")
    for r in tables["best_k_per_condition.csv"]:
        candidates = [s for s in all_k if all(s[k] == r[k] for k in ("dataset", "classifier", "scenario", "family", "condition"))]
        winner = min(candidates, key=lambda s: (-float(s["mean"]), int(s["k"])))
        require(winner["k"] == r["k"], "descriptive per-condition best K")
        close(r["mean"], float(winner["mean"]), "per-condition best BA")
    for r in tests:
        d, clf, comp, cond = r["dataset"], r["classifier"], r["comparator"], r["condition"]
        require(r["irmf_k"] == star(d, "irmf", clf)["k"] and r["comparator_k"] == star(d, comp, clf)["k"], "paired-test K values")
        require(r["n_splits"] == "15" and r["holm_complete"] == "True" and r["contrast"] == "DiD", "paired-test completeness")
        expected = (ba(d, "irmf", cond, clf, r["scenario"]) - ba(d, comp, cond, clf, r["scenario"]) - ba(d, "irmf", clf=clf) + ba(d, comp, clf=clf)) / 100
        close(r["mean_diff"], expected, "DiD reconstructed from BA")
    require(len(tables["primary_tests.csv"]) == 16, "primary test count")
    for r in tables["primary_tests.csv"]:
        require(r in tests and r["family"] == "Primary", "primary CSV / paired CSV")
    for r in tables["prespecified_reading.csv"]:
        t = one(tests, dataset=r["dataset"], family="Primary", condition="nb3_0.4")
        for key, other in (("did", "mean_diff"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("p_nb_holm", "p_nb_holm")):
            close(r[key], float(t[other]), "pre-specified reading / paired tests")
        require(r["irmf_k"] == t["irmf_k"] and r["linear_k"] == t["comparator_k"], "reading K*")

    # The brief's one-decimal accuracy values and component counts are frozen.
    expected_clean = {
        "dronerf": {"linear": (11, 50.5), "irmf": (8, 42.9), "emd": (2, 41.4), "raw": (0, 34.9)},
        "birdclef2023": {"emd": (2, 46.2), "irmf": (11, 44.1), "linear": (11, 41.7), "raw": (0, 37.6)},
        "esc50_animals": {"linear": (11, 60.9), "irmf": (11, 60.2), "emd": (4, 56.2), "raw": (0, 44.3)},
        "synthetic": {"irmf": (9, 81.1), "linear": (6, 68.9), "emd": (4, 54.8)},
    }
    for d, methods in expected_clean.items():
        for method, (k, score) in methods.items():
            require(int(star(d, method)["k"]) == k, f"{d} {method}: expected K{k}")
            close(round(ba(d, method), 1), score, f"{d} {method}: clean BA")
    expected_top = {
        ("dronerf", "linsvm", "shift"): {"irmf": 10.8, "linear": 12.2},
        ("esc50_animals", "lightgbm", "shift"): {"irmf": 38.0, "linear": 30.4, "emd": 18.4},
        ("dronerf", "lightgbm", "shift"): {"irmf": 17.2, "linear": 13.2},
        ("esc50_animals", "linsvm", "matched"): {"irmf": 60.2, "linear": 60.1, "emd": 56.6},
        ("birdclef2023", "logreg", "matched"): {"irmf": 40.2, "emd": 40.0, "linear": 38.7},
        ("dronerf", "linsvm", "matched"): {"irmf": 25.6, "linear": 25.2, "emd": 25.0, "raw": 26.2},
    }
    for (d, clf, scenario), methods in expected_top.items():
        for method, score in methods.items():
            close(round(ba(d, method, "nb3_0.4", clf, scenario), 1), score, f"{d} {clf} {scenario} {method} 40% BA")
    expected_ranges = {
        ("dronerf", "Primary", "linear"): (4.8, 6.1),
        ("birdclef2023", "Primary", "linear"): (-2.1, -1.6),
        ("esc50_animals", "Primary", "linear"): (-0.1, 0.3),
        ("birdclef2023", "S1", "emd"): (10.3, 12.2),
        ("esc50_animals", "S1", "emd"): (8.8, 10.4),
        ("dronerf", "S1", "emd"): (-6.5, -3.8),
        ("dronerf", "S3", "linear"): (9.0, 11.2),
        ("birdclef2023", "S3", "linear"): (-0.7, 0.8),
    }
    for key, limits in expected_ranges.items():
        vals = [100 * float(r["mean_diff"]) for r in group(*key)]
        require(tuple(round(f(vals), 1) for f in (min, max)) == limits, f"{key}: DiD range changed")
    drone_p = max(float(r["p_nb_holm"]) for r in group("dronerf", "Primary"))
    esc_lgb = [r for r in group("esc50_animals", "S3") if r["condition"] != "nb3_0.1"]
    esc_p = max(float(r["p_nb_holm"]) for r in esc_lgb)
    close(drone_p, 0.0021148399730506204, "DroneRF maximum primary Holm p")
    close(esc_p, 0.023429211674279398, "ESC-50 20–40% LightGBM maximum Holm p")
    require(tuple(round(f(100 * float(r["mean_diff"]) for r in esc_lgb), 1) for f in (min, max)) == (6.4, 7.7), "ESC-50 LightGBM DiD range")
    require(all(float(r["p_nb_holm"]) < .05 for d in DATASETS[:3] for r in group(d, "S1", "emd")), "classical EMD contrasts must reject")
    require(all(float(r["p_nb_holm"]) >= .05 for d in DATASETS[1:] for r in group(d, "Primary")), "other primary tests must not reject")
    require(all(float(r["p_nb_holm"]) >= .05 for r in group("birdclef2023", "S3")), "BirdCLEF LightGBM vs linear IMF must not reject")
    require(all(24.9 < ba("dronerf", m, "nb3_0.4", scenario="matched") < 26.3 for m in METHODS), "DroneRF matched BA range")
    require(ba("dronerf", "raw", "nb3_0.4", scenario="matched") == max(ba("dronerf", m, "nb3_0.4", scenario="matched") for m in METHODS), "DroneRF matched raw must be highest")

    h_values = {"dronerf": 2.09416640072853, "birdclef2023": 1.08331489400204,
                "esc50_animals": 0.47263983379285, "synthetic": 1.185305641621}
    for d, h in h_values.items():
        require(f"H_d = {h}" in report, f"{d}: report H value changed")
    for length, schedules in WINDOWS.items():
        for k, windows in schedules.items():
            require(f"K={k}: {','.join(map(str, windows))}" in report, f"L{length} K{k}: report windows changed")

    datasets = []
    for d in DATASETS:
        def numeric(rows, fields):
            return [{**r, **{k: float(r[k]) for k in fields}} for r in rows if r["dataset"] == d]
        datasets.append({
            "key": d, "name": NAMES[d], "fixed": FIXED[d], "classes": CLASSES[d], "chance": 100 / CLASSES[d],
            "h": h_values[d], "length": 1024 if d == "synthetic" else 2048,
            "stars": {clf: {m: {"k": int(star(d, m, clf)["k"]), "ba": ba(d, m, clf=clf)} for m in METHODS} for clf in (FIXED[d], "lightgbm")},
            "scores": numeric(scores, ["k", "mean", "std"]),
            "allK": numeric([r for r in all_k if r["condition"] == "clean" and r["scenario"] == "shift"], ["k", "mean", "std"]),
            "tests": [{**r, "d": 100 * float(r["mean_diff"]), "lo": 100 * float(r["ci_low"]), "hi": 100 * float(r["ci_high"]), "p": float(r["p_nb_holm"])} for r in tests if r["dataset"] == d],
            "stability": numeric([r for r in tables["stability_metrics.csv"] if r["measure"] == "feature_drift" and r["slot"] == "all"], ["k", "median", "p95"]),
            "diagnostics": numeric([r for r in tables["solver_diagnostics.csv"] if r["method"] == "irmf"], "k n_frames n_channels newton_cap_channels newton_cap_frames failed_frames h_median h_ratio_median h_ratio_p95 b_over_h_median b_over_h_p95".split()),
        })
    for d in datasets:
        for model, selected in d["stars"].items():
            for cond in CONDITIONS:
                diag = one(tables["solver_diagnostics.csv"], dataset=d["key"], condition=cond, method="irmf", k=selected["irmf"]["k"])
                close(diag["h_median"], d["h"], "fixed H diagnostic")
                require(int(diag["failed_frames"]) == int(diag["newton_cap_channels"]) == 0, "IRMF solver convergence changed")
                for m in METHODS:
                    one(tables["stability_metrics.csv"], dataset=d["key"], condition=cond, method=m,
                        k=selected[m]["k"], measure="feature_drift", slot="all")

    def relative(path):
        return Path(os.path.relpath(path, target.parent)).as_posix()

    sources = [("Protocol and amendment", relative(protocol_path)), ("Laplace results", "results_laplace.html")]
    for name in ("REPORT.md", *columns, "validation.json"):
        sources.append((name, relative(source / name)))
    sources += [("HTML builder", relative(Path(__file__).resolve())), ("HTML template", relative(TEMPLATE))]
    def k(d, m, clf=None):
        return star(d, m, clf)["k"]
    lead = [
        f"Stopping at a 31-sample window and selecting K on clean data does not give IRMF a general robustness advantage over linear IMF. With classical models, DroneRF has a positive DiD of +4.8 to +6.1 pp at IRMF K{k('dronerf', 'irmf')} vs linear IMF K{k('dronerf', 'linear')}. Holm p ≤ {math.ceil(drone_p * 1000) / 1000:.3f} at every rate. At 40%, accuracy is only {ba('dronerf', 'irmf', 'nb3_0.4'):.1f}% vs {ba('dronerf', 'linear', 'nb3_0.4'):.1f}%, close to 10% chance. This is a floor effect, not usable accuracy.",
        f"The clearest advantage above chance is ESC-50 with LightGBM. IRMF K11 loses 6.4 to 7.7 pp less than linear IMF K11 at 20–40% (Holm p ≤ {math.ceil(esc_p * 1000) / 1000:.3f}). At 40%, IRMF K11 scores {ba('esc50_animals', 'irmf', 'nb3_0.4', 'lightgbm'):.1f}%, linear IMF K11 {ba('esc50_animals', 'linear', 'nb3_0.4', 'lightgbm'):.1f}% and EMD K6 {ba('esc50_animals', 'emd', 'nb3_0.4', 'lightgbm'):.1f}%. Chance is 10%.",
        "Classical primary tests on BirdCLEF at IRMF K11 vs linear IMF K11, ESC-50 at K11 vs K11, and synthetic at IRMF K9 vs linear IMF K6 do not reject; not equivalence. Against EMD, classical IRMF loses 10.3 to 12.2 pp less on BirdCLEF at K11 vs K2 and 8.8 to 10.4 pp less on ESC-50 at K11 vs K4. Both survive Holm. EMD collapses under these spikes.",
    ]
    mismatches = [f"DroneRF primary at IRMF K8 vs linear IMF K11: brief Holm p ≤ 0.002; actual maximum {drone_p:.11f}.",
                  f"ESC-50 LightGBM at IRMF K11 vs linear IMF K11, 20–40%: brief Holm p ≤ 0.023; actual maximum {esc_p:.11f}.",
                  "DroneRF LightGBM at IRMF K11 vs linear IMF K11: DiD reaches 11.2 pp at one decimal, rounded to 11 pp in the brief."]
    fits = tables["fit_costs.csv"]
    require(all(r["selection"] == "grid_search" for r in fits), "all classical fits must rerun grid search")
    return {"datasets": datasets, "windows": WINDOWS, "meta": {
        "line": f"Protocol 2026-10-22 · built {date.today().isoformat()} · 15 paired grouped splits · comp features",
        "lead": lead, "sources": sources, "source": relative(source), "mismatches": mismatches,
        "hashes": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("REPORT.md", *columns, "validation.json")},
        "protocolSha256": hashlib.sha256(protocol.encode()).hexdigest(),
        "fitNote": f"All {len(fits):,} recorded classical fits reran the full grouped C grid. LightGBM used its shared inner early stopping and outer refit.",
        "validationNote": f"The audit recomputed {sum(r['predictions_recomputed'] for r in validation['datasets']):,} predictions. All 400 clean checks passed exactly. Every dataset has 100 pilot exclusions and 15 verified grouped splits. Matrices, fixed H and K selection were recomputed.",
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT / "robustness_v3")
    parser.add_argument("--target", type=Path, default=OUT / "results_v3.html")
    args = parser.parse_args()
    target = args.target.resolve()
    data = build_data(args.source.resolve(), target)
    # Build-time notes on rounding against the hand-written expectations; console only, not page content.
    notes = data["meta"].pop("mismatches")
    template = TEMPLATE.read_text(encoding="utf-8")
    require(template.count("__DATA__") == 1, "template needs exactly one data placeholder")
    require("—" not in template, "template contains an em dash")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {target}; all result assertions passed.")
    for note in notes:
        print(f"Precision note: {note}")


if __name__ == "__main__":
    main()
