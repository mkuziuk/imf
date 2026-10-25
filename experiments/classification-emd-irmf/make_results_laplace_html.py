"""Build the standalone Laplace robustness page from the analysis CSVs.

Run from the repository root:
    .venv/bin/python experiments/classification-emd-irmf/make_results_laplace_html.py

Only the target HTML is written. No experiment code, caches or results are changed.
"""

import argparse
import csv
import json
import math
import os
import subprocess
from datetime import date
from pathlib import Path

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
OUT = REPO / "output" / "classification-emd-irmf"
TEMPLATE = STUDY / "report_templates" / "results_laplace.template.html"
DATASETS = ["dronerf", "birdclef2023", "esc50_animals", "synthetic"]
FIXED = {"dronerf": "linsvm", "birdclef2023": "logreg",
         "esc50_animals": "linsvm", "synthetic": "logreg"}
NAMES = {"dronerf": "DroneRF", "birdclef2023": "BirdCLEF 2023",
         "esc50_animals": "ESC-50 animals", "synthetic": "Synthetic"}
CLASSES = {"dronerf": 10, "birdclef2023": 16, "esc50_animals": 10, "synthetic": 5}
METHODS = ["irmf_h2_a1.41_conv", "imf_linear_a1.41", "emd", "raw"]
CONDITIONS = ["clean", "laplace_0.1", "laplace_0.2", "laplace_0.3", "laplace_0.4"]


def read_csv(source, name, columns):
    with (source / name).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        return list(reader)


def number(value, scale=1):
    if value is None or value.strip().lower() in ("", "nan", "none", "null"):
        return None
    result = float(value) * scale
    return result if math.isfinite(result) else None


def git_value(*args):
    result = subprocess.run(["git", "-C", str(REPO), *args],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def signed(value):
    return f"{value:+.1f}".replace("-", "\u2212")


def lead_text(source):
    """Summary paragraph for the full results; None for a preview.

    The wording encodes the pattern of the 2026-10-22 results. The assertions stop
    a rebuild on different results from printing a summary that no longer fits.
    """
    tests = read_csv(source, "paired_tests.csv", ["dataset", "family", "comparator", "feature_set",
                                                  "condition", "mean_diff", "p_nb_holm"])
    scores = read_csv(source, "score_summary.csv", ["dataset", "scenario", "classifier", "feature_set",
                                                    "method", "condition", "balanced_accuracy_mean"])
    real = ["dronerf", "birdclef2023", "esc50_animals"]
    if not set(real) <= {row["dataset"] for row in tests}:
        return None

    def ba(dataset, method, condition, scenario="shift"):
        return next(100 * float(r["balanced_accuracy_mean"]) for r in scores
                    if r["dataset"] == dataset and r["scenario"] == scenario and r["classifier"] == FIXED[dataset]
                    and r["feature_set"] == "comp" and r["method"] == method and r["condition"] == condition)

    def primary(dataset):
        return [(100 * float(r["mean_diff"]), float(r["p_nb_holm"])) for r in tests
                if r["dataset"] == dataset and r["family"] == "Primary"]

    drone = primary("dronerf")
    others = primary("birdclef2023") + primary("esc50_animals")
    assert all(d > 0 and p < 0.05 for d, p in drone), "DroneRF wording assumes significant positive DiDs"
    assert all(p >= 0.05 for _, p in others), "BirdCLEF/ESC-50 wording assumes no significant DiD"
    irmf, linear = "irmf_h2_a1.41_conv", "imf_linear_a1.41"
    top = "laplace_0.4"
    matched_gap = max(abs(ba(d, irmf, top, "matched") - ba(d, linear, top, "matched"))
                      for d in ("birdclef2023", "esc50_animals"))
    recovered = {d: (ba(d, linear, top, "matched") - ba(d, linear, top)) / (ba(d, linear, "clean") - ba(d, linear, top))
                 for d in real}
    assert recovered["dronerf"] < 0.7 and min(recovered["birdclef2023"], recovered["esc50_animals"]) > 0.8
    return (
        "Laplace noise on 10 to 40% of test samples cuts balanced accuracy sharply for every method trained on "
        f"clean data. At 10%, DroneRF falls from {ba('dronerf', linear, 'clean'):.0f}% to "
        f"{ba('dronerf', linear, 'laplace_0.1'):.0f}% with linear IMF and from {ba('dronerf', irmf, 'clean'):.0f}% "
        f"to {ba('dronerf', irmf, 'laplace_0.1'):.0f}% with IRMF. On DroneRF, IRMF loses "
        f"{min(d for d, _ in drone):.1f} to {max(d for d, _ in drone):.1f} points less than linear IMF at every "
        f"rate (Holm p ≤ {max(p for _, p in drone):.3f}). Both still end near chance at 40% "
        f"({ba('dronerf', irmf, top):.1f}% and {ba('dronerf', linear, top):.1f}%, chance 10%), so the gain mostly "
        "measures how much more linear IMF's higher clean score had to lose. On BirdCLEF and ESC-50 the "
        f"IRMF − linear IMF gap moves by {signed(min(d for d, _ in others))} to {signed(max(d for d, _ in others))} "
        "points and no change survives Holm correction. Models trained on noisy data win back "
        f"{100 * recovered['dronerf']:.0f}% of linear IMF's loss on DroneRF and "
        f"{100 * min(recovered['birdclef2023'], recovered['esc50_animals']):.0f} to "
        f"{100 * max(recovered['birdclef2023'], recovered['esc50_animals']):.0f}% on BirdCLEF and ESC-50, "
        f"where IRMF and linear IMF then end within {matched_gap:.1f} points of each other."
    )


def build_data(source, target):
    tests = read_csv(source, "paired_tests.csv", [
        "dataset", "family", "classifier", "feature_set", "scenario", "comparator",
        "condition", "mean_diff", "ci_low", "ci_high", "p_nb_holm", "holm_family", "status"])
    scores = read_csv(source, "score_summary.csv", [
        "dataset", "scenario", "classifier", "feature_set", "method", "condition",
        "balanced_accuracy_mean"])
    stability = read_csv(source, "stability_metrics.csv", [
        "dataset", "condition", "method", "measure", "feature_set", "median", "p95"])
    diagnostics = read_csv(source, "solver_diagnostics.csv", [
        "dataset", "condition", "n_frames", "n_channels", "h_ratio_median", "h_ratio_p95",
        "newton_cap_channels", "newton_cap_frames", "newton_failed_frames"])
    fits = read_csv(source, "fit_costs.csv", [
        "dataset", "scenario", "classifier", "selection", "clean_matches_impulse_run"])
    present = {row["dataset"] for row in scores + tests}
    unknown = present - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if not present:
        raise ValueError("The results contain no datasets")

    datasets = []
    for dataset in DATASETS:
        if dataset not in present:
            continue
        test_rows = []
        for row in tests:
            if row["dataset"] != dataset or row.get("contrast", "DiD") != "DiD":
                continue
            test_rows.append({
                **{key: row[key] for key in ("family", "classifier", "feature_set", "scenario",
                                            "comparator", "condition", "holm_family", "status")},
                "d": number(row["mean_diff"], 100), "lo": number(row["ci_low"], 100),
                "hi": number(row["ci_high"], 100), "p": number(row["p_nb_holm"]),
                "n": number(row.get("n_splits", "")),
            })
        curves = {}
        for model in (FIXED[dataset], "lightgbm"):
            for feature_set in ("comp", "comp+log_scale"):
                rows = [row for row in scores if row["dataset"] == dataset
                        and row["scenario"] == "shift" and row["classifier"] == model
                        and row["feature_set"] == feature_set]
                curves[f"{model}|{feature_set}"] = {
                    method: {condition: next((number(row["balanced_accuracy_mean"], 100)
                                              for row in rows if row["method"] == method
                                              and row["condition"] == condition), None)
                             for condition in CONDITIONS} for method in METHODS
                }
        datasets.append({
            "key": dataset, "name": NAMES[dataset], "classes": CLASSES[dataset],
            "fixed": FIXED[dataset], "chance": 100 / CLASSES[dataset],
            "curves": curves, "tests": test_rows,
            "stability": [{"condition": row["condition"], "method": row["method"],
                           "median": number(row["median"]), "p95": number(row["p95"])}
                          for row in stability if row["dataset"] == dataset
                          and row["measure"] == "feature_drift" and row["feature_set"] == "comp"
                          and row.get("slot", "all") == "all"],
            "diagnostics": [{"condition": row["condition"],
                             **{key: number(row[key]) for key in (
                                 "n_frames", "n_channels", "h_ratio_median", "h_ratio_p95",
                                 "newton_cap_channels", "newton_cap_frames", "newton_failed_frames")}}
                            for row in diagnostics if row["dataset"] == dataset],
        })

    classical = [row for row in fits if row["dataset"] in present and row["scenario"] == "shift"
                 and row["classifier"] == FIXED[row["dataset"]]]
    reused = [row for row in classical if row["selection"] == "impulse_run_choice"]
    verified = sum(row["clean_matches_impulse_run"].lower() == "true" for row in reused)
    fallback = sum(row["selection"] == "grid_search_fallback" for row in classical)
    grid = sum(row["selection"] == "grid_search" for row in classical)

    def relative(path):
        return Path(os.path.relpath(path, target.parent)).as_posix()

    sources = [("Laplace protocol", relative(STUDY / "robustness_laplace" / "PROTOCOL_laplace.md")),
               ("Impulse experiment page", relative(OUT / "results_new.html") + "#robustness"),
               ("Impulse experiment report", relative(OUT / "robustness" / "REPORT.md")),
               ("Notebook-scale spikes page (robustness v3)", relative(OUT / "results_v3.html"))]
    for name, label in (
        ("REPORT.md", "Laplace experiment report"), ("per_split_scores.csv", "Per-split scores"),
        ("paired_tests.csv", "All paired tests"), ("primary_tests.csv", "Primary tests"),
        ("score_summary.csv", "Score summary"), ("stability_metrics.csv", "Feature stability"),
        ("solver_diagnostics.csv", "Solver diagnostics"), ("prespecified_reading.csv", "Pre-specified reading"),
        ("fit_costs.csv", "Fit costs and C reuse checks"), ("costs.csv", "Feature costs"),
        ("validation.json", "Validation audit"), ("environment_audit.json", "Environment and source hashes"),
        ("failed_frames.csv", "Failed frames"), ("paired_vectors.csv", "Paired DiD vectors")):
        if (source / name).exists():
            sources.append((label, relative(source / name)))
    sources.extend([("HTML builder", relative(Path(__file__).resolve())),
                    ("HTML template", relative(TEMPLATE))])
    return {
        "datasets": datasets,
        "meta": {
            "line": f"Built {date.today().isoformat()} · {git_value('branch', '--show-current')} · "
                    f"{git_value('rev-parse', '--short', 'HEAD')} · "
                    "experiments/classification-emd-irmf/make_results_laplace_html.py",
            "source": relative(source), "sources": sources,
            "protocol": sources[0][1], "impulsePage": sources[1][1], "impulseReport": sources[2][1],
            "missing": [NAMES[key] for key in DATASETS if key not in present],
            "lead": lead_text(source),
        },
        "fits": {"classical": len(classical), "reused": len(reused), "verified": verified,
                 "fallback": fallback, "grid": grid},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT / "robustness_laplace")
    parser.add_argument("--target", type=Path, default=OUT / "results_laplace.html")
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    data = build_data(source, target)
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("Template must contain exactly one __DATA__ placeholder")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {target} ({len(data['datasets'])} datasets)")


if __name__ == "__main__":
    main()
