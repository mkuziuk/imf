#!/usr/bin/env python3
"""Combine supplemental fits with cached, identically budgeted main references."""

import itertools
import argparse
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd

from emdirmf.stats import compare_methods
import analyse_phase3 as audit

OUT = STUDY.parents[1] / "output/classification-emd-irmf/esc50_animals/ml"
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def export(folds, destination, task):
    methods = list(dict.fromkeys(folds.method))
    keys = ["method", "feature_set", "classifier", "split_id"]
    assert not folds.duplicated(keys).any()
    assert all(set(group.split_id) == set(range(15)) for _, group in
               folds.groupby(keys[:-1]))
    folds.to_csv(destination / f"combined_folds_{task}.csv", index=False)
    long = folds.melt(id_vars=keys + ["n_train", "n_test"], value_vars=METRICS,
                      var_name="metric", value_name="value")
    tables = []
    for metric in METRICS:
        table = compare_methods(long, metric=metric, pairs=list(itertools.combinations(methods, 2)),
                                by=("feature_set", "classifier"))
        table["task"] = task
        table["holm_family_size"] = len(table)
        tables.append(table)
    all_pairs = pd.concat(tables, ignore_index=True)
    all_pairs.to_csv(destination / f"paired_{task}.csv", index=False)
    display = all_pairs[all_pairs.metric == "balanced_accuracy"].copy()
    display["difference [95% CI], pp"] = display.apply(lambda row:
        f"{100 * row.mean_diff:+.2f} [{100 * row.ci_low:+.2f}, {100 * row.ci_high:+.2f}]", axis=1)
    (destination / f"PAIRS_{task}.md").write_text("\n".join([
        f"# {destination.name}, {task}: paired balanced accuracy", "",
        f"Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: {len(display)} tests per metric. Wilcoxon is secondary.", "",
        display[["feature_set", "method_a", "method_b", "difference [95% CI], pp",
                 "p_nb_holm", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""]))
    scores = folds.groupby(keys[:-1])[METRICS].agg(["mean", "std"])
    flat = scores.reset_index()
    flat.columns = ["_".join(column).rstrip("_") for column in flat.columns]
    flat.to_csv(destination / f"scores_{task}.csv", index=False)
    return scores


def main():
    written = {}
    seen_methods = set()
    costs = []
    for task in ("animal", "bird"):
        main = pd.read_csv(OUT / "trees" / f"results_folds_{task}.csv")
        main = main[main.classifier == "lightgbm"]
        for name in ("sensitivity", "slots9", "matched", "concat"):
            folder = OUT / name
            path = folder / f"results_folds_{task}.csv"
            if not path.exists():
                continue
            additional = pd.read_csv(path)
            sources = [folder]
            late = OUT / "sensitivity_late"
            if name == "sensitivity" and (late / f"results_folds_{task}.csv").exists():
                additional = pd.concat([additional, pd.read_csv(late / f"results_folds_{task}.csv")],
                                       ignore_index=True)
                sources.append(late)
            validations = []
            for source in sources:
                run = json.loads((source / f"run_{task}.json").read_text())
                source_folds = pd.read_csv(source / f"results_folds_{task}.csv")
                prediction = pd.read_parquet(run["predictions"])
                validations.append(audit.audit_predictions(source_folds, prediction,
                    audit.load_meta("esc50_animals"), task, source.name,
                    n_splits=15, methods=source_folds.method.unique()))
                assert all(0 < json.loads(value)["n_estimators"] <= 150 for value in source_folds.best_params)
                costs.append({"stage": source.name, "task": task, "workers": run["workers"],
                              "wall_seconds": run["seconds"], "n_fits": len(source_folds)})
            if "n_input_features" in additional and name in {"slots9", "concat", "matched"}:
                base = {"slots9": 180, "concat": 540, "matched": 200}[name]
                for feature_set, group in additional.groupby("feature_set"):
                    values = set(group.n_input_features.dropna().astype(int))
                    expected = base + int(feature_set.endswith("log_scale"))
                    assert not values or values == {expected}
            # Reuse references only when the ablation leaves them unchanged.
            if name == "slots9":
                reference = main[main.method.isin(["raw", "emd"])]
            elif name == "matched":
                reference = main[main.method.isin(["raw", "emd", "imf_linear_a1.41",
                                                  "irmf_h2_a1.41", "irmf_h2_a1.41_conv"])]
                for extra_source in (OUT / "sensitivity", OUT / "sensitivity_late"):
                    extra_path = extra_source / f"results_folds_{task}.csv"
                    if extra_path.exists():
                        extra = pd.read_csv(extra_path)
                        reference = pd.concat([reference, extra[extra.method == "irmf_h2_a2_conv"]],
                                              ignore_index=True)
            elif name == "concat":
                reference = main
            else:
                reference = main
            combined = pd.concat([reference, additional], ignore_index=True)
            seen_methods.update(combined.method)
            scores = export(combined, folder, task)
            written[f"{name}_{task}"] = {"fits": len(additional),
                                          "combined_rows": len(combined),
                                          "methods": sorted(combined.method.unique()),
                                          "validation": validations}
            lines = [f"# {name}, {task}", "", "Percentages. References reuse identical main LightGBM fits.", "",
                     scores.mul(100).to_markdown(floatfmt=".2f"), "",
                     "Nadeau-Bengio corrected t-test and CI, Wilcoxon secondary. Holm across all pairs and both feature sets per metric.", ""]
            (folder / f"TABLE_{task}.md").write_text("\n".join(lines))
    (OUT / "sensitivity_validation.json").write_text(json.dumps(written, indent=2) + "\n")
    pd.DataFrame(costs).to_csv(OUT / "supplemental_costs.csv", index=False)
    decomposition = []
    for method in sorted(seen_methods):
        sources = ["emd", method[5:]] if method.startswith("emd__") else [method]
        cpu, slots = 0.0, 0
        for source in sources:
            if source == "raw":
                slots += 1
                continue
            info = json.loads((audit.DATA / "decomp" / f"{source}.json").read_text())
            cpu += info["seconds_per_frame"]["cpu_mean"]
            slots += info["shape"][2]
        decomposition.append({"method": method, "cpu_seconds_per_frame": cpu,
                              "source_slots": slots, "fusion_cost_is_sum": len(sources) > 1})
    pd.DataFrame(decomposition).to_csv(OUT / "all_decomposition_costs.csv", index=False)
    print(json.dumps(written, indent=2))


def export_clip_scores():
    meta = audit.load_meta("esc50_animals")
    rows = []
    for name in ("sensitivity", "sensitivity_late", "matched", "slots9", "concat"):
        for task in ("animal", "bird"):
            source = OUT / name / f"run_{task}.json"
            if not source.exists():
                continue
            run = json.loads(source.read_text())
            predictions = pd.read_parquet(run["predictions"])
            rows.extend(audit.clip_scores(predictions, meta, task, name))
    clips = pd.DataFrame(rows)
    clips.to_csv(OUT / "supplemental_clip_metrics_by_seed.csv", index=False)
    keys = ["task", "runner", "method", "feature_set", "classifier"]
    scores = clips.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    scores.columns = ["_".join(column).rstrip("_") for column in scores.columns]
    scores.to_csv(OUT / "supplemental_clip_scores.csv", index=False)
    print(f"Wrote {len(scores)} supplemental clip score cells.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.clips_only:
        main()
    export_clip_scores()
