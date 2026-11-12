"""Apply the registered resource cuts using CPU timings, never new-input scores."""

import argparse
import hashlib
import json

import pandas as pd

from components.classification import CLASSICAL, FEATURE_SETS
from components.construction import BASE_TAGS, all_tags, parse_tag
from components.integrity import DATASETS, HERE, check_pimf, record_entry
from components.storage import run_dir, write_json

CUTS = ("LightGBM seed 0 only", "LightGBM comp only", "LightGBM K=4,6 plus defaults only")


def selection(level):
    if level not in (0, 1, 2, 3):
        raise ValueError(level)
    tags = all_tags()
    if level == 3:
        tags = [tag for tag in tags if tag in BASE_TAGS or tag == "gd60_trunc_k8" or parse_tag(tag).k in (4, 6)]
    return {"level": level, "cuts": list(CUTS[:level]),
            "lightgbm_seeds": [0] if level else [0, 1, 2],
            "lightgbm_feature_sets": ["comp"] if level >= 2 else list(FEATURE_SETS),
            "lightgbm_tags": tags}


def eligible(row, plan):
    return row["classifier"] != "lightgbm" or (
        row["seed"] in plan["lightgbm_seeds"] and
        row["feature_set"] in plan["lightgbm_feature_sets"] and
        row["method"] in plan["lightgbm_tags"])


def default_digest(dataset):
    digest = hashlib.sha256()
    for classifier in (CLASSICAL[dataset], "lightgbm"):
        for feature_set in FEATURE_SETS:
            for tag in BASE_TAGS:
                for split_id in range(15):
                    path = run_dir(dataset) / "fits" / classifier / feature_set / tag / f"{split_id:02d}.json"
                    digest.update(path.name.encode())
                    digest.update(path.read_bytes())
    return digest.hexdigest()


def project(elapsed_wall, fixed_remaining_cpu, tree_cpu_by_level, threshold_hours=12):
    """First applicable cut tier. Finished work is included and never discarded."""
    projections = []
    for level in range(4):
        total_wall = elapsed_wall + 1.2 * (fixed_remaining_cpu + tree_cpu_by_level[level]) / 6
        projections.append({"level": level, "projected_total_wall_hours": total_wall / 3600,
                            "future_classical_cpu_seconds": fixed_remaining_cpu,
                            "future_lightgbm_cpu_seconds": tree_cpu_by_level[level]})
        if total_wall <= threshold_hours * 3600:
            return {**selection(level), "projections": projections,
                    "projected_total_wall_hours": total_wall / 3600,
                    "actual_completed_runner_wall_seconds": elapsed_wall,
                    "threshold_hours": threshold_hours}
    # All allowed cuts are exhausted; fixed-classical coverage remains mandatory.
    return {**selection(3), "projections": projections,
            "projected_total_wall_hours": projections[-1]["projected_total_wall_hours"],
            "actual_completed_runner_wall_seconds": elapsed_wall, "threshold_hours": threshold_hours}


def calculate_plan():
    check_pimf()
    records, sources = [], {}
    elapsed_wall = 182.04723087500315
    for dataset in DATASETS:
        validation = json.loads((run_dir(dataset) / "default_validation.json").read_text())
        assert validation["passed"]
        completed = sorted(run_dir(dataset).glob("classification_completed_defaults_*.json"))
        assert completed
        elapsed_wall += json.loads(completed[0].read_text())["wall_seconds"]
        for path in run_dir(dataset).glob("processing_completed_*.json"):
            elapsed_wall += json.loads(path.read_text())["wall_seconds"]
        for classifier in (CLASSICAL[dataset], "lightgbm"):
            for feature_set in FEATURE_SETS:
                for tag in BASE_TAGS:
                    for split_id in range(15):
                        path = run_dir(dataset) / "fits" / classifier / feature_set / tag / f"{split_id:02d}.json"
                        row = json.loads(path.read_text())
                        # Only timing and design fields enter the projection.
                        records.append({"dataset": dataset, "classifier": classifier,
                                        "feature_set": feature_set, "method": tag,
                                        "cpu_seconds": row["cpu_seconds"]})
        sources[dataset] = default_digest(dataset)
    table = pd.DataFrame(records)
    means = table.groupby(["dataset", "classifier", "feature_set", "method"]).cpu_seconds.mean()
    fixed, trees = 0.0, [0.0] * 4
    new_tags = [tag for tag in all_tags() if tag not in BASE_TAGS]
    for dataset in DATASETS:
        for tag in new_tags:
            family = parse_tag(tag).family
            source = {"linear": "imf_linear_a1.41", "irmf": "irmf_h2_a1.41_conv",
                      "emd": "emd", "gd60": "irmf_h2_a1.41"}[family]
            for feature_set in FEATURE_SETS:
                fixed += 15 * means[dataset, CLASSICAL[dataset], feature_set, source]
            for level in range(4):
                plan = selection(level)
                if tag in plan["lightgbm_tags"]:
                    for feature_set in plan["lightgbm_feature_sets"]:
                        trees[level] += 5 * len(plan["lightgbm_seeds"]) * means[dataset, "lightgbm", feature_set, source]
    plan = project(elapsed_wall, fixed, trees)
    plan.update(default_checkpoint_sha256=sources,
                projection_rule="Actual completed runner wall plus 1.2 * remaining model CPU / six workers. Each new input conservatively uses its full-default family's mean CPU per feature set.",
                timing_inputs=table.groupby(["dataset", "classifier", "feature_set", "method"]).cpu_seconds.mean().reset_index().to_dict("records"),
                decision_uses_new_input_scores=False)
    return plan


def load_plan():
    path = run_dir("synthetic") / "budget_plan.json"
    return json.loads(path.read_text()) if path.exists() else selection(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-notes", action="store_true")
    args = parser.parse_args()
    for dataset in DATASETS:
        record_entry(dataset, "measured_default_budget")
    path = run_dir("synthetic") / "budget_plan.json"
    if path.exists():
        plan = load_plan()
        assert plan["default_checkpoint_sha256"] == {dataset: default_digest(dataset) for dataset in DATASETS}
    else:
        plan = calculate_plan()
        write_json(path, plan)
    if args.append_notes:
        protocol = HERE / "PROTOCOL_components.md"
        heading = "## Measured default timing decision"
        if heading not in protocol.read_text():
            lines = ["", heading, "", "Before any new-input classification, the completed default fits supplied",
                     "single-thread CPU measurements. The budget uses timing fields only, without",
                     "new-input scores. Completed runner wall time is retained, and remaining fits",
                     "use their full-default family's CPU mean per feature set, divided by six",
                     "with a 20% allowance. This is conservative for inputs with fewer features.", ""]
            for row in plan["projections"]:
                lines.append(f"- Tier {row['level']}: projected total wall {row['projected_total_wall_hours']:.3f} hours.")
            lines += ["", f"Chosen tier {plan['level']}: " + ("; ".join(plan["cuts"]) or "no cuts") + ".",
                      "All fixed-classical families retain all 15 splits and both feature sets.",
                      "Completed default LightGBM fits outside the selected seed/feature scope stay",
                      "on disk and in the score export, flagged as unused for paired analysis.",
                      "The complete timing-only decision and checkpoint hashes are in",
                      "DATA_ROOT/synthetic/components/budget_plan.json.", ""]
            with protocol.open("a") as handle:
                handle.write("\n".join(lines))
    print(json.dumps({key: value for key, value in plan.items() if key != "timing_inputs"}, indent=2))


if __name__ == "__main__":
    main()
