"""Paired classification comparison: converged IRMF from a new pimf release
against the study's Newton prototype, plus the main contrasts with each.

Reads the per-split tables that baselines/run_classical.py (and optionally
ml/run_trees.py) wrote for the two data roots prepared by
pimf_release_update.py:

    OUT/<dataset>/classical/            new release features, plus raw, EMD, linear IMF
    OUT/<dataset>/classical_prototype/  prototype features, same runner and splits

and writes OUT/paired_<runner>.csv. Holm correction runs within each contrast
family across all datasets, classifiers and feature sets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from emdirmf.stats import holm, paired_comparison  # noqa: E402

DATASETS = {"synthetic": "signal", "dronerf": "10class", "birdclef2023": "species", "esc50_animals": "animal"}
CONV = "irmf_h2_a1.41_conv"
RUNNERS = {
    "classical": ("classical", "classical_prototype", "results_folds_{task}.csv"),
    "trees": ("trees", "trees_prototype", "results_folds_{task}.csv"),
}


def per_split(path):
    table = pd.read_csv(path)
    if "subset" in table:
        table = table[table["subset"] == "all"]
    return table


def compare(out, runner):
    new_dir, proto_dir, pattern = RUNNERS[runner]
    rows = []
    for dataset, task in DATASETS.items():
        new_path = out / dataset / new_dir / pattern.format(task=task)
        proto_path = out / dataset / proto_dir / pattern.format(task=task)
        if not (new_path.exists() and proto_path.exists()):
            continue
        new, proto = per_split(new_path), per_split(proto_path)
        proto = proto[proto["method"] == CONV].assign(method="prototype")
        both = pd.concat([new, proto], ignore_index=True)
        for (classifier, feature_set), group in both.groupby(["classifier", "feature_set"]):
            scores = group.pivot_table(index="split_id", columns="method", values="balanced_accuracy")
            sizes = group.drop_duplicates("split_id").set_index("split_id").loc[scores.index]
            contrasts = [
                ("release - prototype", CONV, "prototype"),
                ("release IRMF - EMD", CONV, "emd"),
                ("prototype IRMF - EMD", "prototype", "emd"),
                ("release IRMF - linear IMF", CONV, "imf_linear_a1.41"),
                ("prototype IRMF - linear IMF", "prototype", "imf_linear_a1.41"),
            ]
            for name, a, b in contrasts:
                if a not in scores or b not in scores:
                    continue
                pair = scores[[a, b]].dropna()
                stats = paired_comparison(pair[a], pair[b], sizes.loc[pair.index, "n_train"],
                                          sizes.loc[pair.index, "n_test"])
                rows.append({"dataset": dataset, "classifier": classifier, "feature_set": feature_set,
                             "contrast": name, **stats})
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit(f"no {runner} results found under {out}")
    for column in ("mean_diff", "ci_low", "ci_high", "mean_a", "mean_b"):
        table[column] = 100 * table[column]
    family = table["contrast"].str.replace("release |prototype ", "", regex=True)
    table["family"] = np.where(table["contrast"] == "release - prototype", "release - prototype", family)
    table["source"] = np.where(table["contrast"].str.startswith("prototype"), "prototype", "release")
    table["p_nb_holm"] = np.nan
    for _, index in table.groupby(["family", "source"]).groups.items():
        table.loc[index, "p_nb_holm"] = holm(table.loc[index, "p_nb"].to_numpy())
    table.to_csv(out / f"paired_{runner}.csv", index=False)
    view = table[["dataset", "classifier", "feature_set", "contrast", "mean_diff", "ci_low", "ci_high", "p_nb_holm"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE.parents[2] / "output/classification-emd-irmf/pimf_release/0.5.0"))
    parser.add_argument("--runner", choices=sorted(RUNNERS), default="classical")
    args = parser.parse_args()
    compare(Path(args.out), args.runner)


if __name__ == "__main__":
    main()
