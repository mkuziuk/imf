"""Require complete paired coverage, export statistics, figures and the report."""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from components.classification import CLASSICAL, FEATURE_SETS, population
from components.budget import eligible, load_plan, selection
from components.construction import BASE_TAGS, DEFAULTS, EMD_K, NATIVE_K, TRUNC_K, all_tags, native_ratio, parse_tag
from components.integrity import DATA_ROOT, DATASETS, HERE, check_pimf, record_entry
from components.processing import BLOCK_SIZE
from components.presentation import write_overview
from components.storage import OUTPUT, STUDY, allocated_bytes, file_hash, run_dir, write_json
from emdirmf.decompose import MethodSpec, imf_schedule, scale_order
from emdirmf.stats import holm, paired_comparison

METRICS = ("balanced_accuracy", "macro_f1", "accuracy")
CONTRASTS = (("irmf", "emd"), ("linear", "emd"), ("irmf", "linear"))
COLORS = {"linear": "#2673a8", "irmf": "#c45b28", "emd": "#278b63"}
DISPLAY = {"linear": "Linear IMF", "irmf": "IRMF", "emd": "EMD"}


def verify_code_manifests():
    records = []
    for dataset in DATASETS:
        for name in ("processing_manifest.json", "classification_manifest_defaults.json",
                     "classification_manifest_experiment.json", "budgeted_classification_manifest.json"):
            manifest = json.loads((run_dir(dataset) / name).read_text())
            for relative_path, expected in manifest["code"].items():
                current = file_hash(STUDY / relative_path)
                if current != expected:
                    raise RuntimeError(f"STOP: code changed since {dataset}/{name}: {relative_path}")
                records.append({"dataset": dataset, "manifest": name, "path": relative_path,
                                "sha256": current})
    return records


def registered_pairs(family):
    pairs = []
    if family == "A":
        for k in (3, 4, 5, 6):
            tags = {"irmf": f"irmf_native_k{k}", "linear": f"linear_native_k{k}", "emd": f"emd_k{k}"}
            for a, b in CONTRASTS:
                pairs.append((k, f"{DISPLAY[a]} - {DISPLAY[b]}", tags[a], tags[b]))
    elif family == "B":
        for f in ("linear", "irmf"):
            pairs.extend((k, f"{DISPLAY[f]} K={k} - full", f"{f}_native_k{k}", DEFAULTS[f]) for k in NATIVE_K)
        pairs.extend((k, f"EMD K={k} - K=8", f"emd_k{k}", "emd") for k in EMD_K if k != 8)
    elif family == "C":
        for f in ("linear", "irmf"):
            pairs.extend((k, f"{DISPLAY[f]} k_max={k} - full", f"{f}_trunc_k{k}", DEFAULTS[f]) for k in TRUNC_K)
        pairs.extend([(8, "GD60 k_max=8 - GD60 full", "gd60_trunc_k8", "irmf_h2_a1.41"),
                      (8, "GD60 k_max=8 - EMD K=8", "gd60_trunc_k8", "emd")])
    else:
        raise ValueError(family)
    return pairs


def load_scores(plan=None):
    plan = selection(0) if plan is None else plan
    rows = []
    expected = 0
    for dataset in DATASETS:
        validation = json.loads((run_dir(dataset) / "default_validation.json").read_text())
        if not validation["passed"]:
            raise RuntimeError(f"unverified defaults: {dataset}")
        for classifier in (CLASSICAL[dataset], "lightgbm"):
            for feature_set in FEATURE_SETS:
                for tag in all_tags():
                    for split_id in range(15):
                        seed = population(dataset)[3][split_id].seed
                        use = eligible({"classifier": classifier, "method": tag,
                                        "feature_set": feature_set, "seed": seed}, plan)
                        if tag not in BASE_TAGS and not use:
                            continue
                        expected += 1
                        path = run_dir(dataset) / "fits" / classifier / feature_set / tag / f"{split_id:02d}.json"
                        row = json.loads(path.read_text())
                        assert (row["dataset"], row["method"], row["classifier"], row["feature_set"], row["split_id"]) == (
                            dataset, tag, classifier, feature_set, split_id)
                        rows.append({**row, "analysis_eligible": use})
    table = pd.DataFrame(rows)
    assert len(table) == expected and np.isfinite(table[list(METRICS)]).all().all()
    assert table[list(METRICS)].ge(0).all().all() and table[list(METRICS)].le(1).all().all()
    assert len(table[table.classifier != "lightgbm"]) == 3480
    actual = sum(len(list((run_dir(dataset) / "fits").glob("*/*/*/*.json"))) for dataset in DATASETS)
    assert actual == expected, ("unexpected or incomplete fit coverage", actual, expected)
    return table


def paired_tests(scores, plan=None):
    plan = selection(0) if plan is None else plan
    rows = []
    for family in ("A", "B", "C"):
        for tree, feature_set in ((False, "comp"), (False, "comp+log_scale"),
                                  (True, "comp"), (True, "comp+log_scale")):
            if family == "B" and (tree or feature_set != "comp"):
                continue
            if tree and feature_set not in plan["lightgbm_feature_sets"]:
                continue
            variant = ("lightgbm" if tree else "classical") + "/" + feature_set
            for dataset in DATASETS:
                classifier = "lightgbm" if tree else CLASSICAL[dataset]
                cell = scores[(scores.dataset == dataset) & (scores.classifier == classifier) &
                              (scores.feature_set == feature_set)]
                for k, contrast, a, b in registered_pairs(family):
                    if tree and (a not in plan["lightgbm_tags"] or b not in plan["lightgbm_tags"]):
                        continue
                    first = cell[cell.method == a].sort_values("split_id").reset_index(drop=True)
                    second = cell[cell.method == b].sort_values("split_id").reset_index(drop=True)
                    n_splits = 5 * len(plan["lightgbm_seeds"]) if tree else 15
                    assert len(first) == len(second) == n_splits
                    assert np.array_equal(first.split_id, second.split_id)
                    assert np.array_equal(first.n_train, second.n_train) and np.array_equal(first.n_test, second.n_test)
                    for metric in METRICS:
                        row = {"family": family, "variant": variant, "dataset": dataset,
                               "classifier": classifier, "feature_set": feature_set,
                               "metric": metric, "k": k, "contrast": contrast,
                               "method_a": a, "method_b": b}
                        row.update(paired_comparison(first[metric], second[metric], first.n_train, first.n_test))
                        rows.append(row)
    tests = pd.DataFrame(rows)
    tests["holm_family"] = tests.apply(lambda r: "/".join(
        [r.family, r.variant, r.metric] + ([] if r.family == "A" else [r.dataset])), axis=1)
    for _, indices in tests.groupby("holm_family").groups.items():
        row = tests.loc[indices[0]]
        tree = row.variant.startswith("lightgbm/")
        expected = sum(not tree or (a in plan["lightgbm_tags"] and b in plan["lightgbm_tags"])
                       for _, _, a, b in registered_pairs(row.family))
        expected *= 4 if row.family == "A" else 1
        assert len(indices) == expected
        for kind in ("nb", "wilcoxon"):
            tests.loc[indices, f"p_{kind}_holm"] = holm(tests.loc[indices, f"p_{kind}"])
    return tests


def count_and_padding():
    counts, padding = [], []
    for dataset in DATASETS:
        meta = pd.read_parquet(DATA_ROOT / dataset / "decomp/emd.meta.parquet")
        n = meta.n_imfs.to_numpy()
        row = {"dataset": dataset, "n_channels": len(n), "fraction_ge9": float(np.mean(n >= 9))}
        row.update({f"p{p}": float(np.percentile(n, p)) for p in (1, 5, 50, 95, 99)})
        counts.append(row)
        stack = np.load(DATA_ROOT / dataset / "decomp/emd.npy", mmap_mode="r", allow_pickle=False)
        for k in EMD_K:
            absent = n < k
            # Verify the diagnostic against actual all-zero stored Kth slots.
            zero = np.all(stack[:, :, k - 1, :] == 0, axis=-1).reshape(-1)
            assert np.array_equal(absent, zero)
            padding.append({"dataset": dataset, "k": k, "n_channels": len(n),
                            "padded_channels": int(absent.sum()), "padding_fraction": float(absent.mean())})
    return pd.DataFrame(counts), pd.DataFrame(padding)


def processing_tables():
    costs, energies, caps, failures, verification, pilot, validations = [], [], [], [], [], [], []
    for dataset in DATASETS:
        root = run_dir(dataset)
        shape = json.loads((root / "processing_manifest.json").read_text())["shape"]
        n, channels, length = shape
        verification.append(pd.read_csv(root / "pilot/verification.csv"))
        pilot.append(pd.read_csv(root / "pilot/timing.csv"))
        validations.extend(json.loads((root / "default_validation.json").read_text())["records"])
        for tag in all_tags():
            spec = parse_tag(tag)
            slots = 1 if tag == "raw" else (spec.k + 1 if spec.k else (9 if tag == "emd" else len(imf_schedule(length)[0]) + 1))
            row = {"dataset": dataset, "method": tag, "family": spec.family,
                   "construction": spec.construction, "k": slots - 1,
                   "slots": slots, "comp_features": channels * slots * 20,
                   "comp_log_scale_features": channels * slots * 20 + channels,
                   "n_frames": n, "n_channels": channels, "decomposition_cpu_seconds": 0.0,
                   "feature_cpu_seconds": 0.0, "block_wall_seconds_sum": 0.0,
                   "maximum_reconstruction_error": 0.0}
            if tag in BASE_TAGS:
                row["cost_source"] = "historical cache; no new decomposition"
                if tag != "raw":
                    summary = json.loads((DATA_ROOT / dataset / "decomp" / f"{tag}.json").read_text())
                    row["historical_cpu_seconds_per_frame"] = summary["seconds_per_frame"]["cpu_mean"]
            else:
                row["cost_source"] = "new native fits" if spec.construction == "native" else "cache folding only"
                energy_parts = []
                for start in range(0, n, BLOCK_SIZE):
                    metadata = json.loads((root / "blocks" / tag / f"{start:06d}.json").read_text())
                    for column in ("decomposition_cpu_seconds", "feature_cpu_seconds"):
                        row[column] += metadata[column]
                    row["block_wall_seconds_sum"] += metadata["wall_seconds"]
                    row["maximum_reconstruction_error"] = max(row["maximum_reconstruction_error"], metadata["reconstruction_relative_error"])
                    for channel in metadata["diagnostics"]:
                        for attempt in channel["attempts"]:
                            if attempt["error"] is not None:
                                failures.append({"dataset": dataset, "method": tag,
                                                 "frame_id": channel["frame_id"], "channel": channel["channel"],
                                                 "cap": attempt["cap"], "error": attempt["error"],
                                                 "failed_rows": sum(stage["failed_rows"] for stage in attempt["diagnostics"]),
                                                 "resolved_by_retry": attempt["cap"] == 100})
                            for stage in attempt["cap_stages"]:
                                caps.append({"dataset": dataset, "method": tag, "frame_id": channel["frame_id"],
                                             "channel": channel["channel"], "cap": attempt["cap"], "coarse_stage": stage,
                                             "failed_rows": attempt["diagnostics"][stage]["failed_rows"],
                                             "error": attempt["error"] or ""})
                    if spec.construction == "native":
                        with np.load(root / "blocks" / tag / f"{start:06d}.npz", allow_pickle=False) as block:
                            energy_parts.append(block["relative_energy"].reshape(-1, slots))
                if energy_parts:
                    values = np.concatenate(energy_parts)
                    shared_spec = MethodSpec(tag, "imf_linear" if spec.family == "linear" else "irmf", a=native_ratio(length, spec.k))
                    order = scale_order(shared_spec, length)
                    radii = [h * length for h in imf_schedule(length, native_ratio(length, spec.k))[1]][::-1]
                    for slot in range(slots):
                        energies.append({"dataset": dataset, "method": tag, "k": spec.k, "storage_slot": slot,
                                         "scale_rank": order.index(slot), "radius_samples": radii[slot] if slot < spec.k else np.nan,
                                         "slot_label": f"stage_radius_{radii[slot]:.6g}" if slot < spec.k else "residual_fine",
                                         "relative_energy_median": np.median(values[:, slot]),
                                         "relative_energy_p95": np.percentile(values[:, slot], 95), "n_channel_frames": len(values)})
            row["new_decomposition_cpu_seconds_per_frame"] = row["decomposition_cpu_seconds"] / n
            costs.append(row)
    return (pd.DataFrame(costs), pd.DataFrame(energies),
            pd.DataFrame(caps, columns=["dataset", "method", "frame_id", "channel", "cap", "coarse_stage", "failed_rows", "error"]),
            pd.DataFrame(failures, columns=["dataset", "method", "frame_id", "channel", "cap", "failed_rows", "error", "resolved_by_retry"]),
            pd.concat(verification, ignore_index=True), pd.concat(pilot, ignore_index=True), pd.DataFrame(validations))


def summarize(scores):
    keys = ["dataset", "classifier", "feature_set", "method", "n_features"]
    summary = scores.groupby(keys)[list(METRICS)].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(part for part in col if part) if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.merge(scores.groupby(keys).size().reset_index(name="n_splits"), on=keys, validate="one_to_one")
    best = []
    for (dataset, classifier, feature_set), cell in summary.groupby(["dataset", "classifier", "feature_set"]):
        for family in ("linear", "irmf", "emd"):
            candidates = cell[cell.method.map(lambda tag: parse_tag(tag).family == family and
                                               parse_tag(tag).construction in ("native", "fold", "default"))].copy()
            length = 1024 if dataset == "synthetic" else 2048
            candidates["k"] = candidates.method.map(lambda t: parse_tag(t).k or (8 if family == "emd" else len(imf_schedule(length)[0])))
            winner = candidates.sort_values(["balanced_accuracy_mean", "k"], ascending=[False, True]).iloc[0]
            best.append({"dataset": dataset, "classifier": classifier, "feature_set": feature_set, "family": family,
                         "k": int(winner.k), "method": winner.method,
                         "balanced_accuracy": winner.balanced_accuracy_mean, "n_features": winner.n_features,
                         "n_splits": int(winner.n_splits)})
    return summary, pd.DataFrame(best)


def survival_reading(tests):
    primary = tests[(tests.family == "A") & (tests.variant == "classical/comp") & (tests.metric == "balanced_accuracy")]
    rows = []
    for dataset in DATASETS:
        check = primary[(primary.dataset == dataset) & primary.k.isin([5, 6]) & (primary.method_b.str.startswith("emd"))]
        assert len(check) == 4
        rows.append({"dataset": dataset, "survives_matched_counts": bool(((check.mean_diff > 0) & (check.p_nb_holm < .05)).all()),
                     "positive_and_holm_significant": int(((check.mean_diff > 0) & (check.p_nb_holm < .05)).sum()),
                     "required_contrasts": 4})
    return pd.DataFrame(rows)


def plot_curves(summary, figures):
    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True, sharey=True, constrained_layout=True)
        for col, classifier in enumerate((CLASSICAL[dataset], "lightgbm")):
            for row, feature_set in enumerate(FEATURE_SETS):
                axis = axes[row, col]
                cell = summary[(summary.dataset == dataset) & (summary.classifier == classifier) & (summary.feature_set == feature_set)]
                if cell.empty:
                    axis.text(.5, .5, "Omitted by pre-specified budget cut", ha="center", va="center")
                    axis.set_axis_off()
                    continue
                for family in ("emd", "linear", "irmf"):
                    part = cell[cell.method.map(lambda t: parse_tag(t).family == family and parse_tag(t).construction in ("default", "native", "fold"))].copy()
                    length = 1024 if dataset == "synthetic" else 2048
                    default_k = 8 if family == "emd" else len(imf_schedule(length)[0])
                    part["k"] = part.method.map(lambda t: parse_tag(t).k or default_k)
                    part = part.sort_values("k")
                    axis.plot(part.k, 100 * part.balanced_accuracy_mean, "o-", color=COLORS[family], label=DISPLAY[family])
                    last = part[part.k == default_k].iloc[0]
                    axis.scatter([last.k], [100 * last.balanced_accuracy_mean], marker="s", s=65, color=COLORS[family])
                raw = cell[cell.method == "raw"].iloc[0].balanced_accuracy_mean
                axis.axhline(100 * raw, color="#777777", linestyle=":", label="Raw")
                axis.set_title(f"{classifier}, {feature_set}, {int(cell.n_splits.iloc[0])} splits")
                axis.set_xlabel("Extracted components K, plus one residual")
                axis.set_ylabel("Mean balanced accuracy (%)")
                axis.set_xticks(sorted({*EMD_K, *NATIVE_K, 15 if dataset == "synthetic" else 17}))
                axis.grid(alpha=.2)
                if row == 0 and col == 0:
                    axis.legend(fontsize=8)
        fig.suptitle(f"{dataset}: native count curves; squares mark study defaults")
        fig.savefig(figures / f"ba_vs_k_{dataset}.png", dpi=170)
        plt.close(fig)


def plot_forest(tests, figures):
    primary = tests[(tests.family == "A") & (tests.variant == "classical/comp") & (tests.metric == "balanced_accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        part = primary[primary.dataset == dataset].sort_values(["k", "contrast"]).reset_index(drop=True)
        for i, row in part.iterrows():
            color = "#245e8b" if row.p_nb_holm < .05 else "#777777"
            axis.errorbar(100 * row.mean_diff, i, xerr=[[100 * (row.mean_diff - row.ci_low)],
                                                      [100 * (row.ci_high - row.mean_diff)]],
                          fmt="o", color=color, capsize=3, markersize=4)
        axis.set_yticks(range(len(part)), [f"K={r.k}: {r.contrast}" for r in part.itertuples()], fontsize=8)
        axis.invert_yaxis()
        axis.axvline(0, color="black", linewidth=.8)
        axis.grid(axis="x", alpha=.2)
        axis.set_title(dataset)
        axis.set_xlabel("BA difference, percentage points; corrected 95% CI")
    fig.suptitle("Family A: fixed classical model, comp; blue has Holm p < 0.05 across 48 tests")
    fig.savefig(figures / "family_a_forest.png", dpi=180)
    plt.close(fig)


def markdown_tests(table, include_dataset=True):
    rows = []
    for row in table.itertuples():
        record = {"Dataset": row.dataset} if include_dataset else {}
        record.update({"K": row.k, "Contrast": row.contrast,
                       "Estimate, pp": f"{100*row.mean_diff:+.3f}",
                       "95% CI, pp": f"[{100*row.ci_low:+.3f}, {100*row.ci_high:+.3f}]",
                       "Holm p": f"{row.p_nb_holm:.6g}"})
        rows.append(record)
    return pd.DataFrame(rows).to_markdown(index=False)


def compute_totals(scores, costs, pilot):
    sessions = []
    for dataset in DATASETS:
        for path in sorted(run_dir(dataset).glob("*completed*.json")):
            session = json.loads(path.read_text())
            sessions.append({"dataset": dataset, "file": path.name, **session})
    cpu = scores.cpu_seconds.sum() + costs.decomposition_cpu_seconds.sum() + costs.feature_cpu_seconds.sum() + pilot.cpu_seconds.sum()
    wall = sum(row["wall_seconds"] for row in sessions) + 182.04723087500315
    pilot_entries = [json.loads(path.read_text())["time_utc"] for dataset in DATASETS
                     for path in (run_dir(dataset) / "audit").glob("*.json")
                     if json.loads(path.read_text())["runner"] == "pilot"]
    first_pilot = min(datetime.fromisoformat(stamp) for stamp in pilot_entries)
    analysis_time = datetime.now(timezone.utc)
    return {"measured_cpu_seconds": float(cpu), "runner_wall_seconds": float(wall),
            "classification_cpu_seconds": float(scores.cpu_seconds.sum()),
            "decomposition_and_folding_cpu_seconds": float(costs.decomposition_cpu_seconds.sum()),
            "feature_cpu_seconds": float(costs.feature_cpu_seconds.sum()),
            "native_timing_pilot_cpu_seconds": float(pilot.cpu_seconds.sum()),
            "elapsed_since_first_pilot_seconds": (analysis_time - first_pilot).total_seconds(),
            "first_pilot_utc": first_pilot.isoformat(), "analysis_time_utc": analysis_time.isoformat(),
            "new_data_bytes": allocated_bytes(), "pilot_wall_seconds": 182.04723087500315,
            "native_irmf_channel_decompositions": int((costs.loc[(costs.family == "irmf") &
                (costs.construction == "native"), "n_frames"] * costs.loc[(costs.family == "irmf") &
                (costs.construction == "native"), "n_channels"]).sum()),
            "pilot_cap_hit_attempts": int(pilot.cap_hits.sum()),
            "sessions": sessions,
            "wall_scope": "timed processing and classification sections plus the pilot; classification timers begin after input-array preparation",
            "cpu_scope": "fit CPU, block decomposition and feature CPU, and native timing pilot; excludes normalization, input-array preparation, direct pilot verification, JIT/startup, tests and orchestration"}


def write_report(destination, summary, best, tests, reading, counts, padding, costs, caps, failures, verification, validations, totals, plan):
    primary = tests[(tests.variant == "classical/comp") & (tests.metric == "balanced_accuracy")]
    family_a = primary[primary.family == "A"]
    family_c = primary[primary.family == "C"]
    primary_best = best[(best.feature_set == "comp") & (best.classifier != "lightgbm")].copy()
    primary_best["BA (%)"] = (100 * primary_best.balanced_accuracy).map(lambda x: f"{x:.3f}")
    defaults, changes, best_p = [], [], []
    for row in primary_best.itertuples():
        default_tag = "emd" if row.family == "emd" else DEFAULTS[row.family]
        reference = summary[(summary.dataset == row.dataset) & (summary.classifier == row.classifier) &
                            (summary.feature_set == "comp") & (summary.method == default_tag)].iloc[0]
        defaults.append(f"{100*reference.balanced_accuracy_mean:.3f}")
        changes.append(f"{100*(row.balanced_accuracy-reference.balanced_accuracy_mean):+.3f}")
        pair = primary[(primary.family == "B") & (primary.dataset == row.dataset) &
                       (primary.method_a == row.method) & (primary.method_b == default_tag)]
        best_p.append(f"{pair.iloc[0].p_nb_holm:.6g}" if len(pair) else "reference")
    primary_best["Default BA (%)"] = defaults
    primary_best["Best - default, pp"] = changes
    primary_best["Family B Holm p"] = best_p
    padding_table = padding.pivot(index="dataset", columns="k", values="padding_fraction") * 100
    padding_table = padding_table.rename(columns=lambda k: f"K={k}, % padded").reset_index()
    native_costs = costs[costs.construction == "native"].pivot(
        index=["dataset", "family"], columns="k", values="new_decomposition_cpu_seconds_per_frame")
    native_costs = native_costs.rename(columns=lambda k: f"K={k}").reset_index()
    met = reading.loc[reading.survives_matched_counts, "dataset"].tolist()
    unmet = reading.loc[~reading.survives_matched_counts, "dataset"].tolist()
    outcome = "The registered matched-count criterion requires both IMF methods to beat EMD at K=5 and K=6 after Holm correction. "
    outcome += ("It is met for " + ", ".join(met) + ". " if met else "It is met for none of the four datasets. ")
    if unmet:
        outcome += "It is not met for " + ", ".join(unmet) + "."
    lines = ["# Component count and signal classification", "", outcome.strip(), "", "## Question and design", "",
             "Does the IMF advantage over EMD persist when the extracted component count is matched? "
             "This experiment compares clean prepared frames on four datasets using the existing 15 grouped splits. "
             "Every K has K extracted slots and one residual. Native linear IMF and converged IRMF span radius "
             "0.25 L to 1.5 samples. Linear IMF uses Quadratic; IRMF keeps H=2 sigma_hat and uses the shared "
             "NewtonSmoothAbs lookup-score solver at tolerance 1e-12. Any window reaching 100 iterations "
             "triggers a full-channel retry at cap 1000; unresolved fits stop the run. "
             "EMD uses PyEMD defaults and retains its first K IMFs, with zero padding where needed. A separate "
             "construction truncates the default IMF cascade at the coarsest k_max stages.", "",
             "The pre-registration commit is `0b19623`. The [protocol](../../../experiments/classification-emd-irmf/components/PROTOCOL_components.md) "
             "records tags, exact ratios, the 50-frame label-free pilot and the initial 9.93-hour budget projection. "
             "An audit of historical tree thread counts revised that projection to 10.53 hours, still below the cut threshold. "
             "The study default ends at radius 2 samples and has 15 extracted stages for synthetic, 17 for the "
             "other datasets. Native K therefore changes both count and spacing, with a slightly finer endpoint. "
             "Matched counts control input feature dimension; they do not make the decompositions' scale bands identical. "
             "The shared classical pipelines' training-fold constant-feature filter can leave different effective widths.", "",
             "Fixed models are logistic regression for synthetic and BirdCLEF, linear SVM for DroneRF and ESC-50. "
             "DroneRF retains its historical fixed C=0.1; the others reuse the original grouped training-fold searches. "
             "LightGBM reuses the exact shared fit_tree implementation with one thread. Both comp and comp+log_scale "
             "are evaluated. Each channel-slot contributes 20 unchanged shared features. "
             "[Costs and feature counts](costs.csv) list each input's dimensions, checked against the "
             "actual feature width recorded for every fit.", "",
             "## Cache and baseline checks", "",
             f"The direct checks used 50 frames per dataset and every channel. Across {len(verification)} checks, "
             f"the largest relative L2 difference from direct decomposition was {verification.relative_error.max():.3g}, "
             "below 2e-7 after float32 storage. All native channels assert their stage count and final radius. "
             f"The {len(validations)} recomputed historical split/feature-set/metric checks have maximum absolute "
             f"difference {validations.difference.abs().max():.3g}. "
             "[Direct verification](verification.csv), [default score validation](default_validation.csv).", "",
             "The supplied EMD count table is reproduced below. Percentiles are over channels, including both "
             "DroneRF views. [Counts](emd_counts.csv) and [padding by K](padding_fractions.csv) retain full precision.", "",
             counts.to_markdown(index=False, floatfmt=".6g"), "",
             "## Matched counts, family A", "",
             "All estimates below are IRMF minus comparator or linear IMF minus EMD in BA percentage points. "
             "Intervals use the Nadeau-Bengio corrected variance and are marginal 95% intervals. "
             "Two-sided corrected-t p values use Holm across all 48 tests. Wilcoxon is secondary and saved with "
             "its own Holm adjustment. Macro-F1 and accuracy form separate supporting families. "
             "[All paired tests](paired_tests.csv), [family A source](family_a.csv), "
             "[forest plot](figures/family_a_forest.png), [filterable primary tables](overview.html).", "", markdown_tests(family_a), "",
             "The pre-specified reading requires both IMF-vs-EMD gains to be positive and Holm-significant at "
             "K=5 and K=6. The result for each dataset is:", ""]
    for row in reading.itertuples():
        lines.append(f"- {row.dataset}: {'survives matched counts' if row.survives_matched_counts else 'does not meet the pre-specified criterion'}, "
                     f"{row.positive_and_holm_significant}/4 required contrasts are positive and significant.")
        if not row.survives_matched_counts:
            matched = family_a[(family_a.dataset == row.dataset) & family_a.k.isin([5, 6]) &
                               family_a.method_b.str.startswith("emd")]
            lines.append(f"  The four point gains range from {100*matched.mean_diff.min():+.3f} to "
                         f"{100*matched.mean_diff.max():+.3f} pp.")
            missing = matched[matched.p_nb_holm >= .05]
            if len(missing) == 1:
                failed = missing.iloc[0]
                lines.append(f"  The failed requirement is {failed.contrast} at K={failed.k}: "
                             f"{100*failed.mean_diff:+.3f} pp [{100*failed.ci_low:+.3f}, "
                             f"{100*failed.ci_high:+.3f}], Holm p={failed.p_nb_holm:.6g}.")
    lines += ["", "## Native count curves, family B", "",
              "Each of 17 counts-versus-default contrasts per dataset has its own paired difference and "
              "corrected interval in [family_b.csv](family_b.csv), with Holm within the dataset. "
              "A best observed K is descriptive, chosen from the native grid and the study default; it does not "
              "estimate performance after independent model selection. The primary comp results are:", "",
              primary_best[["dataset", "family", "k", "BA (%)", "Default BA (%)", "Best - default, pp",
                            "Family B Holm p", "n_features"]].to_markdown(index=False), ""]
    for dataset in DATASETS:
        part = primary[(primary.family == "B") & (primary.dataset == dataset)]
        sig = part[part.p_nb_holm < .05]
        lines.append(f"- {dataset}: {len(sig)}/17 counts differ from default after Holm; "
                     f"{int((sig.mean_diff < 0).sum())} are lower and {int((sig.mean_diff > 0).sum())} are higher. "
                     f"[BA curves for both classifiers and feature sets](figures/ba_vs_k_{dataset}.png).")
    lines += ["", "The best observed EMD count is three or four on every dataset with the fixed classical model. "
              "For the datasets where that best-count improvement passes family B's correction:", ""]
    for row in primary_best[primary_best.family == "emd"].itertuples():
        pair = primary[(primary.family == "B") & (primary.dataset == row.dataset) &
                       (primary.method_a == row.method)]
        if len(pair) and pair.iloc[0].p_nb_holm < .05:
            effect = pair.iloc[0]
            lines.append(f"- {row.dataset}, EMD K={row.k}: {100*effect.mean_diff:+.3f} pp "
                         f"[{100*effect.ci_low:+.3f}, {100*effect.ci_high:+.3f}], Holm p={effect.p_nb_holm:.6g}, versus K=8.")
    lines += ["", "## Default k_max truncation, family C", "",
              "These inputs keep the coarsest sequential stages. Their final residual combines every finer "
              "stage and the finest solver residual. They therefore answer a different question from the native "
              "K-stage cascades. Holm spans eight contrasts per dataset. "
              "[Family C source](family_c.csv).", "", markdown_tests(family_c), "",
              "For pimf's GD60/k_max=8 versus the same GD60 full cascade:", ""]
    for row in family_c[family_c.method_b == "irmf_h2_a1.41"].itertuples():
        lines.append(f"- {row.dataset}: {100*row.mean_diff:+.3f} pp "
                     f"[{100*row.ci_low:+.3f}, {100*row.ci_high:+.3f}], Holm p={row.p_nb_holm:.6g}.")
    lines += ["", "## Secondary results", "",
              "The exploratory variants have separate corrections: 48 family-A tests across datasets and "
              "eight family-C tests within each dataset before budget cuts, separately for classical/comp+log_scale, "
              "LightGBM/comp and LightGBM/comp+log_scale. Fixed-classical results retain 15 splits. "
              f"LightGBM uses {5*len(plan['lightgbm_seeds'])} splits on seeds {plan['lightgbm_seeds']}, "
              f"with feature sets {plan['lightgbm_feature_sets']}. "
              + (f"All {int(summary.n_splits.sum()):,} completed fits are in the planned scope. " if not plan['cuts'] else
                 "Completed default fits outside that scope are saved in per_split_scores.csv with analysis_eligible=False. ") +
              "[Scores](per_split_scores.csv), [mean scores](score_summary.csv), "
              "[best observed K](best_k.csv) and [secondary test details](paired_tests.csv) provide the numbers.", ""]
    secondary_rows = []
    for variant in ("classical/comp+log_scale", "lightgbm/comp", "lightgbm/comp+log_scale"):
        part = tests[(tests.variant == variant) & (tests.metric == "balanced_accuracy")]
        if part.empty:
            continue
        a, c = part[part.family == "A"], part[part.family == "C"]
        positive = ((a.mean_diff > 0) & (a.p_nb_holm < .05)).sum()
        negative = ((a.mean_diff < 0) & (a.p_nb_holm < .05)).sum()
        lines.append(f"- {variant}: family A has {positive} positive and {negative} negative Holm-significant "
                     f"contrasts out of {len(a)}. Family C has {int((c.p_nb_holm < .05).sum())}/{len(c)} significant contrasts.")
        for dataset in DATASETS:
            matched = a[(a.dataset == dataset) & a.k.isin([5, 6]) & a.method_b.str.startswith("emd")]
            gd = c[(c.dataset == dataset) & (c.method_b == "irmf_h2_a1.41")].iloc[0]
            secondary_rows.append({"Variant": variant, "Dataset": dataset,
                                   "K=5,6 gains positive/significant": f"{int(((matched.mean_diff > 0) & (matched.p_nb_holm < .05)).sum())}/{len(matched)}",
                                   "GD60 k8 - full, pp": f"{100*gd.mean_diff:+.3f}",
                                   "C Holm p": f"{gd.p_nb_holm:.6g}"})
    lines += ["", "The exploratory summary below counts positive, Holm-significant IMF-vs-EMD gains at "
              "K=5 and K=6, out of the available four contrasts. It also shows the GD60/k_max=8 change "
              "against GD60/full. The primary matched-count reading remains the classical/comp family A.", "",
              pd.DataFrame(secondary_rows).to_markdown(index=False), ""]
    for dataset in DATASETS:
        part = summary[(summary.dataset == dataset) & (summary.feature_set == "comp")]
        raw = part[part.method == "raw"]
        lines.append(f"- {dataset} raw comp BA: " + "; ".join(
            f"{r.classifier} {100*r.balanced_accuracy_mean:.3f}%" for r in raw.itertuples()) + ".")
    lines += ["", "## Slot contents and padding", "",
              "[Slot energies](slot_energies.csv) report median and 95th-percentile relative power for every "
              "native slot, pooled over frame-channels, plus radius and true scale rank. Energy uses the shared feature definition, each "
              "slot's power divided by the sum of slot powers. The decompositions are not orthogonal, so this "
              "is not an additive partition of input power. Slot numbers alone do not equate scales across "
              "methods or across native schedules. The IMF solver residual is the finest content. "
              "[Padding fractions](padding_fractions.csv) report absent Kth EMD slots, checked against all-zero "
              "cached slots. Feature count matches across methods at matched K even where EMD has padding.", "",
              padding_table.to_markdown(index=False, floatfmt=".4f"), "",
              "## Compute cost and solver integrity", "",
              f"Timed processing and classification sections, plus the pilot, total {totals['runner_wall_seconds']/3600:.3f} wall hours. "
              "Classification timers begin after input-array preparation. "
              f"Calendar elapsed time from the first pilot entry to analysis is {totals['elapsed_since_first_pilot_seconds']/3600:.3f} hours, "
              "including tests, commits, implementation and waits between phases. "
              f"The summed measured CPU time is {totals['measured_cpu_seconds']/3600:.3f} hours. CPU includes "
              "classification fits, block decomposition/feature work and native pilot fits; it excludes direct "
              "normalization, input-array preparation, direct pilot verification, worker JIT/startup, tests and orchestration. These are concurrent-load measurements. "
              f"New external data occupy {totals['new_data_bytes']/1e9:.3f} GB against the 60 GB cap. "
              "[Session timings](compute_totals.json), [costs per input](costs.csv), [pilot timings](pilot_timings.csv).", "",
              "Measured native decomposition CPU seconds per complete frame, including every channel, "
              "are below. Shared normalization and feature extraction are separate. These are full-population "
              "measurements rather than the pilot projections.", "",
              native_costs.to_markdown(index=False, floatfmt=".6f"), "",
              f"There are {len(caps)} recorded Newton window-stage cap hits, across "
              f"{len(caps[['dataset','method','frame_id','channel']].drop_duplicates()) if len(caps) else 0} channels. "
              f"The full population includes {totals['native_irmf_channel_decompositions']:,} native IRMF channel decompositions. "
              f"The timing pilot recorded {totals['pilot_cap_hit_attempts']} cap-hit attempts. "
              "Every accepted native IRMF channel converged. There are zero unresolved Newton failures. "
              f"There are {len(failures)} failed solver attempts before any retry. "
              "[Cap events](newton_cap_events.csv) preserves every 100-iteration retry and its failed-row count. "
              "[Failed attempts](newton_failures.csv) includes errors independently of cap status. "
              "pimf stayed at 0.4.0 with all five installed Python-file hashes equal to the initial fingerprint. "
              "Each runner's complete fingerprint record stays under the external components/audit/ directories. "
              "The [final integrity record](integrity.json) also confirms that model, construction and shared "
              "feature code hashes match the manifests used by their runs.", "",
              "## Deviations and limits", "",
              "No fixed-classical dataset, contrast, split, component count or feature set was cut. "
              + ("No LightGBM budget cut was triggered. " if not plan['cuts'] else
                 "The pre-specified LightGBM cuts applied were: " + "; ".join(plan['cuts']) + ". ") +
              "No shared study module or installed pimf file changed. "
              "During setup, the initial pilot and test commands were launched concurrently, an execution oversight. Their combined worker peak "
              "was not recorded, so compliance with the aggregate worker cap cannot be verified for that "
              "initial interval. Subsequent processing, classification and test phases ran sequentially. "
              "The label-free pilot supplied timing only. "
              "The protocol explicitly retained the historical DroneRF C=0.1 rather than the generic SVM grid. "
              "After pre-registration, the historical LightGBM timing sources were found to use two threads for "
              "synthetic and BirdCLEF, rather than the single-thread assumption used in the initial CPU-equivalent "
              "projection. Multiplying those wall times by their declared thread counts raised the projection "
              "from 9.93 to 10.53 hours. This timing correction used no new-input scores and triggered no cut. "
              "The appended protocol run notes and external budget_timing_audit.json preserve the correction. "
              "The new default LightGBM fits then showed higher CPU costs than those historical timings. "
              "Before classifying any new input, the default-only timing measurements were used to freeze "
              f"budget tier {plan['level']}. The uncut projection was {plan['projections'][0]['projected_total_wall_hours']:.3f} hours; "
              f"the selected tier projected {plan['projected_total_wall_hours']:.3f} hours. "
              "[The complete timing-only decision](budget_plan.json) records every "
              "tested cut tier, completed wall time and projected remaining CPU cost. It uses no new-input scores. "
              "The CPU total has the measurement exclusions stated above, so it is not a complete operating-system "
              "accounting of every process. These timing exclusions changed no fitted input or result.", "",
              "This comparison concerns the fixed prepared populations and chosen classifiers. Matching feature "
              "count does not match information per slot, active EMD components, or the scale spacing; "
              "classical training-fold removal of constant features can also leave different effective widths. "
              "Best-K estimates use the same outer folds and remain descriptive.", "",
              "## Reproduce", "", "```bash",
              "cd /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf",
              "export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification",
              "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
              "export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
              "../../.venv/bin/python -m pytest -q tests components/tests",
              "../../.venv/bin/python -m components.processing --workers 6",
              "../../.venv/bin/python -m components.classification --phase defaults --workers 6",
              "../../.venv/bin/python -m components.budget --append-notes",
              "../../.venv/bin/python -m components.run_budgeted --workers 6",
              "../../.venv/bin/python -m components.analysis --out /absolute/path/to/fresh/export",
              "```", "", "Completed checkpoints resume without replacement. The original label-free pilot is already "
              "recorded and is not rerun over its immutable outputs. The full instructions and initial fingerprint "
              "are in the [components README](../../../experiments/classification-emd-irmf/components/README.md), "
              "including detached launches with fresh logs and PID records. The production new-input job "
              "used nohup, a new session, closed stdin and regular-file stdout/stderr.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for dataset in DATASETS:
        record_entry(dataset, "final_analysis")
    check_pimf()
    if not (run_dir("synthetic") / "budget_plan.json").exists():
        raise RuntimeError("Freeze the measured-default timing budget before final analysis")
    plan = load_plan()
    code_records = verify_code_manifests()
    scores = load_scores(plan)
    analytic = scores[scores.analysis_eligible].copy()
    tests = paired_tests(analytic, plan)
    summary, best = summarize(analytic)
    reading = survival_reading(tests)
    counts, padding = count_and_padding()
    costs, energies, caps, failures, verification, pilot, validations = processing_tables()
    feature_counts = {(row.dataset, row.method, feature_set): int(count)
                      for row in costs.itertuples()
                      for feature_set, count in (("comp", row.comp_features),
                                                 ("comp+log_scale", row.comp_log_scale_features))}
    assert all(row.n_features == feature_counts[row.dataset, row.method, row.feature_set]
               for row in scores.itertuples())
    totals = compute_totals(scores, costs, pilot)
    assert totals["new_data_bytes"] <= 60_000_000_000
    # Build a fresh external draft first. Never replace a published export.
    draft = run_dir("synthetic") / f"export_{time.time_ns()}"
    figures = draft / "figures"
    figures.mkdir(parents=True)
    tables = {"per_split_scores": scores, "score_summary": summary, "paired_tests": tests,
              "best_k": best, "survival_reading": reading, "emd_counts": counts,
              "padding_fractions": padding, "costs": costs, "slot_energies": energies,
              "newton_cap_events": caps, "newton_failures": failures, "verification": verification,
              "pilot_timings": pilot, "default_validation": validations}
    for family in ("A", "B", "C"):
        tables[f"family_{family.lower()}"] = tests[(tests.family == family) &
                                                  (tests.variant == "classical/comp") &
                                                  (tests.metric == "balanced_accuracy")]
    for name, table in tables.items():
        table.to_csv(draft / f"{name}.csv", index=False)
    write_json(draft / "compute_totals.json", totals)
    write_json(draft / "budget_plan.json", plan)
    timing_audit = run_dir("synthetic") / "budget_timing_audit.json"
    if timing_audit.exists():
        write_json(draft / "budget_timing_audit.json", json.loads(timing_audit.read_text()))
    write_json(draft / "integrity.json", {"pimf": check_pimf(), "pre_registration_commit": "0b19623",
               "protocol_sha256": file_hash(HERE / "PROTOCOL_components.md"),
               "analysis_sha256": file_hash(Path(__file__)),
               "presentation_sha256": file_hash(HERE / "presentation.py"),
               "verified_code_manifests": code_records})
    plot_curves(summary, figures)
    plot_forest(tests, figures)
    write_report(draft, summary, best, tests, reading, counts, padding, costs, caps, failures, verification, validations, totals, plan)
    write_overview(draft, tests, best, reading, totals)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise RuntimeError(f"refuse to overwrite existing export: {args.out}; fresh draft is {draft}")
    for path in draft.iterdir():
        if path.is_dir():
            shutil.copytree(path, args.out / path.name)
        else:
            shutil.copy2(path, args.out / path.name)
    print(f"Exported {args.out}\n{reading.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
