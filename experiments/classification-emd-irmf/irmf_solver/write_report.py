"""Build REPORT.md from the external audit caches; never copy data into git."""

import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
AUDIT = ROOT / "solver_audit"
HERE = Path(__file__).resolve().parent
NAMES = ("dronerf", "birdclef2023", "esc50_animals")


def table(frame, columns=None, digits=3):
    selected = (frame if columns is None else frame[columns]).copy()
    for column in selected:
        selected[column] = selected[column].map(
            lambda value: (f"{value:.{digits}f}" if np.isfinite(value) else "n/a")
            if isinstance(value, (float, np.floating)) else value
        )
    return selected.to_markdown(index=False, disable_numparse=True)


def main():
    overview = pd.read_csv(AUDIT / "diagnosis_overview.csv")
    stages = pd.read_csv(AUDIT / "diagnosis_stage.csv")
    classes = pd.read_csv(AUDIT / "diagnosis_class.csv")
    benchmark = pd.read_csv(AUDIT / "benchmark.csv")
    validation = json.loads((AUDIT / "validation.json").read_text())
    mechanism = pd.read_csv(AUDIT / "mechanism.csv")
    lines = ["# IRMF convergence and solver audit", "",
             "Measured on 2026-10-20 in the supplied worktree, pimf 0.4.0, Python 3.12. "
             "At most two computational processes ran concurrently, each with one native thread. "
             "Code, shared study files, and the installed pimf package were left unchanged outside this folder. "
             "No full-dataset decomposition was run.", "", "## Existing-cache diagnosis", "",
             "These tables use every successful channel in the original caches. DroneRF has two channels per frame; "
             "all other datasets have one. Times sum channels within a frame. Reconstruction does not establish "
             "convergence, because the residual telescopes for any sequence of fitted components.", "",
             table(overview, ["dataset", "n_frames", "n_channels", "unconverged_stage_pct",
                              "any_unconverged_frame_pct", "wall_s_per_frame_mean", "cpu_s_per_frame_mean"]), "",
             "Stage failure percentages in canonical fine-to-coarse order:", "",
             stages.pivot(index="window", columns="dataset", values="unconverged_pct").reset_index().to_markdown(index=False, floatfmt=".2f"), "",
             "Iteration distributions over all stages, and the stopping ratio "
             "r = final_max_delta / [1e-6 (1 + max |fitted component|)]. The denominator uses cached components, "
             "rather than input amplitude. Float32 caching reproduces every convergence flag exactly. "
             "r > 1 denotes a capped unconverged stage.", "",
             table(overview, ["dataset", "iterations_q05", "iterations_q50", "iterations_q95", "iterations_max",
                              "delta_ratio_q50", "delta_ratio_q95", "delta_ratio_max", "failed_delta_ratio_q50"]), "",
             "Wall seconds per frame:", "",
             table(overview, ["dataset", "wall_seconds_q05", "wall_seconds_q50", "wall_seconds_q95", "wall_seconds_max"]), "",
             "Primary-task class failure percentages below count stage-channel fits. All secondary tasks, "
             "iteration quantiles by window, and delta-ratio quantiles by window are in the external CSV files.", ""]
    primary = {"dronerf": "10class", "birdclef2023": "species", "esc50_animals": "animal", "synthetic": "signal"}
    keep = [row.task == primary[row.dataset] for row in classes.itertuples()]
    lines += [table(classes[keep], ["dataset", "class", "n_channels", "unconverged_stage_pct", "any_unconverged_channel_pct", "H_median"]), "",
              "## Why gradient descent stops late", "",
              "The installed `_gd_fit_windows` starts at the **unweighted** window median, although the objective uses "
              "squared-triangle weights. It sets step = 0.95 H / sqrt(2/pi), so |delta| <= 1.190648 H. "
              "Small H relative to a burst or a bimodal window can therefore impose many transit steps. "
              "More generally, the local convergence factor is approximately "
              "1 - 0.95 sum(w psi') / [sqrt(2/pi)/H]; it approaches one in flat score regions.", "",
              "The single global maximum-delta criterion waits for the slowest of all L windows. Every row keeps "
              "being evaluated until that happens. Its tolerance uses the largest fitted value anywhere in the stage, "
              "so a small last update is also not a uniform bound on distance to each root.", "",
              "The study computes H once as 2 median(|diff(y)|) / (0.6745 sqrt(2)), on the robustly normalized "
              "input channel, then reuses it through the cascade. This is an uncentered median absolute difference, "
              "not median(|diff(y) - median(diff(y))|). The mean-absolute-difference fallback does not occur in these caches. "
              "Smooth, high-amplitude signals can have a small difference-based H despite large local spread.", ""]
    toy = validation["flat_score_toy"]
    lines += [f"At window 7, radius 2 sqrt(2), the center weight is {toy['center_weight']:.8f}. "
              f"A window with center 100 and all other values 0, H=1, has score root {toy['root']:.8f}. "
              f"GD60 returns {toy['gd60']['x']:.8f}, an error of {toy['gd60']['abs_error']:.6g}; "
              f"GD1000 still has error {toy['gd1000']['abs_error']:.6g}. "
              "This verifies the flat-curvature mechanism without relying on a transit-distance explanation.", "",
              "The next table uses five hardest subset frames per dataset. Roots are solved on the exact inputs "
              "seen by each default stage, before its residual is updated. It reports the largest initial "
              "median-to-root distance in H units, the strongest transit lower bound, the smallest relative curvature, "
              "and the median percentage of still-active rows in failed stages.", ""]
    mech = []
    for name, cell in mechanism.groupby("dataset", sort=False):
        bad = cell[~cell.default_converged]
        mech.append({"dataset": name, "distance_H_max": cell.initial_distance_H_max.max(),
                     "move_cap_min_steps_max": cell.move_cap_lower_bound_iterations.max(),
                     "relative_curvature_min": cell.curvature_ratio_min.min(),
                     "failed_active_row_pct_median": bad.still_active_row_pct.median()})
    mech_table = pd.DataFrame(mech)
    mech_table["relative_curvature_min"] = mech_table.relative_curvature_min.map(lambda x: f"{x:.3e}")
    lines += [table(mech_table, digits=5), "", "## Subset sensitivity", "",
              "Seed 20261020, ten frames per primary class sampled across groups, first channel: "
              "100 DroneRF, 160 BirdCLEF, 100 ESC-50. GD60, GD300, and GD1000 call unchanged "
              "`emdirmf.decompose.irmf_components`. Every setting except the iteration cap is identical. "
              "Newton uses the same H, kernel, wrap boundary, schedule, and storage order.", ""]
    comp_rows, feature_rows, costs_all, stack_rows, verification = [], [], [], [], []
    for name in NAMES:
        out = AUDIT / name
        costs = pd.read_csv(out / "costs.csv")
        costs_all.append(costs)
        comp = pd.read_csv(out / "component_differences.csv")
        features = pd.read_csv(out / "feature_differences.csv")
        for (method, reference), cell in comp.groupby(["method", "reference"]):
            if method == reference:
                continue
            comp_rows.append({"dataset": name, "method": method, "reference": reference,
                              "component_rel_l2_median": cell.rel_l2.median(),
                              "component_rel_l2_p95": cell.rel_l2.quantile(.95),
                              "component_rel_l2_max": cell.rel_l2.max(),
                              "component_pct_above_1pct": 100 * (cell.rel_l2 > .01).mean(),
                              "input_rel_l2_p95": cell.input_relative_l2.quantile(.95)})
        for (method, reference), cell in features.groupby(["method", "reference"]):
            if method == reference:
                continue
            for include_scale in (False, True):
                rho = cell.spearman.dropna().to_numpy()
                errors = cell.nrmse.dropna().to_numpy()
                if include_scale:
                    rho = np.r_[rho, 1.0]
                    errors = np.r_[errors, 0.0]
                feature_rows.append({"dataset": name, "method": method, "reference": reference,
                    "feature_set": "+scale" if include_scale else "components",
                    "rho_median": np.median(rho), "rho_p05": np.quantile(rho, .05),
                    "rho_min": rho.min(), "nrmse_median": np.median(errors),
                    "nrmse_p95": np.quantile(errors, .95),
                    "features_rho_below_0.95_pct": 100 * np.mean(rho < .95)})
        cached = np.load(ROOT / name / "decomp" / "irmf_h2_a1.41.npy", mmap_mode="r")
        ids = json.loads((out / "selection.json").read_text())["frame_ids"]
        cache_max = 0.0
        exact_count = 0
        for frame_id in ids:
            with np.load(out / f"frame_{frame_id}.npz") as data:
                cache_delta = data["gd60"].astype(np.float32) - cached[frame_id, 0]
                cache_max = max(cache_max, np.max(np.abs(cache_delta)))
                exact_count += np.array_equal(data["gd60"].astype(np.float32), cached[frame_id, 0])
                for reference in ("gd1000", "newton"):
                    for method in ("gd60", "gd300", "gd1000"):
                        stack_rows.append({"dataset": name, "frame_id": frame_id, "method": method,
                            "reference": reference,
                            "relative_l2": np.linalg.norm(data[method] - data[reference]) / np.linalg.norm(data[reference])})
        verification.append({"dataset": name, "n": len(ids), "bitwise_matches_float32": exact_count, "cache_max_abs_difference": cache_max})
    costs = pd.concat(costs_all)
    comp_summary = pd.DataFrame(comp_rows)
    feature_summary = pd.DataFrame(feature_rows)
    stack = pd.DataFrame(stack_rows)
    comp_summary.to_csv(AUDIT / "component_summary.csv", index=False)
    feature_summary.to_csv(AUDIT / "feature_summary.csv", index=False)
    stack.to_csv(AUDIT / "stack_differences.csv", index=False)
    save = pd.DataFrame(verification)
    save.to_csv(AUDIT / "cache_reproduction.csv", index=False)
    lines += ["The fresh GD60 decompositions reproduce the stored float32 cache bit for bit:", "", table(save), "",
              "Unconverged stage counts and current CPU seconds per channel:", ""]
    cost_summary = []
    for (name, method), cell in costs.groupby(["dataset", "method"]):
        cost_summary.append({"dataset": name, "method": method, "n": len(cell),
             "unconverged_stage_pct": 100 * cell.unconverged_stages.sum() / (17 * len(cell)),
             "frames_any_unconverged_pct": 100 * (cell.unconverged_stages > 0).mean(),
             "cpu_seconds_mean": cell.cpu_seconds.mean(), "wall_seconds_mean": cell.seconds.mean()})
    lines += [table(pd.DataFrame(cost_summary)), "",
              "Component relative L2 errors, each component including the residual counted separately. "
              "The input-relative column divides by the input frame norm and limits the influence of tiny components.", "",
              table(comp_summary, digits=6), "", "Whole component-stack relative L2 errors against Newton:", ""]
    stack_summary = stack[stack.reference == "newton"].groupby(["dataset", "method"]).relative_l2.agg(
        median="median", p95=lambda x: x.quantile(.95), maximum="max").reset_index()
    default_stack = stack_summary[stack_summary.method == "gd60"]
    default_features = feature_summary[(feature_summary.method == "gd60") &
                                      (feature_summary.reference == "newton") &
                                      (feature_summary.feature_set == "components")]
    interpretation = "At max_iter=60, the 95th-percentile whole-stack relative errors are " + ", ".join(
        f"{row.dataset} {100 * row.p95:.2f}%" for row in default_stack.itertuples()) + ". "
    interpretation += "The fractions of feature columns with Spearman correlation below .95 are " + ", ".join(
        f"{row.dataset} {row.features_rho_below_0_95_pct:.1f}%" for row in
        default_features.rename(columns={"features_rho_below_0.95_pct": "features_rho_below_0_95_pct"}).itertuples()) + ". "
    interpretation += "Exact reconstruction therefore does not imply stable components or stable feature values."
    lines += [table(stack_summary, digits=6), "", interpretation, "",
              "Features are the unchanged 20 emdirmf values per slot. Spearman correlations compare each feature "
              "across subset frames. NRMSE divides feature RMSE by that feature's across-frame reference SD. "
              "Constant reference columns are omitted. The shared log_scale is identical by construction. "
              "The complete per-feature MAE, maximum error, and rank correlation are in the external cache.", "",
              table(feature_summary), "", "## Classification sensitivity", "",
              "Exploratory fixed shrinkage LDA, preprocessing learned in each training fold, same saved 5-fold x "
              "3-seed group splits intersected with the subset. DroneRF uses its available binary study split "
              "and labels; BirdCLEF uses species and ESC-50 uses animal labels. Some subset test folds lack "
              "classes. Balanced accuracy uses present test classes; macro-F1 follows the shared metric implementation. "
              "The small sample cannot establish whether IRMF outperforms EMD.", ""]
    folds = pd.read_csv(AUDIT / "classification_folds.csv")
    support = folds.drop_duplicates(["dataset", "split_id"]).groupby("dataset").agg(
        n_train_min=("n_train", "min"), n_train_max=("n_train", "max"),
        n_test_min=("n_test", "min"), n_test_max=("n_test", "max"),
        test_classes_min=("n_test_classes", "min"), test_classes_max=("n_test_classes", "max")).reset_index()
    lines += ["Subset fold support:", "", table(support), ""]
    classification = folds.groupby(["dataset", "feature_set", "method"])[["balanced_accuracy", "macro_f1", "accuracy"]].mean().reset_index()
    lines += [table(classification), "",
              "Paired balanced-accuracy differences are method_a minus method_b. Nadeau-Bengio corrected "
              "resampled t-test and its corrected 95% CI are primary, Wilcoxon secondary. Holm adjustment "
              "covers all 30 rows of this table. Separate macro-F1 and accuracy families, with the same "
              "30-row correction, are retained in classification_comparisons.csv.", ""]
    comparisons = pd.read_csv(AUDIT / "classification_comparisons.csv")
    primary_tests = comparisons[(comparisons.metric == "balanced_accuracy") & (comparisons.method_a == "gd60") & (comparisons.method_b == "newton")]
    lines += [table(comparisons[comparisons.metric == "balanced_accuracy"], ["dataset", "feature_set", "method_a", "method_b",
                   "mean_diff", "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm"]), "",
              f"For GD60 versus Newton, {int((primary_tests.p_nb_holm < .05).sum())} of six "
              "balanced-accuracy comparisons reject after correction. A nonsignificant result on this "
              "small subset is not evidence of equivalence. Full-study classification sensitivity remains to be measured.", "",
              "## Prototype accuracy and speed", "",
              "`NewtonSmoothAbs` uses pimf's public closed-form-solver hook to preserve the existing driver. "
              "It starts at the weighted mean, brackets each score root, uses the exact lookup-table slope "
              "for Newton, and falls back to bisection for unsafe steps or every eighth iteration. "
              "Independent row stopping and a fused single-threaded Numba loop avoid repeated work and "
              "large broadcast temporaries. A row failing its stopping test raises an error.", "",
              "Important numerical distinction: lookup=True's score is piecewise linear and is not the exact "
              "derivative of the interpolated loss returned by SmoothAbs.__call__. This prototype matches the "
              "score equation actually used by pimf. In saturated floating-point tails it can have flat root "
              "intervals; the ideal analytic SmoothAbs objective is strictly convex. No equivalence to the "
              "literal interpolated loss minimizer is claimed.", "",
              f"Independent SciPy brentq checks cover 240 windows at H=.01, 1, 10 and widths 5, 7, 65, 1025, "
              f"including constants and center outliers. Maximum absolute root discrepancy "
              f"{max(r['max_abs_error_vs_brentq'] for r in validation['roots']):.3g}.", "",
              "Timing signals are fixed noisy and smooth/bursty examples at every L, plus a bimodal "
              "stress case at L=1024. Its 20000-iteration failure motivated stopping an in-progress "
              "L=2048 bimodal repeat and omitting larger stress repeats; they are not accuracy checks. "
              "Tight-request pimf uses max_iter=20000 and tol=1e-12, prototype root_tol=1e-12. "
              "JIT compilation is excluded. Each cell is one measured run, so speed ratios are descriptive; "
              "the real-frame subset above supplies a larger timing sample. CPU times reduce interference "
              "from concurrent machine work. A tight-GD run with any unconverged stages is not an accuracy "
              "reference; its runtime is a lower bound on the time to convergence. Such stress cases remain "
              "in the table, with failure counts, rather than being silently dropped.", ""]
    bench_rows = []
    for (length, signal), cell in benchmark.groupby(["length", "signal"]):
        by = cell.set_index("method")
        bench_rows.append({"L": length, "signal": signal,
             "gd60_cpu_s": by.loc["gd60", "cpu_seconds"],
             "tight_cpu_s": by.loc["tight_gd", "cpu_seconds"],
             "newton_cpu_s": by.loc["newton", "cpu_seconds"],
             "speedup_vs_gd60": by.loc["gd60", "cpu_seconds"] / by.loc["newton", "cpu_seconds"],
             "speedup_vs_tight": by.loc["tight_gd", "cpu_seconds"] / by.loc["newton", "cpu_seconds"],
             "newton_rel_l2": by.loc["newton", "relative_l2_vs_tight"],
             "max_component_rel_l2": by.loc["newton", "max_component_relative_l2_vs_tight"],
             "tight_unconverged": by.loc["tight_gd", "unconverged"]})
    bench_summary = pd.DataFrame(bench_rows)
    bench_summary.to_csv(AUDIT / "benchmark_summary.csv", index=False)
    bench_table = bench_summary.copy()
    for column in ("newton_rel_l2", "max_component_rel_l2"):
        bench_table[column] = bench_table[column].map(lambda x: f"{x:.3e}")
    valid_bench = bench_summary[bench_summary.tight_unconverged == 0]
    assert len(valid_bench) > 0 and valid_bench.newton_rel_l2.max() < 1e-6
    lines += [table(bench_table, digits=4), "",
              f"Across the {len(valid_bench)} fully converged tight-GD references, maximum Newton "
              f"stack relative L2 discrepancy is {valid_bench.newton_rel_l2.max():.3g}; "
              f"maximum individual-component relative discrepancy is {valid_bench.max_component_rel_l2.max():.3g}. "
              "Only those rows validate agreement with tightly converged pimf.", "",
              f"Largest reconstruction absolute error in the benchmark: {benchmark.reconstruction_max_abs.max():.3g}.", "",
              "## Recommendation and full-study cost", "",
              "Keep the default GD60 tag as the protocol's primary result. Add a fully converged solver "
              "as `irmf_h2_a1.41_conv`, with separate provenance and convergence diagnostics, then repeat the "
              "full paired classification analysis with and without log_scale. GD1000 is not an unconditional "
              "accuracy reference. Class-dependent solver error can itself become a classification feature. "
              "The proposed `_conv` tag needs dispatcher/parser support before use; those shared modules "
              "were intentionally left untouched under this task's scope restriction.", "",
              "For an upstream pimf change, expose a public lookup-compatible score derivative, retain the "
              "reference solver, and make safeguarded Newton an explicit option. Numba acceleration should "
              "be optional. Existing StageInfo reports iterations=1 on the solve hook; use the prototype's "
              "own per-stage diagnostics when recording Newton work. No installed or source pimf files were changed.", "",
              "Cost estimate uses mean measured Newton CPU seconds, 12,000 two-channel DroneRF frames, "
              "12,000 mono BirdCLEF, and 5,100 mono ESC-50, all L=2048. DroneRF's second-channel estimate "
              "comes from 34 additional H-band frames. Parallel estimates assume linear scaling and exclude "
              "normalization, feature extraction, I/O, JIT startup, and scheduling contention.", ""]
    second = pd.read_csv(AUDIT / "dronerf/second_channel_costs.csv")
    estimates = []
    for name, n_frames in zip(NAMES, (12000, 12000, 5100)):
        channel_mean = costs[(costs.dataset == name) & (costs.method == "newton")].cpu_seconds.mean()
        frame_mean = channel_mean + second.cpu_seconds.mean() if name == "dronerf" else channel_mean
        estimates.append({"dataset": name, "frames": n_frames,
            "cpu_s_per_frame": frame_mean, "one_core_minutes": n_frames * frame_mean / 60,
            "two_core_minutes": n_frames * frame_mean / 120,
            "six_core_minutes": n_frames * frame_mean / 360})
    estimate = pd.DataFrame(estimates)
    estimate.to_csv(AUDIT / "full_cost_estimate.csv", index=False)
    lines += [table(estimate), "",
              f"Total: {estimate.one_core_minutes.sum():.1f} CPU-minutes, "
              f"{estimate.two_core_minutes.sum():.1f} minutes on two cores or "
              f"{estimate.six_core_minutes.sum():.1f} minutes on six cores under ideal scaling. "
              "Allowing 2x for overhead and load gives a conservative planning range. "
              "The full float32 decomposition cache would occupy about 6.06 GB decimal, before metadata.", "",
              "## Files and reproducibility", "",
              "Code: `newton.py`, `run_study.py`, `write_report.py`; instructions: `README.md`. "
              "The README lists exact reproduction commands. No git commit was made. "
              f"All numerical tables and checkpoints remain under `{AUDIT}`. "
              "Top-level files include diagnosis_overview.csv, diagnosis_stage.csv, diagnosis_class.csv, diagnosis_class_stage.csv, "
              "component_summary.csv, feature_summary.csv, stack_differences.csv, cache_reproduction.csv, "
              "classification_folds.csv, classification_comparisons.csv, classification_summary.csv, "
              "benchmark.csv, benchmark_summary.csv, mechanism.csv, validation.json, and full_cost_estimate.csv. "
              "Each dataset subfolder holds selection.json, frame NPZ checkpoints, feature and component "
              "differences, costs.csv, and features.npz.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("wrote", HERE / "REPORT.md")
    print("cost", estimate.to_string(index=False))
    print("stack", stack_summary.to_string(index=False))
    print("features GD60 vs Newton", feature_summary[(feature_summary.method == "gd60") & (feature_summary.reference == "newton")].to_string(index=False))
    print("benchmark", bench_summary.to_string(index=False))


if __name__ == "__main__":
    main()
