"""Reproduce the solver audit without writing caches to the repository."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/Users/mikhail/Datasets/imf-classification/solver_audit/numba")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pimf
from scipy import stats

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emdirmf import decompose as dec
from emdirmf.features import FEATURE_NAMES, frame_features
from emdirmf.stats import holm, paired_comparison
from newton import newton_components

DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
AUDIT = DATA_ROOT / "solver_audit"
TAG = "irmf_h2_a1.41"
DATASETS = ("dronerf", "birdclef2023", "esc50_animals", "synthetic")
SEED = 20261020


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()))


def quantiles(values, prefix=""):
    values = np.asarray(values)
    return {f"{prefix}{key}": float(value) for key, value in zip(
        ("min", "q05", "q50", "q95", "q99", "max"),
        np.quantile(values[np.isfinite(values)], [0, .05, .5, .95, .99, 1]),
    )}


def diagnose():
    """Only existing caches are read; no solver is called here."""
    AUDIT.mkdir(parents=True, exist_ok=True)
    stage_rows, class_rows, class_stage_rows, overview = [], [], [], []
    for name in DATASETS:
        root = DATA_ROOT / name
        summary = json.loads((root / "decomp" / f"{TAG}.json").read_text())
        dataset = json.loads((root / "prepared/dataset.json").read_text())
        meta = pd.read_parquet(root / "decomp" / f"{TAG}.meta.parquet")
        labels = pd.read_parquet(root / "prepared/meta.parquet")
        meta = meta[meta.ok].reset_index(drop=True)
        conv = np.stack(meta.stage_converged)
        iters = np.stack(meta.stage_iterations)
        delta = np.stack(meta.stage_final_delta)
        comps = np.load(root / "decomp" / f"{TAG}.npy", mmap_mode="r")
        # x in the stopping rule is the fitted component, not the input frame.
        max_abs = np.empty_like(delta)
        for start in range(0, len(meta), 64):
            chunk = meta.iloc[start:start + 64]
            values = comps[chunk.frame_id.to_numpy(), chunk.channel.to_numpy(), :-1]
            max_abs[start:start + len(chunk)] = np.max(np.abs(values), axis=-1)
        ratio = delta / (1e-6 * (1 + max_abs))
        assert np.array_equal(conv, ratio <= 1), "float32 cache changed stop classification"
        for slot, window in enumerate(summary["slot_window_sizes"]):
            row = {"dataset": name, "window": window, "n_channels": len(meta),
                   "unconverged_pct": 100 * (~conv[:, slot]).mean(),
                   "iterations_mean": iters[:, slot].mean()}
            row.update(quantiles(iters[:, slot], "iterations_"))
            row.update(quantiles(ratio[:, slot], "delta_ratio_"))
            stage_rows.append(row)
        for task, column in dataset["tasks"].items():
            class_labels = labels.set_index("frame_id").loc[meta.frame_id, column].astype(str).to_numpy()
            for label in sorted(set(class_labels)):
                mask = class_labels == label
                class_rows.append({"dataset": name, "task": task, "class": label,
                                   "n_channels": int(mask.sum()),
                                   "unconverged_stage_pct": 100 * (~conv[mask]).mean(),
                                   "any_unconverged_channel_pct": 100 * (~conv[mask]).any(axis=1).mean(),
                                   "H_median": float(meta.H[mask].median())})
                for slot, window in enumerate(summary["slot_window_sizes"]):
                    class_stage_rows.append({"dataset": name, "task": task, "class": label,
                        "window": window, "n_channels": int(mask.sum()),
                        "unconverged_pct": 100 * (~conv[mask, slot]).mean(),
                        "iterations_median": np.median(iters[mask, slot]),
                        "delta_ratio_median": np.median(ratio[mask, slot]),
                        "delta_ratio_p95": np.quantile(ratio[mask, slot], .95)})
        timing = meta.groupby("frame_id")[["seconds", "cpu_seconds"]].sum()
        row = {"dataset": name, "n_frames": len(timing), "n_channels": len(meta),
               "unconverged_stage_pct": 100 * (~conv).mean(),
               "any_unconverged_channel_pct": 100 * (~conv).any(axis=1).mean(),
               "any_unconverged_frame_pct": 100 * meta.assign(bad=(~conv).any(axis=1)).groupby("frame_id").bad.any().mean(),
               "wall_s_per_frame_mean": float(timing.seconds.mean()),
               "cpu_s_per_frame_mean": float(timing.cpu_seconds.mean()),
               "H_median": float(meta.H.median())}
        row.update(quantiles(timing.seconds, "wall_seconds_"))
        row.update(quantiles(iters.ravel(), "iterations_"))
        row.update(quantiles(ratio.ravel(), "delta_ratio_"))
        if (~conv).any():
            row.update(quantiles(ratio[~conv], "failed_delta_ratio_"))
        overview.append(row)
        print("diagnosis", name, row, flush=True)
    pd.DataFrame(stage_rows).to_csv(AUDIT / "diagnosis_stage.csv", index=False)
    pd.DataFrame(class_rows).to_csv(AUDIT / "diagnosis_class.csv", index=False)
    pd.DataFrame(class_stage_rows).to_csv(AUDIT / "diagnosis_class_stage.csv", index=False)
    pd.DataFrame(overview).to_csv(AUDIT / "diagnosis_overview.csv", index=False)


def select_subset(name, per_class=10):
    """Equal class quotas, round-robin over shuffled groups then frames."""
    root = DATA_ROOT / name
    dataset = json.loads((root / "prepared/dataset.json").read_text())
    meta = pd.read_parquet(root / "prepared/meta.parquet")
    column = dataset["tasks"][dataset["primary_task"]]
    rng = np.random.default_rng(SEED)
    ids = []
    for _, cell in meta.groupby(column, sort=True):
        grouped = [rng.permutation(g.frame_id.to_numpy()).tolist() for _, g in cell.groupby("group", sort=True)]
        grouped = [grouped[i] for i in rng.permutation(len(grouped))]
        chosen = []
        while len(chosen) < per_class:
            for group in grouped:
                if group:
                    chosen.append(group.pop())
                    if len(chosen) == per_class:
                        break
        ids.extend(chosen)
    return np.array(sorted(ids)), meta, dataset


def sensitivity(names, per_class=10):
    for name in names:
        out = AUDIT / name
        out.mkdir(parents=True, exist_ok=True)
        ids, meta, dataset = select_subset(name, per_class)
        save_json(out / "selection.json", {"seed": SEED, "per_class": per_class,
                  "frame_ids": ids.tolist(), "channel": 0,
                  "task": dataset["primary_task"]})
        frames = np.load(DATA_ROOT / name / "prepared/frames.npy", mmap_mode="r")
        # Exclude JIT compilation from measured prototype time.
        newton_components(np.arange(32, dtype=float))
        for index, frame_id in enumerate(ids):
            path = out / f"frame_{frame_id}.npz"
            if path.exists():
                continue
            y, log_scale = dec.normalize_frame(frames[frame_id, 0])
            arrays = {"y": y, "log_scale": np.array(log_scale)}
            infos = {}
            for max_iter in (60, 300, 1000):
                started, cpu = time.perf_counter(), time.process_time()
                components, info = dec.irmf_components(y, max_iter=max_iter)
                info.update(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
                arrays[f"gd{max_iter}"] = components
                infos[f"gd{max_iter}"] = info
            started, cpu = time.perf_counter(), time.process_time()
            components, info = newton_components(y)
            info.update(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
            arrays["newton"] = components
            infos["newton"] = info
            arrays["infos_json"] = np.array(json.dumps(infos))
            np.savez_compressed(path, **arrays)
            print(name, index + 1, "/", len(ids), "frame", frame_id,
                  "seconds", {k: round(v["seconds"], 3) for k, v in infos.items()}, flush=True)
        summarize_sensitivity(name)


def summarize_sensitivity(name):
    out = AUDIT / name
    selection = json.loads((out / "selection.json").read_text())
    ids = selection["frame_ids"]
    all_features = {method: [] for method in ("gd60", "gd300", "gd1000", "newton")}
    differences, costs = [], []
    scales = []
    for frame_id in ids:
        with np.load(out / f"frame_{frame_id}.npz") as data:
            infos = json.loads(str(data["infos_json"]))
            scales.append(float(data["log_scale"]))
            for method in all_features:
                all_features[method].append(frame_features(data[method][None, None])[0])
                info = infos[method]
                costs.append({"dataset": name, "frame_id": frame_id, "method": method,
                              "seconds": info["seconds"], "cpu_seconds": info["cpu_seconds"],
                              "unconverged_stages": sum(not v for v in info.get("stage_converged", []))})
            for reference in ("gd1000", "newton"):
                for method in ("gd60", "gd300", "gd1000"):
                    for slot, (a, b) in enumerate(zip(data[method], data[reference])):
                        differences.append({"dataset": name, "frame_id": frame_id,
                            "method": method, "reference": reference, "slot": slot,
                            "rel_l2": np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12),
                            "input_relative_l2": np.linalg.norm(a - b) / np.linalg.norm(data["y"])})
    pd.DataFrame(differences).to_csv(out / "component_differences.csv", index=False)
    pd.DataFrame(costs).to_csv(out / "costs.csv", index=False)
    all_features = {key: np.array(values) for key, values in all_features.items()}
    np.savez_compressed(out / "features.npz", **all_features, log_scale=scales, frame_ids=ids)
    feature_rows = []
    for reference in ("gd1000", "newton"):
        ref = all_features[reference]
        std = ref.std(axis=0)
        for method in ("gd60", "gd300", "gd1000"):
            values = all_features[method]
            for column in range(ref.shape[1]):
                valid = std[column] > 1e-12
                rho = stats.spearmanr(values[:, column], ref[:, column]).statistic if valid and np.ptp(values[:, column]) > 0 else np.nan
                feature_rows.append({"dataset": name, "method": method, "reference": reference,
                     "slot": column // len(FEATURE_NAMES), "feature": FEATURE_NAMES[column % len(FEATURE_NAMES)],
                     "std_ref": std[column], "spearman": rho,
                     "nrmse": np.sqrt(np.mean((values[:, column] - ref[:, column]) ** 2)) / std[column] if valid else np.nan,
                     "mae": np.mean(np.abs(values[:, column] - ref[:, column])),
                     "max_abs": np.max(np.abs(values[:, column] - ref[:, column]))})
    pd.DataFrame(feature_rows).to_csv(out / "feature_differences.csv", index=False)


def classify(names):
    """Small-sample, fixed LDA check on intersections of the saved study folds."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from emdirmf.cv import split_metrics

    rows = []
    for name in names:
        out = AUDIT / name
        data = np.load(out / "features.npz")
        ids = data["frame_ids"]
        meta = pd.read_parquet(DATA_ROOT / name / "prepared/meta.parquet").iloc[ids]
        # DroneRF's only saved split is binary. Preserve it and evaluate binary.
        task = "2class" if name == "dronerf" else ("species" if name == "birdclef2023" else "animal")
        saved = json.loads((DATA_ROOT / name / f"splits_{task}.json").read_text())
        y = meta[f"y_{task}"].astype(str).to_numpy()
        groups = meta.group.astype(str).to_numpy()
        for record in saved["splits"]:
            test = np.flatnonzero(np.isin(ids, record["test"]))
            train = np.setdiff1d(np.arange(len(ids)), test)
            assert not (set(groups[train]) & set(groups[test]))
            for method in ("gd60", "gd300", "gd1000", "newton"):
                for include_scale in (False, True):
                    x = data[method]
                    if include_scale:
                        x = np.column_stack([x, data["log_scale"]])
                    model = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(),
                        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x[train], y[train])
                        pred = model.predict(x[test])
                    row = {"dataset": name, "task": task, "method": method,
                           "feature_set": "comp+log_scale" if include_scale else "comp",
                           "split_id": record["split_id"], "n_train": len(train), "n_test": len(test),
                           "n_test_classes": len(set(y[test]))}
                    row.update(split_metrics(y[test], pred, sorted(set(y))))
                    rows.append(row)
        pd.DataFrame([r for r in rows if r["dataset"] == name]).to_csv(out / "classification_folds.csv", index=False)
    results = pd.DataFrame(rows)
    comparisons = []
    for (name, feature_set), cell in results.groupby(["dataset", "feature_set"]):
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            pairs = (("gd60", "gd1000"), ("gd300", "gd1000"),
                     ("gd60", "newton"), ("gd300", "newton"), ("gd1000", "newton"))
            for method, reference_method in pairs:
                reference = cell[cell.method == reference_method].sort_values("split_id")
                base = cell[cell.method == method].sort_values("split_id")
                row = {"dataset": name, "feature_set": feature_set, "metric": metric,
                       "method_a": method, "method_b": reference_method}
                row.update(paired_comparison(base[metric], reference[metric], base.n_train, base.n_test))
                comparisons.append(row)
    comparisons = pd.DataFrame(comparisons)
    # One table per metric: 3 datasets x 5 contrasts x 2 scale options = 30 rows.
    for metric in comparisons.metric.unique():
        mask = comparisons.metric == metric
        comparisons.loc[mask, "p_nb_holm"] = holm(comparisons.loc[mask, "p_nb"])
        comparisons.loc[mask, "p_wilcoxon_holm"] = holm(comparisons.loc[mask, "p_wilcoxon"])
    results.to_csv(AUDIT / "classification_folds.csv", index=False)
    comparisons.to_csv(AUDIT / "classification_comparisons.csv", index=False)
    results.groupby(["dataset", "task", "feature_set", "method"])[
        ["balanced_accuracy", "macro_f1", "accuracy"]].agg(["mean", "std"]).to_csv(AUDIT / "classification_summary.csv")


def tight_gd(y):
    h = 2 * dec.noise_sigma_hat(y)[0]
    result = pimf.IMF(pimf.SmoothAbs(h, lookup=True), max_iter=20000, tol=1e-12, workers=1).decompose(
        y, h1=dec.H1, a=dec.SQRT2, k_max=None, h_min=dec.imf_h_min(len(y)),
    )
    return np.vstack([result.imfs[::-1], result.residual[None]]), {
        "unconverged": sum(not stage.converged for stage in result.stages),
        "max_iterations": max(stage.iterations for stage in result.stages),
    }


def benchmark():
    """Noisy/bursty signals at all lengths, plus one bimodal stress case."""
    AUDIT.mkdir(parents=True, exist_ok=True)
    newton_components(np.arange(32, dtype=float))
    rows = []
    for length in (1024, 2048, 4096):
        rng = np.random.default_rng(SEED + length)
        t = np.arange(length)
        signals = {
            "noisy": np.sin(2 * np.pi * .07 * t) + rng.normal(0, .7, length),
            "bursty": (1 + 8 * np.exp(-.5 * ((t - .45 * length) / (.06 * length)) ** 2)) * np.sin(2 * np.pi * .018 * t) + rng.normal(0, .08, length),
        }
        if length == 1024:
            signals["bimodal"] = 3 * np.sign(np.sin(2 * np.pi * t / (.31 * length))) + rng.normal(0, .12, length)
        for kind, signal in signals.items():
            checkpoint = AUDIT / f"benchmark_{length}_{kind}.json"
            if checkpoint.exists():
                rows.extend(json.loads(checkpoint.read_text()))
                continue
            y, _ = dec.normalize_frame(signal)
            values, records = {}, []
            for method in ("gd60", "tight_gd", "newton"):
                wall, cpu = time.perf_counter(), time.process_time()
                if method == "gd60":
                    components, info = dec.irmf_components(y)
                    info = {"unconverged": sum(not v for v in info["stage_converged"])}
                elif method == "tight_gd":
                    components, info = tight_gd(y)
                else:
                    components, info = newton_components(y)
                    info = {"unconverged": sum(stage["failed_rows"] for stage in info["stages"]),
                            "max_iterations": max(stage["iterations_max"] for stage in info["stages"])}
                record = {"length": length, "signal": kind, "method": method,
                          "seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                          "reconstruction_max_abs": float(np.max(np.abs(components.sum(axis=0) - y))), **info}
                values[method] = components
                records.append(record)
                print("benchmark", record, flush=True)
            reference = values["tight_gd"]
            for record in records:
                delta = values[record["method"]] - reference
                record["relative_l2_vs_tight"] = float(np.linalg.norm(delta) / np.linalg.norm(reference))
                record["max_component_relative_l2_vs_tight"] = float(np.max(np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-12)))
            save_json(checkpoint, records)
            rows.extend(records)
    pd.DataFrame(rows).to_csv(AUDIT / "benchmark.csv", index=False)


def validate():
    """Independent brentq roots and a reproducible flat-score example."""
    from scipy.optimize import brentq
    from newton import NewtonSmoothAbs
    from pimf.decompose import _gd_fit_windows
    from pimf.kernels import squared_triangle

    rng = np.random.default_rng(SEED)
    records = []
    for width in (5, 7, 65, 1025):
        weights = squared_triangle.weights(width)
        for h in (.01, 1.0, 10.0):
            windows = rng.normal(size=(20, width))
            windows[0] = 4
            windows[1, width // 2] = 100
            contrast = NewtonSmoothAbs(h)
            roots = contrast.solve(windows, weights)
            expected = []
            for row in windows:
                lo, hi = row[weights > 0].min(), row[weights > 0].max()
                root = lo if lo == hi else brentq(
                    lambda x: np.dot(weights, contrast.psi(row - x)), lo, hi,
                    xtol=1e-12, rtol=1e-14,
                )
                expected.append(root)
            error = np.max(np.abs(roots - expected))
            assert error < 1e-8, (width, h, error)
            records.append({"width": width, "H": h, "n_rows": len(windows),
                            "max_abs_error_vs_brentq": error})
    # The study's noninteger radius gives a center weight close to one half.
    radius = 2 * np.sqrt(2)
    weights = squared_triangle.weights(7, bandwidth=radius)
    windows = np.zeros((1, 7))
    windows[0, 3] = 100
    contrast = NewtonSmoothAbs(1)
    exact = contrast.solve(windows, weights)
    toy = {"weights": weights.tolist(), "center_weight": weights[3], "root": exact[0]}
    for n in (60, 300, 1000, 20000):
        fitted, iterations, delta = _gd_fit_windows(windows, weights, contrast, n, 1e-12)
        toy[f"gd{n}"] = {"x": fitted[0], "iterations": iterations,
                          "delta": delta, "abs_error": abs(fitted[0] - exact[0])}
    save_json(AUDIT / "validation.json", {"roots": records, "flat_score_toy": toy})
    print("validation", json.dumps(toy), flush=True)


def mechanism():
    """Compare default-stage starts to roots on exactly the same stage inputs."""
    from numpy.lib.stride_tricks import sliding_window_view
    from pimf.kernels import squared_triangle
    from newton import NewtonSmoothAbs

    rows = []
    for name in DATASETS[:3]:
        out = AUDIT / name
        selection = json.loads((out / "selection.json").read_text())
        # Select five hardest subset frames by number of capped stages.
        candidates = []
        for frame_id in selection["frame_ids"]:
            with np.load(out / f"frame_{frame_id}.npz") as data:
                info = json.loads(str(data["infos_json"]))["gd60"]
                candidates.append((sum(not c for c in info["stage_converged"]), frame_id))
        for _, frame_id in sorted(candidates, reverse=True)[:5]:
            with np.load(out / f"frame_{frame_id}.npz") as data:
                info = json.loads(str(data["infos_json"]))["gd60"]
                residual = data["y"].copy()
                windows, bandwidths = dec.imf_schedule(len(residual))
                for slot in range(len(windows) - 1, -1, -1):
                    stage = len(windows) - 1 - slot
                    size, h = windows[stage], bandwidths[stage]
                    weights = squared_triangle.weights(size, bandwidth=h * len(residual))
                    view = sliding_window_view(np.pad(residual, size // 2, mode="wrap"), size)
                    contrast = NewtonSmoothAbs(info["H"])
                    roots = contrast.solve(view, weights)
                    median = np.median(view, axis=1)
                    start_distance = np.abs(roots - median) / info["H"]
                    # Lookup score derivative, exactly as in the prototype.
                    from pimf.contrasts import _SCORE_DELTA
                    standardized = (view - roots[:, None]) / info["H"]
                    positions = np.clip(standardized, -8, 8)
                    indices = np.minimum(((positions + 8) * 256).astype(int), 4096)
                    curvature_ratio = np.sum(weights[None] * _SCORE_DELTA[indices] * 256 *
                                             (np.abs(standardized) < 8), axis=1) / np.sqrt(2 / np.pi)
                    stage_fit = data["gd60"][slot]
                    stage_delta = contrast.psi(view - stage_fit[:, None]) @ weights
                    threshold = 1e-6 * (1 + np.max(np.abs(stage_fit)))
                    active = np.abs(.95 * info["H"] / np.sqrt(2 / np.pi) * stage_delta) > threshold
                    rows.append({"dataset": name, "frame_id": frame_id, "window": size,
                                 "default_converged": info["stage_converged"][slot],
                                 "H": info["H"], "initial_distance_H_max": start_distance.max(),
                                 "move_cap_lower_bound_iterations": start_distance.max() / (.95 / np.sqrt(2 / np.pi)),
                                 "curvature_ratio_min": curvature_ratio.min(),
                                 "curvature_ratio_q05": np.quantile(curvature_ratio, .05),
                                 "still_active_row_pct": 100 * active.mean(),
                                 "stage_rel_l2_vs_same_input_root": np.linalg.norm(stage_fit - roots) / max(np.linalg.norm(roots), 1e-12)})
                    residual -= stage_fit
    pd.DataFrame(rows).to_csv(AUDIT / "mechanism.csv", index=False)


def second_channel():
    """Newton timing on spaced DroneRF H-band frames for the cost estimate."""
    name = "dronerf"
    out = AUDIT / name
    ids = json.loads((out / "selection.json").read_text())["frame_ids"]
    frames = np.load(DATA_ROOT / name / "prepared/frames.npy", mmap_mode="r")
    newton_components(np.arange(32, dtype=float))
    rows = []
    for frame_id in ids[::3]:
        y, _ = dec.normalize_frame(frames[frame_id, 1])
        wall, cpu = time.perf_counter(), time.process_time()
        components, info = newton_components(y)
        rows.append({"frame_id": frame_id, "channel": 1, "seconds": time.perf_counter() - wall,
                     "cpu_seconds": time.process_time() - cpu,
                     "max_iterations": max(s["iterations_max"] for s in info["stages"])})
    pd.DataFrame(rows).to_csv(out / "second_channel_costs.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "sensitivity", "summarize", "classify", "benchmark", "validate", "mechanism", "second_channel"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS[:3]))
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "diagnose":
        diagnose()
    elif args.mode == "sensitivity":
        sensitivity(args.datasets, args.per_class)
    elif args.mode == "summarize":
        for name in args.datasets:
            summarize_sensitivity(name)
    elif args.mode == "classify":
        classify(args.datasets)
    elif args.mode == "validate":
        validate()
    elif args.mode == "mechanism":
        mechanism()
    elif args.mode == "second_channel":
        second_channel()
    else:
        benchmark()


if __name__ == "__main__":
    main()
