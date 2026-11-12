"""Compare one pimf worker with all detected CPU cores, preserving the first run."""

import benchmark as base  # Set native-library thread limits before importing NumPy.

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pimf import Epanechnikov, IMF, SmoothAbs
from scipy.optimize import brentq
from scipy.special import erf


def validate_parallel(y, model, workers):
    output = base.package_decompose(y, model)
    reconstruction = base.verify_output(y, output)
    plain = IMF(contrast=SmoothAbs(base.H), kernel=Epanechnikov(), workers=workers)
    uncached = base.package_decompose(y, plain)
    base.verify_output(y, uncached)
    assert np.array_equal(output[0], uncached[0])
    tight_model, _ = base.prepare(len(y), lookup=False, tol=1e-12)
    tight = base.package_decompose(y, tight_model)
    base.verify_output(y, tight)
    tight_error = float(np.max(np.abs(output[0] - tight[0])))
    roots, scores = [], []
    residual = y.copy()
    for k, (h, stage) in enumerate(zip(base.BANDWIDTHS, output[2], strict=True)):
        size = stage["window_size"]
        weights = model.kernel.weights(size, bandwidth=h * len(y))
        windows = sliding_window_view(np.pad(residual, size // 2, mode="wrap"), size)
        score = np.sum(weights * erf((windows - output[0][k, :, None]) / np.sqrt(2)), axis=1)
        scores.append(float(np.max(np.abs(score))))
        for position in [0, 73, 250, 511, 999]:
            values = windows[position]
            root = brentq(
                lambda x: np.dot(weights, erf((values - x) / np.sqrt(2))),
                float(values.min()), float(values.max()), xtol=1e-14, rtol=1e-14,
            )
            roots.append({"component": k + 1, "position": position,
                          "pimf": float(output[0][k, position]), "exact_root": root,
                          "absolute_error": abs(root - output[0][k, position])})
        residual -= output[0][k]
    root_error = float(max(row["absolute_error"] for row in roots))
    assert max(root_error, max(scores), tight_error) < 2e-6
    base.write_csv("parallel_root_validation.csv", roots)
    return {"reconstruction_max_abs": reconstruction, "root_checks": len(roots),
            "root_max_abs_error": root_error, "exact_score_max_abs": max(scores),
            "default_vs_tight_direct_max_abs": tight_error,
            "cached_vs_uncached_bit_identical": True}


def timing_benchmark(workers):
    rng = np.random.default_rng(base.SEED + 1)
    order_rng = np.random.default_rng(base.SEED + 2)
    raw, summaries = [], []
    for n in base.SIZES:
        y = base.true_components(n).sum(axis=0) + rng.normal(0, base.SIGMA, n)
        serial, spectra = base.prepare(n)
        parallel, _ = base.prepare(n, workers=workers)
        runners = {
            "pimf_1": lambda: base.package_decompose(y, serial),
            "pimf_max": lambda: base.package_decompose(y, parallel),
            "fft": lambda: base.fft_decompose(y, spectra),
        }
        warmups = {name: run() for name, run in runners.items()}
        for output in warmups.values():
            base.verify_output(y, output)
        difference = float(np.max(np.abs(warmups["pimf_1"][0] - warmups["pimf_max"][0])))
        assert difference < 2e-6, difference
        rows = []
        for repeat in range(base.REPEATS):
            for method in order_rng.permutation(list(runners)):
                tic = perf_counter()
                output = runners[method]()
                seconds = perf_counter() - tic
                reconstruction = base.verify_output(y, output)
                rows.append({"n": n, "repeat": repeat + 1, "method": str(method),
                             "seconds": seconds, "reconstruction_max_abs": reconstruction,
                             "all_stages_converged": all(stage["converged"] for stage in output[2])})
        raw.extend(rows)
        # pimf 0.4.0 uses at least 64 observations per thread chunk.
        chunk_size = max(64, (n + workers - 1) // workers)
        summary = {"n": n, "requested_workers": workers,
                   "thread_pool_capacity": (n + chunk_size - 1) // chunk_size,
                   "serial_parallel_max_abs": difference}
        for method in runners:
            times = [row["seconds"] for row in rows if row["method"] == method]
            summary[method + "_median_ms"] = 1000 * float(np.median(times))
            summary[method + "_min_ms"] = 1000 * min(times)
            summary[method + "_max_ms"] = 1000 * max(times)
        summary["parallel_speedup"] = summary["pimf_1_median_ms"] / summary["pimf_max_median_ms"]
        summary["parallel_to_fft_ratio"] = summary["pimf_max_median_ms"] / summary["fft_median_ms"]
        summaries.append(summary)
        print(f"n={n}: 1 worker {summary['pimf_1_median_ms']:.3f} ms; "
              f"{workers} workers {summary['pimf_max_median_ms']:.3f} ms; "
              f"FFT {summary['fft_median_ms']:.3f} ms; "
              f"thread speedup {summary['parallel_speedup']:.2f}x", flush=True)
        base.write_csv("parallel_timing_raw.csv", raw)
        base.write_csv("parallel_timing_summary.csv", summaries)
    return summaries


def main():
    workers = os.cpu_count() or 1
    original = json.loads((base.OUTPUT / "results.json").read_text())
    assert original["settings"]["workers"] == 1
    print(f"Detected {workers} CPU cores; native numerical-library threads remain 1", flush=True)
    model, spectra = base.prepare(base.N, workers=workers)
    rng = np.random.default_rng(base.SEED)
    validation_y = base.true_components(base.N).sum(axis=0) + rng.normal(0, base.SIGMA, base.N)
    checks = validate_parallel(validation_y, model, workers)
    pilot = base.monte_carlo(rng, model, spectra, base.REPLICATES, name="parallel")
    with np.load(base.OUTPUT / "pilot_curves.npz") as serial, np.load(base.OUTPUT / "parallel_curves.npz") as parallel:
        assert np.array_equal(serial["inputs"], parallel["inputs"])
        assert np.array_equal(serial["fft"], parallel["fft"])
        difference = float(np.max(np.abs(serial["pimf"] - parallel["pimf"])))
    assert difference < 2e-6
    with (base.OUTPUT / "source_reconciliation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    reconciliation = []
    for row in rows:
        value = pilot["summary"][int(row["component"]) - 1][row["metric"]]
        delta = value - float(row["source_rounded"])
        reconciliation.append({**row, "new_value": value, "difference": delta,
                               "within_rounding": abs(delta) <= float(row["source_rounding_half_unit"])})
    base.write_csv("parallel_source_reconciliation.csv", reconciliation)
    checks.update({"serial_parallel_max_abs_all_20_runs": difference,
                   "same_inputs_and_fft_bit_identical": True,
                   "all_25_entries_within_source_rounding": all(row["within_rounding"] for row in reconciliation)})
    timing = timing_benchmark(workers)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {**original["settings"], "workers": workers},
        "validation": checks, "pilot": pilot, "timing": timing,
        "timing_protocol": {**original["timing_protocol"],
                            "methods": ["pimf_1", "pimf_max", "fft"],
                            "serial_and_fft_retimed_in_same_run": True,
                            "thread_pool_setup_included": True,
                            "chunking": "pimf 0.4.0 minimum 64 observations per chunk"},
        "environment": original["environment"],
        "provenance": {"parallel_benchmark_sha256": base.sha256(Path(__file__)),
                       "benchmark_sha256": base.sha256(Path(base.__file__)),
                       "original_results_sha256": base.sha256(base.OUTPUT / "results.json"),
                       "serial_curves_sha256": base.sha256(base.OUTPUT / "pilot_curves.npz")},
    }
    (base.OUTPUT / "parallel_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Parallel validation:", json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
