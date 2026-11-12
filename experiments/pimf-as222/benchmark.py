"""Reproduce the supplied five-sine comparison with the installed pimf package.

Run from the repository root with .venv/bin/python experiments/pimf-as222/benchmark.py.
The FFT comparator is the supplied report's AS222-style adaptation, not a
translation of historical RESSMO. Independent roots only validate pimf output.
"""

import os

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pimf import Epanechnikov, IMF, Quadratic, SmoothAbs
from scipy.optimize import brentq
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/pimf-as222"
FREQUENCIES = np.array([1, 3, 8, 20, 50])
AMPLITUDES = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
BANDWIDTHS = (0.100, 0.0829, 0.0303, 0.0128, 0.00757)
SEED = 20261004
SIGMA = 0.2
H = 1.0
N = 1000
REPLICATES = 20
SIZES = (250, 500, 1000, 2000, 4000, 8000)
REPEATS = 5
CORRELATION_GATE = 0.995
RELATIVE_RMS_GATE = 0.05


class CachedEpanechnikov(Epanechnikov):
    """Cache the package's unmodified weights, as in the source timing protocol."""

    def __init__(self):
        self.cache = {}

    def weights(self, window_size, *, bandwidth=None):
        key = (window_size, bandwidth)
        if key not in self.cache:
            weights = super().weights(window_size, bandwidth=bandwidth)
            weights.setflags(write=False)
            self.cache[key] = weights
        return self.cache[key]


def true_components(n):
    t = np.arange(n) / n
    return AMPLITUDES[:, None] * np.sin(2 * np.pi * FREQUENCIES[:, None] * t)


def sampled_kernel(n, h):
    distance = np.minimum(np.arange(n), n - np.arange(n)) / n
    weights = 0.75 * np.maximum(0.0, 1.0 - (distance / h) ** 2)
    return weights / weights.sum()


def prepare(n, *, lookup=True, tol=1e-6, workers=1):
    kernel = CachedEpanechnikov()
    for h in BANDWIDTHS:
        radius = h * n
        if abs(radius - round(radius)) < 1e-9:
            radius = float(round(radius))
        kernel.weights(2 * int(np.ceil(radius)) + 1, bandwidth=h * n)
    model = IMF(
        contrast=SmoothAbs(H, lookup=lookup), kernel=kernel,
        boundary="wrap", max_iter=60, tol=tol, workers=workers,
    )
    spectra = [np.fft.rfft(sampled_kernel(n, h)) for h in BANDWIDTHS]
    return model, spectra


def package_decompose(y, model):
    """Use one public decompose call per prescribed, nongeometric bandwidth."""
    residual = y.copy()
    components, stages = [], []
    for h in BANDWIDTHS:
        result = model.decompose(residual, h1=h, k_max=1)
        components.append(result.imfs[0])
        stages.append(asdict(result.stages[0]))
        residual = result.residual
    return np.stack(components), residual, stages


def convolve(values, spectrum):
    return np.fft.irfft(np.fft.rfft(values) * spectrum, n=len(values))


def fft_stage(residual, spectrum):
    pilot = convolve(residual, spectrum)
    errors = residual - pilot
    numerator = convolve(erf(errors / (np.sqrt(2) * H)), spectrum)
    denominator = convolve(
        np.sqrt(2 / np.pi) / H * np.exp(-0.5 * (errors / H) ** 2), spectrum
    )
    if not np.all(np.isfinite(denominator)) or np.min(denominator) <= 0:
        raise ArithmeticError("Nonpositive or nonfinite one-step denominator")
    return pilot + numerator / denominator


def fft_decompose(y, spectra):
    residual = y.copy()
    components = []
    for spectrum in spectra:
        component = fft_stage(residual, spectrum)
        components.append(component)
        residual = residual - component
    return np.stack(components), residual, []


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def correlation(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def verify_output(y, output):
    components, residual, stages = output
    assert components.shape == (5, len(y))
    assert np.all(np.isfinite(components)) and np.all(np.isfinite(residual))
    error = float(np.max(np.abs(components.sum(axis=0) + residual - y)))
    assert error < 1e-12, error
    assert all(stage["converged"] for stage in stages), stages
    return error


def validate(y, model, spectra):
    """Check the objective, sampled kernels, FFT arithmetic and package settings."""
    components, residual, stages = package_decompose(y, model)
    checks = {"reconstruction_max_abs": verify_output(y, (components, residual, stages))}
    positions = [0, 73, 250, 511, 999]
    root_errors, score_errors, kernel_errors, fft_errors, root_rows = [], [], [], [], []
    current = y.copy()
    for k, (h, stage) in enumerate(zip(BANDWIDTHS, stages, strict=True)):
        size = stage["window_size"]
        weights = model.kernel.weights(size, bandwidth=h * len(y))
        radius = size // 2
        offsets = np.arange(-radius, radius + 1)
        q = sampled_kernel(len(y), h)
        embedded = np.zeros(len(y))
        embedded[offsets % len(y)] = weights
        kernel_errors.append(float(np.max(np.abs(q - embedded))))
        windows = sliding_window_view(np.pad(current, radius, mode="wrap"), size)
        score = np.sum(weights * erf((windows - components[k, :, None]) / np.sqrt(2)), axis=1)
        score_errors.append(float(np.max(np.abs(score))))
        for target in positions:
            values = windows[target]
            root = brentq(
                lambda x: np.dot(weights, erf((values - x) / (np.sqrt(2) * H))),
                float(values.min()), float(values.max()), xtol=1e-14, rtol=1e-14,
            )
            error = abs(root - components[k, target])
            root_errors.append(error)
            root_rows.append({"component": k + 1, "position": target, "pimf": float(components[k, target]), "exact_root": root, "absolute_error": error})

        def spatial(values):
            return np.sum(
                weights * sliding_window_view(np.pad(values, radius, mode="wrap"), size),
                axis=1,
            )

        pilot = spatial(current)
        errors = current - pilot
        expected = pilot + spatial(erf(errors / np.sqrt(2))) / spatial(np.sqrt(2 / np.pi) * np.exp(-errors ** 2 / 2))
        fft_errors.append(float(np.max(np.abs(expected - fft_stage(current, spectra[k])))))
        current = current - components[k]
    tight_model, _ = prepare(len(y), lookup=False, tol=1e-12)
    tight = package_decompose(y, tight_model)
    verify_output(y, tight)
    plain = IMF(contrast=SmoothAbs(H), kernel=Epanechnikov())
    uncached = package_decompose(y, plain)
    assert np.array_equal(uncached[0], components)
    linear = IMF(contrast=Quadratic(), kernel=Epanechnikov()).decompose(y, h1=BANDWIDTHS[0], k_max=1).imfs[0]
    ky = convolve(y, spectra[0])
    linear_one_step = ky + convolve(y - ky, spectra[0])
    checks.update({
        "root_checks": len(root_errors),
        "root_max_abs_error": float(max(root_errors)),
        "exact_score_max_abs": float(max(score_errors)),
        "kernel_max_abs_difference": float(max(kernel_errors)),
        "fft_spatial_max_abs_difference": float(max(fft_errors)),
        "default_vs_tight_direct_max_abs": float(np.max(np.abs(components - tight[0]))),
        "cached_vs_uncached_bit_identical": bool(np.array_equal(uncached[0], components)),
        "linear_pimf_vs_fft_max_abs": float(np.max(np.abs(linear - ky))),
        "linear_one_step_identity_max_abs": float(np.max(np.abs(linear_one_step - (2 * ky - convolve(ky, spectra[0]))))),
        "linear_direct_vs_one_step_rms": rms(linear - linear_one_step),
        "stages": stages,
    })
    assert checks["root_max_abs_error"] < 2e-6
    assert checks["exact_score_max_abs"] < 2e-6
    assert checks["kernel_max_abs_difference"] < 1e-14
    assert checks["fft_spatial_max_abs_difference"] < 1e-12
    assert checks["default_vs_tight_direct_max_abs"] < 2e-6
    assert checks["linear_pimf_vs_fft_max_abs"] < 1e-12
    assert checks["linear_one_step_identity_max_abs"] < 1e-12
    write_csv("root_validation.csv", root_rows)
    return checks


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def monte_carlo(rng, model, spectra, count, name="pilot"):
    truth = true_components(N)
    rows, stage_rows, package_curves, fft_curves, inputs = [], [], [], [], []
    totals = {"pimf": 0.0, "fft": 0.0}
    reconstruction = 0.0
    for replicate in range(count):
        y = truth.sum(axis=0) + rng.normal(0, SIGMA, N)
        tic = perf_counter()
        package = package_decompose(y, model)
        totals["pimf"] += perf_counter() - tic
        tic = perf_counter()
        fourier = fft_decompose(y, spectra)
        totals["fft"] += perf_counter() - tic
        reconstruction = max(reconstruction, verify_output(y, package), verify_output(y, fourier))
        package_curves.append(package[0])
        fft_curves.append(fourier[0])
        inputs.append(y)
        for k, (left, right, target) in enumerate(zip(package[0], fourier[0], truth, strict=True)):
            difference = rms(left - right)
            corr = correlation(left, right)
            relative = difference / rms(left)
            rows.append({
                "replicate": replicate + 1, "component": k + 1,
                "frequency": int(FREQUENCIES[k]), "pimf_fft_correlation": corr,
                "difference_rms": difference, "relative_difference": relative,
                "difference_max_abs": float(np.max(np.abs(left - right))),
                "pimf_true_correlation": correlation(left, target),
                "fft_true_correlation": correlation(right, target),
                "pimf_true_rmse": rms(left - target), "fft_true_rmse": rms(right - target),
                "agreement_pass": corr >= CORRELATION_GATE and relative <= RELATIVE_RMS_GATE,
            })
        for k, stage in enumerate(package[2]):
            stage_rows.append({"replicate": replicate + 1, "component": k + 1, **stage})
    metric_names = list(rows[0])[3:-1]
    summaries = []
    for k in range(5):
        subset = [row for row in rows if row["component"] == k + 1]
        summary = {"component": k + 1, "frequency": int(FREQUENCIES[k])}
        for metric in metric_names:
            values = np.array([row[metric] for row in subset])
            summary[metric] = float(values.mean())
            summary[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(count))
        summary["agreement_passes"] = sum(row["agreement_pass"] for row in subset)
        summaries.append(summary)
    write_csv(f"{name}_paired_metrics.csv", rows)
    write_csv(f"{name}_component_summary.csv", summaries)
    write_csv(f"{name}_solver_diagnostics.csv", stage_rows)
    np.savez_compressed(
        OUTPUT / f"{name}_curves.npz", t=np.arange(N) / N, truth=truth,
        inputs=np.asarray(inputs), pimf=np.asarray(package_curves), fft=np.asarray(fft_curves),
    )
    return {
        "replicates": count, "agreement_passed": all(row["agreement_pass"] for row in rows),
        "paired_component_passes": sum(row["agreement_pass"] for row in rows),
        "paired_component_count": len(rows), "reconstruction_max_abs": reconstruction,
        "smoothing_seconds": totals, "summary": summaries,
        "max_solver_iterations": max(row["iterations"] for row in stage_rows),
        "all_stages_converged": all(row["converged"] for row in stage_rows),
    }


def timing_benchmark():
    rng = np.random.default_rng(SEED + 1)
    order_rng = np.random.default_rng(SEED + 2)
    raw, summaries = [], []
    for n in SIZES:
        y = true_components(n).sum(axis=0) + rng.normal(0, SIGMA, n)
        model, spectra = prepare(n)
        runners = {"pimf": lambda: package_decompose(y, model), "fft": lambda: fft_decompose(y, spectra)}
        for run in runners.values():
            verify_output(y, run())
        rows = []
        for repeat in range(REPEATS):
            for method in order_rng.permutation(list(runners)):
                tic = perf_counter()
                output = runners[method]()
                seconds = perf_counter() - tic
                error = verify_output(y, output)
                rows.append({"n": n, "repeat": repeat + 1, "method": str(method), "seconds": seconds, "reconstruction_max_abs": error})
        raw.extend(rows)
        summary = {"n": n}
        for method in runners:
            values = [row["seconds"] for row in rows if row["method"] == method]
            summary[method + "_median_ms"] = 1000 * float(np.median(values))
            summary[method + "_min_ms"] = 1000 * min(values)
            summary[method + "_max_ms"] = 1000 * max(values)
        summary["speedup"] = summary["pimf_median_ms"] / summary["fft_median_ms"]
        summaries.append(summary)
        print(f"n={n}: pimf {summary['pimf_median_ms']:.3f} ms; FFT {summary['fft_median_ms']:.3f} ms", flush=True)
        write_csv("timing_raw.csv", raw)
        write_csv("timing_summary.csv", summaries)
    return summaries


def source_reconciliation(summaries):
    # Rounded values transcribed from page 8 of the supplied ten-page PDF.
    reported = {
        "pimf_fft_correlation": [0.9937, 0.8006, 0.9921, 0.9464, 0.9129],
        "difference_rms": [0.04001, 0.02896, 0.03105, 0.02741, 0.02330],
        "relative_difference": [0.168, 0.649, 0.270, 0.340, 0.492],
        "pimf_true_correlation": [0.8562, 0.9581, 0.9727, 0.8631, 0.6997],
        "fft_true_correlation": [0.7947, 0.6657, 0.9643, 0.8890, 0.7704],
    }
    rounding = {key: 5e-5 for key in reported}
    rounding.update(difference_rms=5e-6, relative_difference=5e-4)
    rows = []
    for k, summary in enumerate(summaries):
        for metric, values in reported.items():
            difference = summary[metric] - values[k]
            rows.append({"component": k + 1, "metric": metric, "source_rounded": values[k], "new_value": summary[metric], "difference": difference, "source_rounding_half_unit": rounding[metric], "within_rounding": abs(difference) <= rounding[metric]})
    write_csv("source_reconciliation.csv", rows)
    return {"all_25_entries_within_source_rounding": all(row["within_rounding"] for row in rows), "maximum_abs_metric_difference": max(abs(row["difference"]) for row in rows)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-timing", action="store_true", help="Run the numerical reproduction without timing measurements")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, spectra = prepare(N)
    rng = np.random.default_rng(SEED)
    validation_y = true_components(N).sum(axis=0) + rng.normal(0, SIGMA, N)
    checks = validate(validation_y, model, spectra)
    print("Validation passed:", json.dumps({key: value for key, value in checks.items() if key != "stages"}), flush=True)
    pilot = monte_carlo(rng, model, spectra, REPLICATES)
    print("Pilot agreement gate:", pilot["agreement_passed"], flush=True)
    extended = monte_carlo(rng, model, spectra, 1000, "extended") if pilot["agreement_passed"] else None
    timing = [] if args.skip_timing else timing_benchmark()
    import pimf
    sources = []
    paths = [
        Path('/Users/mikhail/Downloads/20261008 AS222_vs_IRMF_10_page_report.pdf'),
        Path('/Users/mikhail/Downloads/five_sine_IRMF_benchmark.pdf'),
        ROOT / 'research/pimf-as222/references/hardle_1987_as222.pdf',
        ROOT / 'research/pimf-as222/references/dusoir_1988_as_r73.pdf',
    ]
    for path in paths:
        if path.exists():
            sources.append({"name": path.name, "sha256": sha256(path)})
    package_dir = Path(pimf.__file__).parent
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does pip-installed pimf preserve the supplied fixed-bandwidth comparison with the AS222-style adaptation?",
        "settings": {"n": N, "replicates": REPLICATES, "seed": SEED, "rng": "numpy.random.default_rng / PCG64", "validation_draws_before_pilot": 1, "sigma": SIGMA, "H": H, "frequencies": FREQUENCIES.tolist(), "amplitudes": AMPLITUDES.tolist(), "bandwidths": BANDWIDTHS, "kernel": "classical Epanechnikov", "boundary": "wrap", "lookup": True, "workers": 1, "max_iter": 60, "tol": 1e-6},
        "gate": {"minimum_correlation": CORRELATION_GATE, "maximum_relative_rms": RELATIVE_RMS_GATE, "requirement": "all 20 x 5 individual pairs"},
        "validation": checks, "pilot": pilot,
        "extended": extended, "timing": timing,
        "source_reconciliation": source_reconciliation(pilot["summary"]),
        "timing_protocol": {"repeats": REPEATS, "warmups_per_method_per_size": 1, "order": "randomized method order within each repetition", "kernel_setup": "excluded for both methods", "pimf_weights": "cached unmodified Epanechnikov.weights outputs", "includes": "five public one-stage decompose calls or five FFT stages, and their residual updates", "excludes": "imports, simulation, lookup-table and kernel setup, output verification, metrics, plotting", "input_rng_seed": SEED + 1, "order_rng_seed": SEED + 2},
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if sys.platform == "darwin" else platform.processor(), "packages": {name: importlib.metadata.version(name) for name in ['pimf', 'numpy', 'scipy', 'matplotlib']}, "thread_limits": {name: os.environ[name] for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']}},
        "provenance": {"benchmark_sha256": sha256(Path(__file__)), "pimf_source_hashes": {p.name: sha256(p) for p in sorted(package_dir.glob('*.py'))}, "source_documents": sources, "source_pdf_code_and_arrays_available": False},
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Saved", OUTPUT / "results.json", flush=True)


if __name__ == "__main__":
    main()
