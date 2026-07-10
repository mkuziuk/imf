#!/usr/bin/env python3
"""Non-destructive diagnostics for the first recursive IMF error.

Run from the repository root with:

    .venv/bin/python research/first-imf-recursive-error/diagnostics/run_numerical_diagnosis.py

The script only writes CSV/JSON artifacts beside itself.  It reproduces the
current notebook definitions rather than importing or editing the notebook.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


HERE = Path(__file__).resolve().parent
N = 1000
SIGMA = 0.4
H_ROBUST = 2.0 * SIGMA
A = np.sqrt(2.0)
WINDOW_SIZES = [501, 355, 251, 177, 125, 89, 63, 45, 31]
LINEAR_MC_REPS = 5000
BOUNDARY_MC_REPS = 400
ROBUST_MC_REPS = 120

SQRT_2 = np.sqrt(2.0)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
TARGET_SIGNAL_STD = 0.38545431761087123


def _python_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(filename: str, rows: list[dict]):
    if not rows:
        raise ValueError(f"no rows for {filename}")
    keys = list(rows[0])
    with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _python_value(row[key]) for key in keys})


def odd_ceiling(value):
    size = int(np.ceil(value))
    if size % 2 == 0:
        size += 1
    return max(1, size)


def nearest_odd(value):
    rounded = int(np.round(value))
    if rounded % 2 == 1:
        return max(1, rounded)
    lower = max(1, rounded - 1)
    upper = rounded + 1
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def epanechnikov_weights(window_size):
    """Exact weights used in the notebook (the name is retained verbatim)."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    radius = window_size // 2
    if radius == 0:
        return np.array([1.0])
    offsets = np.arange(-radius, radius + 1)
    u = offsets / radius
    weights = 0.75 * np.maximum(0.0, 1.0 - np.abs(u)) ** 2
    return weights / weights.sum()


def local_linear_filter(y, window_size, boundary="wrap"):
    y = np.asarray(y, dtype=float)
    radius = window_size // 2
    weights = epanechnikov_weights(window_size)
    y_padded = np.pad(y, pad_width=radius, mode=boundary)
    windows = sliding_window_view(y_padded, window_size)
    return windows @ weights


def linear_imf(y, window_sizes=WINDOW_SIZES, boundary="wrap"):
    residual = np.asarray(y, dtype=float).copy()
    imfs = []
    for window_size in window_sizes:
        component = local_linear_filter(residual, window_size, boundary=boundary)
        imfs.append(component)
        residual = residual - component
    return np.asarray(imfs), residual


def rmse(values, axis=None):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values**2, axis=axis))


def circular_kernel(window_size, n=N):
    radius = window_size // 2
    kernel = np.zeros(n, dtype=float)
    for offset, weight in zip(
        np.arange(-radius, radius + 1), epanechnikov_weights(window_size)
    ):
        kernel[offset % n] += weight
    return kernel


def linear_transfer_functions(n=N, window_sizes=WINDOW_SIZES):
    smoother = [np.fft.fft(circular_kernel(size, n=n)) for size in window_sizes]
    residual_transfer = np.ones(n, dtype=complex)
    recursive = []
    for response in smoother:
        recursive.append(response * residual_transfer)
        residual_transfer = residual_transfer * (1.0 - response)
    return np.asarray(smoother), np.asarray(recursive), residual_transfer


def exact_linear_rows():
    smoother, recursive, _ = linear_transfer_functions()
    rows = []
    residual_product = np.ones(N, dtype=complex)
    for stage, (window_size, smooth_response, component_response) in enumerate(
        zip(WINDOW_SIZES, smoother, recursive), start=1
    ):
        single_impulse = np.fft.ifft(smooth_response).real
        recursive_impulse = np.fft.ifft(component_response).real
        single_rms = SIGMA * np.linalg.norm(single_impulse)
        recursive_rms = SIGMA * np.linalg.norm(recursive_impulse)
        if stage == 1:
            adjacent_response = smooth_response
        else:
            adjacent_response = smooth_response * (1.0 - smoother[stage - 2])
        adjacent_rms = SIGMA * np.linalg.norm(np.fft.ifft(adjacent_response))
        rows.append(
            {
                "stage": stage,
                "window_size": window_size,
                "radius": window_size // 2,
                "previous_to_current_window_ratio": (
                    np.nan
                    if stage == 1
                    else WINDOW_SIZES[stage - 2] / window_size
                ),
                "single_pass_exact_rms": single_rms,
                "recursive_exact_rms": recursive_rms,
                "single_scaled_by_a_k_over_2": single_rms / A ** (stage / 2),
                "recursive_scaled_by_a_k_over_2": recursive_rms / A ** (stage / 2),
                "single_scaled_by_a_k_minus_1_over_2": single_rms
                / A ** ((stage - 1) / 2),
                "recursive_scaled_by_a_k_minus_1_over_2": recursive_rms
                / A ** ((stage - 1) / 2),
                "recursive_to_single_sd_ratio": recursive_rms / single_rms,
                "recursive_variance_fraction_of_single": (recursive_rms / single_rms) ** 2,
                "adjacent_only_recursive_rms": adjacent_rms,
                "full_to_adjacent_sd_ratio": recursive_rms / adjacent_rms,
                "component_dc_gain": float(np.real(component_response[0])),
                "single_effective_sample_size": 1.0
                / np.sum(epanechnikov_weights(window_size) ** 2),
                "single_rms_times_sqrt_window": single_rms * np.sqrt(window_size),
            }
        )
        residual_product *= 1.0 - smooth_response
    return rows


def gen_signal(t, seed=2026, target_std=TARGET_SIGNAL_STD):
    t = np.asarray(t, dtype=float)
    rng = np.random.default_rng(seed)
    u = (t - t.min()) / np.ptp(t)

    def center_and_scale(values, scale):
        values = np.asarray(values, dtype=float) - np.mean(values)
        values_std = np.std(values)
        return values if values_std == 0 else scale * values / values_std

    baseline = np.zeros_like(u)
    for harmonic in range(1, 6):
        amplitude = rng.normal(scale=1.0 / harmonic)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        baseline += amplitude * np.sin(2.0 * np.pi * harmonic * u + phase)
    baseline = center_and_scale(baseline, scale=0.36)

    amplitude_envelope = 0.75 + 0.35 * np.sin(
        2.0 * np.pi * (1.3 * u + 0.08) + rng.uniform(0.0, 2.0 * np.pi)
    )
    chirp_phase = 2.0 * np.pi * (
        2.2 * u
        + 5.8 * u**2
        + 0.35 * np.sin(2.0 * np.pi * u + rng.uniform(0.0, 2.0 * np.pi))
    )
    chirp = center_and_scale(amplitude_envelope * np.sin(chirp_phase), scale=0.22)

    transients = np.zeros_like(u)
    for center, width, height in zip(
        rng.uniform(0.08, 0.92, size=8),
        rng.uniform(0.008, 0.04, size=8),
        rng.normal(0.0, 1.0, size=8),
    ):
        transients += height * np.exp(-0.5 * ((u - center) / width) ** 2)
    transients = center_and_scale(transients, scale=0.30)

    texture_raw = rng.normal(size=len(u))
    texture_radius = max(3, nearest_odd(0.02 * len(u)) // 2)
    texture_offsets = np.arange(-texture_radius, texture_radius + 1)
    texture_kernel = np.exp(
        -0.5 * (texture_offsets / max(1.0, texture_radius / 2.5)) ** 2
    )
    texture_kernel = texture_kernel / texture_kernel.sum()
    texture = np.convolve(texture_raw, texture_kernel, mode="same")
    texture = center_and_scale(texture, scale=0.08)

    x = baseline + chirp + transients + texture
    return x * (target_std / np.std(x))


def generate_observation(
    x,
    sigma=SIGMA,
    contamination_prob=0.05,
    contamination_scale=0.1,
    centered_contamination=False,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    gaussian_noise = rng.normal(loc=0.0, scale=sigma, size=len(x))
    contamination_mask = rng.random(len(x)) < contamination_prob
    exponential_noise = rng.exponential(scale=contamination_scale, size=len(x))
    if centered_contamination:
        exponential_noise = exponential_noise - contamination_scale
    contamination = contamination_mask * exponential_noise
    contamination *= rng.choice([-1, 1], size=len(x))
    y = x + gaussian_noise + contamination
    return y, {
        "gaussian_noise": gaussian_noise,
        "contamination": contamination,
        "contamination_mask": contamination_mask,
    }


def erf_approx(x):
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    z = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * z + a4) * z + a3) * z + a2) * z + a1) * z)
    return sign * (1.0 - poly * np.exp(-(ax**2)))


def make_lookup_grid(zmax=8.0, num=4097):
    z_grid = np.linspace(-zmax, zmax, num)
    return {
        "z": z_grid,
        "score": erf_approx(z_grid / SQRT_2),
    }


LOOKUP_GRID = make_lookup_grid()


def lookup_score(residual, scale=H_ROBUST, grid=LOOKUP_GRID):
    z = np.asarray(residual, dtype=float) / scale
    flat = np.interp(
        z.ravel(), grid["z"], grid["score"], left=-1.0, right=1.0
    )
    return flat.reshape(z.shape)


def robust_gd_fit_windows(
    windows, weights, scale=H_ROBUST, max_iter=60, tol=1e-6
):
    windows = np.asarray(windows, dtype=float)
    row_weights = (np.asarray(weights, dtype=float) / np.sum(weights))[None, :]
    x = np.median(windows, axis=1)
    lower = windows.min(axis=1)
    upper = windows.max(axis=1)
    step = 0.95 * scale / SQRT_2_OVER_PI
    for _ in range(max_iter):
        local_score = np.sum(
            row_weights * lookup_score(windows - x[:, None], scale=scale), axis=1
        )
        x_next = np.clip(x + step * local_score, lower, upper)
        max_delta = float(np.max(np.abs(x_next - x)))
        x = x_next
        if max_delta <= tol * (1.0 + float(np.max(np.abs(x)))):
            break
    return x


def local_robust_filter(y, window_size, boundary="wrap"):
    y = np.asarray(y, dtype=float)
    radius = window_size // 2
    y_padded = np.pad(y, pad_width=radius, mode=boundary)
    windows = sliding_window_view(y_padded, window_size)
    return robust_gd_fit_windows(windows, epanechnikov_weights(window_size))


def robust_imf(y, window_sizes=WINDOW_SIZES, boundary="wrap"):
    residual = np.asarray(y, dtype=float).copy()
    imfs = []
    for window_size in window_sizes:
        component = local_robust_filter(residual, window_size, boundary=boundary)
        imfs.append(component)
        residual = residual - component
    return np.asarray(imfs), residual


def observed_seed_rows():
    t = np.linspace(0.0, 1.0, N)
    x = gen_signal(t)
    y_no_contamination, no_contamination = generate_observation(
        x,
        contamination_prob=0.0,
        contamination_scale=0.2,
        rng=np.random.default_rng(777),
    )
    y_contaminated, contaminated = generate_observation(
        x,
        contamination_prob=0.2,
        contamination_scale=0.6,
        rng=np.random.default_rng(777),
    )

    clean_linear, _ = linear_imf(x)
    gaussian_linear, _ = linear_imf(y_no_contamination)
    contaminated_linear, _ = linear_imf(y_contaminated)

    clean_robust, _ = robust_imf(x)
    gaussian_robust, _ = robust_imf(y_no_contamination)
    contaminated_robust, _ = robust_imf(y_contaminated)
    contamination_only_robust, _ = robust_imf(x + contaminated["contamination"])
    contamination_only_linear, _ = linear_imf(x + contaminated["contamination"])

    cases = [
        ("linear_gaussian_only_notebook", "linear", gaussian_linear - clean_linear),
        ("robust_gaussian_only_control", "robust", gaussian_robust - clean_robust),
        ("linear_gaussian_plus_contamination_control", "linear", contaminated_linear - clean_linear),
        ("robust_gaussian_plus_contamination_notebook", "robust", contaminated_robust - clean_robust),
        ("linear_contamination_only_control", "linear", contamination_only_linear - clean_linear),
        ("robust_contamination_only_control", "robust", contamination_only_robust - clean_robust),
    ]
    rows = []
    observed_linear_recursive = None
    observed_linear_single = None
    for case, method, errors in cases:
        if case == "linear_gaussian_only_notebook":
            observed_linear_recursive = errors.copy()
        for stage, (window_size, error) in enumerate(zip(WINDOW_SIZES, errors), start=1):
            rows.append(
                {
                    "case": case,
                    "method": method,
                    "stage": stage,
                    "window_size": window_size,
                    "recursive_rmse": rmse(error),
                    "recursive_scaled_rmse_a_k_over_2": rmse(error) / A ** (stage / 2),
                    "recursive_sup_error": np.max(np.abs(error)),
                    "recursive_scaled_sup_error_a_k_over_2": np.max(np.abs(error))
                    / A ** (stage / 2),
                }
            )
    observed_linear_single = np.asarray(
        [
            local_linear_filter(no_contamination["gaussian_noise"], size)
            for size in WINDOW_SIZES
        ]
    )
    metadata = {
        "realized_contamination_fraction": float(
            np.mean(contaminated["contamination_mask"])
        ),
        "max_abs_contamination": float(np.max(np.abs(contaminated["contamination"]))),
        "linear_notebook_recursive_rmse": rmse(observed_linear_recursive, axis=1).tolist(),
        "linear_notebook_single_pass_rmse": rmse(observed_linear_single, axis=1).tolist(),
    }
    return rows, metadata, observed_linear_recursive, observed_linear_single, clean_robust, x


def summarize_samples(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": np.mean(values),
        "sd": np.std(values),
        "p05": np.quantile(values, 0.05),
        "p50": np.quantile(values, 0.50),
        "p95": np.quantile(values, 0.95),
    }


def linear_monte_carlo(observed_recursive, observed_single):
    smoother, recursive, _ = linear_transfer_functions()
    rng = np.random.default_rng(20260709)
    recursive_rmse = np.empty((LINEAR_MC_REPS, len(WINDOW_SIZES)))
    recursive_sup = np.empty_like(recursive_rmse)
    single_rmse = np.empty_like(recursive_rmse)
    single_sup = np.empty_like(recursive_rmse)
    chunk_size = 250
    for start in range(0, LINEAR_MC_REPS, chunk_size):
        stop = min(start + chunk_size, LINEAR_MC_REPS)
        noise = rng.normal(scale=SIGMA, size=(stop - start, N))
        noise_fft = np.fft.fft(noise, axis=1)
        for stage in range(len(WINDOW_SIZES)):
            rec_values = np.fft.ifft(
                noise_fft * recursive[stage][None, :], axis=1
            ).real
            single_values = np.fft.ifft(
                noise_fft * smoother[stage][None, :], axis=1
            ).real
            recursive_rmse[start:stop, stage] = rmse(rec_values, axis=1)
            recursive_sup[start:stop, stage] = np.max(np.abs(rec_values), axis=1)
            single_rmse[start:stop, stage] = rmse(single_values, axis=1)
            single_sup[start:stop, stage] = np.max(np.abs(single_values), axis=1)

    rows = []
    exact = exact_linear_rows()
    for family, rms_values, sup_values, observed_rms, observed_values in [
        (
            "recursive",
            recursive_rmse,
            recursive_sup,
            rmse(observed_recursive, axis=1),
            observed_recursive,
        ),
        (
            "single_pass",
            single_rmse,
            single_sup,
            rmse(observed_single, axis=1),
            observed_single,
        ),
    ]:
        observed_sup = np.max(np.abs(observed_values), axis=1)
        for stage, window_size in enumerate(WINDOW_SIZES, start=1):
            rms_summary = summarize_samples(rms_values[:, stage - 1])
            sup_summary = summarize_samples(sup_values[:, stage - 1])
            exact_rms = exact[stage - 1][
                "recursive_exact_rms"
                if family == "recursive"
                else "single_pass_exact_rms"
            ]
            rows.append(
                {
                    "family": family,
                    "stage": stage,
                    "window_size": window_size,
                    "exact_sqrt_expected_mean_square": exact_rms,
                    "mc_mean_rmse": rms_summary["mean"],
                    "mc_sd_rmse": rms_summary["sd"],
                    "mc_p05_rmse": rms_summary["p05"],
                    "mc_p50_rmse": rms_summary["p50"],
                    "mc_p95_rmse": rms_summary["p95"],
                    "observed_seed777_rmse": observed_rms[stage - 1],
                    "observed_rmse_percentile": np.mean(
                        rms_values[:, stage - 1] <= observed_rms[stage - 1]
                    ),
                    "mc_mean_scaled_rmse_a_k_over_2": rms_summary["mean"]
                    / A ** (stage / 2),
                    "mc_mean_sup": sup_summary["mean"],
                    "mc_p05_sup": sup_summary["p05"],
                    "mc_p50_sup": sup_summary["p50"],
                    "mc_p95_sup": sup_summary["p95"],
                    "observed_seed777_sup": observed_sup[stage - 1],
                    "observed_sup_percentile": np.mean(
                        sup_values[:, stage - 1] <= observed_sup[stage - 1]
                    ),
                    "mc_mean_scaled_sup_a_k_over_2": sup_summary["mean"]
                    / A ** (stage / 2),
                }
            )
    return rows


def exact_single_pass_boundary_rows():
    rows = []
    original_indices = np.arange(N)
    for boundary in ["wrap", "reflect"]:
        for stage, window_size in enumerate(WINDOW_SIZES, start=1):
            radius = window_size // 2
            padded_indices = np.pad(original_indices, radius, mode=boundary)
            weights = epanechnikov_weights(window_size)
            energy = np.empty(N)
            for target in range(N):
                coefficients = np.bincount(
                    padded_indices[target : target + window_size],
                    weights=weights,
                    minlength=N,
                )
                energy[target] = np.dot(coefficients, coefficients)
            point_sd = SIGMA * np.sqrt(energy)
            edge_mask = np.zeros(N, dtype=bool)
            edge_mask[:radius] = True
            edge_mask[N - radius :] = True
            interior_mask = ~edge_mask
            rows.append(
                {
                    "boundary": boundary,
                    "stage": stage,
                    "window_size": window_size,
                    "domain_exact_rms": SIGMA * np.sqrt(np.mean(energy)),
                    "min_point_sd": np.min(point_sd),
                    "max_point_sd": np.max(point_sd),
                    "first_point_sd": point_sd[0],
                    "center_point_sd": point_sd[N // 2],
                    "edge_region_rms": SIGMA * np.sqrt(np.mean(energy[edge_mask])),
                    "interior_region_rms": (
                        SIGMA * np.sqrt(np.mean(energy[interior_mask]))
                        if np.any(interior_mask)
                        else np.nan
                    ),
                }
            )
    return rows


def boundary_recursive_monte_carlo():
    rng = np.random.default_rng(20260710)
    values = {
        boundary: np.empty((BOUNDARY_MC_REPS, len(WINDOW_SIZES)))
        for boundary in ["wrap", "reflect"]
    }
    for repeat in range(BOUNDARY_MC_REPS):
        noise = rng.normal(scale=SIGMA, size=N)
        for boundary in values:
            components, _ = linear_imf(noise, boundary=boundary)
            values[boundary][repeat] = rmse(components, axis=1)
    rows = []
    for boundary, matrix in values.items():
        for stage, window_size in enumerate(WINDOW_SIZES, start=1):
            summary = summarize_samples(matrix[:, stage - 1])
            rows.append(
                {
                    "boundary": boundary,
                    "stage": stage,
                    "window_size": window_size,
                    "mc_reps": BOUNDARY_MC_REPS,
                    "mean_recursive_rmse": summary["mean"],
                    "p05_recursive_rmse": summary["p05"],
                    "p50_recursive_rmse": summary["p50"],
                    "p95_recursive_rmse": summary["p95"],
                    "mean_scaled_recursive_rmse_a_k_over_2": summary["mean"]
                    / A ** (stage / 2),
                }
            )
    return rows


def robust_monte_carlo(clean_robust, x):
    matrices = {
        "robust_gaussian_only": np.empty((ROBUST_MC_REPS, len(WINDOW_SIZES))),
        "robust_gaussian_plus_contamination": np.empty(
            (ROBUST_MC_REPS, len(WINDOW_SIZES))
        ),
    }
    master = np.random.default_rng(20260711)
    seeds = master.integers(0, 2**63 - 1, size=ROBUST_MC_REPS, dtype=np.int64)
    for repeat, seed in enumerate(seeds):
        y_gaussian, _ = generate_observation(
            x,
            contamination_prob=0.0,
            contamination_scale=0.6,
            rng=np.random.default_rng(int(seed)),
        )
        y_contaminated, _ = generate_observation(
            x,
            contamination_prob=0.2,
            contamination_scale=0.6,
            rng=np.random.default_rng(int(seed)),
        )
        gaussian_robust, _ = robust_imf(y_gaussian)
        contaminated_robust, _ = robust_imf(y_contaminated)
        matrices["robust_gaussian_only"][repeat] = rmse(
            gaussian_robust - clean_robust, axis=1
        )
        matrices["robust_gaussian_plus_contamination"][repeat] = rmse(
            contaminated_robust - clean_robust, axis=1
        )
    rows = []
    for case, matrix in matrices.items():
        for stage, window_size in enumerate(WINDOW_SIZES, start=1):
            summary = summarize_samples(matrix[:, stage - 1])
            rows.append(
                {
                    "case": case,
                    "stage": stage,
                    "window_size": window_size,
                    "mc_reps": ROBUST_MC_REPS,
                    "mean_recursive_rmse": summary["mean"],
                    "sd_recursive_rmse": summary["sd"],
                    "p05_recursive_rmse": summary["p05"],
                    "p50_recursive_rmse": summary["p50"],
                    "p95_recursive_rmse": summary["p95"],
                    "mean_scaled_recursive_rmse_a_k_over_2": summary["mean"]
                    / A ** (stage / 2),
                }
            )
    return rows


def summary_json(exact_rows, observed_rows, boundary_single_rows, boundary_mc_rows, robust_mc_rows):
    exact_single_scaled = np.array(
        [row["single_scaled_by_a_k_over_2"] for row in exact_rows]
    )
    exact_recursive_scaled = np.array(
        [row["recursive_scaled_by_a_k_over_2"] for row in exact_rows]
    )
    seed_case = [
        row
        for row in observed_rows
        if row["case"] == "linear_gaussian_only_notebook"
    ]
    robust_seed_case = [
        row
        for row in observed_rows
        if row["case"] == "robust_gaussian_plus_contamination_notebook"
    ]
    robust_gaussian_seed_case = [
        row
        for row in observed_rows
        if row["case"] == "robust_gaussian_only_control"
    ]
    reflect_stage1 = next(
        row
        for row in boundary_single_rows
        if row["boundary"] == "reflect" and row["stage"] == 1
    )
    wrap_stage1 = next(
        row
        for row in boundary_single_rows
        if row["boundary"] == "wrap" and row["stage"] == 1
    )
    boundary_means = {
        boundary: np.array(
            [
                row["mean_recursive_rmse"]
                for row in boundary_mc_rows
                if row["boundary"] == boundary
            ]
        )
        for boundary in ["wrap", "reflect"]
    }
    boundary_scaled_means = {
        boundary: np.array(
            [
                row["mean_scaled_recursive_rmse_a_k_over_2"]
                for row in boundary_mc_rows
                if row["boundary"] == boundary
            ]
        )
        for boundary in ["wrap", "reflect"]
    }
    robust_mc_stage_means = {
        case: np.array(
            [
                row["mean_scaled_recursive_rmse_a_k_over_2"]
                for row in robust_mc_rows
                if row["case"] == case
            ]
        )
        for case in [
            "robust_gaussian_only",
            "robust_gaussian_plus_contamination",
        ]
    }
    return {
        "linear_exact": {
            "single_pass_scaled_mean": float(np.mean(exact_single_scaled)),
            "single_pass_scaled_cv": float(
                np.std(exact_single_scaled) / np.mean(exact_single_scaled)
            ),
            "recursive_scaled_stage1": float(exact_recursive_scaled[0]),
            "recursive_scaled_stage2_to_9_mean": float(
                np.mean(exact_recursive_scaled[1:])
            ),
            "recursive_scaled_stage3_to_9_mean": float(
                np.mean(exact_recursive_scaled[2:])
            ),
            "stage1_to_stage2_to_9_scaled_ratio": float(
                exact_recursive_scaled[0] / np.mean(exact_recursive_scaled[1:])
            ),
            "stage1_to_stage3_to_9_scaled_ratio": float(
                exact_recursive_scaled[0] / np.mean(exact_recursive_scaled[2:])
            ),
            "stage2_to_9_scaled_cv": float(
                np.std(exact_recursive_scaled[1:])
                / np.mean(exact_recursive_scaled[1:])
            ),
            "stage3_to_9_scaled_cv": float(
                np.std(exact_recursive_scaled[2:])
                / np.mean(exact_recursive_scaled[2:])
            ),
        },
        "seed777": {
            "linear_recursive_rmse": [float(row["recursive_rmse"]) for row in seed_case],
            "linear_recursive_scaled_rmse": [
                float(row["recursive_scaled_rmse_a_k_over_2"]) for row in seed_case
            ],
            "robust_contaminated_recursive_rmse": [
                float(row["recursive_rmse"]) for row in robust_seed_case
            ],
            "robust_contaminated_recursive_scaled_rmse": [
                float(row["recursive_scaled_rmse_a_k_over_2"])
                for row in robust_seed_case
            ],
            "robust_gaussian_recursive_rmse": [
                float(row["recursive_rmse"]) for row in robust_gaussian_seed_case
            ],
            "robust_gaussian_recursive_scaled_rmse": [
                float(row["recursive_scaled_rmse_a_k_over_2"])
                for row in robust_gaussian_seed_case
            ],
        },
        "boundary": {
            "wrap_stage1_domain_exact_single_rms": float(
                wrap_stage1["domain_exact_rms"]
            ),
            "reflect_stage1_domain_exact_single_rms": float(
                reflect_stage1["domain_exact_rms"]
            ),
            "reflect_to_wrap_stage1_domain_rms_ratio": float(
                reflect_stage1["domain_exact_rms"] / wrap_stage1["domain_exact_rms"]
            ),
            "reflect_stage1_endpoint_to_center_sd_ratio": float(
                reflect_stage1["first_point_sd"] / reflect_stage1["center_point_sd"]
            ),
            "mc_stage1_to_stage2_to_9_raw_ratio": {
                boundary: float(values[0] / np.mean(values[1:]))
                for boundary, values in boundary_means.items()
            },
            "mc_stage1_to_stage2_to_9_scaled_ratio": {
                boundary: float(values[0] / np.mean(values[1:]))
                for boundary, values in boundary_scaled_means.items()
            },
        },
        "robust_mc": {
            case: {
                "stage1_scaled_mean": float(values[0]),
                "stage2_to_9_scaled_mean": float(np.mean(values[1:])),
                "stage1_to_stage2_to_9_scaled_ratio": float(
                    values[0] / np.mean(values[1:])
                ),
            }
            for case, values in robust_mc_stage_means.items()
        },
    }


def main():
    start = time.perf_counter()
    exact_rows = exact_linear_rows()
    write_csv("linear_operator_exact.csv", exact_rows)

    observed_rows, metadata, observed_recursive, observed_single, clean_robust, x = (
        observed_seed_rows()
    )
    write_csv("seed777_method_controls.csv", observed_rows)

    linear_mc_rows = linear_monte_carlo(observed_recursive, observed_single)
    write_csv("linear_monte_carlo.csv", linear_mc_rows)

    boundary_single_rows = exact_single_pass_boundary_rows()
    write_csv("single_pass_boundary_exact.csv", boundary_single_rows)

    boundary_mc_rows = boundary_recursive_monte_carlo()
    write_csv("recursive_boundary_monte_carlo.csv", boundary_mc_rows)

    robust_mc_rows = robust_monte_carlo(clean_robust, x)
    write_csv("robust_monte_carlo.csv", robust_mc_rows)

    report = summary_json(
        exact_rows,
        observed_rows,
        boundary_single_rows,
        boundary_mc_rows,
        robust_mc_rows,
    )
    report["run"] = {
        "n": N,
        "sigma": SIGMA,
        "H_robust": H_ROBUST,
        "a": float(A),
        "window_sizes": WINDOW_SIZES,
        "linear_mc_reps": LINEAR_MC_REPS,
        "boundary_mc_reps": BOUNDARY_MC_REPS,
        "robust_mc_reps": ROBUST_MC_REPS,
        "elapsed_seconds": time.perf_counter() - start,
        **metadata,
    }
    (HERE / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
