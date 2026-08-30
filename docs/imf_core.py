"""Canonical NumPy implementation of Intrinsic Multiscale Filtering (IMF).

This module is shared between the research notebooks and the demo website
(where it runs unmodified inside the browser via Pyodide). The definitions
mirror the notebook versions verbatim; the only additions are the kernel
menu, the contrast menu, and the JSON entry point ``run_demo`` used by the
web worker.
"""

from __future__ import annotations

import json
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SQRT_2 = np.sqrt(2.0)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
TARGET_SIGNAL_STD = 0.38545431761087123


# ---------------------------------------------------------------------------
# Window schedule (verbatim from the notebooks)
# ---------------------------------------------------------------------------

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


def make_window_schedule(n, factor=np.sqrt(2.0), min_window_size=31):
    if n <= 0:
        raise ValueError("n must be positive")
    if factor <= 1:
        raise ValueError("factor must be larger than 1")

    first = odd_ceiling(n / 2)
    if first > n:
        first = n if n % 2 == 1 else n - 1

    min_size = nearest_odd(min_window_size)
    if min_size > first:
        return [first]

    sizes = [first]
    current = first
    while current > min_size:
        candidate = nearest_odd(current / factor)
        candidate = min(candidate, current - 2)
        if candidate % 2 == 0:
            candidate -= 1
        if candidate < min_size:
            candidate = min_size
        sizes.append(candidate)
        current = candidate
    return sizes


# ---------------------------------------------------------------------------
# Kernels. "squared_triangular" is the kernel used throughout the notebooks
# (historically named epanechnikov_weights there).
# ---------------------------------------------------------------------------

KERNEL_PROFILES = {
    "squared_triangular": lambda u: 0.75 * np.maximum(0.0, 1.0 - np.abs(u)) ** 2,
    "epanechnikov": lambda u: 0.75 * np.maximum(0.0, 1.0 - u**2),
    "triangular": lambda u: np.maximum(0.0, 1.0 - np.abs(u)),
    "uniform": lambda u: np.ones_like(u),
    "tricube": lambda u: np.maximum(0.0, 1.0 - np.abs(u) ** 3) ** 3,
    "gaussian": lambda u: np.exp(-0.5 * (2.5 * u) ** 2),
}


def kernel_weights(window_size, kind="squared_triangular"):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    radius = window_size // 2
    if radius == 0:
        return np.array([1.0])
    offsets = np.arange(-radius, radius + 1)
    u = offsets / radius
    weights = KERNEL_PROFILES[kind](u)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------

def gen_signal_simple(
    t,
    slow_amp=0.6,
    fast_amp=0.25,
    fast_cycles=6.0,
    bump_amp=0.8,
    bump_pos=0.55,
    bump_width=0.015,
    trend_slope=0.5,
):
    slow = slow_amp * np.sin(2 * np.pi * t)
    fast = fast_amp * np.sin(2 * np.pi * fast_cycles * t)
    bump = bump_amp * np.exp(-((t - bump_pos) ** 2) / (2 * bump_width**2))
    trend = trend_slope * (t - 0.5)
    return slow + fast + bump + trend


def gen_signal_complex(t, seed=2026, target_std=TARGET_SIGNAL_STD):
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


# ---------------------------------------------------------------------------
# Observation models
# ---------------------------------------------------------------------------

def generate_observation(
    x,
    sigma=0.4,
    contamination_prob=0.05,
    contamination_scale=0.1,
    model="additive",
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    gaussian_noise = rng.normal(loc=0.0, scale=sigma, size=len(x))
    contamination_mask = rng.random(len(x)) < contamination_prob
    exponential_noise = rng.exponential(scale=contamination_scale, size=len(x))
    contamination = contamination_mask * exponential_noise
    contamination *= rng.choice([-1, 1], size=len(x))
    if model == "additive":
        y = x + gaussian_noise + contamination
    elif model == "masked":
        y = np.where(contamination_mask, gaussian_noise + contamination, x + gaussian_noise)
    else:
        raise ValueError(f"unknown observation model: {model}")
    return y, {
        "gaussian_noise": gaussian_noise,
        "contamination": contamination,
        "contamination_mask": contamination_mask,
    }


# ---------------------------------------------------------------------------
# Contrast functions and their scores
# ---------------------------------------------------------------------------

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
    poly = ((((a5 * z + a4) * z + a3) * z + a2) * z + a1) * z
    return sign * (1.0 - poly * np.exp(-(ax**2)))


def smooth_abs_score(residual, scale):
    """psi_H(r) = erf(r / (sqrt(2) H)) -- bounded score of the smoothed |.|."""
    return erf_approx(np.asarray(residual, dtype=float) / (SQRT_2 * scale))


def gd_step_size(contrast, scale):
    """0.95 divided by the curvature bound of the contrast."""
    if contrast == "smooth_abs":
        return 0.95 * scale / SQRT_2_OVER_PI
    if contrast == "quadratic":
        return 0.95
    raise ValueError(f"unknown contrast: {contrast}")


def gd_score(residual, contrast, scale):
    if contrast == "smooth_abs":
        return smooth_abs_score(residual, scale)
    if contrast == "quadratic":
        return np.asarray(residual, dtype=float)
    raise ValueError(f"unknown contrast: {contrast}")


# ---------------------------------------------------------------------------
# Local fits
# ---------------------------------------------------------------------------

def _windows(y, window_size, boundary="wrap"):
    y = np.asarray(y, dtype=float)
    radius = window_size // 2
    y_padded = np.pad(y, pad_width=radius, mode=boundary)
    return sliding_window_view(y_padded, window_size)


def local_linear_filter(y, window_size, kernel="squared_triangular", boundary="wrap"):
    return _windows(y, window_size, boundary) @ kernel_weights(window_size, kernel)


def gd_fit_windows(
    windows, weights, contrast="smooth_abs", scale=0.8, max_iter=60, tol=1e-6
):
    windows = np.asarray(windows, dtype=float)
    row_weights = (np.asarray(weights, dtype=float) / np.sum(weights))[None, :]
    x = np.median(windows, axis=1)
    lower = windows.min(axis=1)
    upper = windows.max(axis=1)
    step = gd_step_size(contrast, scale)
    trace = []
    for _ in range(max_iter):
        local_score = np.sum(
            row_weights * gd_score(windows - x[:, None], contrast, scale), axis=1
        )
        x_next = np.clip(x + step * local_score, lower, upper)
        max_delta = float(np.max(np.abs(x_next - x)))
        trace.append(max_delta)
        x = x_next
        if max_delta <= tol * (1.0 + float(np.max(np.abs(x)))):
            break
    return x, trace


def local_gd_filter(
    y,
    window_size,
    kernel="squared_triangular",
    contrast="smooth_abs",
    scale=0.8,
    boundary="wrap",
    max_iter=60,
    tol=1e-6,
):
    return gd_fit_windows(
        _windows(y, window_size, boundary),
        kernel_weights(window_size, kernel),
        contrast=contrast,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Full decompositions
# ---------------------------------------------------------------------------

def linear_imf_with_history(y, window_sizes, kernel="squared_triangular", boundary="wrap"):
    residual = np.asarray(y, dtype=float).copy()
    components = []
    for window_size in window_sizes:
        component = local_linear_filter(residual, window_size, kernel, boundary)
        components.append(component)
        residual = residual - component
    return {"components": np.asarray(components), "residual": residual}


def gd_imf_with_history(
    y,
    window_sizes,
    kernel="squared_triangular",
    contrast="smooth_abs",
    scale=0.8,
    boundary="wrap",
    max_iter=60,
    tol=1e-6,
):
    residual = np.asarray(y, dtype=float).copy()
    components, traces = [], []
    for window_size in window_sizes:
        component, trace = local_gd_filter(
            residual, window_size, kernel, contrast, scale, boundary, max_iter, tol
        )
        components.append(component)
        traces.append(trace)
        residual = residual - component
    return {
        "components": np.asarray(components),
        "residual": residual,
        "traces": traces,
    }


def rmse(values, axis=None):
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.mean(values**2, axis=axis))


# ---------------------------------------------------------------------------
# Entry point for the web worker
# ---------------------------------------------------------------------------

def _round_list(array, digits=5):
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def _round_matrix(matrix, digits=5):
    return [np.round(np.asarray(row, dtype=float), digits).tolist() for row in matrix]


def run_demo(params_json):
    params = json.loads(params_json)
    started = time.perf_counter()

    n = int(params["n"])
    sigma = float(params["sigma"])
    factor = float(params["factor"])
    contrast = params["contrast"]
    kernel = params["kernel"]
    scale = max(float(params["h_ratio"]) * sigma, 1e-3)

    t = np.linspace(0.0, 1.0, n)
    sp = params.get("signal_params", {})
    if params["signal"] == "simple":
        x = gen_signal_simple(
            t,
            slow_amp=float(sp.get("slow_amp", 0.6)),
            fast_amp=float(sp.get("fast_amp", 0.25)),
            fast_cycles=float(sp.get("fast_cycles", 6.0)),
            bump_amp=float(sp.get("bump_amp", 0.8)),
            bump_pos=float(sp.get("bump_pos", 0.55)),
            bump_width=float(sp.get("bump_width", 0.015)),
            trend_slope=float(sp.get("trend_slope", 0.5)),
        )
    else:
        x = gen_signal_complex(
            t,
            seed=int(sp.get("signal_seed", 2026)),
            target_std=TARGET_SIGNAL_STD * float(sp.get("signal_scale", 1.0)),
        )
    rng = np.random.default_rng(int(params["seed"]))
    y, info = generate_observation(
        x,
        sigma=sigma,
        contamination_prob=float(params["p"]),
        contamination_scale=float(params["scale"]),
        model=params["model"],
        rng=rng,
    )
    window_sizes = make_window_schedule(n, factor, int(params["min_window"]))

    linear = linear_imf_with_history(y, window_sizes, kernel)
    linear_ref = linear_imf_with_history(x, window_sizes, kernel)
    gd = gd_imf_with_history(y, window_sizes, kernel, contrast, scale)
    gd_ref = gd_imf_with_history(x, window_sizes, kernel, contrast, scale)

    linear_denoised = linear["components"].sum(axis=0)
    gd_denoised = gd["components"].sum(axis=0)
    stages = np.arange(1, len(window_sizes) + 1)
    normalizer = factor ** (stages / 2.0)

    stage_rmse_linear = rmse(linear["components"] - linear_ref["components"], axis=1)
    stage_rmse_gd = rmse(gd["components"] - gd_ref["components"], axis=1)

    payload = {
        "t": _round_list(t, 6),
        "clean": _round_list(x),
        "observed": _round_list(y),
        "mask_idx": np.flatnonzero(info["contamination_mask"]).tolist(),
        "windows": [int(w) for w in window_sizes],
        "linear": {
            "components": _round_matrix(linear["components"]),
            "residual": _round_list(linear["residual"]),
            "denoised": _round_list(linear_denoised),
        },
        "gd": {
            "components": _round_matrix(gd["components"]),
            "residual": _round_list(gd["residual"]),
            "denoised": _round_list(gd_denoised),
            "traces": [[float(f"{v:.3e}") for v in trace] for trace in gd["traces"]],
            "iters": [len(trace) for trace in gd["traces"]],
        },
        "linear_ref": {"components": _round_matrix(linear_ref["components"])},
        "gd_ref": {"components": _round_matrix(gd_ref["components"])},
        "metrics": {
            "recon_linear": float(
                np.max(np.abs(y - (linear["components"].sum(axis=0) + linear["residual"])))
            ),
            "recon_gd": float(
                np.max(np.abs(y - (gd["components"].sum(axis=0) + gd["residual"])))
            ),
            "rmse_observed": float(rmse(y - x)),
            "rmse_linear": float(rmse(linear_denoised - x)),
            "rmse_gd": float(rmse(gd_denoised - x)),
            "stage_rmse_linear": _round_list(stage_rmse_linear, 6),
            "stage_rmse_gd": _round_list(stage_rmse_gd, 6),
            "stage_rmse_linear_scaled": _round_list(stage_rmse_linear / normalizer, 6),
            "stage_rmse_gd_scaled": _round_list(stage_rmse_gd / normalizer, 6),
            "contam_fraction": float(np.mean(info["contamination_mask"])),
            "gd_step": gd_step_size(contrast, scale),
            "h_value": scale,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return json.dumps(payload)
