#!/usr/bin/env python3
"""Check the accepted identities and finite-design operator assumptions.

This is a numerical diagnostic, not a proof of the nonlinear remainder rate.
Run from any directory. Optional --archive audits the submitted stored results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.special import erf, ndtr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N = 1000
WINDOWS = [501, 355, 251, 177, 125, 89, 63, 45, 31]
SIGMA = 0.4
H = 0.8
FACTOR = np.sqrt(2.0)


def weights(window, kernel="squared_triangle"):
    radius = window // 2
    offsets = np.arange(-radius, radius + 1)
    u = offsets / radius
    values = (1 - np.abs(u)) ** 2 if kernel == "squared_triangle" else 1 - u**2
    return offsets, values / values.sum()


def score(values, scale):
    return erf(values / (np.sqrt(2.0) * scale))


def curvature(values, scale):
    return np.sqrt(2 / np.pi) / scale * np.exp(-0.5 * (values / scale) ** 2)


def gaussian_curvature_deviation(offsets, loss_scale=H, noise_scale=SIGMA):
    """Exact E|rho''(offset + Gaussian noise) - E rho''|, not quadrature."""
    combined = np.hypot(loss_scale, noise_scale)
    mean = curvature(offsets, combined)
    peak = np.sqrt(2 / np.pi) / loss_scale
    threshold = loss_scale * np.sqrt(2 * np.log(peak / mean))
    tilted_mean = offsets * loss_scale**2 / combined**2
    tilted_scale = noise_scale * loss_scale / combined
    tilted_probability = (
        ndtr((threshold - tilted_mean) / tilted_scale)
        - ndtr((-threshold - tilted_mean) / tilted_scale)
    )
    original_probability = (
        ndtr((threshold - offsets) / noise_scale)
        - ndtr((-threshold - offsets) / noise_scale)
    )
    return 2 * mean * (tilted_probability - original_probability)


def smooth(values, indices, weight, scale):
    """Bracketed Newton with a score-residual check and frozen converged rows."""
    windows = values[indices]
    lower, upper = windows.min(1), windows.max(1)
    fit = windows @ weight
    for _ in range(150):
        delta = windows - fit[:, None]
        gradient = score(delta, scale) @ weight
        active = np.abs(gradient) > 2e-14
        if not active.any():
            return fit
        lower = np.where(active & (gradient > 0), fit, lower)
        upper = np.where(active & (gradient < 0), fit, upper)
        hessian = curvature(delta, scale) @ weight
        step = np.divide(gradient, hessian, out=np.zeros_like(fit), where=hessian > 0)
        candidate = fit + step
        bad = (candidate <= lower) | (candidate >= upper) | ~np.isfinite(candidate)
        candidate = np.where(bad, (lower + upper) / 2, candidate)
        fit = np.where(active, candidate, fit)
    raise AssertionError("Local score did not converge")


def load_signal():
    path = ROOT / "research/first-imf-recursive-error/diagnostics/run_numerical_diagnosis.py"
    spec = importlib.util.spec_from_file_location("first_component_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gen_signal(np.linspace(0, 1, N))


def linear_checks():
    rows = []
    for kernel in ["squared_triangle", "epanechnikov"]:
        multipliers = []
        for window in WINDOWS:
            offsets, weight = weights(window, kernel)
            impulse = np.zeros(N)
            np.add.at(impulse, offsets % N, weight)
            multipliers.append(np.fft.fft(impulse))
        for k, multiplier in enumerate(multipliers):
            product = np.ones(N, dtype=complex)
            component_norms = [np.abs(np.fft.ifft(multiplier)).sum()]
            bare_norms = []
            # Include every l=1,...,k and the empty product. In particular,
            # evaluate AFTER the last update to include the longest product.
            for j in range(k - 1, -1, -1):
                product *= 1 - multipliers[j]
                component_norms.append(np.abs(np.fft.ifft(multiplier * product)).sum())
            product = np.ones(N, dtype=complex)
            for j in range(k, -1, -1):
                product *= 1 - multipliers[j]
                bare_norms.append(np.abs(np.fft.ifft(product)).sum())
            prefix = np.prod([1 - w for w in multipliers[:k]], axis=0) if k else np.ones(N)
            rows.append({
                "kernel": kernel, "stage": k + 1, "window": WINDOWS[k],
                "single_residual_l2": float(np.max(np.abs(1 - multiplier))),
                "prefix_l2": float(np.max(np.abs(prefix))),
                "prefix_linf": float(np.abs(np.fft.ifft(prefix)).sum()),
                "prefix_component_linf": float(np.abs(np.fft.ifft(multiplier * prefix)).sum()),
                "all_partial_component_linf": float(max(component_norms)),
                "all_partial_residual_linf": float(max(bare_norms)),
            })
    return rows


def robust_checks(repeats):
    signal = load_signal()
    indices, all_weights, references, targets, kernels, fisher = [], [], [], [], [], []
    residual = signal.copy()
    scale = np.sqrt(H**2 + SIGMA**2)
    identity = np.eye(N)
    tilts = []
    curvature_dispersion = []
    for window in WINDOWS:
        offsets, weight = weights(window)
        index = (np.arange(N)[:, None] + offsets) % N
        target = smooth(residual, index, weight, scale)
        values = curvature(residual[index] - target[:, None], scale) * weight
        information = values.sum(1)
        deviation = gaussian_curvature_deviation(residual[index] - target[:, None])
        curvature_dispersion.append(float(((deviation @ weight) / information).max()))
        kernel = np.zeros((N, N))
        kernel[np.arange(N)[:, None], index] = values / information[:, None]
        tilts.append(float(np.abs(values / information[:, None] - weight).sum(1).max()))
        indices.append(index)
        all_weights.append(weight)
        references.append(residual.copy())
        targets.append(target)
        kernels.append(kernel)
        fisher.append(information)
        residual -= target

    composite_max, bare_max = 1.0, 1.0
    for k, kernel in enumerate(kernels):
        product = identity.copy()
        for j in range(k - 1, -1, -1):
            product = product @ (identity - kernels[j])
            composite_max = max(composite_max, float(np.abs(kernel @ product).sum(1).max()))
        product = identity.copy()
        for j in range(k, -1, -1):
            product = product @ (identity - kernels[j])
            bare_max = max(bare_max, float(np.abs(product).sum(1).max()))

    rng = np.random.default_rng(777)
    max_identity_error = 0.0
    max_reconstruction_error = 0.0
    max_propagation_split_error = 0.0
    sup = {name: [] for name in ["e", "L", "eL", "R", "Phi"]}
    for _ in range(repeats):
        noise = SIGMA * rng.normal(size=N)
        residual = signal + noise
        accumulated = np.zeros(N)
        leading_accumulated = np.zeros(N)
        remainder_accumulated = np.zeros(N)
        components = []
        sample = {name: [] for name in sup}
        for k, kernel in enumerate(kernels):
            index, weight = indices[k], all_weights[k]
            fit = smooth(residual, index, weight, H)
            error = fit - targets[k]
            single = smooth(references[k] + noise, index, weight, H) - targets[k]
            local_leading = (score(references[k][index] + noise[index] - targets[k][:, None], H) @ weight) / fisher[k]
            fisher_remainder = single - local_leading
            propagation = single - error - kernel @ accumulated
            leading = local_leading - kernel @ leading_accumulated
            remainder = fisher_remainder - propagation - kernel @ remainder_accumulated
            unnormalized = (
                curvature(references[k][index] + noise[index] - targets[k][:, None], H)
                * weight / fisher[k][:, None]
            )
            base_fit = single + targets[k]
            fitted = curvature(
                references[k][index] + noise[index] - base_fit[:, None], H
            ) * weight
            jacobian = fitted / fitted.sum(1, keepdims=True)
            population = kernel[np.arange(N)[:, None], index]
            fluctuation = unnormalized - population
            quadratic = (
                single - error - (jacobian * accumulated[index]).sum(1)
            )
            location = ((jacobian - unnormalized) * accumulated[index]).sum(1)
            fluctuation_leading = (
                fluctuation * leading_accumulated[index]
            ).sum(1)
            fluctuation_remainder = (
                fluctuation * remainder_accumulated[index]
            ).sum(1)
            split_error = propagation - (
                quadratic + location + fluctuation_leading + fluctuation_remainder
            )
            max_propagation_split_error = max(
                max_propagation_split_error, float(np.abs(split_error).max())
            )
            max_identity_error = max(max_identity_error, float(np.max(np.abs(error - leading - remainder))))
            for name, value in zip(sup, [error, leading, error - leading, fisher_remainder, propagation]):
                sample[name].append(float(np.max(np.abs(value))))
            accumulated += error
            leading_accumulated += leading
            remainder_accumulated += remainder
            residual -= fit
            components.append(fit)
        max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(np.sum(components, axis=0) + residual - signal - noise))))
        for name in sup:
            sup[name].append(sample[name])
    assert max_identity_error < 1e-11
    assert max_reconstruction_error < 1e-11
    assert max_propagation_split_error < 1e-11
    alpha = np.array([weights(window)[1].max() for window in WINDOWS])
    schedule_constant = max(alpha[j] / (alpha[k] * FACTOR ** (-(k - j))) for k in range(len(WINDOWS)) for j in range(k + 1))
    return {
        "repeats": repeats, "seed": 777, "noise": "Gaussian",
        "minimum_population_curvature": float(np.min(fisher)),
        "tilt_sum": float(sum(tilts)),
        "all_partial_composite_linf": composite_max,
        "all_partial_residual_linf": bare_max,
        "geometric_schedule_constant": float(schedule_constant),
        "curvature_dispersion_by_stage": curvature_dispersion,
        "maximum_curvature_dispersion": max(curvature_dispersion),
        "max_decomposition_error": max_identity_error,
        "max_reconstruction_error": max_reconstruction_error,
        "max_propagation_split_error": max_propagation_split_error,
        "sup": sup,
    }


def archive_checks(path, robust, linear):
    manifest = json.loads((HERE.parent / "source/manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive does not match the reviewed source manifest")
    results = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".npz"):
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                    results[name] = {
                        "stored_repeats": int(data["sup_e"].shape[0]),
                        "median_relative_sup_remainder": np.median(data["sup_eL"] / data["sup_e"], axis=0).tolist(),
                        "tilt_sum": float(data["tilt_inf"].sum()),
                    }
                    if name == "leading_term_mc_gauss.npz":
                        count = min(robust["repeats"], data["sup_e"].shape[0])
                        differences = {
                            key: float(np.max(np.abs(
                                np.array(robust["sup"][key])[:count]
                                - data[f"sup_{key}"][:count]
                            )))
                            for key in robust["sup"]
                        }
                        results[name]["rerun_compared_samples"] = count
                        results[name]["rerun_max_absolute_differences"] = differences
                        assert max(differences.values()) < 1e-9
        original = list(csv.DictReader(io.StringIO(archive.read("linear_contraction_check.csv").decode())))
        for expected, actual in zip(original, linear, strict=True):
            for old, new, digits in [
                ("norm2_I_minus_Wk", "single_residual_l2", 4),
                ("norm2_prod", "prefix_l2", 4),
                ("norminf_prod", "prefix_linf", 3),
                ("norminf_component", "prefix_component_linf", 3),
            ]:
                assert abs(float(expected[old]) - round(actual[new], digits)) < 1e-12
        results["linear_csv_reproduced"] = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    linear = linear_checks()
    robust = robust_checks(args.repeats)
    report = {"scope": "Finite-design checks; no proof of nonlinear remainder rate", "n": N, "windows": WINDOWS, "sigma": SIGMA, "H": H, "robust": robust}
    if args.archive:
        report["submitted_results"] = archive_checks(args.archive, robust, linear)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "linear_partial_products.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=linear[0])
        writer.writeheader()
        writer.writerows(linear)
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in robust.items() if key != "sup"}, indent=2))


if __name__ == "__main__":
    main()
