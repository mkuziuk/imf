#!/usr/bin/env python3
"""Evaluate the proved sufficient conditions; a failed condition is not a failed theorem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from check_recursive_bounds import (
    FACTOR, H, N, SIGMA, WINDOWS, curvature,
    gaussian_curvature_deviation, robust_checks, weights,
)


HERE = Path(__file__).resolve().parent


def theorem_constants(population_curvature, product_bound, schedule_constant, dispersion):
    """Equations for P(A), D and eta in irmf_all_components_bound.tex."""
    l2 = np.sqrt(2 / np.pi) / H
    l3 = np.sqrt(2 / np.pi) * np.exp(-0.5) / H**2
    lam, cb, ca, a = population_curvature, product_bound, schedule_constant, FACTOR
    critical = (a - 1) / (cb * ca)
    if dispersion >= critical:
        return {"dispersion_condition_passes": False, "critical_dispersion": critical}
    delta = (dispersion + critical) / 2
    q = cb * ca * delta / (a - 1)
    cu, cv = np.sqrt(2) + 2 / 3, np.sqrt(2) + 1 / 3
    s1, s2, r = ca / (1 - 1 / a), np.sqrt(ca) / (1 - 1 / np.sqrt(a)), ca / a
    a_lambda = cb / lam * (np.sqrt(2) * s2 + 2 * s1 / 3)
    g_inf, g_two, b_zero = 2 * cb * s1 / lam, 2 * cb * s2 / lam, l2 / lam
    a_r = 2 * l2 * cu * cv / lam**2 + 2 * l3 * cu**2 / lam**3
    c_loc = l2 * cv / lam + 4 * l3 * cu / lam**2
    accumulated_constant = 2 * a_lambda
    p_a = (
        4 * l3 / lam * accumulated_constant**2 * r
        + c_loc * accumulated_constant * np.sqrt(r)
        + b_zero * g_inf * r
        + 32 * np.e * b_zero * g_two * np.sqrt(r)
    )
    remainder_constant = cb * s1 * (a_r + p_a) / (1 - q)
    thresholds = {
        "bernstein_simplification": 1.0,
        "single_pass_hessian": (lam / (4 * l2 * cv))**2,
        "single_pass_location": (lam**2 / (8 * l3 * cu))**2,
        "curvature_dispersion": ((delta - dispersion) / (b_zero * cv))**2,
        "induction_localization": (a_lambda / remainder_constant)**2,
    }
    assert np.isclose(
        remainder_constant,
        cb * s1 * (a_r + p_a) + q * remainder_constant,
        rtol=2e-14,
    )
    assert 0 <= q < 1
    assert min(thresholds.values()) > 0
    return {
        "dispersion_condition_passes": True,
        "critical_dispersion": critical,
        "chosen_delta": delta,
        "induction_coefficient_q": q,
        "leading_constant": a_lambda,
        "accumulated_error_constant": accumulated_constant,
        "single_pass_remainder_constant": a_r,
        "propagation_forcing_constant": p_a,
        "accumulated_remainder_constant": remainder_constant,
        "sample_size_threshold_eta": min(thresholds.values()),
        "individual_thresholds": thresholds,
    }


def check_gaussian_formula():
    differences = []
    for offset in [-0.8, 0.0, 0.6]:
        mean = curvature(offset, np.hypot(H, SIGMA))
        peak = np.sqrt(2 / np.pi) / H
        threshold = H * np.sqrt(2 * np.log(peak / mean))
        knots = [-np.inf, (-threshold - offset) / SIGMA,
                 (threshold - offset) / SIGMA, np.inf]

        def integrand(z):
            return (
                abs(curvature(offset + SIGMA * z, H) - mean)
                * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            )

        numerical = sum(
            quad(integrand, left, right, epsabs=1e-13, epsrel=1e-13)[0]
            for left, right in zip(knots[:-1], knots[1:], strict=True)
        )
        differences.append(abs(numerical - gaussian_curvature_deviation(offset)))
    assert max(differences) < 1e-11
    return float(max(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    checks = robust_checks(3)
    constants = theorem_constants(
        checks["minimum_population_curvature"],
        checks["all_partial_residual_linf"],
        checks["geometric_schedule_constant"],
        checks["maximum_curvature_dispersion"],
    )
    alpha_max = max(weights(window)[1].max() for window in WINDOWS)
    confidence_x = float(np.log(20))
    z_zero = float(np.log(10 * N * len(WINDOWS)))
    z_alpha = (confidence_x + z_zero) * alpha_max
    eta = constants.get("sample_size_threshold_eta", 0.0)
    report = {
        "scope": "Floating-point evaluation of sufficient conditions, not interval certification",
        "n": N,
        "windows": WINDOWS,
        "sigma": SIGMA,
        "H": H,
        "gaussian_deviation_formula_max_quadrature_difference": check_gaussian_formula(),
        "maximum_curvature_dispersion": checks["maximum_curvature_dispersion"],
        "curvature_dispersion_by_stage": checks["curvature_dispersion_by_stage"],
        "population_product_bound": checks["all_partial_residual_linf"],
        "max_propagation_split_error": checks["max_propagation_split_error"],
        "constants": constants,
        "confidence_x": confidence_x,
        "z_times_alpha_max": z_alpha,
        "sample_size_condition_passes": bool(z_alpha <= eta),
        "target_moment_cutoff_x_star": eta / alpha_max - z_zero,
        "paper_target_conditions_applicable": bool(eta / alpha_max > z_zero),
        "conclusion": (
            "The dispersion condition passes. The sample-size condition fails; "
            "this theorem does not certify the n=1000 experiment."
            if constants["dispersion_condition_passes"] and z_alpha > eta
            else "Inspect each sufficient condition in this report."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_components_conditions.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
