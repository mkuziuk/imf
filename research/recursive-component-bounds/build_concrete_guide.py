#!/usr/bin/env python3
"""Compute the eight-observation example in the concrete IRMF guide.

This is a deterministic teaching example, not a certificate of the theorem's
sample-size assumptions. It uses the same smoothed absolute loss as the proof.
Run from any directory with the repository's Python environment.
"""

from pathlib import Path
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "overleaf/irmf_intuitive_guide"
LOSS_WIDTH = 0.8
NOISE_SD = 0.2
SIGNAL = np.array([3.4, 3.4, 2.6, 2.6, 1.4, 1.4, 0.6, 0.6])
NOISE = np.array([0.2, -0.1, 0.3, -0.2, -0.15, 0.05, 0.1, -0.2])
WINDOWS = [7, 5, 3]


def score(values, width):
    return erf(values / (np.sqrt(2) * width))


def curvature(values, width):
    return np.sqrt(2 / np.pi) / width * np.exp(-values**2 / (2 * width**2))


def fit(values, weights, width):
    fitted = np.array([
        brentq(lambda level: row @ score(values - level, width),
               values.min() - 1, values.max() + 1, xtol=1e-14)
        for row in weights
    ])
    residual_scores = np.sum(weights * score(values[None, :] - fitted[:, None], width), axis=1)
    assert np.max(np.abs(residual_scores)) < 2e-13
    return fitted


def coordinates(name, x_values, y_values):
    pairs = " ".join(f"({x:.9g},{y:.12g})" for x, y in zip(x_values, y_values))
    return rf"\def\{name}{{{pairs}}}" + "\n"


def main():
    n = len(SIGNAL)
    reference = SIGNAL.copy()
    residual = SIGNAL + NOISE
    accumulated = np.zeros(n)
    predicted_accumulated = np.zeros(n)
    stages = []
    identity_errors = []
    population_width = np.hypot(LOSS_WIDTH, NOISE_SD)
    for stage, window in enumerate(WINDOWS, 1):
        weights = np.zeros((n, n))
        for position in range(n):
            indices = (position + np.arange(-(window // 2), window // 2 + 1)) % n
            weights[position, indices] = 1 / window
        target = fit(reference, weights, population_width)
        offsets = reference[None, :] - target[:, None]
        expected_curvature = curvature(offsets, population_width)
        information = np.sum(weights * expected_curvature, axis=1)
        population_weights = weights * expected_curvature / information[:, None]
        scores = np.sum(weights * score(offsets + NOISE, LOSS_WIDTH), axis=1)
        direct_prediction = scores / information
        unshifted = reference + NOISE
        single_pass = fit(unshifted, weights, LOSS_WIDTH)
        fitted = fit(residual, weights, LOSS_WIDTH)
        error = fitted - target
        single_remainder = single_pass - target - direct_prediction
        # Proposition 4.1 uses sums. Here its kernel is 1 inside the window,
        # while the recursion above uses normalized weights 1 / window.
        empirical_information = np.sum(weights * curvature(offsets + NOISE, LOSS_WIDTH), axis=1)
        empirical_newton_step = scores / empirical_information
        taylor_remainder = single_pass - target - empirical_newton_step
        curvature_replacement = empirical_newton_step - direct_prediction
        proposition = {
            "kernel_mass": window,
            "nabla": window * scores,
            "F": window * information,
            "F_circle": window * empirical_information,
            "D": np.sqrt(window * information),
            "empirical_newton_step": empirical_newton_step,
            "taylor_remainder": taylor_remainder,
            "curvature_replacement": curvature_replacement,
            "scaled_remainder": np.sqrt(window * information) * single_remainder,
        }
        propagation_remainder = single_pass - fitted - population_weights @ accumulated
        leading = direct_prediction - population_weights @ predicted_accumulated
        remainder = error - leading
        accumulated_remainder = accumulated - predicted_accumulated

        fitted_numerator = weights * curvature(unshifted[None, :] - single_pass[:, None], LOSS_WIDTH)
        fitted_jacobian = fitted_numerator / fitted_numerator.sum(axis=1)[:, None]
        target_numerator = weights * curvature(offsets + NOISE, LOSS_WIDTH) / information[:, None]
        fluctuation = target_numerator - population_weights
        parts = {
            "bending": single_pass - fitted - fitted_jacobian @ accumulated,
            "starting_location": (fitted_jacobian - target_numerator) @ accumulated,
            "fluctuation_times_leading": fluctuation @ predicted_accumulated,
            "fluctuation_times_remainder": fluctuation @ accumulated_remainder,
        }
        identity_errors.extend([
            float(np.max(np.abs(single_remainder - taylor_remainder - curvature_replacement))),
            float(np.max(np.abs(direct_prediction - proposition["nabla"] / proposition["F"]))),
            float(np.max(np.abs(residual - (reference + NOISE - accumulated)))),
            float(np.max(np.abs(error - (direct_prediction + single_remainder
                  - propagation_remainder - population_weights @ accumulated)))),
            float(np.max(np.abs(propagation_remainder - sum(parts.values())))),
        ])
        record = {
            "stage": stage, "window": window, "reference_input": reference.copy(),
            "actual_input": residual.copy(), "target": target, "fit": fitted,
            "error": error, "leading": leading, "remainder": remainder,
            "accumulated_error": accumulated.copy(),
            "predicted_accumulated_error": predicted_accumulated.copy(),
            "accumulated_remainder": accumulated_remainder,
            "score": scores, "population_curvature": information,
            "direct_prediction": direct_prediction, "single_pass": single_pass,
            "single_remainder": single_remainder,
            "propagation_remainder": propagation_remainder,
            "population_weights": population_weights, "fitted_jacobian": fitted_jacobian,
            "curvature_fluctuation": fluctuation, "propagation_parts": parts,
            "proposition_4_1": proposition,
        }
        stages.append(record)
        reference -= target
        residual -= fitted
        accumulated += error
        predicted_accumulated = (np.eye(n) - population_weights) @ predicted_accumulated + direct_prediction

    assert max(identity_errors) < 1e-12
    assert np.max(np.abs(SIGNAL + NOISE - residual - sum(s["fit"] for s in stages))) < 1e-12
    linear_inputs = [SIGNAL + NOISE]
    for window in [8, 4, 2]:
        values = linear_inputs[-1]
        block_fit = np.repeat(values.reshape(-1, window).mean(axis=1), window)
        linear_inputs.append(values - block_fit)
    direct_pair_residual = SIGNAL + NOISE - np.repeat((SIGNAL + NOISE).reshape(-1, 2).mean(axis=1), 2)
    assert np.max(np.abs(linear_inputs[-1] - direct_pair_residual)) < 1e-12

    # Check the two illustrative induction calculations separately from the data.
    budget = 4 / (1 - 0.4)
    assert np.isclose(4 + 0.4 * budget, budget)
    assert budget / np.sqrt(45) < 1
    assert np.isclose(2 * 0.001 / 0.5, 0.004)

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serializable(item) for item in value]
        return value

    output = {"scope": "Computed teaching example, not a theorem certificate",
              "signal": SIGNAL, "noise": NOISE, "noise_status": "Fixed illustrative vector; Gaussian law sets the population reference",
              "loss_width": LOSS_WIDTH, "reference_noise_sd": NOISE_SD,
              "population_loss_width": population_width, "stages": stages,
              "linear_block_residuals": linear_inputs,
              "max_identity_error": max(identity_errors)}
    (GUIDE / "worked_example.json").write_text(json.dumps(serializable(output), indent=2) + "\n")
    lines = ["% Generated by build_concrete_guide.py.\n"]
    lines.append(coordinates("cleanData", np.arange(1, n + 1), SIGNAL))
    lines.append(coordinates("observedData", np.arange(1, n + 1), SIGNAL + NOISE))
    lines.append(coordinates("componentError", np.arange(1, 4), [s["error"][0] for s in stages]))
    lines.append(coordinates("componentLeading", np.arange(1, 4), [s["leading"][0] for s in stages]))
    lines.append(coordinates("componentRemainder", np.arange(1, 4), [s["remainder"][0] for s in stages]))
    # Plot the total score and the two slopes used in the one-pass proof.
    item = stages[2]
    shifts = np.linspace(0, 0.010, 101)
    local_data = (item["reference_input"] + NOISE)[[7, 0, 1]]
    actual_scores = np.sum(score(local_data[None, :] - item["target"][0]
                                 - shifts[:, None], LOSS_WIDTH), axis=1)
    prop = item["proposition_4_1"]
    lines.append(coordinates("proofActualScore", 1000 * shifts, actual_scores))
    lines.append(coordinates("proofEmpiricalTangent", 1000 * shifts,
                             prop["nabla"][0] - prop["F_circle"][0] * shifts))
    lines.append(coordinates("proofPopulationSlope", 1000 * shifts,
                             prop["nabla"][0] - prop["F"][0] * shifts))
    (GUIDE / "example_data.tex").write_text("".join(lines))
    print(f"Computed 3 robust stages; maximum identity error {max(identity_errors):.3g}.")
    for item in stages:
        print(f"Stage {item['stage']}: e={item['error'][0]:+.9f}, "
              f"L={item['leading'][0]:+.9f}, Q={item['remainder'][0]:+.9f}")


if __name__ == "__main__":
    main()
