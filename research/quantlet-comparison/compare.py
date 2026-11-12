"""Reproduce the bounded checks in comparison.md without executing notebook demos.

Run from the imf root with .venv/bin/python research/quantlet-comparison/compare.py.
The QuantLet checkout is expected beside imf, or supplied with --quantlet.
"""

import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imf-comparison-mpl"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_functions(filename):
    """Load original function bodies and imports, omitting experiment side effects."""
    nb = json.loads(filename.read_text())
    nodes = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                    nodes.append(node)
    namespace = {"SQRT_2": np.sqrt(2.0), "SQRT_2_OVER_PI": np.sqrt(2.0 / np.pi)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), namespace)
    namespace["lookup_grid"] = namespace["make_lookup_grid"]()
    return namespace


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantlet", type=Path, default=root.parent / "IRMF-Intrinsic-Robust-Multiscale-Filtering")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    q = load_module(args.quantlet / "Simulations/imd_python_fixedvf.py", "quantlet_simulation")
    local = notebook_functions(root / "experiments/gd-irmf/gd_irmf.ipynb")
    result = {
        "scope": "Small deterministic diagnostics, not the full published simulation or a statistical ranking.",
        "commits": {str(p): subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
                    for p in [root, args.quantlet]},
        "numpy": np.__version__,
    }

    n, radius, H = 256, 50, 0.4
    grid = np.arange(n) / n
    clean = 1.2 * np.sin(2 * np.pi * grid) + 0.5 * np.sin(6 * np.pi * grid)
    rng = np.random.default_rng(20260917)
    observed = clean + rng.normal(0, 0.2, n)
    observed += (rng.random(n) < 0.2) * rng.exponential(0.6, n) * rng.choice([-1, 1], n)
    h = radius / n
    offsets = np.arange(-radius, radius + 1)
    w_ep = q.epanechnikov(offsets / radius)
    w_ep /= w_ep.sum()
    w_sq = local["epanechnikov_weights"](2 * radius + 1)
    windows = sliding_window_view(np.pad(observed, radius, mode="wrap"), 2 * radius + 1)

    def upstream_fit(y, bandwidth=h, loss_scale=H):
        extended_grid, extended_values = q.periodic_extend(y, grid)
        return q.weighted_minimizer(
            grid, extended_grid,
            lambda a, active: q.rho_H(extended_values[active] - a, loss_scale),
            bandwidth, np.min(y) - 4 * loss_scale, np.max(y) + 4 * loss_scale,
        )

    scalar = upstream_fit(observed)
    fitted_ep, trace = local["robust_gd_fit_windows"](windows, w_ep, H, return_trace=True)
    fitted_sq = local["robust_gd_fit_windows"](windows, w_sq, H)
    score_residual = np.sum(w_ep * q.psi_H(windows - fitted_ep[:, None], H), axis=1)
    solver_error = float(np.max(np.abs(scalar - fitted_ep)))
    kernel_rmse = float(np.sqrt(np.mean((fitted_ep - fitted_sq) ** 2)))
    assert solver_error < 1e-4, solver_error

    timings = {}
    for label, fn in [("golden_seconds", lambda: upstream_fit(observed)),
                      ("gd_seconds", lambda: local["robust_gd_fit_windows"](windows, w_ep, H))]:
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        timings[label] = float(np.median(samples))
    result["matched_solvers"] = {
        "n": n, "radius": radius, "H": H, "seed": 20260917,
        "max_abs_difference": solver_error, "gd_iterations": len(trace),
        "max_abs_exact_score_residual": float(np.max(np.abs(score_residual))),
        "kernel_change_rmse": kernel_rmse,
        "kernel_change_max_abs": float(np.max(np.abs(fitted_ep - fitted_sq))),
        "timing_note": "Three repeats, single thread; GD windows already constructed; no population diagnostics in either timed call.",
        **timings,
    }
    z = np.linspace(-8, 8, 20001)
    result["score_approximation"] = {
        "direct_max_error": float(np.max(np.abs(local["exact_score"](z, 1) - q.psi_H(z, 1)))),
        "lookup_max_error": float(np.max(np.abs(local["lookup_score"](z, 1, local["lookup_grid"]) - q.psi_H(z, 1)))),
    }

    # A broad local step exposes the finite iteration cap at small H.
    stress_window = np.where(np.abs(offsets) <= 20, 10.0, 0.0)[None, :]
    stress_gd, stress_trace = local["robust_gd_fit_windows"](stress_window, w_sq, 0.1, return_trace=True)
    stress_scalar = q.golden_section_minimize(
        lambda a: float(np.sum(w_sq * q.rho_H(stress_window[0] - a, 0.1))), 0, 10,
    )
    result["gd_iteration_cap_control"] = {
        "window": "101 points; central 41 equal 10, remaining 60 equal 0; squared-triangular weights",
        "H": 0.1, "default_iterations": len(stress_trace),
        "gd_fit": float(stress_gd[0]), "scalar_fit": stress_scalar,
        "last_update": float(stress_trace[-1]),
        "stopping_threshold": float(1e-6 * (1 + np.abs(stress_gd[0]))),
    }

    # Match all three bandwidths exactly, including noninteger support radii.
    scalar_residual, gd_residual = observed.copy(), observed.copy()
    stage_errors = []
    for bandwidth in h / np.sqrt(2.0) ** np.arange(3):
        scalar_component = upstream_fit(scalar_residual, bandwidth=bandwidth)
        halfwidth = int(np.ceil(n * bandwidth))
        stage_weights = q.epanechnikov(np.arange(-halfwidth, halfwidth + 1) / (n * bandwidth))
        stage_windows = sliding_window_view(np.pad(gd_residual, halfwidth, mode="wrap"), 2 * halfwidth + 1)
        gd_component = local["robust_gd_fit_windows"](stage_windows, stage_weights, H)
        stage_errors.append(float(np.max(np.abs(scalar_component - gd_component))))
        scalar_residual -= scalar_component
        gd_residual -= gd_component
    assert max(stage_errors) < 1e-4, stage_errors
    result["matched_recursion_stage_max_errors"] = stage_errors

    # Exact discrete spectra on a nonduplicated periodic grid.
    spectra = {}
    for name, weights in [("epanechnikov", w_ep), ("squared_triangular", w_sq)]:
        impulse = np.zeros(n)
        impulse[offsets % n] = weights
        eigenvalues = np.fft.fft(impulse).real
        spectra[name] = eigenvalues
        result[name] = {
            "min_eigenvalue": float(eigenvalues.min()),
            "residual_operator_norm": float(np.max(np.abs(1 - eigenvalues))),
            "sum_squared_weights": float(np.sum(weights ** 2)),
            "effective_sample_size": float(1 / np.sum(weights ** 2)),
        }
    assert result["epanechnikov"]["min_eigenvalue"] < 0

    # The upstream generator includes both 0 and 1, then extends with period 1.
    # Construct its exact linear weight matrices to expose duplicate coordinates.
    def weight_matrix(t, bandwidth):
        ext_t, ext_idx = q.periodic_extend(np.arange(len(t)), t)
        unnormalized = q.epanechnikov((t[:, None] - ext_t) / bandwidth) / bandwidth
        weights = unnormalized / unnormalized.sum(axis=1, keepdims=True)
        matrix = np.zeros((len(t), len(t)))
        for column, original in enumerate(ext_idx.astype(int)):
            matrix[:, original] += weights[:, column]
        return matrix, unnormalized.sum(axis=1)

    endpoint_results = {}
    for name, t in [("inclusive", np.arange(65) / 64), ("exclusive", np.arange(64) / 64)]:
        W1, mass = weight_matrix(t, 0.2)
        W2, _ = weight_matrix(t, 0.2 / np.sqrt(2))
        endpoint_results[name] = {
            "n_points": len(t), "kernel_mass_min": float(mass.min()),
            "kernel_mass_max": float(mass.max()),
            "commutator_frobenius_norm": float(np.linalg.norm(W1 @ W2 - W2 @ W1)),
        }
    result["periodic_grid"] = endpoint_results

    # A Gaussian population objective can be computed exactly by widening H.
    clean_fit = upstream_fit(clean, loss_scale=0.1)
    gaussian_population_fit = upstream_fit(clean, loss_scale=np.hypot(0.1, 0.4))
    result["gaussian_population_vs_clean"] = {
        "H": 0.1, "sigma": 0.4,
        "rmse": float(np.sqrt(np.mean((clean_fit - gaussian_population_fit) ** 2))),
        "max_abs_difference": float(np.max(np.abs(clean_fit - gaussian_population_fit))),
    }
    result["constant_signal_zero_replacement_targets"] = [
        {"clean_value": 1.0, "p": p, "H": loss_scale,
         "population_target": q.golden_section_minimize(
             lambda s: float((1 - p) * q.rho_H(1 - s, loss_scale) + p * q.rho_H(-s, loss_scale)), 0, 1, tol=1e-10)}
        for p in [0.2, 0.4] for loss_scale in [0.1, 1.0]
    ]

    # Exercise the actual multi-path routine and its six linearizations.
    for paths in [1, 8]:
        data = q.generate_zero_contaminated_signal(n=48, n_paths=paths, seed=2026, type=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            simulation = q.imd_decompose_with_linearization(data["Y"], data["x_grid"], data["true_signal"],
                                                          K_steps=3, H=1, progress_every=0)
        Y = np.asarray(data["Y"]).reshape(49, paths)
        reconstruction = simulation["tilde_S"].sum(axis=2) + simulation["Y_final"]
        reconstruction_error = float(np.max(np.abs(reconstruction - Y)))
        assert reconstruction_error < 1e-12
        result[f"upstream_simulation_m{paths}"] = {
            "shape": list(simulation["tilde_S"].shape),
            "reconstruction_max_error": reconstruction_error,
            "component_error_max": float(np.max(np.abs(simulation["linear"]["tilde_error"]))),
            "summary": [{name: float(row[name]) for name in simulation["linear"]["summary"].dtype.names}
                        for row in simulation["linear"]["summary"]],
        }
        assert np.isfinite(simulation["tilde_S"]).all()

    # Plot measured distinctions; this is not a method-performance leaderboard.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    u = np.linspace(-1, 1, 501)
    axes[0].plot(u, q.epanechnikov(u), label="QuantLet: Epanechnikov")
    axes[0].plot(u, 1.5 * (1 - np.abs(u)) ** 2, label="imf: squared triangular")
    axes[0].set(title="Different normalized kernels", xlabel="Offset / bandwidth", ylabel="Kernel weight")
    axes[0].legend(fontsize=8)
    axes[1].plot(grid, fitted_ep - scalar, label="GD minus scalar, same kernel")
    axes[1].plot(grid, fitted_sq - fitted_ep, label="Changing only the kernel")
    axes[1].set(title="One-stage fit differences", xlabel="Time", ylabel="Difference in fitted value")
    axes[1].legend(fontsize=8)
    for name, values in spectra.items():
        axes[2].plot(np.arange(25), values[:25], marker=".", label=name.replace("_", " "))
    axes[2].axhline(0, color="grey", linewidth=0.7)
    axes[2].set(title="Discrete smoothing eigenvalues", xlabel="Fourier mode", ylabel="Eigenvalue")
    axes[2].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)

    # Strict JSON: undefined correlations in the m=1 control become null.
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    (output / "results.json").write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(finite_json({k: v for k, v in result.items() if not k.startswith("upstream_simulation")}), indent=2))
    print("Full checks, including linearization summaries, saved to", output / "results.json")


if __name__ == "__main__":
    main()
