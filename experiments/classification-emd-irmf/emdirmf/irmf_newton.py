"""Single-threaded safeguarded roots of pimf 0.4.0's SmoothAbs lookup score.

The public contrast.solve hook leaves pimf's driver, weights and padding intact.
The module-level Numba dispatcher is compiled once per worker and cached on disk.
"""

from functools import lru_cache

import numpy as np
import pimf
from numba import njit
from pimf.contrasts import _SCORE, _SCORE_DELTA

SOLVER_NAME = "emdirmf.irmf_newton.NewtonSmoothAbs"
SOLVER_VERSION = "1.0.0"
PIMF_VERSION = "0.4.0"
ROOT_TOL = 1e-12
ROOT_MAX_ITER = 100
BISECTION_PERIOD = 8


def solver_params():
    """Numerical provenance included only in convergence-tag manifests."""
    return {
        "solver": SOLVER_NAME,
        "solver_version": SOLVER_VERSION,
        "driver": "pimf.IMF",
        "lookup_score_pimf_version": PIMF_VERSION,
        "root_tol": ROOT_TOL,
        "root_max_iter": ROOT_MAX_ITER,
        "root_stopping_rule": "abs(Newton step) or bracket width <= root_tol * (1 + abs(root))",
        "bisection_period": BISECTION_PERIOD,
    }


@njit(cache=True)
def _fit_rows(windows, weights, h, score_table, slope_table, tol, max_iter):
    n_rows, width = windows.shape
    out = np.empty(n_rows)
    iterations = np.zeros(n_rows, dtype=np.int64)
    bisections = np.zeros(n_rows, dtype=np.int64)
    failed = 0
    max_score = 0.0
    for row in range(n_rows):
        lower = np.inf
        upper = -np.inf
        x = 0.0
        for col in range(width):
            if weights[col] > 0:
                value = windows[row, col]
                lower = min(lower, value)
                upper = max(upper, value)
                x += weights[col] * value
        done = False
        for iteration in range(max_iter):
            score = 0.0
            derivative = 0.0
            for col in range(width):
                if weights[col] == 0:
                    continue
                standardized = (windows[row, col] - x) / h
                position = (min(8.0, max(-8.0, standardized)) + 8.0) * 256.0
                index = min(4096, int(position))
                fraction = position - index
                score += weights[col] * (score_table[index] + slope_table[index] * fraction)
                if -8.0 < standardized < 8.0:
                    derivative += weights[col] * slope_table[index] * 256.0 / h
            iterations[row] = iteration + 1
            if score > 0:
                lower = x
            elif score < 0:
                upper = x
            else:
                done = True
                break
            step = score / derivative if derivative > 0 else np.inf
            if derivative > 0 and abs(step) <= tol * (1.0 + abs(x)):
                x = min(upper, max(lower, x + step))
                done = True
                break
            if upper - lower <= tol * (1.0 + abs(x)):
                x = 0.5 * (lower + upper)
                done = True
                break
            candidate = x + step
            if (not np.isfinite(candidate) or candidate <= lower or candidate >= upper
                    or iteration % BISECTION_PERIOD == BISECTION_PERIOD - 1):
                candidate = 0.5 * (lower + upper)
                bisections[row] += 1
            x = candidate
        if not done:
            failed += 1
        # Measure the score at the returned root, including the final correction.
        score = 0.0
        for col in range(width):
            if weights[col] == 0:
                continue
            standardized = (windows[row, col] - x) / h
            position = (min(8.0, max(-8.0, standardized)) + 8.0) * 256.0
            index = min(4096, int(position))
            score += weights[col] * (score_table[index] + slope_table[index] * (position - index))
        max_score = max(max_score, abs(score))
        out[row] = x
    return out, iterations, bisections, failed, max_score


@lru_cache(maxsize=1)
def warm_up():
    """Compile/load the same strided readonly-window signature used by pimf."""
    windows = np.lib.stride_tricks.sliding_window_view(np.zeros(7), 3)
    _fit_rows(windows, np.ones(3) / 3, 1.0, _SCORE, _SCORE_DELTA, ROOT_TOL, ROOT_MAX_ITER)


def initialize_worker():
    """Process-pool initializer for convergence tags only."""
    from .parallel import limit_threads

    limit_threads()
    warm_up()


class NewtonSmoothAbs(pimf.SmoothAbs):
    """Lookup-compatible score roots, independent stopping for every window."""

    def __init__(self, h, *, root_tol=ROOT_TOL, root_max_iter=ROOT_MAX_ITER):
        if pimf.__version__ != PIMF_VERSION:
            raise RuntimeError(f"Newton lookup solver requires pimf {PIMF_VERSION}")
        super().__init__(h, lookup=True)
        if not np.isfinite(root_tol) or root_tol <= 0:
            raise ValueError("root_tol must be finite and positive")
        if not isinstance(root_max_iter, (int, np.integer)) or root_max_iter < 1:
            raise ValueError("root_max_iter must be a positive integer")
        self.root_tol = float(root_tol)
        self.root_max_iter = int(root_max_iter)
        self.diagnostics = []

    def solve(self, windows, weights):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        values, iterations, bisections, failed, score = _fit_rows(
            windows, weights, self.H, _SCORE, _SCORE_DELTA,
            self.root_tol, self.root_max_iter,
        )
        self.diagnostics.append({
            "window_size": windows.shape[1],
            "iterations_mean": float(iterations.mean()),
            "iterations_max": int(iterations.max()),
            "bisection_fallbacks": int(bisections.sum()),
            "bisection_fallbacks_max": int(bisections.max()),
            "failed_rows": int(failed),
            "last_score_max": float(score),
            "converged": failed == 0,
        })
        if failed:
            raise RuntimeError(f"Newton failed to converge for {failed} rows at window {windows.shape[1]}")
        return values


def newton_components(y, *, h_ratio=2.0, a=None, root_tol=ROOT_TOL, root_max_iter=ROOT_MAX_ITER):
    """Shared study settings and storage order for an already normalized channel.

    The H/stages keys retain the audit prototype's public return format.
    Additional stage arrays feed the shared cache metadata.
    """
    from .decompose import H1, SQRT2, _imf_stack, imf_h_min, noise_sigma_hat

    y = np.asarray(y, dtype=np.float64)
    sigma, source = noise_sigma_hat(y)
    contrast = NewtonSmoothAbs(h_ratio * sigma, root_tol=root_tol, root_max_iter=root_max_iter)
    result = pimf.IMF(contrast, workers=1).decompose(
        y, h1=H1, a=SQRT2 if a is None else a, k_max=None, h_min=imf_h_min(len(y)),
    )
    stages = contrast.diagnostics[::-1]
    return _imf_stack(result), {
        "H": contrast.H,
        "sigma_hat": sigma,
        "sigma_source": source,
        "n_components": len(stages),
        "stages": stages,
        "stage_iterations": [stage["iterations_max"] for stage in stages],
        "stage_iterations_mean": [stage["iterations_mean"] for stage in stages],
        "stage_bisection_fallbacks": [stage["bisection_fallbacks"] for stage in stages],
        "stage_bisection_fallbacks_max": [stage["bisection_fallbacks_max"] for stage in stages],
        "stage_max_score_residual": [stage["last_score_max"] for stage in stages],
        "stage_converged": [stage["converged"] for stage in stages],
    }
