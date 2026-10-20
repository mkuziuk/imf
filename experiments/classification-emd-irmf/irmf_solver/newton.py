"""Independent, single-core safeguarded Newton solver for pimf 0.4 SmoothAbs.

The public contrast.solve hook preserves pimf's schedule, padding and weights.
The LUT score and its piecewise derivative are evaluated together in Numba.
No installed package or shared study module is modified.
"""

import numpy as np
import pimf
from numba import njit
from pimf.contrasts import _SCORE, _SCORE_DELTA


@njit(cache=True)
def _fit_rows(windows, weights, h, score_table, slope_table, tol, max_iter):
    n_rows, width = windows.shape
    out = np.empty(n_rows)
    iterations = np.zeros(n_rows, dtype=np.int64)
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
        score = 0.0
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
                # Apply the last correction. Its size bounds the local error.
                x = min(upper, max(lower, x + step))
                done = True
                break
            if upper - lower <= tol * (1.0 + abs(x)):
                x = 0.5 * (lower + upper)
                done = True
                break
            candidate = x + step
            if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
                candidate = 0.5 * (lower + upper)
            # A periodic bisection guarantees bracket contraction even when
            # Newton repeatedly lands just inside one edge of a wide bracket.
            if iteration % 8 == 7:
                candidate = 0.5 * (lower + upper)
            x = candidate
        if not done:
            failed += 1
        max_score = max(max_score, abs(score))
        out[row] = x
    return out, iterations, failed, max_score


class NewtonSmoothAbs(pimf.SmoothAbs):
    """Lookup-compatible score roots, independent stopping for every window."""

    def __init__(self, h, *, root_tol=1e-12, root_max_iter=100):
        super().__init__(h, lookup=True)
        self.root_tol = root_tol
        self.root_max_iter = root_max_iter
        self.diagnostics = []

    def solve(self, windows, weights):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        values, iterations, failed, score = _fit_rows(
            windows, weights, self.H, _SCORE, _SCORE_DELTA,
            self.root_tol, self.root_max_iter,
        )
        self.diagnostics.append({
            "window_size": windows.shape[1],
            "iterations_mean": float(iterations.mean()),
            "iterations_max": int(iterations.max()),
            "failed_rows": int(failed),
            "last_score_max": float(score),
        })
        if failed:
            raise RuntimeError(f"Newton failed to converge for {failed} rows")
        return values


def newton_components(y, *, root_tol=1e-12):
    """Study settings and storage order; takes one already-normalized channel."""
    from emdirmf.decompose import H1, SQRT2, imf_h_min, noise_sigma_hat

    y = np.asarray(y, dtype=np.float64)
    sigma, source = noise_sigma_hat(y)
    contrast = NewtonSmoothAbs(2.0 * sigma, root_tol=root_tol)
    result = pimf.IMF(contrast, workers=1).decompose(
        y, h1=H1, a=SQRT2, k_max=None, h_min=imf_h_min(len(y)),
    )
    return np.vstack([result.imfs[::-1], result.residual[None]]), {
        "H": contrast.H, "sigma_source": source,
        "stages": contrast.diagnostics[::-1],
    }
