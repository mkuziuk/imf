"""Endpoint-safe native schedules and converged Newton decompositions."""
from functools import lru_cache

import numpy as np
import pimf

from emdirmf.decompose import H1, SQRT2, _imf_stack, imf_h_min, noise_sigma_hat
from emdirmf.irmf_newton import NewtonSmoothAbs

EMD_K = (2, 3, 4, 6, 8)
NATIVE_K = {2048: (3, 4, 6, 8, 11), 1024: (3, 4, 6, 8, 9)}
METHODS = ('raw', 'emd', 'linear', 'irmf')
WINDOWS = {
    2048: {3: (1025, 177, 31), 4: (1025, 317, 99, 31),
           6: (1025, 507, 251, 125, 63, 31), 8: (1025, 621, 375, 227, 139, 85, 51, 31),
           11: (1025, 721, 507, 357, 251, 177, 125, 89, 63, 45, 31)},
    1024: {3: (513, 125, 31), 4: (513, 201, 79, 31),
           6: (513, 293, 167, 95, 55, 31), 8: (513, 343, 229, 153, 103, 69, 47, 31),
           9: (513, 361, 253, 179, 125, 89, 63, 45, 31)},
}


def variants(length):
    return ('raw', *(f'emd_k{k}' for k in EMD_K),
            *(f'{m}_k{k}' for m in ('linear', 'irmf') for k in NATIVE_K[length]))


def parse_variant(variant):
    if variant == 'raw':
        return 'raw', 0
    method, k = variant.rsplit('_k', 1)
    if method not in METHODS:
        raise ValueError(variant)
    return method, int(k)


def n_slots(variant):
    return parse_variant(variant)[1] + 1


@lru_cache(maxsize=None)
def native_ratio(length, k):
    if k not in NATIVE_K[length]:
        raise ValueError((length, k))
    ratio = (H1 * length / 15.0) ** (1.0 / (k - 1))
    for _ in range(16):
        h = H1
        for _ in range(k - 1):
            h /= ratio
        if h <= 15.0 / length:
            break
        ratio = float(np.nextafter(ratio, np.inf))
    result = pimf.IMF(pimf.Quadratic(), workers=1).decompose(
        np.zeros(length), h1=H1, a=ratio, h_min=15.0 / length, k_max=None)
    assert tuple(result.window_sizes) == WINDOWS[length][k], (length, k, result.window_sizes)
    assert result.bandwidths[-1] == 15.0 / length
    return ratio


def newton_components(y, *, h_min=None, a=SQRT2, estimator=noise_sigma_hat,
                      root_max_iter=100, root_tol=1e-12, fixed_h=None):
    """Shared solver with an explicit endpoint, schedule and H estimator."""
    y = np.asarray(y, dtype=np.float64)
    sigma, source = estimator(y)
    contrast = NewtonSmoothAbs(2 * sigma if fixed_h is None else fixed_h, root_tol=root_tol, root_max_iter=root_max_iter)
    result = pimf.IMF(contrast, workers=1).decompose(
        y, h1=H1, a=a, k_max=None, h_min=imf_h_min(len(y)) if h_min is None else h_min)
    stages = contrast.diagnostics[::-1]
    return _imf_stack(result), dict(H=contrast.H, sigma_hat=sigma, sigma_source=source,
        stages=stages, stage_iterations=[s['iterations_max'] for s in stages],
        stage_converged=[s['converged'] for s in stages], windows=list(result.window_sizes))


def channel_stack(y, method, k, fixed_h):
    """Return float64 slots, cap-100 retry flag and failure text."""
    a = native_ratio(len(y), k)
    if method == 'linear':
        result = pimf.IMF(pimf.Quadratic(), workers=1).decompose(
            y, h1=H1, a=a, h_min=15.0 / len(y), k_max=None)
        return _imf_stack(result), False, ''
    hit = False
    for cap in (100, 1000):
        try:
            stack, info = newton_components(y, h_min=15.0 / len(y), a=a,
                                            fixed_h=fixed_h, root_max_iter=cap)
            if max(info['stage_iterations']) < cap and all(info['stage_converged']):
                assert stack.shape == (k + 1, len(y))
                return stack, hit, ''
            error = f'Newton reached cap {cap}'
        except RuntimeError as exc:
            if 'Newton failed to converge' not in str(exc):
                raise
            error = str(exc)
        hit = True
    return None, hit, error


def fold_emd(stack, k):
    stack = np.asarray(stack, dtype=np.float64)
    if stack.shape[-2] != 9 or k not in EMD_K:
        raise ValueError('expected 9 EMD slots and a registered K')
    return np.concatenate([stack[..., :k, :], stack[..., k:, :].sum(axis=-2, keepdims=True)], axis=-2)
