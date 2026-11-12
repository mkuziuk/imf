"""New tags, endpoint-safe schedules, and exact cache derivations."""

import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pimf

from emdirmf.decompose import H1, SQRT2, emd_components, noise_sigma_hat, scale_order
from emdirmf.irmf_newton import NewtonSmoothAbs

NATIVE_K = (3, 4, 5, 6, 9, 13)
EMD_K = (3, 4, 5, 6, 7, 8)
TRUNC_K = (4, 8, 12)
DEFAULTS = {"linear": "imf_linear_a1.41", "irmf": "irmf_h2_a1.41_conv"}
BASE_TAGS = ("raw", "emd", *DEFAULTS.values(), "irmf_h2_a1.41")


@dataclass(frozen=True)
class Spec:
    tag: str
    family: str
    construction: str
    k: int | None = None


def parse_tag(tag):
    if tag in BASE_TAGS:
        family = {"raw": "raw", "emd": "emd", "imf_linear_a1.41": "linear",
                  "irmf_h2_a1.41_conv": "irmf", "irmf_h2_a1.41": "gd60"}[tag]
        return Spec(tag, family, "default")
    match = re.fullmatch(r"(linear|irmf)_native_k(3|4|5|6|9|13)", tag)
    if match:
        return Spec(tag, match[1], "native", int(match[2]))
    match = re.fullmatch(r"emd_k(3|4|5|6|7|8)", tag)
    if match:
        return Spec(tag, "emd", "fold", int(match[1]))
    match = re.fullmatch(r"(linear|irmf)_trunc_k(4|8|12)", tag)
    if match:
        return Spec(tag, match[1], "trunc", int(match[2]))
    if tag == "gd60_trunc_k8":
        return Spec(tag, "gd60", "trunc", 8)
    raise ValueError(f"unknown components tag {tag!r}")


def all_tags():
    return [*BASE_TAGS, *[f"{f}_native_k{k}" for f in ("linear", "irmf") for k in NATIVE_K],
            *[f"emd_k{k}" for k in EMD_K if k != 8],
            *[f"{f}_trunc_k{k}" for f in ("linear", "irmf") for k in TRUNC_K],
            "gd60_trunc_k8"]


@lru_cache(maxsize=None)
def native_ratio(length, k):
    """Round a_K upward only until repeated division reaches pimf's endpoint snap."""
    if length not in (1024, 2048) or k not in NATIVE_K:
        raise ValueError("length or K outside pre-registered grid")
    ratio = (H1 * length / 1.5) ** (1.0 / (k - 1))
    for _ in range(16):
        h = H1
        for _ in range(k - 1):
            h /= ratio
        if h <= 1.5 / length:
            break
        ratio = float(np.nextafter(ratio, np.inf))
    result = pimf.IMF(pimf.Quadratic()).decompose(
        np.zeros(length), h1=H1, a=ratio, h_min=1.5 / length, k_max=None)
    assert len(result.stages) == k
    assert result.bandwidths[-1] == 1.5 / length
    return ratio


def native_components(y, family, k):
    y = np.asarray(y, dtype=np.float64)
    length = len(y)
    ratio = native_ratio(length, k)
    cpu, wall = time.process_time(), time.perf_counter()
    attempts = []
    for cap in (100, 1000):
        contrast = pimf.Quadratic() if family == "linear" else NewtonSmoothAbs(
            2 * noise_sigma_hat(y)[0], root_max_iter=cap)
        error = None
        try:
            result = pimf.IMF(contrast, workers=1).decompose(
                y, h1=H1, a=ratio, k_max=None, h_min=1.5 / length)
        except RuntimeError as exc:
            error = str(exc)
        diagnostics = getattr(contrast, "diagnostics", [])
        cap_stages = [i for i, d in enumerate(diagnostics) if d["iterations_max"] >= cap]
        attempts.append({"cap": cap, "cap_stages": cap_stages, "error": error,
                         "diagnostics": diagnostics})
        if error is None and not cap_stages:
            break
        if cap == 1000:
            raise RuntimeError(f"unaccepted Newton fit after retry: {attempts}")
    assert len(result.stages) == k, (length, k, result.bandwidths)
    assert result.bandwidths[-1] * length == 1.5
    stack = np.concatenate([result.imfs[::-1], result.residual[None]], axis=0)
    assert np.all(np.isfinite(stack))
    return stack, {"cpu_seconds": time.process_time() - cpu,
                   "seconds": time.perf_counter() - wall, "a": ratio, "k": k,
                   "last_radius": result.bandwidths[-1] * length, "attempts": attempts}


def fold_emd(stack, k):
    """Fold cached fine-to-coarse slots K..8 into the residue, in float64."""
    stack = np.asarray(stack, dtype=np.float64)
    if stack.shape[-2] != 9 or k not in EMD_K:
        raise ValueError("expected a 9-slot EMD cache and registered K")
    return np.concatenate([stack[..., :k, :], stack[..., k:, :].sum(axis=-2, keepdims=True)], axis=-2)


def truncate_imf(stack, k, source_tag):
    """Keep the first coarse pimf stages, then the folded fine residual."""
    stack = np.asarray(stack, dtype=np.float64)
    order = scale_order(source_tag, stack.shape[-1])
    if len(order) != stack.shape[-2] or k >= len(order):
        raise ValueError("source shape or k_max mismatch")
    # scale_order includes the finest solver residual first. Last k ranks are
    # precisely the first k sequential pimf stages, stored fine to coarse.
    retained, folded = order[-k:], order[:-k]
    return np.concatenate([stack[..., retained, :],
                           stack[..., folded, :].sum(axis=-2, keepdims=True)], axis=-2)


def direct_truncation(y, family, k):
    contrast = pimf.Quadratic() if family == "linear" else (
        pimf.SmoothAbs(2 * noise_sigma_hat(y)[0], lookup=True) if family == "gd60"
        else NewtonSmoothAbs(2 * noise_sigma_hat(y)[0], root_max_iter=1000))
    result = pimf.IMF(contrast, workers=1).decompose(
        y, h1=H1, a=SQRT2, k_max=k, h_min=1.5 / len(y))
    return np.concatenate([result.imfs[::-1], result.residual[None]], axis=0)


def relative_error(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - b) /
                 max(np.linalg.norm(b), 1e-12))
