"""Per-signal decompositions compared in the study (PROTOCOL.md, "Decompositions").

Every method sees the same normalized frame and returns a float64 array of shape
(K + 1, L) in the canonical storage order:

- slots 0 .. K-1: extracted components, fine to coarse;
- slot K: the solver's residual.

For EMD (K = 8) the residual is the residue, i.e. the slowest content (trend plus
any IMFs beyond the eighth). For the IMF methods the extracted components are the
pimf stages reversed (pimf runs coarse to fine), so slot 0 is the smallest window
and slot K-1 the widest; the residual is what the smallest window left behind,
which is the *finest* content. `scale_order` gives the permutation that sorts all
slots strictly fine to coarse.

Schedule shared by both IMF methods: h1 = 0.25, h[k+1] = h[k] / a, k_max = None,
h_min = 1.5 / L. pimf's squared-triangle kernel gives zero weight at |offset| = h L,
so a stage with radius h L <= 1 sample is the identity (it would move the whole
residual into one component). h_min = 1.5 / L stops before that and makes the last
stage radius lie in [1.5, 1.5 a) samples: 3 or 5 nonzero kernel taps (pimf window
size 5 or 7) for both a = sqrt(2) and a = 2, for any L. For power-of-two L and
a = sqrt(2) the last radius is exactly 2 samples (window 5, taps 1/6, 2/3, 1/6).
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pimf

SQRT2 = math.sqrt(2.0)
MAD_TO_SIGMA = 1.4826
DIFF_MAD_TO_SIGMA = 0.6745 * SQRT2
K_EMD = 8
H1 = 0.25
H_MIN_RADIUS = 1.5
IRMF_MAX_ITER = 60
IRMF_TOL = 1e-6
MAIN_TAGS = ("emd", "imf_linear_a1.41", "irmf_h2_a1.41")
ABLATION_TAGS = ("irmf_h1_a1.41", "irmf_h4_a1.41", "irmf_h2_a2", "imf_linear_a2")
ALIASES = {"imf_linear": "imf_linear_a1.41", "irmf": "irmf_h2_a1.41"}

# Scale-normalization outcome codes (also stored in decomposition meta).
SCALE_MAD, SCALE_STD, SCALE_CONSTANT = 0, 1, 2

logging.getLogger("PyEMD").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Method tags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Parsed method tag. family is one of raw, emd, imf_linear, irmf."""

    tag: str
    family: str
    a: float | None = None
    h_ratio: float | None = None
    max_iter: int | None = None

    @property
    def is_decomposition(self):
        return self.family != "raw"

    def params(self):
        if self.family == "emd":
            return {"solver": "PyEMD.EMD", "settings": "defaults", "k_emd": K_EMD}
        if self.family in ("imf_linear", "irmf"):
            params = {
                "solver": "pimf.IMF",
                "contrast": "Quadratic" if self.family == "imf_linear" else "SmoothAbs",
                "kernel": "squared_triangle",
                "boundary": "wrap",
                "h1": H1,
                "a": self.a,
                "k_max": None,
                "h_min_rule": f"{H_MIN_RADIUS} / L",
                "workers": 1,
            }
            if self.family == "irmf":
                params.update(
                    h_ratio=self.h_ratio,
                    H_rule="h_ratio * median(|diff(y)|) / (0.6745 * sqrt(2)), y normalized",
                    lookup=True,
                    max_iter=self.max_iter,
                    tol=IRMF_TOL,
                )
            return params
        return {}


def _a_token(a):
    return "1.41" if math.isclose(a, SQRT2) else f"{a:g}"


def _parse_a(token):
    a = SQRT2 if token == "1.41" else float(token)
    if not a > 1:
        raise ValueError(f"schedule ratio a must exceed 1, got {token}")
    return a


def parse_method(tag):
    """Parse a method tag: raw, emd, imf_linear_a{A}, irmf_h{R}_a{A}[_it{N}].

    A = 1.41 stands for sqrt(2). imf_linear and irmf alias the main settings.
    The optional _it{N} sets the IRMF max_iter (default 60, the pimf default and
    the PROTOCOL setting, which is then omitted from the canonical tag).
    """
    tag = ALIASES.get(tag, tag)
    if tag in ("raw", "emd"):
        return MethodSpec(tag, tag)
    match = re.fullmatch(r"imf_linear_a([0-9.]+)", tag)
    if match:
        a = _parse_a(match.group(1))
        return MethodSpec(f"imf_linear_a{_a_token(a)}", "imf_linear", a=a)
    match = re.fullmatch(r"irmf_h([0-9.]+)_a([0-9.]+)(?:_it([0-9]+))?", tag)
    if match:
        h_ratio, a = float(match.group(1)), _parse_a(match.group(2))
        max_iter = IRMF_MAX_ITER if match.group(3) is None else int(match.group(3))
        if not h_ratio > 0 or max_iter < 1:
            raise ValueError("h_ratio and max_iter must be positive")
        suffix = "" if max_iter == IRMF_MAX_ITER else f"_it{max_iter}"
        return MethodSpec(f"irmf_h{h_ratio:g}_a{_a_token(a)}{suffix}", "irmf", a=a, h_ratio=h_ratio,
                          max_iter=max_iter)
    raise ValueError(f"unknown method tag {tag!r}")


def canonical_tag(tag):
    return parse_method(tag).tag


# ---------------------------------------------------------------------------
# Normalization and noise scale
# ---------------------------------------------------------------------------


def normalize_frame(x, *, return_source=False):
    """Robustly standardize along the last axis: (x - median) / (1.4826 * MAD).

    Zero guard: if MAD is zero the standard deviation is used instead; a constant
    signal is only centered (scale 1). Returns (normalized, log_scale) and, with
    return_source, the code SCALE_MAD / SCALE_STD / SCALE_CONSTANT. Works on any
    array shape; log_scale drops the last axis.
    """
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x, axis=-1, keepdims=True)
    centered = x - median
    scale = MAD_TO_SIGMA * np.median(np.abs(centered), axis=-1, keepdims=True)
    source = np.full(scale.shape, SCALE_MAD, dtype=np.int8)
    use_std = ~(scale > 0)
    if np.any(use_std):
        std = np.std(x, axis=-1, keepdims=True)
        scale = np.where(use_std, std, scale)
        source = np.where(use_std, SCALE_STD, source)
        constant = ~(scale > 0)
        scale = np.where(constant, 1.0, scale)
        source = np.where(constant, SCALE_CONSTANT, source).astype(np.int8)
    normalized = centered / scale
    log_scale = np.log(scale[..., 0])
    if return_source:
        return normalized, log_scale, source[..., 0]
    return normalized, log_scale


def noise_sigma_hat(y):
    """sigma_hat = median(|diff(y)|) / (0.6745 sqrt 2) with guards.

    Returns (sigma_hat, source): source 0 is the protocol estimate; 1 falls back to
    mean(|diff|) * sqrt(pi) / 2 (same Gaussian target) when the median is zero;
    2 means a constant signal, where sigma_hat = 1 is arbitrary but harmless.
    """
    d = np.abs(np.diff(np.asarray(y, dtype=np.float64)))
    sigma = float(np.median(d)) / DIFF_MAD_TO_SIGMA
    if sigma > 0:
        return sigma, 0
    sigma = float(np.mean(d)) * math.sqrt(math.pi) / 2.0
    if sigma > 0:
        return sigma, 1
    return 1.0, 2


# ---------------------------------------------------------------------------
# Schedules and slot bookkeeping
# ---------------------------------------------------------------------------


def imf_h_min(length):
    """Smallest bandwidth of the shared schedule: a radius of 1.5 samples."""
    return H_MIN_RADIUS / length


@lru_cache(maxsize=64)
def _schedule(length, a):
    result = pimf.IMF(pimf.Quadratic()).decompose(
        np.zeros(length), h1=H1, a=a, k_max=None, h_min=imf_h_min(length)
    )
    return tuple(result.window_sizes), tuple(result.bandwidths)


def imf_schedule(length, a=SQRT2):
    """(window_sizes, bandwidths) in pimf order (coarse to fine) for length L."""
    return _schedule(int(length), float(a))


def n_slots(method, length):
    """K + 1 storage slots for a method at frame length L (1 for raw)."""
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    if spec.family == "raw":
        return 1
    if spec.family == "emd":
        return K_EMD + 1
    return len(imf_schedule(length, spec.a)[0]) + 1


def slot_labels(method, length):
    """Human-readable label per storage slot."""
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    if spec.family == "raw":
        return ["frame"]
    if spec.family == "emd":
        return [f"imf{k + 1}" for k in range(K_EMD)] + ["residue_coarse"]
    windows = imf_schedule(length, spec.a)[0][::-1]
    return [f"w{size}" for size in windows] + ["residual_fine"]


def slot_window_sizes(method, length):
    """pimf window size per extracted slot (canonical order); None for EMD/raw."""
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    if spec.family in ("imf_linear", "irmf"):
        return list(imf_schedule(length, spec.a)[0][::-1])
    return None


def scale_order(method, length):
    """Permutation of storage slots that sorts them strictly fine to coarse.

    EMD storage is already fine to coarse. For the IMF methods the residual
    (finest content) moves to the front.
    """
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    count = n_slots(spec, length)
    if spec.family in ("imf_linear", "irmf"):
        return [count - 1] + list(range(count - 1))
    return list(range(count))


# ---------------------------------------------------------------------------
# Per-signal decompositions (input: one normalized channel)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _pyemd():
    from PyEMD import EMD

    return EMD()


def emd_components(y, k_emd=K_EMD):
    """PyEMD.EMD with default sifting; first k_emd IMFs, the rest folded into the residue.

    Returns ((k_emd + 1, L) array, info). Missing IMFs are zero rows; info records
    n_imfs (all IMFs PyEMD extracted) and n_components (kept, <= k_emd).
    """
    y = np.asarray(y, dtype=np.float64)
    solver = _pyemd()
    solver.emd(y)
    imfs, _ = solver.get_imfs_and_residue()
    n_imfs = int(imfs.shape[0])
    kept = min(n_imfs, k_emd)
    components = np.zeros((k_emd + 1, y.size))
    components[:kept] = imfs[:kept]
    components[k_emd] = y - imfs[:kept].sum(axis=0)
    return components, {"n_components": kept, "n_imfs": n_imfs}


def _imf_stack(result):
    return np.vstack([result.imfs[::-1], result.residual[None, :]])


def imf_linear_components(y, a=SQRT2):
    """Linear IMF (closed-form local weighted means) on the shared schedule."""
    y = np.asarray(y, dtype=np.float64)
    result = pimf.IMF(pimf.Quadratic()).decompose(
        y, h1=H1, a=a, k_max=None, h_min=imf_h_min(y.size)
    )
    return _imf_stack(result), {"n_components": len(result.stages)}


def irmf_components(y, h_ratio=2.0, a=SQRT2, max_iter=IRMF_MAX_ITER):
    """Robust IMF with SmoothAbs(H), H = h_ratio * sigma_hat, on the shared schedule.

    Stage diagnostics in info are listed in canonical slot order (fine to coarse).
    """
    y = np.asarray(y, dtype=np.float64)
    sigma_hat, sigma_source = noise_sigma_hat(y)
    H = h_ratio * sigma_hat
    solver = pimf.IMF(pimf.SmoothAbs(H, lookup=True), max_iter=max_iter, tol=IRMF_TOL, workers=1)
    result = solver.decompose(y, h1=H1, a=a, k_max=None, h_min=imf_h_min(y.size))
    stages = result.stages[::-1]
    info = {
        "n_components": len(stages),
        "sigma_hat": sigma_hat,
        "sigma_source": sigma_source,
        "H": H,
        "stage_iterations": [int(stage.iterations) for stage in stages],
        "stage_converged": [bool(stage.converged) for stage in stages],
        "stage_final_delta": [float(stage.final_max_delta) for stage in stages],
    }
    return _imf_stack(result), info


def decompose_signal(y, method):
    """Decompose one normalized channel; returns ((K + 1, L) float64, info dict)."""
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    if spec.family == "emd":
        return emd_components(y)
    if spec.family == "imf_linear":
        return imf_linear_components(y, a=spec.a)
    if spec.family == "irmf":
        return irmf_components(y, h_ratio=spec.h_ratio, a=spec.a, max_iter=spec.max_iter)
    if spec.family == "raw":
        y = np.asarray(y, dtype=np.float64)
        return y[None, :].copy(), {"n_components": 1}
    raise ValueError(f"unsupported method {spec.tag}")


def decompose_frame(frame, method, *, catch=False):
    """Normalize and decompose each channel of a (C, L) frame.

    Returns (components (C, K + 1, L) float64, list of per-channel info dicts).
    Each info carries log_scale, scale_source, seconds and cpu_seconds; with
    catch=True a failing channel becomes NaN with ok=False and the error text.
    """
    spec = method if isinstance(method, MethodSpec) else parse_method(method)
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim == 1:
        frame = frame[None, :]
    n_channels, length = frame.shape
    components = np.full((n_channels, n_slots(spec, length), length), np.nan)
    infos = []
    for channel in range(n_channels):
        wall, cpu = time.perf_counter(), time.process_time()
        info = {"channel": channel, "ok": True, "error": ""}
        try:
            if not np.all(np.isfinite(frame[channel])):
                raise ValueError("non-finite samples in frame")
            normalized, log_scale, source = normalize_frame(frame[channel], return_source=True)
            info.update(log_scale=float(log_scale), scale_source=int(source))
            stack, details = decompose_signal(normalized, spec)
            if not np.all(np.isfinite(stack)):
                raise FloatingPointError("non-finite components")
            components[channel] = stack
            info.update(details)
        except Exception as error:  # record and continue: one bad frame must not stop a batch
            if not catch:
                raise
            info.update(ok=False, error=f"{type(error).__name__}: {error}"[:500])
        info["seconds"] = time.perf_counter() - wall
        info["cpu_seconds"] = time.process_time() - cpu
        infos.append(info)
    return components, infos
