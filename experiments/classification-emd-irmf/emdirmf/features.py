"""Per-component classical features (PROTOCOL.md, "Features").

Every method, and the raw-frame control, goes through the same functions. A
feature depends on its own frame only. Frequencies are normalized (cycles per
sample, 0 to 0.5), so values do not depend on fs.

Per component x of length L (mu = mean, sigma = std):

  energy_frac    mean(x^2) / sum over the frame-channel's slots of mean(x^2)
  log_energy     log(mean(x^2) + 1e-12)
  std, mean_abs  sigma, mean(|x|)
  skew, kurt     standardized 3rd moment, excess 4th moment
  crest          max|x - mu| / sigma
  zcr            crossings of the mean per sample
  extrema_rate   strict local extrema per sample
  hilb_amp_mean, hilb_amp_std   envelope |z| of the analytic signal z of x - mu
  hilb_if_mean, hilb_if_std     instantaneous frequency angle(z[t+1] conj z[t]) / 2 pi,
                                envelope-weighted mean and std
  spec_centroid, spec_bw        Hann-windowed periodogram of x - mu: centroid and spread
  spec_entropy                  normalized Shannon entropy of the periodogram
  spec_peak                     frequency of the periodogram maximum
  hjorth_mob, hjorth_cpx        Hjorth mobility and complexity
  perm_entropy                  permutation entropy, order 4, delay 1, normalized by log 24

A component with sigma < 1e-6 (relative to a unit robust frame scale), such as a
zero-padded EMD slot, is "absent": its energy features are kept and every shape
feature (skew through perm_entropy) is set to 0, so rounding noise cannot produce
arbitrary values.
"""

import re

import numpy as np
from scipy.signal import hilbert

from .decompose import normalize_frame

FEATURE_NAMES = (
    "energy_frac",
    "log_energy",
    "std",
    "mean_abs",
    "skew",
    "kurt",
    "crest",
    "zcr",
    "extrema_rate",
    "hilb_amp_mean",
    "hilb_amp_std",
    "hilb_if_mean",
    "hilb_if_std",
    "spec_centroid",
    "spec_bw",
    "spec_entropy",
    "spec_peak",
    "hjorth_mob",
    "hjorth_cpx",
    "perm_entropy",
)
SHAPE_FEATURES = FEATURE_NAMES[4:]
STD_FLOOR = 1e-6
ENERGY_EPS = 1e-12
PE_ORDER = 4
_PE_PAIRS = [(i, j) for i in range(PE_ORDER) for j in range(i + 1, PE_ORDER)]
_PE_LOG_NORM = np.log(24.0)
_COMPONENT_COLUMN = re.compile(r"^ch\d+_c\d+_")


def _safe_divide(numerator, denominator):
    out = np.zeros(np.broadcast(numerator, denominator).shape)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _permutation_entropy(x):
    """Order-4 permutation entropy per row via the 6 pairwise order comparisons."""
    rows, length = x.shape
    count = length - PE_ORDER + 1
    codes = np.zeros((rows, count), dtype=np.int64)
    for bit, (i, j) in enumerate(_PE_PAIRS):
        codes |= (x[:, i : i + count] < x[:, j : j + count]).astype(np.int64) << bit
    n_codes = 1 << len(_PE_PAIRS)
    codes += (np.arange(rows) * n_codes)[:, None]
    histogram = np.bincount(codes.ravel(), minlength=rows * n_codes).reshape(rows, n_codes)
    p = histogram / count
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return entropy / _PE_LOG_NORM


def component_features(x):
    """Features for each row of x (M, L); returns (M, len(FEATURE_NAMES)).

    energy_frac is set to 1 here (each row its own group); frame-level callers
    replace it with the within-frame fraction.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (M, L)")
    rows, length = x.shape
    out = np.zeros((rows, len(FEATURE_NAMES)))
    column = {name: index for index, name in enumerate(FEATURE_NAMES)}

    power = np.mean(x**2, axis=1)
    mu = np.mean(x, axis=1, keepdims=True)
    centered = x - mu
    sigma = np.sqrt(np.mean(centered**2, axis=1))
    present = sigma >= STD_FLOOR
    sigma_safe = np.where(present, sigma, 1.0)
    standardized = centered / sigma_safe[:, None]

    out[:, column["energy_frac"]] = 1.0
    out[:, column["log_energy"]] = np.log(power + ENERGY_EPS)
    out[:, column["std"]] = sigma
    out[:, column["mean_abs"]] = np.mean(np.abs(x), axis=1)
    out[:, column["skew"]] = np.mean(standardized**3, axis=1)
    out[:, column["kurt"]] = np.mean(standardized**4, axis=1) - 3.0
    out[:, column["crest"]] = np.max(np.abs(standardized), axis=1)

    positive = centered >= 0
    out[:, column["zcr"]] = np.count_nonzero(positive[:, 1:] != positive[:, :-1], axis=1) / (length - 1)
    d1 = np.diff(x, axis=1)
    out[:, column["extrema_rate"]] = np.count_nonzero(d1[:, 1:] * d1[:, :-1] < 0, axis=1) / (length - 2)

    analytic = hilbert(centered, axis=1)
    amplitude = np.abs(analytic)
    out[:, column["hilb_amp_mean"]] = amplitude.mean(axis=1)
    out[:, column["hilb_amp_std"]] = amplitude.std(axis=1)
    inst_freq = np.angle(analytic[:, 1:] * np.conj(analytic[:, :-1])) / (2 * np.pi)
    weight = 0.5 * (amplitude[:, 1:] + amplitude[:, :-1])
    weight_sum = weight.sum(axis=1)
    if_mean = _safe_divide(np.sum(weight * inst_freq, axis=1), weight_sum)
    if_var = _safe_divide(np.sum(weight * (inst_freq - if_mean[:, None]) ** 2, axis=1), weight_sum)
    out[:, column["hilb_if_mean"]] = if_mean
    out[:, column["hilb_if_std"]] = np.sqrt(np.maximum(if_var, 0.0))

    window = np.hanning(length)
    spectrum = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(length)
    total = spectrum.sum(axis=1)
    p = _safe_divide(spectrum, total[:, None])
    centroid = p @ freqs
    out[:, column["spec_centroid"]] = centroid
    out[:, column["spec_bw"]] = np.sqrt(np.maximum(np.sum(p * (freqs[None, :] - centroid[:, None]) ** 2, axis=1), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out[:, column["spec_entropy"]] = entropy / np.log(len(freqs))
    out[:, column["spec_peak"]] = freqs[np.argmax(spectrum, axis=1)]

    var0 = sigma**2
    var1 = np.var(d1, axis=1)
    var2 = np.var(np.diff(d1, axis=1), axis=1)
    mobility = np.sqrt(_safe_divide(var1, var0))
    mobility_d1 = np.sqrt(_safe_divide(var2, var1))
    out[:, column["hjorth_mob"]] = mobility
    out[:, column["hjorth_cpx"]] = _safe_divide(mobility_d1, mobility)

    out[:, column["perm_entropy"]] = _permutation_entropy(x)

    shape_columns = [column[name] for name in SHAPE_FEATURES]
    out[np.ix_(~present, shape_columns)] = 0.0
    nonfinite_rows = ~np.all(np.isfinite(x), axis=1)
    out[nonfinite_rows] = np.nan
    return out


def feature_names(n_channels, n_slots):
    return [
        f"ch{channel}_c{slot}_{name}"
        for channel in range(n_channels)
        for slot in range(n_slots)
        for name in FEATURE_NAMES
    ]


def log_scale_names(n_channels):
    return [f"ch{channel}_log_scale" for channel in range(n_channels)]


def frame_features(components):
    """Features of stacked components (n, C, S, L) -> (n, C * S * F) in feature_names order."""
    components = np.asarray(components, dtype=np.float64)
    n, n_channels, n_slots, length = components.shape
    flat = components.reshape(n * n_channels * n_slots, length)
    values = component_features(flat).reshape(n, n_channels, n_slots, len(FEATURE_NAMES))
    power = np.mean(components**2, axis=-1)
    total = power.sum(axis=-1, keepdims=True)
    values[..., FEATURE_NAMES.index("energy_frac")] = _safe_divide(power, total)
    return values.reshape(n, n_channels * n_slots * len(FEATURE_NAMES))


def raw_components(frames):
    """Normalized frames as single-component stacks (n, C, 1, L), plus log_scale (n, C)."""
    normalized, log_scale = normalize_frame(np.asarray(frames, dtype=np.float64))
    return normalized[:, :, None, :], log_scale


def feature_columns(columns, *, include_log_scale=False):
    """Select model inputs from a features table's columns."""
    selected = [name for name in columns if _COMPONENT_COLUMN.match(name)]
    if include_log_scale:
        selected += [name for name in columns if name.endswith("_log_scale")]
    return selected
