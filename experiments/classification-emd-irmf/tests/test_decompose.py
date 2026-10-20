import math

import numpy as np
import pimf
import pytest
from conftest import multiscale_signal

from emdirmf import decompose as dec

METHODS = ["emd", "imf_linear_a1.41", "irmf_h2_a1.41", "imf_linear_a2", "irmf_h1_a2"]


def test_parse_method_tags():
    assert dec.parse_method("irmf").tag == "irmf_h2_a1.41"
    assert dec.parse_method("imf_linear").tag == "imf_linear_a1.41"
    spec = dec.parse_method("irmf_h4_a2")
    assert (spec.family, spec.h_ratio, spec.a) == ("irmf", 4.0, 2.0)
    assert math.isclose(dec.parse_method("irmf_h1_a1.41").a, math.sqrt(2))
    assert dec.parse_method("irmf_h2_a1.41_it60").tag == "irmf_h2_a1.41"
    spec = dec.parse_method("irmf_h2_a1.41_it300")
    assert spec.tag == "irmf_h2_a1.41_it300" and spec.max_iter == spec.params()["max_iter"] == 300
    with pytest.raises(ValueError):
        dec.parse_method("irmf_h2")
    with pytest.raises(ValueError):
        dec.parse_method("imf_linear_a1")


def test_normalize_frame_and_guards():
    rng = np.random.default_rng(0)
    x = 5 + 3 * rng.normal(size=4001)
    y, log_scale, source = dec.normalize_frame(x, return_source=True)
    assert abs(np.median(y)) < 1e-12
    assert math.isclose(1.4826 * np.median(np.abs(y)), 1.0, rel_tol=1e-9)
    assert math.isclose(log_scale, math.log(1.4826 * np.median(np.abs(x - np.median(x)))))
    assert source == dec.SCALE_MAD
    sparse = np.zeros(100)
    sparse[[3, 50]] = [1.0, -2.0]
    y, log_scale, source = dec.normalize_frame(sparse, return_source=True)
    assert source == dec.SCALE_STD and math.isclose(log_scale, math.log(np.std(sparse)))
    y, log_scale, source = dec.normalize_frame(np.full(64, 7.0), return_source=True)
    assert source == dec.SCALE_CONSTANT and log_scale == 0 and np.all(y == 0)
    batch = rng.normal(size=(3, 2, 128))
    y, log_scale = dec.normalize_frame(batch)
    assert y.shape == batch.shape and log_scale.shape == (3, 2)
    np.testing.assert_allclose(y[1, 0], dec.normalize_frame(batch[1, 0])[0])


@pytest.mark.parametrize("length", [256, 1000, 1024, 1536, 2048, 4096])
@pytest.mark.parametrize("a", [math.sqrt(2), 2.0])
def test_schedule_ends_at_three_or_five_taps(length, a):
    windows, bandwidths = dec.imf_schedule(length, a)
    radius = bandwidths[-1] * length
    assert 1.5 - 1e-9 <= radius < 1.5 * a
    assert windows[-1] in (5, 7) and min(windows) >= 5
    weights = pimf.squared_triangle.weights(windows[-1], bandwidth=radius)
    assert np.count_nonzero(weights) in (3, 5)
    assert all(w1 >= w2 for w1, w2 in zip(windows, windows[1:]))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("length", [1000, 1024])
def test_exact_reconstruction(method, length):
    y, _ = dec.normalize_frame(multiscale_signal(length, seed=length))
    components, info = dec.decompose_signal(y, method)
    assert components.shape == (dec.n_slots(method, length), length)
    assert np.max(np.abs(components.sum(axis=0) - y)) < 1e-10
    assert info["n_components"] <= components.shape[0] - 1


@pytest.mark.parametrize("method", ["imf_linear_a1.41", "irmf_h2_a1.41"])
def test_pimf_slots_are_reversed_stages(method):
    spec = dec.parse_method(method)
    y, _ = dec.normalize_frame(multiscale_signal(512, seed=3))
    components, info = dec.decompose_signal(y, spec)
    contrast = pimf.Quadratic() if spec.family == "imf_linear" else pimf.SmoothAbs(info["H"])
    result = pimf.IMF(contrast).decompose(y, h1=0.25, a=spec.a, k_max=None, h_min=dec.imf_h_min(512))
    np.testing.assert_array_equal(components[:-1], result.imfs[::-1])
    np.testing.assert_array_equal(components[-1], result.residual)
    sizes = dec.slot_window_sizes(spec, 512)
    assert sizes == sorted(sizes) and sizes[0] in (5, 7)
    assert dec.slot_labels(spec, 512)[-1] == "residual_fine"
    assert dec.scale_order(spec, 512)[0] == components.shape[0] - 1


def _centroid(x):
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return np.sum(np.fft.rfftfreq(len(x)) * spectrum) / spectrum.sum()


@pytest.mark.parametrize("method", ["emd", "imf_linear_a1.41", "irmf_h2_a1.41"])
def test_fine_to_coarse_order(method):
    """White noise: spectral centroid falls along scale_order.

    EMD: kept IMFs only (the residue is a trend whose centroid is ill-defined).
    IMF methods: the fine residual first, then every stage from window 5 upwards.
    """
    rng = np.random.default_rng(7)
    y, _ = dec.normalize_frame(rng.normal(size=2048))
    components, info = dec.decompose_signal(y, method)
    kept = info["n_components"]
    order = dec.scale_order(method, 2048)
    order = order[:kept] if method == "emd" else order
    centroids = [_centroid(components[slot]) for slot in order]
    assert all(c1 > c2 for c1, c2 in zip(centroids, centroids[1:])), np.round(centroids, 4)


def test_emd_padding_records_count():
    t = np.arange(1024) / 1024
    y, _ = dec.normalize_frame(np.sin(2 * np.pi * 8 * t) + 0.5 * t)
    components, info = dec.emd_components(y)
    assert components.shape == (dec.K_EMD + 1, 1024)
    assert 1 <= info["n_components"] < dec.K_EMD and info["n_imfs"] == info["n_components"]
    assert np.all(components[info["n_components"] : dec.K_EMD] == 0)
    assert np.max(np.abs(components.sum(axis=0) - y)) < 1e-10


def test_emd_folds_imfs_beyond_eight():
    from PyEMD import EMD

    rng = np.random.default_rng(11)
    y, _ = dec.normalize_frame(rng.normal(size=4096))
    components, info = dec.emd_components(y)
    solver = EMD()
    solver.emd(y)
    imfs, residue = solver.get_imfs_and_residue()
    assert info["n_imfs"] == len(imfs) > dec.K_EMD and info["n_components"] == dec.K_EMD
    np.testing.assert_array_equal(components[: dec.K_EMD], imfs[: dec.K_EMD])
    np.testing.assert_allclose(components[dec.K_EMD], imfs[dec.K_EMD :].sum(axis=0) + residue, atol=1e-10)


def test_emd_constant_frame():
    components, info = dec.emd_components(np.zeros(256))
    assert info["n_imfs"] == 0 and np.all(components == 0)


@pytest.mark.parametrize("method", ["emd", "imf_linear_a1.41", "irmf_h2_a1.41"])
def test_deterministic(method):
    frame = np.stack([multiscale_signal(512, seed=5), multiscale_signal(512, seed=6)])
    first, _ = dec.decompose_frame(frame, method)
    second, _ = dec.decompose_frame(frame, method)
    assert first.tobytes() == second.tobytes()


def test_irmf_bandwidth_rule():
    y, _ = dec.normalize_frame(multiscale_signal(1024, seed=2))
    _, info = dec.irmf_components(y, h_ratio=4.0)
    sigma = np.median(np.abs(np.diff(y))) / (0.6745 * math.sqrt(2))
    assert math.isclose(info["sigma_hat"], sigma) and math.isclose(info["H"], 4 * sigma)
    assert len(info["stage_iterations"]) == info["n_components"] == 15
    assert all(1 <= it <= 60 for it in info["stage_iterations"])


def test_sigma_hat_fallbacks():
    stairs = np.repeat(np.arange(8.0), 64)
    sigma, source = dec.noise_sigma_hat(stairs)
    assert source == 1 and sigma > 0
    assert dec.noise_sigma_hat(np.ones(32)) == (1.0, 2)


def test_decompose_frame_catches_failures():
    frame = np.stack([multiscale_signal(256), multiscale_signal(256, seed=9)])
    frame[1, 10] = np.nan
    components, infos = dec.decompose_frame(frame, "irmf", catch=True)
    assert infos[0]["ok"] and not infos[1]["ok"] and "non-finite" in infos[1]["error"]
    assert np.all(np.isfinite(components[0])) and np.all(np.isnan(components[1]))
    with pytest.raises(ValueError):
        dec.decompose_frame(frame, "irmf")
