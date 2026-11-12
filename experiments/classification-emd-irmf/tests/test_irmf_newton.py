"""Converged roots, schedule compatibility and existing-cache regression checks."""

import json

import numpy as np
import pandas as pd
import pimf
import pytest
from conftest import multiscale_signal, write_prepared

from emdirmf import decompose as dec
from emdirmf.irmf_newton import NewtonSmoothAbs, _fit_rows, solver_params, warm_up
from emdirmf.paths import DEFAULT_DATA_ROOT
from emdirmf.run_decompose import load_decomposition, run_decompose
from emdirmf.run_features import run_features


@pytest.mark.parametrize("tag, canonical", [
    ("irmf_h2_a1.41_conv", "irmf_h2_a1.41_conv"),
    ("irmf_h2.0_a2.00_conv", "irmf_h2_a2_conv"),
    ("irmf_h1_a1.41_conv", "irmf_h1_a1.41_conv"),
    ("irmf_h4_a1.41_conv", "irmf_h4_a1.41_conv"),
    ("irmf_h2_a2_conv", "irmf_h2_a2_conv"),
])
def test_conv_tags_round_trip(tag, canonical):
    spec = dec.parse_method(tag)
    assert spec.tag == canonical and spec.converged and spec.max_iter is None
    assert dec.parse_method(spec.tag) == spec
    assert dec.canonical_tag(tag) == canonical
    params = spec.params()
    assert params.items() >= solver_params().items()
    assert "max_iter" not in params and "tol" not in params
    assert json.loads(json.dumps(params)) == params


@pytest.mark.parametrize("tag", [
    "irmf_h2_a1.41_conv_it60", "irmf_h2_a1.41_it60_conv",
    "imf_linear_a2_conv", "irmf_h0_a2_conv",
])
def test_conv_and_iteration_suffix_are_exclusive(tag):
    with pytest.raises(ValueError):
        dec.parse_method(tag)


@pytest.mark.parametrize("tag", [
    "irmf_h2_a1.41_conv", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv", "irmf_h2_a2_conv",
])
@pytest.mark.parametrize("length", [256, 1000, 1024])
def test_conv_reconstruction_and_stage_order(tag, length):
    y, _ = dec.normalize_frame(multiscale_signal(length, seed=18))
    components, info = dec.decompose_signal(y, tag)
    gd_tag = tag.removesuffix("_conv")
    assert components.shape == (dec.n_slots(gd_tag, length), length)
    np.testing.assert_allclose(components.sum(axis=0), y, atol=1e-12, rtol=1e-12)
    assert dec.slot_labels(tag, length) == dec.slot_labels(gd_tag, length)
    assert dec.scale_order(tag, length) == dec.scale_order(gd_tag, length)
    assert info["n_components"] == len(dec.imf_schedule(length, dec.parse_method(tag).a)[0])
    assert all(info["stage_converged"])
    for key in ("stage_iterations", "stage_iterations_mean", "stage_bisection_fallbacks",
                "stage_max_score_residual"):
        assert len(info[key]) == info["n_components"]
    # A direct pimf solve-hook run verifies both array order and stage diagnostics.
    spec = dec.parse_method(tag)
    contrast = NewtonSmoothAbs(info["H"])
    result = pimf.IMF(contrast, workers=1).decompose(
        y, h1=dec.H1, a=spec.a, k_max=None, h_min=dec.imf_h_min(length),
    )
    np.testing.assert_array_equal(components[:-1], result.imfs[::-1])
    assert [stage["window_size"] for stage in contrast.diagnostics[::-1]] == dec.slot_window_sizes(tag, length)
    assert info["stage_iterations"] == [stage["iterations_max"] for stage in contrast.diagnostics[::-1]]


@pytest.mark.parametrize("h_ratio, a", [(1.0, dec.SQRT2), (2.0, dec.SQRT2), (4.0, dec.SQRT2), (2.0, 2.0)])
@pytest.mark.parametrize("seed", [4, 19])
def test_conv_agrees_with_tightly_converged_pimf(h_ratio, a, seed):
    y, _ = dec.normalize_frame(multiscale_signal(128, seed=seed, impulses=False))
    sigma, _ = dec.noise_sigma_hat(y)
    reference = pimf.IMF(pimf.SmoothAbs(h_ratio * sigma, lookup=True),
                         max_iter=20000, tol=1e-12, workers=1).decompose(
        y, h1=dec.H1, a=a, k_max=None, h_min=dec.imf_h_min(len(y)),
    )
    assert all(stage.converged for stage in reference.stages)
    expected = dec._imf_stack(reference)
    actual, _ = dec.irmf_components(y, h_ratio=h_ratio, a=a, converged=True)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) <= 1e-8
    relative_per_slot = np.linalg.norm(actual - expected, axis=1) / np.maximum(
        np.linalg.norm(expected, axis=1), 1e-12,
    )
    assert relative_per_slot.max() <= 1e-8


def test_score_residual_and_bisection_diagnostics():
    windows = np.array([[-100.0, -100.0, 100.0], [-3.0, 0.0, 80.0], [7.0, 7.0, 7.0]])
    weights = np.array([0.3, 0.4, 0.3])
    contrast = NewtonSmoothAbs(0.5)
    roots = contrast.solve(windows, weights)
    score = np.abs(contrast.psi(windows - roots[:, None]) @ weights).max()
    assert np.isclose(contrast.diagnostics[0]["last_score_max"], score, atol=1e-15, rtol=1e-12)
    assert score < 1e-10
    assert contrast.diagnostics[0]["bisection_fallbacks"] > 0
    assert contrast.diagnostics[0]["converged"]
    failing = NewtonSmoothAbs(0.5, root_max_iter=1)
    with pytest.raises(RuntimeError, match="failed to converge"):
        failing.solve(windows, weights)
    assert not failing.diagnostics[0]["converged"] and failing.diagnostics[0]["failed_rows"] > 0


def test_numba_reuses_window_signature():
    warm_up()
    signatures = list(_fit_rows.signatures)
    for length in (128, 256):
        dec.decompose_signal(multiscale_signal(length), "irmf_h2_a1.41_conv")
    assert _fit_rows.signatures == signatures


def test_conv_pipeline_metadata_and_features(tiny_dataset):
    tag = "irmf_h2_a1.41_conv"
    summary = run_decompose(tiny_dataset, tag, workers=2, chunk_size=8, verbose=False)
    components, meta, stored = load_decomposition(tiny_dataset, tag)
    assert stored == json.loads(json.dumps(summary))
    assert stored["params"].items() >= solver_params().items()
    assert stored["n_failed"] == 0 and stored["irmf"]["channels_all_converged"] == 80
    assert len(meta) == 80 and meta["ok"].all()
    for key in ("stage_iterations", "stage_iterations_mean", "stage_bisection_fallbacks",
                "stage_max_score_residual", "stage_converged"):
        assert np.stack(meta[key]).shape == (80, components.shape[2] - 1)
    table = run_features(tiny_dataset, tag, workers=1, verbose=False)
    assert len(table) == 40 and table["decomp_ok"].all()


def test_conv_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("IMF_CLS_DATA", str(tmp_path))
    frames = np.random.default_rng(2).normal(size=(4, 1, 128))
    meta = pd.DataFrame({"frame_id": range(4), "group": list("aabb"), "y_cls": list("xyxy")})
    write_prepared(tmp_path, "capped", frames, meta)
    original_solve = NewtonSmoothAbs.solve

    def capped_solve(self, windows, weights):
        self.root_max_iter = 1
        return original_solve(self, windows, weights)

    monkeypatch.setattr(NewtonSmoothAbs, "solve", capped_solve)
    summary = run_decompose("capped", "irmf_h2_a1.41_conv", workers=1, verbose=False)
    components, meta, _ = load_decomposition("capped", "irmf_h2_a1.41_conv")
    assert summary["n_failed"] == 4 and not meta["ok"].any()
    assert meta["error"].str.contains("Newton failed to converge").all() and np.isnan(components).all()
    table = run_features("capped", "irmf_h2_a1.41_conv", verbose=False)
    assert not table["decomp_ok"].any() and table.filter(regex="_c[0-9]+_").isna().all().all()


@pytest.mark.parametrize("dataset, tag", [
    ("synthetic", "irmf_h2_a1.41"), ("uas_birds_60ghz", "irmf_h2_a1.41"),
    ("esc50_animals", "irmf_h2_a1.41"), ("birdclef2023", "irmf_h2_a1.41"),
    ("dronerf", "irmf_h2_a1.41"), ("esc50_animals", "irmf_h2_a2"),
])
def test_existing_gd_cache_is_byte_identical(dataset, tag):
    directory = DEFAULT_DATA_ROOT / dataset
    if not (directory / "decomp" / f"{tag}.json").exists():
        pytest.skip("optional pre-integration research cache unavailable")
    with open(directory / "decomp" / f"{tag}.json") as handle:
        cached_summary = json.load(handle)
    assert dec.parse_method(tag).params() == cached_summary["params"]
    frames = np.load(directory / "prepared" / "frames.npy", mmap_mode="r")
    cached = np.load(directory / "decomp" / f"{tag}.npy", mmap_mode="r")
    cached_meta = pd.read_parquet(directory / "decomp" / f"{tag}.meta.parquet")
    for index in (0, 1):
        actual, infos = dec.decompose_frame(frames[index], tag)
        assert actual.astype(np.float32).tobytes() == cached[index].tobytes()
        for channel, info in enumerate(infos):
            reference = cached_meta.query("frame_id == @index and channel == @channel").iloc[0]
            for key in ("stage_iterations", "stage_converged", "stage_final_delta"):
                np.testing.assert_array_equal(info[key], reference[key])
