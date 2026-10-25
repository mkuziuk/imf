import numpy as np
import pimf
import pytest

from components.construction import (
    EMD_K, NATIVE_K, TRUNC_K, all_tags, direct_truncation, fold_emd,
    native_components, native_ratio, parse_tag, relative_error, truncate_imf,
)
from components.integrity import check_pimf
from emdirmf.decompose import decompose_signal, emd_components, normalize_frame
from emdirmf.irmf_newton import NewtonSmoothAbs


@pytest.mark.parametrize("length", [1024, 2048])
@pytest.mark.parametrize("k", NATIVE_K)
@pytest.mark.parametrize("family", ["linear", "irmf"])
def test_native_endpoint_and_reconstruction(length, k, family):
    check_pimf()
    y = normalize_frame(np.random.default_rng(length).normal(size=length))[0]
    stack, info = native_components(y, family, k)
    assert stack.shape == (k + 1, length)
    assert info["last_radius"] == 1.5
    assert native_ratio(length, k) > 1
    assert relative_error(stack.sum(axis=0), y) < 1e-14
    assert relative_error(stack.astype(np.float32).sum(axis=0, dtype=np.float64), y) < 1e-7


@pytest.mark.parametrize("family,tag", [
    ("linear", "imf_linear_a1.41"), ("irmf", "irmf_h2_a1.41_conv"),
    ("gd60", "irmf_h2_a1.41")])
@pytest.mark.parametrize("k", TRUNC_K)
def test_truncation_matches_sequential_solver(family, tag, k):
    y = normalize_frame(np.random.default_rng(9).normal(size=1024))[0]
    full, _ = decompose_signal(y, tag)
    folded = truncate_imf(full.astype(np.float32), k, tag)
    direct = direct_truncation(y, family, k)
    assert relative_error(folded, direct) < 1e-7
    assert relative_error(folded.astype(np.float32).sum(axis=0, dtype=np.float64), y) < 1e-7


@pytest.mark.parametrize("k", EMD_K)
def test_emd_fold_matches_direct_defaults(k):
    y = normalize_frame(np.random.default_rng(32).normal(size=2048))[0]
    full, _ = emd_components(y)
    folded = fold_emd(full.astype(np.float32), k)
    direct, _ = emd_components(y, k_emd=k)
    assert relative_error(folded, direct) < 1e-7
    assert relative_error(folded.astype(np.float32).sum(axis=0, dtype=np.float64), y) < 1e-7


def test_tags_are_unique_and_local():
    assert len(all_tags()) == len(set(all_tags()))
    for tag in all_tags():
        assert parse_tag(tag).tag == tag
    for bad in ("linear_native_k8", "emd_k9", "irmf_trunc_k6"):
        with pytest.raises(ValueError):
            parse_tag(bad)


def test_newton_recomputes_entire_channel_at_larger_cap(monkeypatch):
    import components.construction as construction

    class CapOnLastIteration(NewtonSmoothAbs):
        def solve(self, windows, weights):
            values = super().solve(windows, weights)
            if self.root_max_iter == 100:
                self.diagnostics[-1]["iterations_max"] = 100
            return values

    y = normalize_frame(np.random.default_rng(18).normal(size=1024))[0]
    expected, _ = native_components(y, "irmf", 3)
    monkeypatch.setattr(construction, "NewtonSmoothAbs", CapOnLastIteration)
    actual, info = native_components(y, "irmf", 3)
    assert [attempt["cap"] for attempt in info["attempts"]] == [100, 1000]
    assert info["attempts"][0]["cap_stages"] == [0, 1, 2]
    assert not info["attempts"][1]["cap_stages"]
    np.testing.assert_array_equal(actual, expected)


def test_newton_never_accepts_failed_retry(monkeypatch):
    import components.construction as construction

    class FailedRoot(NewtonSmoothAbs):
        def solve(self, windows, weights):
            super().solve(windows, weights)
            self.diagnostics[-1].update(iterations_max=self.root_max_iter, failed_rows=1)
            raise RuntimeError("injected unconverged row")

    y = normalize_frame(np.random.default_rng(19).normal(size=1024))[0]
    monkeypatch.setattr(construction, "NewtonSmoothAbs", FailedRoot)
    with pytest.raises(RuntimeError, match="unaccepted Newton fit after retry"):
        native_components(y, "irmf", 3)
