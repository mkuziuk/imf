import numpy as np
import pimf
import pytest

from components.construction import (
    EMD_K, NATIVE_K, TRUNC_K, all_tags, direct_truncation, fold_emd,
    native_components, native_ratio, parse_tag, relative_error, truncate_imf,
)
from components.integrity import check_pimf
from emdirmf.decompose import decompose_signal, emd_components, normalize_frame


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
