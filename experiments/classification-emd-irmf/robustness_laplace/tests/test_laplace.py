import numpy as np
import pytest

from robustness.contamination import contaminate as impulse_contaminate
from robustness.contamination import draws
from robustness_laplace.contamination import CONDITIONS, LEVELS, clean_sigma, contaminate


def test_conditions():
    assert CONDITIONS == ('clean', 'laplace_0.1', 'laplace_0.2', 'laplace_0.3', 'laplace_0.4')
    x = np.linspace(-1, 1, 64)
    clean = contaminate(x, 'synthetic', 0, 0, 'clean')
    np.testing.assert_array_equal(clean, x)
    assert clean is not x
    with pytest.raises(ValueError):
        contaminate(x, 'synthetic', 0, 0, 'gauss_0.1')


def test_signed_exponential_with_nested_masks():
    x = np.sin(np.linspace(0, 40, 20000))
    u, e, s, _ = draws('dronerf', 7, 1, x.size)
    sigma = clean_sigma(x)
    previous = np.zeros(x.size, dtype=bool)
    for p in LEVELS:
        noise = contaminate(x, 'dronerf', 7, 1, f'laplace_{p:g}') - x
        np.testing.assert_allclose(noise, (u < p) * sigma * e * s, rtol=0, atol=1e-12)
        mask = noise != 0
        assert np.all(~previous | mask)
        previous = mask


def test_common_random_numbers_with_impulse_run():
    # The impulse run's two-sided p=.04 spikes are the same draws at ten times the scale.
    x = np.cos(np.linspace(0, 25, 8192))
    laplace = contaminate(x, 'birdclef2023', 11, 0, 'laplace_0.1') - x
    impulse = impulse_contaminate(x, 'birdclef2023', 11, 0, 'impulse2_0.04') - x
    spikes = impulse != 0
    assert spikes.any() and np.all(laplace[spikes] != 0)
    np.testing.assert_allclose(impulse[spikes], 10 * laplace[spikes], rtol=1e-12)


def test_laplace_moments():
    x = np.random.default_rng(5).normal(size=400_000)
    sigma = clean_sigma(x)
    noise = contaminate(x, 'synthetic', 1, 0, 'laplace_0.3') - x
    mask = noise != 0
    n = mask.size
    assert abs(mask.mean() - 0.3) < 5 * np.sqrt(0.3 * 0.7 / n)
    hit = noise[mask] / sigma
    # Laplace(0, 1): mean 0, mean |v| = 1, variance 2, symmetric signs.
    assert abs(hit.mean()) < 5 * np.sqrt(2 / hit.size)
    assert abs(np.abs(hit).mean() - 1) < 5 / np.sqrt(hit.size)
    assert abs(hit.var() - 2) < 5 * np.sqrt(20 / hit.size)


def test_constant_channel_gets_no_noise():
    x = np.full(256, 3.0)
    np.testing.assert_array_equal(contaminate(x, 'synthetic', 4, 0, 'laplace_0.4'), x)


def test_extraction_reconstructs_contaminated_signal():
    from emdirmf.decompose import normalize_frame
    from robustness_laplace.processing import extract
    x = np.random.default_rng(13).normal(size=(2, 256))
    for method in ('imf_linear_a1.41', 'irmf_h2_a1.41_conv'):
        values, stack, hits, error = extract(x, 'synthetic', 2, 'laplace_0.2', method)
        assert not error and not hits.any() and stack.dtype == np.float32 and np.isfinite(values).all()
        for channel in range(2):
            expected, _ = normalize_frame(contaminate(x[channel], 'synthetic', 2, channel, 'laplace_0.2'))
            np.testing.assert_allclose(stack[channel].sum(axis=0), expected, atol=5e-6, rtol=1e-6)


@pytest.mark.parametrize('classifier', ['linsvm', 'logreg'])
def test_refit_at_selected_c_equals_grid_search(classifier):
    from robustness_laplace.classification import fit_classical
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 12))
    y = np.array(['a', 'b', 'c'])[(X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=.8, size=300) > 0).astype(int)
                                   + (X[:, 2] > 1).astype(int)]
    groups = np.repeat(np.arange(30), 10).astype(str)
    np.random.seed(1)
    search, params = fit_classical(classifier, X, y, groups)
    np.random.seed(2)
    model, same = fit_classical(classifier, X, y, groups, params)
    assert same == params
    np.testing.assert_array_equal(model.predict(X), search.predict(X))
    np.testing.assert_array_equal(model.named_steps['clf'].coef_, search.best_estimator_.named_steps['clf'].coef_)


def test_incomplete_family_never_uses_fewer_tests():
    from emdirmf.stats import holm
    from robustness_laplace.analysis import holm_padded
    p = np.array([0.001, 0.02, 0.04, 0.3])
    np.testing.assert_allclose(holm_padded(p, 4), holm(p))
    padded = holm_padded(p, 12)
    np.testing.assert_allclose(padded, holm(np.r_[p, np.ones(8)])[:4])
    assert np.all(padded >= holm(p))
    with pytest.raises(AssertionError):
        holm_padded(p, 3)


def test_check_row_catches_wrong_clean_baseline_and_counts():
    from emdirmf.cv import split_metrics
    from robustness_laplace.analyse import check_row
    y = np.array(['a', 'b', 'a', 'b', 'a', 'b'])
    ids = np.arange(6)
    pred = np.array(['a', 'b', 'b', 'b', 'a', 'a'])
    baseline = dict(enumerate(y.tolist()))
    labels = ['a', 'b']
    row = dict(n_test=6, **split_metrics(y, pred, labels),
               **{'clean_'+k: v for k, v in split_metrics(y, y, labels).items()})
    check_row(row, y, labels, ids, pred, baseline)
    with pytest.raises(AssertionError, match='clean-baseline'):
        check_row(dict(row, clean_balanced_accuracy=row['clean_balanced_accuracy']-0.2), y, labels, ids, pred, baseline)
    with pytest.raises(AssertionError, match='test count'):
        check_row(dict(row, n_test=5), y, labels, ids, pred, baseline)


def test_run_record_accepts_one_run_for_all_datasets(monkeypatch, tmp_path):
    import json
    from robustness_laplace import analyse
    from robustness_laplace.contamination import DATASETS
    monkeypatch.setattr(analyse, 'runs_dir', lambda: tmp_path)
    with pytest.raises(RuntimeError):
        analyse.run_record(DATASETS)
    (tmp_path/'run_a.json').write_text(json.dumps(dict(datasets=list(DATASETS), workers=14, wall_seconds=10.0,
                                                       started_utc='2026-10-22T00:00:00')))
    assert analyse.run_record(DATASETS)['workers'] == 14
