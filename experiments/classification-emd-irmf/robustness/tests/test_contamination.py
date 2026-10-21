from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from robustness.contamination import (CONDITIONS, clean_sigma, contaminate, draws)


def _draw_job(frame):
    return draws('synthetic', frame, 0, 1024)


def test_deterministic_across_order_and_workers():
    ids = [2, 8, 0, 31]
    expected = {i: _draw_job(i) for i in ids}
    for workers in (1, 2):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            result = list(pool.map(_draw_job, ids[::-1]))
        for i, actual in zip(ids[::-1], result):
            for a, b in zip(actual, expected[i]):
                np.testing.assert_array_equal(a, b)
    assert not np.array_equal(draws('synthetic', 2, 0, 1024)[0], draws('synthetic', 2, 1, 1024)[0])
    assert not np.array_equal(draws('synthetic', 2, 0, 1024)[0], draws('dronerf', 2, 0, 1024)[0])


def test_nested_masks_and_shared_impulses():
    x = np.linspace(-1, 1, 10000)
    masks = []
    for p in (0.002, 0.01, 0.04):
        a = contaminate(x, 'synthetic', 3, 0, f'impulse1_{p:g}') - x
        b = contaminate(x, 'synthetic', 3, 0, f'impulse2_{p:g}') - x
        np.testing.assert_array_equal(a != 0, b != 0)
        np.testing.assert_allclose(a, np.abs(b), atol=1e-14)
        masks.append(a != 0)
    assert np.all(~masks[0] | masks[1])
    assert np.all(~masks[1] | masks[2])


def test_matched_expected_power():
    # 4 million samples: use the variance of u*e^2 and z^2, not an arbitrary tolerance.
    n = 4_000_000
    u, e, s, z = draws('synthetic', 900, 0, n)
    for p in (0.002, 0.01, 0.04):
        a = np.mean((u < p) * e ** 2)
        b = 2 * p * np.mean(z ** 2)
        se = np.sqrt((24 * p - 4 * p * p + 8 * p * p) / n)
        assert abs(a - b) < 5 * se
        assert abs(a - 2 * p) < 5 * se


def test_scale_fallback_and_clean():
    x = np.array([0., 0., 0., 1.])
    assert clean_sigma(x) == np.std(x)
    assert clean_sigma(np.ones(10)) == 0
    for c in CONDITIONS:
        np.testing.assert_array_equal(contaminate(np.ones(10), 'synthetic', 0, 0, c), np.ones(10))
    np.testing.assert_array_equal(contaminate(x, 'synthetic', 0, 0, 'clean'), x)
    with pytest.raises(ValueError):
        contaminate(x, 'synthetic', 0, 0, 'bad_0.04')
