import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robustness_v3.contamination import CONDITIONS, clean_sigma, contaminate
from robustness_v3.construction import NATIVE_K, WINDOWS, channel_stack, fold_emd, native_ratio, variants


def test_noise_scale_draws_and_constant():
    from robustness.contamination import draws
    x = np.random.default_rng(7).normal(size=2048)
    u, e, s, _ = draws('synthetic', 3, 0, len(x))
    previous = np.zeros(len(x), bool)
    for p in (.1, .2, .3, .4):
        noise = contaminate(x, 'synthetic', 3, 0, f'nb3_{p}')-x
        np.testing.assert_allclose(noise, (u < p)*3*clean_sigma(x)*e*s, atol=1e-14)
        assert np.all(~previous | (noise != 0))
        previous = noise != 0
    np.testing.assert_array_equal(contaminate(np.ones(2048), 'synthetic', 0, 0, CONDITIONS[-1]), 1)
    with pytest.raises(ValueError):
        contaminate(x, 'synthetic', 3, 0, 'laplace_0.4')


@pytest.mark.parametrize('length', [1024, 2048])
def test_exact_endpoint_schedules(length):
    import pimf
    for k in NATIVE_K[length]:
        result = pimf.IMF(pimf.Quadratic(), workers=1).decompose(np.zeros(length), h1=.25,
            a=native_ratio(length, k), h_min=15/length, k_max=None)
        assert tuple(result.window_sizes) == WINDOWS[length][k]
        assert result.bandwidths[-1]*length == 15
        assert len(result.stages) == k


def test_folding_reconstruction():
    stack = np.random.default_rng(2).normal(size=(2, 9, 1024))
    for k in (2, 3, 4, 6, 8):
        folded = fold_emd(stack, k)
        np.testing.assert_array_equal(folded[:, :k], stack[:, :k])
        np.testing.assert_allclose(folded.sum(axis=1), stack.sum(axis=1), atol=1e-14)


def test_fixed_h_calibration_only_uses_pilots():
    from emdirmf.decompose import noise_sigma_hat, normalize_frame
    from robustness_v3.estimators import calibrate_h
    frames = np.random.default_rng(1).normal(size=(110, 2, 1024))
    ids = np.arange(100)
    value = calibrate_h(frames, ids)
    expected = 2*np.median([noise_sigma_hat(normalize_frame(x)[0])[0] for frame in frames[:100] for x in frame])
    assert value['H_d'] == expected and value['n_channels'] == 200
    frames[100:] *= 100
    assert calibrate_h(frames, ids) == value


def test_newton_fixed_h_and_retry(monkeypatch):
    from robustness_v3 import construction as c
    from emdirmf.decompose import normalize_frame
    y = normalize_frame(np.random.default_rng(3).normal(size=1024))[0]
    stack, info = c.newton_components(y, h_min=15/1024, a=native_ratio(1024, 3), fixed_h=.731)
    assert info['H'] == .731
    assert all(info['stage_converged'])
    np.testing.assert_allclose(stack.sum(axis=0), y, atol=1e-12)
    actual = c.newton_components
    caps = []
    def retry(*args, **kwargs):
        caps.append(kwargs['root_max_iter'])
        if kwargs['root_max_iter'] == 100:
            raise RuntimeError('Newton failed to converge: forced test')
        return actual(*args, **kwargs)
    monkeypatch.setattr(c, 'newton_components', retry)
    stack, cap, error = c.channel_stack(y, 'irmf', 3, .731)
    assert caps == [100, 1000] and cap and not error
    monkeypatch.setattr(c, 'newton_components', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Newton failed to converge')))
    stack, cap, error = c.channel_stack(y, 'irmf', 3, .731)
    assert stack is None and cap and error


def test_extraction_raw_precision_one_emd_and_fixed_h(monkeypatch):
    from emdirmf.decompose import normalize_frame, noise_sigma_hat
    from emdirmf.features import frame_features
    from robustness_v3 import processing as p
    calls = []
    original = p.shared_channel_stack
    def count(y, method):
        calls.append(method)
        return original(y, method)
    monkeypatch.setattr(p, 'shared_channel_stack', count)
    frame = np.random.default_rng(14).normal(size=(1, 1024))
    for condition in ('clean', 'nb3_0.4'):
        result, diag = p.extract_all(frame, 'synthetic', 3, condition, .65)
        y, _ = normalize_frame(contaminate(frame[0], 'synthetic', 3, 0, condition))
        assert diag['h'][0] == .65
        assert diag['h_ratio'][0] == noise_sigma_hat(y)[0]/.65
        assert result['raw']['stack'].dtype == np.float64
        np.testing.assert_array_equal(result['raw']['values'], frame_features(y[None, None, None])[0].astype(np.float32))
        for v, r in result.items():
            assert not r['error'] and np.isfinite(r['values']).all()
            if v != 'raw':
                assert r['stack'].dtype == np.float32
            np.testing.assert_allclose(r['stack'].sum(axis=1)[0], y, atol=2e-6, rtol=1e-6)
    assert calls == ['emd', 'emd']


def test_k_tie_break_and_complete_splits():
    from robustness_v3.classification import select_k
    rows = [dict(variant=f'irmf_k{k}', split_id=sid, balanced_accuracy=.75) for k in (8, 3, 6) for sid in range(15)]
    assert select_k(rows)['k'] == 3
    with pytest.raises(ValueError):
        select_k(rows[:-1])


def test_matched_waits_for_method_all_k(monkeypatch, tmp_path):
    from robustness_v3 import classification as c
    monkeypatch.setattr(c, 'root', lambda d: tmp_path)
    monkeypatch.setattr(c, 'dataset_variants', lambda d: ('linear_k3', 'linear_k4'))
    deps = c.shift_dependencies('synthetic', 'linear', 'logreg')
    for args in deps[:-1]:
        path = c.job_checkpoint(args)/'complete.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(rows=[dict(variant=args[3], split_id=args[5], condition='clean', balanced_accuracy=.6)])))
    assert c.matched_jobs('synthetic', 'linear') == []
    last = c.job_checkpoint(deps[-1])/'complete.json'
    last.parent.mkdir(parents=True)
    last.write_text(json.dumps(dict(rows=[dict(variant='linear_k4', split_id=14, condition='clean', balanced_accuracy=.6)])))
    jobs = c.matched_jobs('synthetic', 'linear')
    assert len(jobs) == 15 and {a[3] for _, a in jobs} == {'linear_k3'}


def mock_scores(datasets):
    from robustness_v3.analysis import CLASSICAL
    rows = []
    for d in datasets:
        for clf in (CLASSICAL[d], 'lightgbm'):
            for method in ('raw', 'emd', 'linear', 'irmf'):
                ks = [0] if method == 'raw' else [3, 4]
                for k in ks:
                    v = 'raw' if k == 0 else f'{method}_k{k}'
                    for sid in range(15):
                        for scenario in ('shift', 'matched') if clf != 'lightgbm' else ('shift',):
                            for condition in CONDITIONS if scenario == 'shift' else CONDITIONS[-1:]:
                                rows.append(dict(dataset=d, classifier=clf, scenario=scenario, condition=condition,
                                    feature_set='comp', method=v, variant=v, family=method, k=k, split_id=sid,
                                    balanced_accuracy=.8 + .01*(k==3) - .01*(condition!='clean')*(method!='irmf'),
                                    clean_balanced_accuracy=.8 + .01*(k==3), n_train=800, n_test=200))
    return pd.DataFrame(rows)


def test_families_and_selected_did():
    from robustness_v3.analysis import DATASETS, FAMILY_SIZES, paired_tests, holm_padded
    scores = mock_scores(DATASETS)
    tests, vectors = paired_tests(scores)
    assert tests.groupby('holm_family').size().to_dict() == FAMILY_SIZES
    assert (tests.irmf_k == 3).all() and (tests.comparator_k == 3).all()
    np.testing.assert_allclose(tests.mean_diff, .01)
    assert len(vectors) == len(tests)*15
    partial, _ = paired_tests(scores[scores.dataset == 'dronerf'], ['dronerf'])
    assert not partial.holm_complete.any()
    with pytest.raises(AssertionError):
        holm_padded([.1, .2], 1)


def test_saved_metric_validation_catches_wrong_baseline():
    from robustness_v3.analyse import check_row
    from emdirmf.cv import split_metrics
    y = np.array(['a', 'b', 'a', 'b'])
    ids = np.arange(4)
    pred = np.array(['a', 'a', 'a', 'b'])
    row = dict(n_test=4, **split_metrics(y, pred, ['a', 'b']),
               **{'clean_'+k: v for k, v in split_metrics(y, y, ['a', 'b']).items()})
    check_row(row, y, ['a', 'b'], ids, pred, dict(enumerate(y)))
    with pytest.raises(AssertionError, match='clean-baseline'):
        check_row(dict(row, clean_balanced_accuracy=.2), y, ['a', 'b'], ids, pred, dict(enumerate(y)))


def test_storage_never_overwrites_or_pickles(tmp_path):
    from robustness_v3.storage import save_json, save_npz
    save_json(tmp_path/'a.json', {'a': 1})
    with pytest.raises(FileExistsError):
        save_json(tmp_path/'a.json', {'a': 2})
    assert json.loads((tmp_path/'a.json').read_text()) == {'a': 1}
    with pytest.raises(TypeError):
        save_npz(tmp_path/'bad.npz', x=np.array([{}], dtype=object))
