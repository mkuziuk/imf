import numpy as np
import pytest

from emdirmf.cv import Split
from robustness.classification import excluded_split, predict_conditions
from robustness.processing import channel_stack, extract
from robustness.storage import save_npz


def test_cap_retry_and_final_failure(monkeypatch):
    calls = []
    def solver(y, root_max_iter=100):
        calls.append(root_max_iter)
        if root_max_iter == 100:
            raise RuntimeError('Newton failed to converge for 1 rows')
        return np.ones((2, len(y))), {'stage_converged': [True], 'stage_iterations': [110]}
    monkeypatch.setattr('robustness.processing.newton_components', solver)
    stack, hit, error = channel_stack(np.zeros(10), 'irmf_h2_a1.41_conv')
    assert calls == [100, 1000] and hit and not error and np.isfinite(stack).all()
    def fail(y, **kwargs):
        raise RuntimeError('Newton failed to converge for 1 rows')
    monkeypatch.setattr('robustness.processing.newton_components', fail)
    stack, hit, error = channel_stack(np.zeros(10), 'irmf_h2_a1.41_conv')
    assert stack is None and hit and error


def test_stack_reconstruction_and_feature_storage():
    x = np.random.default_rng(13).normal(size=(2, 128))
    v, stack, hits, err = extract(x, 'synthetic', 2, 'impulse2_0.04', 'imf_linear_a1.41')
    assert not err and not hits.any() and stack.dtype == np.float32
    from emdirmf.decompose import normalize_frame
    from robustness.contamination import contaminate
    for ch in range(2):
        expected, _ = normalize_frame(contaminate(x[ch], 'synthetic', 2, ch, 'impulse2_0.04'))
        np.testing.assert_allclose(stack[ch].sum(axis=0), expected, atol=2e-6, rtol=1e-6)
    assert np.isfinite(v).all()


def test_prediction_does_not_fit_and_uses_common_ids():
    class Model:
        def predict(self, x):
            return (x[:, 0] > 0).astype(int)
        def fit(self, *args):
            raise AssertionError('test leakage: fitting in predict')
    xs = {'clean': np.array([[-1], [2], [-3]]), 'noise': np.array([[4], [5], [-6]])}
    valid = {'clean': np.array([True, True, True]), 'noise': np.array([True, False, True])}
    result = predict_conditions(Model(), None, xs, np.arange(3), valid, ('clean', 'noise'))
    np.testing.assert_array_equal(result['noise'][0], [0, 2])
    np.testing.assert_array_equal(result['noise'][1], [1, 0])
    split = excluded_split(Split(0, 0, 0, np.array([0, 1]), np.array([2])), valid['noise'])
    np.testing.assert_array_equal(split.train, [0])


def test_checkpoints_do_not_replace_or_pickle(tmp_path):
    path = tmp_path/'x.npz'
    save_npz(path, x=np.arange(10))
    with pytest.raises(FileExistsError):
        save_npz(path, x=np.zeros(10))
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['x'], np.arange(10))
    with pytest.raises(TypeError):
        save_npz(tmp_path/'bad.npz', x=np.array([object()]))


def test_raw_preserves_shared_float64_feature_path():
    from emdirmf.features import frame_features, raw_components
    x=np.random.default_rng(99).normal(size=(1,2,1024)).astype(np.float32)
    components,scale=raw_components(x)
    expected=np.r_[frame_features(components)[0].astype(np.float32).astype(float),scale[0]]
    actual,_,_,error=extract(x[0],'synthetic',1,'clean','raw')
    assert not error
    np.testing.assert_array_equal(actual,expected)
