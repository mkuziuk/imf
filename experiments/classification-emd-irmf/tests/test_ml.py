"""Small CPU-only checks for phase-3 leakage, shapes and reproducibility."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from emdirmf.cv import load_splits, make_splits
from emdirmf.decompose import n_slots, normalize_frame, scale_order
from emdirmf.io import load_frames, load_meta
from emdirmf.paths import decomp_dir
from emdirmf.stacks import StackReader
from ml.common import inner_split, select_splits


def test_inner_validation_is_grouped_and_ignores_test_labels(tiny_dataset):
    split = make_splits(tiny_dataset, verbose=False)[0]
    meta = load_meta(tiny_dataset)
    y, groups = meta.y_cls.to_numpy(), meta.group.to_numpy()
    fit, valid = inner_split(split.train, y, groups, 0)
    assert set(fit) | set(valid) == set(split.train)
    assert not set(groups[fit]) & set(groups[valid])
    assert not (set(fit) | set(valid)) & set(split.test)
    changed = y.copy()
    changed[split.test] = "unseen_test_label"
    fit_again, valid_again = inner_split(split.train, changed, groups, 0)
    np.testing.assert_array_equal(fit, fit_again)
    np.testing.assert_array_equal(valid, valid_again)


def test_fold_selection_keeps_shared_ids(tiny_dataset):
    splits = make_splits(tiny_dataset, verbose=False)
    assert [s.split_id for s in select_splits(splits, "3")] == [0, 1, 2]
    assert [s.split_id for s in select_splits(splits, "0,5,10")] == [0, 5, 10]
    assert len(select_splits(splits, quick=True)) == 5
    with pytest.raises(ValueError):
        select_splits(splits, "0,0")
    with pytest.raises(ValueError):
        select_splits(splits, "16")


def test_reader_normalization_scale_order_and_deterministic_batches(tiny_dataset):
    raw = StackReader(tiny_dataset, "raw")
    indices = np.array([7, 0, 19])
    expected, scales = normalize_frame(load_frames(tiny_dataset)[indices])
    np.testing.assert_array_equal(raw.take(indices), expected.astype(np.float32))
    np.testing.assert_array_equal(raw.log_scales(indices), scales.astype(np.float32))
    assert raw.take(indices).shape == (3, 2, 256)
    a = list(raw.batches(np.arange(20), 6, shuffle=True, seed=42))
    b = list(raw.batches(np.arange(20), 6, shuffle=True, seed=42))
    for (ia, xa), (ib, xb) in zip(a, b):
        np.testing.assert_array_equal(ia, ib)
        np.testing.assert_array_equal(xa, xb)
    assert not np.array_equal(np.concatenate([i for i, _ in a]), np.arange(20))

    tag = "imf_linear_a1.41"
    slots = n_slots(tag, 256)
    values = np.broadcast_to(np.arange(2 * slots).reshape(1, 2, slots, 1), (40, 2, slots, 256)).astype(np.float32)
    directory = decomp_dir(tiny_dataset)
    directory.mkdir()
    np.save(directory / f"{tag}.npy", values)
    (directory / f"{tag}.json").write_text(json.dumps({"shape": list(values.shape), "n_failed": 0}))
    reader = StackReader(tiny_dataset, tag)
    expected = values[indices][:, :, scale_order(tag, 256)].reshape(3, 2 * slots, 256)
    np.testing.assert_array_equal(reader.take(indices), expected)
    assert reader.take([0])[0, 0, 0] == slots - 1  # finest residual first
    downsampled = StackReader(tiny_dataset, tag, downsample=3)
    assert downsampled.take(indices).shape == (3, 2 * slots, 86)
    assert reader.source.mode == "r"
    values[0, 0, 0, 0] = np.nan
    np.save(directory / f"{tag}.npy", values)
    with pytest.raises(ValueError, match="non-finite"):
        StackReader(tiny_dataset, tag).take([0])


def test_rocket_scaler_and_head_ignore_held_out_features():
    from ml.run_rocket import fit_head, predict_head

    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 8)).astype(np.float32)
    y = np.tile(["a", "b"], 15)
    train = np.arange(20)
    args = SimpleNamespace(head_batch_size=6, head_epochs=3)
    a = fit_head(X, None, train, y, 0.001, args, 2)
    changed = X.copy()
    changed[20:] += 100000
    b = fit_head(changed, None, train, y, 0.001, args, 2)
    np.testing.assert_allclose(a[0].mean_, X[train].mean(axis=0), atol=1e-7)
    np.testing.assert_array_equal(a[0].mean_, b[0].mean_)
    np.testing.assert_array_equal(a[1].coef_, b[1].coef_)
    np.testing.assert_array_equal(predict_head(X, None, train, a, args), predict_head(X, None, train, b, args))


def test_rocket_bias_fit_uses_only_inner_training(tmp_path):
    from ml.run_rocket import transform_to_disk

    class Reader:
        shape = (40, 1, 16)

        def take(self, indices):
            return np.broadcast_to(np.asarray(indices)[:, None, None], (len(indices), 1, 16)).copy()

        def batches(self, indices, batch_size):
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                yield batch, self.take(batch)

    class Transform:
        def fit(self, X):
            self.fitted_ids = X[:, 0, 0]

        def transform(self, X):
            return X[:, 0, :2]

    model = Transform()
    train, valid = np.arange(20), np.arange(20, 30)
    args = SimpleNamespace(fit_samples=10, batch_size=7)
    values = transform_to_disk(Reader(), model, train, np.r_[train, valid], tmp_path / "features.npy", args, 3)
    assert set(model.fitted_ids) <= set(train)
    assert len(model.fitted_ids) == 10
    assert not set(model.fitted_ids) & set(valid)
    np.testing.assert_array_equal(values[valid, 0], valid)


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "extratrees"])
def test_tree_fit_is_independent_of_outer_test(tiny_dataset, name):
    from ml.run_trees import fit_tree

    make_splits(tiny_dataset, verbose=False)
    split = load_splits(tiny_dataset)[0]
    meta = load_meta(tiny_dataset)
    y = (meta.y_cls.to_numpy() == "b").astype(int)
    groups = meta.group.to_numpy()
    X = np.random.default_rng(7).normal(size=(40, 5)).astype(np.float32)
    args = SimpleNamespace(quick=True, workers=1)
    a, impute_a, params_a = fit_tree(name, args, X, y, groups, split)
    X[split.test] += 1e6
    y[split.test] = 9
    b, impute_b, params_b = fit_tree(name, args, X, y, groups, split)
    assert params_a == params_b
    np.testing.assert_array_equal(impute_a.statistics_, impute_b.statistics_)
    np.testing.assert_array_equal(a.feature_importances_, b.feature_importances_)


def test_cnn_loader_seed_shapes_and_scale_training_only(tiny_dataset):
    import torch
    from ml.run_cnn import CompactResNet, make_loader, scale_inputs, set_seed

    torch.set_num_threads(1)
    reader = StackReader(tiny_dataset, "raw")
    train = np.arange(20)
    y = np.arange(40) % 2
    scales = scale_inputs(reader, train, True, 8)
    np.testing.assert_allclose(scales[train].mean(axis=0), 0, atol=1e-6)
    original = reader.log_scales
    reader.log_scales = lambda indices: original(indices) + (np.asarray(indices) >= 20)[:, None] * 1e5
    changed = scale_inputs(reader, train, True, 8)
    np.testing.assert_array_equal(scales[train], changed[train])
    loader_a = make_loader(reader, train, y, scales, 6, 15, True)
    loader_b = make_loader(reader, train, y, scales, 6, 15, True)
    for a, b in zip(loader_a, loader_b):
        for aa, bb in zip(a, b):
            assert torch.equal(aa, bb)
        assert a[0].shape[1:] == (2, 256)
        assert a[1].shape[1] == 2
    x, side, _ = next(iter(make_loader(reader, train, y, scales, 6, 0, False)))
    set_seed(11)
    a = CompactResNet(2, 2, width=4, scale_features=2)(x, side)
    set_seed(11)
    b = CompactResNet(2, 2, width=4, scale_features=2)(x, side)
    assert a.shape == (6, 2) and torch.equal(a, b)
