import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from baselines import run_classical
from emdirmf import cv, stats
from emdirmf.paths import results_dir
from emdirmf.run_features import run_features


def test_native_probabilities_and_pickle_free_labels(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    y = np.where(X[:, 0] > 0, "bird", "drone").astype(object)
    groups = np.repeat(np.arange(30), 2)
    split = cv.Split(0, 0, 0, np.arange(40), np.arange(40, 60))
    model = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]},
                         cv=StratifiedGroupKFold(2, shuffle=True, random_state=0))
    row, pred, probabilities, classes = cv.fit_predict_split(
        model, X, y, split, groups=groups, return_probabilities=True)
    reference = clone(model).fit(X[split.train], y[split.train], groups=groups[split.train])
    np.testing.assert_array_equal(pred, reference.predict(X[split.test]))
    np.testing.assert_allclose(probabilities, reference.predict_proba(X[split.test]))
    np.testing.assert_array_equal(classes, reference.classes_)
    assert row["balanced_accuracy"] == cv.split_metrics(y[split.test], pred, classes)["balanced_accuracy"]
    assert pred.dtype.kind == classes.dtype.kind == "U"
    np.savez(tmp_path / "labels.npz", y_pred=pred, classes=classes)
    with np.load(tmp_path / "labels.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["y_pred"], pred)
        np.testing.assert_array_equal(saved["classes"], classes)
    assert len(cv.fit_predict_split(model, X, y, split, groups=groups)) == 2
    _, numeric_pred = cv.fit_predict_split(LogisticRegression(), X, (y == "bird").astype(int), split)
    assert numeric_pred.dtype.kind in "iu"


def test_default_pairs_cover_all_ablation_tags():
    main = list(run_classical.DEFAULT_METHODS)
    expected = [("irmf_h2_a1.41", "emd"), ("irmf_h2_a1.41", "imf_linear_a1.41"),
                ("imf_linear_a1.41", "emd"), ("irmf_h2_a1.41", "raw"),
                ("imf_linear_a1.41", "raw"), ("emd", "raw")]
    assert stats.default_pairs(main) == expected
    tags = main + ["irmf_h2_a1.41_conv", "irmf_h1_a1.41_conv", "irmf_h2_a2_conv", "imf_linear_a2"]
    pairs = stats.default_pairs(tags + [tags[-1]])
    assert pairs[:6] == expected
    assert len(pairs) == len(tags) * (len(tags) - 1) // 2
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
    assert ("irmf_h2_a1.41_conv", "emd") in pairs
    assert ("irmf_h2_a1.41", "irmf_h2_a1.41_conv") in pairs
    assert ("irmf_h2_a2_conv", "imf_linear_a2") in pairs


def test_classical_export_isolation_and_cli(tiny_dataset, tmp_path, monkeypatch):
    run_features(tiny_dataset, "raw", verbose=False)
    out = tmp_path / "tables"
    original_job = run_classical._job

    def checkpoint_job(key, estimator, X, y, split, groups, labels):
        # Existing seven-argument wrappers can save incoming truth labels without
        # pickling, even when pandas supplies object/string-backed metadata.
        assert y.dtype.kind == "U"
        np.savez(tmp_path / "truth.npz", y_true=y[split.test])
        return original_job(key, estimator, X, y, split, groups, labels)

    monkeypatch.setattr(run_classical, "_job", checkpoint_job)
    run_classical.run(tiny_dataset, methods=["raw"], classifier_names=["lda"], quick=True, out=out)
    with np.load(tmp_path / "truth.npz", allow_pickle=False) as saved:
        assert saved["y_true"].dtype.kind == "U"
    monkeypatch.setattr(run_classical, "_job", original_job)
    legacy = results_dir(tiny_dataset) / "classical_cls_predictions.parquet"
    legacy_bytes = legacy.read_bytes()
    assert not any(column.startswith("prob_") for column in pd.read_parquet(legacy))
    for tag in ("main", "ablations"):
        assert run_classical.main([
            "--dataset", tiny_dataset, "--methods", "raw", "--classifiers", "lda", "linsvm",
            "--quick", "--workers", "2", "--out", str(out / tag),
            "--results-tag", tag, "--save-probabilities",
        ]) == 0
        path = results_dir(tiny_dataset) / tag / legacy.name
        predictions = pd.read_parquet(path)
        probability_columns = ["prob_a", "prob_b"]
        lda = predictions[predictions.classifier == "lda"]
        svm = predictions[predictions.classifier == "linsvm"]
        np.testing.assert_allclose(lda[probability_columns].sum(axis=1), 1.0)
        assert svm[probability_columns].isna().all().all()
        np.testing.assert_array_equal(
            np.array(["a", "b"])[lda[probability_columns].to_numpy().argmax(axis=1)],
            lda.y_pred.to_numpy())
        for _, cell in predictions.groupby(["classifier", "feature_set"]):
            assert sorted(cell.frame_id) == list(range(40))
        metadata = json.loads((out / tag / "run_cls.json").read_text())
        assert metadata["predictions"] == str(path)
    assert legacy.read_bytes() == legacy_bytes
    pd.testing.assert_frame_equal(
        pd.read_parquet(results_dir(tiny_dataset) / "main" / legacy.name),
        pd.read_parquet(results_dir(tiny_dataset) / "ablations" / legacy.name))


@pytest.mark.parametrize("tag", ["", ".", "..", "../main", "/tmp/main", "main/part", "main\\part"])
def test_results_tag_rejects_paths(tag):
    with pytest.raises(ValueError, match="results_tag"):
        run_classical.run("unused", results_tag=tag)
