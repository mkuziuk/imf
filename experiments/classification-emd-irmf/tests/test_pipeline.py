import json

import numpy as np
import pandas as pd
import pytest
from conftest import write_prepared

from emdirmf import cv, features, stats
from emdirmf import decompose as dec
from emdirmf.io import validate_prepared
from emdirmf.paths import decomp_dir, splits_path
from emdirmf.run_decompose import load_decomposition, run_decompose
from emdirmf.run_features import load_features, run_features


def test_validate_tiny(tiny_dataset):
    report = validate_prepared(tiny_dataset, verbose=False)
    assert report["ok"], report["errors"]
    assert report["summary"]["n_groups"] == 10


def test_validate_flags_problems(tmp_path, monkeypatch):
    monkeypatch.setenv("IMF_CLS_DATA", str(tmp_path))
    frames = np.random.default_rng(0).normal(size=(6, 1, 64))
    frames[2, 0, 5] = np.nan
    meta = pd.DataFrame({"frame_id": [0, 1, 2, 3, 5, 4], "group": list("aabbcc"), "y_cls": list("xyxyxy")})
    write_prepared(tmp_path, "bad", frames, meta)
    report = validate_prepared("bad", verbose=False)
    assert not report["ok"]
    assert any("frame_id" in error for error in report["errors"])
    assert any("non-finite" in error for error in report["errors"])


@pytest.mark.parametrize("tag", ["emd", "irmf_h2_a1.41"])
def test_batch_matches_serial_and_resumes(tiny_dataset, tag):
    summary = run_decompose(tiny_dataset, tag, workers=2, chunk_size=8, verbose=False)
    components, meta, _ = load_decomposition(tiny_dataset, tag)
    frames = np.load(decomp_dir(tiny_dataset).parent / "prepared" / "frames.npy")
    assert components.shape == (40, 2, dec.n_slots(tag, 256), 256) and components.dtype == np.float32
    for index in (0, 13, 39):
        expected, _ = dec.decompose_frame(frames[index], tag)
        np.testing.assert_array_equal(components[index], expected.astype(np.float32))
    assert len(meta) == 80 and meta["ok"].all() and summary["n_failed"] == 0
    reference = np.array(components)

    parts = decomp_dir(tiny_dataset) / f"{tag}.parts"
    (parts / "chunk_000002.parquet").unlink()
    writable = np.load(decomp_dir(tiny_dataset) / f"{tag}.npy", mmap_mode="r+")
    writable[16:24] = 0
    writable.flush()
    del writable
    run_decompose(tiny_dataset, tag, workers=1, chunk_size=8, verbose=False)
    resumed, meta2, summary2 = load_decomposition(tiny_dataset, tag)
    np.testing.assert_array_equal(resumed, reference)
    assert len(summary2["sessions"]) == 2 and summary2["sessions"][1]["chunks"] == 1
    with pytest.raises(RuntimeError):
        run_decompose(tiny_dataset, tag, workers=1, chunk_size=4, verbose=False)


def test_batch_records_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("IMF_CLS_DATA", str(tmp_path))
    frames = np.random.default_rng(3).normal(size=(6, 1, 128))
    frames[4, 0, 7] = np.inf
    meta = pd.DataFrame({"frame_id": range(6), "group": list("aabbcc"), "y_cls": list("xyxyxy")})
    write_prepared(tmp_path, "broken", frames, meta)
    summary = run_decompose("broken", "imf_linear", workers=1, chunk_size=4, verbose=False)
    components, meta_out, _ = load_decomposition("broken", "imf_linear")
    assert summary["n_failed"] == 1 and not meta_out.loc[4, "ok"]
    assert np.all(np.isnan(components[4])) and np.all(np.isfinite(components[:4]))
    table = run_features("broken", "imf_linear", verbose=False)
    assert not table.loc[4, "decomp_ok"] and table.loc[4, features.feature_columns(table.columns)].isna().all()


def test_features_tables(tiny_dataset):
    run_decompose(tiny_dataset, "emd", workers=1, chunk_size=16, verbose=False)
    for tag in ("raw", "emd"):
        run_features(tiny_dataset, tag, verbose=False)
        table = load_features(tiny_dataset, tag)
        slots = dec.n_slots(tag, 256)
        columns = features.feature_columns(table.columns)
        assert len(columns) == 2 * slots * len(features.FEATURE_NAMES)
        assert columns[0] == "ch0_c0_energy_frac" and "ch1_log_scale" not in columns
        assert "ch1_log_scale" in features.feature_columns(table.columns, include_log_scale=True)
        assert np.all(np.isfinite(table[columns + ["ch0_log_scale", "ch1_log_scale"]].to_numpy()))
        for channel in range(2):
            fractions = table[[f"ch{channel}_c{k}_energy_frac" for k in range(slots)]].sum(axis=1)
            np.testing.assert_allclose(fractions, 1.0, rtol=1e-5)


def test_component_features_edge_cases():
    rng = np.random.default_rng(0)
    rows = np.stack([
        np.zeros(512),
        np.full(512, 3.0),
        1e-9 * rng.normal(size=512),
        rng.normal(size=512),
        np.sin(2 * np.pi * 0.1 * np.arange(512)),
        (rng.random(512) < 0.02) * 50.0,
    ])
    values = features.component_features(rows)
    names = list(features.FEATURE_NAMES)
    assert values.shape == (6, len(names)) and np.all(np.isfinite(values))
    shape_columns = [names.index(name) for name in features.SHAPE_FEATURES]
    assert np.all(values[:3][:, shape_columns] == 0)
    tone = values[4]
    assert abs(tone[names.index("spec_peak")] - 0.1) < 0.002
    assert abs(tone[names.index("hilb_if_mean")] - 0.1) < 0.002
    assert abs(tone[names.index("zcr")] - 0.2) < 0.01
    assert 0.9 < values[3, names.index("perm_entropy")] <= 1.0
    assert values[5, names.index("kurt")] > values[3, names.index("kurt")] + 10


def test_splits_no_leakage_and_reuse(tiny_dataset):
    splits = cv.make_splits(tiny_dataset, "cls", verbose=False)
    meta = pd.read_parquet(splits_path(tiny_dataset, "cls").parent / "prepared" / "meta.parquet")
    groups = meta["group"].to_numpy()
    assert len(splits) == 15
    for split in splits:
        assert not set(groups[split.train]) & set(groups[split.test])
        assert len(split.train) + len(split.test) == 40
    again = cv.make_splits(tiny_dataset, "cls", verbose=False)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(splits, again))
    meta.loc[0, "group"] = "g9"
    meta.to_parquet(splits_path(tiny_dataset, "cls").parent / "prepared" / "meta.parquet", index=False)
    with pytest.raises(RuntimeError):
        cv.load_splits(tiny_dataset, "cls")


def test_evaluate_returns_per_split_metrics(tiny_dataset):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    run_features(tiny_dataset, "raw", verbose=False)
    table = load_features(tiny_dataset, "raw")
    X = table[features.feature_columns(table.columns)].to_numpy()
    y = pd.read_parquet(splits_path(tiny_dataset, "cls").parent / "prepared" / "meta.parquet")["y_cls"].to_numpy()
    splits = cv.make_splits(tiny_dataset, "cls", verbose=False)
    result, predictions = cv.evaluate(lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
                                      X, y, splits, return_predictions=True)
    assert len(result) == 15 and {"balanced_accuracy", "macro_f1", "accuracy", "recall_a"} <= set(result)
    assert result["balanced_accuracy"].mean() > 0.9
    assert len(predictions) == 3 * 40


def test_corrected_ttest_matches_formula():
    d = np.array([0.02, 0.03, -0.01, 0.04, 0.01, 0.02, 0.0, 0.03, 0.02, 0.01, 0.05, -0.02, 0.02, 0.03, 0.01])
    out = stats.corrected_resampled_ttest(d, n_train=np.full(15, 800), n_test=np.full(15, 200))
    se = np.sqrt((1 / 15 + 0.25) * d.var(ddof=1))
    assert np.isclose(out["t"], d.mean() / se) and out["df"] == 14
    from scipy import stats as scipy_stats

    assert np.isclose(out["p_nb"], 2 * scipy_stats.t.sf(abs(d.mean() / se), 14))
    assert out["ci_low"] < d.mean() < out["ci_high"]
    naive = scipy_stats.ttest_1samp(d, 0).pvalue
    assert out["p_nb"] > naive
    assert stats.wilcoxon_signed_rank(np.zeros(15))["p_wilcoxon"] == 1.0


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5, 0.2])
    np.testing.assert_allclose(stats.holm(p), multipletests(p, method="holm")[1])
    assert np.isnan(stats.holm([0.1, np.nan])[1])


def test_compare_methods_table():
    rows = []
    rng = np.random.default_rng(0)
    for method, shift in (("emd", 0.0), ("imf_linear_a1.41", 0.02), ("irmf_h2_a1.41", 0.05), ("raw", -0.05)):
        for split_id in range(15):
            rows.append({"method": method, "classifier": "lda", "split_id": split_id, "metric": "balanced_accuracy",
                         "value": 0.7 + shift + rng.normal(0, 0.01), "n_train": 800, "n_test": 200})
    table = stats.compare_methods(pd.DataFrame(rows))
    assert len(table) == 6
    first = table.iloc[0]
    assert (first["method_a"], first["method_b"]) == ("irmf_h2_a1.41", "emd")
    assert 0.03 < first["mean_diff"] < 0.07 and first["p_nb_holm"] >= first["p_nb"]
    assert json.dumps(table.to_dict("records"), default=float)
