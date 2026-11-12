#!/usr/bin/env python3
"""Exploratory LightGBM capacity and feature-fusion checks on all 15 folds."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from emdirmf.cv import load_splits, split_metrics
from emdirmf.decompose import scale_order, slot_labels
from emdirmf.features import feature_columns
from emdirmf.io import load_meta
from emdirmf.paths import features_dir, output_dir, results_dir
from emdirmf.stats import holm, paired_comparison
from ml.run_trees import fit_tree

OUT = output_dir("birdclef2023", "ml", "feature_checks")
IRMF = "irmf_h2_a1.41"
CONV = IRMF + "_conv"


def run():
    args = argparse.Namespace(quick=False, workers=2)
    threadpool_limits(args.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta("birdclef2023")
    y = meta.y_species.astype(str).to_numpy()
    groups = meta.group.astype(str).to_numpy()
    encoder = LabelEncoder().fit(y)
    encoded = encoder.transform(y)
    tables = {tag: pd.read_parquet(features_dir("birdclef2023") / f"{tag}.parquet")
              for tag in ("emd", IRMF, CONV)}
    for table in tables.values():
        np.testing.assert_array_equal(table.frame_id, meta.frame_id)
        assert table.decomp_ok.all()
    splits = load_splits("birdclef2023", "species")
    rows, predictions, details = [], [], []
    started = time.perf_counter()
    for tag in (IRMF, CONV):
        order = scale_order(tag, 2048)
        ranks = np.rint(np.linspace(0, len(order) - 1, 9)).astype(int)
        slots = [order[rank] for rank in ranks]
        all_columns = feature_columns(tables[tag].columns)
        selected = [c for c in all_columns if int(re.search(r"_c(\d+)_", c).group(1)) in slots]
        emd_columns = feature_columns(tables["emd"].columns)
        assert len(selected) == len(emd_columns) == 180
        details.append({"source": tag, "selected_ranks": ranks.tolist(), "storage_slots": slots,
                        "slot_labels": [slot_labels(tag, 2048)[slot] for slot in slots]})
        for kind in ("matched9", "emd_fusion"):
            columns = selected if kind == "matched9" else all_columns
            base = tables[tag][columns].to_numpy(dtype=np.float32)
            if kind == "emd_fusion":
                base = np.column_stack([tables["emd"][emd_columns].to_numpy(dtype=np.float32), base])
            method = tag + "_" + kind
            for with_scale in (False, True):
                feature_set = "comp+log_scale" if with_scale else "comp"
                X = (np.column_stack([base, tables[tag][["ch0_log_scale"]]]).astype(np.float32)
                     if with_scale else base.copy())
                X[~np.isfinite(X)] = np.nan
                for split in splits:
                    tick = time.perf_counter()
                    model, imputer, params = fit_tree("lightgbm", args, X, encoded, groups, split)
                    probabilities = model.predict_proba(imputer.transform(X[split.test]))
                    pred = encoder.inverse_transform(probabilities.argmax(axis=1))
                    row = {"method": method, "feature_set": feature_set, "classifier": "lightgbm",
                           "split_id": split.split_id, "seed": split.seed, "fold": split.fold,
                           "n_train": len(split.train), "n_test": len(split.test), "n_features": X.shape[1],
                           "seconds": time.perf_counter() - tick, "best_params": json.dumps(params)}
                    row.update(split_metrics(y[split.test], pred, encoder.classes_))
                    rows.append(row)
                    predictions.append(pd.DataFrame({"method": method, "feature_set": feature_set,
                        "split_id": split.split_id, "frame_id": split.test, "y_true": y[split.test],
                        "y_pred": pred, **{f"prob_{label}": probabilities[:, index]
                                           for index, label in enumerate(encoder.classes_)}}))
                    pd.DataFrame(rows).to_csv(OUT / "checks_folds_species.csv", index=False)
                    print(method, feature_set, split.split_id, row["balanced_accuracy"], row["seconds"], flush=True)
    pd.concat(predictions, ignore_index=True).to_parquet(
        results_dir("birdclef2023") / "feature_checks_species_predictions.parquet", index=False)
    (OUT / "config.json").write_text(json.dumps({"split_ids": [s.split_id for s in splits],
        "workers": 2, "max_rounds": 500, "early_stopping": 12, "seconds": time.perf_counter() - started,
        "selection": "Nine evenly spaced ranks in true fine-to-coarse order, fixed without labels",
        "matched_slots": details, "fusion": "All EMD and IRMF component features; log_scale included once"}, indent=2) + "\n")


def compare_checks(combined):
    metrics = ["balanced_accuracy", "macro_f1", "accuracy"]
    comparisons = []
    for metric in metrics:
        table = []
        for feature_set in ("comp", "comp+log_scale"):
            for tag in (IRMF, CONV):
                for suffix in ("matched9", "emd_fusion"):
                    a = combined[(combined.method == tag + "_" + suffix) & (combined.feature_set == feature_set)].set_index("split_id")
                    for other in ("emd", tag):
                        b = combined[(combined.method == other) & (combined.feature_set == feature_set)].set_index("split_id")
                        assert set(a.index) == set(b.index) == set(range(15))
                        b = b.loc[a.index]
                        row = {"metric": metric, "feature_set": feature_set,
                               "method_a": tag + "_" + suffix, "method_b": other}
                        row.update(paired_comparison(a[metric], b[metric], a.n_train, a.n_test))
                        table.append(row)
        frame = pd.DataFrame(table)
        frame["p_nb_holm"] = holm(frame.p_nb)
        frame["p_wilcoxon_holm"] = holm(frame.p_wilcoxon)
        comparisons.append(frame)
    return pd.concat(comparisons, ignore_index=True)


def analyse():
    baseline = pd.read_csv(OUT.parent / "all_folds_species.csv")
    baseline = baseline[baseline.classifier == "lightgbm"]
    extra = pd.read_csv(OUT / "checks_folds_species.csv")
    combined = pd.concat([baseline, extra], ignore_index=True)
    metrics = ["balanced_accuracy", "macro_f1", "accuracy"]
    combined.groupby(["method", "feature_set"])[metrics].agg(["mean", "std"]).to_csv(OUT / "summary.csv")
    compare_checks(combined).to_csv(OUT / "comparisons.csv", index=False)

    meta = load_meta("birdclef2023")
    labels = np.unique(meta.y_species.astype(str))
    columns = [f"prob_{label}" for label in labels]
    splits = {s.split_id: s for s in load_splits("birdclef2023", "species")}
    predictions = pd.read_parquet(results_dir("birdclef2023") / "feature_checks_species_predictions.parquet")
    keys = ["method", "feature_set", "split_id"]
    assert not predictions.duplicated(keys + ["frame_id"]).any()
    recording_rows, recording_predictions = [], []
    for key, cell in predictions.groupby(keys, sort=False):
        split = splits[key[-1]]
        cell = cell.sort_values("frame_id").copy()
        np.testing.assert_array_equal(cell.frame_id, np.sort(split.test))
        np.testing.assert_array_equal(cell.y_true, meta.iloc[cell.frame_id].y_species.astype(str))
        assert not set(meta.iloc[split.train].group) & set(meta.iloc[split.test].group)
        probabilities = cell[columns].to_numpy()
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-5)
        np.testing.assert_array_equal(labels[probabilities.argmax(axis=1)], cell.y_pred)
        saved = extra
        for column, value in zip(keys, key):
            saved = saved[saved[column] == value]
        assert len(saved) == 1
        expected = split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels)
        for metric in metrics:
            np.testing.assert_allclose(expected[metric], saved.iloc[0][metric], atol=1e-12)
        cell["recording_id"] = meta.iloc[cell.frame_id].recording_id.to_numpy()
        assert cell.groupby("recording_id").y_true.nunique().max() == 1
        averaged = cell.groupby("recording_id")[columns].mean()
        averaged["y_true"] = cell.groupby("recording_id").y_true.first()
        averaged["y_pred"] = labels[averaged[columns].to_numpy().argmax(axis=1)]
        row = dict(zip(keys, key))
        row.update(classifier="lightgbm", seed=split.seed, fold=split.fold,
                   n_train=meta.iloc[split.train].recording_id.nunique(), n_test=len(averaged))
        row.update(split_metrics(averaged.y_true.to_numpy(), averaged.y_pred.to_numpy(), labels))
        recording_rows.append(row)
        for column in ("method", "feature_set", "split_id", "seed", "fold"):
            averaged[column] = row[column]
        recording_predictions.append(averaged.reset_index())
    recorded = pd.DataFrame(recording_rows)
    recorded.to_csv(OUT / "recording_folds.csv", index=False)
    base_recorded = pd.read_csv(OUT.parent / "recording_folds_species.csv")
    base_recorded = base_recorded[base_recorded.classifier == "lightgbm"]
    compare_checks(pd.concat([base_recorded, recorded], ignore_index=True)).to_csv(
        OUT / "recording_comparisons.csv", index=False)
    rec = pd.concat(recording_predictions, ignore_index=True)
    assert not rec.duplicated(["method", "feature_set", "seed", "recording_id"]).any()
    rec.to_parquet(results_dir("birdclef2023") / "feature_checks_species_recording_predictions.parquet", index=False)
    pooled = []
    for key, cell in rec.groupby(["method", "feature_set", "seed"]):
        assert len(cell) == meta.recording_id.nunique()
        row = dict(zip(["method", "feature_set", "seed"], key))
        row.update(n_recordings=len(cell), **split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels))
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(OUT / "recording_pooled_seeds.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps({"status": "passed", "fits": len(extra),
        "frame_predictions": len(predictions), "recording_predictions": len(rec),
        "complete_seed_cells": len(pooled), "recordings_per_seed": meta.recording_id.nunique()}, indent=2) + "\n")
    print(combined.groupby(["method", "feature_set"])[metrics].mean().to_string())


if __name__ == "__main__":
    analyse() if "--analyse" in sys.argv else run()
