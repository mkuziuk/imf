"""Resumable paired classical and LightGBM fits, without shared runner writes."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from baselines.run_classical import _pipeline, classifiers
from components.construction import BASE_TAGS, all_tags
from components.integrity import DATA_ROOT, DATASETS, check_pimf, record_entry
from components.storage import STUDY, code_hash, file_hash, run_dir, write_json, write_npz
from emdirmf.cv import fit_predict_split, load_splits, split_metrics
from emdirmf.features import feature_columns
from emdirmf.io import label_column, load_info, load_meta
from emdirmf.parallel import limit_threads, single_thread_env
from ml.run_trees import fit_tree
from sklearn.svm import LinearSVC

CLASSICAL = {"synthetic": "logreg", "dronerf": "linsvm",
             "birdclef2023": "logreg", "esc50_animals": "linsvm"}
FEATURE_SETS = ("comp", "comp+log_scale")


def classical_estimator(dataset):
    if dataset == "dronerf":
        return _pipeline(LinearSVC(C=0.1, dual="auto", max_iter=5000))
    return classifiers()[CLASSICAL[dataset]]


def feature_path(dataset, tag):
    return ((DATA_ROOT / dataset / "features") if tag in BASE_TAGS
            else (run_dir(dataset) / "features")) / f"{tag}.parquet"


def array_path(dataset, tag, feature_set, classifier):
    dtype = "float32" if classifier == "lightgbm" else "float64"
    return run_dir(dataset) / "model_inputs" / f"{tag}__{feature_set}__{dtype}.npy"


@lru_cache(maxsize=4)
def population(dataset):
    info, meta = load_info(dataset), load_meta(dataset)
    task = info["primary_task"]
    y = meta[label_column(info, task)].astype(str).to_numpy(dtype=str)
    groups = meta.group.astype(str).to_numpy(dtype=str)
    splits = load_splits(dataset, task)
    assert len(splits) == 15 and {s.seed for s in splits} == {0, 1, 2}
    return task, y, groups, {s.split_id: s for s in splits}


def fit_job(dataset, tag, feature_set, classifier, split_id):
    check_pimf()
    directory = run_dir(dataset) / "fits" / classifier / feature_set / tag
    checkpoint = directory / f"{split_id:02d}.json"
    predictions = directory / f"{split_id:02d}.npz"
    if checkpoint.exists() and predictions.exists():
        return json.loads(checkpoint.read_text()), True
    if checkpoint.exists() or predictions.exists():
        raise RuntimeError(f"incomplete immutable fit checkpoint: {checkpoint}")
    task, y, groups, splits = population(dataset)
    split = splits[split_id]
    X = np.load(array_path(dataset, tag, feature_set, classifier), mmap_mode="r", allow_pickle=False)
    cpu, wall = time.process_time(), time.perf_counter()
    if classifier != "lightgbm":
        row, pred = fit_predict_split(classical_estimator(dataset), X, y, split,
                                      groups=groups, labels=sorted(set(y)))
    else:
        encoder = LabelEncoder().fit(y)
        encoded = encoder.transform(y)
        started = time.perf_counter()
        model, imputer, params = fit_tree("lightgbm", SimpleNamespace(workers=1, quick=False),
                                          X, encoded, groups, split)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pred = encoder.inverse_transform(model.predict(imputer.transform(X[split.test])).astype(int))
        row = {"split_id": split.split_id, "seed": split.seed, "fold": split.fold,
               "n_train": len(split.train), "n_test": len(split.test),
               "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - started,
               "best_params": json.dumps(params)}
        row.update(split_metrics(y[split.test], pred, sorted(set(y))))
    row.update(dataset=dataset, task=task, method=tag, classifier=classifier,
               feature_set=feature_set, n_features=int(X.shape[1]),
               cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
    check_pimf()
    write_npz(predictions, frame_id=split.test, y_true=y[split.test].astype(str), y_pred=pred.astype(str))
    write_json(checkpoint, row)
    return row, False


def prepare_manifest(dataset, tags, phase):
    task, y, _, _ = population(dataset)
    paths = [Path(__file__), STUDY / "baselines/run_classical.py", STUDY / "ml/run_trees.py",
             STUDY / "ml/common.py", STUDY / "emdirmf/cv.py", STUDY / "emdirmf/features.py"]
    sources = {"meta": file_hash(DATA_ROOT / dataset / "prepared/meta.parquet"),
               "splits": file_hash(DATA_ROOT / dataset / f"splits_{task}.json"),
               "features": {tag: file_hash(feature_path(dataset, tag)) for tag in tags}}
    manifest = {"dataset": dataset, "phase": phase, "tags": tags, "code": code_hash(paths),
                "sources": sources, "classical": CLASSICAL[dataset],
                "feature_sets": list(FEATURE_SETS), "lightgbm_seeds": [0, 1, 2]}
    path = run_dir(dataset) / f"classification_manifest_{phase}.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("classification manifest changed; do not reuse stale fits")
    else:
        write_json(path, manifest)
    for tag in tags:
        table = pd.read_parquet(feature_path(dataset, tag))
        assert np.array_equal(table.frame_id, np.arange(len(y)))
        assert table.decomp_ok.all()
        for feature_set in FEATURE_SETS:
            columns = feature_columns(table.columns, include_log_scale=feature_set.endswith("log_scale"))
            for classifier in (CLASSICAL[dataset], "lightgbm"):
                target = array_path(dataset, tag, feature_set, classifier)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                values = table[columns].to_numpy(dtype=np.float32 if classifier == "lightgbm" else np.float64)
                assert np.all(np.isfinite(values))
                with target.open("xb") as handle:
                    np.save(handle, values, allow_pickle=False)


def verify_defaults(dataset):
    task, _, _, _ = population(dataset)
    original = STUDY.parents[1] / "output/classification-emd-irmf" / dataset / "classical" / f"results_folds_{task}.csv"
    stored = pd.read_csv(original)
    stored = stored[(stored.classifier == CLASSICAL[dataset]) & (stored.subset == "all")]
    records = []
    sources = [original]
    ablation = original.parent.parent / "classical_ablations" / f"ablation_folds_{task}.csv"
    if ablation.exists():
        conv = pd.read_csv(ablation)
        conv = conv[(conv.classifier == CLASSICAL[dataset]) & (conv.feature_set.isin(FEATURE_SETS)) &
                    (conv.method == "irmf_h2_a1.41_conv") & (conv.subset == "all")]
        assert len(conv) == 30
        stored = pd.concat([stored, conv], ignore_index=True)
        sources.append(ablation)
    for row in stored.itertuples(index=False):
        checkpoint = run_dir(dataset) / "fits" / CLASSICAL[dataset] / row.feature_set / row.method / f"{row.split_id:02d}.json"
        current = json.loads(checkpoint.read_text())
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            difference = current[metric] - getattr(row, metric)
            records.append({"dataset": dataset, "method": row.method, "feature_set": row.feature_set,
                            "split_id": row.split_id, "metric": metric, "stored": getattr(row, metric),
                            "recomputed": current[metric], "difference": difference})
    assert len(records) == (450 if ablation.exists() else 360)
    worst = max(abs(row["difference"]) for row in records)
    audit = {"dataset": dataset, "historical_sources": {str(path): file_hash(path) for path in sources},
             "n_metric_checks": len(records), "max_absolute_difference": worst,
             "passed": worst < 1e-12, "records": records}
    target = run_dir(dataset) / "default_validation.json"
    if target.exists():
        if json.loads(target.read_text()) != audit:
            raise RuntimeError("default validation changed")
    else:
        write_json(target, audit)
    if not audit["passed"]:
        raise RuntimeError(f"STOP: {dataset} default scores disagree with phase 2, max difference {worst}")
    print(f"{dataset}: all {len(records)} default metrics agree, max difference {worst:.3g}", flush=True)


def run(dataset, phase, workers):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be 1..6")
    record_entry(dataset, f"classification_{phase}")
    single_thread_env()
    limit_threads()
    if phase == "experiment":
        validation = json.loads((run_dir(dataset) / "default_validation.json").read_text())
        if not validation["passed"]:
            raise RuntimeError("defaults must agree before classifying new inputs")
    tags = list(BASE_TAGS) if phase == "defaults" else [tag for tag in all_tags() if tag not in BASE_TAGS]
    prepare_manifest(dataset, tags, phase)
    _, _, _, splits = population(dataset)
    started = time.perf_counter()
    jobs = [(dataset, tag, feature_set, classifier, split_id)
            for classifier in (CLASSICAL[dataset], "lightgbm")
            for tag in tags for feature_set in FEATURE_SETS for split_id in splits]
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=limit_threads) as pool:
        futures = [pool.submit(fit_job, *job) for job in jobs]
        for future in as_completed(futures):
            row, reused = future.result()
            completed += 1
            if completed % 30 == 0 or completed == len(jobs):
                print(f"{dataset}/{phase}: {completed}/{len(jobs)} fits, "
                      f"elapsed {time.perf_counter()-started:.1f}s, last={row['method']}, reused={reused}", flush=True)
    if phase == "defaults":
        verify_defaults(dataset)
    write_json(run_dir(dataset) / f"classification_completed_{phase}_{time.time_ns()}.json",
               {"dataset": dataset, "phase": phase, "wall_seconds": time.perf_counter() - started,
                "workers": workers, "fits": len(jobs)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--phase", choices=("defaults", "experiment"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset, args.phase, args.workers)


if __name__ == "__main__":
    main()
