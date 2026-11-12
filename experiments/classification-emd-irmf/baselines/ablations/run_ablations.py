"""Isolated phase-2b fits, with read-only reuse of verified phase-2 folds.

Run from the worktree with its .venv Python. No shared module is patched.
All estimator definitions come from the unchanged phase-2 driver. Native OOF
probabilities, warnings, exact split indices and estimator signatures are saved.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV
from threadpoolctl import threadpool_limits

REPO = Path("/Users/mikhail/.t3/worktrees/imf/t3-44ed3594")
STUDY = REPO / "experiments/classification-emd-irmf"
DATA = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
OUTPUT = REPO / "output/classification-emd-irmf"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "baselines"))

import run_classical as shared
from emdirmf.cv import load_splits, split_metrics
from emdirmf.features import feature_columns
from emdirmf.io import label_column, load_info, load_meta

MAIN = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41"]
TAGS = ["irmf_h2_a1.41_conv", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv",
        "irmf_h2_a2_conv", "imf_linear_a2"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model):
    return repr(model.get_params(deep=True))


def models_for(dataset, task, names):
    models = shared.classifiers()
    context = json.loads((OUTPUT / dataset / "classical" / f"run_{task}.json").read_text())
    if dataset == "birdclef2023":
        assert context["profile"] == "narrow"
        models["linsvm"].param_grid = context["grids"]["linsvm"]
    if dataset == "dronerf":
        assert (OUTPUT / dataset / "classical/REPORT.md").exists()
        for name, grid in context["grids"].items():
            fixed = context.get("fixed_parameters", {}).get(name)
            if fixed:
                models[name] = models[name].estimator.set_params(**fixed)
            elif grid is not None:
                models[name].param_grid = grid
    for name in names:
        assert name in context["classifiers"]
        fixed = context.get("fixed_parameters", {}).get(name)
        if fixed:
            assert all(models[name].get_params()[p] == value for p, value in fixed.items())
        elif "grids" in context:
            assert getattr(models[name], "param_grid", None) == context["grids"][name]
    assert context["n_splits"] == 15 and not context["quick"]
    assert context["methods"] == MAIN
    return {name: models[name] for name in names}, context


def phase2_predictions(dataset, task):
    folder = DATA / dataset / "results"
    if dataset == "esc50_animals":
        folder /= "phase2_main"
    return folder / f"classical_{task}_predictions.parquet"


def verify_reuse(dataset, task, models, splits, y, labels):
    names = list(models)
    source = OUTPUT / dataset / "classical" / f"results_folds_{task}.csv"
    folds = pd.read_csv(source, float_precision="round_trip")
    folds = folds[folds.classifier.isin(names) & folds.method.isin(MAIN)].copy()
    assert not folds.duplicated(["method", "feature_set", "classifier", "split_id"]).any()
    predictions_path = phase2_predictions(dataset, task)
    predictions = pd.read_parquet(predictions_path)
    predictions = predictions[predictions.classifier.isin(names) & predictions.method.isin(MAIN)]
    checked = 0
    if dataset == "dronerf":
        context = json.loads((OUTPUT / dataset / "classical" / f"run_{task}.json").read_text())
        profile = "reduced" if context["reduced"] else "full"
        manifest_path = (DATA / dataset / "work/classical"
                         / f"checkpoints_{profile}_{task}" / "manifest.json")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["grids"] == context["grids"]
        assert manifest["fixed_parameters"] == context.get("fixed_parameters", {})
        sources = [STUDY / "baselines/run_classical.py", DATA / dataset / f"splits_{task}.json"]
        sources += [DATA / dataset / "features" / f"{tag}.parquet" for tag in MAIN]
        for source in sources:
            assert sha256(source) == manifest["sha256"][str(source)]
    for (method, feature_set, name), cell in folds.groupby(["method", "feature_set", "classifier"], observed=True):
        assert sorted(cell.split_id) == sorted(s.split_id for s in splits)
        pred = predictions[(predictions.method == method) & (predictions.feature_set == feature_set)
                           & (predictions.classifier == name)]
        for split in splits:
            row = cell[cell.split_id == split.split_id].iloc[0]
            assert (row.seed, row.fold, row.n_train, row.n_test) == (
                split.seed, split.fold, len(split.train), len(split.test))
            held = pred[pred.split_id == split.split_id].sort_values("frame_id")
            order = np.argsort(split.test)
            assert np.array_equal(held.frame_id.to_numpy(), split.test[order])
            assert np.array_equal(held.y_true.astype(str).to_numpy(), y[split.test[order]])
            measured = split_metrics(held.y_true.astype(str).to_numpy(), held.y_pred.astype(str).to_numpy(), labels)
            for metric in METRICS:
                assert np.isclose(row[metric], measured[metric], atol=1e-12)
            if dataset == "esc50_animals":
                feature_path = DATA / dataset / "features" / f"{method}.parquet"
                signature = hashlib.sha256((repr(models[name].get_params())
                                           + str(feature_path.stat().st_mtime_ns)
                                           + repr(split.test.tolist())).encode()).hexdigest()[:12]
                checkpoint = (DATA / dataset / "results/phase2_main" / task / "jobs"
                              / f"{method}__{feature_set}__{name}__{split.split_id}__{signature}.npz")
                assert checkpoint.exists(), "phase-2 estimator/features/split signature differs"
            checked += 1
    assert checked == len(MAIN) * 2 * len(names) * 15
    folds["source"] = "phase2_reused"
    return folds, {"fold_file": str(source), "fold_sha256": sha256(source),
                   "prediction_file": str(predictions_path), "verified_fits": checked,
                   "split_sha256": sha256(DATA / dataset / f"splits_{task}.json"),
                   "driver_sha256": sha256(STUDY / "baselines/run_classical.py")}


def ready_table(dataset, tag, meta):
    folder = DATA / dataset
    paths = [folder / "decomp" / f"{tag}.json", folder / "features" / f"{tag}.json",
             folder / "features" / f"{tag}.parquet"]
    if not all(p.exists() for p in paths):
        return None, {"tag": tag, "ready": False, "reason": "missing JSON or features"}
    decomp, features = [json.loads(p.read_text()) for p in paths[:2]]
    assert decomp["tag"] == features["tag"] == tag
    assert not decomp["partial"] and decomp["shape"][0] == len(meta)
    assert features["n_frames"] == len(meta) and features["n_failed_frames"] == 0
    assert features["n_nonfinite_rows_among_ok"] == 0 and decomp["n_failed"] == 0
    assert features["source"]["frames_fingerprint"] == decomp["frames_fingerprint"]
    if tag.endswith("_conv"):
        assert decomp["irmf"]["channels_all_converged"] == np.prod(decomp["shape"][:2])
        assert decomp["params"]["root_tol"] == 1e-12
        assert decomp["params"]["root_max_iter"] == 100
    reference_tag = "irmf_h2_a1.41" if tag.startswith("irmf") else "imf_linear_a1.41"
    reference = json.loads((folder / "decomp" / f"{reference_tag}.json").read_text())
    assert decomp["frames_fingerprint"] == reference["frames_fingerprint"]
    for parameter in ("h1", "h_min_rule", "kernel", "boundary", "workers"):
        assert decomp["params"][parameter] == reference["params"][parameter]
    expected_a = 2 if "_a2" in tag else np.sqrt(2)
    assert np.isclose(decomp["params"]["a"], expected_a)
    if tag.startswith("irmf"):
        assert decomp["params"]["h_ratio"] == float(tag.split("_h")[1].split("_a")[0])
    assert features["n_slots"] == decomp["shape"][2]
    table = pd.read_parquet(paths[2])
    assert np.array_equal(table.frame_id.to_numpy(), meta.frame_id.to_numpy())
    assert table.decomp_ok.all()
    assert np.isfinite(table[feature_columns(table.columns, include_log_scale=True)].to_numpy()).all()
    return table, {"tag": tag, "ready": True, "slots": decomp["shape"][2],
                   "cpu_seconds_per_frame": decomp["seconds_per_frame"]["cpu_mean"],
                   "features_sha256": sha256(paths[2]), "decomposition_sha256": sha256(paths[0])}


def fit_job(dataset, task, key, model, matrix, y, split, groups, labels, signature):
    tag, feature_set, name = key
    folder = DATA / dataset / "results/phase2b" / task / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{tag}__{feature_set}__{name}__{split.split_id:02d}__{signature}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved["frame_id"], split.test)
            assert np.array_equal(saved["y_true"], y[split.test])
            return json.loads(str(saved["row"]))
    np.random.seed(0)
    fitted = clone(model)
    X = np.load(matrix, mmap_mode="r")
    params = {"groups": groups[split.train]} if isinstance(fitted, BaseSearchCV) else {}
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        started = time.perf_counter()
        fitted.fit(X[split.train], y[split.train], **params)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pred = fitted.predict(X[split.test])
        predict_seconds = time.perf_counter() - started
        probability = fitted.predict_proba(X[split.test]) if hasattr(fitted, "predict_proba") else np.empty((len(pred), 0))
    if probability.shape[1]:
        assert np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=1), 1)
        assert np.array_equal(np.asarray(fitted.classes_)[probability.argmax(axis=1)], pred)
    messages = sorted(set(str(w.message) for w in caught))
    row = dict(dataset=dataset, task=task, method=tag, feature_set=feature_set, classifier=name,
               subset="all", source="phase2b", split_id=split.split_id, seed=split.seed, fold=split.fold,
               n_train=len(split.train), n_test=len(split.test), fit_seconds=fit_seconds,
               predict_seconds=predict_seconds, best_params=json.dumps(getattr(fitted, "best_params_", None)),
               warning_count=len(caught), warning_messages=json.dumps(messages))
    selected = getattr(fitted, "best_estimator_", fitted).named_steps["clf"]
    if hasattr(selected, "n_iter_"):
        row["selected_n_iter"] = int(np.max(selected.n_iter_))
        row["selected_max_iter"] = int(selected.max_iter)
    row.update(split_metrics(y[split.test], pred, labels))
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row=json.dumps(row), frame_id=split.test, y_true=y[split.test].astype(str),
                            y_pred=pred.astype(str), probabilities=probability,
                            classes=np.asarray(fitted.classes_, dtype=str), model_signature=model_signature(model))
    temporary.replace(path)
    print(f"done {dataset}/{task} {tag} {feature_set} {name} {split.split_id} "
          f"BA={row['balanced_accuracy']:.4f} {fit_seconds:.1f}s", flush=True)
    return row


def run(dataset, task=None, names=("lda", "logreg", "linsvm"), workers=4, include_gd_a2=False):
    started = time.perf_counter()
    assert 1 <= workers <= 4
    if dataset == "dronerf":
        assert (OUTPUT / dataset / "classical/REPORT.md").exists(), "DroneRF still running"
    info, meta = load_info(dataset), load_meta(dataset)
    task = task or ("10class" if dataset == "dronerf" else info["primary_task"])
    y = meta[label_column(info, task)].astype(str).to_numpy()
    groups = meta.group.astype(str).to_numpy()
    labels = sorted(set(y))
    splits = load_splits(dataset, task)
    models, phase2_context = models_for(dataset, task, names)
    baseline, audit = verify_reuse(dataset, task, models, splits, y, labels)
    folder = DATA / dataset / "results/phase2b" / task
    out = OUTPUT / dataset / "classical_ablations"
    folder.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out / f"reused_folds_{task}.csv", index=False)
    jobs, caches = [], []
    reference_scale = pd.read_parquet(DATA / dataset / "features/raw.parquet")
    for tag in TAGS + ["irmf_h2_a2"]:
        try:
            table, cache = ready_table(dataset, tag, meta)
        except (AssertionError, KeyError, ValueError, OSError) as error:
            if tag != "irmf_h2_a2":
                raise
            table, cache = None, {"tag": tag, "ready": False,
                                  "reason": f"incomplete or inconsistent optional cache: {error}"}
        caches.append(cache)
        if tag == "irmf_h2_a2" and table is not None and not include_gd_a2:
            cache.update(skipped=True, reason="optional GD60 a2 excluded to reserve the primary-tag budget")
            continue
        if table is None:
            assert tag == "irmf_h2_a2", f"required tag incomplete: {tag}"
            continue
        for column in [c for c in table if c.endswith("_log_scale")]:
            assert np.array_equal(table[column].to_numpy(), reference_scale[column].to_numpy())
        for feature_set in shared.FEATURE_SETS:
            columns = feature_columns(table.columns, include_log_scale=feature_set.endswith("log_scale"))
            matrix = folder / f"{tag}__{feature_set}__{cache['features_sha256'][:12]}.npy"
            if not matrix.exists():
                temporary = matrix.with_suffix(".partial")
                with temporary.open("wb") as handle:
                    np.save(handle, table[columns].to_numpy(dtype=np.float64))
                temporary.replace(matrix)
            for name, model in models.items():
                signature = hashlib.sha256((cache["features_sha256"] + model_signature(model)
                                            + audit["split_sha256"] + audit["driver_sha256"]).encode()).hexdigest()[:16]
                for split in splits:
                    jobs.append((dataset, task, (tag, feature_set, name), model, str(matrix),
                                 y, split, groups, labels, signature))
        del table
    context = dict(dataset=dataset, task=task, classifiers=list(names), workers=workers,
                   start=datetime.now(timezone.utc).isoformat(), phase2=phase2_context, reuse=audit,
                   caches=caches, estimator_signatures={name: model_signature(m) for name, m in models.items()},
                   job_count=len(jobs), n_splits=len(splits), protocol="same nested grouped CV; no retuning on test folds")
    (out / f"audit_{task}.json").write_text(json.dumps(context, indent=2))
    print(f"START {dataset}/{task}: {len(jobs)} new fits; {len(baseline)} reused; {workers} workers", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=workers, return_as="generator_unordered", batch_size=1)(
            delayed(fit_job)(*job) for job in jobs)
        rows = list(results)
    new = pd.DataFrame(rows)
    new.to_csv(out / f"ablation_folds_{task}.csv", index=False)
    combined = pd.concat([baseline, new], ignore_index=True)
    combined.to_csv(out / f"results_folds_{task}.csv", index=False)
    context.update(seconds=time.perf_counter() - started, finished=datetime.now(timezone.utc).isoformat())
    (out / f"audit_{task}.json").write_text(json.dumps(context, indent=2))
    print(f"FINISHED {dataset}/{task}: {context['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task")
    parser.add_argument("--classifiers", nargs="+", default=["lda", "logreg", "linsvm"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-secondary", action="store_true",
                        help="Explicitly spend optional budget on a secondary task")
    parser.add_argument("--include-gd-a2", action="store_true")
    args = parser.parse_args()
    primary = "10class" if args.dataset == "dronerf" else load_info(args.dataset)["primary_task"]
    if args.task and args.task != primary and not args.allow_secondary:
        print("SKIP optional secondary task; reserve budget for primary datasets. "
              "Rerun with --allow-secondary when the primary work is covered.", flush=True)
    else:
        run(args.dataset, args.task, args.classifiers, args.workers, args.include_gd_a2)
