"""Run the shared full classical grid, adding resumable OOF probability caches.

No shared code changes. The shared runner otherwise saves hard labels only.
All per-frame caches and logs live outside the repository under DATA_ROOT.
"""

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = STUDY_DIR.parents[1]
DATA_DIR = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification")) / "birdclef2023"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["IMF_CLS_MAX_WORKERS"] = "3"
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR / "baselines"))

import numpy as np
import run_classical as shared
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV
from threadpoolctl import threadpool_limits


def cached_job(key, estimator, X, y, split, groups, labels):
    """Same fit/metrics as the shared worker, retaining native probabilities."""
    task = os.environ["BIRDCLEF_CLASSICAL_TASK"]
    profile = os.environ["BIRDCLEF_CLASSICAL_PROFILE"]
    tag, feature_set, classifier = key
    folder = DATA_DIR / "results" / "classical_phase2" / profile / task
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{tag}__{feature_set}__{classifier}__{split.split_id:02d}.npz"
    # The reduced profile changes only SVM. Other completed pilot fits are exact
    # matches, so reuse them without spending the shared CPU budget twice.
    pilot_path = DATA_DIR / "results" / "classical_phase2" / "full" / task / path.name
    if profile == "narrow" and classifier != "linsvm" and not path.exists() and pilot_path.exists():
        shutil.copyfile(pilot_path, path)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            assert np.array_equal(cached["frame_id"], split.test)
            assert np.array_equal(cached["y_true"], y[split.test])
            return key, json.loads(str(cached["row"])), cached["y_pred"]
    model = clone(estimator)
    fit_params = {"groups": groups[split.train]} if isinstance(model, BaseSearchCV) else {}
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        model.fit(X[split.train], y[split.train], **fit_params)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = model.predict(X[split.test])
        predict_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(X[split.test]) if hasattr(model, "predict_proba") else np.empty((len(split.test), 0))
    row = {
        "split_id": split.split_id, "seed": split.seed, "fold": split.fold,
        "n_train": len(split.train), "n_test": len(split.test),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "best_params": json.dumps(getattr(model, "best_params_", None), default=str),
    }
    row.update(shared.split_metrics(y[split.test], y_pred, labels))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, frame_id=split.test, y_true=y[split.test].astype(str),
                        y_pred=np.asarray(y_pred).astype(str), probabilities=probabilities,
                        classes=np.asarray(model.classes_).astype(str), row=json.dumps(row))
    os.replace(temporary, path)
    print(f"done {task} {tag} {feature_set} {classifier} split={split.split_id} "
          f"BA={row['balanced_accuracy']:.4f} fit={fit_seconds:.1f}s", flush=True)
    return key, row, np.asarray(y_pred)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", choices=["species", "passerine"], required=True)
    parser.add_argument("--profile", choices=["full", "narrow"], default="full")
    args = parser.parse_args()
    os.environ["BIRDCLEF_CLASSICAL_TASK"] = args.task
    os.environ["BIRDCLEF_CLASSICAL_PROFILE"] = args.profile
    scratch = DATA_DIR / "results" / "classical_phase2" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    shared.tempfile.tempdir = str(scratch)
    shared._job = cached_job
    if args.profile == "narrow":
        original_classifiers = shared.classifiers

        def narrowed_classifiers():
            models = original_classifiers()
            models["linsvm"].param_grid = {"clf__C": [0.001, 0.01]}
            return models

        shared.classifiers = narrowed_classifiers
    out = ROOT_DIR / "output" / "classification-emd-irmf" / "birdclef2023" / "classical"
    shared.run("birdclef2023", task=args.task, methods=shared.DEFAULT_METHODS,
               workers=3, quick=False, out=out)
    run_path = out / f"run_{args.task}.json"
    context = json.loads(run_path.read_text())
    pilot_folder = DATA_DIR / "results" / "classical_phase2" / "full" / args.task
    eligible = sum("__linsvm__" not in path.name and ".tmp." not in path.name for path in pilot_folder.glob("*.npz"))
    context.update(profile=args.profile, probability_cache=str(DATA_DIR / "results" / "classical_phase2" / args.profile / args.task),
                   grids={name: getattr(model, "param_grid", None) for name, model in shared.classifiers().items()},
                   pilot_reuse_enabled=args.profile == "narrow", eligible_pilot_checkpoints=eligible)
    run_path.write_text(json.dumps(context, indent=2))


if __name__ == "__main__":
    main()
