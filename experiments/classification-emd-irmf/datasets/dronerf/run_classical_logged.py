"""Run the unchanged classical runner with DroneRF-only fit checkpoints.

Defaults and grids are unchanged unless --reduced-grid is explicitly supplied. Checkpoints can be reused after an
interruption. They are keyed by method, feature set, classifier, and split.
"""
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
STUDY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR / "baselines"))
import numpy as np
import run_classical as baseline
from emdirmf.cv import fit_predict_split


def checkpoint_job(key, estimator, x, y, split, groups, labels):
    directory = Path(os.environ["DRONERF_CLASSICAL_CHECKPOINTS"])
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{key[0]}__{key[1]}__{key[2]}__{split.split_id}"
    json_path = Path(str(stem) + ".json")
    pred_path = Path(str(stem) + ".npy")
    if json_path.exists() and pred_path.exists():
        return key, json.loads(json_path.read_text()), np.load(pred_path, allow_pickle=False)
    row, pred = fit_predict_split(estimator, x, y, split, groups=groups, labels=labels)
    np.save(pred_path, np.asarray(pred).astype(str), allow_pickle=False)
    temporary = Path(str(json_path) + ".tmp")
    temporary.write_text(json.dumps(row))
    temporary.replace(json_path)
    print(f"finished {key} split={split.split_id} fit={row['fit_seconds']:.2f}s "
          f"BA={row['balanced_accuracy']:.4f}", flush=True)
    return key, row, pred


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("DroneRF wrapper: --reduced-grid narrows inner tuning; all outer splits remain.")
        return baseline.main([arg for arg in sys.argv[1:] if arg != "--reduced-grid"])
    if "--dataset" not in sys.argv or sys.argv[sys.argv.index("--dataset") + 1] != "dronerf":
        raise ValueError("This wrapper is restricted to dronerf")
    task = sys.argv[sys.argv.index("--task") + 1]
    reduced = "--reduced-grid" in sys.argv
    if reduced:
        sys.argv.remove("--reduced-grid")
        original_classifiers = baseline.classifiers

        def reduced_classifiers():
            models = original_classifiers()
            models["qda"].param_grid = {"clf__shrinkage": ["auto", 0.2, 0.8]}
            models["logreg"].param_grid = {"clf__C": [0.01, 0.1, 1.0]}
            models["knn"].param_grid = {"clf__n_neighbors": [1, 15]}
            models["linsvm"].param_grid = {"clf__C": [0.01, 0.1]}
            if task == "10class":
                # One candidate: skipping the inner fits leaves the final model unchanged.
                models["linsvm"] = models["linsvm"].estimator.set_params(clf__C=0.1)
            return models

        baseline.classifiers = reduced_classifiers
    cache = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
    directory = cache / "dronerf/work/classical" / f"checkpoints_{'reduced' if reduced else 'full'}_{task}"
    os.environ["DRONERF_CLASSICAL_CHECKPOINTS"] = str(directory)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(cache / "dronerf/work/classical/tmp")
    directory.mkdir(parents=True, exist_ok=True)
    baseline.make_splits("dronerf", task)
    grids = {name: getattr(model, "param_grid", None) for name, model in baseline.classifiers().items()}
    fixed_parameters = {"linsvm": {"clf__C": 0.1}} if reduced and task == "10class" else {}
    for name, parameters in fixed_parameters.items():
        grids[name] = {key: [value] for key, value in parameters.items()}
    dataset_dir = cache / "dronerf"
    inputs = [dataset_dir / "prepared/meta.parquet", dataset_dir / f"splits_{task}.json",
              Path(__file__), STUDY_DIR / "baselines/run_classical.py", STUDY_DIR / "emdirmf/cv.py",
              STUDY_DIR / "emdirmf/features.py"]
    inputs += [dataset_dir / f"features/{tag}.parquet" for tag in baseline.DEFAULT_METHODS]
    signature = {"grids": grids, "fixed_parameters": fixed_parameters, "sklearn": version("scikit-learn"), "numpy": version("numpy"),
                 "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}}
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise RuntimeError(f"Stale checkpoints in {directory}; clear this task cache before rerunning")
    manifest.write_text(json.dumps(signature, indent=2))
    (directory / "grid.json").write_text(json.dumps({
        "task": task, "reduced": reduced, "grids": grids,
        "fixed_parameters": fixed_parameters, "skipped_single_candidate_cv": list(fixed_parameters),
    }, indent=2))
    baseline.Parallel = partial(baseline.Parallel, batch_size=1)
    baseline._job = checkpoint_job
    result = baseline.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else baseline.output_dir("dronerf", "classical")
    context_path = out / f"run_{task}.json"
    context = json.loads(context_path.read_text())
    context.update(json.loads((directory / "grid.json").read_text()))
    context["joblib_batch_size"] = 1
    context_path.write_text(json.dumps(context, indent=2))
    return result


if __name__ == "__main__":
    sys.exit(main())
