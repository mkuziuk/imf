"""New-input classification after the timing-only, pre-specified budget cuts."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from components.budget import eligible, load_plan
from components.classification import (
    CLASSICAL, FEATURE_SETS, fit_job, population, prepare_manifest,
)
from components.construction import BASE_TAGS, all_tags
from components.integrity import DATASETS, check_pimf, record_entry
from components.storage import code_hash, file_hash, run_dir, write_json
from emdirmf.parallel import limit_threads, single_thread_env


def run(dataset, workers):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be 1..6")
    record_entry(dataset, "budgeted_experiment")
    single_thread_env()
    limit_threads()
    path = run_dir("synthetic") / "budget_plan.json"
    if not path.exists():
        raise RuntimeError("Measure all default CPU costs and freeze the budget plan first")
    plan = load_plan()
    assert json.loads((run_dir(dataset) / "default_validation.json").read_text())["passed"]
    tags = [tag for tag in all_tags() if tag not in BASE_TAGS]
    # Original model/feature code is unchanged, preserving all default manifests.
    prepare_manifest(dataset, tags, "experiment")
    manifest = {"budget_sha256": file_hash(path), "code": code_hash([__file__,
                __import__('components.budget', fromlist=['']).__file__]), "workers": workers}
    target = run_dir(dataset) / "budgeted_classification_manifest.json"
    if target.exists():
        assert json.loads(target.read_text()) == manifest
    else:
        write_json(target, manifest)
    _, _, _, splits = population(dataset)
    jobs = []
    for classifier in (CLASSICAL[dataset], "lightgbm"):
        for tag in tags:
            for feature_set in FEATURE_SETS:
                for split in splits.values():
                    if eligible({"classifier": classifier, "method": tag,
                                 "feature_set": feature_set, "seed": split.seed}, plan):
                        jobs.append((dataset, tag, feature_set, classifier, split.split_id))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=limit_threads) as pool:
        futures = [pool.submit(fit_job, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            row, reused = future.result()
            if done % 30 == 0 or done == len(jobs):
                print(f"{dataset}/experiment tier={plan['level']}: {done}/{len(jobs)} fits, "
                      f"elapsed {time.perf_counter()-started:.1f}s, last={row['method']}, reused={reused}", flush=True)
    check_pimf()
    write_json(run_dir(dataset) / f"classification_completed_experiment_{time.time_ns()}.json",
               {"dataset": dataset, "phase": "experiment", "wall_seconds": time.perf_counter()-started,
                "workers": workers, "fits": len(jobs), "budget_level": plan["level"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset, args.workers)


if __name__ == "__main__":
    main()
