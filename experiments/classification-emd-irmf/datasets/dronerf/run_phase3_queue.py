#!/usr/bin/env python3
"""Run one fixed-budget DroneRF phase-3 lane, respecting a wall-clock cutoff."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output/classification-emd-irmf/dronerf/ml"
RUN = Path(__file__).with_name("run_phase3.py")
PYTHON = ROOT / ".venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=["lgbm", "other", "rocket", "rocket_full", "cnn", "et_secondary", "auxiliary", "cnn_secondary", "cnn_full"])
    parser.add_argument("--deadline", default="2026-10-21T00:30:00+00:00")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    manifest = json.loads((OUT / "jobs.json").read_text())["jobs"]
    by_key = {(j["name"], j["task"]): j for j in manifest}
    tasks = ["10class", "4class", "2class"]
    lanes = {
        "lgbm": [("trees_lgbm", "10class"), ("trees_lonly", "10class"),
                 ("trees_lgbm", "4class"), ("trees_lgbm", "2class")],
        "other": [(name, task) for task in tasks for name in ["trees_et", "trees_xgb"]],
        "rocket": [("rocket", task) for task in tasks],
        "rocket_full": [("rocket_extra", "10class")],
        "cnn": [("cnn", task) for task in tasks],
        "cnn_secondary": [("cnn", task) for task in ["4class", "2class"]],
        "cnn_full": [("cnn_extra", "10class"), ("cnn", "4class"), ("cnn", "2class")],
        "et_secondary": [("trees_et", task) for task in ["4class", "2class"]],
        "auxiliary": [("trees_et", task) for task in ["4class", "2class"]] +
                     [("trees_h1", "10class"), ("trees_h4", "10class")],
    }
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    environment["IMF_CLS_DATA"] = "/Users/mikhail/Datasets/imf-classification"
    completed_seconds = {}
    completed_workers = {}
    completed_units = {}
    events_path = OUT / f"queue_{args.lane}.json"
    events = []
    for name, task in lanes[args.lane]:
        job = by_key.get((name, task))
        if job is None:
            continue
        forecast_name = job.get("analysis_runner", name)
        destination = OUT / job["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        existing = destination / f"run_{task}.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            completed_seconds[forecast_name] = previous["seconds"]
            completed_workers[forecast_name] = previous["workers"]
            completed_units[forecast_name] = len(previous["methods"]) * len(previous["split_ids"]) * (
                previous["params"]["epochs"] if job["runner"] == "cnn" else 1)
            continue
        remaining = deadline - time.time()
        plan_path = OUT / "worker_plan.json"
        worker_plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
        planned_workers = worker_plan.get(f"dronerf_phase3_{name}/{task}", 1)
        expected = completed_seconds.get(forecast_name, 0) * 1.2 * completed_workers.get(forecast_name, 1) / planned_workers
        planned_units = len(job["methods"]) * job["folds"] * (job["epochs"] if job["runner"] == "cnn" else 1)
        expected *= planned_units / completed_units.get(forecast_name, planned_units)
        if task != "10class" and remaining < max(600, expected):
            events.append({"name": name, "task": task, "status": "skipped_wall_budget",
                           "remaining_seconds": remaining, "expected_seconds": expected})
            events_path.write_text(json.dumps(events, indent=2) + "\n")
            continue
        fold_argument = ",".join(map(str, job["split_ids"])) if "split_ids" in job else str(job["folds"])
        command = [str(PYTHON), "-u", str(RUN), job["runner"], "--dataset", "dronerf",
                   "--task", task, "--methods", *job["methods"], "--folds", fold_argument,
                   "--workers", "1", "--out", str(destination), "--run-name", "dronerf_phase3_" + name]
        if job["runner"] == "trees":
            command += ["--quick", "--classifiers", *job["classifiers"]]
            if name == "trees_lonly":
                command += ["--l-only"]
        elif job["runner"] == "rocket":
            command += ["--features", "1008", "--head-epochs", "5", "--fit-samples", "128"]
        else:
            command += ["--epochs", str(job["epochs"]), "--patience", "3", "--width", "8", "--downsample", "2",
                        "--device", "mps"]
            if job.get("cache_stacks", False):
                command += ["--cache-stacks"]
        start = time.time()
        event = {"name": name, "task": task, "status": "running", "command": command,
                 "start_utc": datetime.now(timezone.utc).isoformat()}
        events.append(event)
        events_path.write_text(json.dumps(events, indent=2) + "\n")
        print(f"start {name}/{task}", flush=True)
        with open(destination / f"console_{task}.log", "w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        event.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode,
                     seconds=time.time() - start, end_utc=datetime.now(timezone.utc).isoformat())
        completed_seconds[forecast_name] = event["seconds"]
        if existing.exists():
            finished = json.loads(existing.read_text())
            completed_workers[forecast_name] = finished["workers"]
            completed_units[forecast_name] = len(finished["methods"]) * len(finished["split_ids"]) * (
                finished["params"]["epochs"] if job["runner"] == "cnn" else 1)
        events_path.write_text(json.dumps(events, indent=2) + "\n")
        if result.returncode:
            print(f"FAILED {name}/{task}, see {destination / f'console_{task}.log'}", flush=True)
            return result.returncode
        print(f"finished {name}/{task} in {event['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
