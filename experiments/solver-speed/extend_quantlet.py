"""Extend the saved single-threaded QuantLet sweep within a wall-clock budget.

The parent enforces the timeout in a separate process. Completed measurements
and their component checks are saved after each size; partial timings are omitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark_support as benchmark
import numpy as np
from scaling_benchmark import CACHE, DESTINATION, SIZES, make_case, save, summarize


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(result):
    # Preserve the original runner hashes, but verify the solvers and adapter
    # that actually perform these measurements before using cached GD outputs.
    for source in result["metadata"]["sources"]:
        path = Path(source["path"])
        if path.name == "scaling_benchmark.py":
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path


def check_components(result, n):
    with np.load(CACHE / f"{n}_quantlet_1.npz") as data:
        scalar = data["components"]
    with np.load(CACHE / f"{n}_gd_1.npz") as data:
        gd = data["components"]
    runs = [run for run in result["runs"] if run["n"] == n]
    assert {(run["method"], run["workers"]) for run in runs} == {
        ("quantlet", 1), ("gd", 1), ("gd", 4), ("gd", 12)}
    for run in runs:
        with np.load(CACHE / f"{n}_{run['method']}_{run['workers']}.npz") as data:
            components = data["components"]
        assert components.shape == scalar.shape == gd.shape
        own = scalar if run["method"] == "quantlet" else gd
        run["max_component_difference_from_scalar"] = float(np.max(np.abs(components - scalar)))
        run["max_component_difference_from_single_thread"] = float(np.max(np.abs(components - own)))
    passed = all(run["unconverged_chunks"] == 0 and run["reconstruction_max_error"] < 1e-12
                 and run["max_component_difference_from_scalar"] < 1e-4
                 and run["max_component_difference_from_single_thread"] < 1e-4 for run in runs)
    validation = {"n": n, "passed": passed, "scalar_reference_available": True,
                  "max_component_difference": max(run["max_component_difference_from_scalar"] for run in runs),
                  "max_thread_difference": max(run["max_component_difference_from_single_thread"] for run in runs)}
    result["validation"] = [item for item in result["validation"] if item["n"] != n] + [validation]
    result["validation"].sort(key=lambda item: item["n"])
    return passed


def worker(max_n):
    result = json.loads(DESTINATION.read_text())
    verify_sources(result)
    pending = [n for n in SIZES if n <= max_n and
               not any(run["n"] == n and run["method"] == "quantlet" for run in result["runs"])]
    print("QuantLet extension: one untimed warmup at 1,000 points", flush=True)
    benchmark.decompose(make_case(1000), "quantlet", 1)
    for n in pending:
        case = make_case(n)
        assert case["parameters"] == next(item for item in result["cases"] if item["n"] == n)
        for workers in [1, 4, 12]:
            assert (CACHE / f"{n}_gd_{workers}.npz").exists()
        started_utc = utc_now()
        result["current_run"] = {"n": n, "method": "quantlet", "workers": 1, "started_utc": started_utc}
        save(result)
        print(f"Starting QuantLet at {n:,} points, {case['parameters']['stages']} stages", flush=True)
        load = list(os.getloadavg())
        measured = benchmark.decompose(case, "quantlet", 1)
        record = {"n": n, "method": "quantlet", "workers": 1, "seconds": measured["seconds"],
                  "started_utc": started_utc, "finished_utc": utc_now(),
                  "stage_count": len(measured["stages"]), "load_average_start": load,
                  "stages": measured["stages"], "unconverged_chunks": 0,
                  "reconstruction_max_error": float(np.max(np.abs(
                      measured["components"].sum(axis=0) + measured["residual"] - case["y"]))),
                  "max_component_difference_from_scalar": None,
                  "max_component_difference_from_single_thread": None}
        np.savez_compressed(CACHE / f"{n}_quantlet_1.npz",
                            components=measured["components"], residual=measured["residual"])
        result["runs"].append(record)
        passed = check_components(result, n)
        result.pop("current_run", None)
        result["metadata"]["quantlet_max_n"] = n
        save(result)
        summarize(result).to_csv(benchmark.OUTPUT / "scaling_summary.csv", index=False)
        print(f"Finished {n:,}: {record['seconds']:.3f} seconds; output checks {'passed' if passed else 'FAILED'}", flush=True)
        if not passed:
            raise ValueError(f"Output checks failed at {n} points")


def extend(max_n=13667, budget_seconds=2400):
    assert max_n in SIZES and 0 < budget_seconds <= 2400
    result = json.loads(DESTINATION.read_text())
    assert result["metadata"].get("complete") and result["metadata"]["all_output_checks_passed"]
    verify_sources(result)
    pending = [n for n in SIZES if n <= max_n and
               not any(run["n"] == n and run["method"] == "quantlet" for run in result["runs"])]
    if not pending:
        print("All requested QuantLet sizes are already measured.", flush=True)
        return result
    for n in pending:
        for workers in [1, 4, 12]:
            assert (CACHE / f"{n}_gd_{workers}.npz").exists()
    CACHE.mkdir(parents=True, exist_ok=True)
    backup = CACHE / f"before-extension-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    backup.write_bytes(DESTINATION.read_bytes())
    metadata = result["metadata"]
    metadata.setdefault("quantlet_extensions", []).append({
        "started_utc": utc_now(), "requested_max_n": max_n, "pending_sizes": pending,
        "budget_seconds": budget_seconds, "retained_measurements": len(result["runs"]),
        "warmup": "One untimed full QuantLet decomposition at 1000 points before this extension.",
        "order": "Only QuantLet is timed, one thread, sizes ascending; all GD measurements are reused.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_helper_sha256": hashlib.sha256(Path(__file__).with_name("scaling_benchmark.py").read_bytes()).hexdigest(),
        "status": "running"})
    metadata["complete"] = False
    save(result)
    start = time.monotonic()
    status = "completed"
    returncode = 0
    try:
        completed = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--worker", "--max-n", str(max_n)], timeout=budget_seconds)
        returncode = completed.returncode
        if returncode:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "time_limit"
        print("40-minute extension limit reached; partial timing excluded.", flush=True)
    result = json.loads(DESTINATION.read_text())
    extension = result["metadata"]["quantlet_extensions"][-1]
    extension.update(status=status, finished_utc=utc_now(), elapsed_wall_seconds=time.monotonic() - start,
                     completed_sizes=[n for n in pending if any(run["n"] == n and run["method"] == "quantlet"
                                                               for run in result["runs"])])
    if "current_run" in result:
        extension["interrupted_run"] = result.pop("current_run")
        extension["interrupted_timing_included"] = False
    result["metadata"]["quantlet_max_n"] = max(run["n"] for run in result["runs"] if run["method"] == "quantlet")
    result["metadata"]["complete"] = True
    result["metadata"]["all_output_checks_passed"] = all(item["passed"] for item in result["validation"])
    save(result)
    summarize(result).to_csv(benchmark.OUTPUT / "scaling_summary.csv", index=False)
    if returncode:
        raise RuntimeError(f"QuantLet worker exited with status {returncode}; retained completed measurements")
    print(f"Extension {status}: {extension['elapsed_wall_seconds'] / 60:.2f} minutes", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, choices=SIZES, default=13667)
    parser.add_argument("--budget-seconds", type=float, default=2400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.max_n)
    else:
        extend(args.max_n, args.budget_seconds)
