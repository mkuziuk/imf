"""Full-decomposition timings as signal length grows, with fixed GD defaults.

QuantLet uses one worker through 5222 points. GD uses 1, 4 and 12 workers at
all sizes. Completed measurements are checkpointed and reused on resume.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import benchmark_support as benchmark
import numpy as np
import pandas as pd

SIZES = np.rint(np.linspace(1000, 20000, 10)).astype(int).tolist()
CONFIGURATIONS = [("quantlet", 1), ("gd", 1), ("gd", 4), ("gd", 12)]
QUANTLET_MAX_N = 5222
DESTINATION = benchmark.OUTPUT / "scaling.json"
CACHE = benchmark.ROOT / "tmp/solver-speed-scaling"


def make_case(n):
    case = benchmark.make_case("your_gd", n=int(n))
    case["parameters"]["H"] = 2 * case["parameters"]["sigma"]
    case["parameters"]["H_rule"] = "H = 2 * sigma"
    case["parameters"]["key"] = "my_defaults_" + str(n)
    case["parameters"]["window_sizes"] = benchmark.LOCAL["make_window_schedule"](int(n))
    return case


def save(result):
    result["metadata"]["latest_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = DESTINATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(DESTINATION)


def summarize(result):
    return pd.DataFrame([
        {key: run[key] for key in ["n", "method", "workers", "seconds", "stage_count",
                                   "unconverged_chunks", "max_component_difference_from_scalar",
                                   "max_component_difference_from_single_thread"]}
        for run in result["runs"]
    ]).sort_values(["n", "method", "workers"]).reset_index(drop=True)


def run_scaling():
    benchmark.OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        result = json.loads(DESTINATION.read_text())
        assert result["sizes"] == SIZES
        assert result["configurations"] == [list(item) for item in CONFIGURATIONS]
        for source in result["metadata"]["sources"]:
            expected = source["sha256"]
            if source["path"] == str(Path(__file__).resolve()):
                expected = result["metadata"].get("continuation", {}).get("runner_sha256", expected)
            assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == expected, source["path"]
        if result["metadata"].get("complete"):
            print("Using the completed scaling measurements.", flush=True)
            summarize(result).to_csv(benchmark.OUTPUT / "scaling_summary.csv", index=False)
            return result
        print("Resuming", len(result["runs"]), "completed measurements.", flush=True)
    else:
        metadata = benchmark.metadata()
        metadata["sources"].append({"path": str(Path(__file__).resolve()),
                                     "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        metadata["experiment"] = "Full decomposition versus signal length using GD notebook defaults for both solvers."
        metadata["repetitions"] = "One measured run per size and configuration; no concurrent configurations."
        metadata["warmup"] = "One untimed full decomposition at 1000 points for each of the four configurations per process launch."
        metadata["size_rule"] = "Ten evenly spaced sizes from 1000 to 20000, rounded to integers."
        metadata["window_rule"] = "Original make_window_schedule(n): first window about n/2, divide by sqrt(2), stop at 31 points. Stage count grows with n."
        metadata["input_rule"] = "Same signal function, Gaussian sigma=0.2, signed exponential contamination scale=0.2 and probability=0.2, seed=777, H=2*sigma=0.4 and squared-triangular kernel at every size. Identical observations across all four configurations within each size."
        metadata["order"] = "Sizes ascend; configuration order is deterministically shuffled separately for each size with seed 17092026."
        metadata["quantlet_max_n"] = QUANTLET_MAX_N
        result = {"metadata": metadata, "sizes": SIZES,
                  "configurations": [list(item) for item in CONFIGURATIONS],
                  "repeats": 1, "cases": [], "runs": [], "validation": []}
        save(result)

    warmup = make_case(SIZES[0])
    for method, workers in CONFIGURATIONS:
        pending = any((method != "quantlet" or n <= QUANTLET_MAX_N) and
                      not any(run["n"] == n and run["method"] == method and run["workers"] == workers
                              for run in result["runs"]) for n in SIZES)
        if not pending:
            continue
        print("Warmup:", method, workers, flush=True)
        measured = benchmark.decompose(warmup, method, workers)
        assert all(stage.get("unconverged_chunks", 0) == 0 for stage in measured["stages"])

    order_rng = np.random.default_rng(17092026)
    total = sum(method != "quantlet" or n <= QUANTLET_MAX_N
                for n in SIZES for method, _ in CONFIGURATIONS)
    for n in SIZES:
        case = make_case(n)
        if not any(item["n"] == n for item in result["cases"]):
            result["cases"].append(case["parameters"])
        for index in order_rng.permutation(len(CONFIGURATIONS)):
            method, workers = CONFIGURATIONS[int(index)]
            if method == "quantlet" and n > QUANTLET_MAX_N:
                continue
            if any(run["n"] == n and run["method"] == method and run["workers"] == workers for run in result["runs"]):
                continue
            result["current_run"] = {"n": n, "method": method, "workers": workers,
                                      "started_utc": datetime.now(timezone.utc).isoformat()}
            save(result)
            print(f"Starting {len(result['runs'])+1}/{total}: n={n}, {method}, {workers} worker(s), {case['parameters']['stages']} stages", flush=True)
            load = list(os.getloadavg())
            measured = benchmark.decompose(case, method, workers)
            np.savez_compressed(CACHE / f"{n}_{method}_{workers}.npz",
                                components=measured["components"], residual=measured["residual"])
            record = {"n": n, "method": method, "workers": workers,
                      "seconds": measured["seconds"], "stage_count": len(measured["stages"]),
                      "load_average_start": load, "stages": measured["stages"],
                      "unconverged_chunks": sum(stage.get("unconverged_chunks", 0) for stage in measured["stages"]),
                      "reconstruction_max_error": float(np.max(np.abs(measured["components"].sum(axis=0) + measured["residual"] - case["y"]))),
                      "max_component_difference_from_scalar": None,
                      "max_component_difference_from_single_thread": None}
            result["runs"].append(record)
            result.pop("current_run", None)
            save(result)
            print(f"Finished: {record['seconds']:.3f} s; {len(result['runs'])}/{total} measurements", flush=True)

        if not any(item["n"] == n for item in result["validation"]):
            scalar_available = any(run["n"] == n and run["method"] == "quantlet" for run in result["runs"])
            scalar_components = None
            if scalar_available:
                with np.load(CACHE / f"{n}_quantlet_1.npz") as reference:
                    scalar_components = reference["components"]
            with np.load(CACHE / f"{n}_gd_1.npz") as reference:
                gd_components = reference["components"]
            runs = [run for run in result["runs"] if run["n"] == n]
            for record in runs:
                with np.load(CACHE / f"{n}_{record['method']}_{record['workers']}.npz") as data:
                    components = data["components"]
                own = scalar_components if record["method"] == "quantlet" else gd_components
                record["max_component_difference_from_scalar"] = (float(np.max(np.abs(components - scalar_components)))
                                                                   if scalar_available else None)
                record["max_component_difference_from_single_thread"] = float(np.max(np.abs(components - own)))
            passed = all(record["unconverged_chunks"] == 0 and record["reconstruction_max_error"] < 1e-12
                         and (record["max_component_difference_from_scalar"] is None or record["max_component_difference_from_scalar"] < 1e-4)
                         and record["max_component_difference_from_single_thread"] < 1e-4 for record in runs)
            result["validation"].append({"n": n, "passed": passed,
                                         "scalar_reference_available": scalar_available,
                                         "max_component_difference": (max(record["max_component_difference_from_scalar"] for record in runs)
                                                                       if scalar_available else None),
                                         "max_thread_difference": max(record["max_component_difference_from_single_thread"] for record in runs)})
            save(result)
            print("Output checks:", "passed" if passed else "FAILED", "at", n, "points", flush=True)

    result["metadata"]["complete"] = True
    result["metadata"]["all_output_checks_passed"] = all(item["passed"] for item in result["validation"])
    result.pop("current_run", None)
    save(result)
    summarize(result).to_csv(benchmark.OUTPUT / "scaling_summary.csv", index=False)
    return result


if __name__ == "__main__":
    run_scaling()
