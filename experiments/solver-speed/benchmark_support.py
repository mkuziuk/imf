"""Matched single-signal IMF solver benchmarks using the original solver bodies.

Original repositories remain unchanged. GD threads process disjoint target windows.
QuantLet uses only its original single-threaded scalar solver.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

THREAD_ENV = {
    name: "1" for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
}
for name, value in THREAD_ENV.items():
    os.environ[name] = value
os.environ.setdefault("MPLCONFIGDIR", "/tmp/imf-benchmark-mpl")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[2]
QUANTLET = ROOT.parent / "IRMF-Intrinsic-Robust-Multiscale-Filtering"
OUTPUT = Path(__file__).resolve().parent / "results"
LOCAL_SOURCE = ROOT / "experiments/gd-irmf/gd_irmf.ipynb"
UPSTREAM_SOURCE = QUANTLET / "Simulations/imd_python_fixedvf.py"


def load_notebook_functions(filename):
    nodes = []
    for cell in json.loads(filename.read_text())["cells"]:
        if cell["cell_type"] == "code":
            nodes.extend(node for node in ast.parse("".join(cell["source"])).body
                         if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)))
    ns = {"SQRT_2": np.sqrt(2.0), "SQRT_2_OVER_PI": np.sqrt(2.0 / np.pi)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), ns)
    ns["lookup_grid"] = ns["make_lookup_grid"]()
    return ns


def load_upstream():
    spec = importlib.util.spec_from_file_location("quantlet_solver", UPSTREAM_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LOCAL = load_notebook_functions(LOCAL_SOURCE)
Q = load_upstream()
UPSTREAM_KERNEL = Q.epanechnikov


def kernel(offsets, kind):
    offsets = np.asarray(offsets, dtype=float)
    if kind == "squared_triangular":
        return 0.75 * np.maximum(1 - np.abs(offsets), 0) ** 2
    if kind == "epanechnikov":
        return UPSTREAM_KERNEL(offsets)
    raise ValueError(kind)


def make_case(profile, n=None):
    if profile == "your_gd":
        n = n or 1000
        t = np.arange(n) / n
        # Same signal and noise generator as the GD notebook; only time labels
        # change to endpoint-exclusive so both periodic implementations agree.
        clean, _ = LOCAL["gen_signal"](t)
        y, _ = LOCAL["generate_observation"](clean, sigma=0.2, contamination_prob=0.2,
                                               contamination_scale=0.2, rng=np.random.default_rng(777))
        sizes = LOCAL["make_window_schedule"](n, min_window_size=31)
        radii = np.array(sizes, dtype=int) // 2
        params = dict(profile=profile, label="My GD settings", n=n, H=0.4, sigma=0.2,
                      kernel="squared_triangular", noise="Gaussian + signed exponential",
                      contamination_probability=0.2, contamination_scale=0.2, seed=777)
        params["bandwidths"] = (radii / n).tolist()
    elif profile in ["quantlet_zero", "quantlet_mixture"]:
        n = n or 2000
        t = np.arange(n) / n
        noise_type = 1 if profile == "quantlet_zero" else 2
        dat = Q.generate_zero_contaminated_signal(x_grid=t, n_paths=1, case=1, type=noise_type,
                                                 prop=0.2, sd_o=4, sd_c=0.1, seed=2026)
        clean, y = dat["true_signal"], dat["Y"]
        params = dict(profile=profile, label="QuantLet default settings" if noise_type == 1 else "QuantLet Gaussian mixture",
                      n=n, H=1.0, sigma=0.0 if noise_type == 1 else 0.1, kernel="epanechnikov",
                      noise="20% replaced by zero" if noise_type == 1 else "Gaussian mixture, SD 0.1 / 0.4",
                      contamination_probability=0.2, seed=2026)
        params["bandwidths"] = (0.2 / np.sqrt(2) ** np.arange(8)).tolist()
    else:
        raise ValueError(profile)
    params["stages"] = len(params["bandwidths"])
    params["grid"] = "arange(n)/n, endpoint excluded for both methods"
    params["input_sha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    return {"parameters": params, "t": t, "clean": clean, "y": y}


def ranges_for(n, workers):
    size = n if workers == 1 else max(64, int(math.ceil(n / workers)))
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def gd_stage(y, bandwidth, H, kind, workers):
    radius_float = bandwidth * len(y)
    rounded = round(radius_float)
    radius = int(rounded if abs(radius_float - rounded) < 1e-9 else math.ceil(radius_float))
    weights = kernel(np.arange(-radius, radius + 1) / radius_float, kind)
    windows = sliding_window_view(np.pad(y, radius, mode="wrap"), 2 * radius + 1)
    ranges = ranges_for(len(y), workers)

    def chunk(bounds):
        start, stop = bounds
        values, trace = LOCAL["robust_gd_fit_windows"](
            windows[start:stop], weights, H, grid=LOCAL["lookup_grid"],
            max_iter=60, tol=1e-6, return_trace=True,
        )
        threshold = 1e-6 * (1 + float(np.max(np.abs(values))))
        return start, values, len(trace), float(trace[-1]), bool(trace[-1] <= threshold)

    if workers == 1 or len(ranges) == 1:
        chunks = [chunk(bounds) for bounds in ranges]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(chunk, ranges))
    out = np.empty_like(y)
    for start, values, _, _, _ in chunks:
        out[start:start + len(values)] = values
    return out, {"iterations": max(c[2] for c in chunks),
                 "unconverged_chunks": sum(not c[4] for c in chunks),
                 "chunks": len(chunks)}


def quantlet_stage(y, t, bandwidth, H):
    # Exact original scalar objective and search over all target locations.
    ext_t, ext_y = Q.periodic_extend(y, t)
    lower, upper = np.min(y) - 4 * H, np.max(y) + 4 * H
    values = Q.weighted_minimizer(
        t, ext_t, lambda a, active: Q.rho_H(ext_y[active] - a, H),
        bandwidth, lower, upper,
    )
    return values, {"chunks": 1}


def decompose(case, method, workers):
    if method == "quantlet" and workers != 1:
        raise ValueError("QuantLet supports only one worker; threading is available only for GD.")
    params = case["parameters"]
    # Suites run sequentially, so this in-memory kernel selection has no races.
    Q.epanechnikov = lambda v: kernel(v, params["kernel"])
    start = time.perf_counter()
    residual = case["y"].copy()
    components, stages = [], []
    for bandwidth in params["bandwidths"]:
        stage_start = time.perf_counter()
        if method == "gd":
            component, info = gd_stage(residual, bandwidth, params["H"], params["kernel"], workers)
        elif method == "quantlet":
            component, info = quantlet_stage(residual, case["t"], bandwidth, params["H"])
        else:
            raise ValueError(method)
        components.append(component)
        residual -= component
        stages.append({"seconds": time.perf_counter() - stage_start, **info})
    components = np.stack(components)
    elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "components": components, "residual": residual, "stages": stages}


def metadata():
    def command(args):
        return subprocess.check_output(args, text=True).strip()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpus": os.cpu_count(), "platform": platform.platform(),
        "memory_bytes": int(command(["sysctl", "-n", "hw.memsize"])),
        "python": sys.version, "numpy": np.__version__,
        "gil_enabled": sys._is_gil_enabled(), "library_thread_limits": THREAD_ENV,
        "numpy_blas": np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}),
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in [LOCAL_SOURCE, UPSTREAM_SOURCE, Path(__file__)]],
        "commits": {str(p): command(["git", "-C", str(p), "rev-parse", "HEAD"])
                    for p in [ROOT, QUANTLET]},
        "timing_scope": "One complete single-signal decomposition. Includes weights, padding, iteration, residual updates and GD per-stage thread-pool creation. Excludes imports, lookup-table creation, input generation, validation, population fitting, diagnostics and plots.",
        "threading": "GD uses ThreadPoolExecutor over disjoint contiguous target chunks, minimum 64 targets/chunk. Stages remain sequential. QuantLet uses one worker only.",
        "tolerances": {"gd_relative_update": 1e-6, "gd_max_iterations": 60,
                       "golden_absolute_bracket": 1e-6, "golden_max_iterations": 200,
                       "acceptable_component_difference": 1e-4},
        "warmup": "One untimed complete decomposition per method per case at one worker.",
        "repetitions": "Same input per case; deterministic shuffled execution order within each timing repetition; configurations never run concurrently.",
    }


def run_suite(workers=(1, 2, 4, 8, 12, 16), repeats=3, profiles=("your_gd", "quantlet_zero"),
              sizes=None, name="benchmark"):
    """Benchmark the requested GD worker counts against single-threaded QuantLet."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workers_by_method = {"gd": list(workers), "quantlet": [1]}
    result = {"metadata": metadata(), "workers_by_method": workers_by_method, "repeats": repeats,
              "cases": [], "runs": [], "validation": []}
    cases = [make_case(profile, n) for profile in profiles for n in (sizes or [None])]
    references = {}

    def save():
        result["metadata"]["latest_utc"] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / (name + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for case in cases:
        params = case["parameters"]
        key = params["profile"] + "_" + str(params["n"])
        params["key"] = key
        result["cases"].append(params)
        print("Preparing", key, flush=True)
        for method in ["gd", "quantlet"]:
            reference = decompose(case, method, 1)
            references[(key, method)] = reference
            check = {
                "case": key, "method": method,
                "reconstruction_max_error": float(np.max(np.abs(reference["components"].sum(axis=0) + reference["residual"] - case["y"]))),
                "unconverged_chunks": sum(s.get("unconverged_chunks", 0) for s in reference["stages"]),
            }
            assert check["reconstruction_max_error"] < 1e-12, check
            assert check["unconverged_chunks"] == 0, check
            result["validation"].append(check)
        difference = float(np.max(np.abs(references[(key, "gd")]["components"] - references[(key, "quantlet")]["components"])))
        assert difference < 1e-4, (key, difference)
        result["validation"].append({"case": key, "cross_solver_max_component_difference": difference})
        # Check the adapter against the native GD decomposition for its own setup.
        if params["profile"] == "your_gd":
            native = LOCAL["robust_gd_imf_with_history"](
                case["y"], LOCAL["make_window_schedule"](params["n"], min_window_size=31), params["H"], max_workers=1)
            adapter_error = float(np.max(np.abs(native["imfs"] - references[(key, "gd")]["components"])))
            assert adapter_error < 1e-10, adapter_error
            result["validation"].append({"case": key, "native_gd_adapter_max_difference": adapter_error})
        np.savez_compressed(OUTPUT / (key + "_example.npz"), t=case["t"], clean=case["clean"], y=case["y"],
                            gd_components=references[(key, "gd")]["components"],
                            quantlet_components=references[(key, "quantlet")]["components"])
        save()

    tasks = [(case, method, worker) for case in cases
             for method, counts in workers_by_method.items() for worker in counts]
    order_rng = np.random.default_rng(17092026)
    count, total = 0, len(tasks) * repeats
    for repeat in range(repeats):
        for index in order_rng.permutation(len(tasks)):
            case, method, worker = tasks[int(index)]
            key = case["parameters"]["key"]
            load_start = os.getloadavg()
            measured = decompose(case, method, worker)
            reference = references[(key, "quantlet")]
            own = references[(key, method)]
            difference = float(np.max(np.abs(measured["components"] - reference["components"])))
            thread_difference = float(np.max(np.abs(measured["components"] - own["components"])))
            unconverged = sum(s.get("unconverged_chunks", 0) for s in measured["stages"])
            assert difference < 1e-4 and thread_difference < 1e-4 and unconverged == 0, (key, method, worker, difference, unconverged)
            result["runs"].append({"case": key, "method": method, "workers": worker,
                                   "repeat": repeat + 1, "seconds": measured["seconds"],
                                   "max_component_difference_from_scalar": difference,
                                   "max_component_difference_from_single_thread": thread_difference,
                                   "unconverged_chunks": unconverged,
                                   "load_average_start": list(load_start), "stages": measured["stages"]})
            count += 1
            print(f"{count}/{total}: {key}, {method}, {worker} workers: {measured['seconds']:.3f} s", flush=True)
            save()
    result["metadata"]["complete"] = True
    save()
    summary = summarize(result)
    summary.to_csv(OUTPUT / (name + "_summary.csv"), index=False)
    return result


def summarize(result):
    frame = pd.DataFrame(result["runs"])
    frame = frame.groupby(["case", "method", "workers"], as_index=False).agg(
        seconds=("seconds", "median"), minimum=("seconds", "min"), maximum=("seconds", "max"),
        repeats=("seconds", "size"), max_difference=("max_component_difference_from_scalar", "max"),
        thread_difference=("max_component_difference_from_single_thread", "max"),
    )
    baseline = frame[frame.workers == 1].set_index(["case", "method"])["seconds"].to_dict()
    frame["speedup"] = [baseline[(row.case, row.method)] / row.seconds for row in frame.itertuples()]
    return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    result = run_suite(workers=(1, 4), repeats=1, name="pilot") if args.pilot else run_suite()
    print(summarize(result).to_string(index=False))
