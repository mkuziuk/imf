"""Run the requested shared CLIs sequentially with at most two CPU workers.

Run from the repository root with .venv/bin/python -I and this script's path.
Completed outputs are reused; interrupted decompositions resume via their CLI.
Logs and invocation timings stay under DATA_ROOT/solver_integration/.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
RUN_DIR = DATA_ROOT / "solver_integration"
DATASETS = ("synthetic", "uas_birds_60ghz", "esc50_animals", "birdclef2023", "dronerf")
CONV_TAGS = ("irmf_h2_a1.41_conv", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv", "irmf_h2_a2_conv")
WORKERS = 2


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def run_one(dataset, tag, module, records):
    decomposition = module == "run_decompose"
    kind = "decomp" if decomposition else "features"
    summary_path = DATA_ROOT / dataset / kind / f"{tag}.json"
    existing = read_json(summary_path)
    suffix = ".npy" if decomposition else ".parquet"
    complete = existing and not existing.get("partial") and summary_path.with_suffix(suffix).exists()
    command = [sys.executable, "-m", f"emdirmf.{module}", "--dataset", dataset,
               "--method", tag, "--workers", str(WORKERS)]
    record = {"dataset": dataset, "tag": tag, "kind": kind, "command": command,
              "cwd": str(STUDY_DIR), "workers": WORKERS,
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    log_path = RUN_DIR / "cli" / f"{dataset}.{tag}.{kind}.log"
    started = time.perf_counter()
    if complete:
        record["status"] = "reused"
        print(f"reuse {dataset}/{tag} {kind}", flush=True)
    else:
        print(f"start {dataset}/{tag} {kind}, workers={WORKERS}, log={log_path}", flush=True)
        with log_path.open("a") as handle:
            handle.write(f"\n{record['started']} {' '.join(command)}\n")
            handle.flush()
            subprocess.run(command, cwd=STUDY_DIR, check=True, stdout=handle, stderr=subprocess.STDOUT)
        record["status"] = "completed"
    record["elapsed_seconds"] = time.perf_counter() - started
    records.append(record)
    temporary = RUN_DIR / "invocations.tmp"
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(RUN_DIR / "invocations.json")
    summary = read_json(summary_path)
    if summary is None or summary.get("partial"):
        raise RuntimeError(f"incomplete {summary_path}")
    failed = summary.get("n_failed", 0) if decomposition else summary.get("n_failed_frames", 0)
    if failed or summary.get("n_nonfinite_rows_among_ok", 0):
        raise RuntimeError(f"failures or nonfinite features in {summary_path}")
    print(f"done {dataset}/{tag} {kind}: {record['status']}, {record['elapsed_seconds']:.1f} s", flush=True)


def main():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["IMF_CLS_MAX_WORKERS"] = str(WORKERS)
    (RUN_DIR / "cli").mkdir(parents=True, exist_ok=True)
    records = read_json(RUN_DIR / "invocations.json") or []
    process_snapshot = subprocess.run(["ps", "-Ao", "pid,ppid,pcpu,etime,command"],
                                      check=True, capture_output=True, text=True).stdout
    (RUN_DIR / "processes_at_start.txt").write_text(process_snapshot)
    for tag in CONV_TAGS:
        for dataset in DATASETS:
            run_one(dataset, tag, "run_decompose", records)
            run_one(dataset, tag, "run_features", records)
    for dataset in ("uas_birds_60ghz", "birdclef2023", "dronerf", "esc50_animals"):
        run_one(dataset, "imf_linear_a2", "run_decompose", records)
        run_one(dataset, "imf_linear_a2", "run_features", records)
    print("All requested decompositions and features are complete.", flush=True)


if __name__ == "__main__":
    main()
