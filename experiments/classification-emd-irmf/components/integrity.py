"""Check the installed pimf against the fingerprint captured before any run."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pimf

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
DATASETS = ("synthetic", "dronerf", "birdclef2023", "esc50_animals")


def fingerprint():
    directory = Path(pimf.__file__).resolve().parent
    return {"version": pimf.__version__, "files": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.py"))
    }}


def check_pimf():
    current = fingerprint()
    expected = json.loads((HERE / "pimf_baseline.json").read_text())
    if current != expected or current["version"] != "0.4.0":
        raise RuntimeError("STOP: installed pimf version or Python-file hashes changed")
    return current


def record_entry(dataset, runner):
    current = check_pimf()
    directory = DATA_ROOT / dataset / "components" / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    record = {"time_utc": now, "runner": runner, "pimf": current}
    with (directory / f"{now.replace(':', '-')}_{os.getpid()}.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    print(f"pimf audit {runner}: {json.dumps(current, sort_keys=True)}", flush=True)
    return record
