#!/usr/bin/env python3
"""Run available ESC-50 LightGBM sensitivities without changing shared code."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
ROOT = STUDY.parents[1]
OUT = ROOT / "output/classification-emd-irmf/esc50_animals/ml"
sys.path.insert(0, str(STUDY))

from emdirmf.paths import dataset_dir
from emdirmf.run_features import run_features


def ready(tag, features=True):
    directory = dataset_dir("esc50_animals")
    summary = directory / "decomp" / f"{tag}.json"
    if not summary.exists():
        return False
    info = json.loads(summary.read_text())
    if info.get("partial") or info.get("n_failed") or info["shape"][0] != 5100:
        return False
    if not features:
        return True
    table = directory / "features" / f"{tag}.parquet"
    metadata = directory / "features" / f"{tag}.json"
    if not table.exists() or not metadata.exists():
        return False
    info = json.loads(metadata.read_text())
    return info["n_frames"] == 5100 and not info["n_failed_frames"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args()
    manifest = {"started": datetime.now(timezone.utc).isoformat(), "features_created": [], "stages": []}
    manifest_path = OUT / "supplemental_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def stage(name, methods, *, slots=None, fusion=False):
        if not methods:
            return
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        for task in ("animal", "bird"):
            command = [sys.executable, "-u", str(HERE / ("run_concat.py" if fusion else "run_phase3.py"))]
            if not fusion:
                command.append("trees")
            command.extend(["--dataset", "esc50_animals", "--task", task,
                            "--methods", *methods, "--folds", "15", "--workers", str(options.workers),
                            "--classifiers", "lightgbm", "--lightgbm-rounds", "150", "--out", str(folder),
                            "--run-name", f"esc50_phase3_{name}"])
            if slots is not None:
                command.extend(["--slots", str(slots)])
            started = time.perf_counter()
            with (folder / f"stdout_{task}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest["stages"].append({"name": name, "task": task, "methods": methods,
                                       "seconds": time.perf_counter() - started, "command": command})
            save()

    h_candidates = ["irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv", "irmf_h2_a2_conv"]
    manifest["h_checkpoint"] = {tag: ready(tag) for tag in h_candidates}
    save()
    stage("sensitivity", [tag for tag in h_candidates if manifest["h_checkpoint"][tag]])
    for tag in ("imf_linear_a2", "irmf_h2_a2"):
        if ready(tag, features=False) and not ready(tag):
            started = time.perf_counter()
            run_features("esc50_animals", tag, workers=1)
            manifest["features_created"].append({"tag": tag, "seconds": time.perf_counter() - started})
            save()
    stage("matched", [tag for tag in ("imf_linear_a2", "irmf_h2_a2") if ready(tag)])
    stage("slots9", ["imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"], slots=9)
    stage("concat", ["irmf_h2_a1.41", "irmf_h2_a1.41_conv"], fusion=True)
    manifest["final_readiness"] = {tag: ready(tag) for tag in h_candidates}
    if ready("irmf_h2_a2_conv") and not manifest["h_checkpoint"]["irmf_h2_a2_conv"]:
        stage("sensitivity_late", ["irmf_h2_a2_conv"])
    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    save()
    print(manifest_path)


if __name__ == "__main__":
    main()
