#!/usr/bin/env python3
"""Run prioritized tree families sequentially, with separate prediction exports."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "output/classification-emd-irmf/birdclef2023/ml"
PYTHON = REPO / ".venv/bin/python"
WRAPPER = Path(__file__).with_name("run_phase3.py")
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="species")
    parser.add_argument("--classifiers", nargs="+", choices=("lightgbm", "extratrees", "xgboost"),
                        default=["lightgbm", "extratrees", "xgboost"])
    options = parser.parse_args()
    task = options.task
    jobs = [(classifier, "1,2,3,4,5,6,7,8,9,10,11,12,13,14"
             if classifier == "lightgbm" and task == "species" else "15")
            for classifier in options.classifiers]
    environment = dict(os.environ, IMF_CLS_DATA="/Users/mikhail/Datasets/imf-classification")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    commands = []
    for classifier, folds in jobs:
        name = f"trees_{classifier}" + ("_remaining" if classifier == "lightgbm" and task == "species" else "")
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", str(WRAPPER), "trees", "--run-name", name,
                   "--dataset", "birdclef2023", "--task", task, "--methods", *METHODS,
                   "--classifiers", classifier, "--folds", folds, "--workers", "2", "--out", str(directory)]
        commands.append(command)
    suffix = "" if options.classifiers == ["lightgbm", "extratrees", "xgboost"] else "_" + "_".join(options.classifiers)
    (OUT / f"tree_commands_{task}{suffix}.json").write_text(json.dumps(commands, indent=2) + "\n")
    for command in commands:
        name = command[command.index("--run-name") + 1]
        print("Starting", name, task, flush=True)
        with (OUT / f"{name}_{task}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print("Finished", name, task, flush=True)


if __name__ == "__main__":
    main()
