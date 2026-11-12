"""Serial queue for the first three datasets; four fit workers in total."""

import argparse
import subprocess
import sys
from pathlib import Path

from run_ablations import DATA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--include-secondary", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_ablations.py")
    queue = [("uas_birds_60ghz", "bird_drone"), ("esc50_animals", "animal"),
             ("birdclef2023", "species")]
    if args.include_secondary:
        queue += [("uas_birds_60ghz", "target3"), ("esc50_animals", "bird"),
                  ("birdclef2023", "passerine")]
    for dataset, task in queue:
        folder = DATA / dataset / "results/phase2b"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"RUN {dataset}/{task}", flush=True)
        with (folder / f"{task}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(runner), "--dataset", dataset,
                            "--task", task, "--allow-secondary"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE {dataset}/{task}", flush=True)
