"""Repair this runner's early object-dtype label arrays without unpickling.

The saved frame IDs and current fingerprint-verified prepared metadata supply
the labels. Stored metrics must reproduce before a checkpoint is replaced.
Predictions, probabilities, fit timing and estimator signatures are preserved.
"""

import json

import numpy as np

from run_ablations import DATA, METRICS, OUTPUT
from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import label_column, load_info, load_meta


def main():
    checks = []
    for dataset in ("uas_birds_60ghz", "esc50_animals", "birdclef2023", "dronerf"):
        data_dir = DATA / dataset
        if not (data_dir / "results/phase2b").exists():
            continue
        info, meta = load_info(dataset), load_meta(dataset)
        for task_dir in sorted((data_dir / "results/phase2b").iterdir()):
            if not (task_dir / "jobs").is_dir():
                continue
            task = task_dir.name
            y = meta[label_column(info, task)].astype(str).to_numpy().astype(str)
            labels = sorted(set(y))
            splits = {s.split_id: s for s in load_splits(dataset, task)}
            repaired, total = 0, 0
            for path in sorted((task_dir / "jobs").glob("*.npz")):
                with np.load(path, allow_pickle=False) as saved:
                    row = json.loads(str(saved["row"]))
                    assert (row["dataset"], row["task"], row["source"]) == (dataset, task, "phase2b")
                    frame_id = saved["frame_id"]
                    assert np.array_equal(frame_id, splits[row["split_id"]].test)
                    truth, pred = y[frame_id], saved["y_pred"]
                    metrics = split_metrics(truth, pred, labels)
                    for metric in METRICS:
                        assert np.isclose(row[metric], metrics[metric], atol=1e-12, rtol=0)
                    try:
                        assert np.array_equal(saved["y_true"], truth)
                        needs_repair = False
                    except ValueError as error:
                        if "Object arrays cannot be loaded" not in str(error):
                            raise
                        needs_repair = True
                    if needs_repair:
                        arrays = {key: saved[key] for key in saved.files if key != "y_true"}
                        arrays["y_true"] = truth
                if needs_repair:
                    temporary = path.with_suffix(".labels.partial")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **arrays)
                    temporary.replace(path)
                    repaired += 1
                total += 1
            checks.append(dict(dataset=dataset, task=task, checkpoints=total,
                               repaired_object_label_arrays=repaired,
                               all_stored_metrics_reproduced=True))
    target = OUTPUT / "uas_birds_60ghz/classical_ablations/checkpoint_label_repair.json"
    target.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
