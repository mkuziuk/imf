"""Validate actual held-out results and independently recompute corrected CIs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_ablations import DATA, METRICS, OUTPUT, STUDY, sha256
from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import label_column, load_info, load_meta


def main():
    checks = []
    for out in sorted(OUTPUT.glob("*/classical_ablations")):
        dataset = out.parent.name
        for path in sorted(out.glob("ablation_folds_*.csv")):
            task = path.stem.removeprefix("ablation_folds_")
            audit = json.loads((out / f"audit_{task}.json").read_text())
            assert audit["finished"]
            assert sha256(STUDY / "baselines/run_classical.py") == audit["reuse"]["driver_sha256"]
            assert sha256(Path(audit["reuse"]["fold_file"])) == audit["reuse"]["fold_sha256"]
            assert sha256(DATA / dataset / f"splits_{task}.json") == audit["reuse"]["split_sha256"]
            meta, info = load_meta(dataset), load_info(dataset)
            y = meta[label_column(info, task)].astype(str).to_numpy()
            labels = sorted(set(y))
            splits = {s.split_id: s for s in load_splits(dataset, task)}
            folds = pd.read_csv(path, float_precision="round_trip")
            assert len(folds) == audit["job_count"]
            assert not folds.duplicated(["method", "feature_set", "classifier", "split_id"]).any()
            checkpoints = list((DATA / dataset / "results/phase2b" / task / "jobs").glob("*.npz"))
            seen = set()
            probability_fits = 0
            for checkpoint in checkpoints:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    row = json.loads(str(saved["row"]))
                    key = row["method"], row["feature_set"], row["classifier"], row["split_id"]
                    assert key not in seen
                    seen.add(key)
                    split = splits[row["split_id"]]
                    assert np.array_equal(saved["frame_id"], split.test)
                    assert np.array_equal(saved["y_true"], y[split.test])
                    pred = saved["y_pred"]
                    metrics = split_metrics(y[split.test], pred, labels)
                    for metric in METRICS:
                        assert np.isclose(metrics[metric], row[metric], atol=1e-12)
                    probability = saved["probabilities"]
                    if probability.shape[1]:
                        assert probability.shape == (len(pred), len(saved["classes"]))
                        assert np.isfinite(probability).all()
                        assert np.allclose(probability.sum(axis=1), 1)
                        assert np.array_equal(saved["classes"][probability.argmax(axis=1)], pred)
                        probability_fits += 1
            assert len(seen) == len(folds)
            all_folds = pd.read_csv(out / f"results_folds_{task}.csv", float_precision="round_trip")
            paired = pd.read_csv(out / f"paired_fixed_classifier_{task}.csv", float_precision="round_trip")
            for _, row in paired.iterrows():
                cell = all_folds[(all_folds.classifier == row.classifier)
                                 & (all_folds.feature_set == row.feature_set)]
                a = cell[cell.method == row.method_a].sort_values("split_id")
                b = cell[cell.method == row.method_b].sort_values("split_id")
                d = a[row.metric].to_numpy() - b[row.metric].to_numpy()
                se = np.sqrt((1 / len(d) + np.mean(a.n_test / a.n_train)) * np.var(d, ddof=1))
                half = stats.t.ppf(.975, len(d) - 1) * se
                assert np.allclose([row.mean_diff, row.se_corrected, row.ci_low, row.ci_high],
                                   [d.mean(), se, d.mean() - half, d.mean() + half])
            checks.append(dict(dataset=dataset, task=task, new_fits=len(seen),
                               probability_fits=probability_fits, corrected_intervals=len(paired), passed=True))
    target = OUTPUT / "uas_birds_60ghz/classical_ablations/validation.json"
    target.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
