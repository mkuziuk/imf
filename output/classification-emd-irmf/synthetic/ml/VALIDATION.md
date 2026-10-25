# Phase-3 validation

Created `ml/run_trees.py`, `run_rocket.py`, `run_cnn.py`, `common.py`, `estimate_costs.py`, `README.md`, `__init__.py`, `emdirmf/stacks.py`, and `tests/test_ml.py`; updated study requirements. Results, importance, curves and cost tables are under this folder. Predictions are in `DATA_ROOT/synthetic/results/`. No commits, real-data fits or cache changes.

Run from the repository root:

```bash
PY=.venv/bin/python
ML=experiments/classification-emd-irmf/ml
$PY $ML/run_trees.py --dataset synthetic --quick --workers 2
$PY $ML/run_rocket.py --dataset synthetic --quick --workers 2 --classifiers minirocket multirocket
$PY $ML/run_cnn.py --dataset synthetic --quick --workers 2 --device mps
$PY $ML/estimate_costs.py --calibrate-head
```

All accept task/method/fold selection. Trees use grouped inner validation for boosting rounds or leaf size. ROCKET uses train-fitted biases, incremental standardization and a tuned streaming logistic head. CNN uses three residual blocks, widths 16/32/64, GroupNorm and mean/max pooling. Inner validation selects epochs; a fresh model refits all outer training data. Raw uses the shared robust normalization. Read-only stacks stream in true scale order; optional anti-aliased downsampling treats every method equally.

Synthetic balanced accuracy, mean over shared split IDs 0–4. Quick budgets: 60 boosting rounds, 64 ExtraTrees trees, 672 ROCKET features/5 head epochs, CNN at most 6 selection/refit epochs. These are smoke-test results.

comp:

|            |   LGBM |   XGB |   ExtraTrees |   MiniRocket |   MultiRocket |   CNN |
|:-----------|-------:|------:|-------------:|-------------:|--------------:|------:|
| Raw        |  0.337 | 0.330 |        0.356 |        0.500 |         0.435 | 0.572 |
| EMD        |  0.407 | 0.393 |        0.354 |        0.416 |         0.350 | 0.225 |
| Linear IMF |  0.701 | 0.697 |        0.629 |        0.543 |         0.462 | 0.558 |
| IRMF       |  0.851 | 0.836 |        0.769 |        0.580 |         0.552 | 0.501 |

comp+log_scale:

|            |   LGBM |   XGB |   ExtraTrees |   MiniRocket |   MultiRocket |   CNN |
|:-----------|-------:|------:|-------------:|-------------:|--------------:|------:|
| Raw        |  0.290 | 0.283 |        0.320 |        0.518 |         0.425 | 0.576 |
| EMD        |  0.386 | 0.359 |        0.353 |        0.408 |         0.354 | 0.220 |
| Linear IMF |  0.687 | 0.665 |        0.635 |        0.541 |         0.433 | 0.565 |
| IRMF       |  0.846 | 0.824 |        0.763 |        0.577 |         0.543 | 0.509 |

Median fit+predict seconds per method/feature-set/fold, in table-column order: 2.29, 5.02, 0.50, 5.37, 2.91, 5.68. Runner wall times: trees 280 s, ROCKET 155 s, CNN 221 s.

59 tests passed. Saved-output audit verified 240 fold rows, 72,000 predictions, paired fold membership and recomputed metrics. All per-class recalls, timings, corrected t-tests, Wilcoxon and Holm tables are saved.

Real-data planning estimates, minutes per primary task, 16 cores at 50% scaling efficiency plus MPS; four inputs and both log-scale variants. No decomposition/feature generation included. Allow roughly 0.5–2×; early stopping may reduce full budgets further.

Five-fold quick pilot:

| Dataset       |   All trees |   MiniRocket |   MultiRocket |   CNN |
|:--------------|------------:|-------------:|--------------:|------:|
| dronerf       |        39.6 |         11.5 |           8.3 |  59.1 |
| birdclef2023  |        31.2 |         11.4 |           9.3 |  53.5 |
| esc50_animals |         8.4 |          3.9 |           3.0 |  22.8 |

Full CPU budgets over 15 folds; CNN five folds, 30+30 epoch ceiling:

| Dataset       |   All trees |   MiniRocket |   MultiRocket |   CNN |
|:--------------|------------:|-------------:|--------------:|------:|
| dronerf       |      1108.1 |        533.3 |         387.6 | 290.3 |
| birdclef2023  |       880.7 |        538.1 |         444.2 | 262.7 |
| esc50_animals |       235.5 |        182.7 |         140.2 | 111.6 |

Shared-package caveat: the default IRMF synthetic cache has unconverged stages in 984/1500 frames, despite exact reconstruction. Solver defaults were preserved. The protocol’s `workers` argument belongs to the pimf constructor, as its README clarifies. Optional 2-D spectrogram reference was not implemented.
