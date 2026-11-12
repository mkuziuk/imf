# Classification robustness v3

Binding design: [PROTOCOL_v3.md](PROTOCOL_v3.md).

## Status on 2026-10-22

The amendment resolves the raw precision and H decisions. All 23 combined v3 and
Laplace tests pass. The new clean gate passes exactly on all 400 comparisons,
50 EMD and 50 raw frames for each dataset. Synthetic end-to-end and analysis preview passed: 540 fits, 2,460 score rows,
688,800 saved predictions rescored, zero failures and zero cap retries. The
synthetic run took 88.18 s with 14 workers. Preview and figures are in
`/tmp/robustness_v3_preview/`.

The real DroneRF, BirdCLEF and ESC-50 run launched detached at
2026-10-22T10:46:14Z, PID **80987**, with one pool of 14 workers. It is healthy:
DroneRF clean chunks are completing, all six thread environment variables are 1
in every worker, and all recorded module hashes still match. Code modules are
frozen for this run. Log:
`/Users/mikhail/Datasets/imf-classification/robustness_v3_runs/20261022T104614188721Z.log`.

Timing covered all 100 excluded pilot IDs per dataset, every K and condition,
with no failures or retries. The projection for the three real datasets is about
9.61 hours on 14 workers, including a prior full-grid fit-cost estimate dominated
by DroneRF. It is a projection, not a measured real-run duration. The conservative
uncompressed v3 storage projection is 6.62 GB including synthetic, below 30 GB.
Timing and launch/health evidence are saved in `provenance/*round2*`.

Synthetic clean-selected results, percentages, shift clean to 40%:

| Classifier | Method | K* | Clean BA | 40% BA |
|---|---|---:|---:|---:|
| logreg | Raw | 0 | 48.09 | 32.56 |
| logreg | EMD | 4 | 54.84 | 30.75 |
| logreg | Linear IMF | 6 | 68.89 | 41.90 |
| logreg | IRMF | 9 | 81.06 | 51.14 |
| LightGBM | Raw | 0 | 34.45 | 30.55 |
| LightGBM | EMD | 3 | 42.10 | 23.25 |
| LightGBM | Linear IMF | 9 | 66.09 | 34.55 |
| LightGBM | IRMF | 9 | 77.81 | 37.00 |

Raw K=0 means one raw slot, with no decomposition. The synthetic 40% primary
DiD is -2.9226 BA points, corrected 95% CI [-9.1248, 3.2796], Holm p=1 across its
separate 18-test family. The test does not reject; this is not equivalence.
Matched logreg BA at 40% is raw 33.97%, EMD 32.11%, linear 47.20%, IRMF 59.75%.

## Dated deviations and resolutions

2026-10-22: the first draft cast normalized raw samples to float32, which failed
7/200 raw cache comparisons: five synthetic and two DroneRF. All 200 EMD K=8
comparisons were exact. The original per-dataset `clean_verification.json` files
remain unchanged. No tolerance was relaxed and no contamination or classifier
fit ran before that gate was resolved. The dated amendment restores the shared
float64 raw samples; decomposition stacks remain float32. The new evidence is
`clean_verification_v2.json`, referenced by `manifest_v2.json`, manifest version 2.
All 400 amended comparisons are exactly equal after float32 feature rounding.

2026-10-22: at the user's choice, H is fixed per dataset in normalized units.
Calibration is twice the median E1 estimate over every channel of that dataset's
100 excluded clean pilot frames, each normalized with `normalize_frame`. H does
not change by input, condition, channel or K. The pilot report's fixed-H control
used split-0 training channels; this experiment uses the excluded pilots, as the
amendment requires. Each input's own E1, source code, E1/H_d and b/H_d remain in
the numeric diagnostics for every variant. `h_ratio` denotes E1/H_d.

| Dataset | Pilot channels | H_d |
|---|---:|---:|
| synthetic | 100 | 1.1853056416210017 |
| DroneRF | 200 | 2.0941664007285286 |
| BirdCLEF 2023 | 100 | 1.0833148940020392 |
| ESC-50 animals | 100 | 0.47263983379285046 |

2026-10-22: constant-channel contamination is interpreted as zero when
`noise_sigma_hat` returns source code 2. That estimator returns an arbitrary
positive fallback of 1 for constants, but the protocol expressly says constant
channels receive no noise. The H estimator itself keeps the shared fallbacks.

## Pilot provenance

The c=3 IQR rule retains E1: E2 had lower per-slot L2 in 8/12 cells and no worse
scale stability in 4/12, below the required 9 and 7. The original pilot copy and
`h_decision.json` remain as first-round evidence. The amended pilot report is
[provenance/sigma_pilot_REPORT_v2.md](provenance/sigma_pilot_REPORT_v2.md), with
SHA-256 `9bb6f2c66f6da7a54090be2bf0c5cbe2aedf70f6adb12574a01eb86b07eb4e8b`.
The amended decision is [provenance/h_decision_v2.json](provenance/h_decision_v2.json).
Each feature manifest records both the report hash and its full fixed-H calibration.

## Implementation

`construction.py` checks all protocol windows and the radius-15 endpoint.
`processing.py` computes fresh clean and contaminated features, runs PyEMD once
per channel, folds EMD slots, and retains full stacks only for pilot IDs.
`classification.py` uses the shared full classical grid and LightGBM runner;
no C choices or models are reused. K* is selected separately for each method and
classifier over all 15 clean scores. Matched classical jobs become eligible only
after all K and split jobs for that method have completed. `k_star.json` records
the dataset selections; individual selections hash their dependencies.

`run.py` uses one process pool across datasets and a dependency-aware longest-first
queue, immutable checkpoints, an active-run lock and detached logs.
`analyse.py` verifies all source and input hashes, matrix content, K* and its
dependencies, every prediction metric and its clean baseline, and retained pilot
IDs before producing a report. Families are Primary 12, S1 12, S3 24, S5 6 and
synthetic 18. A subset preview pads absent real-family tests with p=1.
`analysis.py` and `reporting.py` produce all-K BA, best K per condition,
clean-selected best-vs-best BA, paired tests, solver and H diagnostics, and figures.

## Commands

Run from `experiments/classification-emd-irmf`:

```bash
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=../../.venv/bin/python
$PY -m pytest -q robustness_v3/tests robustness_laplace/tests
$PY -m robustness_v3.run --workers 14 --verify-only
$PY -m robustness_v3.run --workers 14 --datasets synthetic
$PY -m robustness_v3.analyse --datasets synthetic --out /tmp/robustness_v3_preview
$PY -m robustness_v3.pilot --workers 14 --frames 100 --out /tmp/robustness_v3_timing
$PY -m robustness_v3.run --workers 14 --datasets dronerf birdclef2023 esc50_animals --detach
$PY -m robustness_v3.analyse
```

The final command defaults to `output/classification-emd-irmf/robustness_v3/`.
All report destinations must be new. Outputs are immutable and NumPy loads and
saves prohibit pickle. The v3 external output budget is 30 GB. Do not change
modules recorded in the manifests after the real run begins.
