# Impulsive-contamination experiment

Read [the pre-registration](PROTOCOL_robustness.md). This experiment writes only
new robustness outputs and reads prepared frames, clean caches and saved splits.
The excluded pilot subset, seed mapping and analysis families are fixed there.

Run from `experiments/classification-emd-irmf/`:

```bash
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=../../.venv/bin/python
$PY -m pytest -q tests robustness/tests
$PY -m robustness.pilot --dataset synthetic --workers 6
$PY -m robustness.pilot --dataset dronerf --workers 6
$PY -m robustness.pilot --dataset birdclef2023 --workers 6
$PY -m robustness.pilot --dataset esc50_animals --workers 6
```

The implementation adds these resumable commands after pre-registration:

```bash
$PY -m robustness.run --dataset synthetic --workers 6
$PY -m robustness.run --dataset dronerf --workers 6
$PY -m robustness.run --dataset birdclef2023 --workers 6
$PY -m robustness.run --dataset esc50_animals --workers 6
$PY -m robustness.analyse
```

Completed outputs are reused without replacement. A failed/incompatible checkpoint
requires inspection; no overwrite flag is provided. Full component stacks exist
only for the 100 fixed pilot/figure frames per condition. Numeric NPZ files use
allow_pickle=False; estimators are never pickled. Small report tables and figures
will live in `output/classification-emd-irmf/robustness/`.

The active feature provenance files are `manifest_v2.json` and
`clean_verification_v2.json`. Each `features/<condition>/part_<start>.npz`
contains all four methods' numeric arrays. The failed first raw-precision audit
is preserved separately; it produced no contaminated outputs. The report builder
verifies exact clean reproduction and recomputes all per-split metrics from the
saved Unicode predictions. `run_real` runs the three real datasets serially.

The 2026-10-21 amendment adds all 87 synthetic contrasts in one separate Holm
family. Real-data family membership remains as pre-registered. Every summary
figure includes all four datasets. `analyse --out <new-directory>` can render
another report copy without replacing an existing artifact directory.
