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
