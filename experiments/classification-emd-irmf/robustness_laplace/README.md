# Laplace-contamination experiment

Read [the protocol](PROTOCOL_laplace.md). It repeats the [impulse experiment](../robustness/README.md)
with Laplace noise on 10–40% of samples and no Gaussian control. The code only writes new
files: each dataset's `robustness_laplace/` directory, `robustness_laplace_runs/` under the
data root, and `output/classification-emd-irmf/robustness_laplace/`.

Run from `experiments/classification-emd-irmf/`:

```bash
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=../../.venv/bin/python
$PY -m pytest -q robustness_laplace/tests
$PY -m robustness_laplace.run --workers 14 --datasets synthetic
$PY -m robustness_laplace.run --workers 14 --datasets dronerf birdclef2023 esc50_animals --detach
$PY -m robustness_laplace.analyse
```

`--detach` prints the PID and log path. One pool serves every dataset. It runs the clean
verification, then all feature chunks, then the fits, longest first. Each matched fit
waits for its clean-trained fit. Completed checkpoints are reused, so rerunning the same
command resumes. Do not edit the modules while a run is active: the manifests record
their SHA-256 and `analyse` refuses to report if they change.

`analyse --datasets synthetic --out /tmp/preview` writes a preview for a subset. The
default output directory is reserved for the full four-dataset report.

The run is faster than the impulse run for three reasons. Clean classical models are
refit at the C the impulse run selected, and each refit is kept only if its clean
predictions match that run's exactly. Matched training covers 40% only. All datasets
share one 14-worker pool. IRMF uses the same Numba Newton solver
(`emdirmf/irmf_newton.py`) as before.
