# Component-count experiment

Read [PROTOCOL_components.md](PROTOCOL_components.md) before running. It fixes
the matched-count, native-count and default-k_max questions before new input
classification. No shared module, original report, installed pimf file or raw
dataset is modified.

Run from the study directory in this checkout:

```bash
cd /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
study_python=../../.venv/bin/python
"$study_python" -m pytest -q tests components/tests
"$study_python" -m components.pilot
"$study_python" -m components.processing --workers 6
"$study_python" -m components.classification --phase defaults --workers 6
"$study_python" -m components.classification --phase experiment --workers 6
"$study_python" -m components.analysis
```

The pilot is append-only and should run only once in this data root. Processing
and classification resume completed immutable block/fit checkpoints; an
incomplete checkpoint is an explicit error. Default validation precedes new
input classification. Each entry records and checks pimf against
[pimf_baseline.json](pimf_baseline.json). A changed version or hash stops the run.
The final analysis requires complete expected coverage and verified defaults.

Use `--datasets synthetic` to restrict a runner for diagnosis. Stored native
stacks, features, diagnostics, predictions and logs remain in each external
dataset's components/ directory. Only small tables, plots and the final report
are committed under output/classification-emd-irmf/components/. Output exports
are created once; use a fresh named export destination for a later rerun.
