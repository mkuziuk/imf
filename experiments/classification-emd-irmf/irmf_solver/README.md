# IRMF convergence audit and solver prototype

`newton.py` re-exports `NewtonSmoothAbs` and `newton_components` from
`emdirmf/irmf_newton.py`. The shared implementation uses pimf's public
`solve(windows, weights)` hook without patching pimf. It starts at the weighted mean and solves each window's
monotone score equation with safeguarded Newton steps and periodic bisection.
Each row stops independently. A single-threaded Numba loop fuses score and
derivative evaluation without allocating an L-by-window temporary per iteration.

The derivative is the slope of pimf 0.4.0's actual piecewise-linear lookup score,
including its saturated tails. Using the Gaussian analytic derivative would
instead mix two numerical problems. Kernel weights, wrap padding, H, schedule,
and component order come from the unchanged study code and pimf driver. The
prototype solves the score equation used by pimf; pimf's interpolated loss has a
different exact derivative. The private lookup-table imports pin this prototype
to pimf 0.4.0. A library implementation should expose a public score derivative.

The original audit below predates integration. `INTEGRATION.md` records the
converged shared tags, complete caches, diagnostics and GD60 differences.

All commands below run from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594`.
Use the absolute interpreter and script paths when invoking elsewhere.

```bash
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py diagnose
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py sensitivity
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py validate
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py benchmark
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py mechanism
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py second_channel
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_study.py classify
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/write_report.py
```

Every command is sequential, uses one computational process, and limits native
thread pools to one. Two independent dataset commands can run concurrently.
Sensitivity uses seed 20261020 and ten frames per primary class, sampled across
groups: 100 DroneRF, 160 BirdCLEF, and 100 ESC-50 frames, first channel only.
The benchmark excludes JIT compilation and compares GD60, GD at max_iter=20000
and tol=1e-12, and Newton at root_tol=1e-12 on noisy and bursty signals at
each length, plus a bimodal stress case at L=1024. `validate`
checks 240 independent scalar roots against SciPy brentq, including constant
windows and center outliers, and records a window-7 flat-score example.

All numerical outputs, resumable per-frame NPZ files, and Numba caches go under
`${IMF_CLS_DATA:-/Users/mikhail/Datasets/imf-classification}/solver_audit/`.
Remove a frame's NPZ to recompute it; remove benchmark JSON checkpoints to
rebenchmark. Use a separate data root when changing the study settings or seed.
No full-dataset decomposition is performed.

`diagnosis_stage.csv` and `diagnosis_class.csv` cover all cached stages and classes;
`diagnosis_overview.csv` includes full iteration, delta-ratio, and timing quantiles.
The stopping-rule denominator uses the cached fitted component's maximum, with
a check that float32 caching reproduces every recorded convergence flag.
Subset component errors and feature errors are recorded against both GD1000
and Newton, including individual components, feature names, Spearman correlation
across frames, and feature RMSE divided by reference across-frame SD.

The classification check is deliberately small. It uses fixed shrinkage LDA,
training-fold preprocessing, and intersections of the study's saved 15 grouped
folds with the subset. DroneRF uses the available binary split and binary labels;
the sampling quota remains balanced across its ten primary classes. Some small
test folds omit classes. Balanced accuracy, macro-F1, accuracy, and per-class
recalls use the unchanged shared metric function. Both component-only and
component-plus-log_scale versions are evaluated. Nadeau-Bengio corrected tests
are primary, Wilcoxon secondary, and Holm correction covers all 30 comparisons
in each metric table. These checks cannot establish full-study superiority.

`REPORT.md` is the readable record; the data-root CSV and JSON files preserve
the detailed measurements. No package installation or git commit is needed.
