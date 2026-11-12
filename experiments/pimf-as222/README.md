# pimf versus AS222-style FFT

Reproduce the supplied five-sine experiment with `pip install pimf==0.4.0`
replacing the report's hand-written direct solver. The concise PDF and editable
LaTeX are in `overleaf/pimf_as222_comparison.*`.

## Run

From the repository root, with Python 3.14 for the recorded dependency versions:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/pimf-as222/requirements.txt
.venv/bin/python experiments/pimf-as222/benchmark.py
.venv/bin/python experiments/pimf-as222/parallel_benchmark.py
.venv/bin/python experiments/pimf-as222/build_report.py
latexmk -pdf -cd overleaf/pimf_as222_comparison.tex
```

The benchmarks overwrite their respective outputs. To rebuild the report from
saved measurements, run only the final two commands. `--skip-timing` runs the
numerical checks and Monte Carlo experiment without timing, but replaces
`results.json` with a result that cannot build the runtime report.

## Exact comparison settings

- Periodic grid `arange(1000) / 1000`; five frequencies `1, 3, 8, 20, 50`.
- Amplitudes `.30, .25, .20, .15, .10`; iid Gaussian noise with standard deviation `.2`.
- Classical Epanechnikov kernel; bandwidths `.1, .0829, .0303, .0128, .00757`; `H=1`.
- `default_rng(20261004)` with one validation observation vector consumed first,
  followed by 20 paired observation vectors. The PDFs specify the seed and skipped
  draw but not the generator. PCG64 reproduces all 25 rounded table entries.
- Every stage uses `IMF(contrast=SmoothAbs(1), kernel=Epanechnikov(), boundary="wrap")`
  through its public `decompose(residual, h1=h, k_max=1)` API. The returned residual
  feeds the next call. Explicit integer window sizes would change fractional kernel
  widths, and the default geometric schedule would change the experiment.
- Retain the package's defaults `lookup=True`, `tol=1e-6`, `max_iter=60`.
  The initial run uses `workers=1`; the parallel run uses `workers=os.cpu_count()`,
  which is 16 on the recorded Apple M4 Max. The report presents the parallel run.
  This differs from the source report's safeguarded Newton solver and `1e-12` score
  tolerance. Independent roots and a tighter package run quantify the difference.
- The FFT comparator uses the source report's exact Gaussian score and its derivative,
  and the FFT of the same sampled discrete kernel. Each method evolves its own residual.

The historical papers were downloaded from Oxford Academic and checked directly.
Härdle's algorithm dates to 1987. Dusoir's 1988 remark corrects two initialization
errors. The original AS222 uses Huber scores, Gaussian smoothing and discretization
of a random design. This benchmark preserves the supplied comparison's adaptation;
it does not claim to translate or time the original Fortran or recreate its tables.

## Checks and results

The executable validates 25 local fits against exact-score Brent roots, the full
validation signal's score residuals, all five kernels against independent sampled
weights, and FFT versus spatial evaluation of the same one-step expression. It also
checks reconstruction, stage convergence, cached versus uncached package equality,
the default versus tighter solver, and the linear-score operator identity.

The inherited agreement gate requires correlation at least `.995` and relative RMS
difference at most `.05` in every one of the 100 pairs. All 100 pairs fail. The
conditional 1,000-replicate extension therefore remains untriggered. The program
contains the extension logic should a changed experiment pass the gate.

The new package results agree with every source accuracy entry within its displayed
rounding. The first source table corresponds to the direct solver, so the CSV uses
`pimf_*` names for the replacement's corresponding metrics. This is not a bitwise
curve comparison: the source `benchmark.py`, JSON, CSV and NPZ files were not supplied.

Fresh timing uses one input per size, one warmup and five repetitions per method.
The two methods run sequentially in randomized order within each repetition. Exact
package kernel weights are precomputed using a cache subclass; the solver is
unchanged and outputs match uncached calls bit for bit. The FFT spectra are also
precomputed. Timing includes all five smoothing calls and residual updates, but
excludes simulation, setup, checks, metrics and plotting. The initial results use one worker.
No speed claim against the source report's historical direct timings is made.

The parallel benchmark retimes one worker, 16 workers and FFT sequentially in one
randomized protocol, retaining native numerical-library thread limits of one.
Thread-pool creation for every stage is included. The package imposes a minimum
chunk size of 64 samples, so requested `workers=16` gives pool capacities of 4, 8,
16, 16, 16 and 16 across the six sizes. At `n=1000`, 16 workers take 39.29 ms versus
12.99 ms with one worker. At `n=8000`, they take 143.88 ms versus 793.37 ms, a 5.51x
speedup. More workers benefit the larger arrays in this experiment.

All 100 parallel stages converge. Parallel and serial component arrays differ by
at most `1.35e-7` over the 20 paired inputs, and all 25 source-table entries still
match within rounding. Independent root, exact-score, tight-solver, reconstruction
and cached/uncached checks also pass for the parallel configuration. Thread chunks
stop independently, so bitwise equality across worker counts is not required.

Generated data in `output/pimf-as222/`:

- `results.json`: parameters, measurements, checks, package versions and hashes.
- `pilot_paired_metrics.csv`: all 100 paired comparisons, including true-sine RMSE.
- `pilot_component_summary.csv`: per-component means and Monte Carlo standard errors.
- `pilot_solver_diagnostics.csv`: all 100 package stage diagnostics.
- `pilot_curves.npz`: all paired inputs and component arrays, plus true components.
- `root_validation.csv`: all 25 independent root checks.
- `timing_raw.csv` and `timing_summary.csv`: individual timings and observed ranges.
- `source_reconciliation.csv`: every rounded source-table comparison.
- `parallel_results.json`: the 16-worker run, checks and fresh three-configuration timings.
- `parallel_*.csv` and `parallel_curves.npz`: corresponding raw timings, paired metrics,
  summaries, solver diagnostics, root checks, source comparisons and saved curves.

`research/pimf-as222/references/` contains the two publisher PDFs. The supplied
comparison PDFs stay in Downloads; optional file hashes are recorded when present.
Reading or rerunning the experiment does not require those PDFs.
