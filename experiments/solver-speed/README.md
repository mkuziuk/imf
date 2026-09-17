# IMF solver speed

Open `solver_speed_comparison.ipynb` for the executed experiment and figures. The experiment compares the lookup-based GD solver in `../gd-irmf/gd_irmf.ipynb` with QuantLet's scalar golden-section search.

Each solver receives the same observations, kernel, H and bandwidth schedule within a test case. The two cases use the GD notebook settings and QuantLet's default zero-replacement settings. Both use a grid without the repeated endpoint. Timings cover a complete single-signal decomposition, including GD thread-pool creation at every stage.

The original implementations remain unchanged. `benchmark_support.py` loads their solver functions and distributes GD's independent target windows across threads. QuantLet runs only on one thread; requests for more workers are rejected. Original stopping rules are retained; output agreement and GD convergence are checked separately.

## Run

Use the repository's `requirements.txt`. The QuantLet clone must be next to the imf repository, in `../IRMF-Intrinsic-Robust-Multiscale-Filtering`. The recorded source commits and file hashes are in `results/benchmark.json`.

From the imf repository root, execute the notebook with a fresh kernel:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace experiments/solver-speed/solver_speed_comparison.ipynb --ExecutePreprocessor.timeout=2400
```

The notebook sets numerical-library thread limits before importing NumPy. By default, it reads the saved measurements and rebuilds the charts. Set `RERUN_BENCHMARK = True` for fresh timings. This tests GD with 1, 2, 4, 8, 12 and 16 workers and QuantLet with one worker, with three repetitions per configuration. The 42 runs execute sequentially in shuffled order and replace the saved results.

The current results retain the original GD runs and QuantLet one-thread runs. Timing values and measurement-time source hashes are unchanged. The JSON's `scope_revision` records the filtering step and current adapter hash. QuantLet's threaded wrapper and its results have been removed.

To rebuild only the report from completed results:

```bash
.venv/bin/python experiments/solver-speed/report_builder.py
```

Report generation uses the bundled `report_style.css`, adapted from the `ssh-html-report` dark template. It writes `tmp/reports/imf-solver-speed.html` and refreshes `tmp/index.html`.

## Results

- `results/benchmark.json`: every timing, validation, machine details and source hashes.
- `results/benchmark_summary.csv`: median, observed range and thread speedup.
- `results/figures/`: SVG charts and PNG previews.
- `results/*_example.npz`: inputs and single-thread reference components.
- `results/pilot*`: preliminary checks, excluded from the final report.

Times measure these inputs on this machine. They exclude imports, lookup-table creation, plotting and pooled-reference calculations. The experiment checks solver agreement, not signal-recovery quality across different settings.

## Signal-length sweep

`scaling_benchmark.py` times full decompositions at 10 evenly spaced sizes from 1,000 to 20,000 points. Both solvers use the GD notebook defaults, including sigma = 0.2, H = 2 × sigma = 0.4, the squared-triangular kernel and the same noise model. GD uses 1, 4 and 12 workers at all sizes. QuantLet always uses one worker.

The original pass measured QuantLet through 5,222 points, then stopped because the projected full sweep exceeded the requested 40-minute limit. `extend_quantlet.py` adds larger QuantLet measurements and checks their components against the saved GD outputs. It reuses all earlier timings. The current measured endpoint is recorded in `results/scaling.json` and shown in the report.

The original window-schedule rule stays fixed: start with about half the signal length, divide by sqrt(2), and stop at 31 points. Larger signals therefore have more stages. The chart measures the full default workflow, including this change in stage count.

There is one measured run per included size and configuration, after a 1,000-point warmup. No configurations run concurrently. The original pass shuffled configuration order within each size; the extension runs only QuantLet in ascending size order. The measurements do not estimate run-to-run variability. GD runs are checked for convergence, reconstruction and agreement across thread counts. Agreement with QuantLet is checked at every size where it was measured.

To resume or reuse the original sweep:

```bash
.venv/bin/python -u experiments/solver-speed/scaling_benchmark.py
```

To extend QuantLet through 13,667 points with a hard 40-minute budget for the added work:

```bash
.venv/bin/python -u experiments/solver-speed/extend_quantlet.py --max-n 13667 --budget-seconds 2400
```

The extension uses a separate process with a parent-enforced timeout. Each completed size is saved with its output checks. An interrupted measurement is excluded. Already measured sizes are reused, so rerunning this command after completion does not repeat them. This requires the cached GD component arrays in `tmp/solver-speed-scaling/`. The JSON preserves the original stop decision and source hashes, and records each extension separately.

The notebook reads completed results by default. `RERUN_SIZE_SWEEP` can resume the original experiment, and `EXTEND_QUANTLET` enables the bounded extension. To measure the original sweep again from scratch, move `results/scaling.json` aside first.

## Scaling-law fits and chart controls

The HTML report switches between seconds and time relative to each method's own time at 1,000 points. Both displays support linear and log–log axes. A checkbox shows or hides the fitted power laws. The report opens with the relative view. All controls and figures work offline.

The relative view divides each measured time by its measured 1,000-point baseline. Dashed curves divide each original fit by its own fitted 1,000-point value, giving `(n / 1000) ** p`. Both start at 1, and the original fitted powers stay unchanged. A higher relative curve means a larger proportional increase; the seconds view shows the actual waiting time.

`scaling_charts.py` fits `time = a * (n / 1000) ** p` by unweighted least squares in log space. The fitted curves are straight on log–log axes and stop at each solver's measured endpoints. A separate shared-range fit uses only sizes measured for all four configurations. The report shows the power p and the time multiplier `2 ** p` for doubling the signal length. These are descriptive fits over the measured range, not asymptotic complexity bounds or estimates of timing uncertainty.

- `results/scaling.json`: every timing, setting, source hash and output check.
- `results/scaling_summary.csv`: times and component-agreement checks.
- `results/scaling_normalized.csv`: measured times, each configuration's 1,000-point baseline, and their ratios.
- `results/scaling_fits.json` and `.csv`: coefficients, doubling factors, log-space R squared, fit ranges and shared-range fits. JSON includes a hash of the timing inputs.
- `results/figures/scaling_linear.*` and `scaling_loglog.*`: measured curves. Files ending in `_fit` add the fitted curves.
- `results/figures/scaling_relative_linear.*` and `scaling_relative_loglog.*`: the normalized views, also with optional `_fit` curves.
- `scaling_report.py`: scale controls and simple explanations.
- `tmp/solver-speed-scaling/`: temporary component arrays and snapshots from before extensions.

Once measurements finish, rebuild the HTML with `report_builder.py` or execute the notebook. The interactive size chart appears immediately after the original timing block.
