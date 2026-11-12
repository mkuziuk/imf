# Converged IRMF integration

Verified 2026-10-20T20:26:53+00:00. pimf 0.4.0, Python 3.12.

`emdirmf/irmf_newton.py` now owns the audited lookup-compatible safeguarded Newton solver. The prototype module is a thin import shim. The public pimf solve hook preserves normalization, H estimation and fallbacks, squared-triangle kernel, wrap boundary, bandwidth schedule and slot order. No installed pimf files or PROTOCOL.md were changed. No git commit was made.

Tags are `irmf_h2_a1.41_conv` and the ablations `irmf_h1_a1.41_conv`, `irmf_h4_a1.41_conv`, `irmf_h2_a2_conv`. `_conv` excludes `_it{N}`. Solver version 1.0.0 uses root tolerance 1e-12, a 100-iteration cap, and bisection every eighth update or when a Newton candidate leaves its bracket. The stopping test bounds the root step or bracket width by `1e-12 * (1 + abs(root))`. Parameters record this provenance. Numba warms one strided-window signature per worker; it uses no parallel Numba loop.

## Validation and outputs

The complete suite passed: **98 tests**. It covers tag canonicalization, suffix exclusion, exact float64 reconstruction, tight-GD agreement at max_iter=20000 and tol=1e-12 within 1e-8 relative L2 per component, identical stage order, failure propagation to NaN caches/features, multiprocessing metadata, JIT signature reuse, and byte-identical float32 GD cache slices and stage diagnostics. GD regression checks use frames 0 and 1 of all five datasets plus ESC-50's a=2 cache. Existing GD parameter dictionaries match the stored manifests exactly.

Every listed output has a full float32 decomposition, meta parquet, JSON, and feature parquet/JSON under DATA_ROOT. Features report zero failed frames and zero nonfinite rows. Full-cache reconstruction was checked after float32 storage.

| Dataset | h2/a1.41 conv | h1/a1.41 conv | h4/a1.41 conv | h2/a2 conv | linear a2 |
|---|---|---|---|---|---|
| synthetic | complete | complete | complete | complete | outside requested scope |
| uas_birds_60ghz | complete | complete | complete | complete | complete |
| esc50_animals | complete | complete | complete | complete | complete |
| birdclef2023 | complete | complete | complete | complete | complete |
| dronerf | complete | complete | complete | complete | complete |

BirdCLEF, DroneRF and ESC-50 linear-a2 decompositions were reused. Their missing features were generated, including ESC-50. UAS linear-a2 decomposition and features were generated. The interrupted ESC-50 `irmf_h1_a1.41` GD cache was left alone.

## Convergence and timings

All channel-stages converged for every convergence tag. There were no decomposition failures. Per-stage arrays in the meta parquet record maximum/mean row iterations, total/maximum row bisection fallbacks, maximum absolute weighted score at returned roots, and convergence. `stage_iterations` is the maximum over window rows, rather than pimf's solve-hook placeholder of 1.

All runs used two worker processes with one native thread each, sequential datasets and tags. Other phase-2 processes were active. Decomposition wall times include process startup, worker JIT warmup and cache I/O. CPU time sums measured channel fits and excludes worker warmup. Feature times are computation wall times reported by the shared CLI, before table serialization.

| Dataset | Tag | Decomp wall s | Fit CPU s | Features s | Mean row iterations | Stage max iterations, p50/p95/max | Bisections total, per root | Max score residual |
|---|---|---:|---:|---:|---:|---|---|---:|
| synthetic | irmf_h2_a1.41_conv | 13.4 | 23.7 | 1.9 | 3.183 | 4/7/28 | 60787, 0.0026383 | 1.33e-15 |
| synthetic | irmf_h1_a1.41_conv | 14.0 | 24.8 | 1.8 | 3.505 | 5/13/28 | 227096, 0.0098566 | 2.11e-15 |
| synthetic | irmf_h4_a1.41_conv | 12.5 | 21.7 | 1.9 | 2.929 | 4/6/22 | 7377, 0.00032018 | 1.26e-15 |
| synthetic | irmf_h2_a2_conv | 8.3 | 13.6 | 1.7 | 3.213 | 4/7/21 | 39515, 0.0032157 | 1.33e-15 |
| uas_birds_60ghz | irmf_h2_a1.41_conv | 13.8 | 24.6 | 1.9 | 3.270 | 4/5/28 | 4270, 0.00018155 | 1.25e-15 |
| uas_birds_60ghz | irmf_h1_a1.41_conv | 14.7 | 26.6 | 1.8 | 3.647 | 5/7/30 | 27981, 0.0011897 | 1.2e-15 |
| uas_birds_60ghz | irmf_h4_a1.41_conv | 12.7 | 22.2 | 1.9 | 2.944 | 4/5/20 | 368, 1.5646e-05 | 1.23e-15 |
| uas_birds_60ghz | irmf_h2_a2_conv | 8.6 | 14.2 | 1.7 | 3.329 | 4/6/28 | 4407, 0.00035132 | 1.22e-15 |
| esc50_animals | irmf_h2_a1.41_conv | 166.3 | 328.5 | 7.0 | 3.348 | 5/7/67 | 1.16058e+06, 0.0065362 | 2.78e-13 |
| esc50_animals | irmf_h1_a1.41_conv | 183.8 | 363.5 | 6.7 | 3.665 | 5/20/76 | 2.5997e+06, 0.014641 | 2.84e-13 |
| esc50_animals | irmf_h4_a1.41_conv | 150.9 | 297.6 | 6.7 | 3.037 | 4/6/67 | 617416, 0.0034772 | 7.7e-14 |
| esc50_animals | irmf_h2_a2_conv | 96.8 | 190.0 | 4.4 | 3.433 | 5/7/64 | 716534, 0.0076224 | 9.91e-14 |
| birdclef2023 | irmf_h2_a1.41_conv | 349.7 | 695.0 | 14.6 | 3.067 | 4/6/64 | 914321, 0.0021885 | 3.1e-14 |
| birdclef2023 | irmf_h1_a1.41_conv | 376.0 | 747.6 | 14.3 | 3.350 | 5/11/74 | 2.47691e+06, 0.0059286 | 3.1e-14 |
| birdclef2023 | irmf_h4_a1.41_conv | 328.0 | 650.6 | 14.0 | 2.822 | 4/5/62 | 374984, 0.00089754 | 4.66e-15 |
| birdclef2023 | irmf_h2_a2_conv | 203.4 | 402.5 | 8.6 | 3.130 | 4/6/70 | 611844, 0.0027662 | 3.1e-14 |
| dronerf | irmf_h2_a1.41_conv | 649.7 | 1291.3 | 26.4 | 3.003 | 4/7/75 | 3.58079e+07, 0.042854 | 6.26e-14 |
| dronerf | irmf_h1_a1.41_conv | 671.1 | 1333.4 | 28.8 | 3.245 | 4/19/78 | 5.23781e+07, 0.062684 | 1.19e-13 |
| dronerf | irmf_h4_a1.41_conv | 627.9 | 1247.4 | 28.3 | 2.795 | 3/6/77 | 2.27588e+07, 0.027237 | 2.84e-14 |
| dronerf | irmf_h2_a2_conv | 376.6 | 747.0 | 16.4 | 3.018 | 4/7/77 | 1.96611e+07, 0.044445 | 6.26e-14 |

Completed CLI invocations took 4569.2 s in total. Convergence-tag fit CPU time sums to 8465.9 s. Largest float32 reconstruction relative L2 error was 3.89e-08. `cache_summary.csv` retains counts, per-root maximum fallback counts, timings and reconstruction errors at full precision. All run logs and invocation records are under `/Users/mikhail/Datasets/imf-classification/solver_integration`.

## GD60 versus convergence

The table compares the unchanged primary GD60 tag `irmf_h2_a1.41` against `irmf_h2_a1.41_conv` on every successful matched channel. Each cell is median / 95th percentile of `norm(GD60_component - conv_component) / max(norm(conv_component), 1e-12)`, computed from the float32 caches in float64. The residual is included separately. Relative errors can be large when the converged component has little energy. These are component differences, not classification results.

| Slot/window | synthetic | uas_birds_60ghz | esc50_animals | birdclef2023 | dronerf |
|---|---:|---:|---:|---:|---:|
| w5 | 0.0341 / 0.0609 | 3.11e-07 / 0.00456 | 7.13e-06 / 0.159 | 7.97e-07 / 0.0634 | 8.48e-08 / 0.35 |
| w7 | 0.0539 / 0.0933 | 3e-07 / 0.00392 | 5.28e-06 / 0.111 | 5.13e-07 / 0.056 | 8.41e-08 / 0.604 |
| w9 | 6.64e-07 / 2.5e-06 | 3.79e-07 / 3.17e-06 | 3.68e-06 / 0.0923 | 4.6e-07 / 0.0349 | 1.49e-07 / 0.748 |
| w13 | 3.37e-07 / 1.52e-06 | 4.1e-07 / 3.41e-06 | 2.7e-06 / 0.0845 | 5.74e-07 / 0.0432 | 1.81e-07 / 0.708 |
| w17 | 2.74e-07 / 6.63e-07 | 4.38e-07 / 2.74e-06 | 2.45e-06 / 0.0718 | 8.07e-07 / 0.0507 | 2.26e-07 / 0.616 |
| w25 | 2.65e-07 / 6.39e-07 | 4.62e-07 / 2.46e-06 | 2.56e-06 / 0.057 | 1.19e-06 / 0.0439 | 2.77e-07 / 0.591 |
| w33 | 3.42e-07 / 1.07e-06 | 5.1e-07 / 2.78e-06 | 2.66e-06 / 0.0519 | 1.95e-06 / 0.0427 | 3.56e-07 / 0.445 |
| w47 | 5.43e-07 / 1.66e-06 | 5.98e-07 / 3.24e-06 | 3.36e-06 / 0.0467 | 3.18e-06 / 0.0439 | 4.56e-07 / 0.288 |
| w65 | 8.85e-07 / 2.6e-06 | 8.38e-07 / 4.27e-06 | 4.58e-06 / 0.0404 | 5.13e-06 / 0.0354 | 6.07e-07 / 0.165 |
| w91 | n/a | 1.11e-06 / 4.29e-06 | n/a | n/a | n/a |
| w93 | 1.35e-06 / 3.69e-06 | n/a | 6.56e-06 / 0.0339 | 7.52e-06 / 0.0302 | 7.32e-07 / 0.0952 |
| w127 | n/a | 1.57e-06 / 4.74e-06 | n/a | n/a | n/a |
| w129 | 2e-06 / 4.85e-06 | n/a | 9.16e-06 / 0.0348 | 1.05e-05 / 0.0205 | 9.47e-07 / 0.0578 |
| w179 | n/a | 2.52e-06 / 7.4e-06 | n/a | n/a | n/a |
| w183 | 1.62e-06 / 3.93e-06 | n/a | 1.21e-05 / 0.0227 | 1.31e-05 / 0.0159 | 1.46e-06 / 0.0366 |
| w251 | n/a | 3.79e-06 / 1.36e-05 | n/a | n/a | n/a |
| w257 | 1.09e-06 / 2.97e-06 | n/a | 1.5e-05 / 0.017 | 1.58e-05 / 0.0113 | 1.82e-06 / 0.0167 |
| w355 | n/a | 5.58e-06 / 2.3e-05 | n/a | n/a | n/a |
| w365 | 7.62e-07 / 2.08e-06 | n/a | 1.77e-05 / 0.0158 | 1.83e-05 / 0.00747 | 1.89e-06 / 0.00639 |
| w501 | n/a | 2.47e-06 / 1.87e-05 | n/a | n/a | n/a |
| w513 | 3.7e-07 / 7.99e-07 | n/a | 2.09e-05 / 0.0101 | 2.1e-05 / 0.00557 | 1.85e-06 / 0.00106 |
| w727 | n/a | n/a | 2.09e-05 / 0.00636 | 2.18e-05 / 0.00354 | 1.84e-06 / 0.000183 |
| w1025 | n/a | n/a | 1.15e-05 / 0.00224 | 8.68e-06 / 0.00113 | 1.22e-06 / 4.31e-05 |
| residual_fine | 0.000351 / 0.00105 | 2.7e-07 / 0.000644 | 9.42e-06 / 0.213 | 1.7e-06 / 0.0564 | 8.06e-08 / 4.42 |

Exact per-slot statistics and matched-channel counts are in `solver_integration/gd60_conv_component_differences.csv` under DATA_ROOT.

## Reproduction and suggested protocol wording

```bash
.venv/bin/python -m pytest experiments/classification-emd-irmf/tests -q
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/run_integration.py
.venv/bin/python -I experiments/classification-emd-irmf/irmf_solver/summarize_integration.py
```

Suggested addition to PROTOCOL.md, which remains unchanged:

> Keep the default GD60 IRMF tag as the primary analysis. Add `irmf_h2_a1.41_conv` as the convergence sensitivity analysis, with corresponding H-ratio and a=2 ablations. The `_conv` tag uses safeguarded Newton roots of pimf 0.4.0's lookup score, solver version 1.0.0, root tolerance 1e-12 and at most 100 iterations per window. It shares GD's normalization, H rule, kernel, boundary, schedule and slot order. `_conv` and `_it{N}` are mutually exclusive. Record per-stage iteration counts, bisection fallbacks, maximum score residual and convergence. A stage that misses convergence marks its channel as failed. Repeat paired classification comparisons with and without log_scale using the same grouped splits.
