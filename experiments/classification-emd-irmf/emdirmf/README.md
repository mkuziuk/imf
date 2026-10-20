# emdirmf

This package holds the code shared by the EMD vs IMF vs IRMF classification
study: loading, decomposition, features, cross-validation and paired statistics.
`../PROTOCOL.md` is the binding spec. This file documents how the package
implements it.

## Setup

Run commands from `experiments/classification-emd-irmf/`, or put that folder on
`sys.path`. Scripts in `baselines/` and `ml/` insert it themselves.

```bash
cd experiments/classification-emd-irmf
PY=../../.venv/bin/python
$PY -m emdirmf.validate synthetic
$PY -m emdirmf.run_decompose --dataset synthetic --method emd imf_linear irmf --workers 6
$PY -m emdirmf.run_features  --dataset synthetic --method raw emd imf_linear irmf
$PY baselines/run_classical.py --dataset synthetic --workers 6 --breakdown noise_regime
$PY -m pytest -q tests
```

`DATA_ROOT` comes from `IMF_CLS_DATA` (default `/Users/mikhail/Datasets/imf-classification`).
Worker counts are capped at 6. Set `IMF_CLS_MAX_WORKERS` to change the cap.
Every worker process runs with one BLAS/OpenMP thread.

## Method tags

| tag | method |
|---|---|
| `raw` | normalized frame as a single component (no decomposition) |
| `emd` | `PyEMD.EMD()` default sifting, first 8 IMFs + residue (9 slots) |
| `imf_linear_a1.41` (alias `imf_linear`) | `pimf.IMF(pimf.Quadratic())`, a = sqrt 2 |
| `irmf_h2_a1.41` (alias `irmf`) | `pimf.IMF(pimf.SmoothAbs(H))`, H = 2 sigma_hat, a = sqrt 2 |
| `irmf_h1_a1.41`, `irmf_h4_a1.41` | h_ratio ablations |
| `irmf_h2_a2`, `imf_linear_a2` | a = 2 ablations (about half the slots) |

`parse_method(tag)` accepts any `irmf_h{R}_a{A}[_it{N}]` and `imf_linear_a{A}`.
`A = 1.41` means sqrt 2 exactly. `_it{N}` overrides the IRMF `max_iter` (default 60,
omitted from the canonical tag) for sensitivity runs.

## Decomposition rules (`decompose.py`)

- `normalize_frame(x)` computes `(x - median) / (1.4826 MAD)` along the last axis and
  returns `(normalized, log_scale)`. If MAD is 0 it divides by the std instead. A
  constant channel is centered and gets scale 1. `scale_source` records which case
  applied: 0 MAD, 1 std, 2 constant.
- `irmf` sets `H = h_ratio * sigma_hat`, where
  `sigma_hat = median|diff(y)| / (0.6745 sqrt 2)` on the normalized channel. If
  that median is 0, it falls back to `mean|diff| sqrt(pi) / 2`. pimf defaults
  apply: `lookup=True`, `max_iter=60`, `tol=1e-6`, `workers=1`.
- The two IMF methods share one schedule: `h1 = 0.25`, `h[k+1] = h[k] / a`,
  `k_max=None`, `h_min = 1.5 / L`. The squared-triangle kernel gives zero weight at
  |offset| = hL, so a stage with radius hL <= 1 sample is the identity: it would
  move the whole residual into one component. With `h_min = 1.5 / L` the last
  radius always lies in [1.5, 1.5a) samples, which means 3 or 5 nonzero taps (pimf
  window 5 or 7) for any L and for both a = sqrt 2 and a = 2. For power-of-two L
  and a = sqrt 2 the last stage has radius 2 (window 5, taps 1/6, 2/3, 1/6). Stage
  counts K at a = sqrt 2 are 15, 17 and 19 for L = 1024, 2048 and 4096; at a = 2
  they are 8, 9 and 10.
- Storage order (PROTOCOL): slots `0..K-1` hold the extracted components from fine
  to coarse, and slot `K` holds the solver residual.
  - EMD: `imf1..imf8`, then `residue_coarse`. The residue is the input minus the
    kept IMFs, so IMFs 9 and up are folded into it. If EMD finds fewer than 8 IMFs,
    the missing rows are zero and `n_components` / `n_imfs` give the true count.
  - IMF methods: the pimf stages are reversed (`w5, w7, ..., w513`), then
    `residual_fine`. The pimf residual is what the smallest window left behind, so
    it is the *finest* content, not a trend. `scale_order(tag, L)` returns the
    permutation that sorts every slot strictly from fine to coarse; `slot_labels`
    names the slots.
- Per-signal functions: `emd_components`, `imf_linear_components`,
  `irmf_components` return `((K+1, L) float64, info)`. `decompose_signal(y, tag)`
  dispatches between them. `decompose_frame(frame (C, L), tag, catch=)` normalizes
  and decomposes each channel and records timing.

## Files under `DATA_ROOT/NAME/`

```
prepared/frames.npy, meta.parquet, dataset.json     (dataset agents)
decomp/TAG.npy           float32 memmap (N, C, K+1, L)
decomp/TAG.meta.parquet  per (frame_id, channel): ok, error, n_components, n_imfs (EMD),
                         sigma_hat, H, stage_iterations, stage_converged, n_unconverged (IRMF),
                         log_scale, scale_source, seconds, cpu_seconds
decomp/TAG.json          params, schedule, slot labels, versions, s/frame, failures,
                         EMD IMF-count histogram, IRMF per-slot non-convergence
decomp/TAG.parts/        resume state (manifest + one parquet per finished chunk)
features/TAG.parquet     frame_id, decomp_ok, ch{c}_log_scale, ch{c}_c{k}_{feature}
features/TAG.json        feature list, slot labels, timing
splits_<task>.json       shared CV splits (test indices per seed and fold)
results/classical_<task>_predictions.parquet   out-of-fold predictions
```

`run_decompose` processes chunks of 64 frames on a process pool. The parent writes
each finished chunk into the memmap, flushes it, and only then writes the chunk
marker, so an interrupted run resumes where it stopped. A frame that raises is
stored as NaN with `ok=False` and its error text, and its features become NaN.
If the frames, parameters or chunk size changed, the cache is rejected; pass
`--overwrite`. A cache from a `--limit` run is replaced automatically.

## Features (`features.py`)

Each component gets the same 20 features, computed from its own frame only.
Frequencies are in cycles per sample.

`energy_frac, log_energy, std, mean_abs, skew, kurt, crest, zcr, extrema_rate,
hilb_amp_mean, hilb_amp_std, hilb_if_mean, hilb_if_std, spec_centroid, spec_bw,
spec_entropy, spec_peak, hjorth_mob, hjorth_cpx, perm_entropy`

- `zcr` counts crossings of the component mean.
- The instantaneous-frequency statistics are weighted by the Hilbert envelope.
- The spectral features use a Hann-windowed periodogram.
- Permutation entropy uses order 4 and is normalized.
- A component with std < 1e-6 (for example an EMD padding row) gets 0 for every
  shape feature.

Columns are named `ch{c}_c{k}_{feature}`, plus `ch{c}_log_scale`.
`feature_columns(columns, include_log_scale=False)` selects model inputs.
Throughput is about 1 ms per frame at L = 1024 with 16 slots.

## Evaluation (`cv.py`, `stats.py`)

- `make_splits(name, task)` builds `StratifiedGroupKFold(5, shuffle=True)` with
  seeds 0, 1 and 2, giving 15 splits. It saves them once, refuses to reuse them if
  frames, groups or labels have changed, and asserts that no group appears on both
  sides of a split. `load_splits` reloads them.
- `evaluate(factory_or_estimator, X, y, splits, groups=, n_jobs=, return_predictions=)`
  returns one row per split: balanced accuracy, macro-F1, accuracy, `recall_<class>`,
  fit and predict seconds, and the best parameters. When the estimator is a
  `GridSearchCV`, `groups` are passed through so that the inner CV is grouped too.
- `compare_methods(long_df, metric=, pairs=, by=)` expects a long table with columns
  method, split_id, metric, value, n_train, n_test and the `by` columns. For each
  pair it reports the mean difference with a 95% CI (Nadeau-Bengio corrected
  standard error), the corrected t-test, a Wilcoxon signed-rank test, win/tie/loss
  counts, and Holm-adjusted p-values over the whole table. The default pairs are
  irmf-emd, irmf-imf_linear, imf_linear-emd, and each method against raw.

`baselines/run_classical.py` runs phase 2. It computes Kruskal-Wallis H and
epsilon² and binned Miller-Madow mutual information per feature. It then fits
LDA (Ledoit-Wolf), QDA (tuned covariance shrinkage), Gaussian NB, L2 logistic
regression (C tuned), kNN (k tuned) and linear SVM (C tuned). Each runs as
imputer, constant filter, StandardScaler and classifier, with tuning by grouped
inner 3-fold CV. Each runs with and without `log_scale`. `--breakdown COLUMN`
adds metrics per subset of a meta column. Outputs go to
`output/classification-emd-irmf/NAME/classical/`.

## Timing (one process, CPU seconds per frame and channel)

Measured on this machine with other jobs running. The input was a multiscale test
signal with impulses.

| L | emd | imf_linear_a1.41 | irmf_h2_a1.41 | irmf_h2_a2 | features (a = sqrt 2 slots) |
|---|---|---|---|---|---|
| 1024 | 0.013 | 0.0014 | 0.088 | 0.052 | 0.9 ms |
| 2048 | 0.032 | 0.0050 | 0.32 | 0.19 | 1.9 ms |
| 4096 | 0.080 | 0.020 | 1.23 | 0.73 | 4.1 ms |

With 6 workers, IRMF on 20,000 frames at L = 2048 takes about 18 minutes.

## Implementation choices

- EMD uses PyEMD (EMD-signal 1.10.0), as PROTOCOL specifies. It is deterministic,
  its residue is exact (`S - sum(imfs)`), and it is the reference Python
  implementation. The `emd` package is 3-10x faster, but EMD costs at most 15% of
  IRMF, so speed does not justify changing the solver.
- IRMF often stops at `max_iter = 60` on the second-finest stage (window 7, radius
  2.8 samples). On `synthetic`, 66% of frames do not converge there, 3-4% fail at
  windows 5 and 9, and only 516 of 1,500 frames converge at every stage. The decomposition stays exact,
  because the residual absorbs whatever the stage leaves, but that component is a
  partial M-estimate. The per-slot rates are recorded in `TAG.json` under
  `irmf.unconverged_fraction_per_slot`.
- Bursty real frames (std well above the MAD scale, e.g. a dronerf frame with std 33
  and max 123 in MAD units) can miss convergence at every stage: the clipped
  gradient step is about 1.2 H per iteration, which is small next to the spread of
  the window. Such a channel costs 1.75 s instead of about 0.3 s at L = 2048, and
  even `_it1000` converges only 9 of its 17 stages.
