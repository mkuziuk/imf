# Phase-3 machine learning

Run from the repository root with `.venv/bin/python`. The default methods are
`raw emd imf_linear_a1.41 irmf_h2_a1.41`; aliases and cached ablations also work.
The runners read prepared data, existing shared splits and caches. They never
create decompositions, features or splits. A failed decomposition stops a run
instead of silently changing its paired test population.

```bash
.venv/bin/python experiments/classification-emd-irmf/ml/run_trees.py \
  --dataset synthetic --quick --workers 2
.venv/bin/python experiments/classification-emd-irmf/ml/run_rocket.py \
  --dataset synthetic --quick --workers 2 --classifiers minirocket multirocket
.venv/bin/python experiments/classification-emd-irmf/ml/run_cnn.py \
  --dataset synthetic --quick --workers 2 --device mps
```

Every runner accepts `--task`, `--methods`, `--workers`, `--quick`, `--folds`,
`--device`, `--log-scale both|without|with`, and `--out`. `--workers` limits CPU
threads, with folds processed sequentially and no child data-loader processes.
Use at most 2 while the decomposition jobs are active. CPU models accept
`--device cpu|auto`; CNN accepts `cpu|mps|auto` and prefers MPS automatically.

`--quick` selects the first seed's five folds and reduces model budgets.
`--folds 5` selects the first five shared splits regardless of `--quick`;
`--folds 0,5,10` selects those exact split IDs. Record the same selection for
every compared method. Even one split can be run, but inferential comparisons
require at least two. Full default runs use all 15 shared splits.

All preprocessing and model selection fit training data only. Each outer fold
uses the first class-complete split from a seeded, grouped, stratified 3-fold
partition of its training rows for validation. Too few groups or missing
training classes cause an explicit error. After selecting the budget or
regularization, every runner refits on the entire outer training fold.

## Trees

`--classifiers lightgbm xgboost extratrees` is the default. LightGBM has 15
leaves, XGBoost depth 4, learning rate 0.05 and at most 500 rounds. Weighted
validation log loss with patience 12 selects boosting rounds. ExtraTrees uses
300 trees and selects leaf size 1 or 3 by validation balanced accuracy. Quick
runs cap boosting at 60 rounds and use 64 ExtraTrees trees. Median imputation
and balanced sample weights fit each training partition separately.

Both component-only and component-plus-`log_scale` features run by default.
`importance_folds_<task>.csv` sums normalized model importance over features
and original input channels by component storage slot. The companion
`importance_scales_<task>.csv` gives fold means and SDs in true scale order.
The finest IMF residual has scale rank 0. Log-scale is a separate rank -1 row.
LightGBM split count, XGBoost gain and ExtraTrees impurity importance have
different meanings, so compare them within model families.

## ROCKET

MiniRocket is the default; add `--classifiers minirocket multirocket` for both.
Both use aeon transformers and a logistic head fitted with averaged SGD.
`--features 10080` is the default target feature count for either transformer;
quick runs use 672. MultiRocket pools four statistics from the signal and its
first difference, so its kernel budget is target/8. aeon rounds counts down
to multiples of 84, with a minimum of 84 kernels. Actual dimensions are saved.
The same total budget applies to every method, rather than multiplying it by
component count.
aeon's random kernels combine at most nine channels at once, even on a wider
stack; the same rule applies to all methods.

Transform biases use at most `--fit-samples 256` randomly selected training
frames. Inner transforms fit inner training only. Inner validation chooses
logistic regularization alpha from 0.0001, 0.001 and 0.01. Final transforms,
incremental StandardScaler moments and heads refit outer training only. SGD
uses balanced training weights, deterministic shuffled epochs, 15 passes by
default and 5 in quick mode; override with `--head-epochs`. Head updates are
serial; per-class thread-pool overhead costs more than these small batches.

`--batch-size 32` bounds input-stack reads and `--head-batch-size 128` bounds
head/scaler updates. Transformed features use temporary float32 disk memmaps.
At N=12000 and 10080 features, one transform file uses about 484 MB; inner
and outer files are used sequentially. In-memory stack batches for 36
channels and L=2048 use about 9.4 MB; the 256-frame bias sample uses 75.5 MB,
plus aeon scratch arrays and short-lived copies. The 4 GB stack is never
materialized. Both log-scale variants reuse transforms, so their timing rows
include shared work; summing those rows overstates actual run wall time.

## CNN

The same compact 1-D residual architecture is used for each method: a
stride-2 convolution, three residual blocks of widths 16, 32 and 64, kernel
size 7, GroupNorm, temporal mean/max pooling and a linear head. Only the
number of input channels changes. With `log_scale`, train-standardized scale
values join the pooled features. AdamW uses learning rate 0.001, weight decay
0.001, balanced cross entropy and gradient norm capped at 5. There is no data
augmentation. `--width` changes all widths together and must be divisible by 4.

Default training caps at `--epochs 30`, `--patience 5`, batch size 64. Quick
uses 6 epochs and patience 2. Inner validation loss selects the epoch count;
a freshly seeded model then trains on all outer training frames for that
count. `curves_<task>.csv` records both selection and refit stages. All
PyTorch/NumPy seeds are fixed and deterministic PyTorch operations are
required. MPS and CPU can still give different numerical results.

The DataLoader reads frames from read-only memmaps with `num_workers=0`.
Only a batch and its activations move to MPS. For a cheaper first real-data
pass, use `--folds 5` on every method. Do not pick methods or folds using test
scores.

## Shared stack treatment and outputs

Stack channels are input channel first, then components from fine to coarse
according to `emdirmf.decompose.scale_order`. Raw frames use exactly
`normalize_frame`, with one channel per input channel. There is no per-channel
standardization that would inflate zero-padded components. Both stack runners
accept `--downsample D`: a common anti-aliased `resample_poly` reduction on
every channel of every method. Default D=1 preserves all samples. Coarse
components are not selectively dropped or decimated.

Each runner writes under `output/classification-emd-irmf/<dataset>/ml/<runner>/`:

- `results_folds_<task>.csv`, including per-class recall, fit/predict seconds,
  selected parameters and split IDs;
- `results_long_<task>.csv`, including every metric, recall and timing;
- `comparisons_<task>.csv`, all pairs within classifier and feature set, for
  balanced accuracy, macro-F1 and accuracy. The shared statistics module
  supplies corrected resampled t-tests, corrected 95% CIs, Wilcoxon, and Holm
  within each metric's full table;
- `summary_<task>.md` and `run_<task>.json`, with effective parameters,
  versions, stack order, source decomposition costs and fold timings.

Out-of-fold predictions are written only to
`<DATA_ROOT>/<dataset>/results/<runner>_<task>_predictions.parquet`, with
frame IDs, true/predicted labels and split IDs. A rerun of the same runner/task
replaces its result artifacts. Use `--out` for alternate small-table locations;
the predictions still follow the protocol's fixed results path.

`estimate_costs.py --calibrate-head` reads the saved synthetic timing tables and
benchmarks only synthetic-sized logistic heads on seeded random features. It
writes per-method/fold and per-dataset planning estimates under `synthetic/ml/costs/`.
No real dataset arrays are opened. Tables give five-fold quick pilots and full
CPU-model budgets over 15 folds, plus a five-fold CNN ceiling of 30 selection
and 30 refit epochs. The assumptions and a broad 0.5x-2x planning range are
saved explicitly. Early stopping can make full boosted-tree/CNN runs much cheaper.

Tests are CPU-only and use temporary data:

```bash
.venv/bin/python -m pytest experiments/classification-emd-irmf/tests -q
```
