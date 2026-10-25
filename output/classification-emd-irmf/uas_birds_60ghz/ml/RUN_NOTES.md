# Phase-3 reproduction and choices

Run everything from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594`.
Python is `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python`.
External data root is `/Users/mikhail/Datasets/imf-classification`.
The two tasks are `bird_drone` and `target3`, as declared in `dataset.json`.

## Fixed model budgets

All nine inputs use all 15 saved outer splits, both without and with the two
shared I/Q `log_scale` columns. Methods are raw, EMD, linear IMF a=sqrt(2),
GD60 IRMF H=2 a=sqrt(2), converged Newton IRMF H=2 a=sqrt(2), converged IRMF
H=2 a=2, linear IMF a=2, converged H=1 and converged H=4 a=sqrt(2).

| Model | Budget |
|---|---|
| LightGBM | 15 leaves, at most 500 boosting rounds, weighted validation log loss, patience 12 |
| XGBoost | depth 4, at most 500 boosting rounds, weighted validation log loss, patience 12 |
| ExtraTrees | 300 trees, inner leaf-size choice 1 or 3 |
| MiniRocket | 1,008 actual features, 128 bias-fit frames, five logistic SGD passes |
| MultiRocket | target 1,008, rounded by aeon to 672 actual features; same bias and head budgets |
| 1-D residual CNN | width 8, at most 20 epochs, patience 4, batch size 64, MPS |

Every model sees all 1,000 time samples. No downsampling is applied. Each
ROCKET model has a fixed total feature budget independent of decomposition
channel count. CNN widths and depth are common; input channels change.
The CNN has two raw channels, 18 EMD/a=2 channels, or 32 a=sqrt(2) channels.
Tree feature counts are 40, 360 or 640, plus two when log scale is included.
The matched-capacity a=2 comparisons therefore match EMD in input channels
and component-feature count. Their bandwidth locations and adaptive/fixed
component definitions still differ.

Budgets were fixed after one-split timing pilots on raw and GD60 IRMF, both
log-scale variants. Trees took 9.83 seconds with full shared-runner budgets;
MiniRocket and MultiRocket together took 10.88 seconds; CNN took 16.02 seconds. Pilot test scores did not select folds, methods or budgets.
The stack budgets follow the smaller ESC-50 phase-3 experiment; this study
uses all 15 folds for CNN and MultiRocket as well as the other models.

Exactly three queues run concurrently, each with one CPU thread; the CNN uses
MPS and no data-loader workers. Environment variables also cap BLAS/OpenMP
and Numba at one. Queues process `bird_drone` then `target3` sequentially.
Other agents share the machine and MPS, so wall times are not isolated speed
benchmarks. No decomposition, feature cache or split is regenerated.

## Commands

```bash
ROOT=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594
cd "$ROOT"
LOCAL="$ROOT/experiments/classification-emd-irmf/datasets/uas_birds_60ghz"
# Launch these three queues concurrently, each exactly once.
"$LOCAL/run_phase3_queue.sh" trees &
TREE_PHASE3_PID=$!
"$LOCAL/run_phase3_queue.sh" rocket &
ROCKET_PHASE3_PID=$!
"$LOCAL/run_phase3_queue.sh" cnn &
CNN_PHASE3_PID=$!
wait "$TREE_PHASE3_PID" "$ROCKET_PHASE3_PID" "$CNN_PHASE3_PID"
# After all three queues finish:
"$ROOT/.venv/bin/python" "$LOCAL/analyse_phase3.py"
```

The queue script contains every effective CLI flag and uses absolute paths.
Per-fit logs are under `ml/logs/`; CSV checkpoints and run JSON files live in
each runner directory. The local wrapper copies and adapts the ESC-50 wrapper.
It exports native probabilities and checkpoints without changing shared model
training, train-only preprocessing, grouped validation or refitting.
`ml/`, `emdirmf/`, `PROTOCOL.md` and other datasets are unchanged.

SGD class probabilities can saturate into ties. The exporter and audit require
the native predicted class to attain the maximum probability, allowing a tied
maximum. They preserve native labels and probabilities. A different first
argmax among tied maxima is counted in `validation.json`.

## Inference and errors

The primary frame table contains all ten pairs among the five main methods,
both feature sets and six models. Holm covers 120 tests per task and metric.
The sensitivity table contains seven prespecified pairs across the same
models and feature sets, giving 84 tests per task and metric. Those pairs
compare each a=2 method with EMD, the two a=2 methods with each other, each
with its a=sqrt(2) counterpart, and H=1/H=4 with converged H=2.
Nadeau-Bengio is primary; Wilcoxon is secondary. Intervals are individual
corrected 95% intervals, without simultaneous multiplicity coverage.

The combined analysis uses observed pose-group counts, 35-37 training and
8-10 test groups, for the correction, as in the phase-2 radar analysis. The shared runner's own
comparison CSVs retain its frame-count ratios; headline verdicts use
`paired_all_metrics.csv` and `paired_requested.csv` from this local analysis.
Every method/model uses the same splits and test population. No model is
selected by test scores for confirmatory inference.

Each recording contributes 16 held-out cell predictions. Mean native class
probabilities and majority voting each produce one recording prediction;
exact ties choose the alphabetically first label without consulting truth.
Recording scores pool all 49 recordings within each seed, then average the
three seeds. Recording inferential tests use 15 fold scores with pose-group
correction and separate 120/84-test Holm families at each aggregation level.
A pose with several recordings remains one group. There are 45 pose groups,
including only nine P3P groups, and one indoor campaign.

`per_class_errors.csv` and `confusion.csv` count repeated out-of-fold prediction
occurrences. Each cell is evaluated once per seed, so the denominator is
2,352 cells or 147 recordings per method/model/feature set/task. These are
correlated evaluations, not additional independent samples.
`frame_errors_by_recording.csv`, `recording_errors.csv`, `group_errors.csv` and
`error_counts.csv` retain both repeated-error totals and the count of distinct
recordings/groups ever wrong across all three seeds. Frame and recording
counts answer different questions and are reported separately.

## Power calculation

For each paired BA contrast, the corrected standard error is
`sqrt((1/15 + mean(n_test_groups/n_train_groups)) * var(fold_differences, ddof=1))`,
with 14 degrees of freedom.
The minimum observed difference needed for nominal significance is the
appropriate two-sided t critical value times this standard error. The
80% detectable difference solves the two-sided noncentral-t power equation,
using noncentrality `delta / corrected_se` and the observed paired variance.
Conditional-normal integration over the chi-square denominator avoids tiny-tail
NaNs in this SciPy version's noncentral-t CDF. The integration verifies nominal
alpha at zero noncentrality and 80% power at the solved value.
`power.csv` reports nominal alpha .05 and conservative Holm first-step alpha
`.05 / table_size`. The latter guarantees the first step, rather than predicting
the data-dependent Holm rank. Zero observed variance has no estimable planning
MDE and is marked undefined. These plug-in sensitivity estimates are not
observed power, proof of equivalence, or a new-session sample-size forecast.

Small score/test/error/cost tables stay in the repository. All frame-level
probabilities and recording prediction rows remain under the external
`DATA_ROOT/uas_birds_60ghz/results/uas_phase3_*` paths. Cached decomposition CPU
seconds/frame exclude model fitting. ROCKET timing rows share transforms, so
summing both feature-set times double-counts part of the work.
No git commit is made.

Rendering after analysis:

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python \
 /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/render_phase3.py
```

The observed mean pose-group test/train ratios are 0.250386 for `bird_drone`
and 0.250515 for `target3`. Their corrected standard errors are respectively
2.18078 and 2.18122 times the naive paired-fold standard error. Some folds
hold out eight or ten pose groups because repeated recordings share a pose;
there is no assumption that each fold contains exactly nine groups.

## Completed audit

All 3,240 evaluation fits and 508,032 out-of-fold predictions passed.
The target3 ROCKET exports contain 2,461 labels that differ from the first
probability argmax while attaining a maximal probability. Native labels and
probabilities are preserved. Across 31,752 recording decisions there are
seven exact mean-probability ties and 40 majority ties; alphabetical tie
breaks apply. `completion.json` records elapsed time since the timing pilot.
