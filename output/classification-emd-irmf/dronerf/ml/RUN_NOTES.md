# DroneRF phase-3 reproduction

Run from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594` with
`/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python`.
`IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification`.

## Population and model budgets

The 12,000 activity-gated frames, 227 segment groups and saved outer splits
are unchanged. Each frame has 2,048 samples at 40 MS/s, spanning 51.2 microseconds. L and H are
nonsimultaneous views of a segment. Neither coherence nor phase alignment is
assumed. All model-fitted preprocessing and inner budget selection use outer-training data only.
The shared grouped inner holdout selects rounds, leaf size, head regularization
or epochs, followed by a fresh outer-training refit.

The main methods are `raw`, `emd`, `imf_linear_a1.41`, `irmf_h2_a1.41` and
`irmf_h2_a1.41_conv`. The default IRMF uses GD60; the converged tag uses the
cached safeguarded Newton solution at every stage. Matched-capacity checks
add `irmf_h2_a2_conv` and `imf_linear_a2`. There are 1/9/18/10 slots per view
for raw/EMD/default-schedule IMF/a=2 IMF. The a=2 comparison narrows the
component-count difference; it is not an exact nine-slot match.

Timing pilots use primary-task split 0, raw and converged IRMF, both scale
variants, one CPU thread and concurrent runs. Pilot predictions are excluded
from inference. Pilot wall times were 141.7 s for trees, 249.5 s for MiniRocket and 106.8 s for the two-epoch CNN. Their tables, parameters and clocks are in `pilot_*` and
`pilot_costs.json`. Pilot scores do not select methods or folds.

| Family | Folds | Common per-method budget |
|---|---:|---|
| LightGBM | 15 | 60 rounds maximum, 15 leaves, patience 12 |
| XGBoost | 15 | 60 rounds maximum, depth 4, patience 12 |
| ExtraTrees | 15 | 64 trees, grouped-inner leaf grid 1/3 |
| MiniRocket | 10 primary, 5 secondary | 1,008 features, five head passes, 128 bias-fit frames |
| 1-D residual CNN | 15 primary, 5 secondary | width 8, eight epochs maximum, patience 3, MPS |

The tree `--quick` flag changes capacity, while explicit `--folds 15`
retains all three seeds. Secondary stack tasks use all five folds of seed 0.
Primary MiniRocket combines seeds 0 and 1, and the primary CNN combines all
fifteen folds from three seeds. CNN uses
the same anti-aliased downsampling by two for every input, producing 1,024
samples per stack channel. MiniRocket retains all 2,048 samples and the same
total transformed feature budget regardless of stack width. These are
restricted-budget comparisons rather than accuracy ceilings.

Every primary-task LightGBM and XGBoost fit in the seven-method comparison
selected the 60-round cap (210/210 for each family). The primary CNN selected
eight epochs in 88/210 fits; its median selected epoch was seven. Increasing
cross-validation coverage does not remove these capacity limits.

An initial twelve-epoch CNN launch was stopped before its first evaluation
fit completed. Extrapolating the measured two-epoch wide-stack fit across
seven methods, both scale variants and five folds exceeded the wall budget.
The eight-epoch cap was fixed from timing, with no main CNN scores available.
The abandoned launch is recorded in `interrupted_pilot_extension.json`.

Four main lanes use one CPU thread each. A separate L-only job occupies the fourth CPU-model slot while the CNN uses MPS. The initial allocation is four CPU model slots plus one DroneRF MPS training job; later expansion is recorded below.
Folds are serial and data loaders have no child workers. Other agents share
the machine and MPS. Secondary 4-class and 2-class jobs run only when the
previous measured task suggests they fit before the cutoff. Queue JSON files
record commands, timestamps, failures and skipped jobs. Secondary MiniRocket
uses four threads for 4class and six for 2class, while secondary XGBoost uses
two. The full process inventory allowed expansion; the six-thread job starts
after XGBoost has completed. `worker_plan.json` and effective run JSONs retain
the allocation; it changes hardware allocation only.

## Local extensions and outputs

`datasets/dronerf/run_phase3.py` copies the ESC-50 wrapper and invokes the
unchanged shared runners. It captures native probabilities and records one
parquet part per evaluation fit outside the repository. Small fold checkpoints
are saved after every fit. Trees export their actual feature counts.
`--l-only` keeps the L-view features and the same prepared population, including
frames accepted by H-only activity. The LightGBM view ablation uses the same
capacity, folds and both log-scale variants as its two-view comparison.

Complete out-of-fold predictions are under
`DATA_ROOT/dronerf/results/dronerf_phase3_*_predictions.parquet`.
The similarly named `_parts` directories hold per-fit checkpoints. No
decomposition, feature table, prepared data or shared split is overwritten.
The large component stacks are read through bounded memmap batches.

SGD may return probability ties after one-vs-rest logistic saturation.
Exports verify that the native predicted label has maximum probability, while
allowing ties. Frame metrics use native predictions; segment metrics use the
argmax of average native probabilities. These definitions can differ in tied
cases, and the audit reports the frame tie count.

`analyse_phase3.py` follows the ESC-50 audit. It checks exact fold membership,
train/test group separation, complete method/model/feature-set coverage,
class presence, probability ranges and sums, and every frame metric and recall
against saved predictions. Shared log-scale values must be identical across
methods, feature caches must have no failed decompositions and every `_conv`
channel must report convergence.

## Inference and aggregation

Balanced accuracy is primary; macro-F1 and accuracy accompany every model
and both feature sets in `scores.csv` and `SCORES_ALL_METRICS.md`.
Paired effects use the Nadeau-Bengio corrected resampled t-test,
`SE=sqrt((1/J + mean(n_test/n_train))*var(d, ddof=1))`, with `J-1` degrees of
freedom. Wilcoxon is secondary. Confidence intervals are individual corrected
95% intervals, not simultaneous intervals.

For each task and metric, the primary Holm family contains all ten pairs of
the five main methods, both feature sets and every completed main model
family. With five families this is 100 tests. The a=2 table has three pairs
among EMD, linear a=2 and converged IRMF a=2, both feature sets and the same
families, giving 30 tests. Optional H sensitivity and L-only tables have
separate declared Holm families. The individual shared-runner comparison
CSVs retain their runner-specific corrections; headline inference uses the
combined tables from this analysis.

For each method, classifier and seed, the analysis averages held-out class
probabilities over all frames in each segment, then pools the 227 segments
before scoring. No segment spans test folds within a seed. These pooled
scores are averaged over three seeds for trees and the primary CNN, two for
primary MiniRocket, and one for secondary stack tasks and the spectrogram reference.
`segment_metrics_by_fold.csv` and `paired_segment.csv` separately provide
fold-level inference, with test/train group counts in the correction.

Selecting a best family from outer-fold means is exploratory. L-only results
and pooled segment scores are descriptive sensitivity checks. Repeated seeds
do not create independent recordings. Each BUI is one session, so segment
grouping does not establish unseen-session drone or flight-mode recognition.
Activity gating, short context, normalized SNR and quantization cues also
limit comparison with published RF/STFT results.

## Commands

The queue script and `jobs.json` retain the exact commands for each job.
To reproduce the fixed budgets from the worktree root:

```bash
study_root=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594
study_python="$study_root/.venv/bin/python"
study_local="$study_root/experiments/classification-emd-irmf/datasets/dronerf"
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
"$study_python" "$study_local/analyse_phase3.py" --snapshot-inputs
"$study_python" "$study_local/run_phase3_queue.py" lgbm
"$study_python" "$study_local/run_phase3_queue.py" other
"$study_python" "$study_local/run_phase3_queue.py" rocket
"$study_python" "$study_local/run_phase3_queue.py" cnn
"$study_python" "$study_local/run_phase3_queue.py" auxiliary
"$study_python" "$study_local/run_phase3_queue.py" cnn_full
"$study_python" "$study_local/run_phase3_queue.py" rocket_full
"$study_python" "$study_local/analyse_phase3.py"
"$study_python" "$study_local/render_phase3.py"
```

Use a future `--deadline` when rerunning the queues. Existing completed runs
are skipped. For an interim audit of completed jobs, pass `--partial` to the
analysis. Cached decomposition seconds per two-view frame are reported
separately from model fitting. MiniRocket's two feature-set timing rows reuse
transforms and must not be summed as independent wall time. Pilot and
abandoned-launch time belongs in total study elapsed time.

The input manifest fingerprints 56 prepared, cached and shared-code files. Refresh it with `--snapshot-inputs` before a new study, then verify it during analysis. A brief extra launch was stopped before any evaluation completed when the full process inventory revealed active background joblib workers; `interrupted_resource_extensions.json` records it. The fourth CPU slot subsequently runs secondary ExtraTrees and then H=1/H=4 LightGBM sequentially.

## Exact reader cache and expanded primary CNN CV

The full process inventory at 23:01:47 UTC found no heavy Python work outside
DroneRF, including no remaining joblib workers. The permitted expanded CPU
allocation was then used for two cache-builder processes and, later, four
MiniRocket threads and two XGBoost threads on secondary tasks. After XGBoost
finished, another process check found no outside heavy Python work and the
2class MiniRocket allocation rose to six threads. The model
capacities, splits and training-only selection rules are unchanged.

`stack_cache.py` writes the shared reader's float32, anti-aliased tensors to
read-only memmaps under `DATA_ROOT/dronerf/results/representations`. It builds
32 frames at a time and checks original single-frame reads at 16 fixed frame
IDs per method. All seven methods passed exact array equality. Building all
seven caches with two processes took 15.1 seconds. Normalization and the FIR
remain deterministic per frame, with no fitted cross-frame statistics.

A full eight-epoch raw/converged-IRMF pilot on split 0 took 52.4 seconds,
versus 289.2 seconds of corresponding original fit/predict work. All 9,740
native labels and probability vectors were bit-identical, and selected epochs
were identical. `cache_pilot_validation.json` and `stack_cache_validation.json`
retain the evidence. The optimization changes data reading only.

That timing justified adding all remaining primary CNN split IDs 5-14 for
every method at the existing eight-epoch cap, followed by both secondary
tasks with the same seven methods and model settings. The extension was
chosen from compute cost and uses the entire remaining shared seed set.
`cnn` contains the original five primary folds; `cnn_extra` contains the
additional ten. `analysis_runner=cnn` in `jobs.json` combines them for all
frame/segment summaries and corrected tests, while preserving separate
physical exports and clocks. The analysis rejects duplicate split records.

The raw log-STFT reference uses full-resolution frames, two separate view
channels, 256-sample Hann windows, hop 64, zero padding, and log1p power. Its
129 x 33 images are cached as memmaps. Three Conv2d/GroupNorm/ReLU/MaxPool2d
blocks of widths 8/16/32 use global mean/max pooling and the shared grouped
selection/refit loop. The timing pilot took 30.2 seconds; the main reference
uses five folds, eight epochs, patience 3 and two CPU threads. This is a
separate architecture and sampling pipeline from the downsampled 1-D CNN,
shown descriptively rather than as a decomposition contrast.

To prepare the exact caches and run the CNN extension, after the original
CNN queue has finished:

```bash
"$study_python" "$study_local/stack_cache.py" --dataset dronerf \
  --methods raw emd imf_linear_a1.41 irmf_h2_a1.41 irmf_h2_a1.41_conv irmf_h2_a2_conv imf_linear_a2 \
  --workers 2 --out "$study_root/output/classification-emd-irmf/dronerf/ml/stack_cache_validation.json"
"$study_python" "$study_local/run_phase3_queue.py" cnn_full
"$study_python" "$study_local/analyse_phase3.py"
"$study_python" "$study_local/render_phase3.py"
```

Exact standalone reference, L-only, cache-builder and cache-pilot commands
are retained in `standalone_commands.json`. Run JSONs also retain effective parameters. Native five-fold runner summaries remain available;
`all_folds.csv`, `scores.csv`, `TABLES.md`, and the central paired tables are
the authoritative combined results. Initial default five-fold stack budgets
were fixed before test scores; the CNN CV extension and hardware reallocation
were timing decisions, with model settings retained.

## Timing-based MiniRocket extension

All original jobs finished by 00:07:58 UTC. A fresh full Python-process
inventory found no outside heavy work, and an eight-thread raw/converged-IRMF
split-0 calibration took 67.6 seconds. Its 9,740 native labels and probability
vectors were bit-identical to the original one-thread pilot, with zero maximum
probability difference. `rocket8_pilot_validation.json` retains its independent
prediction audit. The timing pilot is excluded from inference.

The fixed original five-fold timings were decomposed into shared transform,
prediction transform and head work. The calibration ratios forecast 1,376.5
seconds for a further complete seed across all seven methods. A 15% timing
buffer gives 1,583.0 seconds, versus 1,602.2 seconds available before the
00:37 UTC model cutoff, which reserves three minutes for the final audit.
Two seeds would require 3,166.0 buffered seconds and did not fit. The choice
therefore adds exactly shared split IDs 5-9, at the same 1,008 features,
five head passes and 128 bias-fit frames. Scores and p-values are not stopping
criteria. `rocket_extension_forecast.json` retains the calculation.

`rocket_extra` preserves the seed-1 native outputs separately. Its
`analysis_runner=rocket` setting merges it with original seed 0 in every
combined score and corrected test, with duplicate split/seed checks. The
eight-thread allocation begins only after all other model jobs have finished.
The run's exact command is in `queue_rocket_full.json`; the pilot command is
also in `standalone_commands.json`.

## Files written

The local dataset folder contains six new scripts: `run_phase3.py`,
`run_phase3_queue.py`, `analyse_phase3.py`, `stack_cache.py`,
`run_spectrogram.py` and `render_phase3.py`. Shared ML/decomposition code,
the protocol and other datasets are unchanged.

This output folder contains the concise `REPORT.md`, per-run tables and
configuration JSONs, the combined `TABLES.md`, `SCORES_ALL_METRICS.md`,
`PAIRS.md`, `ABLATIONS.md`, `overview.html`, and CSVs for fold scores,
all-metric paired tests, segment scores, convergence strata, view comparisons,
per-class recalls, component importance and costs. `validation.json`,
the input manifest, timing pilots and exact-cache checks retain validation
evidence. Large prediction parquets and deterministic representation memmaps
live under the external data root. No git commit was made.

## Final validation

All 21 registered jobs completed. The independent full audit passed all 2,740 evaluated model/split fits and 6,576,000 prediction rows, with zero pending jobs. It verified all 56 input/source fingerprints, the shared scale values, convergence flags, split/group membership, native probabilities, per-class recalls and all three reported metrics. The combined report table also matches the audited score CSV cell for cell.

The last model finished at 2026-10-21T00:35:58.932470+00:00; the audit finished at 2026-10-21T00:37:40.621101+00:00. Wall time from the first pilot through the audit was 176.3 minutes, including discarded launches and cache work. Initial document/skill reading precedes that anchor. `study_clock.json` retains the clocks. Native sources and prepared/decomposition/feature caches are unchanged. All six local scripts compile. No commit.
