# Impulsive contamination and classification robustness

Pre-registration dated 2026-10-21. This file supplements, and does not edit,
[the binding study protocol](../PROTOCOL.md). No contaminated-condition
classification score has been computed before this protocol's commit.

## Question and estimand

Does IRMF's robust M-estimation improve classification when prepared real frames
receive additive impulses? The primary estimand on each paired split is

DiD(c) = [BA_IRMF(c) - BA_linear(c)] - [BA_IRMF(clean) - BA_linear(clean)].

All tests are two-sided. The confirmatory family has exactly 18 BA tests: three
real datasets, two impulse types, three p values, shift, fixed classical model,
comp. Holm corrects all 18 together. Use the unchanged
`emdirmf.stats.corrected_resampled_ttest`, `wilcoxon_signed_rank`, and `holm`.
The Nadeau-Bengio test is primary; Wilcoxon is secondary. Report marginal 95%
corrected CIs and the actual mean test/train frame-count ratio after exclusions.
Repeated folds are not independent observations.

On a dataset robustness helps under the pre-specified rule if DiD at p=.04 is
positive and primary-family Holm p < .05 for at least one impulse type, and its
matched-power Gaussian DiD is smaller. This rule tests relative robustness,
not absolute IRMF superiority or an accuracy gain over clean signals.

## Populations and splits

| Dataset, RNG index | Task | Prepared N | C | L | Fixed classical family |
|---|---|---:|---:|---:|---|
| synthetic, 0 | signal | 1500 | 1 | 1024 | logistic regression |
| dronerf, 1 | 10class | 12000 | 2 | 2048 | linear SVM |
| birdclef2023, 2 | species | 12000 | 1 | 2048 | logistic regression |
| esc50_animals, 3 | animal | 5100 | 1 | 2048 | linear SVM |

Read existing `splits_<task>.json` with `load_splits`. Keep their five grouped
folds and seeds 0, 1, 2. Never regenerate splits. Synthetic runs first end to end
and contributes to no inferential family. Exclude 60 GHz radar entirely.

Select 100 pilot/figure frame IDs per dataset with
`default_rng(SeedSequence([20261021, dataset_index, 991])).choice(N, 100,
replace=False)`, sorted. Pilot frames are excluded from both fitting and testing
for all conditions, including clean. Thus the real analysis populations are
11900, 11900 and 5000, unless solver failures further reduce them. Preserve the
stored group partitions. The fixed pilot subset supplies the only retained
contaminated component stacks. Clean-feature verification uses 50 evenly spaced
frame IDs and is descriptive.

## Inputs and contamination

Inputs are `raw`, `emd`, `imf_linear_a1.41`, `irmf_h2_a1.41_conv`. Use the shared
normalization, schedule, storage slot order and features unchanged. EMD retains
eight IMFs and a residue. IRMF uses safeguarded Newton roots, tolerance 1e-12,
cap 100. Never run contaminated GD60. H = 2 sigma_hat is calculated from the
contaminated normalized channel actually seen by the method. Record the median
and p95 channel-level H_contaminated/H_clean by condition.

Apply contamination to the raw prepared channel in float64 before normalization.
For each dataset/frame/channel, draw u, e, s, z in that order from
`default_rng(SeedSequence([20261021, dataset_index, frame_id, channel]))`:
uniform(0,1), exponential(1), choice([-1,1]), normal(0,1), each length L.
Draws do not depend on condition, worker, scheduling or method. Sigma is
1.4826 MAD of the clean raw channel, with std fallback when MAD is zero. A
constant channel has sigma zero and receives no injected noise.

A=10; p is .002, .01 or .04. Conditions are clean and nine transforms:

- `impulse1_p`: x + (u < p) A sigma e.
- `impulse2_p`: x + (u < p) A sigma e s.
- `gauss_p`: x + A sigma sqrt(2p) z.

Impulse masks are nested, and one/two-sided impulses share locations and
magnitudes. The Gaussian matches expected squared-noise power since E[e²]=2.
This uses the additive exponential magnitude model in `docs/imf_core.py`, with
the independently specified seed/draw order above. Tests cover determinism under
worker/order changes, nesting, positions, fallback and expected power using
sampling-error bounds.

If any channel reaches the Newton cap, repeat that channel from its original
normalized input at cap 1000. Count channel retries and frame retries. Accept
only converged finite fits. Any final solver or feature failure excludes that
frame from every method for the condition. Save failed IDs and error text.
For comparisons against clean, restrict clean and contaminated predictions to
the same condition-valid held-out IDs. Shift models train once on the clean
valid training population. For impulse-minus-Gaussian contrasts use the common
valid IDs of both conditions and clean, rescoring retained predictions if needed.
If failures make a fold or inner grouped validation lose classes, stop rather
than silently change the task.

## Features and classification

The 20 shared component features are computed from float32 component stacks,
converted back to float64 by `frame_features`, exactly as existing caches do.
Only feature values are stored as float32; log_scale retains float64 precision.
Verify a fresh clean decomposition/features pipeline against every input's
cached features on 50 frames per dataset. Require exact float32 equality or at
most float32 rounding error. If verification fails, recompute clean features
through the same pipeline under robustness, document it, and never overwrite the
old caches. Cached clean log_scale must also agree.

Run comp and comp+log_scale. All imputation, variance filtering, scaling and
model selection stay inside training folds. Import `classifiers()` from
`baselines/run_classical.py` without mutation. SVM is LinearSVC(dual='auto',
max_iter=5000), grouped inner three-fold C in [.001,.01,.1,1]. Logistic regression
uses max_iter=3000 and C in [.001,.01,.1,1,10]. These are the current shared
runner's exact grids; the historical DroneRF 10-class budget used fixed C=.1.
The new experiment uses the requested shared definitions, not that historical
budget restriction. Fixed means fixed classifier family, not a C selected on
outer test scores.

Import `ml.run_trees.fit_tree` for LightGBM, workers=1, quick=False. It uses
500 maximum rounds, learning_rate=.05, 15 leaves, min_child_samples=10,
reg_lambda=1, deterministic=True and force_col_wise=True. Weighted inner grouped
validation chooses rounds by log loss, with 12-round early stopping. Refit the
imputer and balanced class weights on the full outer training fold. Seed is
100*split.seed+split.fold. No score-dependent budget selection.

Primary shift trains once per split, method, feature set and classifier on clean
features and predicts every condition from that model. Secondary matched trains
and tests at the same condition. Initially matched classical covers all nine
conditions and matched LightGBM the three p=.04 conditions, subject only to the
pre-specified timing cuts below. Save BA, macro-F1, accuracy, training/test sizes,
selected parameters, timing and numeric/string predictions without pickle.

## Exploratory families and descriptive measures

All secondary tests are explicitly exploratory. Each numbered family has its own
Holm correction, separately for corrected-t and Wilcoxon. BA is the only tested
metric; macro-F1 and accuracy remain saved descriptive metrics.

- S1, 18 tests: primary impulse DiDs, IRMF minus EMD.
- S2, 27 tests: classical comp shift IRMF minus linear IMF, nine Gaussian DiDs
  and eighteen impulse-minus-Gaussian DiDs at matching p.
- S3, 54 tests: LightGBM comp shift, all nine conditions, both IRMF comparators.
- S4, 108 tests: comp+log_scale shift, both classifiers, all nine conditions,
  both IRMF comparators.
- S5: matched, both feature sets and both comparators, relative to the
  corresponding clean-trained/clean-tested baseline. Under the pilot cuts this
  is 36 tests, three datasets x three types at p=.04 x two feature sets x two
  comparators, classical only. Without cuts the initial family is 144 tests.

Clean predictions are paired on split_id and test-frame IDs. Do not substitute
unpaired differences or tests of contaminated method gaps for the DiD.

For all frames, compute mean absolute standardized feature change against clean,
using the clean all-frame mean and population SD per feature; use SD=1 where
SD=0. Save separate comp and comp+log_scale drift. These all-frame statistics
are descriptive and never enter classification. For EMD and both IMF methods,
compute per-slot and whole-stack relative L2 = norm(contaminated-clean) /
max(norm(clean),1e-12). Compute within channels then summarize channel values.
Report medians and p95 by dataset/method/condition/slot; EMD slots can mode-mix.
Save all input BA degradation curves descriptively.

## Files, resumption and resource limits

Code lives only in `robustness/`. Shared code and installed pimf stay unchanged.
Python is the root `.venv/bin/python`, from this study directory.
DATA_ROOT is `/Users/mikhail/Datasets/imf-classification`, via IMF_CLS_DATA.
Never inspect, execute or modify raw directories. No downloads or network.
Always use `np.load(..., allow_pickle=False)`. No pickled model checkpoints.

External output layout under each dataset's `robustness/`:

- `pilot/costs.csv`, `pilot/run.json`: excluded pilot timings and IDs.
- `features/<condition>/<method>/part_<start>.npz`: completed numeric feature,
  stack-change and diagnostic chunks; completion uses exclusive creation or
  atomic new-file publication. Resume completed chunks without overwriting.
- `figures/<condition>/<method>/`: stacks of only the fixed 100 IDs.
- `clean_verification.json`, `manifest.json`: cache verification, provenance.
- `classification/<scenario>/<classifier>/<method>/<feature_set>/split_<id>*/`:
  one completed fit checkpoint with predictions, metrics, timing and parameters.
- `runs/`: execution costs and dependency checks.

Repository outputs are new `output/classification-emd-irmf/robustness/REPORT.md`,
CSVs and PNG/PDF figures. Preserve all existing reports, caches and results.
Resumption checks manifests against parameters and input/source fingerprints;
incompatible existing outputs cause an error, never a silent overwrite.

At most six worker processes, one native thread each using
`emdirmf.parallel.limit_threads`. Decompose and extract features within each
worker, discarding full stacks except the 100 fixed frames. Check total newly
written external data against 60 GB; estimated retained subsets plus features,
predictions and checkpoint metrics are below 3 GB. No full contaminated stacks.

Record installed pimf version and SHA-256 of every top-level pimf Python file at
start and before final analysis. Verify at every runner entry as well. Stop on
any change. Starting hashes are committed in `provenance/pimf_start.json`.

## Pilot and run notes before classification

Timing pilot completed on 100 frames per dataset at impulse2_.04 and gauss_.04,
with six single-thread workers. No pilot classification was performed. Zero cap
hits and zero failed channels. Per-frame CPU seconds averaged across those two
conditions are given in [pilot_costs.csv](provenance/pilot_costs.csv):

| Dataset | EMD | Linear IMF | Converged IRMF |
|---|---:|---:|---:|
| Synthetic | .02849 | .00179 | .01520 |
| DroneRF | .11591 | .01169 | .11016 |
| BirdCLEF | .07419 | .00594 | .05907 |
| ESC-50 | .10999 | .00597 | .06173 |

The decomposition-only extrapolation is 49,566 CPU seconds, or 2.29 hours on six
workers, including synthetic. It excludes full feature extraction. Each condition
is projected from the average of the two measured high-contamination conditions.
Fit costs use historical clean fixed-family/LightGBM per-fold timing tables,
with a 13-fit workload multiplier for DroneRF's change from one fixed-C fit to
four candidates x three inner fits plus refit. These are conservative worker-time
proxies, not measured CPU. Full-grid C costs can differ substantially.

[projection.json](provenance/projection.json) records estimated CPU/worker seconds
and six-worker wall seconds: initial 42.51 h; matched LightGBM seed 0 only 42.15 h;
drop matched LightGBM 41.97 h; matched classical only p=.04 18.27 h. Apply every
permitted cut in order. The remaining projection still exceeds 12 h; the design
permits no further cuts, so keep every shift condition and the entire primary
family. The 12-hour trigger is used for the specified cuts, not to invent an
additional cut to confirmatory work. Record actual costs separately and do not
represent the planning proxy as an observed CPU total.

Tests passed before pre-registration: 108 existing and 4 new tests, 112 total.
Every later commit requires the existing suite plus robustness tests. Commits
are local, with explicit paths and the required Co-Authored-By trailer.
