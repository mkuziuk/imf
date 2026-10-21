# Component-count experiment

This protocol is committed before classification of any new input. The parent
study's PROTOCOL.md remains binding. Clean prepared frames are the entire
population. The question is how the number and construction of components affect
classification, and whether the IMF advantage over EMD survives matched counts.

## Population and paired evaluation

| Dataset | Task | N | Channels | L | Fixed classical model |
|---|---|---:|---:|---:|---|
| synthetic | signal | 1500 | 1 | 1024 | logistic regression |
| dronerf | 10class | 12000 | 2 | 2048 | linear SVM |
| birdclef2023 | species | 12000 | 1 | 2048 | logistic regression |
| esc50_animals | animal | 5100 | 1 | 2048 | linear SVM |

Use the existing splits_<task>.json, all five grouped folds for seeds 0, 1, 2.
Never recreate splits. Every comparison pairs identical split IDs and frames.
There is no contamination, selection by labels, or radar dataset in this study.
Timing-pilot frames stay in the analysis.

## Inputs and tags

K means extracted components; a residual adds one slot. Storage is fine to coarse
among extracted slots, followed by the solver residual. For IMF this residual is
the finest content. Use emdirmf.decompose.scale_order for cache derivations and
scale ranks. Components has its own parser and never extends the shared parser.

1. Native cascades use `linear_native_k{K}` and `irmf_native_k{K}` for
   K = 3, 4, 5, 6, 9, 13. Both use h1=.25, h_min=1.5/L, k_max=None and
   a_K = (.25 L / 1.5) ** (1 / (K-1)). This spans the same allowed scale range
   with wider spacing. Round the ratio upward by the minimum number of float64
   ULPs needed for pimf's last-stage snap at rtol=1e-14. Assert exactly K stages
   and last radius 1.5 on every channel. The study default uses a=sqrt(2),
   15 stages at L=1024 or 17 at L=2048, and last radius 2. Thus the endpoint
   differs slightly from the default and is not silently presented as identical.
   Linear uses pimf.IMF(Quadratic()). IRMF uses the unchanged NewtonSmoothAbs
   lookup-score roots, root_tol=1e-12, H=2*sigma_hat and the shared noise fallbacks.
   Start each channel at root_max_iter=100. Any window reaching 100 iterations,
   including one converging on the last iteration, triggers a full-channel rerun
   with cap=1000. Record every hit, attempted diagnostic, and failure. Never
   accept an unconverged fit or proceed after an unresolved failure.
2. EMD uses `emd_k{K}` for K=3,4,5,6,7. `emd` is K=8, the study default;
   `emd_k8` is an accepted alias for construction tests only, without duplicate
   classification. Fold cached slots K through 8 into a single residue in
   float64, then cast to float32. Keep the first K default PyEMD IMFs and their
   existing zero padding. Verify all K=3..8 on 50 frames/dataset against direct
   PyEMD default sifting. Report `n_imfs < K` as padding at the Kth extracted slot.
3. Default-schedule truncation uses `linear_trunc_k{K}` and `irmf_trunc_k{K}`
   for k_max=4,8,12, derived from the existing full cascades. Retain the coarsest
   K extracted slots in their storage order. Sum every finer stage and the
   original solver residual in float64 into the final slot, then cast to float32.
   Verify against direct pimf with k_max=K on 50 frames/dataset. The full-pimf
   default comparison `gd60_trunc_k8` uses the same derivation from the GD60
   `irmf_h2_a1.41` cache; verify it directly too.

Default controls are `raw`, `emd`, `imf_linear_a1.41`,
`irmf_h2_a1.41_conv` and `irmf_h2_a1.41`. Reuse their existing feature tables,
but recompute their classification scores. For the four phase-2 controls with
stored scores, compare every recomputed split/feature-set metric with the stored
phase-2 table before using results. Audit discrepancies rather than silently
replacing either source.

## Features and classifiers

Use emdirmf.features.frame_features unchanged on float32 component stacks.
There are 20*C*(K+1) component features and C additional log_scale features.
Raw has 20*C features plus the optional C. Report the actual counts. Relative
slot energy follows the shared energy_frac definition, slot power divided by
the sum of slot powers in that frame-channel. Save its median and p95 per slot.

Use both comp and comp+log_scale. All imputation, removal of constant features,
scaling and model selection fit inside training folds. Reuse the estimators in
baselines/run_classical.py. Logistic regression tunes C=.001,.01,.1,1,10 with
grouped inner 3-fold CV, seed 0, max_iter=3000. ESC-50 SVM uses the shared grid
C=.001,.01,.1,1 and max_iter=5000. DroneRF uses the historical primary-task
fixed C=.1 pipeline, as documented in its primary_svm_grid.json. The fixed
single candidate needs no inner search and applies to every input.

LightGBM reuses ml/run_trees.fit_tree, with workers=1 per fit and quick=False.
It uses 500 candidate rounds, learning_rate=.05, num_leaves=15,
min_child_samples=10, reg_lambda=1, balanced training/validation weights,
deterministic=True and force_col_wise=True. The shared grouped inner holdout
selects rounds by validation log loss, patience 12, and refits on outer training
rows. Use the shared seed rule, seed*100+fold. Classical arrays are float64;
LightGBM arrays are float32, exactly as in their respective shared runners.

## Outcomes and inferential families

Save balanced accuracy, macro-F1, accuracy and per-class recall for each split.
BA is the primary endpoint. Use emdirmf.stats.paired_comparison and holm for
two-sided Nadeau-Bengio corrected resampled t tests, corrected marginal 95%
intervals and secondary Wilcoxon tests. Holm-adjust the two p-value types
separately. Never treat overlapping folds as independent observations.

- Family A is the fixed classical model with comp. For each dataset and
  K=3,4,5,6, compare native IRMF_K minus EMD_K, native linear_K minus EMD_K,
  and native IRMF_K minus native linear_K. Correct all 48 BA tests together.
  The advantage survives matched counts on a dataset only if both IMF-vs-EMD
  contrasts are positive and Holm-significant at both K=5 and K=6.
- Family B compares each native linear/IRMF K=3,4,5,6,9,13 with its converged
  study default, and each EMD K=3,4,5,6,7 with EMD K=8. Correct the 17 BA tests
  within each dataset. Use fixed classical/comp for inference. Plot all three
  methods' mean BA against K separately for each dataset and both classifiers.
- Family C compares both converged IMF families' k_max=4,8,12 against the full
  default, plus GD60/k_max=8 versus GD60/full and versus EMD/K=8. Correct the
  eight BA tests within each dataset.
- Exploratory families A and C are repeated independently for classical with
  comp+log_scale, LightGBM with comp, and LightGBM with comp+log_scale. Family A
  correction spans 48 tests in each variant; family C correction spans eight
  tests/dataset in each variant. No pooling of these exploratory p-values with
  the primary family. Macro-F1 and accuracy use separate matching families and
  are descriptive supporting endpoints. Report LightGBM count curves without
  assigning them the primary family B interpretation.

Best K is the largest observed mean BA among that method's native count grid and
study default, reported descriptively with no claim that it is an unbiased tuned
performance estimate. In a tie choose the smaller K. Truncations are reported
separately and do not enter that best-K selection.

## Integrity, storage and compute

Use only prepared/, decomp/, features/ and existing splits JSONs as data sources.
Never access raw/ or the concurrent robustness experiment's directories. Use
the checkout's .venv/bin/python and explicit allow_pickle=False on NumPy reads.
No download or network request is part of the experiment. pimf remains 0.4.0,
unmodified. pimf_baseline.json records every installed pimf/*.py SHA-256 at the
start. Every runner records version and hashes under the dataset's components/
audit/ and checks for equality; check again before final analysis and stop on
any change. Shared study modules remain unchanged.

All new large data, checkpoints, predictions, feature tables and logs go under
DATA_ROOT/<dataset>/components/. Native float32 stacks are retained in block
NPZs if the total new-data allocation stays below 60,000,000,000 bytes. Features
and metrics take priority. Derived inputs need no duplicate full stacks.
At most six worker processes, one native thread each via limit_threads. Datasets
run sequentially; do not overlap six-worker processing and classification pools.

Code layout is construction.py, integrity.py, pilot.py, storage.py,
processing.py, classification.py and analysis.py, with tests/ and this protocol.
External layout is audit/, pilot/, blocks/<tag>/<start>.npz plus JSON diagnostics,
features/<tag>.parquet, fits/<classifier>/<feature_set>/<tag>/<split>.json and
numeric prediction NPZs, and immutable runner manifests. Small result CSVs,
REPORT.md and figures go to output/classification-emd-irmf/components/.
Existing caches and reports are never overwritten. New completed checkpoints
are reused only with matching source/code/split fingerprints.

## Pilot and budget rule

The label-free pilot uses 50 frame indices per dataset chosen without labels by
default_rng(20261021), without replacement, sorted. Every native family/K is
timed on each complete frame including both DroneRF channels. Worker warmup is
excluded from fit CPU time. CPU means project to all prepared frames. Divide
total projected CPU by six for ideal wall time and report measured contention.
Classification projection uses historical fit times, no new classification
pilot, with a conservative allowance for overhead. Record the numbers below
before committing this protocol.

If the projected six-worker wall time exceeds 12 hours, apply cuts in order:
LightGBM seed 0 only, then LightGBM comp only, then LightGBM K=4,6 and defaults
only. Recalculate after each cut. Never cut fixed-classical families A, B or C.
Document every applied cut and every later deviation in REPORT.md.

## Pre-registration run notes

Captured on 2026-10-21 before new-input classification. The existing EMD counts
match the supplied table. The exact ratios used by the implementation are:

| K | a_K at L=1024 | a_K at L=2048 |
|---:|---:|---:|
| 3 | 13.063945294843617 | 18.475208614068027 |
| 4 | 5.5468901948050782 | 6.9886437178903913 |
| 5 | 3.6144080144393795 | 4.298279727294168 |
| 6 | 2.7953084750863173 | 3.2109662470409233 |
| 9 | 1.9011596499082815 | 2.0732292992561554 |
| 13 | 1.5346607722884436 | 1.6259164506005404 |

Pilot mean fit CPU seconds per complete frame, including all channels:

| Dataset | Family | K=3 | K=4 | K=5 | K=6 | K=9 | K=13 |
|---|---|---:|---:|---:|---:|---:|---:|
| synthetic | linear | 0.000751 | 0.000719 | 0.000777 | 0.000838 | 0.001143 | 0.001531 |
| synthetic | irmf | 0.005286 | 0.006132 | 0.006289 | 0.007152 | 0.009812 | 0.013316 |
| dronerf | linear | 0.004412 | 0.004433 | 0.004909 | 0.005650 | 0.007169 | 0.009943 |
| dronerf | irmf | 0.035516 | 0.041312 | 0.045733 | 0.048350 | 0.067627 | 0.088140 |
| birdclef2023 | linear | 0.002447 | 0.002391 | 0.002544 | 0.002840 | 0.003770 | 0.005120 |
| birdclef2023 | irmf | 0.020222 | 0.022286 | 0.023807 | 0.027360 | 0.038984 | 0.049260 |
| esc50_animals | linear | 0.002183 | 0.002458 | 0.002605 | 0.002767 | 0.003696 | 0.005049 |
| esc50_animals | irmf | 0.021693 | 0.023303 | 0.026058 | 0.028397 | 0.037662 | 0.051704 |

The native decomposition projection is 7910.0 CPU seconds, or
0.366 ideal hours on six workers. No label was read by the timing pilot.
The historical classical-time projection assigns each family its existing
per-fit mean, across both feature sets, to every requested tag in that family.
For converged IRMF it conservatively uses historical GD60 feature-fit cost.
Historical LightGBM per-fit means provide the corresponding tree projection.
These are CPU-equivalent estimates from wall timings in single-thread workers;
they include historical machine contention and are not exact new fit benchmarks.

Projected fixed-classical cost is 139108.7 CPU-equivalent seconds;
LightGBM is 31737.2. Adding 20% for feature extraction, I/O, startup and
concurrent load gives a total projected wall time of 9.93 hours.
This is below 12 hours. No budget cuts apply. Retain LightGBM on all 15 splits,
both feature sets and the complete count grid. Native stack allocation is
31.541 GB before small overhead, below the 60 GB cap.

Direct-cache verification covers the same 50 frames/dataset, every channel,
EMD K=3..8, both converged IMF truncations at k=4,8,12, and GD60/k=8.
The acceptance threshold is relative L2 error <2e-7 after float32 storage,
consistent with independent rounding of cached stages and residuals.
The unit tests additionally assert reconstruction and every native endpoint.
