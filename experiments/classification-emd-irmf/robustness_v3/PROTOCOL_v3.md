# Notebook-scale contamination, schedule stopping at 31 samples, best K

Protocol dated 2026-10-22, written before any result of this experiment existed. It
follows the Laplace experiment ([PROTOCOL_laplace.md](../robustness_laplace/PROTOCOL_laplace.md))
and changes only what is listed here. Its SHA-256 goes into every run manifest.

## Why

A notebook audit (/tmp/notebook_check/REPORT.md) reproduced the earlier notebooks'
component-error gains for IRMF with our solver. It also found that our classification
pipeline differs from the notebooks in three ways that matter:

1. the IMF schedule continued to 5-sample windows, where IRMF's errors are larger;
2. contamination was scaled to the signal (b = 1 robust signal SD), not to the
   noise level that sets H, so it sat below the robust threshold (b/H about 0.4 to 1);
3. each method used a fixed number of components, although the component-count
   experiment found EMD best with 3 or 4 IMFs.

The user asked to stop the schedule near 31 samples, to scale the contamination as in
the notebooks, and to compare each method at its best number of components, stating K.

## Contamination

Conditions are clean and `nb3_p` for p = 0.1, 0.2, 0.3 and 0.4:

    x + (u < p) * b * e * s,   b = 3 * sigma_noise(x_clean).

sigma_noise is `emdirmf.decompose.noise_sigma_hat` applied to the clean raw channel,
median(|diff(x)|) / (0.6745 sqrt 2), the scale that H = 2 sigma uses. So b = 1.5 H on
clean data, the notebooks' usual relation of contamination to noise (residual-distribution
notebook 3 sigma, real-vs-calculated slides about 3.3 sigma, limits notebooks 2 sigma).
u, e, s are the earlier runs' common random numbers (`robustness.contamination.draws`),
so masks are nested across p and match the Laplace run's positions. There is no Gaussian
term. b is defined with this estimator whatever the H decision below. A channel with
sigma_noise = 0 receives no noise. Injection precedes the shared median/MAD normalization.

## Decompositions and K

Same normalization as before. Window radius r = h L, window 2r + 1 samples.

- Linear IMF and IRMF: K stages from h1 = 0.25 (window 1025 at L = 2048, 513 at
  L = 1024) down to radius 15 (window 31), with ratio a_K = (0.25 L / 15)^(1/(K - 1))
  snapped as in `components.construction.native_ratio`, endpoint h_min = 15 / L. Content
  finer than the last window stays in the residual slot. K grid: 3, 4, 6, 8, 11 at
  L = 2048 and 3, 4, 6, 8, 9 at L = 1024 (the largest K has a close to sqrt 2).
  The resulting pimf window sizes, checked before any run:

  | L | K | a | Windows |
  |---:|---:|---:|---|
  | 2048 | 3 | 5.8424 | 1025, 177, 31 |
  | 2048 | 4 | 3.2438 | 1025, 317, 99, 31 |
  | 2048 | 6 | 2.0260 | 1025, 507, 251, 125, 63, 31 |
  | 2048 | 8 | 1.6559 | 1025, 621, 375, 227, 139, 85, 51, 31 |
  | 2048 | 11 | 1.4234 | 1025, 721, 507, 357, 251, 177, 125, 89, 63, 45, 31 |
  | 1024 | 3 | 4.1312 | 513, 125, 31 |
  | 1024 | 4 | 2.5746 | 513, 201, 79, 31 |
  | 1024 | 6 | 1.7637 | 513, 293, 167, 95, 55, 31 |
  | 1024 | 8 | 1.4998 | 513, 343, 229, 153, 103, 69, 47, 31 |
  | 1024 | 9 | 1.4257 | 513, 361, 253, 179, 125, 89, 63, 45, 31 |
- IRMF: converged safeguarded Newton roots (tolerance 1e-12, cap 100, cap-1000 retry),
  H = 2 sigma_hat recomputed on the input being decomposed. sigma_hat is the current
  median-of-differences estimator unless the pre-specified pilot below adopts the IQR
  estimator.
- EMD: one PyEMD run per channel with the study's settings (8 IMFs plus residue). K grid
  2, 3, 4, 6, 8: keep the first K IMFs and add the rest to the residue, as in the
  component-count experiment.
- Raw: unchanged, one variant.

Features: the shared 20 features per slot from float32 stacks, comp only. log_scale
variants are not run.

## The IQR pilot, decided before any classification

/tmp/sigma_pilot/REPORT.md compares the current sigma_hat (E1) with
IQR(diff(y)) / (1.349 sqrt 2) (E2) on the 100 excluded pilot frames per dataset. E2
replaces E1 for IRMF's H only if, at c = 3, it has lower median per-slot relative L2
against the clean decomposition in at least 9 of the 12 real-dataset x p cells and is not
worse on scale stability in a majority of them. The decision and the pilot report's
SHA-256 are recorded in the manifest before contaminated features are computed.

## Classification

Unchanged splits (5 grouped folds x 3 seeds), the same 100 excluded pilot frames, the
fixed classical family per dataset with the full C grid, and LightGBM with the shared
settings. Shift fits train on clean data for every method and K and predict every
condition. Matched fits train and test at p = 0.4, fixed classical model, at each
method's selected K only.

Clean classical models cannot reuse earlier C choices, because every feature table is new.

## Best K

For each dataset, method and classifier, K* is the K with the highest mean clean
balanced accuracy over the 15 splits (ties go to the smaller K). The same K* is used at
every noise level, and every table and figure states it. Selecting on clean outer-fold
scores makes clean scores slightly optimistic for every method alike; the report says so.
Balanced accuracy for every K is also reported, together with the best K at each noise
level, descriptively.

## Tests

The same estimand, at K*:
DiD(p) = [BA_IRMF*(p) - BA_linear*(p)] - [BA_IRMF*(clean) - BA_linear*(clean)].
Two-sided Nadeau-Bengio corrected t-tests, Wilcoxon secondary, Holm within families:

- Primary, 12: IRMF* minus linear IMF*, shift, fixed classical model, four p, three
  real datasets.
- S1, 12: IRMF* minus EMD*, same.
- S3, 24: LightGBM shift, each method at its LightGBM K*, both comparators.
- S5, 6: matched at p = 0.4, fixed classical model, both comparators.

Synthetic is one separate 18-test family. Reading: IRMF is more robust on a dataset if
the 40% primary DiD is positive with Holm p < .05; a negative one with Holm p < .05
favors linear IMF; otherwise the test does not reject, which is not equivalence. The
report also gives the absolute best-vs-best balanced accuracy at each noise level.

## Outputs

Code only in `robustness_v3/`. External data in each dataset's `robustness_v3/`
directory, report in `output/classification-emd-irmf/robustness_v3/`. No existing cache,
checkpoint or report is modified. pimf 0.4.0 source hashes must still match.

## Amendment, 2026-10-22, before any contaminated feature or classification result

1. The IQR pilot (/tmp/sigma_pilot/REPORT.md, copied to provenance/) kept E1. At c = 3
   E2 had lower component error in 8 of 12 real cells and no worse stability in 4, below
   the 9 and 7 required; every E2/E1 error ratio was between 0.996 and 1.005.
2. The same pilot found that a fixed H per dataset, estimated from clean data, lowered
   component error in all 12 real cells at every c (2 to 15% at c = 3). At the user's
   choice, IRMF now uses a fixed H per dataset instead of a per-input estimate:
   H_d = 2 x the median of E1 (`noise_sigma_hat`) over every channel of the dataset's 100
   excluded pilot frames, each normalized clean with `normalize_frame`. H_d is in
   normalized units and is used unchanged for every frame, for clean and contaminated
   inputs and for every K. The pilot frames enter no classification, so no training or
   test frame sets H. The contamination scale b keeps its per-channel definition above.
   Diagnostics still record the per-input E1 value and its ratio to H_d.
3. The clean raw-feature check failed on 7 of 200 frames because the draft computed raw
   features from float32 samples. Raw features use the shared float64 normalized samples,
   as in the impulse and Laplace runs ("raw: unchanged"). The float32 stacks remain for
   EMD, linear IMF and IRMF.
