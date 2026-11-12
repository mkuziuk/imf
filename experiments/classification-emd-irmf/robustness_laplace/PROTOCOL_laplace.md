# Laplace contamination and classification robustness

Protocol dated 2026-10-22, written before any Laplace-contaminated feature or
score existed. It reuses the impulse experiment's design
([PROTOCOL_robustness.md](../robustness/PROTOCOL_robustness.md)) and changes only
what is listed here. Its SHA-256 is recorded in every run manifest.

## Question and estimand

Does IRMF keep more of its classification performance than linear IMF when a
growing fraction of samples carries Laplace noise? The estimand on each paired
split is unchanged:

DiD(p) = [BA_IRMF(p) - BA_linear(p)] - [BA_IRMF(clean) - BA_linear(clean)].

## Contamination

At the user's request there is no Gaussian term and no Gaussian control
condition. Conditions are clean (p = 0) and `laplace_p` for p = 0.1, 0.2, 0.3
and 0.4:

    x + (u < p) * A * sigma * e * s,   A = 1.

This is the signed-exponential contamination of `docs/imf_core.py` and the
real-vs-calculated notebooks: with probability p a sample receives Laplace(0, b)
noise, b = A sigma. Those notebooks set the Laplace scale near the clean signal's
standard deviation, so b is one robust clean SD here. sigma is 1.4826 MAD of the
clean raw channel, with std fallback; a constant channel receives no noise.
u ~ U(0, 1), e ~ Exp(1) and s ~ {-1, 1} are the impulse experiment's draws, from
`default_rng(SeedSequence([20261021, dataset_index, frame_id, channel]))` in the
same order. The masks are therefore nested across p, and the noise is common to
all methods. Injection precedes the shared median/MAD normalization.

## Unchanged from the impulse experiment

Datasets synthetic, DroneRF 10class, BirdCLEF 2023 species and ESC-50 animals;
no 60 GHz radar. Inputs raw, EMD, linear IMF and converged IRMF (safeguarded
Numba Newton roots, tolerance 1e-12, cap 100 with a cap-1000 retry). H is
recomputed from each contaminated channel. The stored grouped five-fold,
three-seed splits, the 100 excluded pilot frames per dataset, the shared 20
features from float32 stacks, comp and comp+log_scale, the classical family per
dataset (linear SVM for DroneRF and ESC-50, logistic regression for BirdCLEF and
synthetic) with grouped inner C selection, and the shared LightGBM settings.
Failure handling, prediction checks and stability measures are the same.

## Speed changes, which do not change any estimand

1. Clean-trained (shift) classical models. The impulse run already selected C
   by grouped inner search on these exact clean training rows. Each shift fit
   refits the pipeline at that C. LinearSVC uses the deterministic primal solver
   here (rows exceed features) and logistic regression uses lbfgs, so the refit
   is the same model. The refit is accepted only if its clean predictions equal
   the stored ones on every test frame. Otherwise the full grid search reruns,
   and the fallback is recorded.
2. LightGBM shift fits rerun in full. Agreement of their clean predictions with
   the impulse run is recorded, not required.
3. Matched classical training covers p = 0.4 only, as the impulse run kept only
   its top level. No matched LightGBM.
4. All four datasets share one 14-worker pool, one native thread per worker.
   Feature chunks run first, then fits in descending expected cost.

Clean-feature verification reruns on 50 evenly spaced frames per dataset and
must pass before any contaminated feature is computed.

## Tests

Two-sided Nadeau-Bengio corrected t-tests, with Wilcoxon as secondary, Holm
within each family, marginal 95% corrected CIs. Real-data families:

- Primary, 12 tests: IRMF minus linear IMF, shift, fixed classical model, comp,
  four p values, three datasets.
- S1, 12 tests: the same contrasts against EMD.
- S3, 24 tests: LightGBM comp shift, both comparators.
- S4, 48 tests: comp+log_scale shift, both classifiers, both comparators.
- S5, 12 tests: matched at p = 0.4, both feature sets, both comparators, relative
  to the clean-trained, clean-tested baseline.

The impulse run's S2 tested Gaussian controls and has no counterpart. Synthetic
forms one separate 36-test Holm family and enters no real-data family.

Pre-specified reading: IRMF is more robust than linear IMF on a dataset if the
p = 0.4 primary DiD is positive with Holm p < .05. A negative DiD with Holm
p < .05 means linear IMF is more robust. The four p values are also read as a
dose-response curve, descriptively.

## Outputs

Code lives only in `robustness_laplace/`. External data go to each dataset's
`robustness_laplace/` directory, with the impulse run's immutable checkpoint
layout. The report goes to `output/classification-emd-irmf/robustness_laplace/`.
No existing cache, checkpoint or report is modified. The pimf 0.4.0 source
hashes from the impulse run must still match at every entry point.
