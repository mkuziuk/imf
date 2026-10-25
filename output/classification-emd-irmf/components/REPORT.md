# Component count and signal classification

The registered matched-count criterion requires both IMF methods to beat EMD at K=5 and K=6 after Holm correction. It is met for synthetic, dronerf. It is not met for birdclef2023, esc50_animals.

## Question and design

Does the IMF advantage over EMD persist when the extracted component count is matched? This experiment compares clean prepared frames on four datasets using the existing 15 grouped splits. Every K has K extracted slots and one residual. Native linear IMF and converged IRMF span radius 0.25 L to 1.5 samples. Linear IMF uses Quadratic; IRMF keeps H=2 sigma_hat and uses the shared NewtonSmoothAbs lookup-score solver at tolerance 1e-12. Any window reaching 100 iterations triggers a full-channel retry at cap 1000; unresolved fits stop the run. EMD uses PyEMD defaults and retains its first K IMFs, with zero padding where needed. A separate construction truncates the default IMF cascade at the coarsest k_max stages.

The pre-registration commit is `0b19623`. The [protocol](../../../experiments/classification-emd-irmf/components/PROTOCOL_components.md) records tags, exact ratios, the 50-frame label-free pilot and the initial 9.93-hour budget projection. An audit of historical tree thread counts revised that projection to 10.53 hours, still below the cut threshold. The study default ends at radius 2 samples and has 15 extracted stages for synthetic, 17 for the other datasets. Native K therefore changes both count and spacing, with a slightly finer endpoint. Matched counts control input feature dimension; they do not make the decompositions' scale bands identical. The shared classical pipelines' training-fold constant-feature filter can leave different effective widths.

Fixed models are logistic regression for synthetic and BirdCLEF, linear SVM for DroneRF and ESC-50. DroneRF retains its historical fixed C=0.1; the others reuse the original grouped training-fold searches. LightGBM reuses the exact shared fit_tree implementation with one thread. Both comp and comp+log_scale are evaluated. Each channel-slot contributes 20 unchanged shared features. [Costs and feature counts](costs.csv) list each input's dimensions, checked against the actual feature width recorded for every fit.

## Cache and baseline checks

The direct checks used 50 frames per dataset and every channel. Across 3250 checks, the largest relative L2 difference from direct decomposition was 3.74e-08, below 2e-7 after float32 storage. All native channels assert their stage count and final radius. The 1710 recomputed historical split/feature-set/metric checks have maximum absolute difference 1.11e-16. [Direct verification](verification.csv), [default score validation](default_validation.csv).

The supplied EMD count table is reproduced below. Percentiles are over channels, including both DroneRF views. [Counts](emd_counts.csv) and [padding by K](padding_fractions.csv) retain full precision.

| dataset       |   n_channels |   fraction_ge9 |   p1 |   p5 |   p50 |   p95 |   p99 |
|:--------------|-------------:|---------------:|-----:|-----:|------:|------:|------:|
| synthetic     |         1500 |     0.00333333 |    6 |    6 |     7 |     8 |     8 |
| dronerf       |        24000 |     0.432792   |    7 |    8 |     8 |     9 |     9 |
| birdclef2023  |        12000 |     0.0379167  |    5 |    6 |     7 |     8 |     9 |
| esc50_animals |         5100 |     0.0105882  |    5 |    6 |     7 |     8 |     9 |

## Matched counts, family A

All estimates below are IRMF minus comparator or linear IMF minus EMD in BA percentage points. Intervals use the Nadeau-Bengio corrected variance and are marginal 95% intervals. Two-sided corrected-t p values use Holm across all 48 tests. Wilcoxon is secondary and saved with its own Holm adjustment. Macro-F1 and accuracy form separate supporting families. [All paired tests](paired_tests.csv), [family A source](family_a.csv), [forest plot](figures/family_a_forest.png), [filterable primary tables](overview.html).

| Dataset       |   K | Contrast          |   Estimate, pp | 95% CI, pp         |      Holm p |
|:--------------|----:|:------------------|---------------:|:-------------------|------------:|
| synthetic     |   3 | IRMF - EMD        |         15.2   | [+11.123, +19.277] | 4.67884e-05 |
| synthetic     |   3 | Linear IMF - EMD  |         16.822 | [+12.146, +21.499] | 6.66722e-05 |
| synthetic     |   3 | IRMF - Linear IMF |         -1.622 | [-6.229, +2.985]   | 1           |
| synthetic     |   4 | IRMF - EMD        |         26.6   | [+22.383, +30.817] | 9.10746e-08 |
| synthetic     |   4 | Linear IMF - EMD  |         19.089 | [+14.330, +23.848] | 2.15277e-05 |
| synthetic     |   4 | IRMF - Linear IMF |          7.511 | [+4.547, +10.475]  | 0.00228236  |
| synthetic     |   5 | IRMF - EMD        |         28.4   | [+23.678, +33.122] | 1.61715e-07 |
| synthetic     |   5 | Linear IMF - EMD  |         19.511 | [+14.224, +24.798] | 5.11494e-05 |
| synthetic     |   5 | IRMF - Linear IMF |          8.889 | [+5.664, +12.113]  | 0.00105985  |
| synthetic     |   6 | IRMF - EMD        |         31.711 | [+27.776, +35.647] | 3.70229e-09 |
| synthetic     |   6 | Linear IMF - EMD  |         21.578 | [+15.249, +27.907] | 0.000118952 |
| synthetic     |   6 | IRMF - Linear IMF |         10.133 | [+5.492, +14.775]  | 0.00846733  |
| dronerf       |   3 | IRMF - EMD        |          3.845 | [+2.429, +5.261]   | 0.00119234  |
| dronerf       |   3 | Linear IMF - EMD  |          7.325 | [+5.479, +9.171]   | 2.37691e-05 |
| dronerf       |   3 | IRMF - Linear IMF |         -3.48  | [-4.520, -2.440]   | 0.000142089 |
| dronerf       |   4 | IRMF - EMD        |          4.003 | [+2.963, +5.043]   | 3.3312e-05  |
| dronerf       |   4 | Linear IMF - EMD  |          9.954 | [+8.209, +11.700]  | 3.06773e-07 |
| dronerf       |   4 | IRMF - Linear IMF |         -5.951 | [-7.803, -4.100]   | 0.000214579 |
| dronerf       |   5 | IRMF - EMD        |          6.949 | [+5.503, +8.395]   | 2.49625e-06 |
| dronerf       |   5 | Linear IMF - EMD  |         13.714 | [+11.508, +15.920] | 1.07646e-07 |
| dronerf       |   5 | IRMF - Linear IMF |         -6.764 | [-8.024, -5.505]   | 6.3132e-07  |
| dronerf       |   6 | IRMF - EMD        |          7.778 | [+6.373, +9.184]   | 4.42181e-07 |
| dronerf       |   6 | Linear IMF - EMD  |         14.089 | [+12.132, +16.046] | 1.6309e-08  |
| dronerf       |   6 | IRMF - Linear IMF |         -6.311 | [-7.405, -5.216]   | 2.73851e-07 |
| birdclef2023  |   3 | IRMF - EMD        |         -0.379 | [-2.398, +1.640]   | 1           |
| birdclef2023  |   3 | Linear IMF - EMD  |         -0.733 | [-2.508, +1.043]   | 1           |
| birdclef2023  |   3 | IRMF - Linear IMF |          0.354 | [-0.884, +1.592]   | 1           |
| birdclef2023  |   4 | IRMF - EMD        |          1.127 | [+0.084, +2.170]   | 0.57886     |
| birdclef2023  |   4 | Linear IMF - EMD  |          0.523 | [-0.926, +1.971]   | 1           |
| birdclef2023  |   4 | IRMF - Linear IMF |          0.604 | [-0.654, +1.862]   | 1           |
| birdclef2023  |   5 | IRMF - EMD        |          1.351 | [-0.218, +2.920]   | 1           |
| birdclef2023  |   5 | Linear IMF - EMD  |          1.403 | [+0.007, +2.798]   | 0.685565    |
| birdclef2023  |   5 | IRMF - Linear IMF |         -0.052 | [-1.649, +1.546]   | 1           |
| birdclef2023  |   6 | IRMF - EMD        |          2.032 | [+0.541, +3.522]   | 0.222019    |
| birdclef2023  |   6 | Linear IMF - EMD  |          1.972 | [+0.598, +3.346]   | 0.172009    |
| birdclef2023  |   6 | IRMF - Linear IMF |          0.06  | [-1.345, +1.465]   | 1           |
| esc50_animals |   3 | IRMF - EMD        |          3.851 | [+0.730, +6.972]   | 0.340088    |
| esc50_animals |   3 | Linear IMF - EMD  |          3.519 | [+0.162, +6.876]   | 0.617491    |
| esc50_animals |   3 | IRMF - Linear IMF |          0.332 | [-2.333, +2.996]   | 1           |
| esc50_animals |   4 | IRMF - EMD        |          4.558 | [+0.874, +8.242]   | 0.340088    |
| esc50_animals |   4 | Linear IMF - EMD  |          5.276 | [+2.305, +8.248]   | 0.0422304   |
| esc50_animals |   4 | IRMF - Linear IMF |         -0.718 | [-2.651, +1.214]   | 1           |
| esc50_animals |   5 | IRMF - EMD        |          5.993 | [+1.229, +10.758]  | 0.329012    |
| esc50_animals |   5 | Linear IMF - EMD  |          6.204 | [+2.926, +9.482]   | 0.0269475   |
| esc50_animals |   5 | IRMF - Linear IMF |         -0.211 | [-3.374, +2.952]   | 1           |
| esc50_animals |   6 | IRMF - EMD        |          7.218 | [+4.038, +10.398]  | 0.00621027  |
| esc50_animals |   6 | Linear IMF - EMD  |          7.197 | [+5.461, +8.934]   | 1.49244e-05 |
| esc50_animals |   6 | IRMF - Linear IMF |          0.02  | [-2.302, +2.343]   | 1           |

The pre-specified reading requires both IMF-vs-EMD gains to be positive and Holm-significant at K=5 and K=6. The result for each dataset is:

- synthetic: survives matched counts, 4/4 required contrasts are positive and significant.
- dronerf: survives matched counts, 4/4 required contrasts are positive and significant.
- birdclef2023: does not meet the pre-specified criterion, 0/4 required contrasts are positive and significant.
  The four point gains range from +1.351 to +2.032 pp.
- esc50_animals: does not meet the pre-specified criterion, 3/4 required contrasts are positive and significant.
  The four point gains range from +5.993 to +7.218 pp.
  The failed requirement is IRMF - EMD at K=5: +5.993 pp [+1.229, +10.758], Holm p=0.329012.

## Native count curves, family B

Each of 17 counts-versus-default contrasts per dataset has its own paired difference and corrected interval in [family_b.csv](family_b.csv), with Holm within the dataset. A best observed K is descriptive, chosen from the native grid and the study default; it does not estimate performance after independent model selection. The primary comp results are:

| dataset       | family   |   k |   BA (%) |   Default BA (%) |   Best - default, pp | Family B Holm p   |   n_features |
|:--------------|:---------|----:|---------:|-----------------:|---------------------:|:------------------|-------------:|
| birdclef2023  | linear   |  17 |   48.856 |           48.856 |                0     | reference         |          360 |
| birdclef2023  | irmf     |  17 |   49.568 |           49.568 |                0     | reference         |          360 |
| birdclef2023  | emd      |   3 |   45.929 |           43.45  |                2.479 | 0.0264356         |           80 |
| dronerf       | linear   |  17 |   57.471 |           57.471 |                0     | reference         |          720 |
| dronerf       | irmf     |  17 |   50.359 |           50.359 |                0     | reference         |          720 |
| dronerf       | emd      |   3 |   40.649 |           38.73  |                1.919 | 0.00467754        |          160 |
| esc50_animals | linear   |  13 |   65.291 |           65.269 |                0.021 | 1                 |          280 |
| esc50_animals | irmf     |  13 |   65.332 |           64.976 |                0.356 | 1                 |          280 |
| esc50_animals | emd      |   4 |   56.517 |           54.319 |                2.198 | 0.177419          |          100 |
| synthetic     | linear   |   9 |   74.667 |           73.022 |                1.644 | 1                 |          200 |
| synthetic     | irmf     |  15 |   84.689 |           84.689 |                0     | reference         |          320 |
| synthetic     | emd      |   4 |   55.311 |           50.422 |                4.889 | 0.119053          |          100 |

- synthetic: 1/17 counts differ from default after Holm; 1 are lower and 0 are higher. [BA curves for both classifiers and feature sets](figures/ba_vs_k_synthetic.png).
- dronerf: 11/17 counts differ from default after Holm; 8 are lower and 3 are higher. [BA curves for both classifiers and feature sets](figures/ba_vs_k_dronerf.png).
- birdclef2023: 11/17 counts differ from default after Holm; 8 are lower and 3 are higher. [BA curves for both classifiers and feature sets](figures/ba_vs_k_birdclef2023.png).
- esc50_animals: 2/17 counts differ from default after Holm; 2 are lower and 0 are higher. [BA curves for both classifiers and feature sets](figures/ba_vs_k_esc50_animals.png).

The best observed EMD count is three or four on every dataset with the fixed classical model. For the datasets where that best-count improvement passes family B's correction:

- birdclef2023, EMD K=3: +2.479 pp [+0.974, +3.983], Holm p=0.0264356, versus K=8.
- dronerf, EMD K=3: +1.919 pp [+0.995, +2.843], Holm p=0.00467754, versus K=8.

## Default k_max truncation, family C

These inputs keep the coarsest sequential stages. Their final residual combines every finer stage and the finest solver residual. They therefore answer a different question from the native K-stage cascades. Holm spans eight contrasts per dataset. [Family C source](family_c.csv).

| Dataset       |   K | Contrast                   |   Estimate, pp | 95% CI, pp         |      Holm p |
|:--------------|----:|:---------------------------|---------------:|:-------------------|------------:|
| synthetic     |   4 | Linear IMF k_max=4 - full  |         -9.333 | [-14.445, -4.222]  | 0.00465436  |
| synthetic     |   8 | Linear IMF k_max=8 - full  |         -7.622 | [-10.950, -4.295]  | 0.000914787 |
| synthetic     |  12 | Linear IMF k_max=12 - full |          1.289 | [-1.056, +3.634]   | 0.379903    |
| synthetic     |   4 | IRMF k_max=4 - full        |        -19.289 | [-24.813, -13.765] | 2.04981e-05 |
| synthetic     |   8 | IRMF k_max=8 - full        |         -8.533 | [-11.068, -5.999]  | 2.64928e-05 |
| synthetic     |  12 | IRMF k_max=12 - full       |          1.556 | [-0.866, +3.977]   | 0.379903    |
| synthetic     |   8 | GD60 k_max=8 - GD60 full   |         -8.756 | [-11.482, -6.029]  | 3.7337e-05  |
| synthetic     |   8 | GD60 k_max=8 - EMD K=8     |         25.667 | [+20.749, +30.585] | 1.81762e-07 |
| dronerf       |   4 | Linear IMF k_max=4 - full  |        -15.636 | [-16.592, -14.680] | 3.85306e-14 |
| dronerf       |   8 | Linear IMF k_max=8 - full  |        -12.191 | [-13.711, -10.671] | 3.27803e-10 |
| dronerf       |  12 | Linear IMF k_max=12 - full |         -6.73  | [-7.940, -5.521]   | 3.00922e-08 |
| dronerf       |   4 | IRMF k_max=4 - full        |        -14.363 | [-15.950, -12.776] | 8.05969e-11 |
| dronerf       |   8 | IRMF k_max=8 - full        |        -11.436 | [-12.594, -10.278] | 2.94971e-11 |
| dronerf       |  12 | IRMF k_max=12 - full       |         -7.253 | [-8.600, -5.905]   | 3.06963e-08 |
| dronerf       |   8 | GD60 k_max=8 - GD60 full   |        -10.364 | [-11.153, -9.576]  | 6.8995e-13  |
| dronerf       |   8 | GD60 k_max=8 - EMD K=8     |          0.436 | [-0.839, +1.710]   | 0.475586    |
| birdclef2023  |   4 | Linear IMF k_max=4 - full  |        -12.507 | [-14.241, -10.773] | 2.03494e-09 |
| birdclef2023  |   8 | Linear IMF k_max=8 - full  |        -12.231 | [-13.811, -10.651] | 1.05268e-09 |
| birdclef2023  |  12 | Linear IMF k_max=12 - full |         -5.739 | [-6.772, -4.706]   | 3.05631e-08 |
| birdclef2023  |   4 | IRMF k_max=4 - full        |        -11.434 | [-12.976, -9.892]  | 1.63245e-09 |
| birdclef2023  |   8 | IRMF k_max=8 - full        |        -10.61  | [-12.376, -8.843]  | 1.49651e-08 |
| birdclef2023  |  12 | IRMF k_max=12 - full       |         -4.725 | [-6.095, -3.356]   | 6.68549e-06 |
| birdclef2023  |   8 | GD60 k_max=8 - GD60 full   |        -10.284 | [-11.920, -8.647]  | 1.04174e-08 |
| birdclef2023  |   8 | GD60 k_max=8 - EMD K=8     |         -4.231 | [-5.687, -2.775]   | 2.18464e-05 |
| esc50_animals |   4 | Linear IMF k_max=4 - full  |        -14.937 | [-19.015, -10.858] | 8.47682e-06 |
| esc50_animals |   8 | Linear IMF k_max=8 - full  |        -10.366 | [-14.070, -6.663]  | 0.000129268 |
| esc50_animals |  12 | Linear IMF k_max=12 - full |         -3.179 | [-6.544, +0.187]   | 0.124543    |
| esc50_animals |   4 | IRMF k_max=4 - full        |        -15.157 | [-17.610, -12.703] | 2.07566e-08 |
| esc50_animals |   8 | IRMF k_max=8 - full        |         -9.841 | [-11.672, -8.009]  | 1.10026e-07 |
| esc50_animals |  12 | IRMF k_max=12 - full       |         -4.242 | [-7.355, -1.128]   | 0.0334432   |
| esc50_animals |   8 | GD60 k_max=8 - GD60 full   |         -9.746 | [-11.658, -7.833]  | 1.84853e-07 |
| esc50_animals |   8 | GD60 k_max=8 - EMD K=8     |          1.221 | [-1.460, +3.902]   | 0.345181    |

For pimf's GD60/k_max=8 versus the same GD60 full cascade:

- synthetic: -8.756 pp [-11.482, -6.029], Holm p=3.7337e-05.
- dronerf: -10.364 pp [-11.153, -9.576], Holm p=6.8995e-13.
- birdclef2023: -10.284 pp [-11.920, -8.647], Holm p=1.04174e-08.
- esc50_animals: -9.746 pp [-11.658, -7.833], Holm p=1.84853e-07.

## Secondary results

The exploratory variants have separate corrections: 48 family-A tests across datasets and eight family-C tests within each dataset before budget cuts, separately for classical/comp+log_scale, LightGBM/comp and LightGBM/comp+log_scale. Fixed-classical results retain 15 splits. LightGBM uses 15 splits on seeds [0, 1, 2], with feature sets ['comp', 'comp+log_scale']. All 6,960 completed fits are in the planned scope. [Scores](per_split_scores.csv), [mean scores](score_summary.csv), [best observed K](best_k.csv) and [secondary test details](paired_tests.csv) provide the numbers.

- classical/comp+log_scale: family A has 22 positive and 2 negative Holm-significant contrasts out of 48. Family C has 28/32 significant contrasts.
- lightgbm/comp: family A has 16 positive and 4 negative Holm-significant contrasts out of 48. Family C has 27/32 significant contrasts.
- lightgbm/comp+log_scale: family A has 18 positive and 3 negative Holm-significant contrasts out of 48. Family C has 27/32 significant contrasts.

The exploratory summary below counts positive, Holm-significant IMF-vs-EMD gains at K=5 and K=6, out of the available four contrasts. It also shows the GD60/k_max=8 change against GD60/full. The primary matched-count reading remains the classical/comp family A.

| Variant                  | Dataset       | K=5,6 gains positive/significant   |   GD60 k8 - full, pp |    C Holm p |
|:-------------------------|:--------------|:-----------------------------------|---------------------:|------------:|
| classical/comp+log_scale | synthetic     | 4/4                                |               -9.911 | 0.000159084 |
| classical/comp+log_scale | dronerf       | 4/4                                |               -8.574 | 1.68696e-10 |
| classical/comp+log_scale | birdclef2023  | 0/4                                |              -10.349 | 4.83819e-08 |
| classical/comp+log_scale | esc50_animals | 3/4                                |               -9.715 | 1.7686e-05  |
| lightgbm/comp            | synthetic     | 4/4                                |              -14.267 | 0.000647487 |
| lightgbm/comp            | dronerf       | 4/4                                |               -6.357 | 1.16422e-06 |
| lightgbm/comp            | birdclef2023  | 0/4                                |               -4.802 | 6.61371e-06 |
| lightgbm/comp            | esc50_animals | 0/4                                |               -7.931 | 0.000638922 |
| lightgbm/comp+log_scale  | synthetic     | 4/4                                |              -16.667 | 0.000749835 |
| lightgbm/comp+log_scale  | dronerf       | 4/4                                |               -5.251 | 2.35524e-06 |
| lightgbm/comp+log_scale  | birdclef2023  | 0/4                                |               -4.726 | 4.32531e-07 |
| lightgbm/comp+log_scale  | esc50_animals | 3/4                                |               -7.759 | 0.000162572 |

- synthetic raw comp BA: lightgbm 35.467%; logreg 47.444%.
- dronerf raw comp BA: lightgbm 58.685%; linsvm 34.947%.
- birdclef2023 raw comp BA: lightgbm 46.834%; logreg 37.612%.
- esc50_animals raw comp BA: lightgbm 52.369%; linsvm 44.336%.

## Slot contents and padding

[Slot energies](slot_energies.csv) report median and 95th-percentile relative power for every native slot, pooled over frame-channels, plus radius and true scale rank. Energy uses the shared feature definition, each slot's power divided by the sum of slot powers. The decompositions are not orthogonal, so this is not an additive partition of input power. Slot numbers alone do not equate scales across methods or across native schedules. The IMF solver residual is the finest content. [Padding fractions](padding_fractions.csv) report absent Kth EMD slots, checked against all-zero cached slots. Feature count matches across methods at matched K even where EMD has padding.

| dataset       |   K=3, % padded |   K=4, % padded |   K=5, % padded |   K=6, % padded |   K=7, % padded |   K=8, % padded |
|:--------------|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
| birdclef2023  |          0.0000 |          0.0000 |          0.0917 |          1.8167 |         15.3167 |         61.9333 |
| dronerf       |          0.0000 |          0.0000 |          0.0000 |          0.0000 |          0.0000 |          2.1583 |
| esc50_animals |          0.0000 |          0.0000 |          0.2549 |          3.0588 |         25.0392 |         75.9804 |
| synthetic     |          0.0000 |          0.0000 |          0.0000 |          0.2000 |         17.8667 |         78.8000 |

## Compute cost and solver integrity

Timed processing and classification sections, plus the pilot, total 5.283 wall hours. Classification timers begin after input-array preparation. Calendar elapsed time from the first pilot entry to analysis is 5.415 hours, including tests, commits, implementation and waits between phases. The summed measured CPU time is 29.910 hours. CPU includes classification fits, block decomposition/feature work and native pilot fits; it excludes direct normalization, input-array preparation, direct pilot verification, worker JIT/startup, tests and orchestration. These are concurrent-load measurements. New external data occupy 39.627 GB against the 60 GB cap. [Session timings](compute_totals.json), [costs per input](costs.csv), [pilot timings](pilot_timings.csv).

Measured native decomposition CPU seconds per complete frame, including every channel, are below. Shared normalization and feature extraction are separate. These are full-population measurements rather than the pilot projections.

| dataset       | family   |      K=3 |      K=4 |      K=5 |      K=6 |      K=9 |     K=13 |
|:--------------|:---------|---------:|---------:|---------:|---------:|---------:|---------:|
| birdclef2023  | irmf     | 0.018174 | 0.020147 | 0.022568 | 0.025126 | 0.033286 | 0.042994 |
| birdclef2023  | linear   | 0.001980 | 0.002127 | 0.002335 | 0.002614 | 0.003446 | 0.004560 |
| dronerf       | irmf     | 0.031803 | 0.035345 | 0.040044 | 0.046004 | 0.061278 | 0.082581 |
| dronerf       | linear   | 0.003764 | 0.004096 | 0.004602 | 0.005094 | 0.006623 | 0.008822 |
| esc50_animals | irmf     | 0.020080 | 0.022854 | 0.024068 | 0.026485 | 0.035412 | 0.047848 |
| esc50_animals | linear   | 0.001892 | 0.002040 | 0.002264 | 0.002505 | 0.003299 | 0.004406 |
| synthetic     | irmf     | 0.004896 | 0.005476 | 0.006130 | 0.006958 | 0.009400 | 0.012901 |
| synthetic     | linear   | 0.000541 | 0.000614 | 0.000679 | 0.000776 | 0.001021 | 0.001415 |

There are 0 recorded Newton window-stage cap hits, across 0 channels. The full population includes 255,600 native IRMF channel decompositions. The timing pilot recorded 0 cap-hit attempts. Every accepted native IRMF channel converged. There are zero unresolved Newton failures. There are 0 failed solver attempts before any retry. [Cap events](newton_cap_events.csv) preserves every 100-iteration retry and its failed-row count. [Failed attempts](newton_failures.csv) includes errors independently of cap status. pimf stayed at 0.4.0 with all five installed Python-file hashes equal to the initial fingerprint. Each runner's complete fingerprint record stays under the external components/audit/ directories. The [final integrity record](integrity.json) also confirms that model, construction and shared feature code hashes match the manifests used by their runs.

## Deviations and limits

No fixed-classical dataset, contrast, split, component count or feature set was cut. No LightGBM budget cut was triggered. No shared study module or installed pimf file changed. During setup, the initial pilot and test commands were launched concurrently, an execution oversight. Their combined worker peak was not recorded, so compliance with the aggregate worker cap cannot be verified for that initial interval. Subsequent processing, classification and test phases ran sequentially. The label-free pilot supplied timing only. The protocol explicitly retained the historical DroneRF C=0.1 rather than the generic SVM grid. After pre-registration, the historical LightGBM timing sources were found to use two threads for synthetic and BirdCLEF, rather than the single-thread assumption used in the initial CPU-equivalent projection. Multiplying those wall times by their declared thread counts raised the projection from 9.93 to 10.53 hours. This timing correction used no new-input scores and triggered no cut. The appended protocol run notes and external budget_timing_audit.json preserve the correction. The new default LightGBM fits then showed higher CPU costs than those historical timings. Before classifying any new input, the default-only timing measurements were used to freeze budget tier 0. The uncut projection was 11.148 hours; the selected tier projected 11.148 hours. [The complete timing-only decision](budget_plan.json) records every tested cut tier, completed wall time and projected remaining CPU cost. It uses no new-input scores. The CPU total has the measurement exclusions stated above, so it is not a complete operating-system accounting of every process. These timing exclusions changed no fitted input or result.

This comparison concerns the fixed prepared populations and chosen classifiers. Matching feature count does not match information per slot, active EMD components, or the scale spacing; classical training-fold removal of constant features can also leave different effective widths. Best-K estimates use the same outer folds and remain descriptive.

## Reproduce

```bash
cd /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf
export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
../../.venv/bin/python -m pytest -q tests components/tests
../../.venv/bin/python -m components.processing --workers 6
../../.venv/bin/python -m components.classification --phase defaults --workers 6
../../.venv/bin/python -m components.budget --append-notes
../../.venv/bin/python -m components.run_budgeted --workers 6
../../.venv/bin/python -m components.analysis --out /absolute/path/to/fresh/export
```

Completed checkpoints resume without replacement. The original label-free pilot is already recorded and is not rerun over its immutable outputs. The full instructions and initial fingerprint are in the [components README](../../../experiments/classification-emd-irmf/components/README.md), including detached launches with fresh logs and PID records. The production new-input job used nohup, a new session, closed stdin and regular-file stdout/stderr.
