# esc50_phase3_rocket: esc50_animals / bird

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 2124.2 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.7493 | 0.0414 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.7450 | 0.0544 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.7615 | 0.0440 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.7617 | 0.0506 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.7547 | 0.0398 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.7489 | 0.0452 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.7534 | 0.0382 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.7474 | 0.0511 |
| ('raw', 'comp', 'minirocket')                          | 0.7546 | 0.0347 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.7525 | 0.0355 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw              | emd                |      0.0053 |  -0.0313 |    0.0418 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | imf_linear_a1.41   |     -0.0069 |  -0.0364 |    0.0226 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41      |     -0.0001 |  -0.0320 |    0.0318 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | irmf_h2_a1.41_conv |      0.0012 |  -0.0267 |    0.0291 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | imf_linear_a1.41   |     -0.0122 |  -0.0464 |    0.0221 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41      |     -0.0054 |  -0.0343 |    0.0236 |      1.0000 |            1.0000 |
| comp           | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0041 |  -0.0343 |    0.0262 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0068 |  -0.0180 |    0.0316 |      1.0000 |            1.0000 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0081 |  -0.0161 |    0.0323 |      1.0000 |            1.0000 |
| comp           | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0013 |  -0.0133 |    0.0159 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | emd                |      0.0075 |  -0.0314 |    0.0465 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | imf_linear_a1.41   |     -0.0092 |  -0.0440 |    0.0256 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41      |      0.0036 |  -0.0279 |    0.0352 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41_conv |      0.0051 |  -0.0296 |    0.0397 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd              | imf_linear_a1.41   |     -0.0167 |  -0.0455 |    0.0120 |      1.0000 |            0.4309 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41      |     -0.0039 |  -0.0334 |    0.0256 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0025 |  -0.0278 |    0.0229 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0128 |  -0.0146 |    0.0403 |      1.0000 |            0.5729 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0143 |  -0.0211 |    0.0496 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0014 |  -0.0224 |    0.0253 |      1.0000 |            1.0000 |
