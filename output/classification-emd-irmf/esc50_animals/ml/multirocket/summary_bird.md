# esc50_phase3_multirocket: esc50_animals / bird

50 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 338.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                         |   mean |    std |
|:--------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'multirocket')                          | 0.7373 | 0.0516 |
| ('emd', 'comp+log_scale', 'multirocket')                | 0.7339 | 0.0447 |
| ('imf_linear_a1.41', 'comp', 'multirocket')             | 0.7497 | 0.0313 |
| ('imf_linear_a1.41', 'comp+log_scale', 'multirocket')   | 0.7482 | 0.0398 |
| ('irmf_h2_a1.41', 'comp', 'multirocket')                | 0.7481 | 0.0440 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'multirocket')      | 0.7410 | 0.0573 |
| ('irmf_h2_a1.41_conv', 'comp', 'multirocket')           | 0.7482 | 0.0447 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'multirocket') | 0.7362 | 0.0588 |
| ('raw', 'comp', 'multirocket')                          | 0.7498 | 0.0328 |
| ('raw', 'comp+log_scale', 'multirocket')                | 0.7415 | 0.0326 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | multirocket  | raw              | emd                |      0.0125 |  -0.0281 |    0.0530 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | imf_linear_a1.41   |      0.0001 |  -0.0248 |    0.0250 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41      |      0.0017 |  -0.0311 |    0.0345 |      1.0000 |            1.0000 |
| comp           | multirocket  | raw              | irmf_h2_a1.41_conv |      0.0016 |  -0.0305 |    0.0337 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | imf_linear_a1.41   |     -0.0124 |  -0.0531 |    0.0283 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41      |     -0.0108 |  -0.0351 |    0.0136 |      1.0000 |            1.0000 |
| comp           | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0109 |  -0.0309 |    0.0092 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0016 |  -0.0305 |    0.0338 |      1.0000 |            1.0000 |
| comp           | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0015 |  -0.0307 |    0.0338 |      1.0000 |            1.0000 |
| comp           | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0001 |  -0.0052 |    0.0050 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | emd                |      0.0077 |  -0.0364 |    0.0518 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | imf_linear_a1.41   |     -0.0066 |  -0.0477 |    0.0345 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41      |      0.0005 |  -0.0623 |    0.0633 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | raw              | irmf_h2_a1.41_conv |      0.0053 |  -0.0602 |    0.0708 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | imf_linear_a1.41   |     -0.0143 |  -0.0392 |    0.0105 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41      |     -0.0072 |  -0.0395 |    0.0252 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | emd              | irmf_h2_a1.41_conv |     -0.0024 |  -0.0327 |    0.0279 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0071 |  -0.0355 |    0.0498 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0119 |  -0.0263 |    0.0501 |      1.0000 |            1.0000 |
| comp+log_scale | multirocket  | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0048 |  -0.0138 |    0.0234 |      1.0000 |            1.0000 |
