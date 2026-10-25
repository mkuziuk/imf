# esc50_phase3_rocket: esc50_animals / animal

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 3696.0 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'minirocket')                          | 0.5728 | 0.0365 |
| ('emd', 'comp+log_scale', 'minirocket')                | 0.5795 | 0.0431 |
| ('imf_linear_a1.41', 'comp', 'minirocket')             | 0.6081 | 0.0376 |
| ('imf_linear_a1.41', 'comp+log_scale', 'minirocket')   | 0.6073 | 0.0398 |
| ('irmf_h2_a1.41', 'comp', 'minirocket')                | 0.5893 | 0.0370 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'minirocket')      | 0.5932 | 0.0352 |
| ('irmf_h2_a1.41_conv', 'comp', 'minirocket')           | 0.5925 | 0.0333 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'minirocket') | 0.5947 | 0.0345 |
| ('raw', 'comp', 'minirocket')                          | 0.5649 | 0.0355 |
| ('raw', 'comp+log_scale', 'minirocket')                | 0.5719 | 0.0350 |

MiniRocket/MultiRocket biases are fitted on a fixed random subset of training frames only. Inner biases, StandardScaler and logistic-head alpha selection never use validation/test samples for fitting.
The logistic head uses SGD with averaged weights, balanced training weights and fixed shuffled epochs. Alpha is selected from {0.0001, 0.001, 0.01} on grouped inner validation; all stages refit on outer train.
Stacks, transforms, scaler moments and head updates are batched. Transformed float32 features live in temporary disk memmaps. Both feature sets share transform work; their individual timing rows include that shared work.
The same total target feature budget, fit sample count and downsampling apply to every method. aeon rounds kernel counts to multiples of 84; actual output dimensions are recorded per fold.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | minirocket   | raw              | emd                |     -0.0079 |  -0.0426 |    0.0268 |      1.0000 |            1.0000 |
| comp           | minirocket   | raw              | imf_linear_a1.41   |     -0.0432 |  -0.0747 |   -0.0118 |      0.2114 |            0.0012 |
| comp           | minirocket   | raw              | irmf_h2_a1.41      |     -0.0244 |  -0.0534 |    0.0046 |      1.0000 |            0.0302 |
| comp           | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0276 |  -0.0497 |   -0.0056 |      0.3195 |            0.0012 |
| comp           | minirocket   | emd              | imf_linear_a1.41   |     -0.0353 |  -0.0630 |   -0.0077 |      0.3016 |            0.0033 |
| comp           | minirocket   | emd              | irmf_h2_a1.41      |     -0.0165 |  -0.0447 |    0.0117 |      1.0000 |            0.1626 |
| comp           | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0197 |  -0.0434 |    0.0039 |      1.0000 |            0.0302 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0188 |  -0.0020 |    0.0397 |      1.0000 |            0.0341 |
| comp           | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0156 |  -0.0064 |    0.0376 |      1.0000 |            0.1245 |
| comp           | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0032 |  -0.0187 |    0.0122 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | emd                |     -0.0076 |  -0.0518 |    0.0366 |      1.0000 |            1.0000 |
| comp+log_scale | minirocket   | raw              | imf_linear_a1.41   |     -0.0354 |  -0.0678 |   -0.0031 |      0.5791 |            0.0073 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41      |     -0.0212 |  -0.0510 |    0.0086 |      1.0000 |            0.0591 |
| comp+log_scale | minirocket   | raw              | irmf_h2_a1.41_conv |     -0.0228 |  -0.0517 |    0.0061 |      1.0000 |            0.0244 |
| comp+log_scale | minirocket   | emd              | imf_linear_a1.41   |     -0.0278 |  -0.0659 |    0.0103 |      1.0000 |            0.0403 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41      |     -0.0136 |  -0.0474 |    0.0201 |      1.0000 |            0.6769 |
| comp+log_scale | minirocket   | emd              | irmf_h2_a1.41_conv |     -0.0152 |  -0.0481 |    0.0177 |      1.0000 |            0.3875 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0142 |  -0.0128 |    0.0412 |      1.0000 |            0.2046 |
| comp+log_scale | minirocket   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0126 |  -0.0178 |    0.0431 |      1.0000 |            0.5676 |
| comp+log_scale | minirocket   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0016 |  -0.0267 |    0.0236 |      1.0000 |            1.0000 |
