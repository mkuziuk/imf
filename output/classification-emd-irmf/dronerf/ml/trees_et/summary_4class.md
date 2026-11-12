# dronerf_phase3_trees_et: dronerf / 4class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 640.8 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.7962 | 0.0209 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.8429 | 0.0193 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.8880 | 0.0107 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.8949 | 0.0127 |
| ('imf_linear_a2', 'comp', 'extratrees')                | 0.8878 | 0.0122 |
| ('imf_linear_a2', 'comp+log_scale', 'extratrees')      | 0.8947 | 0.0127 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.8624 | 0.0135 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.8769 | 0.0125 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.8616 | 0.0096 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.8758 | 0.0111 |
| ('irmf_h2_a2_conv', 'comp', 'extratrees')              | 0.8592 | 0.0114 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'extratrees')    | 0.8806 | 0.0131 |
| ('raw', 'comp', 'extratrees')                          | 0.8399 | 0.0184 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.8890 | 0.0159 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | extratrees   | raw                | emd                |      0.0437 |   0.0280 |    0.0594 |      0.0009 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a1.41   |     -0.0480 |  -0.0603 |   -0.0357 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41      |     -0.0225 |  -0.0344 |   -0.0105 |      0.0206 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0217 |  -0.0358 |   -0.0076 |      0.0738 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0193 |  -0.0310 |   -0.0076 |      0.0503 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a2      |     -0.0478 |  -0.0600 |   -0.0357 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a1.41   |     -0.0917 |  -0.1120 |   -0.0715 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41      |     -0.0662 |  -0.0882 |   -0.0441 |      0.0004 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0654 |  -0.0865 |   -0.0443 |      0.0003 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0630 |  -0.0823 |   -0.0437 |      0.0002 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a2      |     -0.0916 |  -0.1119 |   -0.0713 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0256 |   0.0143 |    0.0368 |      0.0050 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0263 |   0.0166 |    0.0361 |      0.0011 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0287 |   0.0197 |    0.0378 |      0.0003 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0002 |  -0.0059 |    0.0062 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0008 |  -0.0075 |    0.0090 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0032 |  -0.0034 |    0.0098 |      1.0000 |            0.2111 |
| comp           | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0254 |  -0.0351 |   -0.0157 |      0.0015 |            0.0026 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0024 |  -0.0032 |    0.0080 |      1.0000 |            0.3186 |
| comp           | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0261 |  -0.0357 |   -0.0166 |      0.0010 |            0.0026 |
| comp           | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0286 |  -0.0376 |   -0.0195 |      0.0003 |            0.0026 |
| comp+log_scale | extratrees   | raw                | emd                |      0.0461 |   0.0311 |    0.0612 |      0.0004 |            0.0026 |
| comp+log_scale | extratrees   | raw                | imf_linear_a1.41   |     -0.0059 |  -0.0146 |    0.0027 |      1.0000 |            0.0537 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41      |      0.0120 |   0.0007 |    0.0233 |      0.4606 |            0.0037 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41_conv |      0.0132 |   0.0040 |    0.0224 |      0.1072 |            0.0026 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a2_conv    |      0.0083 |  -0.0030 |    0.0196 |      1.0000 |            0.0369 |
| comp+log_scale | extratrees   | raw                | imf_linear_a2      |     -0.0058 |  -0.0136 |    0.0021 |      1.0000 |            0.0483 |
| comp+log_scale | extratrees   | emd                | imf_linear_a1.41   |     -0.0520 |  -0.0669 |   -0.0371 |      0.0001 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41      |     -0.0341 |  -0.0454 |   -0.0227 |      0.0004 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0329 |  -0.0456 |   -0.0202 |      0.0015 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0378 |  -0.0510 |   -0.0246 |      0.0007 |            0.0026 |
| comp+log_scale | extratrees   | emd                | imf_linear_a2      |     -0.0519 |  -0.0666 |   -0.0371 |      0.0001 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0179 |   0.0089 |    0.0269 |      0.0146 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0191 |   0.0131 |    0.0251 |      0.0003 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0142 |   0.0068 |    0.0216 |      0.0182 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0002 |  -0.0037 |    0.0041 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0012 |  -0.0034 |    0.0058 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0037 |  -0.0111 |    0.0037 |      1.0000 |            0.2875 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0178 |  -0.0265 |   -0.0090 |      0.0131 |            0.0026 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0049 |  -0.0114 |    0.0016 |      1.0000 |            0.0369 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0190 |  -0.0248 |   -0.0131 |      0.0002 |            0.0026 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0141 |  -0.0223 |   -0.0058 |      0.0407 |            0.0026 |
