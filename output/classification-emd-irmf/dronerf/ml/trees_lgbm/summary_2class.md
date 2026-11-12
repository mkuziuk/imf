# dronerf_phase3_trees_lgbm: dronerf / 2class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 492.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.9505 | 0.0203 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.9691 | 0.0169 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.9705 | 0.0166 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.9715 | 0.0160 |
| ('imf_linear_a2', 'comp', 'lightgbm')                | 0.9736 | 0.0111 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm')      | 0.9755 | 0.0114 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.9696 | 0.0158 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.9771 | 0.0135 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.9698 | 0.0153 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.9768 | 0.0127 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              | 0.9719 | 0.0129 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    | 0.9758 | 0.0115 |
| ('raw', 'comp', 'lightgbm')                          | 0.9465 | 0.0207 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.9750 | 0.0123 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw                | emd                |     -0.0040 |  -0.0158 |    0.0078 |      1.0000 |            1.0000 |
| comp           | lightgbm     | raw                | imf_linear_a1.41   |     -0.0240 |  -0.0404 |   -0.0077 |      0.2895 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0231 |  -0.0436 |   -0.0025 |      1.0000 |            0.0044 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0233 |  -0.0425 |   -0.0040 |      0.7725 |            0.0044 |
| comp           | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0254 |  -0.0463 |   -0.0046 |      0.7479 |            0.0026 |
| comp           | lightgbm     | raw                | imf_linear_a2      |     -0.0271 |  -0.0464 |   -0.0077 |      0.3809 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a1.41   |     -0.0200 |  -0.0332 |   -0.0068 |      0.2442 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0190 |  -0.0366 |   -0.0014 |      1.0000 |            0.0060 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0192 |  -0.0356 |   -0.0028 |      0.8637 |            0.0315 |
| comp           | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0214 |  -0.0374 |   -0.0054 |      0.4666 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a2      |     -0.0230 |  -0.0398 |   -0.0063 |      0.4069 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0010 |  -0.0093 |    0.0112 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0008 |  -0.0076 |    0.0092 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0014 |  -0.0129 |    0.0101 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0030 |  -0.0134 |    0.0073 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0002 |  -0.0041 |    0.0037 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0024 |  -0.0101 |    0.0053 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |     -0.0040 |  -0.0170 |    0.0089 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0022 |  -0.0108 |    0.0065 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0038 |  -0.0151 |    0.0075 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0017 |  -0.0137 |    0.0104 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | emd                |      0.0059 |  -0.0073 |    0.0191 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a1.41   |      0.0035 |  -0.0123 |    0.0193 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0021 |  -0.0129 |    0.0086 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0018 |  -0.0105 |    0.0069 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0008 |  -0.0121 |    0.0105 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a2      |     -0.0006 |  -0.0187 |    0.0176 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a1.41   |     -0.0024 |  -0.0187 |    0.0139 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0080 |  -0.0194 |    0.0034 |      1.0000 |            0.0814 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0077 |  -0.0163 |    0.0009 |      1.0000 |            0.0044 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0067 |  -0.0174 |    0.0039 |      1.0000 |            0.1282 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a2      |     -0.0064 |  -0.0280 |    0.0152 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0056 |  -0.0189 |    0.0076 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |     -0.0053 |  -0.0173 |    0.0067 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |     -0.0043 |  -0.0170 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0041 |  -0.0164 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0003 |  -0.0046 |    0.0052 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0013 |  -0.0058 |    0.0083 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |      0.0016 |  -0.0151 |    0.0183 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0010 |  -0.0041 |    0.0060 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |      0.0013 |  -0.0146 |    0.0171 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |      0.0003 |  -0.0147 |    0.0153 |      1.0000 |            1.0000 |
