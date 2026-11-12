# dronerf_phase3_trees_lgbm: dronerf / 4class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1514.6 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.8301 | 0.0192 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.8853 | 0.0181 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.8829 | 0.0132 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.8976 | 0.0145 |
| ('imf_linear_a2', 'comp', 'lightgbm')                | 0.8833 | 0.0127 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm')      | 0.8985 | 0.0110 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.8682 | 0.0137 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.8950 | 0.0155 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.8671 | 0.0135 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.8951 | 0.0147 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              | 0.8689 | 0.0135 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    | 0.8955 | 0.0147 |
| ('raw', 'comp', 'lightgbm')                          | 0.8304 | 0.0176 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.8878 | 0.0169 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw                | emd                |      0.0004 |  -0.0100 |    0.0107 |      1.0000 |            1.0000 |
| comp           | lightgbm     | raw                | imf_linear_a1.41   |     -0.0525 |  -0.0639 |   -0.0410 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0378 |  -0.0471 |   -0.0285 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0367 |  -0.0456 |   -0.0278 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0385 |  -0.0489 |   -0.0281 |      0.0001 |            0.0026 |
| comp           | lightgbm     | raw                | imf_linear_a2      |     -0.0529 |  -0.0638 |   -0.0420 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a1.41   |     -0.0528 |  -0.0652 |   -0.0404 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0381 |  -0.0511 |   -0.0251 |      0.0007 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0371 |  -0.0490 |   -0.0251 |      0.0004 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0389 |  -0.0500 |   -0.0277 |      0.0001 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a2      |     -0.0533 |  -0.0661 |   -0.0404 |      0.0000 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0147 |   0.0071 |    0.0223 |      0.0279 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0158 |   0.0095 |    0.0221 |      0.0030 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0140 |   0.0076 |    0.0204 |      0.0099 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0004 |  -0.0059 |    0.0051 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0011 |  -0.0030 |    0.0051 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0007 |  -0.0085 |    0.0070 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |     -0.0151 |  -0.0217 |   -0.0085 |      0.0069 |            0.0026 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0018 |  -0.0075 |    0.0039 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0162 |  -0.0225 |   -0.0099 |      0.0026 |            0.0026 |
| comp           | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0144 |  -0.0207 |   -0.0080 |      0.0073 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | emd                |      0.0025 |  -0.0072 |    0.0122 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a1.41   |     -0.0098 |  -0.0185 |   -0.0011 |      0.5644 |            0.0085 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0072 |  -0.0133 |   -0.0011 |      0.4878 |            0.0085 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0073 |  -0.0133 |   -0.0012 |      0.4878 |            0.0085 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0077 |  -0.0146 |   -0.0008 |      0.5644 |            0.0197 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a2      |     -0.0107 |  -0.0202 |   -0.0011 |      0.5644 |            0.0028 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a1.41   |     -0.0123 |  -0.0208 |   -0.0038 |      0.2016 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0097 |  -0.0164 |   -0.0030 |      0.2016 |            0.0028 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0098 |  -0.0179 |   -0.0016 |      0.4878 |            0.0028 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0102 |  -0.0182 |   -0.0022 |      0.3676 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a2      |     -0.0132 |  -0.0230 |   -0.0033 |      0.2952 |            0.0026 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0026 |  -0.0028 |    0.0081 |      1.0000 |            0.5301 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0026 |  -0.0022 |    0.0073 |      1.0000 |            0.4092 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0021 |  -0.0040 |    0.0082 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |     -0.0009 |  -0.0082 |    0.0065 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0000 |  -0.0035 |    0.0034 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0005 |  -0.0044 |    0.0034 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |     -0.0035 |  -0.0109 |    0.0040 |      1.0000 |            0.8284 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0005 |  -0.0048 |    0.0039 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0034 |  -0.0103 |    0.0035 |      1.0000 |            0.5776 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0030 |  -0.0108 |    0.0049 |      1.0000 |            1.0000 |
