# dronerf_phase3_trees_xgb: dronerf / 2class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 407.0 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |    std |
|:----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'xgboost')                          | 0.9495 | 0.0200 |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.9614 | 0.0157 |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.9719 | 0.0158 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.9736 | 0.0123 |
| ('imf_linear_a2', 'comp', 'xgboost')                | 0.9721 | 0.0073 |
| ('imf_linear_a2', 'comp+log_scale', 'xgboost')      | 0.9754 | 0.0066 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.9680 | 0.0127 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.9737 | 0.0124 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.9682 | 0.0112 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.9731 | 0.0115 |
| ('irmf_h2_a2_conv', 'comp', 'xgboost')              | 0.9682 | 0.0105 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'xgboost')    | 0.9733 | 0.0073 |
| ('raw', 'comp', 'xgboost')                          | 0.9441 | 0.0217 |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.9612 | 0.0166 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | xgboost      | raw                | emd                |     -0.0054 |  -0.0201 |    0.0093 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | imf_linear_a1.41   |     -0.0278 |  -0.0488 |   -0.0068 |      0.5076 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41      |     -0.0239 |  -0.0435 |   -0.0042 |      0.6802 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0241 |  -0.0435 |   -0.0047 |      0.6231 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0241 |  -0.0461 |   -0.0022 |      0.9683 |            0.0026 |
| comp           | xgboost      | raw                | imf_linear_a2      |     -0.0280 |  -0.0519 |   -0.0041 |      0.7962 |            0.0026 |
| comp           | xgboost      | emd                | imf_linear_a1.41   |     -0.0224 |  -0.0359 |   -0.0090 |      0.1266 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41      |     -0.0185 |  -0.0313 |   -0.0056 |      0.3314 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0187 |  -0.0328 |   -0.0046 |      0.5076 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0187 |  -0.0348 |   -0.0026 |      0.7962 |            0.0035 |
| comp           | xgboost      | emd                | imf_linear_a2      |     -0.0226 |  -0.0435 |   -0.0016 |      1.0000 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0040 |  -0.0055 |    0.0134 |      1.0000 |            0.9077 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0037 |  -0.0060 |    0.0134 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0037 |  -0.0059 |    0.0133 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0002 |  -0.0165 |    0.0162 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0003 |  -0.0058 |    0.0052 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0003 |  -0.0083 |    0.0078 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0041 |  -0.0171 |    0.0089 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0000 |  -0.0078 |    0.0077 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0039 |  -0.0162 |    0.0084 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0039 |  -0.0151 |    0.0074 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | emd                |     -0.0001 |  -0.0099 |    0.0096 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a1.41   |     -0.0124 |  -0.0251 |    0.0004 |      1.0000 |            0.0222 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41      |     -0.0125 |  -0.0222 |   -0.0027 |      0.5827 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0118 |  -0.0202 |   -0.0034 |      0.3684 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0121 |  -0.0290 |    0.0049 |      1.0000 |            0.0222 |
| comp+log_scale | xgboost      | raw                | imf_linear_a2      |     -0.0142 |  -0.0356 |    0.0073 |      1.0000 |            0.0772 |
| comp+log_scale | xgboost      | emd                | imf_linear_a1.41   |     -0.0122 |  -0.0227 |   -0.0017 |      0.7962 |            0.0120 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41      |     -0.0123 |  -0.0218 |   -0.0028 |      0.5378 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0117 |  -0.0210 |   -0.0024 |      0.6055 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0119 |  -0.0261 |    0.0023 |      1.0000 |            0.0165 |
| comp+log_scale | xgboost      | emd                | imf_linear_a2      |     -0.0140 |  -0.0327 |    0.0046 |      1.0000 |            0.0483 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |     -0.0001 |  -0.0064 |    0.0063 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0006 |  -0.0070 |    0.0081 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0003 |  -0.0107 |    0.0113 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0018 |  -0.0148 |    0.0112 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |      0.0006 |  -0.0032 |    0.0045 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0004 |  -0.0095 |    0.0102 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0017 |  -0.0167 |    0.0133 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |     -0.0003 |  -0.0100 |    0.0095 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0023 |  -0.0174 |    0.0127 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0021 |  -0.0128 |    0.0086 |      1.0000 |            1.0000 |
