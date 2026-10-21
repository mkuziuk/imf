# dronerf_phase3_trees_xgb: dronerf / 4class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1170.1 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |    std |
|:----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'xgboost')                          | 0.8144 | 0.0152 |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.8617 | 0.0170 |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.8650 | 0.0149 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.8800 | 0.0160 |
| ('imf_linear_a2', 'comp', 'xgboost')                | 0.8674 | 0.0129 |
| ('imf_linear_a2', 'comp+log_scale', 'xgboost')      | 0.8794 | 0.0147 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.8477 | 0.0158 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.8775 | 0.0185 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.8484 | 0.0153 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.8790 | 0.0174 |
| ('irmf_h2_a2_conv', 'comp', 'xgboost')              | 0.8473 | 0.0141 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'xgboost')    | 0.8762 | 0.0175 |
| ('raw', 'comp', 'xgboost')                          | 0.8114 | 0.0181 |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.8616 | 0.0186 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | xgboost      | raw                | emd                |     -0.0030 |  -0.0157 |    0.0096 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw                | imf_linear_a1.41   |     -0.0536 |  -0.0708 |   -0.0363 |      0.0004 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41      |     -0.0363 |  -0.0524 |   -0.0202 |      0.0072 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0370 |  -0.0519 |   -0.0221 |      0.0031 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0359 |  -0.0523 |   -0.0195 |      0.0090 |            0.0026 |
| comp           | xgboost      | raw                | imf_linear_a2      |     -0.0560 |  -0.0715 |   -0.0405 |      0.0001 |            0.0026 |
| comp           | xgboost      | emd                | imf_linear_a1.41   |     -0.0505 |  -0.0648 |   -0.0362 |      0.0001 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41      |     -0.0333 |  -0.0469 |   -0.0197 |      0.0035 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0340 |  -0.0464 |   -0.0215 |      0.0014 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0328 |  -0.0453 |   -0.0204 |      0.0018 |            0.0026 |
| comp           | xgboost      | emd                | imf_linear_a2      |     -0.0530 |  -0.0653 |   -0.0408 |      0.0000 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0172 |   0.0077 |    0.0268 |      0.0298 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0166 |   0.0071 |    0.0260 |      0.0352 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0177 |   0.0092 |    0.0262 |      0.0128 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | imf_linear_a2      |     -0.0025 |  -0.0091 |    0.0041 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0007 |  -0.0052 |    0.0039 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0004 |  -0.0065 |    0.0073 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0197 |  -0.0263 |   -0.0131 |      0.0006 |            0.0026 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0011 |  -0.0060 |    0.0082 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0191 |  -0.0257 |   -0.0124 |      0.0008 |            0.0026 |
| comp           | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0202 |  -0.0260 |   -0.0143 |      0.0001 |            0.0026 |
| comp+log_scale | xgboost      | raw                | emd                |     -0.0001 |  -0.0074 |    0.0072 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw                | imf_linear_a1.41   |     -0.0183 |  -0.0255 |   -0.0112 |      0.0025 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41      |     -0.0158 |  -0.0239 |   -0.0078 |      0.0178 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0174 |  -0.0260 |   -0.0088 |      0.0150 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0146 |  -0.0217 |   -0.0075 |      0.0142 |            0.0026 |
| comp+log_scale | xgboost      | raw                | imf_linear_a2      |     -0.0178 |  -0.0271 |   -0.0085 |      0.0201 |            0.0026 |
| comp+log_scale | xgboost      | emd                | imf_linear_a1.41   |     -0.0182 |  -0.0247 |   -0.0118 |      0.0010 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41      |     -0.0157 |  -0.0230 |   -0.0085 |      0.0091 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0173 |  -0.0231 |   -0.0115 |      0.0006 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0145 |  -0.0218 |   -0.0072 |      0.0166 |            0.0026 |
| comp+log_scale | xgboost      | emd                | imf_linear_a2      |     -0.0177 |  -0.0251 |   -0.0102 |      0.0046 |            0.0026 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0025 |  -0.0029 |    0.0079 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0010 |  -0.0043 |    0.0063 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0037 |  -0.0009 |    0.0083 |      1.0000 |            0.0186 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | imf_linear_a2      |      0.0006 |  -0.0038 |    0.0050 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0015 |  -0.0064 |    0.0033 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0012 |  -0.0030 |    0.0055 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0019 |  -0.0088 |    0.0050 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0028 |  -0.0021 |    0.0077 |      1.0000 |            0.3232 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0004 |  -0.0067 |    0.0059 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0032 |  -0.0100 |    0.0037 |      1.0000 |            1.0000 |
