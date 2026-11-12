# trees_xgboost: birdclef2023 / passerine

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 596.7 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |    std |
|:----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'xgboost')                          | 0.7966 | 0.0123 |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.7950 | 0.0131 |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.8171 | 0.0118 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.8170 | 0.0122 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.8109 | 0.0143 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.8097 | 0.0134 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.8105 | 0.0143 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.8097 | 0.0153 |
| ('raw', 'comp', 'xgboost')                          | 0.7733 | 0.0213 |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.7726 | 0.0192 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | xgboost      | raw              | emd                |     -0.0232 |  -0.0408 |   -0.0057 |      0.0912 |            0.0012 |
| comp           | xgboost      | raw              | imf_linear_a1.41   |     -0.0438 |  -0.0639 |   -0.0237 |      0.0046 |            0.0012 |
| comp           | xgboost      | raw              | irmf_h2_a1.41      |     -0.0376 |  -0.0536 |   -0.0215 |      0.0026 |            0.0012 |
| comp           | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0371 |  -0.0526 |   -0.0216 |      0.0022 |            0.0012 |
| comp           | xgboost      | emd              | imf_linear_a1.41   |     -0.0206 |  -0.0282 |   -0.0130 |      0.0008 |            0.0012 |
| comp           | xgboost      | emd              | irmf_h2_a1.41      |     -0.0143 |  -0.0219 |   -0.0067 |      0.0129 |            0.0012 |
| comp           | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0139 |  -0.0209 |   -0.0069 |      0.0094 |            0.0012 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0063 |  -0.0016 |    0.0141 |      0.3293 |            0.0079 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0067 |  -0.0003 |    0.0137 |      0.2976 |            0.0034 |
| comp           | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0004 |  -0.0027 |    0.0035 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw              | emd                |     -0.0224 |  -0.0365 |   -0.0082 |      0.0395 |            0.0012 |
| comp+log_scale | xgboost      | raw              | imf_linear_a1.41   |     -0.0444 |  -0.0613 |   -0.0275 |      0.0010 |            0.0012 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41      |     -0.0371 |  -0.0504 |   -0.0238 |      0.0006 |            0.0012 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0371 |  -0.0501 |   -0.0241 |      0.0005 |            0.0012 |
| comp+log_scale | xgboost      | emd              | imf_linear_a1.41   |     -0.0220 |  -0.0309 |   -0.0131 |      0.0018 |            0.0012 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41      |     -0.0147 |  -0.0238 |   -0.0056 |      0.0370 |            0.0012 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0147 |  -0.0240 |   -0.0054 |      0.0395 |            0.0012 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0073 |  -0.0003 |    0.0149 |      0.2976 |            0.0031 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0073 |   0.0001 |    0.0146 |      0.2865 |            0.0012 |
| comp+log_scale | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0000 |  -0.0068 |    0.0069 |      1.0000 |            1.0000 |
