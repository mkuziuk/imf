# esc50_phase3_slots9: esc50_animals / animal

90 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 455.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.6637 | 0.0252 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.6749 | 0.0263 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.6554 | 0.0307 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.6531 | 0.0256 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.6552 | 0.0292 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6562 | 0.0246 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0083 |  -0.0074 |    0.0241 |      0.8277 |            0.1060 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0085 |  -0.0022 |    0.0192 |      0.4387 |            0.0051 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0002 |  -0.0090 |    0.0094 |      0.9660 |            0.9780 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0218 |  -0.0029 |    0.0464 |      0.3948 |            0.0171 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0187 |  -0.0003 |    0.0377 |      0.3205 |            0.0058 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0031 |  -0.0113 |    0.0051 |      0.8638 |            0.2140 |
