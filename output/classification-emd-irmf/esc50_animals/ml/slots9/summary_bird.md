# esc50_phase3_slots9: esc50_animals / bird

90 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 74.3 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.7734 | 0.0489 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.7748 | 0.0532 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.7655 | 0.0378 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.7626 | 0.0378 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.7673 | 0.0383 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.7659 | 0.0402 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0079 |  -0.0156 |    0.0313 |      1.0000 |            0.3823 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0062 |  -0.0190 |    0.0314 |      1.0000 |            0.6057 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0017 |  -0.0087 |    0.0053 |      1.0000 |            0.6057 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0122 |  -0.0177 |    0.0421 |      1.0000 |            0.4730 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0089 |  -0.0202 |    0.0381 |      1.0000 |            0.6057 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0033 |  -0.0126 |    0.0060 |      1.0000 |            0.6057 |
