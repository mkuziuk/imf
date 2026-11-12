# esc50_phase3_matched: esc50_animals / bird

60 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 54.1 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                 |   mean |    std |
|:------------------------------------------------|-------:|-------:|
| ('imf_linear_a2', 'comp', 'lightgbm')           | 0.7782 | 0.0477 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm') | 0.7772 | 0.0549 |
| ('irmf_h2_a2', 'comp', 'lightgbm')              | 0.7768 | 0.0393 |
| ('irmf_h2_a2', 'comp+log_scale', 'lightgbm')    | 0.7761 | 0.0410 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a      | method_b   |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:--------------|:-----------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | imf_linear_a2 | irmf_h2_a2 |      0.0014 |  -0.0285 |    0.0313 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a2 | irmf_h2_a2 |      0.0011 |  -0.0426 |    0.0448 |      1.0000 |            1.0000 |
