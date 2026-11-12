# esc50_phase3_matched: esc50_animals / animal

60 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 312.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                 |   mean |    std |
|:------------------------------------------------|-------:|-------:|
| ('imf_linear_a2', 'comp', 'lightgbm')           | 0.6526 | 0.0343 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm') | 0.6548 | 0.0321 |
| ('irmf_h2_a2', 'comp', 'lightgbm')              | 0.6550 | 0.0394 |
| ('irmf_h2_a2', 'comp+log_scale', 'lightgbm')    | 0.6556 | 0.0371 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a      | method_b   |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:--------------|:-----------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | imf_linear_a2 | irmf_h2_a2 |     -0.0024 |  -0.0241 |    0.0192 |      1.0000 |            0.7788 |
| comp+log_scale | lightgbm     | imf_linear_a2 | irmf_h2_a2 |     -0.0008 |  -0.0252 |    0.0236 |      1.0000 |            0.8040 |
