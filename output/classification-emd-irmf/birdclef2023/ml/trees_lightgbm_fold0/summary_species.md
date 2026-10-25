# trees_lightgbm_fold0: birdclef2023 / species

10 fits on shared split IDs [0]; wall time 89.9 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |   std |
|:-----------------------------------------------------|-------:|------:|
| ('emd', 'comp', 'lightgbm')                          | 0.4870 |   nan |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.4855 |   nan |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.5166 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.5039 |   nan |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.5217 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.5172 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.5201 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.5251 |   nan |
| ('raw', 'comp', 'lightgbm')                          | 0.4644 |   nan |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.4610 |   nan |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
