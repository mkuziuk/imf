# dronerf_pilot_trees: dronerf / 10class

12 fits on shared split IDs [0]; wall time 141.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |   std |
|:-------------------------------------------------------|-------:|------:|
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.6234 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.6447 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.5908 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.6363 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.6800 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.6178 |   nan |
| ('raw', 'comp', 'extratrees')                          | 0.5911 |   nan |
| ('raw', 'comp', 'lightgbm')                            | 0.5834 |   nan |
| ('raw', 'comp', 'xgboost')                             | 0.5332 |   nan |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.6370 |   nan |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.6457 |   nan |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.5759 |   nan |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
