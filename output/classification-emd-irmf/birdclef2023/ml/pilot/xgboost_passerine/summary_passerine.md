# trees_xgboost_passerine_pilot: birdclef2023 / passerine

10 fits on shared split IDs [0]; wall time 58.3 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |   std |
|:----------------------------------------------------|-------:|------:|
| ('emd', 'comp', 'xgboost')                          | 0.8125 |   nan |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.8046 |   nan |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.8396 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.8373 |   nan |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.8354 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.8340 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.8349 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.8361 |   nan |
| ('raw', 'comp', 'xgboost')                          | 0.7957 |   nan |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.7893 |   nan |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
