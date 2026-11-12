# Tree importance by scale

Normalized within each fitted model, then averaged across folds. Each component sums its 20 statistics.
Scale rank 0 is finest; residual_fine is the finest IMF/IRMF content. Importance is descriptive, not causal.
Split count, gain and impurity are different measures and should not be compared between model families.

| task   | classifier   | method             | feature_set    | importance_type   | top five scales                                                |
|:-------|:-------------|:-------------------|:---------------|:------------------|:---------------------------------------------------------------|
| animal | extratrees   | emd                | comp           | impurity          | imf1 25.9%, imf2 18.1%, imf3 16.2%, imf4 12.6%, imf5 8.7%      |
| animal | extratrees   | emd                | comp+log_scale | impurity          | imf1 25.6%, imf2 17.8%, imf3 15.9%, imf4 12.5%, imf5 8.6%      |
| animal | extratrees   | imf_linear_a1.41   | comp           | impurity          | w9 7.2%, w13 7.1%, w93 6.7%, w7 6.7%, w129 6.5%                |
| animal | extratrees   | imf_linear_a1.41   | comp+log_scale | impurity          | w13 7.3%, w9 7.2%, w7 6.6%, w129 6.6%, w93 6.5%                |
| animal | extratrees   | irmf_h2_a1.41      | comp           | impurity          | w9 7.1%, w129 7.0%, w65 6.8%, w93 6.8%, w17 6.8%               |
| animal | extratrees   | irmf_h2_a1.41      | comp+log_scale | impurity          | w129 7.0%, w9 7.0%, w93 6.9%, w13 6.8%, w65 6.7%               |
| animal | extratrees   | irmf_h2_a1.41_conv | comp           | impurity          | w9 7.1%, w93 6.9%, w129 6.9%, w65 6.9%, w13 6.8%               |
| animal | extratrees   | irmf_h2_a1.41_conv | comp+log_scale | impurity          | w129 7.1%, w93 7.0%, w9 7.0%, w13 6.9%, w17 6.7%               |
| animal | extratrees   | raw                | comp           | impurity          | frame 100.0%                                                   |
| animal | extratrees   | raw                | comp+log_scale | impurity          | frame 94.0%, log_scale 6.0%                                    |
| animal | lightgbm     | emd                | comp           | split_count       | imf1 31.7%, imf2 17.8%, imf3 17.2%, imf4 14.0%, imf5 6.8%      |
| animal | lightgbm     | emd                | comp+log_scale | split_count       | imf1 28.7%, imf2 16.8%, imf3 16.3%, imf4 13.1%, log_scale 7.3% |
| animal | lightgbm     | imf_linear_a1.41   | comp           | split_count       | residual_fine 10.2%, w13 7.6%, w17 7.1%, w5 6.8%, w47 6.7%     |
| animal | lightgbm     | imf_linear_a1.41   | comp+log_scale | split_count       | residual_fine 9.8%, w13 7.2%, w17 6.8%, w5 6.4%, w47 6.4%      |
| animal | lightgbm     | irmf_h2_a1.41      | comp           | split_count       | residual_fine 10.9%, w17 7.3%, w5 7.0%, w9 6.6%, w33 6.6%      |
| animal | lightgbm     | irmf_h2_a1.41      | comp+log_scale | split_count       | residual_fine 10.3%, w17 6.8%, w9 6.5%, w5 6.5%, w65 6.4%      |
| animal | lightgbm     | irmf_h2_a1.41_conv | comp           | split_count       | residual_fine 11.1%, w17 7.2%, w5 7.1%, w9 6.7%, w13 6.6%      |
| animal | lightgbm     | irmf_h2_a1.41_conv | comp+log_scale | split_count       | residual_fine 10.4%, w17 6.7%, w5 6.5%, w9 6.5%, w65 6.4%      |
| animal | lightgbm     | raw                | comp           | split_count       | frame 100.0%                                                   |
| animal | lightgbm     | raw                | comp+log_scale | split_count       | frame 87.8%, log_scale 12.2%                                   |
| animal | xgboost      | emd                | comp           | gain              | imf1 31.5%, imf2 16.3%, imf3 15.4%, imf4 13.4%, imf5 6.9%      |
| animal | xgboost      | emd                | comp+log_scale | gain              | imf1 32.0%, imf2 16.2%, imf3 15.7%, imf4 13.4%, imf5 6.5%      |
| animal | xgboost      | imf_linear_a1.41   | comp           | gain              | w13 10.6%, w129 8.0%, w65 7.2%, w5 7.1%, w9 7.0%               |
| animal | xgboost      | imf_linear_a1.41   | comp+log_scale | gain              | w13 11.9%, w129 8.2%, w65 7.3%, w9 6.9%, w33 6.4%              |
| animal | xgboost      | irmf_h2_a1.41      | comp           | gain              | w13 10.0%, w33 8.3%, w129 8.3%, w65 7.9%, w9 7.7%              |
| animal | xgboost      | irmf_h2_a1.41      | comp+log_scale | gain              | w13 10.8%, w33 8.2%, w65 7.9%, w129 7.9%, w9 7.2%              |
| animal | xgboost      | irmf_h2_a1.41_conv | comp           | gain              | w13 10.3%, w33 8.5%, w65 8.2%, w129 7.7%, w9 7.5%              |
| animal | xgboost      | irmf_h2_a1.41_conv | comp+log_scale | gain              | w13 10.5%, w33 8.4%, w65 8.3%, w129 7.6%, w9 7.4%              |
| animal | xgboost      | raw                | comp           | gain              | frame 100.0%                                                   |
| animal | xgboost      | raw                | comp+log_scale | gain              | frame 95.5%, log_scale 4.5%                                    |
| bird   | extratrees   | emd                | comp           | impurity          | imf2 20.0%, imf1 18.4%, imf3 18.0%, imf4 15.8%, imf5 9.0%      |
| bird   | extratrees   | emd                | comp+log_scale | impurity          | imf2 19.5%, imf1 18.3%, imf3 18.0%, imf4 15.5%, imf5 9.1%      |
| bird   | extratrees   | imf_linear_a1.41   | comp           | impurity          | w65 9.2%, w93 8.2%, w47 6.9%, w129 6.4%, w17 6.2%              |
| bird   | extratrees   | imf_linear_a1.41   | comp+log_scale | impurity          | w65 9.2%, w93 8.4%, w47 6.6%, w129 6.2%, w25 6.2%              |
| bird   | extratrees   | irmf_h2_a1.41      | comp           | impurity          | w93 9.3%, w65 9.3%, w129 7.8%, w47 7.6%, w183 7.1%             |
| bird   | extratrees   | irmf_h2_a1.41      | comp+log_scale | impurity          | w65 9.5%, w93 9.3%, w129 7.9%, w47 7.5%, w25 6.7%              |
| bird   | extratrees   | irmf_h2_a1.41_conv | comp           | impurity          | w93 9.4%, w65 9.4%, w129 7.9%, w47 7.4%, w183 7.0%             |
| bird   | extratrees   | irmf_h2_a1.41_conv | comp+log_scale | impurity          | w93 9.4%, w65 9.2%, w129 8.0%, w47 7.0%, w183 6.8%             |
| bird   | extratrees   | raw                | comp           | impurity          | frame 100.0%                                                   |
| bird   | extratrees   | raw                | comp+log_scale | impurity          | frame 94.6%, log_scale 5.4%                                    |
| bird   | lightgbm     | emd                | comp           | split_count       | imf1 30.2%, imf2 23.8%, imf3 22.8%, imf4 14.3%, imf5 3.0%      |
| bird   | lightgbm     | emd                | comp+log_scale | split_count       | imf1 27.4%, imf2 22.0%, imf3 21.3%, imf4 13.3%, log_scale 7.7% |
| bird   | lightgbm     | imf_linear_a1.41   | comp           | split_count       | w9 11.3%, w65 11.1%, w17 8.3%, w7 7.6%, residual_fine 7.5%     |
| bird   | lightgbm     | imf_linear_a1.41   | comp+log_scale | split_count       | w65 10.9%, w9 10.9%, w17 7.7%, residual_fine 7.0%, w7 6.8%     |
| bird   | lightgbm     | irmf_h2_a1.41      | comp           | split_count       | w65 12.5%, w17 9.7%, w9 8.8%, w93 7.4%, w7 7.4%                |
| bird   | lightgbm     | irmf_h2_a1.41      | comp+log_scale | split_count       | w65 12.1%, w17 9.8%, w9 8.4%, w93 7.2%, residual_fine 7.1%     |
| bird   | lightgbm     | irmf_h2_a1.41_conv | comp           | split_count       | w65 11.4%, w17 9.7%, w9 8.2%, w7 7.8%, w93 7.7%                |
| bird   | lightgbm     | irmf_h2_a1.41_conv | comp+log_scale | split_count       | w65 11.5%, w17 9.7%, w9 7.8%, w93 7.8%, w7 7.0%                |
| bird   | lightgbm     | raw                | comp           | split_count       | frame 100.0%                                                   |
| bird   | lightgbm     | raw                | comp+log_scale | split_count       | frame 87.3%, log_scale 12.7%                                   |
| bird   | xgboost      | emd                | comp           | gain              | imf4 25.9%, imf3 21.2%, imf1 20.6%, imf2 16.6%, imf5 5.7%      |
| bird   | xgboost      | emd                | comp+log_scale | gain              | imf4 25.7%, imf1 20.9%, imf3 20.6%, imf2 17.1%, imf5 5.2%      |
| bird   | xgboost      | imf_linear_a1.41   | comp           | gain              | w9 13.0%, w65 12.2%, w47 7.4%, w25 6.7%, w17 6.0%              |
| bird   | xgboost      | imf_linear_a1.41   | comp+log_scale | gain              | w9 12.7%, w65 12.1%, w47 7.4%, w25 6.6%, w17 5.8%              |
| bird   | xgboost      | irmf_h2_a1.41      | comp           | gain              | w65 18.5%, w17 9.4%, w93 8.5%, w13 6.2%, w9 6.1%               |
| bird   | xgboost      | irmf_h2_a1.41      | comp+log_scale | gain              | w65 18.6%, w17 9.6%, w93 8.6%, w9 6.1%, w13 5.9%               |
| bird   | xgboost      | irmf_h2_a1.41_conv | comp           | gain              | w65 18.2%, w17 9.4%, w93 7.9%, w13 7.0%, w9 6.3%               |
| bird   | xgboost      | irmf_h2_a1.41_conv | comp+log_scale | gain              | w65 18.2%, w17 9.2%, w93 7.9%, w13 6.9%, w9 6.3%               |
| bird   | xgboost      | raw                | comp           | gain              | frame 100.0%                                                   |
| bird   | xgboost      | raw                | comp+log_scale | gain              | frame 96.2%, log_scale 3.8%                                    |
