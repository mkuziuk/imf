# Tree importance by scale

Normalized per model, then averaged over folds. Descriptive, not causal. Finest IMF residual has rank zero. Split counts, gain and impurity differ.

| task       | classifier   | method             | feature_set    | importance_type   | top scales                                                        |
|:-----------|:-------------|:-------------------|:---------------|:------------------|:------------------------------------------------------------------|
| bird_drone | extratrees   | emd                | comp           | impurity          | imf4 22.3%, imf5 21.4%, imf1 18.8%, imf2 16.4%, imf3 9.2%         |
| bird_drone | extratrees   | emd                | comp+log_scale | impurity          | imf4 22.7%, imf5 20.6%, imf1 18.9%, imf2 16.2%, imf3 8.7%         |
| bird_drone | extratrees   | imf_linear_a1.41   | comp           | impurity          | w127 23.2%, w179 16.1%, w91 15.8%, w251 8.2%, w65 6.2%            |
| bird_drone | extratrees   | imf_linear_a1.41   | comp+log_scale | impurity          | w127 22.9%, w179 16.2%, w91 15.7%, w251 8.3%, w65 6.1%            |
| bird_drone | extratrees   | imf_linear_a2      | comp           | impurity          | w127 34.1%, w65 24.4%, residual_fine 10.0%, w251 9.9%, w5 8.2%    |
| bird_drone | extratrees   | imf_linear_a2      | comp+log_scale | impurity          | w127 33.9%, w65 24.1%, residual_fine 10.6%, w251 9.6%, w5 8.2%    |
| bird_drone | extratrees   | irmf_h1_a1.41_conv | comp           | impurity          | w127 23.0%, w179 14.8%, w91 12.2%, residual_fine 12.1%, w5 8.2%   |
| bird_drone | extratrees   | irmf_h1_a1.41_conv | comp+log_scale | impurity          | w127 23.0%, w179 14.3%, residual_fine 12.6%, w91 12.4%, w5 8.3%   |
| bird_drone | extratrees   | irmf_h2_a1.41      | comp           | impurity          | w127 23.8%, w179 15.9%, w91 13.9%, residual_fine 9.5%, w251 8.5%  |
| bird_drone | extratrees   | irmf_h2_a1.41      | comp+log_scale | impurity          | w127 23.5%, w179 15.3%, w91 14.0%, residual_fine 10.2%, w251 8.4% |
| bird_drone | extratrees   | irmf_h2_a1.41_conv | comp           | impurity          | w127 23.3%, w179 15.7%, w91 13.5%, residual_fine 9.7%, w251 8.3%  |
| bird_drone | extratrees   | irmf_h2_a1.41_conv | comp+log_scale | impurity          | w127 23.1%, w179 14.8%, w91 13.7%, residual_fine 10.3%, w251 8.2% |
| bird_drone | extratrees   | irmf_h2_a2_conv    | comp           | impurity          | w127 33.3%, w65 18.9%, residual_fine 18.3%, w251 9.8%, w5 6.8%    |
| bird_drone | extratrees   | irmf_h2_a2_conv    | comp+log_scale | impurity          | w127 33.1%, w65 19.6%, residual_fine 18.3%, w251 9.3%, w5 6.7%    |
| bird_drone | extratrees   | irmf_h4_a1.41_conv | comp           | impurity          | w127 23.2%, w179 16.5%, w91 14.9%, w251 8.5%, residual_fine 7.7%  |
| bird_drone | extratrees   | irmf_h4_a1.41_conv | comp+log_scale | impurity          | w127 23.0%, w179 16.4%, w91 14.7%, w251 8.5%, residual_fine 8.1%  |
| bird_drone | extratrees   | raw                | comp           | impurity          | frame 100.0%                                                      |
| bird_drone | extratrees   | raw                | comp+log_scale | impurity          | frame 92.5%, log_scale 7.5%                                       |
| bird_drone | lightgbm     | emd                | comp           | split_count       | imf4 23.4%, imf1 22.4%, imf5 21.1%, imf2 18.2%, imf3 5.0%         |
| bird_drone | lightgbm     | emd                | comp+log_scale | split_count       | imf4 23.3%, imf1 18.7%, imf5 18.0%, imf2 15.5%, log_scale 8.6%    |
| bird_drone | lightgbm     | imf_linear_a1.41   | comp           | split_count       | residual_fine 21.9%, w127 17.7%, w91 8.6%, w65 8.5%, w5 8.1%      |
| bird_drone | lightgbm     | imf_linear_a1.41   | comp+log_scale | split_count       | residual_fine 21.5%, w127 16.8%, w65 8.8%, w91 8.5%, w5 7.6%      |
| bird_drone | lightgbm     | imf_linear_a2      | comp           | split_count       | residual_fine 26.9%, w65 21.0%, w127 19.2%, w5 11.0%, w251 6.3%   |
| bird_drone | lightgbm     | imf_linear_a2      | comp+log_scale | split_count       | residual_fine 25.7%, w65 19.7%, w127 19.2%, w5 10.6%, w251 6.5%   |
| bird_drone | lightgbm     | irmf_h1_a1.41_conv | comp           | split_count       | residual_fine 25.7%, w127 15.3%, w91 9.6%, w5 8.9%, w179 8.6%     |
| bird_drone | lightgbm     | irmf_h1_a1.41_conv | comp+log_scale | split_count       | residual_fine 25.4%, w127 15.0%, w91 9.9%, w179 8.2%, w5 8.0%     |
| bird_drone | lightgbm     | irmf_h2_a1.41      | comp           | split_count       | residual_fine 25.1%, w127 14.7%, w91 9.6%, w179 8.5%, w65 7.1%    |
| bird_drone | lightgbm     | irmf_h2_a1.41      | comp+log_scale | split_count       | residual_fine 24.8%, w127 14.7%, w91 9.9%, w179 8.3%, w65 6.8%    |
| bird_drone | lightgbm     | irmf_h2_a1.41_conv | comp           | split_count       | residual_fine 22.9%, w127 14.4%, w91 9.3%, w179 9.1%, w5 8.6%     |
| bird_drone | lightgbm     | irmf_h2_a1.41_conv | comp+log_scale | split_count       | residual_fine 23.0%, w127 14.3%, w91 9.4%, w179 8.9%, w5 8.2%     |
| bird_drone | lightgbm     | irmf_h2_a2_conv    | comp           | split_count       | residual_fine 32.0%, w65 18.9%, w127 18.9%, w5 6.7%, w33 6.2%     |
| bird_drone | lightgbm     | irmf_h2_a2_conv    | comp+log_scale | split_count       | residual_fine 31.0%, w65 18.6%, w127 18.1%, w5 6.8%, w33 6.1%     |
| bird_drone | lightgbm     | irmf_h4_a1.41_conv | comp           | split_count       | residual_fine 23.0%, w127 15.4%, w91 10.3%, w65 8.4%, w5 7.1%     |
| bird_drone | lightgbm     | irmf_h4_a1.41_conv | comp+log_scale | split_count       | residual_fine 22.6%, w127 15.6%, w91 10.7%, w65 7.8%, w5 6.8%     |
| bird_drone | lightgbm     | raw                | comp           | split_count       | frame 100.0%                                                      |
| bird_drone | lightgbm     | raw                | comp+log_scale | split_count       | frame 81.2%, log_scale 18.8%                                      |
| bird_drone | xgboost      | emd                | comp           | gain              | imf4 53.6%, imf5 16.1%, imf2 15.5%, imf1 8.9%, imf3 3.5%          |
| bird_drone | xgboost      | emd                | comp+log_scale | gain              | imf4 54.9%, imf2 15.2%, imf5 15.0%, imf1 8.1%, log_scale 3.2%     |
| bird_drone | xgboost      | imf_linear_a1.41   | comp           | gain              | w127 38.6%, w179 25.1%, residual_fine 8.8%, w65 5.4%, w91 4.4%    |
| bird_drone | xgboost      | imf_linear_a1.41   | comp+log_scale | gain              | w127 36.5%, w179 25.9%, residual_fine 8.5%, w65 5.4%, w5 4.4%     |
| bird_drone | xgboost      | imf_linear_a2      | comp           | gain              | w127 47.4%, w65 26.0%, residual_fine 10.9%, w5 5.8%, w251 3.3%    |
| bird_drone | xgboost      | imf_linear_a2      | comp+log_scale | gain              | w127 45.5%, w65 27.2%, residual_fine 10.9%, w5 5.7%, w251 3.3%    |
| bird_drone | xgboost      | irmf_h1_a1.41_conv | comp           | gain              | w127 54.2%, w179 18.0%, residual_fine 9.2%, w91 4.5%, w5 4.5%     |
| bird_drone | xgboost      | irmf_h1_a1.41_conv | comp+log_scale | gain              | w127 53.9%, w179 17.9%, residual_fine 9.2%, w5 4.4%, w91 4.3%     |
| bird_drone | xgboost      | irmf_h2_a1.41      | comp           | gain              | w127 40.5%, w179 28.4%, residual_fine 9.3%, w91 5.1%, w65 4.5%    |
| bird_drone | xgboost      | irmf_h2_a1.41      | comp+log_scale | gain              | w127 40.0%, w179 28.1%, residual_fine 9.2%, w91 5.2%, w65 4.4%    |
| bird_drone | xgboost      | irmf_h2_a1.41_conv | comp           | gain              | w127 41.2%, w179 26.9%, residual_fine 9.1%, w5 5.0%, w91 4.8%     |
| bird_drone | xgboost      | irmf_h2_a1.41_conv | comp+log_scale | gain              | w127 40.9%, w179 27.1%, residual_fine 9.0%, w5 5.2%, w91 4.7%     |
| bird_drone | xgboost      | irmf_h2_a2_conv    | comp           | gain              | w127 64.2%, residual_fine 14.5%, w65 11.2%, w5 4.3%, w33 2.0%     |
| bird_drone | xgboost      | irmf_h2_a2_conv    | comp+log_scale | gain              | w127 64.1%, residual_fine 13.8%, w65 10.7%, w5 4.3%, w33 2.1%     |
| bird_drone | xgboost      | irmf_h4_a1.41_conv | comp           | gain              | w127 44.4%, w179 19.0%, residual_fine 9.3%, w91 9.0%, w65 6.3%    |
| bird_drone | xgboost      | irmf_h4_a1.41_conv | comp+log_scale | gain              | w127 44.2%, w179 18.2%, residual_fine 9.4%, w91 8.4%, w65 6.2%    |
| bird_drone | xgboost      | raw                | comp           | gain              | frame 100.0%                                                      |
| bird_drone | xgboost      | raw                | comp+log_scale | gain              | frame 96.2%, log_scale 3.8%                                       |
| target3    | extratrees   | emd                | comp           | impurity          | imf1 23.7%, imf4 22.5%, imf5 19.7%, imf2 13.2%, imf3 9.4%         |
| target3    | extratrees   | emd                | comp+log_scale | impurity          | imf1 23.7%, imf4 22.5%, imf5 19.1%, imf2 13.0%, imf3 9.0%         |
| target3    | extratrees   | imf_linear_a1.41   | comp           | impurity          | w127 18.0%, w91 15.0%, w179 12.6%, w65 7.4%, w251 6.7%            |
| target3    | extratrees   | imf_linear_a1.41   | comp+log_scale | impurity          | w127 17.8%, w91 15.1%, w179 12.4%, w65 7.5%, w251 6.8%            |
| target3    | extratrees   | imf_linear_a2      | comp           | impurity          | w127 26.6%, w65 22.8%, w251 11.4%, residual_fine 11.1%, w5 8.6%   |
| target3    | extratrees   | imf_linear_a2      | comp+log_scale | impurity          | w127 26.3%, w65 23.1%, residual_fine 11.4%, w251 10.7%, w5 8.5%   |
| target3    | extratrees   | irmf_h1_a1.41_conv | comp           | impurity          | w127 18.1%, w91 13.0%, residual_fine 12.3%, w179 11.2%, w5 8.1%   |
| target3    | extratrees   | irmf_h1_a1.41_conv | comp+log_scale | impurity          | w127 18.4%, w91 13.4%, residual_fine 12.3%, w179 11.0%, w5 8.3%   |
| target3    | extratrees   | irmf_h2_a1.41      | comp           | impurity          | w127 18.4%, w91 14.2%, w179 12.4%, residual_fine 9.4%, w251 6.9%  |
| target3    | extratrees   | irmf_h2_a1.41      | comp+log_scale | impurity          | w127 18.9%, w91 14.3%, w179 12.4%, residual_fine 9.3%, w251 6.9%  |
| target3    | extratrees   | irmf_h2_a1.41_conv | comp           | impurity          | w127 18.2%, w91 13.9%, w179 12.3%, residual_fine 9.4%, w5 6.8%    |
| target3    | extratrees   | irmf_h2_a1.41_conv | comp+log_scale | impurity          | w127 18.5%, w91 14.0%, w179 12.0%, residual_fine 9.5%, w5 6.8%    |
| target3    | extratrees   | irmf_h2_a2_conv    | comp           | impurity          | w127 26.3%, w65 19.6%, residual_fine 17.9%, w251 10.0%, w5 7.2%   |
| target3    | extratrees   | irmf_h2_a2_conv    | comp+log_scale | impurity          | w127 26.5%, w65 19.7%, residual_fine 17.8%, w251 9.7%, w5 7.1%    |
| target3    | extratrees   | irmf_h4_a1.41_conv | comp           | impurity          | w127 17.9%, w91 14.6%, w179 12.8%, residual_fine 7.4%, w65 6.9%   |
| target3    | extratrees   | irmf_h4_a1.41_conv | comp+log_scale | impurity          | w127 18.1%, w91 14.7%, w179 12.7%, residual_fine 7.7%, w65 6.8%   |
| target3    | extratrees   | raw                | comp           | impurity          | frame 100.0%                                                      |
| target3    | extratrees   | raw                | comp+log_scale | impurity          | frame 94.1%, log_scale 5.9%                                       |
| target3    | lightgbm     | emd                | comp           | split_count       | imf1 26.0%, imf5 18.9%, imf4 18.9%, imf2 15.8%, imf3 6.9%         |
| target3    | lightgbm     | emd                | comp+log_scale | split_count       | imf1 24.0%, imf4 19.3%, imf5 17.1%, imf2 14.5%, imf3 7.1%         |
| target3    | lightgbm     | imf_linear_a1.41   | comp           | split_count       | w91 14.5%, w127 11.8%, residual_fine 11.6%, w5 7.2%, w355 6.5%    |
| target3    | lightgbm     | imf_linear_a1.41   | comp+log_scale | split_count       | w91 14.4%, w127 11.7%, residual_fine 11.2%, w5 7.2%, w355 6.5%    |
| target3    | lightgbm     | imf_linear_a2      | comp           | split_count       | w65 22.7%, residual_fine 17.7%, w127 14.9%, w17 8.8%, w251 8.7%   |
| target3    | lightgbm     | imf_linear_a2      | comp+log_scale | split_count       | w65 22.0%, residual_fine 17.1%, w127 14.4%, w17 9.1%, w251 8.6%   |
| target3    | lightgbm     | irmf_h1_a1.41_conv | comp           | split_count       | residual_fine 15.1%, w91 13.4%, w127 9.1%, w17 8.3%, w5 8.0%      |
| target3    | lightgbm     | irmf_h1_a1.41_conv | comp+log_scale | split_count       | residual_fine 15.2%, w91 13.4%, w127 9.1%, w17 8.3%, w5 7.8%      |
| target3    | lightgbm     | irmf_h2_a1.41      | comp           | split_count       | w91 15.4%, residual_fine 14.7%, w127 9.1%, w33 7.4%, w5 6.6%      |
| target3    | lightgbm     | irmf_h2_a1.41      | comp+log_scale | split_count       | w91 15.6%, residual_fine 14.6%, w127 9.0%, w33 7.3%, w5 6.6%      |
| target3    | lightgbm     | irmf_h2_a1.41_conv | comp           | split_count       | w91 15.5%, residual_fine 13.5%, w127 9.4%, w5 7.7%, w33 7.0%      |
| target3    | lightgbm     | irmf_h2_a1.41_conv | comp+log_scale | split_count       | w91 15.9%, residual_fine 13.4%, w127 9.4%, w5 7.7%, w33 6.9%      |
| target3    | lightgbm     | irmf_h2_a2_conv    | comp           | split_count       | w65 19.9%, residual_fine 19.6%, w127 12.8%, w17 12.7%, w33 8.1%   |
| target3    | lightgbm     | irmf_h2_a2_conv    | comp+log_scale | split_count       | w65 20.2%, residual_fine 19.4%, w17 12.6%, w127 12.6%, w33 7.8%   |
| target3    | lightgbm     | irmf_h4_a1.41_conv | comp           | split_count       | w91 15.3%, residual_fine 11.9%, w127 9.7%, w5 7.1%, w33 6.8%      |
| target3    | lightgbm     | irmf_h4_a1.41_conv | comp+log_scale | split_count       | w91 15.5%, residual_fine 11.8%, w127 9.7%, w5 6.9%, w33 6.9%      |
| target3    | lightgbm     | raw                | comp           | split_count       | frame 100.0%                                                      |
| target3    | lightgbm     | raw                | comp+log_scale | split_count       | frame 82.8%, log_scale 17.2%                                      |
| target3    | xgboost      | emd                | comp           | gain              | imf4 24.3%, imf1 21.2%, imf5 19.4%, imf2 12.0%, imf3 10.7%        |
| target3    | xgboost      | emd                | comp+log_scale | gain              | imf4 24.1%, imf1 21.4%, imf5 20.0%, imf2 12.2%, imf3 9.6%         |
| target3    | xgboost      | imf_linear_a1.41   | comp           | gain              | w91 21.2%, w127 19.0%, w179 9.3%, w355 9.1%, w5 7.9%              |
| target3    | xgboost      | imf_linear_a1.41   | comp+log_scale | gain              | w91 21.2%, w127 18.7%, w179 9.4%, w355 8.8%, w5 7.9%              |
| target3    | xgboost      | imf_linear_a2      | comp           | gain              | w127 36.7%, w65 26.1%, residual_fine 8.5%, w251 7.6%, w5 7.0%     |
| target3    | xgboost      | imf_linear_a2      | comp+log_scale | gain              | w127 36.7%, w65 25.6%, residual_fine 8.6%, w251 7.6%, w5 7.1%     |
| target3    | xgboost      | irmf_h1_a1.41_conv | comp           | gain              | w127 25.8%, w91 23.8%, w179 12.6%, residual_fine 8.5%, w5 4.9%    |
| target3    | xgboost      | irmf_h1_a1.41_conv | comp+log_scale | gain              | w127 25.8%, w91 24.2%, w179 12.5%, residual_fine 8.5%, w5 4.8%    |
| target3    | xgboost      | irmf_h2_a1.41      | comp           | gain              | w91 25.2%, w179 22.0%, w127 15.8%, residual_fine 5.6%, w5 5.3%    |
| target3    | xgboost      | irmf_h2_a1.41      | comp+log_scale | gain              | w91 25.4%, w179 21.2%, w127 15.7%, residual_fine 5.7%, w5 5.5%    |
| target3    | xgboost      | irmf_h2_a1.41_conv | comp           | gain              | w91 25.0%, w179 21.9%, w127 16.2%, w5 5.6%, residual_fine 5.2%    |
| target3    | xgboost      | irmf_h2_a1.41_conv | comp+log_scale | gain              | w91 24.9%, w179 22.0%, w127 16.2%, w5 5.5%, residual_fine 5.1%    |
| target3    | xgboost      | irmf_h2_a2_conv    | comp           | gain              | w127 37.0%, w65 21.9%, residual_fine 11.6%, w17 8.4%, w5 6.2%     |
| target3    | xgboost      | irmf_h2_a2_conv    | comp+log_scale | gain              | w127 36.7%, w65 21.9%, residual_fine 11.6%, w17 8.5%, w5 6.2%     |
| target3    | xgboost      | irmf_h4_a1.41_conv | comp           | gain              | w91 24.4%, w127 17.3%, w179 13.1%, w355 7.5%, w5 7.2%             |
| target3    | xgboost      | irmf_h4_a1.41_conv | comp+log_scale | gain              | w91 24.1%, w127 17.3%, w179 12.9%, w355 7.5%, w5 7.3%             |
| target3    | xgboost      | raw                | comp           | gain              | frame 100.0%                                                      |
| target3    | xgboost      | raw                | comp+log_scale | gain              | frame 96.2%, log_scale 3.8%                                       |
