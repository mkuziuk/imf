# sensitivity, animal: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 56 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                | -7.09 [-12.83, -1.35]     |  0.55347    |         0.0079346 |
| comp           | raw                | imf_linear_a1.41   | -14.81 [-19.89, -9.73]    |  0.0011489  |         0.003418  |
| comp           | raw                | irmf_h2_a1.41      | -13.95 [-19.35, -8.55]    |  0.003557   |         0.003418  |
| comp           | raw                | irmf_h2_a1.41_conv | -13.52 [-18.89, -8.15]    |  0.0043184  |         0.003418  |
| comp           | raw                | irmf_h1_a1.41_conv | -12.86 [-18.01, -7.71]    |  0.0045452  |         0.003418  |
| comp           | raw                | irmf_h4_a1.41_conv | -14.42 [-19.40, -9.43]    |  0.0012198  |         0.003418  |
| comp           | raw                | irmf_h2_a2_conv    | -12.51 [-18.12, -6.89]    |  0.01175    |         0.003418  |
| comp           | emd                | imf_linear_a1.41   | -7.72 [-10.87, -4.57]     |  0.0050515  |         0.003418  |
| comp           | emd                | irmf_h2_a1.41      | -6.86 [-10.42, -3.30]     |  0.035319   |         0.003418  |
| comp           | emd                | irmf_h2_a1.41_conv | -6.43 [-9.54, -3.31]      |  0.021379   |         0.003418  |
| comp           | emd                | irmf_h1_a1.41_conv | -5.78 [-9.06, -2.49]      |  0.066388   |         0.003418  |
| comp           | emd                | irmf_h4_a1.41_conv | -7.33 [-10.20, -4.46]     |  0.0039649  |         0.003418  |
| comp           | emd                | irmf_h2_a2_conv    | -5.42 [-8.04, -2.80]      |  0.021379   |         0.003418  |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41      | +0.86 [-0.34, +2.06]      |  1          |         0.093994  |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv | +1.29 [+0.02, +2.56]      |  1          |         0.003418  |
| comp           | imf_linear_a1.41   | irmf_h1_a1.41_conv | +1.94 [+0.51, +3.38]      |  0.34635    |         0.003418  |
| comp           | imf_linear_a1.41   | irmf_h4_a1.41_conv | +0.39 [-0.72, +1.50]      |  1          |         0.66602   |
| comp           | imf_linear_a1.41   | irmf_h2_a2_conv    | +2.30 [+1.08, +3.52]      |  0.042224   |         0.003418  |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.43 [-0.75, +1.62]      |  1          |         0.84381   |
| comp           | irmf_h2_a1.41      | irmf_h1_a1.41_conv | +1.09 [-0.04, +2.21]      |  1          |         0.025513  |
| comp           | irmf_h2_a1.41      | irmf_h4_a1.41_conv | -0.47 [-2.17, +1.23]      |  1          |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a2_conv    | +1.44 [-0.20, +3.08]      |  1          |         0.030518  |
| comp           | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv | +0.65 [-0.36, +1.67]      |  1          |         0.10254   |
| comp           | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv | -0.90 [-2.72, +0.92]      |  1          |         0.23016   |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +1.01 [-0.38, +2.40]      |  1          |         0.093994  |
| comp           | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv | -1.55 [-3.19, +0.08]      |  1          |         0.019653  |
| comp           | irmf_h1_a1.41_conv | irmf_h2_a2_conv    | +0.35 [-1.11, +1.82]      |  1          |         1         |
| comp           | irmf_h4_a1.41_conv | irmf_h2_a2_conv    | +1.91 [+0.61, +3.21]      |  0.22315    |         0.003418  |
| comp+log_scale | raw                | emd                | -6.12 [-11.07, -1.16]     |  0.55347    |         0.003418  |
| comp+log_scale | raw                | imf_linear_a1.41   | -13.90 [-18.87, -8.94]    |  0.0016677  |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a1.41      | -13.50 [-18.83, -8.17]    |  0.0041505  |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a1.41_conv | -12.82 [-18.18, -7.46]    |  0.0062964  |         0.003418  |
| comp+log_scale | raw                | irmf_h1_a1.41_conv | -12.58 [-17.66, -7.50]    |  0.0048332  |         0.003418  |
| comp+log_scale | raw                | irmf_h4_a1.41_conv | -13.73 [-18.96, -8.51]    |  0.003039   |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a2_conv    | -11.76 [-17.80, -5.71]    |  0.033741   |         0.003418  |
| comp+log_scale | emd                | imf_linear_a1.41   | -7.79 [-10.27, -5.31]     |  0.00053991 |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a1.41      | -7.39 [-10.15, -4.63]     |  0.0026185  |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a1.41_conv | -6.70 [-9.42, -3.98]      |  0.0050257  |         0.003418  |
| comp+log_scale | emd                | irmf_h1_a1.41_conv | -6.46 [-9.49, -3.43]      |  0.016851   |         0.003418  |
| comp+log_scale | emd                | irmf_h4_a1.41_conv | -7.62 [-10.10, -5.14]     |  0.00065782 |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a2_conv    | -5.64 [-8.68, -2.60]      |  0.045603   |         0.003418  |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41      | +0.40 [-0.94, +1.74]      |  1          |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv | +1.09 [-0.23, +2.41]      |  1          |         0.053711  |
| comp+log_scale | imf_linear_a1.41   | irmf_h1_a1.41_conv | +1.32 [-0.20, +2.84]      |  1          |         0.047241  |
| comp+log_scale | imf_linear_a1.41   | irmf_h4_a1.41_conv | +0.17 [-1.09, +1.43]      |  1          |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2_conv    | +2.15 [+0.22, +4.08]      |  0.79178    |         0.0049438 |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.69 [-0.13, +1.50]      |  1          |         0.025513  |
| comp+log_scale | irmf_h2_a1.41      | irmf_h1_a1.41_conv | +0.92 [-0.23, +2.08]      |  1          |         0.047241  |
| comp+log_scale | irmf_h2_a1.41      | irmf_h4_a1.41_conv | -0.23 [-1.57, +1.10]      |  1          |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2_conv    | +1.75 [+0.31, +3.18]      |  0.56034    |         0.0079346 |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv | +0.24 [-1.24, +1.72]      |  1          |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv | -0.92 [-1.95, +0.11]      |  1          |         0.030518  |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +1.06 [-0.57, +2.69]      |  1          |         0.093994  |
| comp+log_scale | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv | -1.15 [-2.70, +0.39]      |  1          |         0.080566  |
| comp+log_scale | irmf_h1_a1.41_conv | irmf_h2_a2_conv    | +0.82 [-0.40, +2.05]      |  1          |         0.093994  |
| comp+log_scale | irmf_h4_a1.41_conv | irmf_h2_a2_conv    | +1.98 [+0.31, +3.65]      |  0.60897    |         0.0079346 |
