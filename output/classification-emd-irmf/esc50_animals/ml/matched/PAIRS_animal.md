# matched, animal: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 56 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                | -7.09 [-12.83, -1.35]     |  0.57255    |         0.0079346 |
| comp           | raw                | imf_linear_a1.41   | -14.81 [-19.89, -9.73]    |  0.0011489  |         0.003418  |
| comp           | raw                | irmf_h2_a1.41      | -13.95 [-19.35, -8.55]    |  0.003557   |         0.003418  |
| comp           | raw                | irmf_h2_a1.41_conv | -13.52 [-18.89, -8.15]    |  0.0043607  |         0.003418  |
| comp           | raw                | irmf_h2_a2_conv    | -12.51 [-18.12, -6.89]    |  0.012043   |         0.003418  |
| comp           | raw                | imf_linear_a2      | -12.89 [-17.46, -8.31]    |  0.0015663  |         0.003418  |
| comp           | raw                | irmf_h2_a2         | -13.13 [-18.89, -7.38]    |  0.0099674  |         0.003418  |
| comp           | emd                | imf_linear_a1.41   | -7.72 [-10.87, -4.57]     |  0.005292   |         0.003418  |
| comp           | emd                | irmf_h2_a1.41      | -6.86 [-10.42, -3.30]     |  0.034309   |         0.003418  |
| comp           | emd                | irmf_h2_a1.41_conv | -6.43 [-9.54, -3.31]      |  0.021942   |         0.003418  |
| comp           | emd                | irmf_h2_a2_conv    | -5.42 [-8.04, -2.80]      |  0.021942   |         0.003418  |
| comp           | emd                | imf_linear_a2      | -5.80 [-7.81, -3.79]      |  0.0012574  |         0.003418  |
| comp           | emd                | irmf_h2_a2         | -6.04 [-8.90, -3.18]      |  0.01875    |         0.003418  |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41      | +0.86 [-0.34, +2.06]      |  1          |         0.11414   |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv | +1.29 [+0.02, +2.56]      |  1          |         0.003418  |
| comp           | imf_linear_a1.41   | irmf_h2_a2_conv    | +2.30 [+1.08, +3.52]      |  0.040982   |         0.003418  |
| comp           | imf_linear_a1.41   | imf_linear_a2      | +1.92 [-0.22, +4.06]      |  1          |         0.05249   |
| comp           | imf_linear_a1.41   | irmf_h2_a2         | +1.67 [+0.35, +3.00]      |  0.5165     |         0.005127  |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.43 [-0.75, +1.62]      |  1          |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a2_conv    | +1.44 [-0.20, +3.08]      |  1          |         0.032043  |
| comp           | irmf_h2_a1.41      | imf_linear_a2      | +1.06 [-1.67, +3.79]      |  1          |         0.53638   |
| comp           | irmf_h2_a1.41      | irmf_h2_a2         | +0.82 [-0.84, +2.47]      |  1          |         0.57495   |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +1.01 [-0.38, +2.40]      |  1          |         0.11414   |
| comp           | irmf_h2_a1.41_conv | imf_linear_a2      | +0.63 [-1.67, +2.93]      |  1          |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2         | +0.38 [-1.35, +2.12]      |  1          |         1         |
| comp           | irmf_h2_a2_conv    | imf_linear_a2      | -0.38 [-2.16, +1.41]      |  1          |         1         |
| comp           | irmf_h2_a2_conv    | irmf_h2_a2         | -0.62 [-1.50, +0.25]      |  1          |         0.14355   |
| comp           | imf_linear_a2      | irmf_h2_a2         | -0.24 [-2.41, +1.92]      |  1          |         1         |
| comp+log_scale | raw                | emd                | -6.12 [-11.07, -1.16]     |  0.57255    |         0.003418  |
| comp+log_scale | raw                | imf_linear_a1.41   | -13.90 [-18.87, -8.94]    |  0.0016356  |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a1.41      | -13.50 [-18.83, -8.17]    |  0.0042388  |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a1.41_conv | -12.82 [-18.18, -7.46]    |  0.0066036  |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a2_conv    | -11.76 [-17.80, -5.71]    |  0.032804   |         0.003418  |
| comp+log_scale | raw                | imf_linear_a2      | -11.90 [-15.92, -7.88]    |  0.00098752 |         0.003418  |
| comp+log_scale | raw                | irmf_h2_a2         | -11.98 [-18.01, -5.95]    |  0.028561   |         0.003418  |
| comp+log_scale | emd                | imf_linear_a1.41   | -7.79 [-10.27, -5.31]     |  0.00053991 |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a1.41      | -7.39 [-10.15, -4.63]     |  0.0025672  |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a1.41_conv | -6.70 [-9.42, -3.98]      |  0.0052595  |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a2_conv    | -5.64 [-8.68, -2.60]      |  0.044221   |         0.003418  |
| comp+log_scale | emd                | imf_linear_a2      | -5.79 [-8.08, -3.49]      |  0.0043607  |         0.003418  |
| comp+log_scale | emd                | irmf_h2_a2         | -5.86 [-8.75, -2.98]      |  0.024111   |         0.003418  |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41      | +0.40 [-0.94, +1.74]      |  1          |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv | +1.09 [-0.23, +2.41]      |  1          |         0.063782  |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2_conv    | +2.15 [+0.22, +4.08]      |  0.85513    |         0.005127  |
| comp+log_scale | imf_linear_a1.41   | imf_linear_a2      | +2.00 [-0.04, +4.04]      |  1          |         0.019653  |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2         | +1.92 [-0.19, +4.04]      |  1          |         0.00354   |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.69 [-0.13, +1.50]      |  1          |         0.025513  |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2_conv    | +1.75 [+0.31, +3.18]      |  0.58109    |         0.0079346 |
| comp+log_scale | irmf_h2_a1.41      | imf_linear_a2      | +1.60 [-0.58, +3.78]      |  1          |         0.09668   |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2         | +1.52 [-0.20, +3.25]      |  1          |         0.0079346 |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +1.06 [-0.57, +2.69]      |  1          |         0.11414   |
| comp+log_scale | irmf_h2_a1.41_conv | imf_linear_a2      | +0.91 [-1.16, +2.99]      |  1          |         0.80298   |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2         | +0.84 [-1.06, +2.73]      |  1          |         0.83252   |
| comp+log_scale | irmf_h2_a2_conv    | imf_linear_a2      | -0.15 [-2.64, +2.35]      |  1          |         1         |
| comp+log_scale | irmf_h2_a2_conv    | irmf_h2_a2         | -0.22 [-1.20, +0.75]      |  1          |         1         |
| comp+log_scale | imf_linear_a2      | irmf_h2_a2         | -0.08 [-2.52, +2.36]      |  1          |         1         |
