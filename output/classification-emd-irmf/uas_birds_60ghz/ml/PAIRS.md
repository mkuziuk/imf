# Paired effects

A minus B, percentage points, individual corrected 95% intervals.
Holm covers 120 primary or 84 sensitivity tests per task/metric. Correction uses the mean observed pose-group test/train ratio.

### bird_drone, comp, compact_resnet

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +0.03 [-2.89, +2.94]  | 0.98398 |           1 |     0.19727  |                 1 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.23 [-3.08, +2.62]  | 0.86541 |           1 |     0.54895  |                 1 |
| irmf_h2_a1.41 - raw                     | primary           | -0.08 [-3.84, +3.68]  | 0.96429 |           1 |     0.82575  |                 1 |
| irmf_h2_a1.41_conv - emd                | primary           | -0.34 [-2.58, +1.90]  | 0.74981 |           1 |     0.45976  |                 1 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.37 [-2.59, +1.86]  | 0.72818 |           1 |     0.69824  |                 1 |
| imf_linear_a1.41 - emd                  | primary           | +0.26 [-3.08, +3.59]  | 0.87101 |           1 |     0.43206  |                 1 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +0.66 [-3.85, +5.17]  | 0.75953 |           1 |     0.69083  |                 1 |
| imf_linear_a2 - emd                     | sensitivity       | +1.46 [-1.95, +4.88]  | 0.3743  |           1 |     0.059284 |                 1 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.81 [-3.64, +2.03]  | 0.55148 |           1 |     0.10205  |                 1 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +1.00 [-2.63, +4.63]  | 0.56528 |           1 |     0.3396   |                 1 |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +1.20 [-1.42, +3.83]  | 0.34165 |           1 |     0.054932 |                 1 |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.55 [-1.87, +2.96]  | 0.63559 |           1 |     0.33105  |                 1 |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.46 [-2.52, +3.44]  | 0.74446 |           1 |     0.39941  |                 1 |

### bird_drone, comp, extratrees

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.82 [-0.19, +5.82]  | 0.063819 |           1 |   0.00048828 |          0.058594 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +1.14 [-0.31, +2.59]  | 0.11381  |           1 |   0.0012207  |          0.14038  |
| irmf_h2_a1.41 - raw                     | primary           | +4.57 [-0.78, +9.91]  | 0.088053 |           1 |   0.0012207  |          0.14038  |
| irmf_h2_a1.41_conv - emd                | primary           | +2.76 [-0.44, +5.97]  | 0.085804 |           1 |   0.0012207  |          0.14038  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.05 [-0.48, +0.38]  | 0.79966  |           1 |   1          |          1        |
| imf_linear_a1.41 - emd                  | primary           | +1.67 [-1.21, +4.56]  | 0.23354  |           1 |   0.030672   |          1        |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.37 [-0.50, +5.25]  | 0.098138 |           1 |   0.0028579  |          0.23149  |
| imf_linear_a2 - emd                     | sensitivity       | +1.62 [-1.35, +4.60]  | 0.26189  |           1 |   0.043518   |          1        |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.75 [-0.50, +2.01]  | 0.21938  |           1 |   0.017578   |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.39 [-1.53, +0.75]  | 0.47687  |           1 |   0.11133    |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.05 [-0.50, +0.40]  | 0.80669  |           1 |   0.8125     |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.41 [-0.42, +1.24]  | 0.30553  |           1 |   0.033203   |          1        |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.08 [-0.71, +0.56]  | 0.79911  |           1 |   0.44141    |          1        |

### bird_drone, comp, lightgbm

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.75 [+0.40, +5.11]  | 0.025167 |           1 |   0.00088336 |           0.10335 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +1.09 [-0.42, +2.60]  | 0.14342  |           1 |   0.0019531  |           0.21094 |
| irmf_h2_a1.41 - raw                     | primary           | +3.06 [-0.02, +6.13]  | 0.051034 |           1 |   0.0023236  |           0.24166 |
| irmf_h2_a1.41_conv - emd                | primary           | +2.89 [+0.37, +5.41]  | 0.027339 |           1 |   0.00098244 |           0.11396 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.14 [-0.32, +0.60]  | 0.52625  |           1 |   0.3125     |           1       |
| imf_linear_a1.41 - emd                  | primary           | +1.66 [-0.96, +4.29]  | 0.19516  |           1 |   0.015551   |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.70 [-0.17, +5.57]  | 0.063122 |           1 |   0.0019244  |           0.15973 |
| imf_linear_a2 - emd                     | sensitivity       | +2.13 [-0.03, +4.28]  | 0.05263  |           1 |   0.0018677  |           0.15689 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.57 [-0.95, +2.09]  | 0.43195  |           1 |   0.082031   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.19 [-0.81, +0.43]  | 0.52154  |           1 |   0.30469    |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.47 [-0.95, +1.88]  | 0.49204  |           1 |   0.23242    |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.08 [-1.14, +0.99]  | 0.87991  |           1 |   0.86133    |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.34 [-1.21, +0.54]  | 0.42142  |           1 |   0.11719    |           1       |

### bird_drone, comp, minirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +0.53 [-1.88, +2.95]  | 0.64162 |           1 |     0.31494  |                 1 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.11 [-1.96, +1.73]  | 0.89596 |           1 |     0.60791  |                 1 |
| irmf_h2_a1.41 - raw                     | primary           | +1.14 [-2.20, +4.47]  | 0.47743 |           1 |     0.089528 |                 1 |
| irmf_h2_a1.41_conv - emd                | primary           | +0.56 [-1.79, +2.91]  | 0.61754 |           1 |     0.32946  |                 1 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.02 [-1.06, +1.11]  | 0.96245 |           1 |     0.92188  |                 1 |
| imf_linear_a1.41 - emd                  | primary           | +0.65 [-1.37, +2.66]  | 0.50067 |           1 |     0.22139  |                 1 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +0.18 [-2.13, +2.49]  | 0.86903 |           1 |     0.47001  |                 1 |
| imf_linear_a2 - emd                     | sensitivity       | +0.47 [-1.97, +2.91]  | 0.68675 |           1 |     0.51315  |                 1 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.29 [-2.11, +1.54]  | 0.7401  |           1 |     0.61182  |                 1 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.38 [-1.60, +0.84]  | 0.51718 |           1 |     0.1958   |                 1 |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.18 [-1.65, +1.29]  | 0.79627 |           1 |     0.69629  |                 1 |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.11 [-1.47, +1.24]  | 0.86288 |           1 |     0.69434  |                 1 |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.22 [-1.07, +1.52]  | 0.71818 |           1 |     0.58984  |                 1 |

### bird_drone, comp, multirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | -0.17 [-2.95, +2.61]  | 0.89754 |           1 |    0.97733   |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.20 [-2.74, +2.33]  | 0.86478 |           1 |    0.74756   |           1       |
| irmf_h2_a1.41 - raw                     | primary           | +2.25 [-0.53, +5.03]  | 0.10454 |           1 |    0.0028487 |           0.29057 |
| irmf_h2_a1.41_conv - emd                | primary           | -0.76 [-2.96, +1.45]  | 0.47356 |           1 |    0.13976   |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.59 [-1.74, +0.56]  | 0.2927  |           1 |    0.041748  |           1       |
| imf_linear_a1.41 - emd                  | primary           | +0.03 [-2.78, +2.85]  | 0.97929 |           1 |    0.8423    |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | -0.58 [-2.92, +1.77]  | 0.60664 |           1 |    0.181     |           1       |
| imf_linear_a2 - emd                     | sensitivity       | -0.16 [-2.90, +2.57]  | 0.89997 |           1 |    0.75153   |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.41 [-2.44, +1.61]  | 0.66812 |           1 |    0.52965   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.18 [-0.90, +1.26]  | 0.72451 |           1 |    0.35083   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.20 [-1.60, +1.20]  | 0.7663  |           1 |    0.64819   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.01 [-2.39, +2.38]  | 0.99511 |           1 |    0.71094   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.68 [-1.59, +2.95]  | 0.53232 |           1 |    0.24878   |           1       |

### bird_drone, comp, xgboost

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.07 [-0.39, +4.54]  | 0.093044 |           1 |    0.0031127 |           0.31127 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.88 [-0.43, +2.18]  | 0.17186  |           1 |    0.0048828 |           0.45898 |
| irmf_h2_a1.41 - raw                     | primary           | +3.10 [-0.29, +6.49]  | 0.069688 |           1 |    0.001674  |           0.18581 |
| irmf_h2_a1.41_conv - emd                | primary           | +1.91 [-0.51, +4.32]  | 0.11268  |           1 |    0.0037372 |           0.36625 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.17 [-0.80, +0.47]  | 0.58414  |           1 |    0.32812   |           1       |
| imf_linear_a1.41 - emd                  | primary           | +1.20 [-0.87, +3.26]  | 0.23375  |           1 |    0.021846  |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.55 [-1.48, +4.57]  | 0.29089  |           1 |    0.056833  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +1.33 [-0.69, +3.34]  | 0.1803   |           1 |    0.015764  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.22 [-1.78, +2.23]  | 0.81571  |           1 |    0.6084    |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.36 [-1.61, +0.89]  | 0.54995  |           1 |    0.38574   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.13 [-1.02, +1.27]  | 0.8134   |           1 |    0.59375   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.04 [-1.87, +1.79]  | 0.96494  |           1 |    0.9873    |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.38 [-1.75, +1.00]  | 0.56409  |           1 |    0.27539   |           1       |

### bird_drone, comp+log_scale, compact_resnet

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +0.74 [-3.02, +4.49]  | 0.6803  |           1 |    0.48964   |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -1.41 [-5.17, +2.35]  | 0.4345  |           1 |    0.1574    |           1       |
| irmf_h2_a1.41 - raw                     | primary           | -1.00 [-4.69, +2.69]  | 0.57015 |           1 |    0.22067   |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +1.24 [-1.91, +4.40]  | 0.41263 |           1 |    0.088274  |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.51 [-2.99, +4.01]  | 0.76069 |           1 |    0.93433   |           1       |
| imf_linear_a1.41 - emd                  | primary           | +2.15 [-0.76, +5.05]  | 0.13564 |           1 |    0.0058662 |           0.54492 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.93 [-1.30, +5.16]  | 0.22067 |           1 |    0.015722  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +2.14 [-1.14, +5.42]  | 0.18364 |           1 |    0.010515  |           0.8307  |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.21 [-2.89, +2.47]  | 0.87003 |           1 |    0.82007   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.69 [-2.34, +3.71]  | 0.63357 |           1 |    0.31543   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.01 [-2.60, +2.59]  | 0.9955  |           1 |    0.87915   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.55 [-2.12, +3.22]  | 0.66616 |           1 |    0.36621   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.78 [-0.97, +2.52]  | 0.3553  |           1 |    0.060632  |           1       |

### bird_drone, comp+log_scale, extratrees

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.34 [-0.57, +5.25]  | 0.10699  |           1 |    0.0024414 |           0.25146 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.89 [-0.28, +2.07]  | 0.12564  |           1 |    0.0058594 |           0.54492 |
| irmf_h2_a1.41 - raw                     | primary           | +2.68 [-2.09, +7.45]  | 0.24772  |           1 |    0.018374  |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +2.45 [-0.47, +5.38]  | 0.093475 |           1 |    0.001709  |           0.18799 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.11 [-0.44, +0.67]  | 0.66559  |           1 |    0.4375    |           1       |
| imf_linear_a1.41 - emd                  | primary           | +1.45 [-1.29, +4.19]  | 0.27631  |           1 |    0.035119  |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.03 [-0.54, +4.60]  | 0.11234  |           1 |    0.0019531 |           0.16016 |
| imf_linear_a2 - emd                     | sensitivity       | +1.15 [-1.37, +3.68]  | 0.34432  |           1 |    0.082415  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.88 [-0.63, +2.39]  | 0.23229  |           1 |    0.012695  |           0.97754 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.42 [-1.12, +0.27]  | 0.21123  |           1 |    0.0078125 |           0.625   |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.30 [-1.06, +0.47]  | 0.4243   |           1 |    0.15625   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.02 [-0.86, +0.91]  | 0.96046  |           1 |    0.98535   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.28 [-1.14, +0.58]  | 0.49932  |           1 |    0.42383   |           1       |

### bird_drone, comp+log_scale, lightgbm

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +1.75 [-0.45, +3.96]  | 0.11049  |           1 |    0.0021355 |           0.22423 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.97 [-0.20, +2.14]  | 0.098037 |           1 |    0.0019531 |           0.21094 |
| irmf_h2_a1.41 - raw                     | primary           | +1.88 [-1.23, +4.99]  | 0.21537  |           1 |    0.0095215 |           0.84741 |
| irmf_h2_a1.41_conv - emd                | primary           | +1.83 [-0.37, +4.03]  | 0.095403 |           1 |    0.00161   |           0.18032 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.08 [-0.46, +0.62]  | 0.75599  |           1 |    0.8125    |           1       |
| imf_linear_a1.41 - emd                  | primary           | +0.78 [-1.49, +3.06]  | 0.47193  |           1 |    0.15137   |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.49 [-0.81, +3.80]  | 0.18699  |           1 |    0.010515  |           0.8307  |
| imf_linear_a2 - emd                     | sensitivity       | +1.04 [-0.97, +3.05]  | 0.28474  |           1 |    0.044396  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.45 [-1.24, +2.15]  | 0.57707  |           1 |    0.3125    |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.34 [-0.92, +0.24]  | 0.23132  |           1 |    0.035156  |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.26 [-0.98, +1.49]  | 0.66266  |           1 |    0.52344   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.03 [-1.10, +1.17]  | 0.94846  |           1 |    0.75      |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.40 [-1.32, +0.52]  | 0.36755  |           1 |    0.0625    |           1       |

### bird_drone, comp+log_scale, minirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +0.70 [-1.45, +2.86]  | 0.49381 |           1 |     0.14838  |                 1 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.20 [-1.44, +1.03]  | 0.72663 |           1 |     0.47754  |                 1 |
| irmf_h2_a1.41 - raw                     | primary           | +0.95 [-1.87, +3.76]  | 0.48219 |           1 |     0.067871 |                 1 |
| irmf_h2_a1.41_conv - emd                | primary           | +0.75 [-2.05, +3.54]  | 0.57608 |           1 |     0.27856  |                 1 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.04 [-1.00, +1.08]  | 0.93277 |           1 |     0.9375   |                 1 |
| imf_linear_a1.41 - emd                  | primary           | +0.91 [-0.82, +2.64]  | 0.27954 |           1 |     0.023193 |                 1 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +0.20 [-2.41, +2.80]  | 0.8728  |           1 |     0.36548  |                 1 |
| imf_linear_a2 - emd                     | sensitivity       | +0.24 [-2.81, +3.29]  | 0.86859 |           1 |     0.79807  |                 1 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.04 [-2.88, +2.80]  | 0.97534 |           1 |     0.84766  |                 1 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.55 [-1.94, +0.84]  | 0.41093 |           1 |     0.14062  |                 1 |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.67 [-2.96, +1.62]  | 0.54085 |           1 |     0.13184  |                 1 |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.32 [-1.46, +0.83]  | 0.56378 |           1 |     0.19141  |                 1 |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.08 [-1.27, +1.11]  | 0.89257 |           1 |     0.62109  |                 1 |

### bird_drone, comp+log_scale, multirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | -0.54 [-2.56, +1.48]  | 0.57674 |           1 |      0.20873 |                 1 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.35 [-2.49, +1.78]  | 0.72752 |           1 |      0.55053 |                 1 |
| irmf_h2_a1.41 - raw                     | primary           | +1.07 [-1.00, +3.14]  | 0.28707 |           1 |      0.03057 |                 1 |
| irmf_h2_a1.41_conv - emd                | primary           | -0.56 [-2.26, +1.13]  | 0.48805 |           1 |      0.18653 |                 1 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.02 [-1.19, +1.14]  | 0.96494 |           1 |      0.83545 |                 1 |
| imf_linear_a1.41 - emd                  | primary           | -0.18 [-2.05, +1.69]  | 0.83586 |           1 |      0.48584 |                 1 |
| irmf_h2_a2_conv - emd                   | sensitivity       | -0.30 [-2.55, +1.96]  | 0.78333 |           1 |      0.72925 |                 1 |
| imf_linear_a2 - emd                     | sensitivity       | -0.32 [-2.67, +2.04]  | 0.7781  |           1 |      0.71172 |                 1 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.02 [-2.42, +2.47]  | 0.98568 |           1 |      0.77698 |                 1 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.27 [-1.44, +1.97]  | 0.74134 |           1 |      0.23262 |                 1 |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.13 [-1.43, +1.16]  | 0.83011 |           1 |      0.46924 |                 1 |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.28 [-1.52, +2.07]  | 0.74498 |           1 |      0.59717 |                 1 |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.55 [-0.88, +1.98]  | 0.4242  |           1 |      0.11523 |                 1 |

### bird_drone, comp+log_scale, xgboost

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +1.40 [-0.51, +3.31]  | 0.13782 |           1 |    0.0068637 |           0.6246  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.90 [-0.41, +2.20]  | 0.16163 |           1 |    0.0039062 |           0.37556 |
| irmf_h2_a1.41 - raw                     | primary           | +1.46 [-1.37, +4.29]  | 0.28731 |           1 |    0.035419  |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +1.17 [-0.75, +3.10]  | 0.2114  |           1 |    0.014186  |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.23 [-0.64, +0.18]  | 0.25684 |           1 |    0.0625    |           1       |
| imf_linear_a1.41 - emd                  | primary           | +0.50 [-1.50, +2.50]  | 0.59959 |           1 |    0.24829   |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +0.94 [-1.29, +3.17]  | 0.38202 |           1 |    0.055413  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +0.76 [-0.93, +2.45]  | 0.35175 |           1 |    0.063777  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.18 [-1.75, +2.11]  | 0.84697 |           1 |    0.45459   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.24 [-1.49, +1.02]  | 0.69314 |           1 |    0.52588   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.26 [-0.87, +1.39]  | 0.62928 |           1 |    0.30469   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.08 [-1.76, +1.93]  | 0.92419 |           1 |    0.87744   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.25 [-1.63, +1.13]  | 0.70386 |           1 |    0.61328   |           1       |

### target3, comp, compact_resnet

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +5.42 [-0.21, +11.04] | 0.057963 |           1 |   0.0014665  |          0.14681  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.10 [-2.68, +2.89]  | 0.9372   |           1 |   0.93189    |          1        |
| irmf_h2_a1.41 - raw                     | primary           | +0.45 [-3.12, +4.03]  | 0.79042  |           1 |   0.67017    |          1        |
| irmf_h2_a1.41_conv - emd                | primary           | +5.00 [-0.40, +10.40] | 0.067004 |           1 |   0.00012207 |          0.014648 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.42 [-3.14, +2.31]  | 0.74768  |           1 |   0.35929    |          1        |
| imf_linear_a1.41 - emd                  | primary           | +5.31 [-1.24, +11.86] | 0.10395  |           1 |   0.0031308  |          0.28216  |
| irmf_h2_a2_conv - emd                   | sensitivity       | +6.60 [-1.54, +14.73] | 0.1038   |           1 |   0.0075861  |          0.55378  |
| imf_linear_a2 - emd                     | sensitivity       | +7.12 [+1.37, +12.87] | 0.018835 |           1 |   0.00012207 |          0.010254 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.52 [-5.68, +4.64]  | 0.83175  |           1 |   0.87817    |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +1.60 [-4.39, +7.59]  | 0.57643  |           1 |   0.16684    |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +1.81 [-0.78, +4.39]  | 0.15581  |           1 |   0.0096715  |          0.67701  |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.56 [-3.32, +4.43]  | 0.76295  |           1 |   0.36291    |          1        |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +1.15 [-3.42, +5.71]  | 0.59911  |           1 |   0.25532    |          1        |

### target3, comp, extratrees

| A - B                                   | inference_table   | difference [95% CI]   |      p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|----------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +3.54 [+0.78, +6.31]  | 0.015726  |     1       |   0.00079744 |          0.092503 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +1.01 [-1.13, +3.14]  | 0.32848   |     1       |   0.080078   |          1        |
| irmf_h2_a1.41 - raw                     | primary           | +6.88 [+2.06, +11.69] | 0.0083878 |     0.98976 |   0.00064839 |          0.076509 |
| irmf_h2_a1.41_conv - emd                | primary           | +3.02 [+0.40, +5.64]  | 0.026727  |     1       |   0.0010765  |          0.11411  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.52 [-1.58, +0.54]  | 0.3113    |     1       |   0.046875   |          1        |
| imf_linear_a1.41 - emd                  | primary           | +2.53 [-0.83, +5.90]  | 0.12814   |     1       |   0.0062542  |          0.48059  |
| irmf_h2_a2_conv - emd                   | sensitivity       | +3.16 [+0.63, +5.69]  | 0.018103  |     1       |   0.00088124 |          0.070523 |
| imf_linear_a2 - emd                     | sensitivity       | +2.40 [-0.78, +5.58]  | 0.12838   |     1       |   0.014234   |          0.93457  |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +0.76 [-1.43, +2.95]  | 0.46688   |     1       |   0.14258    |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.14 [-0.70, +0.98]  | 0.72802   |     1       |   0.51562    |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.14 [-1.34, +1.06]  | 0.80755   |     1       |   0.52344    |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.97 [-0.55, +2.49]  | 0.19172   |     1       |   0.0097656  |          0.67701  |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.14 [-1.69, +1.41]  | 0.85032   |     1       |   0.71094    |          1        |

### target3, comp, lightgbm

| A - B                                   | inference_table   | difference [95% CI]   |      p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|----------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.12 [-0.24, +4.48]  | 0.074757  |     1       |   0.0056704  |          0.44796  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +1.94 [-0.15, +4.04]  | 0.066166  |     1       |   0.0042101  |          0.36207  |
| irmf_h2_a1.41 - raw                     | primary           | +4.97 [+1.71, +8.22]  | 0.0056139 |     0.67366 |   0.00080135 |          0.092503 |
| irmf_h2_a1.41_conv - emd                | primary           | +2.26 [+0.51, +4.00]  | 0.014783  |     1       |   0.0012013  |          0.12374  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.14 [-1.17, +1.45]  | 0.82373   |     1       |   1          |          1        |
| imf_linear_a1.41 - emd                  | primary           | +0.17 [-1.85, +2.20]  | 0.85688   |     1       |   0.47778    |          1        |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.60 [+0.31, +2.89]  | 0.018882  |     1       |   0.0012013  |          0.094901 |
| imf_linear_a2 - emd                     | sensitivity       | +0.21 [-2.08, +2.50]  | 0.84814   |     1       |   0.70553    |          1        |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.39 [-1.01, +3.79]  | 0.23448   |     1       |   0.012207   |          0.81787  |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.66 [-1.99, +0.67]  | 0.30674   |     1       |   0.060547   |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.03 [-3.25, +3.32]  | 0.98222   |     1       |   0.9248     |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.66 [-0.36, +1.68]  | 0.18781   |     1       |   0.0058594  |          0.43945  |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -1.11 [-3.00, +0.77]  | 0.22687   |     1       |   0.023438   |          1        |

### target3, comp, minirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +3.09 [-0.82, +7.00]  | 0.11222 |           1 |    0.0019531 |           0.18555 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +0.31 [-1.40, +2.02]  | 0.70122 |           1 |    0.22754   |           1       |
| irmf_h2_a1.41 - raw                     | primary           | +1.84 [-0.56, +4.24]  | 0.12231 |           1 |    0.0019531 |           0.18555 |
| irmf_h2_a1.41_conv - emd                | primary           | +3.02 [-0.92, +6.97]  | 0.12275 |           1 |    0.0043378 |           0.36872 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.07 [-1.85, +1.71]  | 0.9344  |           1 |    0.68555   |           1       |
| imf_linear_a1.41 - emd                  | primary           | +2.78 [-0.68, +6.24]  | 0.10705 |           1 |    0.0031007 |           0.28216 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.84 [-2.37, +6.05]  | 0.36393 |           1 |    0.07337   |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +2.33 [-1.09, +5.74]  | 0.1657  |           1 |    0.0068359 |           0.50586 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.49 [-2.34, +1.36]  | 0.58177 |           1 |    0.33032   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -1.18 [-3.34, +0.98]  | 0.26053 |           1 |    0.016113  |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.45 [-1.97, +1.07]  | 0.53447 |           1 |    0.23071   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.14 [-2.05, +1.77]  | 0.87824 |           1 |    0.75586   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.49 [-1.45, +2.42]  | 0.59915 |           1 |    0.33203   |           1       |

### target3, comp, multirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.85 [-3.14, +8.83]  | 0.32501 |           1 |    0.032638  |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.90 [-4.44, +2.64]  | 0.59288 |           1 |    0.28125   |           1       |
| irmf_h2_a1.41 - raw                     | primary           | +1.84 [-2.09, +5.77]  | 0.33202 |           1 |    0.040927  |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +2.40 [-3.99, +8.78]  | 0.43433 |           1 |    0.12293   |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.45 [-2.41, +1.51]  | 0.62882 |           1 |    0.38257   |           1       |
| imf_linear_a1.41 - emd                  | primary           | +3.75 [-2.81, +10.31] | 0.24043 |           1 |    0.014359  |           0.97638 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.36 [-2.27, +7.00]  | 0.29298 |           1 |    0.015659  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +3.16 [-1.00, +7.32]  | 0.12525 |           1 |    0.0044736 |           0.34447 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.80 [-3.90, +2.30]  | 0.5892  |           1 |    0.090576  |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.03 [-2.84, +2.77]  | 0.97922 |           1 |    0.71973   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.59 [-5.84, +4.66]  | 0.81282 |           1 |    0.77724   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.69 [-2.11, +3.49]  | 0.60312 |           1 |    0.27975   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.80 [-2.03, +3.63]  | 0.55501 |           1 |    0.29251   |           1       |

### target3, comp, xgboost

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.12 [-1.35, +5.58]  | 0.21107  |           1 |   0.016881   |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +2.57 [-0.15, +5.29]  | 0.061917 |           1 |   0.0019163  |           0.18396 |
| irmf_h2_a1.41 - raw                     | primary           | +4.97 [+0.79, +9.14]  | 0.023007 |           1 |   0.00096462 |           0.10707 |
| irmf_h2_a1.41_conv - emd                | primary           | +1.91 [-1.82, +5.64]  | 0.29125  |           1 |   0.05333    |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.21 [-0.78, +0.36]  | 0.44823  |           1 |   0.1875     |           1       |
| imf_linear_a1.41 - emd                  | primary           | -0.45 [-2.91, +2.01]  | 0.69963  |           1 |   0.44879    |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.56 [-1.54, +4.66]  | 0.298    |           1 |   0.034417   |           1       |
| imf_linear_a2 - emd                     | sensitivity       | -0.07 [-3.21, +3.07]  | 0.96289  |           1 |   0.71899    |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.63 [-0.72, +3.98]  | 0.15894  |           1 |   0.01416    |           0.93457 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.35 [-2.37, +1.68]  | 0.71882  |           1 |   0.41846    |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.38 [-2.79, +3.56]  | 0.80025  |           1 |   0.67285    |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +1.04 [-0.96, +3.05]  | 0.2836   |           1 |   0.039062   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -1.60 [-3.91, +0.71]  | 0.16043  |           1 |   0.0078125  |           0.5625  |

### target3, comp+log_scale, compact_resnet

| A - B                                   | inference_table   | difference [95% CI]   |      p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|----------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +5.52 [+0.53, +10.51] | 0.032412  |     1       |   0.00146    |          0.14681  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.45 [-4.83, +3.93]  | 0.82815   |     1       |   0.44286    |          1        |
| irmf_h2_a1.41 - raw                     | primary           | +0.14 [-3.39, +3.67]  | 0.93396   |     1       |   0.94985    |          1        |
| irmf_h2_a1.41_conv - emd                | primary           | +5.07 [-1.44, +11.57] | 0.11684   |     1       |   0.0038455  |          0.33456  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.45 [-3.84, +2.94]  | 0.77918   |     1       |   0.39049    |          1        |
| imf_linear_a1.41 - emd                  | primary           | +5.97 [+0.04, +11.91] | 0.048747  |     1       |   0.0037372  |          0.33261  |
| irmf_h2_a2_conv - emd                   | sensitivity       | +5.94 [+0.74, +11.14] | 0.028041  |     1       |   0.0007994  |          0.065551 |
| imf_linear_a2 - emd                     | sensitivity       | +7.60 [+2.49, +12.72] | 0.0065724 |     0.55208 |   0.00065002 |          0.053952 |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -1.67 [-5.30, +1.96]  | 0.34162   |     1       |   0.077571   |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.87 [-4.88, +6.61]  | 0.75066   |     1       |   0.48953    |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +1.63 [-3.63, +6.89]  | 0.5164    |     1       |   0.25551    |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.62 [-3.74, +4.99]  | 0.76337   |     1       |   0.71138    |          1        |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.59 [-4.10, +5.29]  | 0.79137   |     1       |   0.36646    |          1        |

### target3, comp+log_scale, extratrees

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +3.47 [+0.48, +6.47]  | 0.026147 |           1 |   0.0011898  |          0.12374  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +1.08 [-0.58, +2.73]  | 0.18488  |           1 |   0.0058594  |          0.45703  |
| irmf_h2_a1.41 - raw                     | primary           | +4.65 [+0.95, +8.36]  | 0.017421 |           1 |   0.00098477 |          0.10707  |
| irmf_h2_a1.41_conv - emd                | primary           | +3.26 [+0.64, +5.88]  | 0.018258 |           1 |   0.0012905  |          0.13163  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.21 [-1.24, +0.82]  | 0.67134  |           1 |   0.40625    |          1        |
| imf_linear_a1.41 - emd                  | primary           | +2.40 [-0.90, +5.69]  | 0.14099  |           1 |   0.015297   |          1        |
| irmf_h2_a2_conv - emd                   | sensitivity       | +3.61 [+0.91, +6.32]  | 0.012535 |           1 |   0.00087066 |          0.070523 |
| imf_linear_a2 - emd                     | sensitivity       | +2.50 [-0.63, +5.63]  | 0.10868  |           1 |   0.0081491  |          0.57859  |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.11 [-0.95, +3.17]  | 0.26613  |           1 |   0.040728   |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | +0.35 [-0.83, +1.53]  | 0.53847  |           1 |   0.16406    |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.10 [-1.20, +1.41]  | 0.86669  |           1 |   0.73828    |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.45 [-1.80, +2.71]  | 0.67411  |           1 |   0.379      |          1        |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.38 [-1.69, +0.93]  | 0.54257  |           1 |   0.19629    |          1        |

### target3, comp+log_scale, lightgbm

| A - B                                   | inference_table   | difference [95% CI]   |     p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|---------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +1.74 [-0.82, +4.29]  | 0.16744  |           1 |   0.0087891  |          0.62402  |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +2.19 [+0.31, +4.06]  | 0.025379 |           1 |   0.00024414 |          0.029053 |
| irmf_h2_a1.41 - raw                     | primary           | +3.12 [-2.15, +8.40]  | 0.22497  |           1 |   0.011995   |          0.83964  |
| irmf_h2_a1.41_conv - emd                | primary           | +1.70 [-0.43, +3.83]  | 0.10918  |           1 |   0.0046387  |          0.37573  |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.03 [-1.03, +0.96]  | 0.94132  |           1 |   0.84375    |          1        |
| imf_linear_a1.41 - emd                  | primary           | -0.45 [-2.62, +1.71]  | 0.66153  |           1 |   0.31422    |          1        |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.01 [-0.65, +2.66]  | 0.21307  |           1 |   0.018307   |          1        |
| imf_linear_a2 - emd                     | sensitivity       | -0.14 [-2.71, +2.43]  | 0.90924  |           1 |   0.64795    |          1        |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.15 [-1.07, +3.37]  | 0.28705  |           1 |   0.017578   |          1        |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.69 [-2.13, +0.75]  | 0.31848  |           1 |   0.029297   |          1        |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.31 [-2.61, +3.23]  | 0.82177  |           1 |   0.63623    |          1        |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.87 [-0.24, +1.97]  | 0.11471  |           1 |   0.0039062  |          0.30469  |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -1.22 [-3.27, +0.84]  | 0.22529  |           1 |   0.017578   |          1        |

### target3, comp+log_scale, minirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.92 [-1.28, +7.12]  | 0.15844 |           1 |    0.0075857 |           0.55376 |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -0.03 [-1.87, +1.80]  | 0.96818 |           1 |    0.81152   |           1       |
| irmf_h2_a1.41 - raw                     | primary           | +1.84 [-1.92, +5.60]  | 0.31147 |           1 |    0.04834   |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +2.95 [-1.83, +7.73]  | 0.2064  |           1 |    0.013105  |           0.90427 |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.03 [-1.28, +1.35]  | 0.95555 |           1 |    0.94141   |           1       |
| imf_linear_a1.41 - emd                  | primary           | +2.95 [-1.10, +7.00]  | 0.1401  |           1 |    0.0069328 |           0.51303 |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.71 [-1.78, +7.20]  | 0.21649 |           1 |    0.019729  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +1.70 [-2.81, +6.21]  | 0.43182 |           1 |    0.093309  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.01 [-0.68, +2.70]  | 0.22186 |           1 |    0.01001   |           0.68066 |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.24 [-2.04, +1.55]  | 0.77553 |           1 |    0.64258   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -1.25 [-2.96, +0.46]  | 0.13954 |           1 |    0.0048828 |           0.37109 |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.03 [-1.57, +1.64]  | 0.96361 |           1 |    0.77832   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.14 [-2.06, +1.79]  | 0.87917 |           1 |    1         |           1       |

### target3, comp+log_scale, multirocket

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +2.19 [-4.37, +8.75]  | 0.48627 |           1 |    0.21191   |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | -1.60 [-3.55, +0.36]  | 0.10129 |           1 |    0.0027831 |           0.25604 |
| irmf_h2_a1.41 - raw                     | primary           | +0.73 [-2.76, +4.22]  | 0.66121 |           1 |    0.50488   |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +2.92 [-3.71, +9.54]  | 0.36092 |           1 |    0.063712  |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | +0.73 [-1.62, +3.08]  | 0.51683 |           1 |    0.035193  |           1       |
| imf_linear_a1.41 - emd                  | primary           | +3.78 [-3.26, +10.83] | 0.2687  |           1 |    0.02304   |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +2.64 [-3.97, +9.25]  | 0.4062  |           1 |    0.086426  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | +3.33 [-2.70, +9.37]  | 0.25612 |           1 |    0.021303  |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | -0.69 [-4.08, +2.69]  | 0.66666 |           1 |    0.2308    |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.28 [-2.93, +2.38]  | 0.82583 |           1 |    0.54891   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | -0.45 [-4.73, +3.83]  | 0.82429 |           1 |    0.53125   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +0.24 [-2.42, +2.91]  | 0.84788 |           1 |    0.72778   |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -0.28 [-3.47, +2.91]  | 0.85448 |           1 |    0.48889   |           1       |

### target3, comp+log_scale, xgboost

| A - B                                   | inference_table   | difference [95% CI]   |    p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:----------------------------------------|:------------------|:----------------------|--------:|------------:|-------------:|------------------:|
| irmf_h2_a1.41 - emd                     | primary           | +1.87 [-1.67, +5.42]  | 0.2752  |           1 |    0.032935  |           1       |
| irmf_h2_a1.41 - imf_linear_a1.41        | primary           | +2.57 [-0.52, +5.66]  | 0.09584 |           1 |    0.0044736 |           0.37131 |
| irmf_h2_a1.41 - raw                     | primary           | +2.64 [-3.06, +8.33]  | 0.33712 |           1 |    0.077907  |           1       |
| irmf_h2_a1.41_conv - emd                | primary           | +1.67 [-2.09, +5.43]  | 0.35803 |           1 |    0.077872  |           1       |
| irmf_h2_a1.41_conv - irmf_h2_a1.41      | primary           | -0.21 [-0.95, +0.54]  | 0.55792 |           1 |    0.39062   |           1       |
| imf_linear_a1.41 - emd                  | primary           | -0.69 [-3.49, +2.11]  | 0.60312 |           1 |    0.2672    |           1       |
| irmf_h2_a2_conv - emd                   | sensitivity       | +1.49 [-1.33, +4.32]  | 0.27557 |           1 |    0.030503  |           1       |
| imf_linear_a2 - emd                     | sensitivity       | -0.10 [-2.99, +2.78]  | 0.93937 |           1 |    0.8643    |           1       |
| irmf_h2_a2_conv - imf_linear_a2         | sensitivity       | +1.60 [-0.98, +4.17]  | 0.20502 |           1 |    0.02002   |           1       |
| irmf_h2_a2_conv - irmf_h2_a1.41_conv    | sensitivity       | -0.17 [-2.40, +2.05]  | 0.86957 |           1 |    0.68335   |           1       |
| imf_linear_a2 - imf_linear_a1.41        | sensitivity       | +0.59 [-2.56, +3.74]  | 0.69413 |           1 |    0.55103   |           1       |
| irmf_h1_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | +1.08 [-1.10, +3.25]  | 0.30591 |           1 |    0.039062  |           1       |
| irmf_h4_a1.41_conv - irmf_h2_a1.41_conv | sensitivity       | -1.56 [-4.22, +1.10]  | 0.22807 |           1 |    0.01709   |           1       |
