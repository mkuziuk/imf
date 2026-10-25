# concat, animal: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 42 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b                | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:------------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                     | -7.09 [-12.83, -1.35]     |  0.40079    |         0.0057983 |
| comp           | raw                | imf_linear_a1.41        | -14.81 [-19.89, -9.73]    |  0.00085106 |         0.0025635 |
| comp           | raw                | irmf_h2_a1.41           | -13.95 [-19.35, -8.55]    |  0.0024681  |         0.0025635 |
| comp           | raw                | irmf_h2_a1.41_conv      | -13.52 [-18.89, -8.15]    |  0.0029102  |         0.0025635 |
| comp           | raw                | emd__irmf_h2_a1.41      | -13.64 [-18.48, -8.80]    |  0.0011545  |         0.0025635 |
| comp           | raw                | emd__irmf_h2_a1.41_conv | -13.25 [-17.92, -8.58]    |  0.0011023  |         0.0025635 |
| comp           | emd                | imf_linear_a1.41        | -7.72 [-10.87, -4.57]     |  0.0033894  |         0.0025635 |
| comp           | emd                | irmf_h2_a1.41           | -6.86 [-10.42, -3.30]     |  0.023209   |         0.0025635 |
| comp           | emd                | irmf_h2_a1.41_conv      | -6.43 [-9.54, -3.31]      |  0.013775   |         0.0025635 |
| comp           | emd                | emd__irmf_h2_a1.41      | -6.55 [-9.39, -3.72]      |  0.0052903  |         0.0025635 |
| comp           | emd                | emd__irmf_h2_a1.41_conv | -6.16 [-8.64, -3.68]      |  0.0031811  |         0.0025635 |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41           | +0.86 [-0.34, +2.06]      |  1          |         0.10071   |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv      | +1.29 [+0.02, +2.56]      |  0.89372    |         0.0025635 |
| comp           | imf_linear_a1.41   | emd__irmf_h2_a1.41      | +1.17 [+0.27, +2.06]      |  0.30853    |         0.0025635 |
| comp           | imf_linear_a1.41   | emd__irmf_h2_a1.41_conv | +1.56 [-0.12, +3.24]      |  1          |         0.034241  |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv      | +0.43 [-0.75, +1.62]      |  1          |         1         |
| comp           | irmf_h2_a1.41      | emd__irmf_h2_a1.41      | +0.31 [-0.83, +1.45]      |  1          |         1         |
| comp           | irmf_h2_a1.41      | emd__irmf_h2_a1.41_conv | +0.70 [-1.27, +2.67]      |  1          |         1         |
| comp           | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41      | -0.12 [-1.10, +0.85]      |  1          |         1         |
| comp           | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41_conv | +0.27 [-1.35, +1.88]      |  1          |         1         |
| comp           | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv | +0.39 [-0.82, +1.60]      |  1          |         0.91577   |
| comp+log_scale | raw                | emd                     | -6.12 [-11.07, -1.16]     |  0.40079    |         0.0025635 |
| comp+log_scale | raw                | imf_linear_a1.41        | -13.90 [-18.87, -8.94]    |  0.0011866  |         0.0025635 |
| comp+log_scale | raw                | irmf_h2_a1.41           | -13.50 [-18.83, -8.17]    |  0.0028258  |         0.0025635 |
| comp+log_scale | raw                | irmf_h2_a1.41_conv      | -12.82 [-18.18, -7.46]    |  0.0040679  |         0.0025635 |
| comp+log_scale | raw                | emd__irmf_h2_a1.41      | -13.60 [-18.89, -8.31]    |  0.0025123  |         0.0025635 |
| comp+log_scale | raw                | emd__irmf_h2_a1.41_conv | -13.00 [-18.43, -7.58]    |  0.0040679  |         0.0025635 |
| comp+log_scale | emd                | imf_linear_a1.41        | -7.79 [-10.27, -5.31]     |  0.00039529 |         0.0025635 |
| comp+log_scale | emd                | irmf_h2_a1.41           | -7.39 [-10.15, -4.63]     |  0.0018484  |         0.0025635 |
| comp+log_scale | emd                | irmf_h2_a1.41_conv      | -6.70 [-9.42, -3.98]      |  0.0033894  |         0.0025635 |
| comp+log_scale | emd                | emd__irmf_h2_a1.41      | -7.49 [-9.82, -5.15]      |  0.00032006 |         0.0025635 |
| comp+log_scale | emd                | emd__irmf_h2_a1.41_conv | -6.89 [-9.49, -4.29]      |  0.0019646  |         0.0025635 |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41           | +0.40 [-0.94, +1.74]      |  1          |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv      | +1.09 [-0.23, +2.41]      |  1          |         0.053711  |
| comp+log_scale | imf_linear_a1.41   | emd__irmf_h2_a1.41      | +0.30 [-0.82, +1.42]      |  1          |         1         |
| comp+log_scale | imf_linear_a1.41   | emd__irmf_h2_a1.41_conv | +0.90 [-0.82, +2.62]      |  1          |         0.39197   |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv      | +0.69 [-0.13, +1.50]      |  1          |         0.020874  |
| comp+log_scale | irmf_h2_a1.41      | emd__irmf_h2_a1.41      | -0.10 [-1.43, +1.23]      |  1          |         1         |
| comp+log_scale | irmf_h2_a1.41      | emd__irmf_h2_a1.41_conv | +0.50 [-0.77, +1.77]      |  1          |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41      | -0.79 [-2.12, +0.55]      |  1          |         0.10071   |
| comp+log_scale | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41_conv | -0.19 [-1.14, +0.76]      |  1          |         1         |
| comp+log_scale | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv | +0.60 [-0.71, +1.90]      |  1          |         0.66431   |
