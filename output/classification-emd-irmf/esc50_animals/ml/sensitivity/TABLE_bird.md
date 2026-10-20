# sensitivity, bird

Percentages. References reuse identical main LightGBM fits.

|                                                      |   ('balanced_accuracy', 'mean') |   ('balanced_accuracy', 'std') |   ('macro_f1', 'mean') |   ('macro_f1', 'std') |   ('accuracy', 'mean') |   ('accuracy', 'std') |
|:-----------------------------------------------------|--------------------------------:|-------------------------------:|-----------------------:|----------------------:|-----------------------:|----------------------:|
| ('emd', 'comp', 'lightgbm')                          |                           75.81 |                           4.18 |                  73.80 |                  4.21 |                  76.48 |                  4.05 |
| ('emd', 'comp+log_scale', 'lightgbm')                |                           75.45 |                           4.12 |                  73.69 |                  4.35 |                  76.50 |                  4.28 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             |                           78.21 |                           4.36 |                  76.75 |                  3.85 |                  79.66 |                  3.28 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   |                           78.35 |                           4.51 |                  76.82 |                  4.11 |                  79.66 |                  3.61 |
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')           |                           77.81 |                           3.07 |                  76.11 |                  2.82 |                  78.88 |                  2.69 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           77.82 |                           3.03 |                  76.21 |                  2.87 |                  79.01 |                  2.75 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                |                           78.07 |                           3.46 |                  76.53 |                  3.22 |                  79.35 |                  2.99 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      |                           77.65 |                           4.01 |                  76.10 |                  3.71 |                  78.97 |                  3.31 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           |                           77.88 |                           3.51 |                  76.33 |                  3.33 |                  79.15 |                  3.07 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           77.62 |                           3.69 |                  75.99 |                  3.38 |                  78.83 |                  3.04 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              |                           78.00 |                           3.74 |                  76.35 |                  3.61 |                  79.10 |                  3.42 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    |                           77.43 |                           3.91 |                  75.86 |                  3.54 |                  78.76 |                  3.26 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')           |                           77.79 |                           3.81 |                  76.30 |                  3.38 |                  79.22 |                  2.94 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           77.65 |                           4.18 |                  76.18 |                  3.88 |                  79.11 |                  3.40 |
| ('raw', 'comp', 'lightgbm')                          |                           73.86 |                           4.89 |                  71.33 |                  4.72 |                  73.84 |                  4.56 |
| ('raw', 'comp+log_scale', 'lightgbm')                |                           73.19 |                           4.61 |                  70.96 |                  4.62 |                  73.66 |                  4.48 |

Nadeau-Bengio corrected t-test and CI, Wilcoxon secondary. Holm across all pairs and both feature sets per metric.
