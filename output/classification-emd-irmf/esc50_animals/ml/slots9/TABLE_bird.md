# slots9, bird

Percentages. References reuse identical main LightGBM fits.

|                                                      |   ('balanced_accuracy', 'mean') |   ('balanced_accuracy', 'std') |   ('macro_f1', 'mean') |   ('macro_f1', 'std') |   ('accuracy', 'mean') |   ('accuracy', 'std') |
|:-----------------------------------------------------|--------------------------------:|-------------------------------:|-----------------------:|----------------------:|-----------------------:|----------------------:|
| ('emd', 'comp', 'lightgbm')                          |                           75.81 |                           4.18 |                  73.80 |                  4.21 |                  76.48 |                  4.05 |
| ('emd', 'comp+log_scale', 'lightgbm')                |                           75.45 |                           4.12 |                  73.69 |                  4.35 |                  76.50 |                  4.28 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             |                           77.34 |                           4.89 |                  75.69 |                  4.48 |                  78.57 |                  3.89 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   |                           77.48 |                           5.32 |                  75.91 |                  4.82 |                  78.86 |                  4.10 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                |                           76.55 |                           3.78 |                  74.96 |                  3.57 |                  77.89 |                  3.27 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      |                           76.26 |                           3.78 |                  74.78 |                  3.52 |                  77.82 |                  3.18 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           |                           76.73 |                           3.83 |                  75.11 |                  3.63 |                  78.00 |                  3.32 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           76.59 |                           4.02 |                  74.99 |                  3.75 |                  77.93 |                  3.37 |
| ('raw', 'comp', 'lightgbm')                          |                           73.86 |                           4.89 |                  71.33 |                  4.72 |                  73.84 |                  4.56 |
| ('raw', 'comp+log_scale', 'lightgbm')                |                           73.19 |                           4.61 |                  70.96 |                  4.62 |                  73.66 |                  4.48 |

Nadeau-Bengio corrected t-test and CI, Wilcoxon secondary. Holm across all pairs and both feature sets per metric.
