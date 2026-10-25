# Primary-task ablations

Balanced accuracy; differences in percentage points. Individual Nadeau–Bengio 95% intervals, with Holm p from the complete 100-test main table.

## Solver convergence

| classifier     | feature_set    | contrast          | effect [95% CI]; Holm p         |
|:---------------|:---------------|:------------------|:--------------------------------|
| lightgbm       | comp           | Conv minus linear | -4.83 [-6.18, -3.47]; 0.0001551 |
| lightgbm       | comp+log_scale | Conv minus linear | -2.35 [-3.16, -1.53]; 0.001275  |
| extratrees     | comp           | Conv minus linear | -5.16 [-6.30, -4.03]; 9.581e-06 |
| extratrees     | comp+log_scale | Conv minus linear | -4.07 [-5.41, -2.73]; 0.0007679 |
| xgboost        | comp           | Conv minus linear | -5.46 [-7.12, -3.80]; 0.0003596 |
| xgboost        | comp+log_scale | Conv minus linear | -3.26 [-4.54, -1.98]; 0.003899  |
| minirocket     | comp           | Conv minus linear | -2.20 [-3.68, -0.71]; 0.348     |
| minirocket     | comp+log_scale | Conv minus linear | -1.05 [-2.73, +0.63]; 1         |
| compact_resnet | comp           | Conv minus linear | -1.26 [-4.12, +1.61]; 1         |
| compact_resnet | comp+log_scale | Conv minus linear | -0.81 [-3.05, +1.44]; 1         |
| lightgbm       | comp           | Conv minus GD60   | +0.35 [-0.40, +1.09]; 1         |
| lightgbm       | comp+log_scale | Conv minus GD60   | +0.12 [-0.34, +0.58]; 1         |
| extratrees     | comp           | Conv minus GD60   | +0.15 [-0.64, +0.94]; 1         |
| extratrees     | comp+log_scale | Conv minus GD60   | +0.62 [-0.28, +1.52]; 1         |
| xgboost        | comp           | Conv minus GD60   | +0.41 [-0.34, +1.16]; 1         |
| xgboost        | comp+log_scale | Conv minus GD60   | +0.38 [-0.22, +0.97]; 1         |
| minirocket     | comp           | Conv minus GD60   | +0.28 [-0.11, +0.66]; 1         |
| minirocket     | comp+log_scale | Conv minus GD60   | +0.13 [-0.67, +0.94]; 1         |
| compact_resnet | comp           | Conv minus GD60   | -1.10 [-2.84, +0.65]; 1         |
| compact_resnet | comp+log_scale | Conv minus GD60   | -0.31 [-3.37, +2.74]; 1         |

Scores on the same GD60-converged/unconverged frame populations, pooled per seed and then averaged over available seeds. The 1,997 unconverged frames and 10,003 converged frames each contain all ten classes. These are descriptive associations; solver failure does not randomly assign frames to difficulty strata.

| classifier     | feature_set    | method             | gd60_all_converged   |   n_frames |   BA % |
|:---------------|:---------------|:-------------------|:---------------------|-----------:|-------:|
| compact_resnet | comp           | imf_linear_a1.41   | False                |       1997 |  28.56 |
| compact_resnet | comp           | imf_linear_a1.41   | True                 |      10003 |  30.54 |
| compact_resnet | comp           | irmf_h2_a1.41      | False                |       1997 |  30.27 |
| compact_resnet | comp           | irmf_h2_a1.41      | True                 |      10003 |  29.93 |
| compact_resnet | comp           | irmf_h2_a1.41_conv | False                |       1997 |  25.37 |
| compact_resnet | comp           | irmf_h2_a1.41_conv | True                 |      10003 |  29.43 |
| compact_resnet | comp+log_scale | imf_linear_a1.41   | False                |       1997 |  29.47 |
| compact_resnet | comp+log_scale | imf_linear_a1.41   | True                 |      10003 |  34.31 |
| compact_resnet | comp+log_scale | irmf_h2_a1.41      | False                |       1997 |  30.68 |
| compact_resnet | comp+log_scale | irmf_h2_a1.41      | True                 |      10003 |  33.23 |
| compact_resnet | comp+log_scale | irmf_h2_a1.41_conv | False                |       1997 |  27.69 |
| compact_resnet | comp+log_scale | irmf_h2_a1.41_conv | True                 |      10003 |  33.73 |
| extratrees     | comp           | imf_linear_a1.41   | False                |       1997 |  61.54 |
| extratrees     | comp           | imf_linear_a1.41   | True                 |      10003 |  67.85 |
| extratrees     | comp           | irmf_h2_a1.41      | False                |       1997 |  46.65 |
| extratrees     | comp           | irmf_h2_a1.41      | True                 |      10003 |  64.25 |
| extratrees     | comp           | irmf_h2_a1.41_conv | False                |       1997 |  46.95 |
| extratrees     | comp           | irmf_h2_a1.41_conv | True                 |      10003 |  64.38 |
| extratrees     | comp+log_scale | imf_linear_a1.41   | False                |       1997 |  61.37 |
| extratrees     | comp+log_scale | imf_linear_a1.41   | True                 |      10003 |  68.57 |
| extratrees     | comp+log_scale | irmf_h2_a1.41      | False                |       1997 |  47.17 |
| extratrees     | comp+log_scale | irmf_h2_a1.41      | True                 |      10003 |  65.64 |
| extratrees     | comp+log_scale | irmf_h2_a1.41_conv | False                |       1997 |  47.11 |
| extratrees     | comp+log_scale | irmf_h2_a1.41_conv | True                 |      10003 |  66.32 |
| lightgbm       | comp           | imf_linear_a1.41   | False                |       1997 |  65.5  |
| lightgbm       | comp           | imf_linear_a1.41   | True                 |      10003 |  70.19 |
| lightgbm       | comp           | irmf_h2_a1.41      | False                |       1997 |  49.5  |
| lightgbm       | comp           | irmf_h2_a1.41      | True                 |      10003 |  66.95 |
| lightgbm       | comp           | irmf_h2_a1.41_conv | False                |       1997 |  51.1  |
| lightgbm       | comp           | irmf_h2_a1.41_conv | True                 |      10003 |  66.94 |
| lightgbm       | comp+log_scale | imf_linear_a1.41   | False                |       1997 |  65.18 |
| lightgbm       | comp+log_scale | imf_linear_a1.41   | True                 |      10003 |  71.7  |
| lightgbm       | comp+log_scale | irmf_h2_a1.41      | False                |       1997 |  51.07 |
| lightgbm       | comp+log_scale | irmf_h2_a1.41      | True                 |      10003 |  71.33 |
| lightgbm       | comp+log_scale | irmf_h2_a1.41_conv | False                |       1997 |  51.73 |
| lightgbm       | comp+log_scale | irmf_h2_a1.41_conv | True                 |      10003 |  71.33 |
| minirocket     | comp           | imf_linear_a1.41   | False                |       1997 |  40.71 |
| minirocket     | comp           | imf_linear_a1.41   | True                 |      10003 |  50.6  |
| minirocket     | comp           | irmf_h2_a1.41      | False                |       1997 |  36.73 |
| minirocket     | comp           | irmf_h2_a1.41      | True                 |      10003 |  48.38 |
| minirocket     | comp           | irmf_h2_a1.41_conv | False                |       1997 |  38.17 |
| minirocket     | comp           | irmf_h2_a1.41_conv | True                 |      10003 |  48.49 |
| minirocket     | comp+log_scale | imf_linear_a1.41   | False                |       1997 |  40.37 |
| minirocket     | comp+log_scale | imf_linear_a1.41   | True                 |      10003 |  52.3  |
| minirocket     | comp+log_scale | irmf_h2_a1.41      | False                |       1997 |  36.72 |
| minirocket     | comp+log_scale | irmf_h2_a1.41      | True                 |      10003 |  51.52 |
| minirocket     | comp+log_scale | irmf_h2_a1.41_conv | False                |       1997 |  37.78 |
| minirocket     | comp+log_scale | irmf_h2_a1.41_conv | True                 |      10003 |  51.57 |
| xgboost        | comp           | imf_linear_a1.41   | False                |       1997 |  60.97 |
| xgboost        | comp           | imf_linear_a1.41   | True                 |      10003 |  65.52 |
| xgboost        | comp           | irmf_h2_a1.41      | False                |       1997 |  46.2  |
| xgboost        | comp           | irmf_h2_a1.41      | True                 |      10003 |  61.03 |
| xgboost        | comp           | irmf_h2_a1.41_conv | False                |       1997 |  47.31 |
| xgboost        | comp           | irmf_h2_a1.41_conv | True                 |      10003 |  61.3  |
| xgboost        | comp+log_scale | imf_linear_a1.41   | False                |       1997 |  60.94 |
| xgboost        | comp+log_scale | imf_linear_a1.41   | True                 |      10003 |  67.26 |
| xgboost        | comp+log_scale | irmf_h2_a1.41      | False                |       1997 |  45.43 |
| xgboost        | comp+log_scale | irmf_h2_a1.41      | True                 |      10003 |  65.66 |
| xgboost        | comp+log_scale | irmf_h2_a1.41_conv | False                |       1997 |  47.51 |
| xgboost        | comp+log_scale | irmf_h2_a1.41_conv | True                 |      10003 |  65.69 |

## H sensitivity

| method             |   comp |   comp+log_scale |
|:-------------------|-------:|-----------------:|
| irmf_h1_a1.41_conv |  62.86 |            67.14 |
| irmf_h2_a1.41_conv |  64.68 |            68.35 |
| irmf_h4_a1.41_conv |  66.17 |            68.95 |

The six H-sensitivity contrasts form a separate Holm table in `paired_all_metrics.csv`.

## L-only LightGBM

The same 12,000 prepared frames, folds and capacities are retained. L-only therefore includes frames accepted by the original H-only activity gate. Family selection from outer-fold means is exploratory.

| method_a              | feature_set    | L-only / L+H BA %   | L minus L+H [95% CI]; Holm p    |
|:----------------------|:---------------|:--------------------|:--------------------------------|
| raw__L                | comp           | 53.69 / 57.04       | -3.35 [-4.58, -2.13]; 4.058e-05 |
| emd__L                | comp           | 52.05 / 56.49       | -4.44 [-5.57, -3.31]; 3.013e-06 |
| imf_linear_a1.41__L   | comp           | 63.19 / 69.51       | -6.32 [-7.51, -5.14]; 1.699e-07 |
| irmf_h2_a1.41__L      | comp           | 58.32 / 64.34       | -6.01 [-7.31, -4.72]; 6.735e-07 |
| irmf_h2_a1.41_conv__L | comp           | 59.00 / 64.68       | -5.68 [-7.10, -4.26]; 3.013e-06 |
| raw__L                | comp+log_scale | 59.50 / 63.68       | -4.18 [-5.27, -3.10]; 3.013e-06 |
| emd__L                | comp+log_scale | 58.09 / 63.06       | -4.96 [-6.56, -3.36]; 2.155e-05 |
| imf_linear_a1.41__L   | comp+log_scale | 64.56 / 70.70       | -6.14 [-7.54, -4.74]; 1.166e-06 |
| irmf_h2_a1.41__L      | comp+log_scale | 61.51 / 68.23       | -6.72 [-8.09, -5.35]; 3.9e-07   |
| irmf_h2_a1.41_conv__L | comp+log_scale | 62.18 / 68.35       | -6.18 [-7.40, -4.96]; 2.997e-07 |

Holm covers all ten method/scale variants in the band table. The a=2 capacity controls are in `TABLES.md` and the separate 30-test capacity table in `PAIRS.md`. All three metrics remain available in the CSV exports.
