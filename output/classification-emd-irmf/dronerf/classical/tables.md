# DroneRF numerical tables

All scores are percentages unless marked otherwise. Differences are percentage points.
Method names below abbreviate the four main tags. See METHODS.md for selection and Holm families.

## Best classifier per method, descriptive

| task    | feature_set    | method     | classifier   |   balanced_accuracy |   macro_f1 |   accuracy |
|:--------|:---------------|:-----------|:-------------|--------------------:|-----------:|-----------:|
| 10class | comp           | EMD        | logreg       |              38.784 |     37.996 |     38.764 |
| 10class | comp+log_scale | EMD        | linsvm       |              42.916 |     41.473 |     42.831 |
| 10class | comp           | linear IMF | linsvm       |              57.471 |     56.760 |     57.478 |
| 10class | comp+log_scale | linear IMF | linsvm       |              59.487 |     58.774 |     59.482 |
| 10class | comp           | IRMF       | linsvm       |              49.530 |     48.392 |     49.503 |
| 10class | comp+log_scale | IRMF       | logreg       |              53.613 |     53.181 |     53.608 |
| 10class | comp           | raw        | knn          |              41.939 |     40.802 |     41.927 |
| 10class | comp+log_scale | raw        | knn          |              47.694 |     46.819 |     47.646 |
| 2class  | comp           | EMD        | linsvm       |              82.535 |     84.013 |     94.533 |
| 2class  | comp+log_scale | EMD        | logreg       |              86.513 |     86.955 |     95.382 |
| 2class  | comp           | linear IMF | linsvm       |              94.141 |     93.688 |     97.712 |
| 2class  | comp+log_scale | linear IMF | linsvm       |              95.368 |     94.833 |     98.126 |
| 2class  | comp           | IRMF       | linsvm       |              92.894 |     92.074 |     97.104 |
| 2class  | comp+log_scale | IRMF       | linsvm       |              94.185 |     93.568 |     97.664 |
| 2class  | comp           | raw        | knn          |              86.634 |     86.523 |     95.145 |
| 2class  | comp+log_scale | raw        | knn          |              90.250 |     90.016 |     96.406 |
| 4class  | comp           | EMD        | linsvm       |              73.698 |     73.948 |     69.013 |
| 4class  | comp+log_scale | EMD        | linsvm       |              79.866 |     79.952 |     77.382 |
| 4class  | comp           | linear IMF | linsvm       |              85.021 |     84.634 |     80.456 |
| 4class  | comp+log_scale | linear IMF | linsvm       |              87.417 |     87.103 |     83.974 |
| 4class  | comp           | IRMF       | linsvm       |              80.909 |     80.363 |     74.360 |
| 4class  | comp+log_scale | IRMF       | linsvm       |              86.211 |     85.647 |     82.635 |
| 4class  | comp           | raw        | knn          |              76.877 |     76.950 |     72.356 |
| 4class  | comp+log_scale | raw        | knn          |              82.712 |     82.784 |     80.323 |

## 2class paired comparisons, common classifier linsvm

The p-values retain the 72-test task/metric Holm family across every classifier and both feature sets.

### balanced_accuracy

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +10.359 [+7.598, +13.120]  | 1.2809e-06 |  5.1236e-05 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -1.247 [-2.290, -0.205]    | 0.022415   |  0.31381    |   0.00018311 |         0.0043945 |
| comp           | linear IMF minus EMD  | +11.606 [+8.910, +14.302]  | 2.4852e-07 |  1.1929e-05 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +17.056 [+13.478, +20.634] | 7.0982e-08 |  3.5491e-06 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +18.303 [+15.194, +21.412] | 4.8512e-09 |  2.7652e-07 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +6.697 [+3.558, +9.836]    | 0.00043146 |  0.011649   |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +7.836 [+5.767, +9.906]    | 1.1505e-06 |  4.7172e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -1.183 [-2.104, -0.262]    | 0.015513   |  0.2327     |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +9.019 [+6.530, +11.508]   | 1.9154e-06 |  7.4702e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +16.551 [+13.431, +19.670] | 1.8442e-08 |  9.7744e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +17.734 [+14.666, +20.802] | 6.1492e-09 |  3.4435e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +8.714 [+6.491, +10.938]   | 7.6477e-07 |  3.2885e-05 |   6.1035e-05 |         0.0043945 |

### macro_f1

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +8.061 [+5.753, +10.369]   | 2.9248e-06 |  0.00011407 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -1.614 [-2.435, -0.793]    | 0.00086458 |  0.02075    |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +9.675 [+7.195, +12.155]   | 8.0944e-07 |  3.8043e-05 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +12.537 [+9.459, +15.615]  | 4.8425e-07 |  2.3728e-05 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +14.151 [+11.048, +17.254] | 1.2281e-07 |  6.6316e-06 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +4.476 [+1.506, +7.446]    | 0.0060263  |  0.10245    |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +6.784 [+4.905, +8.664]    | 2.0031e-06 |  8.6477e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -1.265 [-2.029, -0.502]    | 0.0031755  |  0.059954   |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +8.049 [+5.734, +10.364]   | 3.0704e-06 |  0.00011668 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +12.916 [+9.439, +16.393]  | 1.4369e-06 |  6.6099e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +14.181 [+10.780, +17.582] | 3.6548e-07 |  1.8274e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +6.132 [+3.117, +9.146]    | 0.00065015 |  0.016254   |   6.1035e-05 |         0.0043945 |

### accuracy

| feature_set    | pair                  | difference, 95% CI      |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +2.571 [+1.837, +3.304] | 2.815e-06  |  0.00010979 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -0.608 [-0.910, -0.307] | 0.00068818 |  0.018581   |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +3.179 [+2.422, +3.937] | 3.3902e-07 |  1.7629e-05 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +3.498 [+2.507, +4.488] | 2.5734e-06 |  0.00010294 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +4.106 [+3.098, +5.115] | 4.8684e-07 |  2.4829e-05 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +0.927 [-0.037, +1.892] | 0.05832    |  0.81649    |   0.0015259  |         0.018311  |
| comp+log_scale | IRMF minus EMD        | +2.346 [+1.712, +2.980] | 1.5074e-06 |  6.4816e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -0.462 [-0.733, -0.190] | 0.0026436  |  0.05816    |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +2.808 [+2.033, +3.583] | 1.9253e-06 |  8.0865e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +3.913 [+2.701, +5.126] | 7.0828e-06 |  0.00024081 |   0.00065331 |         0.010986  |
| comp+log_scale | linear IMF minus raw  | +4.375 [+3.219, +5.531] | 1.1564e-06 |  5.0881e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +1.567 [+0.410, +2.724] | 0.011541   |  0.20774    |   6.1035e-05 |         0.0043945 |

## 4class paired comparisons, common classifier linsvm

The p-values retain the 72-test task/metric Holm family across every classifier and both feature sets.

### balanced_accuracy

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +7.211 [+5.736, +8.686]    | 5.1811e-08 |  1.917e-06  |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -4.112 [-5.094, -3.129]    | 3.4965e-07 |  1.014e-05  |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +11.322 [+9.809, +12.835]  | 2.0732e-10 |  1.078e-08  |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +12.391 [+10.870, +13.912] | 6.6607e-11 |  3.5968e-09 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +16.503 [+14.768, +18.238] | 8.2216e-12 |  5.1796e-10 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +5.181 [+3.904, +6.457]    | 5.0629e-07 |  1.3164e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +6.345 [+5.026, +7.665]    | 6.3481e-08 |  2.2218e-06 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -1.206 [-1.882, -0.530]    | 0.0018514  |  0.012977   |   0.00012207 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +7.551 [+6.165, +8.938]    | 1.3228e-08 |  5.4233e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +13.133 [+11.224, +15.042] | 6.3318e-10 |  2.9126e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +14.339 [+12.337, +16.340] | 3.7051e-10 |  1.7785e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +6.787 [+5.541, +8.034]    | 1.3246e-08 |  5.4233e-07 |   6.1035e-05 |         0.0043945 |

### macro_f1

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +6.415 [+5.197, +7.633]    | 2.0218e-08 |  8.2343e-07 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -4.271 [-5.262, -3.280]    | 2.4564e-07 |  7.2939e-06 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +10.686 [+9.476, +11.896]  | 2.2434e-11 |  1.2338e-09 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +11.592 [+10.291, +12.892] | 1.9804e-11 |  1.109e-09  |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +15.863 [+14.382, +17.343] | 1.6225e-12 |  1.0987e-10 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +5.176 [+3.953, +6.400]    | 3.0636e-07 |  8.2717e-06 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +5.695 [+4.614, +6.776]    | 2.0084e-08 |  8.2343e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -1.456 [-2.193, -0.719]    | 0.00082738 |  0.0074464  |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +7.151 [+5.845, +8.457]    | 1.2329e-08 |  5.1781e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +12.139 [+10.354, +13.924] | 7.3805e-10 |  3.5426e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +13.595 [+11.609, +15.581] | 6.7498e-10 |  3.3749e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +6.444 [+5.002, +7.886]    | 1.5736e-07 |  5.3503e-06 |   6.1035e-05 |         0.0043945 |

### accuracy

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +5.347 [+3.964, +6.731]    | 9.0457e-07 |  2.7137e-05 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -6.096 [-7.388, -4.804]    | 8.0395e-08 |  3.055e-06  |   0.00065331 |         0.0071322 |
| comp           | linear IMF minus EMD  | +11.443 [+10.221, +12.665] | 1.0128e-11 |  6.2795e-10 |   0.00065331 |         0.0071322 |
| comp           | IRMF minus raw        | +10.646 [+8.804, +12.487]  | 6.1339e-09 |  2.883e-07  |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +16.742 [+14.856, +18.627] | 2.0868e-11 |  1.2312e-09 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +5.298 [+3.558, +7.039]    | 1.3375e-05 |  0.00027313 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +5.253 [+4.088, +6.418]    | 1.4151e-07 |  4.9529e-06 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -1.340 [-2.214, -0.465]    | 0.0054096  |  0.043276   |   0.00012207 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +6.592 [+5.404, +7.781]    | 1.0457e-08 |  4.703e-07  |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +11.227 [+9.513, +12.942]  | 1.2087e-09 |  6.2853e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +12.567 [+10.799, +14.335] | 4.0924e-10 |  2.2099e-08 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +5.975 [+4.538, +7.411]    | 3.784e-07  |  1.2109e-05 |   6.1035e-05 |         0.0043945 |

## 10class paired comparisons, common classifier linsvm

The p-values retain the 72-test task/metric Holm family across every classifier and both feature sets.

### balanced_accuracy

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +10.800 [+9.909, +11.691]  | 3.0039e-13 |  2.0126e-11 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -7.941 [-9.166, -6.715]    | 1.3926e-09 |  5.8491e-08 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +18.741 [+17.055, +20.427] | 9.821e-13  |  6.3836e-11 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +14.583 [+12.982, +16.185] | 1.4828e-11 |  8.0069e-10 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +22.524 [+20.334, +24.714] | 2.8371e-12 |  1.759e-10  |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +3.783 [+2.248, +5.318]    | 0.00011488 |  0.0021826  |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +10.667 [+9.802, +11.533]  | 2.3942e-13 |  1.6281e-11 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -5.904 [-7.010, -4.798]    | 1.7083e-08 |  6.3207e-07 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +16.572 [+15.304, +17.840] | 1.0664e-13 |  7.4648e-12 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +14.537 [+13.165, +15.910] | 1.9014e-12 |  1.1979e-10 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +20.442 [+18.749, +22.134] | 3.1394e-13 |  2.072e-11  |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +3.870 [+2.470, +5.270]    | 3.6827e-05 |  0.0008102  |   6.1035e-05 |         0.0043945 |

### macro_f1

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +11.453 [+10.662, +12.244] | 2.5803e-14 |  1.832e-12  |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -8.368 [-9.804, -6.933]    | 5.5161e-09 |  2.2064e-07 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +19.821 [+18.145, +21.498] | 4.2123e-13 |  2.6959e-11 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +17.549 [+15.659, +19.439] | 1.1389e-11 |  6.2638e-10 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +25.917 [+23.526, +28.309] | 1.3898e-12 |  8.7559e-11 |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +6.096 [+4.382, +7.810]    | 2.3692e-06 |  6.3968e-05 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +11.151 [+10.342, +11.960] | 5.0906e-14 |  3.5634e-12 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -6.150 [-7.379, -4.920]    | 3.893e-08  |  1.4015e-06 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +17.301 [+15.921, +18.681] | 1.885e-13  |  1.2441e-11 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +16.850 [+15.235, +18.465] | 2.3372e-12 |  1.4257e-10 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +23.000 [+21.279, +24.721] | 7.8349e-14 |  5.2494e-12 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +5.699 [+4.172, +7.226]    | 1.358e-06  |  3.9381e-05 |   6.1035e-05 |         0.0043945 |

### accuracy

| feature_set    | pair                  | difference, 95% CI         |       p_nb |   p_nb_holm |   p_wilcoxon |   p_wilcoxon_holm |
|:---------------|:----------------------|:---------------------------|-----------:|------------:|-------------:|------------------:|
| comp           | IRMF minus EMD        | +10.830 [+9.902, +11.758]  | 5.0637e-13 |  3.4433e-11 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus linear IMF | -7.975 [-9.334, -6.616]    | 5.0536e-09 |  2.0214e-07 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus EMD  | +18.806 [+17.021, +20.590] | 2.0381e-12 |  1.3451e-10 |   6.1035e-05 |         0.0043945 |
| comp           | IRMF minus raw        | +14.656 [+12.872, +16.440] | 5.9438e-11 |  3.0314e-09 |   6.1035e-05 |         0.0043945 |
| comp           | linear IMF minus raw  | +22.632 [+20.194, +25.069] | 1.1393e-11 |  6.722e-10  |   6.1035e-05 |         0.0043945 |
| comp           | EMD minus raw         | +3.826 [+2.236, +5.416]    | 0.00014446 |  0.0024558  |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus EMD        | +10.733 [+9.837, +11.628]  | 3.4979e-13 |  2.4135e-11 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus linear IMF | -5.919 [-7.162, -4.676]    | 7.1794e-08 |  2.6564e-06 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus EMD  | +16.652 [+15.181, +18.122] | 7.6177e-13 |  5.1038e-11 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | IRMF minus raw        | +14.622 [+13.063, +16.181] | 9.9715e-12 |  5.9829e-10 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | linear IMF minus raw  | +20.541 [+18.581, +22.501] | 2.1968e-12 |  1.4279e-10 |   6.1035e-05 |         0.0043945 |
| comp+log_scale | EMD minus raw         | +3.889 [+2.360, +5.419]    | 8.5149e-05 |  0.001714   |   6.1035e-05 |         0.0043945 |

## IRMF minus EMD by classifier

| task    | feature_set    | classifier   |   mean_diff_pp |       p_nb |   p_nb_holm |   p_wilcoxon_holm |
|:--------|:---------------|:-------------|---------------:|-----------:|------------:|------------------:|
| 2class  | comp           | lda          |        12.839  | 6.1855e-08 |  3.1546e-06 |         0.0043945 |
| 2class  | comp           | qda          |         7.8459 | 0.0045323  |  0.090181   |         0.0043945 |
| 2class  | comp           | gnb          |        25.662  | 8.0117e-14 |  5.5281e-12 |         0.0043945 |
| 2class  | comp           | logreg       |        10.478  | 3.9026e-06 |  0.00014253 |         0.0043945 |
| 2class  | comp           | knn          |        21.705  | 2.0201e-11 |  1.3131e-09 |         0.0043945 |
| 2class  | comp           | linsvm       |        10.359  | 1.2809e-06 |  5.1236e-05 |         0.0043945 |
| 2class  | comp+log_scale | lda          |         9.9303 | 9.6677e-06 |  0.0003287  |         0.0043945 |
| 2class  | comp+log_scale | qda          |         4.1468 | 0.065062   |  0.5205     |         0.0043945 |
| 2class  | comp+log_scale | gnb          |        25.643  | 7.8595e-14 |  5.5016e-12 |         0.0043945 |
| 2class  | comp+log_scale | logreg       |         7.3185 | 3.4151e-05 |  0.0010587  |         0.0043945 |
| 2class  | comp+log_scale | knn          |        20.066  | 6.4424e-11 |  3.9943e-09 |         0.0043945 |
| 2class  | comp+log_scale | linsvm       |         7.8363 | 1.1505e-06 |  4.7172e-05 |         0.0043945 |
| 4class  | comp           | lda          |         8.1974 | 3.0397e-08 |  1.1551e-06 |         0.0043945 |
| 4class  | comp           | qda          |        -3.2864 | 0.00016284 |  0.0022216  |         0.0043945 |
| 4class  | comp           | gnb          |        16.033  | 2.1634e-12 |  1.4495e-10 |         0.0043945 |
| 4class  | comp           | logreg       |         6.7365 | 7.9248e-08 |  2.6152e-06 |         0.0043945 |
| 4class  | comp           | knn          |        18.59   | 2.5997e-11 |  1.5598e-09 |         0.0043945 |
| 4class  | comp           | linsvm       |         7.2106 | 5.1811e-08 |  1.917e-06  |         0.0043945 |
| 4class  | comp+log_scale | lda          |         8.456  | 3.4249e-09 |  1.4727e-07 |         0.0043945 |
| 4class  | comp+log_scale | qda          |        -3.0725 | 0.00015868 |  0.0022216  |         0.0043945 |
| 4class  | comp+log_scale | gnb          |        15.842  | 1.9043e-12 |  1.295e-10  |         0.0043945 |
| 4class  | comp+log_scale | logreg       |         5.9403 | 6.5246e-08 |  2.2218e-06 |         0.0043945 |
| 4class  | comp+log_scale | knn          |        17.009  | 6.8613e-11 |  3.6365e-09 |         0.0043945 |
| 4class  | comp+log_scale | linsvm       |         6.3454 | 6.3481e-08 |  2.2218e-06 |         0.0043945 |
| 10class | comp           | lda          |         9.4445 | 8.5866e-12 |  5.152e-10  |         0.0043945 |
| 10class | comp           | qda          |        -2.1222 | 3.0273e-06 |  8.1737e-05 |         0.0043945 |
| 10class | comp           | gnb          |         4.8778 | 5.6905e-07 |  1.7072e-05 |         0.0043945 |
| 10class | comp           | logreg       |        10.559  | 3.8747e-10 |  1.7436e-08 |         0.0043945 |
| 10class | comp           | knn          |        12.568  | 1.3491e-11 |  7.69e-10   |         0.0043945 |
| 10class | comp           | linsvm       |        10.8    | 3.0039e-13 |  2.0126e-11 |         0.0043945 |
| 10class | comp+log_scale | lda          |        10.553  | 1.7057e-10 |  8.1874e-09 |         0.0043945 |
| 10class | comp+log_scale | qda          |        -2.1542 | 3.5086e-06 |  9.1222e-05 |         0.0043945 |
| 10class | comp+log_scale | gnb          |         3.8135 | 9.4498e-06 |  0.0002268  |         0.0043945 |
| 10class | comp+log_scale | logreg       |        10.705  | 1.5553e-09 |  6.3769e-08 |         0.0043945 |
| 10class | comp+log_scale | knn          |        12.189  | 2.2068e-11 |  1.1696e-09 |         0.0043945 |
| 10class | comp+log_scale | linsvm       |        10.667  | 2.3942e-13 |  1.6281e-11 |         0.0043945 |

## 10-class recall, IRMF versus EMD

| feature_set    | class     |   mean_a |   mean_b |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:----------|---------:|---------:|------------:|---------:|----------:|------------:|------------------:|
| comp           | BUI 00000 |    93.44 |    84.46 |      8.985  |   4.849  |    13.12  |   0.1583    |           0.04395 |
| comp           | BUI 10000 |    39.83 |    19.02 |     20.81   |  15.67   |    25.96  |   0.0003504 |           0.04395 |
| comp           | BUI 10001 |    34.03 |    33.05 |      0.9799 |  -2.179  |     4.139 |   1         |           1       |
| comp           | BUI 10010 |    19.35 |    14.18 |      5.169  |   1.3    |     9.039 |   1         |           0.04395 |
| comp           | BUI 10011 |    36.82 |    19.2  |     17.62   |  12      |    23.24  |   0.005616  |           0.04395 |
| comp           | BUI 10100 |    54.64 |    40.17 |     14.47   |   9.14   |    19.8   |   0.0231    |           0.04395 |
| comp           | BUI 10101 |    39.1  |    22.01 |     17.09   |  12.47   |    21.71  |   0.0009702 |           0.04395 |
| comp           | BUI 10110 |    28.31 |    21.62 |      6.688  |   2.293  |    11.08  |   1         |           0.2574  |
| comp           | BUI 10111 |    57.43 |    41.76 |     15.67   |   9.992  |    21.35  |   0.01993   |           0.04395 |
| comp           | BUI 11000 |    92.34 |    91.83 |      0.5133 |  -1.1    |     2.127 |   1         |           1       |
| comp+log_scale | BUI 00000 |    93.61 |    87.83 |      5.783  |   2.171  |     9.396 |   1         |           0.2574  |
| comp+log_scale | BUI 10000 |    52.98 |    23.7  |     29.28   |  24.71   |    33.85  |   1.136e-06 |           0.04395 |
| comp+log_scale | BUI 10001 |    38.83 |    36.5  |      2.329  |  -1.585  |     6.243 |   1         |           1       |
| comp+log_scale | BUI 10010 |    24.66 |    21.35 |      3.314  |  -0.7446 |     7.373 |   1         |           0.8379  |
| comp+log_scale | BUI 10011 |    36.9  |    24.2  |     12.7    |   5.037  |    20.36  |   1         |           0.04395 |
| comp+log_scale | BUI 10100 |    52.85 |    41.03 |     11.83   |   7.081  |    16.57  |   0.04956   |           0.04395 |
| comp+log_scale | BUI 10101 |    44.12 |    33.01 |     11.11   |   6.769  |    15.45  |   0.0392    |           0.2574  |
| comp+log_scale | BUI 10110 |    37.06 |    22.38 |     14.68   |  10.67   |    18.69  |   0.0011    |           0.04395 |
| comp+log_scale | BUI 10111 |    62.57 |    47.56 |     15.01   |   8.954  |    21.06  |   0.05185   |           0.2574  |
| comp+log_scale | BUI 11000 |    92.25 |    91.6  |      0.6481 |  -0.7698 |     2.066 |   1         |           1       |

## Most informative 10-class scales

Sorted by mean MI across 20 features per slot; channel 0=L and 1=H. Descriptive, all frames.

| method     |   channel | slot_label   |   epsilon2_mean |   mi_mean |
|:-----------|----------:|:-------------|----------------:|----------:|
| linear IMF |         0 | w47          |          0.2391 |    0.2926 |
| linear IMF |         0 | w65          |          0.2530 |    0.2861 |
| linear IMF |         0 | w33          |          0.2150 |    0.2760 |
| IRMF       |         0 | w47          |          0.1964 |    0.2511 |
| IRMF       |         0 | w65          |          0.2016 |    0.2427 |
| IRMF       |         0 | w33          |          0.1802 |    0.2408 |
| EMD        |         0 | imf1         |          0.1310 |    0.2199 |
| raw        |         0 | frame        |          0.1008 |    0.1800 |
| linear IMF |         1 | w9           |          0.1310 |    0.1365 |
| IRMF       |         1 | w9           |          0.1220 |    0.1300 |
| linear IMF |         1 | w7           |          0.1064 |    0.1268 |
| IRMF       |         1 | w7           |          0.1011 |    0.1229 |
| EMD        |         0 | imf2         |          0.0548 |    0.1209 |
| linear IMF |         1 | w5           |          0.0963 |    0.1206 |
| IRMF       |         1 | w5           |          0.0921 |    0.1167 |
| EMD        |         0 | imf3         |          0.0714 |    0.1099 |
| EMD        |         1 | imf1         |          0.0886 |    0.1073 |
| raw        |         1 | frame        |          0.0655 |    0.0787 |
| EMD        |         1 | imf2         |          0.0487 |    0.0640 |
| EMD        |         1 | imf3         |          0.0258 |    0.0411 |

## L-only versus both views

| task    | feature_set    | method     | classifier   |   L_BA |   both_BA | difference, 95% CI      |       p_nb |   p_nb_holm |   p_wilcoxon_holm |
|:--------|:---------------|:-----------|:-------------|-------:|----------:|:------------------------|-----------:|------------:|------------------:|
| 10class | comp           | EMD        | linsvm       | 34.675 |    38.73  | -4.055 [-5.243, -2.868] | 3.7698e-06 |  1.5079e-05 |        0.00048828 |
| 10class | comp+log_scale | EMD        | linsvm       | 38.775 |    42.916 | -4.141 [-5.806, -2.476] | 0.00010553 |  0.00021107 |        0.00048828 |
| 10class | comp           | linear IMF | linsvm       | 51.146 |    57.471 | -6.325 [-8.082, -4.568] | 2.0675e-06 |  1.0338e-05 |        0.00048828 |
| 10class | comp+log_scale | linear IMF | linsvm       | 53.342 |    59.487 | -6.145 [-7.548, -4.742] | 2.0142e-07 |  1.6114e-06 |        0.00048828 |
| 10class | comp           | IRMF       | linsvm       | 43.57  |    49.53  | -5.961 [-7.481, -4.440] | 7.6232e-07 |  4.5739e-06 |        0.00048828 |
| 10class | comp+log_scale | IRMF       | linsvm       | 47.578 |    53.583 | -6.006 [-7.508, -4.503] | 6.0576e-07 |  4.2403e-06 |        0.00048828 |
| 10class | comp           | raw        | linsvm       | 31.938 |    34.947 | -3.009 [-4.344, -1.674] | 0.0002647  |  0.0002647  |        0.00048828 |
| 10class | comp+log_scale | raw        | linsvm       | 34.663 |    39.046 | -4.383 [-5.848, -2.918] | 1.6053e-05 |  4.8159e-05 |        0.00048828 |

## Log-scale-only control

| task    | classifier   |   balanced_accuracy |   macro_f1 |   accuracy |
|:--------|:-------------|--------------------:|-----------:|-----------:|
| 10class | linsvm       |              19.566 |     11.287 |     19.440 |

## IRMF accuracy by convergence status

Means over splits. n_test is the mean subset size per fold; common-class BA uses classes present in both statuses.
This association is descriptive, since status changes which frames are evaluated.

| task    | feature_set    | any_unconverged   |   balanced_accuracy |   macro_f1 |   accuracy |   common_class_balanced_accuracy |   n_test |   n_common_classes |
|:--------|:---------------|:------------------|--------------------:|-----------:|-----------:|---------------------------------:|---------:|-------------------:|
| 10class | comp           | False             |              50.527 |     49.402 |     50.843 |                           50.527 | 2000.600 |             10.000 |
| 10class | comp           | True              |              41.304 |     39.661 |     42.722 |                           41.304 |  399.400 |             10.000 |
| 10class | comp+log_scale | False             |              55.311 |     54.406 |     55.539 |                           55.311 | 2000.600 |             10.000 |
| 10class | comp+log_scale | True              |              41.959 |     40.471 |     43.624 |                           41.959 |  399.400 |             10.000 |
| 2class  | comp           | False             |              93.470 |     93.088 |     97.224 |                           93.470 | 2000.600 |              2.000 |
| 2class  | comp           | True              |              83.406 |     77.690 |     96.503 |                           83.406 |  399.400 |              2.000 |
| 2class  | comp+log_scale | False             |              94.678 |     94.419 |     97.770 |                           94.678 | 2000.600 |              2.000 |
| 2class  | comp+log_scale | True              |              86.582 |     81.284 |     97.121 |                           86.582 |  399.400 |              2.000 |
| 4class  | comp           | False             |              80.829 |     80.441 |     74.103 |                           80.829 | 2000.600 |              4.000 |
| 4class  | comp           | True              |              77.796 |     76.286 |     75.639 |                           77.796 |  399.400 |              4.000 |
| 4class  | comp+log_scale | False             |              86.889 |     86.493 |     83.559 |                           86.889 | 2000.600 |              4.000 |
| 4class  | comp+log_scale | True              |              78.068 |     76.786 |     77.996 |                           78.068 |  399.400 |              4.000 |

## Cached decomposition cost

| method           |   cpu_seconds_per_two_band_frame |   decomposition_wall_seconds |   failed |   wall_seconds_per_two_band_frame |   cpu_seconds_total |   channel_0_cpu_seconds |   channel_1_cpu_seconds |
|:-----------------|---------------------------------:|-----------------------------:|---------:|----------------------------------:|--------------------:|------------------------:|------------------------:|
| raw              |                         0        |                        0     |        0 |                        nan        |              nan    |             nan         |             nan         |
| emd              |                         0.029429 |                       61.335 |        0 |                          0.029708 |              353.15 |               0.013777  |               0.015652  |
| imf_linear_a1.41 |                         0.012638 |                       33.202 |        0 |                          0.013321 |              151.66 |               0.0063374 |               0.0063007 |
| irmf_h2_a1.41    |                         1.2787   |                     2625.8   |        0 |                          1.3008   |            15344    |               0.76097   |               0.5177    |

Stage-by-class rates: convergence_stages.csv. Frame-level class rates: convergence_classes.csv.

Complete classifier/metric comparisons: paired_all_metrics.csv. All per-class pairs: paired_class_recall.csv.
