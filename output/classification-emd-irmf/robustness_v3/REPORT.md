# Classification robustness v3

Protocol and amendment dated 2026-10-22. Noise is added before shared median/MAD normalization: `x + (u < p) * 3 noise_sigma_hat(x_clean_raw) * e * s`, with common random numbers and p=.1,.2,.3,.4. A constant channel receives no contamination. There is no Gaussian term.

IRMF uses converged safeguarded Newton, tolerance 1e-12, cap 100 and cap-1000 retry. H is fixed in normalized units per dataset, twice the median E1 across every channel of its 100 excluded clean pilot frames. Those frames enter no classifier fit or score. E1 is retained by the pre-specified IQR rule, 8/12 lower-error and 4/12 no-worse-stability cells. Pilot report SHA-256: `9bb6f2c66f6da7a54090be2bf0c5cbe2aedf70f6adb12574a01eb86b07eb4e8b`.

Features are the shared 20 per slot, comp only. Raw normalized samples stay float64; decomposition stacks are float32. EMD runs once per channel with 8 IMFs plus residue; lower K folds the remaining slots into the residue. IMF and IRMF keep the fine content below the 31-sample final window in their residual slot. Their extracted slots are stored fine to coarse, followed by the residual.

## Fixed H and windows

### DroneRF

H_d = 2.09416640072853; 200 pilot channels. EMD K=2,3,4,6,8. Linear and IRMF windows, coarse to fine: K=3: 1025,177,31; K=4: 1025,317,99,31; K=6: 1025,507,251,125,63,31; K=8: 1025,621,375,227,139,85,51,31; K=11: 1025,721,507,357,251,177,125,89,63,45,31.

### BirdCLEF 2023

H_d = 1.08331489400204; 100 pilot channels. EMD K=2,3,4,6,8. Linear and IRMF windows, coarse to fine: K=3: 1025,177,31; K=4: 1025,317,99,31; K=6: 1025,507,251,125,63,31; K=8: 1025,621,375,227,139,85,51,31; K=11: 1025,721,507,357,251,177,125,89,63,45,31.

### ESC-50 animals

H_d = 0.47263983379285; 100 pilot channels. EMD K=2,3,4,6,8. Linear and IRMF windows, coarse to fine: K=3: 1025,177,31; K=4: 1025,317,99,31; K=6: 1025,507,251,125,63,31; K=8: 1025,621,375,227,139,85,51,31; K=11: 1025,721,507,357,251,177,125,89,63,45,31.

### Synthetic

H_d = 1.185305641621; 100 pilot channels. EMD K=2,3,4,6,8. Linear and IRMF windows, coarse to fine: K=3: 513,125,31; K=4: 513,201,79,31; K=6: 513,293,167,95,55,31; K=8: 513,343,229,153,103,69,47,31; K=9: 513,361,253,179,125,89,63,45,31.

## Clean-selected K*

Each classifier selects its own K* by mean clean balanced accuracy over all 15 grouped splits, ties to smaller K. The same K* applies at every condition. Selecting on clean outer-fold scores makes clean scores slightly optimistic for every method alike. These are descriptive selection scores, without nested K selection.

| dataset       | classifier   | method   |   k |   mean_clean_ba |
|:--------------|:-------------|:---------|----:|----------------:|
| birdclef2023  | lightgbm     | emd      |   3 |        0.505118 |
| birdclef2023  | lightgbm     | irmf     |  11 |        0.504927 |
| birdclef2023  | lightgbm     | linear   |  11 |        0.500733 |
| birdclef2023  | lightgbm     | raw      |   0 |        0.466794 |
| birdclef2023  | logreg       | emd      |   2 |        0.462417 |
| birdclef2023  | logreg       | irmf     |  11 |        0.440737 |
| birdclef2023  | logreg       | linear   |  11 |        0.416754 |
| birdclef2023  | logreg       | raw      |   0 |        0.375896 |
| dronerf       | lightgbm     | emd      |   2 |        0.590986 |
| dronerf       | lightgbm     | irmf     |  11 |        0.609172 |
| dronerf       | lightgbm     | linear   |  11 |        0.659413 |
| dronerf       | lightgbm     | raw      |   0 |        0.585324 |
| dronerf       | linsvm       | emd      |   2 |        0.414039 |
| dronerf       | linsvm       | irmf     |   8 |        0.429283 |
| dronerf       | linsvm       | linear   |  11 |        0.504870 |
| dronerf       | linsvm       | raw      |   0 |        0.348788 |
| esc50_animals | lightgbm     | emd      |   6 |        0.594131 |
| esc50_animals | lightgbm     | irmf     |  11 |        0.628279 |
| esc50_animals | lightgbm     | linear   |  11 |        0.623557 |
| esc50_animals | lightgbm     | raw      |   0 |        0.519943 |
| esc50_animals | linsvm       | emd      |   4 |        0.561695 |
| esc50_animals | linsvm       | irmf     |  11 |        0.601983 |
| esc50_animals | linsvm       | linear   |  11 |        0.609256 |
| esc50_animals | linsvm       | raw      |   0 |        0.443390 |
| synthetic     | lightgbm     | emd      |   3 |        0.421010 |
| synthetic     | lightgbm     | irmf     |   9 |        0.778077 |
| synthetic     | lightgbm     | linear   |   9 |        0.660875 |
| synthetic     | lightgbm     | raw      |   0 |        0.344496 |
| synthetic     | logreg       | emd      |   4 |        0.548418 |
| synthetic     | logreg       | irmf     |   9 |        0.810562 |
| synthetic     | logreg       | linear   |   6 |        0.688859 |
| synthetic     | logreg       | raw      |   0 |        0.480904 |

## Pre-specified reading at 40%

| dataset       | condition   |   irmf_k |   linear_k |       did |    ci_low |   ci_high |   p_nb_holm | verdict                          | status                     |
|:--------------|:------------|---------:|-----------:|----------:|----------:|----------:|------------:|:---------------------------------|:---------------------------|
| dronerf       | nb3_0.4     |        8 |         11 |  0.060683 |  0.046396 |  0.074970 |    0.000004 | IRMF more robust                 | primary                    |
| birdclef2023  | nb3_0.4     |       11 |         11 | -0.020519 | -0.040651 | -0.000387 |    0.327767 | does not reject; not equivalence | primary                    |
| esc50_animals | nb3_0.4     |       11 |         11 | -0.001258 | -0.050218 |  0.047703 |    1.000000 | does not reject; not equivalence | primary                    |
| synthetic     | nb3_0.4     |        9 |          6 | -0.029226 | -0.091248 |  0.032796 |    1.000000 | does not reject; not equivalence | synthetic, separate family |

DiD = [BA_IRMF*(p) - BA_comparator*(p)] - [BA_IRMF*(clean) - BA_comparator*(clean)]. Two-sided Nadeau-Bengio corrected t-tests use 15 paired splits. Wilcoxon is secondary. Holm families are Primary 12, S1 12, S3 24 and S5 6 over the three real datasets. Synthetic uses a separate family of 18. Partial previews pad absent family tests with p=1. A positive significant Primary DiD at 40% favors IRMF robustness; a negative significant one favors linear IMF. Failure to reject is not equivalence.

Matched classical models train and test at 40%, at each method's classical K*. Each is scheduled only after that method's clean shift fits finish. Every classical fit runs the full shared grouped C grid afresh. LightGBM uses the shared inner early stopping and outer refit. No model is pickled.

## Best-vs-best balanced accuracy

Fractions, mean and sample SD over 15 splits. K* is fixed by clean scores.

| dataset       | scenario   | classifier   | family   | variant    |   k | condition   |     mean |      std |
|:--------------|:-----------|:-------------|:---------|:-----------|----:|:------------|---------:|---------:|
| birdclef2023  | matched    | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.399871 | 0.016693 |
| birdclef2023  | matched    | logreg       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.402002 | 0.018467 |
| birdclef2023  | matched    | logreg       | linear   | linear_k11 |  11 | nb3_0.4     | 0.387121 | 0.019639 |
| birdclef2023  | matched    | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.334608 | 0.014491 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.505118 | 0.019362 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.186990 | 0.011310 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.153708 | 0.011083 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.125286 | 0.014097 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.105965 | 0.016493 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.504927 | 0.024422 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.258805 | 0.019081 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.213719 | 0.012893 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.198415 | 0.014233 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.187179 | 0.016528 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.500733 | 0.021822 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.261710 | 0.018791 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.213134 | 0.018435 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.189000 | 0.013387 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.175349 | 0.013250 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.466794 | 0.021291 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.199539 | 0.014697 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.157822 | 0.014575 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.146469 | 0.018396 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.141147 | 0.019472 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | clean       | 0.462417 | 0.014542 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.1     | 0.148699 | 0.013775 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.2     | 0.117316 | 0.012059 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.3     | 0.107470 | 0.008785 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.110210 | 0.015360 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | clean       | 0.440737 | 0.014280 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.243474 | 0.019376 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.217979 | 0.015391 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.204062 | 0.014790 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.191278 | 0.015772 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | clean       | 0.416754 | 0.014600 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.1     | 0.235704 | 0.012103 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.2     | 0.213794 | 0.009742 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.3     | 0.199501 | 0.011776 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.4     | 0.187815 | 0.011375 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | clean       | 0.375896 | 0.009855 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.154059 | 0.018711 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.132870 | 0.013476 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.122513 | 0.017691 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.112062 | 0.020275 |
| dronerf       | matched    | linsvm       | emd      | emd_k2     |   2 | nb3_0.4     | 0.249657 | 0.009036 |
| dronerf       | matched    | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.256314 | 0.009970 |
| dronerf       | matched    | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.251905 | 0.007976 |
| dronerf       | matched    | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.261931 | 0.008459 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.590986 | 0.022917 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.194352 | 0.006833 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.177536 | 0.007622 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.161974 | 0.010973 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.148777 | 0.007339 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.609172 | 0.015951 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.218512 | 0.008647 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.198112 | 0.012489 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.185313 | 0.010382 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.171972 | 0.010516 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.659413 | 0.016836 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.156742 | 0.011274 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.141367 | 0.006328 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.134103 | 0.005751 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.131973 | 0.004512 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.585324 | 0.025684 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.161756 | 0.007698 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.148466 | 0.007900 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.142920 | 0.005591 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.138177 | 0.004237 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | clean       | 0.414039 | 0.012176 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.1     | 0.158578 | 0.011446 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.2     | 0.139945 | 0.009156 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.3     | 0.132863 | 0.007428 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.4     | 0.130726 | 0.006936 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | clean       | 0.429283 | 0.010346 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.108464 | 0.013527 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.100960 | 0.015845 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.103134 | 0.012382 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.107525 | 0.010823 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.504870 | 0.015033 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.136412 | 0.012093 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.2     | 0.125712 | 0.008900 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.3     | 0.123258 | 0.006046 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.122430 | 0.006076 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.348788 | 0.015338 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.176991 | 0.006960 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.164069 | 0.013559 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.139261 | 0.022344 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.122124 | 0.010596 |
| esc50_animals | matched    | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.566014 | 0.026865 |
| esc50_animals | matched    | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.601639 | 0.042996 |
| esc50_animals | matched    | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.601009 | 0.038363 |
| esc50_animals | matched    | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.438038 | 0.030736 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.594131 | 0.043076 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.371810 | 0.037947 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.2     | 0.265493 | 0.047515 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.3     | 0.210552 | 0.045293 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.4     | 0.184108 | 0.040063 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.628279 | 0.037024 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.473946 | 0.036992 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.409785 | 0.043935 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.390987 | 0.045642 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.380295 | 0.039924 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.623557 | 0.045645 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.430943 | 0.046109 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.340989 | 0.054771 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.309036 | 0.066539 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.304021 | 0.051942 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.519943 | 0.028011 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.287538 | 0.027557 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.226136 | 0.030865 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.211823 | 0.034347 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.216617 | 0.030782 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | clean       | 0.561695 | 0.051406 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.1     | 0.276763 | 0.024412 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.2     | 0.201088 | 0.016906 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.3     | 0.157995 | 0.020289 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.138591 | 0.021386 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | clean       | 0.601983 | 0.029505 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.420553 | 0.030282 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.341662 | 0.022203 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.294389 | 0.024516 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.267022 | 0.029177 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.609256 | 0.040018 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.426727 | 0.021323 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.2     | 0.346012 | 0.022696 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.3     | 0.303042 | 0.022182 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.275554 | 0.032621 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.443390 | 0.042978 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.223748 | 0.016872 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.209262 | 0.007786 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.193830 | 0.010721 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.176377 | 0.026003 |
| synthetic     | matched    | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.321105 | 0.033474 |
| synthetic     | matched    | logreg       | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.597480 | 0.043984 |
| synthetic     | matched    | logreg       | linear   | linear_k6  |   6 | nb3_0.4     | 0.472035 | 0.028085 |
| synthetic     | matched    | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.339689 | 0.035203 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.421010 | 0.042275 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.328291 | 0.030788 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.276580 | 0.030321 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.250895 | 0.032747 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.232549 | 0.032448 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | clean       | 0.778077 | 0.051871 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.688646 | 0.062617 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.2     | 0.565020 | 0.061471 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.3     | 0.449081 | 0.061042 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.369970 | 0.059056 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | clean       | 0.660875 | 0.050018 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.1     | 0.561903 | 0.036846 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.2     | 0.464989 | 0.040403 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.3     | 0.390211 | 0.036061 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.4     | 0.345470 | 0.026254 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.344496 | 0.052852 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.352821 | 0.054920 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.324926 | 0.039276 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.323485 | 0.048995 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.305507 | 0.044857 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | clean       | 0.548418 | 0.048763 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.1     | 0.450450 | 0.026238 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.2     | 0.382772 | 0.035453 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.3     | 0.339237 | 0.030190 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.307533 | 0.035831 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | clean       | 0.810562 | 0.025773 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.738305 | 0.047685 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.2     | 0.656438 | 0.050827 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.3     | 0.588132 | 0.040003 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.511439 | 0.033784 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | clean       | 0.688859 | 0.041277 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.1     | 0.577750 | 0.034255 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.2     | 0.510174 | 0.041077 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.3     | 0.459072 | 0.039938 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.4     | 0.418961 | 0.037718 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | clean       | 0.480904 | 0.049685 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.442697 | 0.035932 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.388441 | 0.031430 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.349624 | 0.023862 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.325640 | 0.027871 |

## All K and conditions

| dataset       | scenario   | classifier   | family   | variant    |   k | condition   |     mean |      std |   size |
|:--------------|:-----------|:-------------|:---------|:-----------|----:|:------------|---------:|---------:|-------:|
| birdclef2023  | matched    | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.399871 | 0.016693 |     15 |
| birdclef2023  | matched    | logreg       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.402002 | 0.018467 |     15 |
| birdclef2023  | matched    | logreg       | linear   | linear_k11 |  11 | nb3_0.4     | 0.387121 | 0.019639 |     15 |
| birdclef2023  | matched    | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.334608 | 0.014491 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.504622 | 0.020623 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.191470 | 0.011479 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.158939 | 0.013636 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.131860 | 0.009217 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.114662 | 0.008047 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.505118 | 0.019362 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.186990 | 0.011310 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.153708 | 0.011083 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.125286 | 0.014097 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.105965 | 0.016493 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k4     |   4 | clean       | 0.500535 | 0.018947 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.1     | 0.187274 | 0.012425 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.2     | 0.151671 | 0.014442 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.3     | 0.124602 | 0.014127 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.4     | 0.108936 | 0.015064 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.502607 | 0.016973 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.180733 | 0.009252 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.2     | 0.144609 | 0.011706 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.3     | 0.120089 | 0.016268 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.4     | 0.105001 | 0.014460 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k8     |   8 | clean       | 0.501588 | 0.016815 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.1     | 0.182099 | 0.006371 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.2     | 0.143860 | 0.010930 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.3     | 0.120040 | 0.013729 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.4     | 0.103955 | 0.013587 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.504927 | 0.024422 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.258805 | 0.019081 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.213719 | 0.012893 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.198415 | 0.014233 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.187179 | 0.016528 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k3    |   3 | clean       | 0.491730 | 0.023656 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.230330 | 0.012971 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.194425 | 0.009491 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.181729 | 0.010446 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.172428 | 0.010088 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k4    |   4 | clean       | 0.498033 | 0.027698 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.236839 | 0.015461 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.202766 | 0.013330 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.188447 | 0.013140 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.175953 | 0.011292 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k6    |   6 | clean       | 0.503166 | 0.025482 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.246394 | 0.017766 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.205135 | 0.012823 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.187903 | 0.010915 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.177671 | 0.010255 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k8    |   8 | clean       | 0.500332 | 0.025130 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.250320 | 0.018008 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.207380 | 0.015449 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.192876 | 0.018185 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.183715 | 0.018071 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.500733 | 0.021822 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.261710 | 0.018791 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.213134 | 0.018435 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.189000 | 0.013387 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.175349 | 0.013250 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k3  |   3 | clean       | 0.487411 | 0.027648 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.1     | 0.213694 | 0.013227 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.2     | 0.186069 | 0.011398 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.3     | 0.175788 | 0.011776 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.4     | 0.169065 | 0.010603 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k4  |   4 | clean       | 0.491214 | 0.030508 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.1     | 0.228520 | 0.014129 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.2     | 0.192386 | 0.011785 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.3     | 0.174311 | 0.012548 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.4     | 0.162279 | 0.014635 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k6  |   6 | clean       | 0.493877 | 0.025650 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.1     | 0.242209 | 0.011094 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.2     | 0.200733 | 0.010322 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.3     | 0.185573 | 0.011661 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.4     | 0.173802 | 0.010096 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | clean       | 0.497918 | 0.025652 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.1     | 0.253472 | 0.012927 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.2     | 0.211883 | 0.010372 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.189263 | 0.012469 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.177562 | 0.012847 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.466794 | 0.021291 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.199539 | 0.014697 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.157822 | 0.014575 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.146469 | 0.018396 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.141147 | 0.019472 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | clean       | 0.462417 | 0.014542 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.1     | 0.148699 | 0.013775 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.2     | 0.117316 | 0.012059 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.3     | 0.107470 | 0.008785 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.110210 | 0.015360 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k3     |   3 | clean       | 0.458696 | 0.019098 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.1     | 0.143298 | 0.011074 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.2     | 0.106481 | 0.012026 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.3     | 0.099620 | 0.010014 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.4     | 0.103352 | 0.013319 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k4     |   4 | clean       | 0.453788 | 0.019279 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.1     | 0.145259 | 0.010010 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.2     | 0.112350 | 0.010339 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.3     | 0.100916 | 0.009048 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.103277 | 0.011761 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k6     |   6 | clean       | 0.446371 | 0.017938 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.1     | 0.136745 | 0.010205 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.2     | 0.106991 | 0.011707 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.3     | 0.099138 | 0.009942 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.4     | 0.101338 | 0.010857 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k8     |   8 | clean       | 0.432178 | 0.015352 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.1     | 0.140834 | 0.012679 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.2     | 0.107994 | 0.010309 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.3     | 0.101573 | 0.010056 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.4     | 0.103125 | 0.010174 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | clean       | 0.440737 | 0.014280 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.243474 | 0.019376 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.217979 | 0.015391 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.204062 | 0.014790 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.191278 | 0.015772 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k3    |   3 | clean       | 0.428905 | 0.010697 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.180449 | 0.022760 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.152612 | 0.018807 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.143224 | 0.014234 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.137256 | 0.011893 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k4    |   4 | clean       | 0.425468 | 0.012441 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.193399 | 0.022512 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.167215 | 0.020773 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.152856 | 0.013309 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.145006 | 0.010742 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k6    |   6 | clean       | 0.431809 | 0.013654 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.208268 | 0.022880 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.179598 | 0.022113 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.164431 | 0.014908 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.156472 | 0.012827 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k8    |   8 | clean       | 0.435848 | 0.014706 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.229321 | 0.021000 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.203955 | 0.017150 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.188719 | 0.012357 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.177206 | 0.012120 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | clean       | 0.416754 | 0.014600 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.1     | 0.235704 | 0.012103 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.2     | 0.213794 | 0.009742 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.3     | 0.199501 | 0.011776 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.4     | 0.187815 | 0.011375 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k3  |   3 | clean       | 0.413598 | 0.013945 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.1     | 0.205620 | 0.018380 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.2     | 0.177586 | 0.016545 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.3     | 0.164584 | 0.016868 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.4     | 0.152401 | 0.014669 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k4  |   4 | clean       | 0.409377 | 0.013771 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.1     | 0.207613 | 0.015797 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.2     | 0.182520 | 0.016223 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.3     | 0.167963 | 0.014786 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.4     | 0.155305 | 0.012142 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k6  |   6 | clean       | 0.413452 | 0.012339 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.1     | 0.215779 | 0.017178 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.2     | 0.197851 | 0.013711 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.3     | 0.181765 | 0.012830 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.4     | 0.171448 | 0.013998 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k8  |   8 | clean       | 0.411019 | 0.015747 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.1     | 0.228154 | 0.014076 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.2     | 0.207189 | 0.012627 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.3     | 0.195098 | 0.012994 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.4     | 0.183727 | 0.012209 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | clean       | 0.375896 | 0.009855 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.154059 | 0.018711 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.132870 | 0.013476 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.122513 | 0.017691 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.112062 | 0.020275 |     15 |
| dronerf       | matched    | linsvm       | emd      | emd_k2     |   2 | nb3_0.4     | 0.249657 | 0.009036 |     15 |
| dronerf       | matched    | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.256314 | 0.009970 |     15 |
| dronerf       | matched    | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.251905 | 0.007976 |     15 |
| dronerf       | matched    | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.261931 | 0.008459 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.590986 | 0.022917 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.194352 | 0.006833 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.177536 | 0.007622 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.161974 | 0.010973 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.148777 | 0.007339 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.587610 | 0.021862 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.141352 | 0.004366 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.136729 | 0.005235 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.133192 | 0.003675 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.131051 | 0.004664 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k4     |   4 | clean       | 0.582301 | 0.022211 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.1     | 0.138932 | 0.004054 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.2     | 0.133510 | 0.004137 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.3     | 0.132073 | 0.003946 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.4     | 0.130447 | 0.004655 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.577038 | 0.022831 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.137190 | 0.005605 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.2     | 0.133289 | 0.004655 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.3     | 0.132464 | 0.004272 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.4     | 0.129698 | 0.004753 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k8     |   8 | clean       | 0.575351 | 0.020908 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.1     | 0.137561 | 0.004441 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.2     | 0.133873 | 0.004608 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.3     | 0.132151 | 0.003884 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.4     | 0.131052 | 0.004609 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.609172 | 0.015951 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.218512 | 0.008647 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.198112 | 0.012489 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.185313 | 0.010382 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.171972 | 0.010516 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k3    |   3 | clean       | 0.598531 | 0.018600 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.207586 | 0.014004 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.187922 | 0.013636 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.171625 | 0.013095 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.162535 | 0.014920 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k4    |   4 | clean       | 0.598888 | 0.016491 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.208312 | 0.014592 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.181221 | 0.015868 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.163671 | 0.015070 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.151062 | 0.012043 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k6    |   6 | clean       | 0.603466 | 0.017534 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.211652 | 0.010823 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.189582 | 0.011664 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.175996 | 0.011265 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.160406 | 0.009460 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k8    |   8 | clean       | 0.605486 | 0.014959 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.217533 | 0.012530 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.193890 | 0.012616 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.175347 | 0.011271 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.165854 | 0.013917 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.659413 | 0.016836 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.156742 | 0.011274 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.141367 | 0.006328 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.134103 | 0.005751 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.131973 | 0.004512 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k3  |   3 | clean       | 0.647508 | 0.019642 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.1     | 0.149227 | 0.006589 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.2     | 0.134803 | 0.005245 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.3     | 0.129680 | 0.004694 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.4     | 0.125291 | 0.003330 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k4  |   4 | clean       | 0.651745 | 0.018614 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.1     | 0.151846 | 0.008605 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.2     | 0.138036 | 0.005318 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.3     | 0.131241 | 0.004341 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.4     | 0.125659 | 0.005628 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | clean       | 0.655601 | 0.016980 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.1     | 0.165053 | 0.011245 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.2     | 0.148601 | 0.008894 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.3     | 0.137595 | 0.006356 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.4     | 0.131676 | 0.007044 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k8  |   8 | clean       | 0.656247 | 0.015252 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.1     | 0.160243 | 0.010467 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.2     | 0.146314 | 0.008972 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.136287 | 0.009017 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.130993 | 0.007736 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.585324 | 0.025684 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.161756 | 0.007698 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.148466 | 0.007900 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.142920 | 0.005591 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.138177 | 0.004237 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | clean       | 0.414039 | 0.012176 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.1     | 0.158578 | 0.011446 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.2     | 0.139945 | 0.009156 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.3     | 0.132863 | 0.007428 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.4     | 0.130726 | 0.006936 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | clean       | 0.409844 | 0.016783 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.1     | 0.159611 | 0.010302 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.2     | 0.143392 | 0.009818 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.3     | 0.138442 | 0.009877 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.4     | 0.135541 | 0.009841 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k4     |   4 | clean       | 0.406988 | 0.014693 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.1     | 0.151919 | 0.010453 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.2     | 0.136757 | 0.008651 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.3     | 0.134518 | 0.008134 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.133304 | 0.007073 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k6     |   6 | clean       | 0.394129 | 0.013983 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.1     | 0.153268 | 0.009431 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.2     | 0.137544 | 0.008608 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.3     | 0.136186 | 0.008292 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.4     | 0.134743 | 0.009810 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k8     |   8 | clean       | 0.388191 | 0.011679 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.1     | 0.146852 | 0.010437 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.2     | 0.132970 | 0.008779 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.3     | 0.130683 | 0.009868 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.4     | 0.130544 | 0.008991 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | clean       | 0.426353 | 0.010413 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.114276 | 0.011617 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.105064 | 0.013195 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.108424 | 0.011447 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.111193 | 0.009752 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k3    |   3 | clean       | 0.403121 | 0.010978 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.101957 | 0.010385 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.090587 | 0.011150 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.092980 | 0.012010 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.099786 | 0.010477 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k4    |   4 | clean       | 0.409153 | 0.011075 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.105856 | 0.011790 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.094926 | 0.013707 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.097019 | 0.013279 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.101042 | 0.013301 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k6    |   6 | clean       | 0.422359 | 0.011948 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.102332 | 0.011940 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.094912 | 0.013882 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.097034 | 0.014224 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.103030 | 0.012442 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | clean       | 0.429283 | 0.010346 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.108464 | 0.013527 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.100960 | 0.015845 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.103134 | 0.012382 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.107525 | 0.010823 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.504870 | 0.015033 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.136412 | 0.012093 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.2     | 0.125712 | 0.008900 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.3     | 0.123258 | 0.006046 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.122430 | 0.006076 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k3  |   3 | clean       | 0.476041 | 0.014738 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.1     | 0.124444 | 0.006226 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.2     | 0.112337 | 0.009853 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.3     | 0.113896 | 0.008665 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.4     | 0.116280 | 0.007138 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k4  |   4 | clean       | 0.484510 | 0.014100 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.1     | 0.124826 | 0.006254 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.2     | 0.116457 | 0.008986 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.3     | 0.116607 | 0.007724 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.4     | 0.118381 | 0.005341 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k6  |   6 | clean       | 0.500503 | 0.014301 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.1     | 0.127249 | 0.007447 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.2     | 0.120241 | 0.005524 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.3     | 0.118739 | 0.003734 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.4     | 0.119146 | 0.003533 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k8  |   8 | clean       | 0.503709 | 0.015249 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.1     | 0.132050 | 0.009057 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.2     | 0.121842 | 0.006611 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.3     | 0.119267 | 0.004231 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.4     | 0.120499 | 0.002588 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.348788 | 0.015338 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.176991 | 0.006960 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.164069 | 0.013559 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.139261 | 0.022344 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.122124 | 0.010596 |     15 |
| esc50_animals | matched    | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.566014 | 0.026865 |     15 |
| esc50_animals | matched    | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.601639 | 0.042996 |     15 |
| esc50_animals | matched    | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.601009 | 0.038363 |     15 |
| esc50_animals | matched    | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.438038 | 0.030736 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.573424 | 0.041304 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.367263 | 0.020455 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.277430 | 0.042423 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.221962 | 0.036913 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.198299 | 0.039479 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.586160 | 0.038094 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.368244 | 0.038769 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.256951 | 0.042439 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.209115 | 0.039154 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.187201 | 0.034484 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k4     |   4 | clean       | 0.589502 | 0.043265 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.1     | 0.366318 | 0.043010 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.2     | 0.263879 | 0.044633 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.3     | 0.212228 | 0.040003 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.4     | 0.188627 | 0.036660 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.594131 | 0.043076 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.371810 | 0.037947 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.2     | 0.265493 | 0.047515 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.3     | 0.210552 | 0.045293 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.4     | 0.184108 | 0.040063 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k8     |   8 | clean       | 0.593638 | 0.038813 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.1     | 0.369995 | 0.035346 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.2     | 0.266331 | 0.044717 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.3     | 0.207784 | 0.037068 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.4     | 0.181467 | 0.034104 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.628279 | 0.037024 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.473946 | 0.036992 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.409785 | 0.043935 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.390987 | 0.045642 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.380295 | 0.039924 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k3    |   3 | clean       | 0.591001 | 0.035804 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.433646 | 0.065901 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.350529 | 0.061755 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.323936 | 0.062123 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.324924 | 0.046362 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k4    |   4 | clean       | 0.599744 | 0.040102 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.453832 | 0.056757 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.363411 | 0.056744 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.331191 | 0.061837 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.322988 | 0.046476 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k6    |   6 | clean       | 0.615588 | 0.042050 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.467070 | 0.057610 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.384129 | 0.053609 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.359795 | 0.048771 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.346164 | 0.036689 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k8    |   8 | clean       | 0.623322 | 0.033499 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.467213 | 0.042750 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.397303 | 0.046254 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.380495 | 0.047906 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.370484 | 0.042903 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.623557 | 0.045645 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.430943 | 0.046109 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.340989 | 0.054771 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.309036 | 0.066539 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.304021 | 0.051942 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k3  |   3 | clean       | 0.605906 | 0.043259 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.1     | 0.387463 | 0.053981 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.2     | 0.294489 | 0.053028 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.3     | 0.265246 | 0.050070 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.4     | 0.268474 | 0.036202 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k4  |   4 | clean       | 0.608812 | 0.038655 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.1     | 0.422855 | 0.053170 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.2     | 0.308861 | 0.059273 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.3     | 0.278124 | 0.056191 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.4     | 0.281540 | 0.041066 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k6  |   6 | clean       | 0.618854 | 0.041662 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.1     | 0.429642 | 0.058824 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.2     | 0.334092 | 0.056223 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.3     | 0.297864 | 0.051326 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.4     | 0.296123 | 0.041334 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k8  |   8 | clean       | 0.618353 | 0.043091 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.1     | 0.419180 | 0.053897 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.2     | 0.327478 | 0.062312 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.302450 | 0.059573 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.297214 | 0.043750 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.519943 | 0.028011 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.287538 | 0.027557 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.226136 | 0.030865 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.211823 | 0.034347 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.216617 | 0.030782 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | clean       | 0.550538 | 0.040781 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.1     | 0.256700 | 0.017628 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.2     | 0.206868 | 0.016175 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.3     | 0.160925 | 0.018838 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.4     | 0.134248 | 0.015154 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k3     |   3 | clean       | 0.557280 | 0.047989 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.1     | 0.255889 | 0.024380 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.2     | 0.198529 | 0.009769 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.3     | 0.156827 | 0.014085 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.4     | 0.138001 | 0.017358 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | clean       | 0.561695 | 0.051406 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.1     | 0.276763 | 0.024412 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.2     | 0.201088 | 0.016906 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.3     | 0.157995 | 0.020289 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.138591 | 0.021386 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k6     |   6 | clean       | 0.550741 | 0.046339 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.1     | 0.264895 | 0.017743 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.2     | 0.195039 | 0.016636 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.3     | 0.155710 | 0.015886 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k6     |   6 | nb3_0.4     | 0.137143 | 0.019246 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k8     |   8 | clean       | 0.546219 | 0.048351 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.1     | 0.261919 | 0.015538 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.2     | 0.196233 | 0.017756 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.3     | 0.156794 | 0.016344 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k8     |   8 | nb3_0.4     | 0.137035 | 0.021103 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | clean       | 0.601983 | 0.029505 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.420553 | 0.030282 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.341662 | 0.022203 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.294389 | 0.024516 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.267022 | 0.029177 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k3    |   3 | clean       | 0.574894 | 0.032223 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.362200 | 0.025446 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.337324 | 0.019432 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.313710 | 0.015392 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.289745 | 0.015708 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k4    |   4 | clean       | 0.589818 | 0.034422 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.383225 | 0.026646 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.345022 | 0.022725 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.320781 | 0.025651 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.292817 | 0.033461 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | clean       | 0.601014 | 0.035191 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.405006 | 0.028801 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.352845 | 0.024880 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.325004 | 0.030749 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.297679 | 0.035307 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | clean       | 0.599630 | 0.035691 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.419165 | 0.031229 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.353703 | 0.025804 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.313045 | 0.029228 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.281890 | 0.034080 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.609256 | 0.040018 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.426727 | 0.021323 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.2     | 0.346012 | 0.022696 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.3     | 0.303042 | 0.022182 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.275554 | 0.032621 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k3  |   3 | clean       | 0.580432 | 0.035592 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.1     | 0.361605 | 0.028193 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.2     | 0.329730 | 0.017496 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.3     | 0.315049 | 0.023252 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k3  |   3 | nb3_0.4     | 0.295715 | 0.026263 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | clean       | 0.594094 | 0.043582 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.1     | 0.380849 | 0.017134 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.2     | 0.348143 | 0.016890 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.3     | 0.330152 | 0.020989 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.4     | 0.310646 | 0.025794 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k6  |   6 | clean       | 0.601267 | 0.036820 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.1     | 0.390997 | 0.022335 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.2     | 0.340589 | 0.020599 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.3     | 0.314841 | 0.030417 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k6  |   6 | nb3_0.4     | 0.296359 | 0.035082 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k8  |   8 | clean       | 0.605194 | 0.038820 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.1     | 0.413528 | 0.021647 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.2     | 0.341562 | 0.020425 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.3     | 0.304721 | 0.028081 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k8  |   8 | nb3_0.4     | 0.279366 | 0.032891 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.443390 | 0.042978 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.223748 | 0.016872 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.209262 | 0.007786 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.193830 | 0.010721 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.176377 | 0.026003 |     15 |
| synthetic     | matched    | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.321105 | 0.033474 |     15 |
| synthetic     | matched    | logreg       | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.597480 | 0.043984 |     15 |
| synthetic     | matched    | logreg       | linear   | linear_k6  |   6 | nb3_0.4     | 0.472035 | 0.028085 |     15 |
| synthetic     | matched    | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.339689 | 0.035203 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.380343 | 0.052603 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.337371 | 0.050576 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.308485 | 0.049880 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.260042 | 0.035188 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.239685 | 0.034829 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.421010 | 0.042275 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.1     | 0.328291 | 0.030788 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.2     | 0.276580 | 0.030321 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.3     | 0.250895 | 0.032747 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | nb3_0.4     | 0.232549 | 0.032448 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | clean       | 0.403665 | 0.036863 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.1     | 0.325078 | 0.039209 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.2     | 0.299979 | 0.030680 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.3     | 0.272241 | 0.036635 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.4     | 0.245404 | 0.033030 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.417694 | 0.040782 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.325601 | 0.030280 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.2     | 0.295301 | 0.032142 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.3     | 0.268646 | 0.024381 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.4     | 0.245373 | 0.029695 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | clean       | 0.414624 | 0.041881 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.1     | 0.337383 | 0.043141 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.2     | 0.297286 | 0.031122 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.3     | 0.269105 | 0.030411 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.4     | 0.248829 | 0.027652 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k3    |   3 | clean       | 0.675579 | 0.049475 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.609782 | 0.045637 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.494255 | 0.046804 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.407811 | 0.041862 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.354426 | 0.037295 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k4    |   4 | clean       | 0.714376 | 0.052986 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.627463 | 0.042205 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.513057 | 0.041581 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.419322 | 0.045302 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.353042 | 0.039907 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k6    |   6 | clean       | 0.749274 | 0.046256 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.661276 | 0.051304 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.556907 | 0.041362 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.446695 | 0.036715 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.370194 | 0.029181 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | clean       | 0.765132 | 0.050838 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.676703 | 0.058711 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.579144 | 0.071004 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.450207 | 0.053187 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.370892 | 0.044602 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | clean       | 0.778077 | 0.051871 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.688646 | 0.062617 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.2     | 0.565020 | 0.061471 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.3     | 0.449081 | 0.061042 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.369970 | 0.059056 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k3  |   3 | clean       | 0.560703 | 0.049771 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.1     | 0.496473 | 0.034455 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.2     | 0.404579 | 0.040582 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.3     | 0.337296 | 0.035653 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k3  |   3 | nb3_0.4     | 0.301802 | 0.033983 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k4  |   4 | clean       | 0.599823 | 0.050261 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.1     | 0.499426 | 0.051300 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.2     | 0.402307 | 0.058071 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.3     | 0.346438 | 0.042006 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k4  |   4 | nb3_0.4     | 0.322213 | 0.040947 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k6  |   6 | clean       | 0.654756 | 0.045210 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.1     | 0.527886 | 0.035942 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.2     | 0.425738 | 0.046853 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.3     | 0.365623 | 0.039364 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.4     | 0.341665 | 0.039201 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | clean       | 0.655948 | 0.045743 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.1     | 0.559239 | 0.043942 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.2     | 0.458298 | 0.041841 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.400987 | 0.036415 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.354890 | 0.034603 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | clean       | 0.660875 | 0.050018 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.1     | 0.561903 | 0.036846 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.2     | 0.464989 | 0.040403 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.3     | 0.390211 | 0.036061 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.4     | 0.345470 | 0.026254 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.344496 | 0.052852 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.352821 | 0.054920 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.324926 | 0.039276 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.323485 | 0.048995 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.305507 | 0.044857 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k2     |   2 | clean       | 0.495205 | 0.050526 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.1     | 0.429033 | 0.044972 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.2     | 0.339017 | 0.040557 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.3     | 0.292580 | 0.033480 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.253633 | 0.033472 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k3     |   3 | clean       | 0.535380 | 0.054329 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.1     | 0.443569 | 0.040151 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.2     | 0.369361 | 0.039254 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.3     | 0.334043 | 0.030788 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k3     |   3 | nb3_0.4     | 0.292030 | 0.032638 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | clean       | 0.548418 | 0.048763 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.1     | 0.450450 | 0.026238 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.2     | 0.382772 | 0.035453 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.3     | 0.339237 | 0.030190 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.307533 | 0.035831 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k6     |   6 | clean       | 0.515471 | 0.043082 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.1     | 0.432290 | 0.031727 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.2     | 0.359985 | 0.042879 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.3     | 0.324628 | 0.037626 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k6     |   6 | nb3_0.4     | 0.298938 | 0.032767 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k8     |   8 | clean       | 0.506434 | 0.040396 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.1     | 0.407323 | 0.028483 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.2     | 0.340344 | 0.030912 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.3     | 0.324050 | 0.032244 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k8     |   8 | nb3_0.4     | 0.294953 | 0.029391 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k3    |   3 | clean       | 0.764302 | 0.030761 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.1     | 0.680604 | 0.033319 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.2     | 0.586176 | 0.039489 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.3     | 0.521251 | 0.039212 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k3    |   3 | nb3_0.4     | 0.485147 | 0.044329 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k4    |   4 | clean       | 0.777637 | 0.040371 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.1     | 0.695168 | 0.042131 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.2     | 0.610682 | 0.049492 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.3     | 0.553578 | 0.041709 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k4    |   4 | nb3_0.4     | 0.495854 | 0.046604 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k6    |   6 | clean       | 0.804425 | 0.024235 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.1     | 0.714515 | 0.033763 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.2     | 0.632491 | 0.049250 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.562498 | 0.043684 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.500474 | 0.047144 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | clean       | 0.795475 | 0.030153 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.1     | 0.732055 | 0.028823 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.651802 | 0.043996 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.577817 | 0.040534 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.515320 | 0.044283 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | clean       | 0.810562 | 0.025773 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.738305 | 0.047685 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.2     | 0.656438 | 0.050827 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.3     | 0.588132 | 0.040003 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.4     | 0.511439 | 0.033784 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k3  |   3 | clean       | 0.664270 | 0.041272 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.1     | 0.560796 | 0.045501 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.2     | 0.466187 | 0.041985 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.3     | 0.431267 | 0.043566 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k3  |   3 | nb3_0.4     | 0.410658 | 0.036805 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k4  |   4 | clean       | 0.674711 | 0.037808 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.1     | 0.562090 | 0.040306 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.2     | 0.488394 | 0.033005 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.3     | 0.448419 | 0.039972 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k4  |   4 | nb3_0.4     | 0.414587 | 0.031532 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | clean       | 0.688859 | 0.041277 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.1     | 0.577750 | 0.034255 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.2     | 0.510174 | 0.041077 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.3     | 0.459072 | 0.039938 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | nb3_0.4     | 0.418961 | 0.037718 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k8  |   8 | clean       | 0.681657 | 0.037339 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.1     | 0.585262 | 0.042609 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.2     | 0.515567 | 0.031730 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.3     | 0.469342 | 0.038562 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k8  |   8 | nb3_0.4     | 0.430766 | 0.025930 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | clean       | 0.686609 | 0.035668 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.1     | 0.602792 | 0.042719 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.2     | 0.528562 | 0.034309 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.3     | 0.480792 | 0.031835 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.4     | 0.437966 | 0.025772 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | clean       | 0.480904 | 0.049685 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.442697 | 0.035932 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.388441 | 0.031430 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.349624 | 0.023862 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.325640 | 0.027871 |     15 |

## Descriptive best K at each condition

These maxima use the condition's own outer-fold scores and do not define the hypothesis tests.

| dataset       | scenario   | classifier   | family   | variant    |   k | condition   |     mean |      std |   size |
|:--------------|:-----------|:-------------|:---------|:-----------|----:|:------------|---------:|---------:|-------:|
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | clean       | 0.810562 | 0.025773 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | clean       | 0.778077 | 0.051871 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.738305 | 0.047685 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k6  |   6 | clean       | 0.688859 | 0.041277 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k9    |   9 | nb3_0.1     | 0.688646 | 0.062617 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | clean       | 0.660875 | 0.050018 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.659413 | 0.016836 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.2     | 0.656438 | 0.050827 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.628279 | 0.037024 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.623557 | 0.045645 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.609256 | 0.040018 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.609172 | 0.015951 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.1     | 0.602792 | 0.042719 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | clean       | 0.601983 | 0.029505 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | clean       | 0.594131 | 0.043076 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | clean       | 0.590986 | 0.022917 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k9    |   9 | nb3_0.3     | 0.588132 | 0.040003 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.585324 | 0.025684 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.579144 | 0.071004 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.1     | 0.561903 | 0.036846 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | clean       | 0.561695 | 0.051406 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | clean       | 0.548418 | 0.048763 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.2     | 0.528562 | 0.034309 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.519943 | 0.028011 |     15 |
| synthetic     | shift      | logreg       | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.515320 | 0.044283 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.505118 | 0.019362 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | clean       | 0.504927 | 0.024422 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | clean       | 0.504870 | 0.015033 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | clean       | 0.500733 | 0.021822 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | clean       | 0.480904 | 0.049685 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.3     | 0.480792 | 0.031835 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.473946 | 0.036992 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.466794 | 0.021291 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k9  |   9 | nb3_0.2     | 0.464989 | 0.040403 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | clean       | 0.462417 | 0.014542 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.1     | 0.450450 | 0.026238 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.3     | 0.450207 | 0.053187 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.443390 | 0.042978 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.442697 | 0.035932 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | clean       | 0.440737 | 0.014280 |     15 |
| synthetic     | shift      | logreg       | linear   | linear_k9  |   9 | nb3_0.4     | 0.437966 | 0.025772 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.430943 | 0.046109 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k8    |   8 | clean       | 0.429283 | 0.010346 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.426727 | 0.021323 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k3     |   3 | clean       | 0.421010 | 0.042275 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.420553 | 0.030282 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | clean       | 0.416754 | 0.014600 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k2     |   2 | clean       | 0.414039 | 0.012176 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.409785 | 0.043935 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.400987 | 0.036415 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.390987 | 0.045642 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.388441 | 0.031430 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.2     | 0.382772 | 0.035453 |     15 |
| esc50_animals | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.380295 | 0.039924 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | clean       | 0.375896 | 0.009855 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k6     |   6 | nb3_0.1     | 0.371810 | 0.037947 |     15 |
| synthetic     | shift      | lightgbm     | irmf     | irmf_k8    |   8 | nb3_0.4     | 0.370892 | 0.044602 |     15 |
| synthetic     | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.354890 | 0.034603 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k8    |   8 | nb3_0.2     | 0.353703 | 0.025804 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.352821 | 0.054920 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.349624 | 0.023862 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | clean       | 0.348788 | 0.015338 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.2     | 0.348143 | 0.016890 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | clean       | 0.344496 | 0.052852 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.340989 | 0.054771 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.3     | 0.339237 | 0.030190 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.1     | 0.337383 | 0.043141 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.3     | 0.330152 | 0.020989 |     15 |
| synthetic     | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.325640 | 0.027871 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.3     | 0.325004 | 0.030749 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.324926 | 0.039276 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.323485 | 0.048995 |     15 |
| esc50_animals | shift      | linsvm       | linear   | linear_k4  |   4 | nb3_0.4     | 0.310646 | 0.025794 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.3     | 0.309036 | 0.066539 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.308485 | 0.049880 |     15 |
| synthetic     | shift      | logreg       | emd      | emd_k4     |   4 | nb3_0.4     | 0.307533 | 0.035831 |     15 |
| synthetic     | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.305507 | 0.044857 |     15 |
| esc50_animals | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.304021 | 0.051942 |     15 |
| esc50_animals | shift      | linsvm       | irmf     | irmf_k6    |   6 | nb3_0.4     | 0.297679 | 0.035307 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.287538 | 0.027557 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.277430 | 0.042423 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.1     | 0.276763 | 0.024412 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k4     |   4 | nb3_0.3     | 0.272241 | 0.036635 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.1     | 0.261710 | 0.018791 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.258805 | 0.019081 |     15 |
| synthetic     | shift      | lightgbm     | emd      | emd_k8     |   8 | nb3_0.4     | 0.248829 | 0.027652 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.243474 | 0.019376 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.1     | 0.235704 | 0.012103 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.226136 | 0.030865 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.223748 | 0.016872 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.221962 | 0.036913 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.218512 | 0.008647 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.217979 | 0.015391 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.216617 | 0.030782 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.2     | 0.213794 | 0.009742 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.213719 | 0.012893 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.2     | 0.213134 | 0.018435 |     15 |
| esc50_animals | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.211823 | 0.034347 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.209262 | 0.007786 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.2     | 0.206868 | 0.016175 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.204062 | 0.014790 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.199539 | 0.014697 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.3     | 0.199501 | 0.011776 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.198415 | 0.014233 |     15 |
| esc50_animals | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.198299 | 0.039479 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.198112 | 0.012489 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.194352 | 0.006833 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.193830 | 0.010721 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.1     | 0.191470 | 0.011479 |     15 |
| birdclef2023  | shift      | logreg       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.191278 | 0.015772 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.3     | 0.189263 | 0.012469 |     15 |
| birdclef2023  | shift      | logreg       | linear   | linear_k11 |  11 | nb3_0.4     | 0.187815 | 0.011375 |     15 |
| birdclef2023  | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.187179 | 0.016528 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.185313 | 0.010382 |     15 |
| birdclef2023  | shift      | lightgbm     | linear   | linear_k8  |   8 | nb3_0.4     | 0.177562 | 0.012847 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.177536 | 0.007622 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.1     | 0.176991 | 0.006960 |     15 |
| esc50_animals | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.176377 | 0.026003 |     15 |
| dronerf       | shift      | lightgbm     | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.171972 | 0.010516 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.1     | 0.165053 | 0.011245 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.2     | 0.164069 | 0.013559 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.161974 | 0.010973 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.1     | 0.161756 | 0.007698 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k2     |   2 | nb3_0.3     | 0.160925 | 0.018838 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.1     | 0.159611 | 0.010302 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.2     | 0.158939 | 0.013636 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.157822 | 0.014575 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.1     | 0.154059 | 0.018711 |     15 |
| dronerf       | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.148777 | 0.007339 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.1     | 0.148699 | 0.013775 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.2     | 0.148601 | 0.008894 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.2     | 0.148466 | 0.007900 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.146469 | 0.018396 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.2     | 0.143392 | 0.009818 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.3     | 0.142920 | 0.005591 |     15 |
| birdclef2023  | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.141147 | 0.019472 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.3     | 0.139261 | 0.022344 |     15 |
| esc50_animals | shift      | linsvm       | emd      | emd_k4     |   4 | nb3_0.4     | 0.138591 | 0.021386 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.3     | 0.138442 | 0.009877 |     15 |
| dronerf       | shift      | lightgbm     | raw      | raw        |   0 | nb3_0.4     | 0.138177 | 0.004237 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k6  |   6 | nb3_0.3     | 0.137595 | 0.006356 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.1     | 0.136412 | 0.012093 |     15 |
| dronerf       | shift      | linsvm       | emd      | emd_k3     |   3 | nb3_0.4     | 0.135541 | 0.009841 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.2     | 0.132870 | 0.013476 |     15 |
| dronerf       | shift      | lightgbm     | linear   | linear_k11 |  11 | nb3_0.4     | 0.131973 | 0.004512 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.3     | 0.131860 | 0.009217 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.2     | 0.125712 | 0.008900 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.3     | 0.123258 | 0.006046 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.3     | 0.122513 | 0.017691 |     15 |
| dronerf       | shift      | linsvm       | linear   | linear_k11 |  11 | nb3_0.4     | 0.122430 | 0.006076 |     15 |
| dronerf       | shift      | linsvm       | raw      | raw        |   0 | nb3_0.4     | 0.122124 | 0.010596 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.2     | 0.117316 | 0.012059 |     15 |
| birdclef2023  | shift      | lightgbm     | emd      | emd_k2     |   2 | nb3_0.4     | 0.114662 | 0.008047 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.1     | 0.114276 | 0.011617 |     15 |
| birdclef2023  | shift      | logreg       | raw      | raw        |   0 | nb3_0.4     | 0.112062 | 0.020275 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.4     | 0.111193 | 0.009752 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.4     | 0.110210 | 0.015360 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.3     | 0.108424 | 0.011447 |     15 |
| birdclef2023  | shift      | logreg       | emd      | emd_k2     |   2 | nb3_0.3     | 0.107470 | 0.008785 |     15 |
| dronerf       | shift      | linsvm       | irmf     | irmf_k11   |  11 | nb3_0.2     | 0.105064 | 0.013195 |     15 |

## Paired tests

| dataset       | family   | classifier   | feature_set   | scenario   | comparator   | condition   | contrast   |   irmf_k |   comparator_k |   n_splits | holm_family   | status                     |   mean_diff |    ci_low |   ci_high |   se_corrected |         t |   df |     p_nb |   n_test_over_n_train |   wilcoxon_stat |   p_wilcoxon |   p_nb_holm |   p_wilcoxon_holm | holm_complete   |
|:--------------|:---------|:-------------|:--------------|:-----------|:-------------|:------------|:-----------|---------:|---------------:|-----------:|:--------------|:---------------------------|------------:|----------:|----------:|---------------:|----------:|-----:|---------:|----------------------:|----------------:|-------------:|------------:|------------------:|:----------------|
| synthetic     | Primary  | logreg       | comp          | shift      | linear       | nb3_0.1     | DiD        |        9 |              6 |         15 | synthetic     | synthetic, separate family |    0.038852 | -0.038107 |  0.115811 |       0.035882 |  1.082769 |   14 | 0.297208 |              0.250010 |       24.000000 |     0.041260 |    1.000000 |          0.495117 | True            |
| synthetic     | Primary  | logreg       | comp          | shift      | linear       | nb3_0.2     | DiD        |        9 |              6 |         15 | synthetic     | synthetic, separate family |    0.024561 | -0.044775 |  0.093897 |       0.032328 |  0.759760 |   14 | 0.460001 |              0.250010 |       35.000000 |     0.168823 |    1.000000 |          1.000000 | True            |
| synthetic     | Primary  | logreg       | comp          | shift      | linear       | nb3_0.3     | DiD        |        9 |              6 |         15 | synthetic     | synthetic, separate family |    0.007357 | -0.062111 |  0.076824 |       0.032389 |  0.227134 |   14 | 0.823603 |              0.250010 |       46.000000 |     0.454285 |    1.000000 |          1.000000 | True            |
| synthetic     | Primary  | logreg       | comp          | shift      | linear       | nb3_0.4     | DiD        |        9 |              6 |         15 | synthetic     | synthetic, separate family |   -0.029226 | -0.091248 |  0.032796 |       0.028918 | -1.010665 |   14 | 0.329326 |              0.250010 |       25.000000 |     0.047913 |    1.000000 |          0.527039 | True            |
| synthetic     | S1       | logreg       | comp          | shift      | emd          | nb3_0.1     | DiD        |        9 |              4 |         15 | synthetic     | synthetic, separate family |    0.025711 | -0.049644 |  0.101066 |       0.035134 |  0.731793 |   14 | 0.476367 |              0.250010 |       36.000000 |     0.187622 |    1.000000 |          1.000000 | True            |
| synthetic     | S1       | logreg       | comp          | shift      | emd          | nb3_0.2     | DiD        |        9 |              4 |         15 | synthetic     | synthetic, separate family |    0.011522 | -0.088553 |  0.111597 |       0.046660 |  0.246944 |   14 | 0.808536 |              0.250010 |       50.000000 |     0.599487 |    1.000000 |          1.000000 | True            |
| synthetic     | S1       | logreg       | comp          | shift      | emd          | nb3_0.3     | DiD        |        9 |              4 |         15 | synthetic     | synthetic, separate family |   -0.013249 | -0.097272 |  0.070774 |       0.039175 | -0.338201 |   14 | 0.740230 |              0.250010 |       54.000000 |     0.761536 |    1.000000 |          1.000000 | True            |
| synthetic     | S1       | logreg       | comp          | shift      | emd          | nb3_0.4     | DiD        |        9 |              4 |         15 | synthetic     | synthetic, separate family |   -0.058239 | -0.135166 |  0.018689 |       0.035867 | -1.623724 |   14 | 0.126729 |              0.250010 |        6.000000 |     0.000854 |    1.000000 |          0.012817 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.1     | DiD        |        9 |              9 |         15 | synthetic     | synthetic, separate family |    0.009541 | -0.062286 |  0.081368 |       0.033489 |  0.284889 |   14 | 0.779900 |              0.250010 |       53.000000 |     0.719727 |    1.000000 |          1.000000 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.2     | DiD        |        9 |              9 |         15 | synthetic     | synthetic, separate family |   -0.017172 | -0.097497 |  0.063153 |       0.037451 | -0.458505 |   14 | 0.653627 |              0.250010 |       45.000000 |     0.421204 |    1.000000 |          1.000000 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.3     | DiD        |        9 |              9 |         15 | synthetic     | synthetic, separate family |   -0.058333 | -0.152330 |  0.035664 |       0.043826 | -1.331017 |   14 | 0.204454 |              0.250010 |       16.000000 |     0.010254 |    1.000000 |          0.133301 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.4     | DiD        |        9 |              9 |         15 | synthetic     | synthetic, separate family |   -0.092702 | -0.178896 | -0.006507 |       0.040188 | -2.306707 |   14 | 0.036878 |              0.250010 |        4.000000 |     0.000427 |    0.590050 |          0.006836 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.1     | DiD        |        9 |              3 |         15 | synthetic     | synthetic, separate family |    0.003288 | -0.048791 |  0.055367 |       0.024282 |  0.135394 |   14 | 0.894228 |              0.250010 |       59.000000 |     0.977966 |    1.000000 |          1.000000 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.2     | DiD        |        9 |              3 |         15 | synthetic     | synthetic, separate family |   -0.068628 | -0.140645 |  0.003390 |       0.033578 | -2.043833 |   14 | 0.060258 |              0.250010 |        8.000000 |     0.001526 |    0.903868 |          0.021362 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.3     | DiD        |        9 |              3 |         15 | synthetic     | synthetic, separate family |   -0.158881 | -0.273534 | -0.044228 |       0.053457 | -2.972158 |   14 | 0.010093 |              0.250010 |        3.000000 |     0.000305 |    0.171585 |          0.005188 | True            |
| synthetic     | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.4     | DiD        |        9 |              3 |         15 | synthetic     | synthetic, separate family |   -0.219646 | -0.316694 | -0.122597 |       0.045248 | -4.854216 |   14 | 0.000255 |              0.250010 |        0.000000 |     0.000061 |    0.004594 |          0.001099 | True            |
| synthetic     | S5       | logreg       | comp          | matched    | linear       | nb3_0.4     | DiD        |        9 |              6 |         15 | synthetic     | synthetic, separate family |    0.003742 | -0.058921 |  0.066404 |       0.029216 |  0.128065 |   14 | 0.899919 |              0.250010 |       55.000000 |     0.803955 |    1.000000 |          1.000000 | True            |
| synthetic     | S5       | logreg       | comp          | matched    | emd          | nb3_0.4     | DiD        |        9 |              4 |         15 | synthetic     | synthetic, separate family |    0.014231 | -0.079684 |  0.108146 |       0.043788 |  0.324997 |   14 | 0.749989 |              0.250010 |       47.000000 |     0.488708 |    1.000000 |          1.000000 | True            |
| dronerf       | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.1     | DiD        |        8 |             11 |         15 | Primary       | primary                    |    0.047639 |  0.026780 |  0.068497 |       0.009725 |  4.898435 |   14 | 0.000235 |              0.250008 |        0.000000 |     0.000061 |    0.002115 |          0.000732 | True            |
| dronerf       | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.2     | DiD        |        8 |             11 |         15 | Primary       | primary                    |    0.050835 |  0.035079 |  0.066592 |       0.007347 |  6.919621 |   14 | 0.000007 |              0.250008 |        0.000000 |     0.000061 |    0.000071 |          0.000732 | True            |
| dronerf       | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.3     | DiD        |        8 |             11 |         15 | Primary       | primary                    |    0.055463 |  0.041824 |  0.069102 |       0.006359 |  8.721994 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000005 |          0.000732 | True            |
| dronerf       | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.4     | DiD        |        8 |             11 |         15 | Primary       | primary                    |    0.060683 |  0.046396 |  0.074970 |       0.006661 |  9.109712 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000004 |          0.000732 | True            |
| dronerf       | S1       | linsvm       | comp          | shift      | emd          | nb3_0.1     | DiD        |        8 |              2 |         15 | S1            | exploratory                |   -0.065359 | -0.082818 | -0.047899 |       0.008140 | -8.028925 |   14 | 0.000001 |              0.250008 |        0.000000 |     0.000061 |    0.000012 |          0.000732 | True            |
| dronerf       | S1       | linsvm       | comp          | shift      | emd          | nb3_0.2     | DiD        |        8 |              2 |         15 | S1            | exploratory                |   -0.054229 | -0.075570 | -0.032888 |       0.009950 | -5.450083 |   14 | 0.000086 |              0.250008 |        0.000000 |     0.000061 |    0.000599 |          0.000732 | True            |
| dronerf       | S1       | linsvm       | comp          | shift      | emd          | nb3_0.3     | DiD        |        8 |              2 |         15 | S1            | exploratory                |   -0.044973 | -0.063256 | -0.026690 |       0.008524 | -5.275875 |   14 | 0.000117 |              0.250008 |        0.000000 |     0.000061 |    0.000703 |          0.000732 | True            |
| dronerf       | S1       | linsvm       | comp          | shift      | emd          | nb3_0.4     | DiD        |        8 |              2 |         15 | S1            | exploratory                |   -0.038445 | -0.056198 | -0.020692 |       0.008277 | -4.644551 |   14 | 0.000379 |              0.250008 |        0.000000 |     0.000061 |    0.001137 |          0.000732 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.1     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.112011 |  0.092863 |  0.131160 |       0.008928 | 12.546138 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000000 |          0.001465 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.2     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.106986 |  0.086650 |  0.127321 |       0.009481 | 11.283800 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000000 |          0.001465 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.3     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.101451 |  0.086322 |  0.116580 |       0.007054 | 14.382120 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000000 |          0.001465 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.090240 |  0.071834 |  0.108645 |       0.008582 | 10.515615 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000001 |          0.001465 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.1     | DiD        |       11 |              2 |         15 | S3            | exploratory                |    0.005974 | -0.013321 |  0.025268 |       0.008996 |  0.664027 |   14 | 0.517460 |              0.250008 |       41.000000 |     0.302795 |    1.000000 |          1.000000 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.2     | DiD        |       11 |              2 |         15 | S3            | exploratory                |    0.002389 | -0.016769 |  0.021547 |       0.008932 |  0.267467 |   14 | 0.793010 |              0.250008 |       51.000000 |     0.638672 |    1.000000 |          1.000000 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.3     | DiD        |       11 |              2 |         15 | S3            | exploratory                |    0.005153 | -0.018555 |  0.028861 |       0.011054 |  0.466153 |   14 | 0.648276 |              0.250008 |       37.000000 |     0.207764 |    1.000000 |          1.000000 | True            |
| dronerf       | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.4     | DiD        |       11 |              2 |         15 | S3            | exploratory                |    0.005009 | -0.018732 |  0.028749 |       0.011069 |  0.452486 |   14 | 0.657851 |              0.250008 |       45.000000 |     0.421204 |    1.000000 |          1.000000 | True            |
| dronerf       | S5       | linsvm       | comp          | matched    | linear       | nb3_0.4     | DiD        |        8 |             11 |         15 | S5            | exploratory                |    0.079997 |  0.068746 |  0.091247 |       0.005246 | 15.250017 |   14 | 0.000000 |              0.250008 |        0.000000 |     0.000061 |    0.000000 |          0.000366 | True            |
| dronerf       | S5       | linsvm       | comp          | matched    | emd          | nb3_0.4     | DiD        |        8 |              2 |         15 | S5            | exploratory                |   -0.008587 | -0.022802 |  0.005629 |       0.006628 | -1.295552 |   14 | 0.216085 |              0.250008 |       20.000000 |     0.021545 |    0.864339 |          0.086182 | True            |
| birdclef2023  | Primary  | logreg       | comp          | shift      | linear       | nb3_0.1     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.016212 | -0.042768 |  0.010343 |       0.012381 | -1.309424 |   14 | 0.211474 |              0.250001 |       17.000000 |     0.012451 |    1.000000 |          0.062256 | True            |
| birdclef2023  | Primary  | logreg       | comp          | shift      | linear       | nb3_0.2     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.019797 | -0.039291 | -0.000302 |       0.009089 | -2.178012 |   14 | 0.046995 |              0.250001 |        6.000000 |     0.000854 |    0.327767 |          0.006836 | True            |
| birdclef2023  | Primary  | logreg       | comp          | shift      | linear       | nb3_0.3     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.019422 | -0.037927 | -0.000917 |       0.008628 | -2.251074 |   14 | 0.040971 |              0.250001 |        6.000000 |     0.000854 |    0.327767 |          0.006836 | True            |
| birdclef2023  | Primary  | logreg       | comp          | shift      | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.020519 | -0.040651 | -0.000387 |       0.009387 | -2.185991 |   14 | 0.046299 |              0.250001 |        7.000000 |     0.001160 |    0.327767 |          0.006958 | True            |
| birdclef2023  | S1       | logreg       | comp          | shift      | emd          | nb3_0.1     | DiD        |       11 |              2 |         15 | S1            | exploratory                |    0.116456 |  0.083224 |  0.149688 |       0.015494 |  7.516056 |   14 | 0.000003 |              0.250001 |        0.000000 |     0.000061 |    0.000022 |          0.000732 | True            |
| birdclef2023  | S1       | logreg       | comp          | shift      | emd          | nb3_0.2     | DiD        |       11 |              2 |         15 | S1            | exploratory                |    0.122344 |  0.099875 |  0.144813 |       0.010476 | 11.678258 |   14 | 0.000000 |              0.250001 |        0.000000 |     0.000061 |    0.000000 |          0.000732 | True            |
| birdclef2023  | S1       | logreg       | comp          | shift      | emd          | nb3_0.3     | DiD        |       11 |              2 |         15 | S1            | exploratory                |    0.118272 |  0.094358 |  0.142186 |       0.011150 | 10.607571 |   14 | 0.000000 |              0.250001 |        0.000000 |     0.000061 |    0.000000 |          0.000732 | True            |
| birdclef2023  | S1       | logreg       | comp          | shift      | emd          | nb3_0.4     | DiD        |       11 |              2 |         15 | S1            | exploratory                |    0.102748 |  0.076387 |  0.129110 |       0.012291 |  8.359748 |   14 | 0.000001 |              0.250001 |        0.000000 |     0.000061 |    0.000008 |          0.000732 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.1     | DiD        |       11 |             11 |         15 | S3            | exploratory                |   -0.007100 | -0.031620 |  0.017421 |       0.011433 | -0.621003 |   14 | 0.544576 |              0.250001 |       36.000000 |     0.187622 |    1.000000 |          1.000000 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.2     | DiD        |       11 |             11 |         15 | S3            | exploratory                |   -0.003609 | -0.027457 |  0.020238 |       0.011119 | -0.324616 |   14 | 0.750271 |              0.250001 |       48.000000 |     0.524475 |    1.000000 |          1.000000 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.3     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.005221 | -0.014420 |  0.024863 |       0.009158 |  0.570135 |   14 | 0.577622 |              0.250001 |       38.000000 |     0.229309 |    1.000000 |          1.000000 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.007636 | -0.014393 |  0.029664 |       0.010271 |  0.743421 |   14 | 0.469520 |              0.250001 |       31.000000 |     0.106995 |    1.000000 |          0.855957 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.1     | DiD        |       11 |              3 |         15 | S3            | exploratory                |    0.072004 |  0.039234 |  0.104775 |       0.015279 |  4.712582 |   14 | 0.000333 |              0.250001 |        0.000000 |     0.000061 |    0.004331 |          0.001465 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.2     | DiD        |       11 |              3 |         15 | S3            | exploratory                |    0.060201 |  0.040119 |  0.080283 |       0.009363 |  6.429478 |   14 | 0.000016 |              0.250001 |        0.000000 |     0.000061 |    0.000236 |          0.001465 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.3     | DiD        |       11 |              3 |         15 | S3            | exploratory                |    0.073319 |  0.050435 |  0.096204 |       0.010670 |  6.871708 |   14 | 0.000008 |              0.250001 |        0.000000 |     0.000061 |    0.000130 |          0.001465 | True            |
| birdclef2023  | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.4     | DiD        |       11 |              3 |         15 | S3            | exploratory                |    0.081404 |  0.054628 |  0.108181 |       0.012484 |  6.520457 |   14 | 0.000014 |              0.250001 |        0.000000 |     0.000061 |    0.000217 |          0.001465 | True            |
| birdclef2023  | S5       | logreg       | comp          | matched    | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | S5            | exploratory                |   -0.009101 | -0.025139 |  0.006936 |       0.007477 | -1.217160 |   14 | 0.243663 |              0.250001 |       20.000000 |     0.021545 |    0.864339 |          0.086182 | True            |
| birdclef2023  | S5       | logreg       | comp          | matched    | emd          | nb3_0.4     | DiD        |       11 |              2 |         15 | S5            | exploratory                |    0.023812 |  0.009529 |  0.038095 |       0.006659 |  3.575628 |   14 | 0.003042 |              0.250001 |        0.000000 |     0.000061 |    0.015211 |          0.000366 | True            |
| esc50_animals | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.1     | DiD        |       11 |             11 |         15 | Primary       | primary                    |    0.001100 | -0.033236 |  0.035435 |       0.016009 |  0.068690 |   14 | 0.946208 |              0.250116 |       57.000000 |     0.890381 |    1.000000 |          1.000000 | True            |
| esc50_animals | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.2     | DiD        |       11 |             11 |         15 | Primary       | primary                    |    0.002923 | -0.035374 |  0.041220 |       0.017856 |  0.163705 |   14 | 0.872303 |              0.250116 |       51.000000 |     0.638672 |    1.000000 |          1.000000 | True            |
| esc50_animals | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.3     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.001380 | -0.043576 |  0.040817 |       0.019674 | -0.070127 |   14 | 0.945085 |              0.250116 |       57.000000 |     0.890381 |    1.000000 |          1.000000 | True            |
| esc50_animals | Primary  | linsvm       | comp          | shift      | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | Primary       | primary                    |   -0.001258 | -0.050218 |  0.047703 |       0.022828 | -0.055097 |   14 | 0.956839 |              0.250116 |       57.000000 |     0.890381 |    1.000000 |          1.000000 | True            |
| esc50_animals | S1       | linsvm       | comp          | shift      | emd          | nb3_0.1     | DiD        |       11 |              4 |         15 | S1            | exploratory                |    0.103502 |  0.039457 |  0.167547 |       0.029861 |  3.466156 |   14 | 0.003781 |              0.250116 |        0.000000 |     0.000061 |    0.003781 |          0.000732 | True            |
| esc50_animals | S1       | linsvm       | comp          | shift      | emd          | nb3_0.2     | DiD        |       11 |              4 |         15 | S1            | exploratory                |    0.100285 |  0.055884 |  0.144687 |       0.020702 |  4.844256 |   14 | 0.000260 |              0.250116 |        0.000000 |     0.000061 |    0.001040 |          0.000732 | True            |
| esc50_animals | S1       | linsvm       | comp          | shift      | emd          | nb3_0.3     | DiD        |       11 |              4 |         15 | S1            | exploratory                |    0.096106 |  0.054814 |  0.137398 |       0.019252 |  4.991890 |   14 | 0.000197 |              0.250116 |        0.000000 |     0.000061 |    0.000987 |          0.000732 | True            |
| esc50_animals | S1       | linsvm       | comp          | shift      | emd          | nb3_0.4     | DiD        |       11 |              4 |         15 | S1            | exploratory                |    0.088143 |  0.040259 |  0.136027 |       0.022326 |  3.948062 |   14 | 0.001457 |              0.250116 |        0.000000 |     0.000061 |    0.002915 |          0.000732 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.1     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.038280 | -0.007356 |  0.083917 |       0.021278 |  1.799070 |   14 | 0.093593 |              0.250116 |       11.000000 |     0.003357 |    0.842341 |          0.030212 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.2     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.064073 |  0.030079 |  0.098067 |       0.015850 |  4.042588 |   14 | 0.001211 |              0.250116 |        0.000000 |     0.000061 |    0.014528 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.3     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.077228 |  0.032866 |  0.121590 |       0.020684 |  3.733737 |   14 | 0.002224 |              0.250116 |        0.000000 |     0.000061 |    0.023429 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | S3            | exploratory                |    0.071552 |  0.030689 |  0.112415 |       0.019052 |  3.755540 |   14 | 0.002130 |              0.250116 |        0.000000 |     0.000061 |    0.023429 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.1     | DiD        |       11 |              6 |         15 | S3            | exploratory                |    0.067988 |  0.044839 |  0.091137 |       0.010793 |  6.299171 |   14 | 0.000020 |              0.250116 |        0.000000 |     0.000061 |    0.000274 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.2     | DiD        |       11 |              6 |         15 | S3            | exploratory                |    0.110144 |  0.087784 |  0.132504 |       0.010425 | 10.565213 |   14 | 0.000000 |              0.250116 |        0.000000 |     0.000061 |    0.000001 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.3     | DiD        |       11 |              6 |         15 | S3            | exploratory                |    0.146286 |  0.115428 |  0.177144 |       0.014387 | 10.167695 |   14 | 0.000000 |              0.250116 |        0.000000 |     0.000061 |    0.000001 |          0.001465 | True            |
| esc50_animals | S3       | lightgbm     | comp          | shift      | emd          | nb3_0.4     | DiD        |       11 |              6 |         15 | S3            | exploratory                |    0.162039 |  0.133435 |  0.190644 |       0.013337 | 12.149664 |   14 | 0.000000 |              0.250116 |        0.000000 |     0.000061 |    0.000000 |          0.001465 | True            |
| esc50_animals | S5       | linsvm       | comp          | matched    | linear       | nb3_0.4     | DiD        |       11 |             11 |         15 | S5            | exploratory                |    0.007904 | -0.016587 |  0.032394 |       0.011419 |  0.692191 |   14 | 0.500138 |              0.250116 |       35.000000 |     0.168823 |    1.000000 |          0.337646 | True            |
| esc50_animals | S5       | linsvm       | comp          | matched    | emd          | nb3_0.4     | DiD        |       11 |              4 |         15 | S5            | exploratory                |   -0.004663 | -0.060611 |  0.051286 |       0.026086 | -0.178740 |   14 | 0.860703 |              0.250116 |       55.000000 |     0.803955 |    1.000000 |          0.803955 | True            |

## Solver and H diagnostics

`h_ratio` means the input's E1 estimate divided by fixed H_d, not an H inflation factor. `b_over_h` converts raw b through the input's normalization scale before dividing by H_d. Per-frame, per-channel diagnostics and per-slot relative L2 are retained in the numeric chunks for every variant, including clean. Full stacks are retained only for the 100 pilot IDs.

| dataset       | condition   | variant    | method   |   k |   n_frames |   n_channels |   newton_cap_channels |   newton_cap_frames |   failed_frames |   sigma_fallback_channels |   h_median |    h_p95 |   h_ratio_median |   h_ratio_p95 |   sigma_hat_median |   sigma_hat_p95 |   b_over_h_median |   b_over_h_p95 |
|:--------------|:------------|:-----------|:---------|----:|-----------:|-------------:|----------------------:|--------------------:|----------------:|--------------------------:|-----------:|---------:|-----------------:|--------------:|-------------------:|----------------:|------------------:|---------------:|
| synthetic     | clean       | raw        | raw      |   0 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | emd_k2     | emd      |   2 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | emd_k3     | emd      |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | emd_k4     | emd      |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | emd_k6     | emd      |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | emd_k8     | emd      |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | linear_k3  | linear   |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | linear_k4  | linear   |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | linear_k6  | linear   |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | linear_k8  | linear   |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | linear_k9  | linear   |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | irmf_k3    | irmf     |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | irmf_k4    | irmf     |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | irmf_k6    | irmf     |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | irmf_k8    | irmf     |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | clean       | irmf_k9    | irmf     |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.492510 |      0.609998 |           0.583775 |        0.723035 |          1.477529 |       1.829995 |
| synthetic     | nb3_0.1     | raw        | raw      |   0 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | emd_k2     | emd      |   2 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | emd_k3     | emd      |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | emd_k4     | emd      |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | emd_k6     | emd      |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | emd_k8     | emd      |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | linear_k3  | linear   |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | linear_k4  | linear   |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | linear_k6  | linear   |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | linear_k8  | linear   |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | linear_k9  | linear   |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | irmf_k3    | irmf     |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | irmf_k4    | irmf     |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | irmf_k6    | irmf     |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | irmf_k8    | irmf     |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.1     | irmf_k9    | irmf     |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.532058 |      0.654342 |           0.630652 |        0.775596 |          1.391255 |       1.716174 |
| synthetic     | nb3_0.2     | raw        | raw      |   0 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | emd_k2     | emd      |   2 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | emd_k3     | emd      |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | emd_k4     | emd      |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | emd_k6     | emd      |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | emd_k8     | emd      |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | linear_k3  | linear   |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | linear_k4  | linear   |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | linear_k6  | linear   |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | linear_k8  | linear   |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | linear_k9  | linear   |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | irmf_k3    | irmf     |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | irmf_k4    | irmf     |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | irmf_k6    | irmf     |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | irmf_k8    | irmf     |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.2     | irmf_k9    | irmf     |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.576346 |      0.704394 |           0.683146 |        0.834922 |          1.303041 |       1.590276 |
| synthetic     | nb3_0.3     | raw        | raw      |   0 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | emd_k2     | emd      |   2 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | emd_k3     | emd      |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | emd_k4     | emd      |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | emd_k6     | emd      |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | emd_k8     | emd      |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | linear_k3  | linear   |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | linear_k4  | linear   |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | linear_k6  | linear   |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | linear_k8  | linear   |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | linear_k9  | linear   |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | irmf_k3    | irmf     |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | irmf_k4    | irmf     |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | irmf_k6    | irmf     |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | irmf_k8    | irmf     |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.3     | irmf_k9    | irmf     |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.627554 |      0.766250 |           0.743844 |        0.908241 |          1.217751 |       1.470286 |
| synthetic     | nb3_0.4     | raw        | raw      |   0 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | emd_k2     | emd      |   2 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | emd_k3     | emd      |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | emd_k4     | emd      |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | emd_k6     | emd      |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | emd_k8     | emd      |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | linear_k3  | linear   |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | linear_k4  | linear   |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | linear_k6  | linear   |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | linear_k8  | linear   |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | linear_k9  | linear   |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | irmf_k3    | irmf     |   3 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | irmf_k4    | irmf     |   4 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | irmf_k6    | irmf     |   6 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | irmf_k8    | irmf     |   8 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| synthetic     | nb3_0.4     | irmf_k9    | irmf     |   9 |       1500 |         1500 |                     0 |                   0 |               0 |                         0 |   1.185306 | 1.185306 |         0.685747 |      0.827252 |           0.812820 |        0.980547 |          1.131360 |       1.345253 |
| dronerf       | clean       | raw        | raw      |   0 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | emd_k2     | emd      |   2 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | emd_k3     | emd      |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | emd_k4     | emd      |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | emd_k6     | emd      |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | emd_k8     | emd      |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | linear_k3  | linear   |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | linear_k4  | linear   |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | linear_k6  | linear   |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | linear_k8  | linear   |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | linear_k11 | linear   |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | irmf_k3    | irmf     |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | irmf_k4    | irmf     |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | irmf_k6    | irmf     |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | irmf_k8    | irmf     |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | clean       | irmf_k11   | irmf     |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.491129 |      0.562751 |           1.028505 |        1.178495 |          1.473386 |       1.688254 |
| dronerf       | nb3_0.1     | raw        | raw      |   0 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | emd_k2     | emd      |   2 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | emd_k3     | emd      |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | emd_k4     | emd      |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | emd_k6     | emd      |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | emd_k8     | emd      |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | linear_k3  | linear   |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | linear_k4  | linear   |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | linear_k6  | linear   |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | linear_k8  | linear   |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | linear_k11 | linear   |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | irmf_k3    | irmf     |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | irmf_k4    | irmf     |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | irmf_k6    | irmf     |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | irmf_k8    | irmf     |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.1     | irmf_k11   | irmf     |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.506476 |      0.590889 |           1.060646 |        1.237420 |          1.314463 |       1.519429 |
| dronerf       | nb3_0.2     | raw        | raw      |   0 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | emd_k2     | emd      |   2 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | emd_k3     | emd      |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | emd_k4     | emd      |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | emd_k6     | emd      |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | emd_k8     | emd      |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | linear_k3  | linear   |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | linear_k4  | linear   |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | linear_k6  | linear   |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | linear_k8  | linear   |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | linear_k11 | linear   |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | irmf_k3    | irmf     |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | irmf_k4    | irmf     |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | irmf_k6    | irmf     |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | irmf_k8    | irmf     |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.2     | irmf_k11   | irmf     |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.543708 |      0.665247 |           1.138614 |        1.393138 |          1.152690 |       1.519429 |
| dronerf       | nb3_0.3     | raw        | raw      |   0 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | emd_k2     | emd      |   2 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | emd_k3     | emd      |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | emd_k4     | emd      |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | emd_k6     | emd      |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | emd_k8     | emd      |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | linear_k3  | linear   |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | linear_k4  | linear   |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | linear_k6  | linear   |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | linear_k8  | linear   |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | linear_k11 | linear   |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | irmf_k3    | irmf     |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | irmf_k4    | irmf     |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | irmf_k6    | irmf     |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | irmf_k8    | irmf     |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.3     | irmf_k11   | irmf     |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.594192 |      0.675302 |           1.244337 |        1.414194 |          1.012953 |       1.350603 |
| dronerf       | nb3_0.4     | raw        | raw      |   0 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | emd_k2     | emd      |   2 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | emd_k3     | emd      |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | emd_k4     | emd      |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | emd_k6     | emd      |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | emd_k8     | emd      |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | linear_k3  | linear   |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | linear_k4  | linear   |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | linear_k6  | linear   |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | linear_k8  | linear   |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | linear_k11 | linear   |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | irmf_k3    | irmf     |   3 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | irmf_k4    | irmf     |   4 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | irmf_k6    | irmf     |   6 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | irmf_k8    | irmf     |   8 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| dronerf       | nb3_0.4     | irmf_k11   | irmf     |  11 |      12000 |        24000 |                     0 |                   0 |               0 |                         0 |   2.094166 | 2.094166 |         0.647917 |      0.675302 |           1.356845 |        1.414194 |          1.012953 |       1.146833 |
| birdclef2023  | clean       | raw        | raw      |   0 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | emd_k2     | emd      |   2 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | emd_k3     | emd      |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | emd_k4     | emd      |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | emd_k6     | emd      |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | emd_k8     | emd      |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | linear_k3  | linear   |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | linear_k4  | linear   |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | linear_k6  | linear   |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | linear_k8  | linear   |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | linear_k11 | linear   |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | irmf_k3    | irmf     |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | irmf_k4    | irmf     |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | irmf_k6    | irmf     |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | irmf_k8    | irmf     |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | clean       | irmf_k11   | irmf     |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.444451 |      0.905766 |           0.481481 |        0.981230 |          1.333353 |       2.717299 |
| birdclef2023  | nb3_0.1     | raw        | raw      |   0 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | emd_k2     | emd      |   2 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | emd_k3     | emd      |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | emd_k4     | emd      |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | emd_k6     | emd      |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | emd_k8     | emd      |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | linear_k3  | linear   |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | linear_k4  | linear   |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | linear_k6  | linear   |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | linear_k8  | linear   |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | linear_k11 | linear   |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | irmf_k3    | irmf     |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | irmf_k4    | irmf     |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | irmf_k6    | irmf     |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | irmf_k8    | irmf     |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.1     | irmf_k11   | irmf     |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.486084 |      0.971157 |           0.526582 |        1.052068 |          1.259199 |       2.484559 |
| birdclef2023  | nb3_0.2     | raw        | raw      |   0 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | emd_k2     | emd      |   2 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | emd_k3     | emd      |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | emd_k4     | emd      |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | emd_k6     | emd      |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | emd_k8     | emd      |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | linear_k3  | linear   |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | linear_k4  | linear   |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | linear_k6  | linear   |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | linear_k8  | linear   |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | linear_k11 | linear   |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | irmf_k3    | irmf     |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | irmf_k4    | irmf     |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | irmf_k6    | irmf     |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | irmf_k8    | irmf     |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.2     | irmf_k11   | irmf     |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.533893 |      1.035450 |           0.578374 |        1.121718 |          1.181743 |       2.278186 |
| birdclef2023  | nb3_0.3     | raw        | raw      |   0 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | emd_k2     | emd      |   2 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | emd_k3     | emd      |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | emd_k4     | emd      |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | emd_k6     | emd      |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | emd_k8     | emd      |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | linear_k3  | linear   |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | linear_k4  | linear   |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | linear_k6  | linear   |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | linear_k8  | linear   |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | linear_k11 | linear   |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | irmf_k3    | irmf     |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | irmf_k4    | irmf     |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | irmf_k6    | irmf     |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | irmf_k8    | irmf     |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.3     | irmf_k11   | irmf     |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.586625 |      1.091173 |           0.635500 |        1.182084 |          1.105565 |       2.077981 |
| birdclef2023  | nb3_0.4     | raw        | raw      |   0 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | emd_k2     | emd      |   2 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | emd_k3     | emd      |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | emd_k4     | emd      |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | emd_k6     | emd      |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | emd_k8     | emd      |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | linear_k3  | linear   |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | linear_k4  | linear   |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | linear_k6  | linear   |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | linear_k8  | linear   |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | linear_k11 | linear   |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | irmf_k3    | irmf     |   3 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | irmf_k4    | irmf     |   4 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | irmf_k6    | irmf     |   6 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | irmf_k8    | irmf     |   8 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| birdclef2023  | nb3_0.4     | irmf_k11   | irmf     |  11 |      12000 |        12000 |                     0 |                   0 |               0 |                         0 |   1.083315 | 1.083315 |         0.642988 |      1.135268 |           0.696559 |        1.229852 |          1.032591 |       1.886150 |
| esc50_animals | clean       | raw        | raw      |   0 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | emd_k2     | emd      |   2 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | emd_k3     | emd      |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | emd_k4     | emd      |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | emd_k6     | emd      |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | emd_k8     | emd      |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | linear_k3  | linear   |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | linear_k4  | linear   |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | linear_k6  | linear   |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | linear_k8  | linear   |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | linear_k11 | linear   |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | irmf_k3    | irmf     |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | irmf_k4    | irmf     |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | irmf_k6    | irmf     |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | irmf_k8    | irmf     |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | clean       | irmf_k11   | irmf     |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.534902 |      1.120705 |           0.252816 |        0.529690 |          1.604707 |       3.362116 |
| esc50_animals | nb3_0.1     | raw        | raw      |   0 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | emd_k2     | emd      |   2 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | emd_k3     | emd      |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | emd_k4     | emd      |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | emd_k6     | emd      |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | emd_k8     | emd      |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | linear_k3  | linear   |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | linear_k4  | linear   |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | linear_k6  | linear   |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | linear_k8  | linear   |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | linear_k11 | linear   |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | irmf_k3    | irmf     |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | irmf_k4    | irmf     |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | irmf_k6    | irmf     |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | irmf_k8    | irmf     |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.1     | irmf_k11   | irmf     |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.599754 |      1.220405 |           0.283467 |        0.576812 |          1.553683 |       3.152753 |
| esc50_animals | nb3_0.2     | raw        | raw      |   0 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | emd_k2     | emd      |   2 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | emd_k3     | emd      |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | emd_k4     | emd      |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | emd_k6     | emd      |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | emd_k8     | emd      |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | linear_k3  | linear   |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | linear_k4  | linear   |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | linear_k6  | linear   |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | linear_k8  | linear   |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | linear_k11 | linear   |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | irmf_k3    | irmf     |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | irmf_k4    | irmf     |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | irmf_k6    | irmf     |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | irmf_k8    | irmf     |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.2     | irmf_k11   | irmf     |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.677031 |      1.358803 |           0.319992 |        0.642224 |          1.511442 |       2.948145 |
| esc50_animals | nb3_0.3     | raw        | raw      |   0 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | emd_k2     | emd      |   2 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | emd_k3     | emd      |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | emd_k4     | emd      |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | emd_k6     | emd      |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | emd_k8     | emd      |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | linear_k3  | linear   |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | linear_k4  | linear   |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | linear_k6  | linear   |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | linear_k8  | linear   |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | linear_k11 | linear   |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | irmf_k3    | irmf     |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | irmf_k4    | irmf     |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | irmf_k6    | irmf     |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | irmf_k8    | irmf     |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.3     | irmf_k11   | irmf     |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.768807 |      1.503885 |           0.363369 |        0.710796 |          1.453558 |       2.757653 |
| esc50_animals | nb3_0.4     | raw        | raw      |   0 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | emd_k2     | emd      |   2 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | emd_k3     | emd      |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | emd_k4     | emd      |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | emd_k6     | emd      |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | emd_k8     | emd      |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | linear_k3  | linear   |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | linear_k4  | linear   |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | linear_k6  | linear   |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | linear_k8  | linear   |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | linear_k11 | linear   |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | irmf_k3    | irmf     |   3 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | irmf_k4    | irmf     |   4 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | irmf_k6    | irmf     |   6 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | irmf_k8    | irmf     |   8 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |
| esc50_animals | nb3_0.4     | irmf_k11   | irmf     |  11 |       5100 |         5100 |                     0 |                   0 |               0 |                         0 |   0.472640 | 0.472640 |         0.868185 |      1.629974 |           0.410339 |        0.770391 |          1.402070 |       2.584898 |

## Verification and dated deviations

2026-10-22: the first raw gate failed on 7/200 raw checks because the draft cast raw samples to float32. All 200 EMD checks passed exactly. The original `clean_verification.json` files remain unchanged. The amendment restores float64 raw samples. The passing checks are separately saved as `clean_verification_v2.json`, referenced by `manifest_v2.json`, version 2. No tolerance was relaxed and contamination began only after this new gate passed.

2026-10-22: constant-channel noise is zero when the shared estimator returns its constant fallback source code 2. H calibration retains the shared E1 fallbacks. Fixed H uses the excluded pilots, as amended, rather than the pilot report's split-0 training calibration.

DroneRF verification: 100/100 pass, 100/100 exact.

BirdCLEF 2023 verification: 100/100 pass, 100/100 exact.

ESC-50 animals verification: 100/100 pass, 100/100 exact.

Synthetic verification: 100/100 pass, 100/100 exact.

## Cost and reproduction

External v3 outputs currently occupy 5.880 GB of the 30 GB budget.

| dataset       | condition   | variant    | method   |   k |   n_frames |   cpu_seconds |   frame_wall_seconds |   cpu_s_per_frame |   failed_frames |   cap_channels |
|:--------------|:------------|:-----------|:---------|----:|-----------:|--------------:|---------------------:|------------------:|----------------:|---------------:|
| synthetic     | clean       | raw        | raw      |   0 |       1500 |        0.5615 |               0.5639 |            0.0004 |               0 |              0 |
| synthetic     | clean       | emd_k2     | emd      |   2 |       1500 |        0.6820 |               0.6925 |            0.0005 |               0 |              0 |
| synthetic     | clean       | emd_k3     | emd      |   3 |       1500 |        0.7152 |               0.7211 |            0.0005 |               0 |              0 |
| synthetic     | clean       | emd_k4     | emd      |   4 |       1500 |        0.7889 |               0.7926 |            0.0005 |               0 |              0 |
| synthetic     | clean       | emd_k6     | emd      |   6 |       1500 |        0.9527 |               0.9693 |            0.0006 |               0 |              0 |
| synthetic     | clean       | emd_k8     | emd      |   8 |       1500 |       24.9654 |              25.1415 |            0.0166 |               0 |              0 |
| synthetic     | clean       | linear_k3  | linear   |   3 |       1500 |        1.7404 |               1.7507 |            0.0012 |               0 |              0 |
| synthetic     | clean       | linear_k4  | linear   |   4 |       1500 |        1.9861 |               2.0014 |            0.0013 |               0 |              0 |
| synthetic     | clean       | linear_k6  | linear   |   6 |       1500 |        2.6079 |               2.6345 |            0.0017 |               0 |              0 |
| synthetic     | clean       | linear_k8  | linear   |   8 |       1500 |        3.2391 |               3.2637 |            0.0022 |               0 |              0 |
| synthetic     | clean       | linear_k9  | linear   |   9 |       1500 |        3.5723 |               3.6173 |            0.0024 |               0 |              0 |
| synthetic     | clean       | irmf_k3    | irmf     |   3 |       1500 |       10.0560 |              10.1395 |            0.0067 |               0 |              0 |
| synthetic     | clean       | irmf_k4    | irmf     |   4 |       1500 |       12.1813 |              12.2639 |            0.0081 |               0 |              0 |
| synthetic     | clean       | irmf_k6    | irmf     |   6 |       1500 |       16.9132 |              17.0606 |            0.0113 |               0 |              0 |
| synthetic     | clean       | irmf_k8    | irmf     |   8 |       1500 |       21.6454 |              21.8108 |            0.0144 |               0 |              0 |
| synthetic     | clean       | irmf_k9    | irmf     |   9 |       1500 |       24.0664 |              24.2566 |            0.0160 |               0 |              0 |
| synthetic     | nb3_0.1     | raw        | raw      |   0 |       1500 |        0.5637 |               0.5682 |            0.0004 |               0 |              0 |
| synthetic     | nb3_0.1     | emd_k2     | emd      |   2 |       1500 |        0.6826 |               0.6926 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.1     | emd_k3     | emd      |   3 |       1500 |        0.7071 |               0.7230 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.1     | emd_k4     | emd      |   4 |       1500 |        0.7903 |               0.8032 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.1     | emd_k6     | emd      |   6 |       1500 |        0.9710 |               1.0157 |            0.0006 |               0 |              0 |
| synthetic     | nb3_0.1     | emd_k8     | emd      |   8 |       1500 |       23.4203 |              23.8927 |            0.0156 |               0 |              0 |
| synthetic     | nb3_0.1     | linear_k3  | linear   |   3 |       1500 |        1.7550 |               1.7871 |            0.0012 |               0 |              0 |
| synthetic     | nb3_0.1     | linear_k4  | linear   |   4 |       1500 |        2.0019 |               2.0363 |            0.0013 |               0 |              0 |
| synthetic     | nb3_0.1     | linear_k6  | linear   |   6 |       1500 |        2.6260 |               2.6819 |            0.0018 |               0 |              0 |
| synthetic     | nb3_0.1     | linear_k8  | linear   |   8 |       1500 |        3.2671 |               3.3082 |            0.0022 |               0 |              0 |
| synthetic     | nb3_0.1     | linear_k9  | linear   |   9 |       1500 |        3.5736 |               3.6504 |            0.0024 |               0 |              0 |
| synthetic     | nb3_0.1     | irmf_k3    | irmf     |   3 |       1500 |       10.2311 |              10.4101 |            0.0068 |               0 |              0 |
| synthetic     | nb3_0.1     | irmf_k4    | irmf     |   4 |       1500 |       12.4805 |              12.7054 |            0.0083 |               0 |              0 |
| synthetic     | nb3_0.1     | irmf_k6    | irmf     |   6 |       1500 |       17.2212 |              17.5170 |            0.0115 |               0 |              0 |
| synthetic     | nb3_0.1     | irmf_k8    | irmf     |   8 |       1500 |       22.1031 |              22.4992 |            0.0147 |               0 |              0 |
| synthetic     | nb3_0.1     | irmf_k9    | irmf     |   9 |       1500 |       24.5792 |              25.0074 |            0.0164 |               0 |              0 |
| synthetic     | nb3_0.2     | raw        | raw      |   0 |       1500 |        0.5762 |               0.5903 |            0.0004 |               0 |              0 |
| synthetic     | nb3_0.2     | emd_k2     | emd      |   2 |       1500 |        0.7027 |               0.7137 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.2     | emd_k3     | emd      |   3 |       1500 |        0.7240 |               0.7333 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.2     | emd_k4     | emd      |   4 |       1500 |        0.7992 |               0.8089 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.2     | emd_k6     | emd      |   6 |       1500 |        0.9615 |               0.9829 |            0.0006 |               0 |              0 |
| synthetic     | nb3_0.2     | emd_k8     | emd      |   8 |       1500 |       23.0204 |              23.4669 |            0.0153 |               0 |              0 |
| synthetic     | nb3_0.2     | linear_k3  | linear   |   3 |       1500 |        1.7254 |               1.7562 |            0.0012 |               0 |              0 |
| synthetic     | nb3_0.2     | linear_k4  | linear   |   4 |       1500 |        1.9642 |               1.9952 |            0.0013 |               0 |              0 |
| synthetic     | nb3_0.2     | linear_k6  | linear   |   6 |       1500 |        2.5777 |               2.6230 |            0.0017 |               0 |              0 |
| synthetic     | nb3_0.2     | linear_k8  | linear   |   8 |       1500 |        3.2177 |               3.2817 |            0.0021 |               0 |              0 |
| synthetic     | nb3_0.2     | linear_k9  | linear   |   9 |       1500 |        3.5247 |               3.5693 |            0.0023 |               0 |              0 |
| synthetic     | nb3_0.2     | irmf_k3    | irmf     |   3 |       1500 |       10.1804 |              10.3521 |            0.0068 |               0 |              0 |
| synthetic     | nb3_0.2     | irmf_k4    | irmf     |   4 |       1500 |       12.4949 |              12.6514 |            0.0083 |               0 |              0 |
| synthetic     | nb3_0.2     | irmf_k6    | irmf     |   6 |       1500 |       17.2995 |              17.5705 |            0.0115 |               0 |              0 |
| synthetic     | nb3_0.2     | irmf_k8    | irmf     |   8 |       1500 |       22.2893 |              22.7181 |            0.0149 |               0 |              0 |
| synthetic     | nb3_0.2     | irmf_k9    | irmf     |   9 |       1500 |       24.8144 |              25.1713 |            0.0165 |               0 |              0 |
| synthetic     | nb3_0.3     | raw        | raw      |   0 |       1500 |        0.5647 |               0.5836 |            0.0004 |               0 |              0 |
| synthetic     | nb3_0.3     | emd_k2     | emd      |   2 |       1500 |        0.6787 |               0.6825 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.3     | emd_k3     | emd      |   3 |       1500 |        0.7048 |               0.7112 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.3     | emd_k4     | emd      |   4 |       1500 |        0.7874 |               0.7977 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.3     | emd_k6     | emd      |   6 |       1500 |        0.9471 |               0.9595 |            0.0006 |               0 |              0 |
| synthetic     | nb3_0.3     | emd_k8     | emd      |   8 |       1500 |       21.3752 |              21.6644 |            0.0143 |               0 |              0 |
| synthetic     | nb3_0.3     | linear_k3  | linear   |   3 |       1500 |        1.7365 |               1.7616 |            0.0012 |               0 |              0 |
| synthetic     | nb3_0.3     | linear_k4  | linear   |   4 |       1500 |        1.9820 |               2.0310 |            0.0013 |               0 |              0 |
| synthetic     | nb3_0.3     | linear_k6  | linear   |   6 |       1500 |        2.5926 |               2.6236 |            0.0017 |               0 |              0 |
| synthetic     | nb3_0.3     | linear_k8  | linear   |   8 |       1500 |        3.2175 |               3.2519 |            0.0021 |               0 |              0 |
| synthetic     | nb3_0.3     | linear_k9  | linear   |   9 |       1500 |        3.5305 |               3.5782 |            0.0024 |               0 |              0 |
| synthetic     | nb3_0.3     | irmf_k3    | irmf     |   3 |       1500 |       10.2886 |              10.4817 |            0.0069 |               0 |              0 |
| synthetic     | nb3_0.3     | irmf_k4    | irmf     |   4 |       1500 |       12.4792 |              12.6341 |            0.0083 |               0 |              0 |
| synthetic     | nb3_0.3     | irmf_k6    | irmf     |   6 |       1500 |       17.3565 |              17.5899 |            0.0116 |               0 |              0 |
| synthetic     | nb3_0.3     | irmf_k8    | irmf     |   8 |       1500 |       22.3260 |              22.7144 |            0.0149 |               0 |              0 |
| synthetic     | nb3_0.3     | irmf_k9    | irmf     |   9 |       1500 |       24.9518 |              25.3074 |            0.0166 |               0 |              0 |
| synthetic     | nb3_0.4     | raw        | raw      |   0 |       1500 |        0.5619 |               0.5720 |            0.0004 |               0 |              0 |
| synthetic     | nb3_0.4     | emd_k2     | emd      |   2 |       1500 |        0.6760 |               0.6906 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.4     | emd_k3     | emd      |   3 |       1500 |        0.7105 |               0.7339 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.4     | emd_k4     | emd      |   4 |       1500 |        0.7950 |               0.8056 |            0.0005 |               0 |              0 |
| synthetic     | nb3_0.4     | emd_k6     | emd      |   6 |       1500 |        0.9578 |               0.9706 |            0.0006 |               0 |              0 |
| synthetic     | nb3_0.4     | emd_k8     | emd      |   8 |       1500 |       21.2781 |              21.6589 |            0.0142 |               0 |              0 |
| synthetic     | nb3_0.4     | linear_k3  | linear   |   3 |       1500 |        1.7402 |               1.7727 |            0.0012 |               0 |              0 |
| synthetic     | nb3_0.4     | linear_k4  | linear   |   4 |       1500 |        1.9798 |               2.0300 |            0.0013 |               0 |              0 |
| synthetic     | nb3_0.4     | linear_k6  | linear   |   6 |       1500 |        2.5916 |               2.6605 |            0.0017 |               0 |              0 |
| synthetic     | nb3_0.4     | linear_k8  | linear   |   8 |       1500 |        3.2262 |               3.2799 |            0.0022 |               0 |              0 |
| synthetic     | nb3_0.4     | linear_k9  | linear   |   9 |       1500 |        3.5298 |               3.5900 |            0.0024 |               0 |              0 |
| synthetic     | nb3_0.4     | irmf_k3    | irmf     |   3 |       1500 |       10.2696 |              10.4913 |            0.0068 |               0 |              0 |
| synthetic     | nb3_0.4     | irmf_k4    | irmf     |   4 |       1500 |       12.7113 |              12.9192 |            0.0085 |               0 |              0 |
| synthetic     | nb3_0.4     | irmf_k6    | irmf     |   6 |       1500 |       17.3917 |              17.6737 |            0.0116 |               0 |              0 |
| synthetic     | nb3_0.4     | irmf_k8    | irmf     |   8 |       1500 |       22.2279 |              22.6453 |            0.0148 |               0 |              0 |
| synthetic     | nb3_0.4     | irmf_k9    | irmf     |   9 |       1500 |       24.9008 |              25.3757 |            0.0166 |               0 |              0 |
| dronerf       | clean       | raw        | raw      |   0 |      12000 |        7.4445 |               7.4923 |            0.0006 |               0 |              0 |
| dronerf       | clean       | emd_k2     | emd      |   2 |      12000 |       13.0128 |              13.1454 |            0.0011 |               0 |              0 |
| dronerf       | clean       | emd_k3     | emd      |   3 |      12000 |       15.2663 |              15.4359 |            0.0013 |               0 |              0 |
| dronerf       | clean       | emd_k4     | emd      |   4 |      12000 |       17.6480 |              17.8440 |            0.0015 |               0 |              0 |
| dronerf       | clean       | emd_k6     | emd      |   6 |      12000 |       22.8513 |              23.1082 |            0.0019 |               0 |              0 |
| dronerf       | clean       | emd_k8     | emd      |   8 |      12000 |      388.3313 |             392.8455 |            0.0324 |               0 |              0 |
| dronerf       | clean       | linear_k3  | linear   |   3 |      12000 |       69.8156 |              70.5318 |            0.0058 |               0 |              0 |
| dronerf       | clean       | linear_k4  | linear   |   4 |      12000 |       81.6925 |              82.5285 |            0.0068 |               0 |              0 |
| dronerf       | clean       | linear_k6  | linear   |   6 |      12000 |      108.9944 |             110.2526 |            0.0091 |               0 |              0 |
| dronerf       | clean       | linear_k8  | linear   |   8 |      12000 |      137.6063 |             139.1181 |            0.0115 |               0 |              0 |
| dronerf       | clean       | linear_k11 | linear   |  11 |      12000 |      181.1698 |             182.9996 |            0.0151 |               0 |              0 |
| dronerf       | clean       | irmf_k3    | irmf     |   3 |      12000 |      496.0805 |             501.6616 |            0.0413 |               0 |              0 |
| dronerf       | clean       | irmf_k4    | irmf     |   4 |      12000 |      592.6270 |             598.7993 |            0.0494 |               0 |              0 |
| dronerf       | clean       | irmf_k6    | irmf     |   6 |      12000 |      809.0201 |             817.7768 |            0.0674 |               0 |              0 |
| dronerf       | clean       | irmf_k8    | irmf     |   8 |      12000 |     1035.7516 |            1046.9524 |            0.0863 |               0 |              0 |
| dronerf       | clean       | irmf_k11   | irmf     |  11 |      12000 |     1377.3442 |            1393.4958 |            0.1148 |               0 |              0 |
| dronerf       | nb3_0.1     | raw        | raw      |   0 |      12000 |        7.4240 |               7.4564 |            0.0006 |               0 |              0 |
| dronerf       | nb3_0.1     | emd_k2     | emd      |   2 |      12000 |       13.0888 |              13.2540 |            0.0011 |               0 |              0 |
| dronerf       | nb3_0.1     | emd_k3     | emd      |   3 |      12000 |       15.2883 |              15.3838 |            0.0013 |               0 |              0 |
| dronerf       | nb3_0.1     | emd_k4     | emd      |   4 |      12000 |       17.6390 |              17.7803 |            0.0015 |               0 |              0 |
| dronerf       | nb3_0.1     | emd_k6     | emd      |   6 |      12000 |       22.8425 |              23.0778 |            0.0019 |               0 |              0 |
| dronerf       | nb3_0.1     | emd_k8     | emd      |   8 |      12000 |      832.1002 |             838.4124 |            0.0693 |               0 |              0 |
| dronerf       | nb3_0.1     | linear_k3  | linear   |   3 |      12000 |       69.0360 |              69.3960 |            0.0058 |               0 |              0 |
| dronerf       | nb3_0.1     | linear_k4  | linear   |   4 |      12000 |       80.8093 |              81.3890 |            0.0067 |               0 |              0 |
| dronerf       | nb3_0.1     | linear_k6  | linear   |   6 |      12000 |      107.9481 |             108.5883 |            0.0090 |               0 |              0 |
| dronerf       | nb3_0.1     | linear_k8  | linear   |   8 |      12000 |      136.0844 |             137.0297 |            0.0113 |               0 |              0 |
| dronerf       | nb3_0.1     | linear_k11 | linear   |  11 |      12000 |      179.0047 |             180.2674 |            0.0149 |               0 |              0 |
| dronerf       | nb3_0.1     | irmf_k3    | irmf     |   3 |      12000 |      516.5602 |             519.7459 |            0.0430 |               0 |              0 |
| dronerf       | nb3_0.1     | irmf_k4    | irmf     |   4 |      12000 |      616.7985 |             621.4744 |            0.0514 |               0 |              0 |
| dronerf       | nb3_0.1     | irmf_k6    | irmf     |   6 |      12000 |      846.9162 |             853.6728 |            0.0706 |               0 |              0 |
| dronerf       | nb3_0.1     | irmf_k8    | irmf     |   8 |      12000 |     1082.6308 |            1090.0355 |            0.0902 |               0 |              0 |
| dronerf       | nb3_0.1     | irmf_k11   | irmf     |  11 |      12000 |     1440.7736 |            1450.5430 |            0.1201 |               0 |              0 |
| dronerf       | nb3_0.2     | raw        | raw      |   0 |      12000 |        7.4512 |               7.4836 |            0.0006 |               0 |              0 |
| dronerf       | nb3_0.2     | emd_k2     | emd      |   2 |      12000 |       13.1242 |              13.2006 |            0.0011 |               0 |              0 |
| dronerf       | nb3_0.2     | emd_k3     | emd      |   3 |      12000 |       15.3437 |              15.4216 |            0.0013 |               0 |              0 |
| dronerf       | nb3_0.2     | emd_k4     | emd      |   4 |      12000 |       17.7772 |              17.9253 |            0.0015 |               0 |              0 |
| dronerf       | nb3_0.2     | emd_k6     | emd      |   6 |      12000 |       22.9943 |              23.1516 |            0.0019 |               0 |              0 |
| dronerf       | nb3_0.2     | emd_k8     | emd      |   8 |      12000 |      731.7569 |             736.7892 |            0.0610 |               0 |              0 |
| dronerf       | nb3_0.2     | linear_k3  | linear   |   3 |      12000 |       68.9590 |              69.3817 |            0.0057 |               0 |              0 |
| dronerf       | nb3_0.2     | linear_k4  | linear   |   4 |      12000 |       80.8174 |              81.3888 |            0.0067 |               0 |              0 |
| dronerf       | nb3_0.2     | linear_k6  | linear   |   6 |      12000 |      108.1521 |             108.7923 |            0.0090 |               0 |              0 |
| dronerf       | nb3_0.2     | linear_k8  | linear   |   8 |      12000 |      136.3205 |             137.2601 |            0.0114 |               0 |              0 |
| dronerf       | nb3_0.2     | linear_k11 | linear   |  11 |      12000 |      179.4805 |             180.7486 |            0.0150 |               0 |              0 |
| dronerf       | nb3_0.2     | irmf_k3    | irmf     |   3 |      12000 |      524.9223 |             529.0170 |            0.0437 |               0 |              0 |
| dronerf       | nb3_0.2     | irmf_k4    | irmf     |   4 |      12000 |      627.8714 |             632.8654 |            0.0523 |               0 |              0 |
| dronerf       | nb3_0.2     | irmf_k6    | irmf     |   6 |      12000 |      861.9943 |             867.9983 |            0.0718 |               0 |              0 |
| dronerf       | nb3_0.2     | irmf_k8    | irmf     |   8 |      12000 |     1101.5824 |            1109.1511 |            0.0918 |               0 |              0 |
| dronerf       | nb3_0.2     | irmf_k11   | irmf     |  11 |      12000 |     1468.4377 |            1477.6327 |            0.1224 |               0 |              0 |
| dronerf       | nb3_0.3     | raw        | raw      |   0 |      12000 |        7.4474 |               7.4624 |            0.0006 |               0 |              0 |
| dronerf       | nb3_0.3     | emd_k2     | emd      |   2 |      12000 |       13.1331 |              13.2003 |            0.0011 |               0 |              0 |
| dronerf       | nb3_0.3     | emd_k3     | emd      |   3 |      12000 |       15.3650 |              15.4472 |            0.0013 |               0 |              0 |
| dronerf       | nb3_0.3     | emd_k4     | emd      |   4 |      12000 |       17.7606 |              17.8979 |            0.0015 |               0 |              0 |
| dronerf       | nb3_0.3     | emd_k6     | emd      |   6 |      12000 |       22.8939 |              23.2034 |            0.0019 |               0 |              0 |
| dronerf       | nb3_0.3     | emd_k8     | emd      |   8 |      12000 |      652.8105 |             657.4580 |            0.0544 |               0 |              0 |
| dronerf       | nb3_0.3     | linear_k3  | linear   |   3 |      12000 |       69.2484 |              69.9266 |            0.0058 |               0 |              0 |
| dronerf       | nb3_0.3     | linear_k4  | linear   |   4 |      12000 |       80.9843 |              81.5012 |            0.0067 |               0 |              0 |
| dronerf       | nb3_0.3     | linear_k6  | linear   |   6 |      12000 |      107.9256 |             108.6924 |            0.0090 |               0 |              0 |
| dronerf       | nb3_0.3     | linear_k8  | linear   |   8 |      12000 |      136.2736 |             136.9653 |            0.0114 |               0 |              0 |
| dronerf       | nb3_0.3     | linear_k11 | linear   |  11 |      12000 |      179.2865 |             180.3460 |            0.0149 |               0 |              0 |
| dronerf       | nb3_0.3     | irmf_k3    | irmf     |   3 |      12000 |      529.3865 |             532.9472 |            0.0441 |               0 |              0 |
| dronerf       | nb3_0.3     | irmf_k4    | irmf     |   4 |      12000 |      634.4436 |             638.1026 |            0.0529 |               0 |              0 |
| dronerf       | nb3_0.3     | irmf_k6    | irmf     |   6 |      12000 |      869.4481 |             875.1594 |            0.0725 |               0 |              0 |
| dronerf       | nb3_0.3     | irmf_k8    | irmf     |   8 |      12000 |     1110.5578 |            1118.1636 |            0.0925 |               0 |              0 |
| dronerf       | nb3_0.3     | irmf_k11   | irmf     |  11 |      12000 |     1480.2038 |            1489.4794 |            0.1234 |               0 |              0 |
| dronerf       | nb3_0.4     | raw        | raw      |   0 |      12000 |        7.4644 |               7.5044 |            0.0006 |               0 |              0 |
| dronerf       | nb3_0.4     | emd_k2     | emd      |   2 |      12000 |       13.1039 |              13.2108 |            0.0011 |               0 |              0 |
| dronerf       | nb3_0.4     | emd_k3     | emd      |   3 |      12000 |       15.3058 |              15.5200 |            0.0013 |               0 |              0 |
| dronerf       | nb3_0.4     | emd_k4     | emd      |   4 |      12000 |       17.6623 |              17.8515 |            0.0015 |               0 |              0 |
| dronerf       | nb3_0.4     | emd_k6     | emd      |   6 |      12000 |       22.8167 |              22.9234 |            0.0019 |               0 |              0 |
| dronerf       | nb3_0.4     | emd_k8     | emd      |   8 |      12000 |      593.5515 |             597.9854 |            0.0495 |               0 |              0 |
| dronerf       | nb3_0.4     | linear_k3  | linear   |   3 |      12000 |       69.0836 |              69.5105 |            0.0058 |               0 |              0 |
| dronerf       | nb3_0.4     | linear_k4  | linear   |   4 |      12000 |       80.9434 |              81.3722 |            0.0067 |               0 |              0 |
| dronerf       | nb3_0.4     | linear_k6  | linear   |   6 |      12000 |      107.8930 |             108.4771 |            0.0090 |               0 |              0 |
| dronerf       | nb3_0.4     | linear_k8  | linear   |   8 |      12000 |      136.3511 |             137.3058 |            0.0114 |               0 |              0 |
| dronerf       | nb3_0.4     | linear_k11 | linear   |  11 |      12000 |      179.4486 |             180.4954 |            0.0150 |               0 |              0 |
| dronerf       | nb3_0.4     | irmf_k3    | irmf     |   3 |      12000 |      530.5456 |             533.7495 |            0.0442 |               0 |              0 |
| dronerf       | nb3_0.4     | irmf_k4    | irmf     |   4 |      12000 |      635.2033 |             638.8959 |            0.0529 |               0 |              0 |
| dronerf       | nb3_0.4     | irmf_k6    | irmf     |   6 |      12000 |      869.8876 |             875.3771 |            0.0725 |               0 |              0 |
| dronerf       | nb3_0.4     | irmf_k8    | irmf     |   8 |      12000 |     1112.0151 |            1118.9753 |            0.0927 |               0 |              0 |
| dronerf       | nb3_0.4     | irmf_k11   | irmf     |  11 |      12000 |     1480.9773 |            1492.0682 |            0.1234 |               0 |              0 |
| birdclef2023  | clean       | raw        | raw      |   0 |      12000 |        5.4221 |               5.4668 |            0.0005 |               0 |              0 |
| birdclef2023  | clean       | emd_k2     | emd      |   2 |      12000 |        7.7898 |               7.8414 |            0.0006 |               0 |              0 |
| birdclef2023  | clean       | emd_k3     | emd      |   3 |      12000 |        8.6554 |               8.7460 |            0.0007 |               0 |              0 |
| birdclef2023  | clean       | emd_k4     | emd      |   4 |      12000 |        9.8293 |               9.9407 |            0.0008 |               0 |              0 |
| birdclef2023  | clean       | emd_k6     | emd      |   6 |      12000 |       12.4704 |              12.5795 |            0.0010 |               0 |              0 |
| birdclef2023  | clean       | emd_k8     | emd      |   8 |      12000 |      159.6279 |             161.1090 |            0.0133 |               0 |              0 |
| birdclef2023  | clean       | linear_k3  | linear   |   3 |      12000 |       35.7392 |              36.0629 |            0.0030 |               0 |              0 |
| birdclef2023  | clean       | linear_k4  | linear   |   4 |      12000 |       41.7952 |              42.2366 |            0.0035 |               0 |              0 |
| birdclef2023  | clean       | linear_k6  | linear   |   6 |      12000 |       55.3080 |              55.7819 |            0.0046 |               0 |              0 |
| birdclef2023  | clean       | linear_k8  | linear   |   8 |      12000 |       69.7768 |              70.4508 |            0.0058 |               0 |              0 |
| birdclef2023  | clean       | linear_k11 | linear   |  11 |      12000 |       91.0426 |              91.9684 |            0.0076 |               0 |              0 |
| birdclef2023  | clean       | irmf_k3    | irmf     |   3 |      12000 |      259.0330 |             261.3581 |            0.0216 |               0 |              0 |
| birdclef2023  | clean       | irmf_k4    | irmf     |   4 |      12000 |      307.7281 |             310.5676 |            0.0256 |               0 |              0 |
| birdclef2023  | clean       | irmf_k6    | irmf     |   6 |      12000 |      419.5206 |             423.2866 |            0.0350 |               0 |              0 |
| birdclef2023  | clean       | irmf_k8    | irmf     |   8 |      12000 |      533.7775 |             539.0234 |            0.0445 |               0 |              0 |
| birdclef2023  | clean       | irmf_k11   | irmf     |  11 |      12000 |      707.0690 |             713.6485 |            0.0589 |               0 |              0 |
| birdclef2023  | nb3_0.1     | raw        | raw      |   0 |      12000 |        5.4859 |               5.5156 |            0.0005 |               0 |              0 |
| birdclef2023  | nb3_0.1     | emd_k2     | emd      |   2 |      12000 |        8.0491 |               8.0851 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.1     | emd_k3     | emd      |   3 |      12000 |        8.8405 |               8.8863 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.1     | emd_k4     | emd      |   4 |      12000 |        9.9317 |               9.9824 |            0.0008 |               0 |              0 |
| birdclef2023  | nb3_0.1     | emd_k6     | emd      |   6 |      12000 |       12.5595 |              12.6485 |            0.0010 |               0 |              0 |
| birdclef2023  | nb3_0.1     | emd_k8     | emd      |   8 |      12000 |      474.0429 |             477.1366 |            0.0395 |               0 |              0 |
| birdclef2023  | nb3_0.1     | linear_k3  | linear   |   3 |      12000 |       35.6885 |              35.8455 |            0.0030 |               0 |              0 |
| birdclef2023  | nb3_0.1     | linear_k4  | linear   |   4 |      12000 |       41.4518 |              41.6704 |            0.0035 |               0 |              0 |
| birdclef2023  | nb3_0.1     | linear_k6  | linear   |   6 |      12000 |       55.0718 |              55.4025 |            0.0046 |               0 |              0 |
| birdclef2023  | nb3_0.1     | linear_k8  | linear   |   8 |      12000 |       69.3300 |              69.7593 |            0.0058 |               0 |              0 |
| birdclef2023  | nb3_0.1     | linear_k11 | linear   |  11 |      12000 |       90.8852 |              91.4417 |            0.0076 |               0 |              0 |
| birdclef2023  | nb3_0.1     | irmf_k3    | irmf     |   3 |      12000 |      268.3467 |             270.0447 |            0.0224 |               0 |              0 |
| birdclef2023  | nb3_0.1     | irmf_k4    | irmf     |   4 |      12000 |      320.2773 |             322.4352 |            0.0267 |               0 |              0 |
| birdclef2023  | nb3_0.1     | irmf_k6    | irmf     |   6 |      12000 |      436.9039 |             439.9326 |            0.0364 |               0 |              0 |
| birdclef2023  | nb3_0.1     | irmf_k8    | irmf     |   8 |      12000 |      555.5840 |             558.7321 |            0.0463 |               0 |              0 |
| birdclef2023  | nb3_0.1     | irmf_k11   | irmf     |  11 |      12000 |      736.6724 |             741.0167 |            0.0614 |               0 |              0 |
| birdclef2023  | nb3_0.2     | raw        | raw      |   0 |      12000 |        5.4946 |               5.5119 |            0.0005 |               0 |              0 |
| birdclef2023  | nb3_0.2     | emd_k2     | emd      |   2 |      12000 |        8.1090 |               8.1599 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.2     | emd_k3     | emd      |   3 |      12000 |        8.9001 |               9.0127 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.2     | emd_k4     | emd      |   4 |      12000 |        9.9876 |              10.0337 |            0.0008 |               0 |              0 |
| birdclef2023  | nb3_0.2     | emd_k6     | emd      |   6 |      12000 |       12.5678 |              12.6928 |            0.0010 |               0 |              0 |
| birdclef2023  | nb3_0.2     | emd_k8     | emd      |   8 |      12000 |      443.4229 |             446.0379 |            0.0370 |               0 |              0 |
| birdclef2023  | nb3_0.2     | linear_k3  | linear   |   3 |      12000 |       35.8077 |              36.0121 |            0.0030 |               0 |              0 |
| birdclef2023  | nb3_0.2     | linear_k4  | linear   |   4 |      12000 |       41.6818 |              42.0004 |            0.0035 |               0 |              0 |
| birdclef2023  | nb3_0.2     | linear_k6  | linear   |   6 |      12000 |       55.2684 |              55.6337 |            0.0046 |               0 |              0 |
| birdclef2023  | nb3_0.2     | linear_k8  | linear   |   8 |      12000 |       69.5747 |              69.9167 |            0.0058 |               0 |              0 |
| birdclef2023  | nb3_0.2     | linear_k11 | linear   |  11 |      12000 |       90.6519 |              91.0607 |            0.0076 |               0 |              0 |
| birdclef2023  | nb3_0.2     | irmf_k3    | irmf     |   3 |      12000 |      271.7547 |             273.4110 |            0.0226 |               0 |              0 |
| birdclef2023  | nb3_0.2     | irmf_k4    | irmf     |   4 |      12000 |      324.5993 |             327.0099 |            0.0270 |               0 |              0 |
| birdclef2023  | nb3_0.2     | irmf_k6    | irmf     |   6 |      12000 |      442.0495 |             444.5217 |            0.0368 |               0 |              0 |
| birdclef2023  | nb3_0.2     | irmf_k8    | irmf     |   8 |      12000 |      563.8280 |             568.5330 |            0.0470 |               0 |              0 |
| birdclef2023  | nb3_0.2     | irmf_k11   | irmf     |  11 |      12000 |      748.0480 |             753.5468 |            0.0623 |               0 |              0 |
| birdclef2023  | nb3_0.3     | raw        | raw      |   0 |      12000 |        5.5173 |               5.5394 |            0.0005 |               0 |              0 |
| birdclef2023  | nb3_0.3     | emd_k2     | emd      |   2 |      12000 |        8.1465 |               8.1813 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.3     | emd_k3     | emd      |   3 |      12000 |        8.9139 |               8.9541 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.3     | emd_k4     | emd      |   4 |      12000 |        9.9920 |              10.0589 |            0.0008 |               0 |              0 |
| birdclef2023  | nb3_0.3     | emd_k6     | emd      |   6 |      12000 |       12.5791 |              12.6496 |            0.0010 |               0 |              0 |
| birdclef2023  | nb3_0.3     | emd_k8     | emd      |   8 |      12000 |      380.6625 |             383.8431 |            0.0317 |               0 |              0 |
| birdclef2023  | nb3_0.3     | linear_k3  | linear   |   3 |      12000 |       35.7567 |              35.9816 |            0.0030 |               0 |              0 |
| birdclef2023  | nb3_0.3     | linear_k4  | linear   |   4 |      12000 |       41.5617 |              41.8729 |            0.0035 |               0 |              0 |
| birdclef2023  | nb3_0.3     | linear_k6  | linear   |   6 |      12000 |       55.1885 |              55.6849 |            0.0046 |               0 |              0 |
| birdclef2023  | nb3_0.3     | linear_k8  | linear   |   8 |      12000 |       69.4663 |              70.2645 |            0.0058 |               0 |              0 |
| birdclef2023  | nb3_0.3     | linear_k11 | linear   |  11 |      12000 |       90.8692 |              91.4656 |            0.0076 |               0 |              0 |
| birdclef2023  | nb3_0.3     | irmf_k3    | irmf     |   3 |      12000 |      273.1812 |             274.9477 |            0.0228 |               0 |              0 |
| birdclef2023  | nb3_0.3     | irmf_k4    | irmf     |   4 |      12000 |      325.5943 |             327.6310 |            0.0271 |               0 |              0 |
| birdclef2023  | nb3_0.3     | irmf_k6    | irmf     |   6 |      12000 |      444.2918 |             446.7381 |            0.0370 |               0 |              0 |
| birdclef2023  | nb3_0.3     | irmf_k8    | irmf     |   8 |      12000 |      565.9628 |             569.3556 |            0.0472 |               0 |              0 |
| birdclef2023  | nb3_0.3     | irmf_k11   | irmf     |  11 |      12000 |      750.1607 |             754.7281 |            0.0625 |               0 |              0 |
| birdclef2023  | nb3_0.4     | raw        | raw      |   0 |      12000 |        5.5143 |               5.5666 |            0.0005 |               0 |              0 |
| birdclef2023  | nb3_0.4     | emd_k2     | emd      |   2 |      12000 |        8.1356 |               8.1633 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.4     | emd_k3     | emd      |   3 |      12000 |        8.8851 |               8.9393 |            0.0007 |               0 |              0 |
| birdclef2023  | nb3_0.4     | emd_k4     | emd      |   4 |      12000 |        9.9632 |              10.0212 |            0.0008 |               0 |              0 |
| birdclef2023  | nb3_0.4     | emd_k6     | emd      |   6 |      12000 |       12.5221 |              12.5770 |            0.0010 |               0 |              0 |
| birdclef2023  | nb3_0.4     | emd_k8     | emd      |   8 |      12000 |      331.3026 |             333.5951 |            0.0276 |               0 |              0 |
| birdclef2023  | nb3_0.4     | linear_k3  | linear   |   3 |      12000 |       35.9753 |              36.1661 |            0.0030 |               0 |              0 |
| birdclef2023  | nb3_0.4     | linear_k4  | linear   |   4 |      12000 |       41.7516 |              42.2149 |            0.0035 |               0 |              0 |
| birdclef2023  | nb3_0.4     | linear_k6  | linear   |   6 |      12000 |       55.1929 |              55.4865 |            0.0046 |               0 |              0 |
| birdclef2023  | nb3_0.4     | linear_k8  | linear   |   8 |      12000 |       69.3634 |              69.7358 |            0.0058 |               0 |              0 |
| birdclef2023  | nb3_0.4     | linear_k11 | linear   |  11 |      12000 |       90.8095 |              91.3499 |            0.0076 |               0 |              0 |
| birdclef2023  | nb3_0.4     | irmf_k3    | irmf     |   3 |      12000 |      272.6428 |             274.5658 |            0.0227 |               0 |              0 |
| birdclef2023  | nb3_0.4     | irmf_k4    | irmf     |   4 |      12000 |      324.9864 |             326.9242 |            0.0271 |               0 |              0 |
| birdclef2023  | nb3_0.4     | irmf_k6    | irmf     |   6 |      12000 |      444.3041 |             447.5167 |            0.0370 |               0 |              0 |
| birdclef2023  | nb3_0.4     | irmf_k8    | irmf     |   8 |      12000 |      565.2002 |             569.8116 |            0.0471 |               0 |              0 |
| birdclef2023  | nb3_0.4     | irmf_k11   | irmf     |  11 |      12000 |      750.6264 |             755.7930 |            0.0626 |               0 |              0 |
| esc50_animals | clean       | raw        | raw      |   0 |       5100 |        2.2839 |               2.3032 |            0.0004 |               0 |              0 |
| esc50_animals | clean       | emd_k2     | emd      |   2 |       5100 |        3.3004 |               3.3289 |            0.0006 |               0 |              0 |
| esc50_animals | clean       | emd_k3     | emd      |   3 |       5100 |        3.6590 |               3.6984 |            0.0007 |               0 |              0 |
| esc50_animals | clean       | emd_k4     | emd      |   4 |       5100 |        4.1624 |               4.1970 |            0.0008 |               0 |              0 |
| esc50_animals | clean       | emd_k6     | emd      |   6 |       5100 |        5.2678 |               5.3588 |            0.0010 |               0 |              0 |
| esc50_animals | clean       | emd_k8     | emd      |   8 |       5100 |       79.4238 |              80.1596 |            0.0156 |               0 |              0 |
| esc50_animals | clean       | linear_k3  | linear   |   3 |       5100 |       15.0941 |              15.2991 |            0.0030 |               0 |              0 |
| esc50_animals | clean       | linear_k4  | linear   |   4 |       5100 |       17.5822 |              17.7739 |            0.0034 |               0 |              0 |
| esc50_animals | clean       | linear_k6  | linear   |   6 |       5100 |       23.2856 |              23.4964 |            0.0046 |               0 |              0 |
| esc50_animals | clean       | linear_k8  | linear   |   8 |       5100 |       29.3610 |              29.7132 |            0.0058 |               0 |              0 |
| esc50_animals | clean       | linear_k11 | linear   |  11 |       5100 |       38.3753 |              38.7181 |            0.0075 |               0 |              0 |
| esc50_animals | clean       | irmf_k3    | irmf     |   3 |       5100 |      124.8670 |             126.1021 |            0.0245 |               0 |              0 |
| esc50_animals | clean       | irmf_k4    | irmf     |   4 |       5100 |      148.9298 |             150.2005 |            0.0292 |               0 |              0 |
| esc50_animals | clean       | irmf_k6    | irmf     |   6 |       5100 |      202.6524 |             204.2962 |            0.0397 |               0 |              0 |
| esc50_animals | clean       | irmf_k8    | irmf     |   8 |       5100 |      258.0211 |             260.0633 |            0.0506 |               0 |              0 |
| esc50_animals | clean       | irmf_k11   | irmf     |  11 |       5100 |      342.2704 |             345.2136 |            0.0671 |               0 |              0 |
| esc50_animals | nb3_0.1     | raw        | raw      |   0 |       5100 |        2.3043 |               2.3044 |            0.0005 |               0 |              0 |
| esc50_animals | nb3_0.1     | emd_k2     | emd      |   2 |       5100 |        3.3983 |               3.4009 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.1     | emd_k3     | emd      |   3 |       5100 |        3.7278 |               3.7900 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.1     | emd_k4     | emd      |   4 |       5100 |        4.1745 |               4.1832 |            0.0008 |               0 |              0 |
| esc50_animals | nb3_0.1     | emd_k6     | emd      |   6 |       5100 |        5.2597 |               5.2667 |            0.0010 |               0 |              0 |
| esc50_animals | nb3_0.1     | emd_k8     | emd      |   8 |       5100 |      249.9098 |             250.6169 |            0.0490 |               0 |              0 |
| esc50_animals | nb3_0.1     | linear_k3  | linear   |   3 |       5100 |       15.1172 |              15.1598 |            0.0030 |               0 |              0 |
| esc50_animals | nb3_0.1     | linear_k4  | linear   |   4 |       5100 |       17.5812 |              17.6219 |            0.0034 |               0 |              0 |
| esc50_animals | nb3_0.1     | linear_k6  | linear   |   6 |       5100 |       23.3469 |              23.4026 |            0.0046 |               0 |              0 |
| esc50_animals | nb3_0.1     | linear_k8  | linear   |   8 |       5100 |       29.2962 |              29.3590 |            0.0057 |               0 |              0 |
| esc50_animals | nb3_0.1     | linear_k11 | linear   |  11 |       5100 |       38.1989 |              38.3000 |            0.0075 |               0 |              0 |
| esc50_animals | nb3_0.1     | irmf_k3    | irmf     |   3 |       5100 |      126.9015 |             127.1902 |            0.0249 |               0 |              0 |
| esc50_animals | nb3_0.1     | irmf_k4    | irmf     |   4 |       5100 |      150.8471 |             151.1743 |            0.0296 |               0 |              0 |
| esc50_animals | nb3_0.1     | irmf_k6    | irmf     |   6 |       5100 |      205.9490 |             206.4482 |            0.0404 |               0 |              0 |
| esc50_animals | nb3_0.1     | irmf_k8    | irmf     |   8 |       5100 |      261.9441 |             262.6537 |            0.0514 |               0 |              0 |
| esc50_animals | nb3_0.1     | irmf_k11   | irmf     |  11 |       5100 |      347.1289 |             347.9673 |            0.0681 |               0 |              0 |
| esc50_animals | nb3_0.2     | raw        | raw      |   0 |       5100 |        2.3195 |               2.3212 |            0.0005 |               0 |              0 |
| esc50_animals | nb3_0.2     | emd_k2     | emd      |   2 |       5100 |        3.4381 |               3.4424 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.2     | emd_k3     | emd      |   3 |       5100 |        3.7476 |               3.7509 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.2     | emd_k4     | emd      |   4 |       5100 |        4.1993 |               4.2058 |            0.0008 |               0 |              0 |
| esc50_animals | nb3_0.2     | emd_k6     | emd      |   6 |       5100 |        5.3156 |               5.3250 |            0.0010 |               0 |              0 |
| esc50_animals | nb3_0.2     | emd_k8     | emd      |   8 |       5100 |      234.9623 |             235.5560 |            0.0461 |               0 |              0 |
| esc50_animals | nb3_0.2     | linear_k3  | linear   |   3 |       5100 |       15.1737 |              15.2030 |            0.0030 |               0 |              0 |
| esc50_animals | nb3_0.2     | linear_k4  | linear   |   4 |       5100 |       17.6075 |              17.6491 |            0.0035 |               0 |              0 |
| esc50_animals | nb3_0.2     | linear_k6  | linear   |   6 |       5100 |       23.3092 |              23.4096 |            0.0046 |               0 |              0 |
| esc50_animals | nb3_0.2     | linear_k8  | linear   |   8 |       5100 |       29.3972 |              29.4808 |            0.0058 |               0 |              0 |
| esc50_animals | nb3_0.2     | linear_k11 | linear   |  11 |       5100 |       38.4317 |              38.5351 |            0.0075 |               0 |              0 |
| esc50_animals | nb3_0.2     | irmf_k3    | irmf     |   3 |       5100 |      127.7359 |             128.2554 |            0.0250 |               0 |              0 |
| esc50_animals | nb3_0.2     | irmf_k4    | irmf     |   4 |       5100 |      152.1947 |             152.6877 |            0.0298 |               0 |              0 |
| esc50_animals | nb3_0.2     | irmf_k6    | irmf     |   6 |       5100 |      207.1290 |             207.6134 |            0.0406 |               0 |              0 |
| esc50_animals | nb3_0.2     | irmf_k8    | irmf     |   8 |       5100 |      263.9105 |             264.6252 |            0.0517 |               0 |              0 |
| esc50_animals | nb3_0.2     | irmf_k11   | irmf     |  11 |       5100 |      349.5109 |             350.4032 |            0.0685 |               0 |              0 |
| esc50_animals | nb3_0.3     | raw        | raw      |   0 |       5100 |        2.3323 |               2.3384 |            0.0005 |               0 |              0 |
| esc50_animals | nb3_0.3     | emd_k2     | emd      |   2 |       5100 |        3.4525 |               3.4587 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.3     | emd_k3     | emd      |   3 |       5100 |        3.7751 |               3.7838 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.3     | emd_k4     | emd      |   4 |       5100 |        4.2257 |               4.2495 |            0.0008 |               0 |              0 |
| esc50_animals | nb3_0.3     | emd_k6     | emd      |   6 |       5100 |        5.3356 |               5.3581 |            0.0010 |               0 |              0 |
| esc50_animals | nb3_0.3     | emd_k8     | emd      |   8 |       5100 |      188.6009 |             189.0512 |            0.0370 |               0 |              0 |
| esc50_animals | nb3_0.3     | linear_k3  | linear   |   3 |       5100 |       15.2063 |              15.2620 |            0.0030 |               0 |              0 |
| esc50_animals | nb3_0.3     | linear_k4  | linear   |   4 |       5100 |       17.6484 |              17.7006 |            0.0035 |               0 |              0 |
| esc50_animals | nb3_0.3     | linear_k6  | linear   |   6 |       5100 |       23.3175 |              23.3728 |            0.0046 |               0 |              0 |
| esc50_animals | nb3_0.3     | linear_k8  | linear   |   8 |       5100 |       29.3173 |              29.3809 |            0.0057 |               0 |              0 |
| esc50_animals | nb3_0.3     | linear_k11 | linear   |  11 |       5100 |       38.2817 |              38.3916 |            0.0075 |               0 |              0 |
| esc50_animals | nb3_0.3     | irmf_k3    | irmf     |   3 |       5100 |      128.0089 |             128.2548 |            0.0251 |               0 |              0 |
| esc50_animals | nb3_0.3     | irmf_k4    | irmf     |   4 |       5100 |      152.7056 |             153.1012 |            0.0299 |               0 |              0 |
| esc50_animals | nb3_0.3     | irmf_k6    | irmf     |   6 |       5100 |      207.2469 |             207.6728 |            0.0406 |               0 |              0 |
| esc50_animals | nb3_0.3     | irmf_k8    | irmf     |   8 |       5100 |      264.8618 |             265.5707 |            0.0519 |               0 |              0 |
| esc50_animals | nb3_0.3     | irmf_k11   | irmf     |  11 |       5100 |      350.3339 |             351.3226 |            0.0687 |               0 |              0 |
| esc50_animals | nb3_0.4     | raw        | raw      |   0 |       5100 |        2.3360 |               2.3383 |            0.0005 |               0 |              0 |
| esc50_animals | nb3_0.4     | emd_k2     | emd      |   2 |       5100 |        3.4590 |               3.4634 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.4     | emd_k3     | emd      |   3 |       5100 |        3.7687 |               3.7728 |            0.0007 |               0 |              0 |
| esc50_animals | nb3_0.4     | emd_k4     | emd      |   4 |       5100 |        4.2151 |               4.2219 |            0.0008 |               0 |              0 |
| esc50_animals | nb3_0.4     | emd_k6     | emd      |   6 |       5100 |        5.2865 |               5.2962 |            0.0010 |               0 |              0 |
| esc50_animals | nb3_0.4     | emd_k8     | emd      |   8 |       5100 |      156.5092 |             156.9409 |            0.0307 |               0 |              0 |
| esc50_animals | nb3_0.4     | linear_k3  | linear   |   3 |       5100 |       15.2495 |              15.2838 |            0.0030 |               0 |              0 |
| esc50_animals | nb3_0.4     | linear_k4  | linear   |   4 |       5100 |       17.6669 |              17.7125 |            0.0035 |               0 |              0 |
| esc50_animals | nb3_0.4     | linear_k6  | linear   |   6 |       5100 |       23.3708 |              23.4628 |            0.0046 |               0 |              0 |
| esc50_animals | nb3_0.4     | linear_k8  | linear   |   8 |       5100 |       29.3296 |              29.4036 |            0.0058 |               0 |              0 |
| esc50_animals | nb3_0.4     | linear_k11 | linear   |  11 |       5100 |       38.5509 |              38.6363 |            0.0076 |               0 |              0 |
| esc50_animals | nb3_0.4     | irmf_k3    | irmf     |   3 |       5100 |      127.4500 |             127.7913 |            0.0250 |               0 |              0 |
| esc50_animals | nb3_0.4     | irmf_k4    | irmf     |   4 |       5100 |      152.7307 |             153.2303 |            0.0299 |               0 |              0 |
| esc50_animals | nb3_0.4     | irmf_k6    | irmf     |   6 |       5100 |      207.4573 |             208.0277 |            0.0407 |               0 |              0 |
| esc50_animals | nb3_0.4     | irmf_k8    | irmf     |   8 |       5100 |      264.2430 |             264.9647 |            0.0518 |               0 |              0 |
| esc50_animals | nb3_0.4     | irmf_k11   | irmf     |  11 |       5100 |      350.6253 |             351.4780 |            0.0688 |               0 |              0 |

Fit CPU total: 76007.69 s. Completed run records: {"workers": 14, "records": [{"started_utc": "2026-10-22T10:42:40.096417+00:00", "finished_utc": "2026-10-22T10:44:08.280015+00:00", "wall_seconds": 88.18318316689692, "workers": 14, "datasets": ["synthetic"], "stages": {"synthetic": {"clean_seconds_from_start": 11.176038458012044, "features_seconds_from_start": 60.58212433301378}}, "pimf": {"version": "0.4.0", "files": {"__init__.py": "fc7c0a84b39b1a4787a7973ded0eff50a882ee05b8f1d68a995941c1667c3391", "_erf.py": "01ffbc30bd9e50752980716d5c8c01d6be94ba2e724b253d5ccdc48117ab465e", "contrasts.py": "38556222cc9645c21285faaa8d388b3b53bd2c87758a79f03c64e9f000013c2f", "decompose.py": "f71fe8ba5e08f2f5ce7f3572c762346543ac124712760f74411f2253d3551688", "kernels.py": "fb65d469cb7fb594734e768bc8cde8a6926214c985c5f044e9f4517d3a220c11"}}}, {"started_utc": "2026-10-22T10:46:15.264933+00:00", "finished_utc": "2026-10-22T13:33:30.608670+00:00", "wall_seconds": 10035.480208375026, "workers": 14, "datasets": ["dronerf", "birdclef2023", "esc50_animals"], "stages": {"dronerf": {"clean_seconds_from_start": 399.5128910000203, "features_seconds_from_start": 2829.9532275409438}, "birdclef2023": {"clean_seconds_from_start": 603.1925223330036, "features_seconds_from_start": 3926.9388612080365}, "esc50_animals": {"clean_seconds_from_start": 699.8722303329268, "features_seconds_from_start": 4448.858191916021}}, "pimf": {"version": "0.4.0", "files": {"__init__.py": "fc7c0a84b39b1a4787a7973ded0eff50a882ee05b8f1d68a995941c1667c3391", "_erf.py": "01ffbc30bd9e50752980716d5c8c01d6be94ba2e724b253d5ccdc48117ab465e", "contrasts.py": "38556222cc9645c21285faaa8d388b3b53bd2c87758a79f03c64e9f000013c2f", "decompose.py": "f71fe8ba5e08f2f5ce7f3572c762346543ac124712760f74411f2253d3551688", "kernels.py": "fb65d469cb7fb594734e768bc8cde8a6926214c985c5f044e9f4517d3a220c11"}}}]}.

Run from `experiments/classification-emd-irmf` with `IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification` and OMP, OpenBLAS, MKL, VECLIB, NumExpr and Numba thread variables set to 1.

```bash
../../.venv/bin/python -m pytest -q robustness_v3/tests robustness_laplace/tests
../../.venv/bin/python -m robustness_v3.run --workers 14 --datasets dronerf birdclef2023 esc50_animals --detach
../../.venv/bin/python -m robustness_v3.analyse
```

Source/input hashes, fixed H calibration, cached-feature verification, every saved prediction metric and its clean baseline are checked by analyse before report publication. CSV score units are fractions. Figures use percentage points.
