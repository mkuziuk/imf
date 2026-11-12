# IRMF noise-scale pilot

Decision at c = 3: **KEEP E1**. E2 has lower component error in 8/12 real-dataset cells. Its stability criterion is no worse in 4/12 cells. The pre-specified requirements are at least 9 component-error wins and at least 7 stability cells. The rule is applied to full-precision values, so table rounding does not determine a win.

| c | E2 lower component error, /12 | E2 stability no worse, /12 | Rule outcome |
| --- | --- | --- | --- |
| 2 | 6 | 7 | KEEP E1 |
| 3 | 8 | 4 | KEEP E1 |
| 12 | 9 | 6 | KEEP E1 |

The main run took 168.74 seconds with 8 workers. It evaluated 100 excluded frames per dataset, 500 channels in total, including both DroneRF channels. No classification was run. All generated files and caches are under `/tmp/sigma_pilot/`. No repository or dataset file was edited, staged or committed.

Both fixed-H controls have lower component error than both adaptive estimators in all 12 real-data cells at each c. This informational result supports testing a training-calibrated H next; it does not change the E1-versus-E2 decision.

## Protocol and units

The protocol was written to `protocol.json` before validation or pilot outcomes were inspected. The full grid is p = 0, .1, .2, .3, .4 and c = 2, 3, 12. Evaluation IDs are exactly `robustness.contamination.subset_ids(dataset, n_frames)`; each ID list is saved as `pilot_ids_<dataset>.csv`. Frames and other NumPy arrays are loaded with `mmap_mode="r"` and `allow_pickle=False`.

For each raw prepared channel x, set `b = c * noise_sigma_hat(x)[0]` and add `(u < p) * b * e * s`, using the unmodified `robustness.contamination.draws` common random numbers. The Gaussian draw returned by that helper is unused. Normalize each clean or contaminated input separately with `emdirmf.decompose.normalize_frame`. E1 is the existing `noise_sigma_hat`; E2 uses linear empirical 25th/75th quantiles of signed first differences, divided by `1.349 * sqrt(2)`. E2 uses the same mean-absolute-difference fallback and constant-signal fallback as E1. No extra estimator enters the comparison.

Each adaptive method sets `H = 2 * sigma_hat` on the current normalized input. Its reference is the clean channel decomposed with that same method, using that clean input's H. Criterion 1 is the median over channels of `abs(log(sigma_p / sigma_0))`. Criterion 2 uses `sigma_true = meta.sigma * meta.gain / normalization_scale_p`, with the current input's normalization scale. Criterion 3 pools every channel and extracted slot in a dataset cell, then takes the median of `norm(C_p - C_0) / norm(C_0)`. The residual is excluded. This is a median over channel-slot pairs, rather than a median of each channel's median. All clean component norms were positive; no denominator floor or exclusion was needed.

"Clean" means the original prepared frame before this pilot's added contamination. Synthetic prepared frames already contain Gaussian noise and, in two of the three regimes, baseline impulses. Synthetic accuracy targets only the known per-frame Gaussian SD, not the total background-plus-signal scale. The clean diff estimate used to define b is not an oracle Gaussian SD.

Threshold-unit convention: H is in normalized units. The reported `b/H` is `(b_raw / normalization_scale_p) / H_p`; directly dividing raw b by normalized H would be invalid. `H_p/H_0` uses normalized thresholds, as requested. The CSV also includes `H_raw_inflation = (H_p * scale_p) / (H_0 * scale_0)`, which separates the effect of changing normalization. At p = 0, b is a nominal scale that is not injected. E1 then has b/H exactly c/2.

## The decision cells

Each row is a real dataset and p > 0. Lower component error and lower stability values are better. A component win is strict; stability allows equality. The primary decision uses c = 3. `E2/E1` is the ratio of the two cell medians, not the median of channel-wise ratios.

### c = 3

| Dataset | p | Error E1 | Error E2 | E2/E1 | E2 error win | Stability E1 | Stability E2 | E2 stable no worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dronerf | 0.100000 | 0.782000 | 0.780538 | 0.998130 | yes | 0.112390 | 0.121060 | no |
| dronerf | 0.200000 | 1.104651 | 1.104830 | 1.000163 | no | 0.188394 | 0.175440 | yes |
| dronerf | 0.300000 | 1.403513 | 1.405243 | 1.001233 | no | 0.222967 | 0.222896 | yes |
| dronerf | 0.400000 | 1.687734 | 1.690296 | 1.001518 | no | 0.237225 | 0.242528 | no |
| birdclef2023 | 0.100000 | 0.916625 | 0.915090 | 0.998325 | yes | 0.081495 | 0.081831 | no |
| birdclef2023 | 0.200000 | 1.327983 | 1.324718 | 0.997541 | yes | 0.179316 | 0.179966 | no |
| birdclef2023 | 0.300000 | 1.603531 | 1.603145 | 0.999759 | yes | 0.260411 | 0.267009 | no |
| birdclef2023 | 0.400000 | 1.774078 | 1.771699 | 0.998659 | yes | 0.350916 | 0.349096 | yes |
| esc50_animals | 0.100000 | 0.205634 | 0.205080 | 0.997302 | yes | 0.108399 | 0.109007 | no |
| esc50_animals | 0.200000 | 0.310496 | 0.311772 | 1.004112 | no | 0.228773 | 0.231652 | no |
| esc50_animals | 0.300000 | 0.391158 | 0.390327 | 0.997876 | yes | 0.351907 | 0.351022 | yes |
| esc50_animals | 0.400000 | 0.470342 | 0.470157 | 0.999605 | yes | 0.481620 | 0.484640 | no |


### c = 2

| Dataset | p | Error E1 | Error E2 | E2/E1 | E2 error win | Stability E1 | Stability E2 | E2 stable no worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dronerf | 0.100000 | 0.637586 | 0.633487 | 0.993571 | yes | 0.090303 | 0.100466 | no |
| dronerf | 0.200000 | 0.888788 | 0.888645 | 0.999839 | yes | 0.128514 | 0.122529 | yes |
| dronerf | 0.300000 | 1.069009 | 1.069764 | 1.000706 | no | 0.170529 | 0.175237 | no |
| dronerf | 0.400000 | 1.266496 | 1.266784 | 1.000227 | no | 0.172895 | 0.170696 | yes |
| birdclef2023 | 0.100000 | 0.741319 | 0.739976 | 0.998189 | yes | 0.065779 | 0.063447 | yes |
| birdclef2023 | 0.200000 | 1.064039 | 1.063191 | 0.999204 | yes | 0.135656 | 0.135534 | yes |
| birdclef2023 | 0.300000 | 1.266694 | 1.267340 | 1.000510 | no | 0.202439 | 0.202390 | yes |
| birdclef2023 | 0.400000 | 1.405759 | 1.403112 | 0.998117 | yes | 0.270369 | 0.269356 | yes |
| esc50_animals | 0.100000 | 0.156740 | 0.156766 | 1.000168 | no | 0.086520 | 0.089212 | no |
| esc50_animals | 0.200000 | 0.234716 | 0.234766 | 1.000212 | no | 0.188003 | 0.187413 | yes |
| esc50_animals | 0.300000 | 0.300988 | 0.301745 | 1.002517 | no | 0.274071 | 0.277609 | no |
| esc50_animals | 0.400000 | 0.354677 | 0.354063 | 0.998269 | yes | 0.366964 | 0.372992 | no |


### c = 12

| Dataset | p | Error E1 | Error E2 | E2/E1 | E2 error win | Stability E1 | Stability E2 | E2 stable no worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dronerf | 0.100000 | 1.174945 | 1.176343 | 1.001190 | no | 0.241162 | 0.182322 | yes |
| dronerf | 0.200000 | 2.088169 | 2.081946 | 0.997020 | yes | 0.288228 | 0.288380 | no |
| dronerf | 0.300000 | 3.074661 | 3.083175 | 1.002769 | no | 0.471230 | 0.464927 | yes |
| dronerf | 0.400000 | 3.544310 | 3.538095 | 0.998246 | yes | 0.847298 | 0.836306 | yes |
| birdclef2023 | 0.100000 | 1.473957 | 1.467615 | 0.995697 | yes | 0.130835 | 0.129533 | yes |
| birdclef2023 | 0.200000 | 2.334096 | 2.331774 | 0.999005 | yes | 0.296878 | 0.295912 | yes |
| birdclef2023 | 0.300000 | 2.990248 | 2.983940 | 0.997890 | yes | 0.502681 | 0.509241 | no |
| birdclef2023 | 0.400000 | 3.810787 | 3.807951 | 0.999256 | yes | 0.847273 | 0.851250 | no |
| esc50_animals | 0.100000 | 0.368030 | 0.367854 | 0.999522 | yes | 0.150803 | 0.152108 | no |
| esc50_animals | 0.200000 | 0.557800 | 0.557775 | 0.999955 | yes | 0.335811 | 0.335394 | yes |
| esc50_animals | 0.300000 | 0.725851 | 0.727491 | 1.002260 | no | 0.602975 | 0.603423 | no |
| esc50_animals | 0.400000 | 0.929684 | 0.928812 | 0.999062 | yes | 1.011118 | 1.015120 | no |


At c = 3 the E2/E1 component-error ratios span 0.997302 to 1.004112. E1 and E2 target the same population scale when the signed difference distribution is symmetric about zero, because then its IQR is twice its median absolute value, and 1.349 is twice .6745. This observation explains why an IQR replacement need not cure contamination sensitivity; the decision above comes from the measured component errors.

## Synthetic accuracy and stability

Entries are channel medians of absolute log error. Lower is better. The p = 0 accuracy is repeated over c because no new noise is added at p = 0.

| c | p | Accuracy E1 | Accuracy E2 | Stability E1 | Stability E2 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.000000 | 0.105484 | 0.104124 | 0.000000 | 0.000000 |
| 2 | 0.100000 | 0.206585 | 0.203715 | 0.064469 | 0.068193 |
| 2 | 0.200000 | 0.313795 | 0.311369 | 0.122196 | 0.122545 |
| 2 | 0.300000 | 0.430624 | 0.425462 | 0.183730 | 0.181420 |
| 2 | 0.400000 | 0.547211 | 0.542485 | 0.250848 | 0.248928 |
| 3 | 0.000000 | 0.105484 | 0.104124 | 0.000000 | 0.000000 |
| 3 | 0.100000 | 0.244317 | 0.248661 | 0.084096 | 0.083346 |
| 3 | 0.200000 | 0.391107 | 0.385915 | 0.163909 | 0.164622 |
| 3 | 0.300000 | 0.538170 | 0.537484 | 0.242056 | 0.237719 |
| 3 | 0.400000 | 0.706775 | 0.701266 | 0.332144 | 0.329827 |
| 12 | 0.000000 | 0.105484 | 0.104124 | 0.000000 | 0.000000 |
| 12 | 0.100000 | 0.325984 | 0.321198 | 0.120343 | 0.120856 |
| 12 | 0.200000 | 0.579215 | 0.579461 | 0.272852 | 0.273204 |
| 12 | 0.300000 | 0.949802 | 0.949407 | 0.501352 | 0.500640 |
| 12 | 0.400000 | 1.498191 | 1.488528 | 0.898817 | 0.878071 |


## b/H and normalized H inflation

Every entry is the median of the per-channel ratio. This table uses correctly matched amplitude units. The p = 0 rows show the nominal outlier scale relative to the clean H. A value below 1 means the contamination's exponential scale is below H; individual exponential draws can still exceed H.

### c = 2

| Dataset | p | b/H E1 | b/H E2 | H_p/H_0 E1 | H_p/H_0 E2 |
| --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 1.000000 | 1.001320 | 1.000000 | 1.000000 |
| synthetic | 0.100000 | 0.900344 | 0.900823 | 1.066593 | 1.070573 |
| synthetic | 0.200000 | 0.804102 | 0.805810 | 1.129976 | 1.130370 |
| synthetic | 0.300000 | 0.723124 | 0.725273 | 1.201691 | 1.198919 |
| synthetic | 0.400000 | 0.648016 | 0.649260 | 1.285115 | 1.282652 |
| dronerf | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| dronerf | 0.100000 | 0.875722 | 0.877470 | 1.078575 | 1.091395 |
| dronerf | 0.200000 | 0.759615 | 0.762278 | 1.118171 | 1.119516 |
| dronerf | 0.300000 | 0.687996 | 0.689680 | 1.171105 | 1.172117 |
| dronerf | 0.400000 | 0.634971 | 0.634589 | 1.180840 | 1.181440 |
| birdclef2023 | 0.000000 | 1.000000 | 1.000427 | 1.000000 | 1.000000 |
| birdclef2023 | 0.100000 | 0.897198 | 0.898427 | 1.067991 | 1.065503 |
| birdclef2023 | 0.200000 | 0.801373 | 0.804971 | 1.145292 | 1.145148 |
| birdclef2023 | 0.300000 | 0.713615 | 0.711508 | 1.224385 | 1.224325 |
| birdclef2023 | 0.400000 | 0.639568 | 0.642364 | 1.310448 | 1.309124 |
| esc50_animals | 0.000000 | 1.000000 | 1.000938 | 1.000000 | 1.000000 |
| esc50_animals | 0.100000 | 0.899005 | 0.900938 | 1.090373 | 1.093312 |
| esc50_animals | 0.200000 | 0.803586 | 0.804844 | 1.206837 | 1.206127 |
| esc50_animals | 0.300000 | 0.720028 | 0.719658 | 1.315308 | 1.319971 |
| esc50_animals | 0.400000 | 0.644550 | 0.647286 | 1.443348 | 1.452072 |


### c = 3

| Dataset | p | b/H E1 | b/H E2 | H_p/H_0 E1 | H_p/H_0 E2 |
| --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 1.500000 | 1.501980 | 1.000000 | 1.000000 |
| synthetic | 0.100000 | 1.301441 | 1.305993 | 1.087734 | 1.086918 |
| synthetic | 0.200000 | 1.129578 | 1.132394 | 1.178107 | 1.178948 |
| synthetic | 0.300000 | 0.975527 | 0.978879 | 1.273866 | 1.268352 |
| synthetic | 0.400000 | 0.823434 | 0.828168 | 1.393954 | 1.390728 |
| dronerf | 0.000000 | 1.500000 | 1.500000 | 1.000000 | 1.000000 |
| dronerf | 0.100000 | 1.250477 | 1.261916 | 1.111111 | 1.111111 |
| dronerf | 0.200000 | 1.090909 | 1.092120 | 1.174088 | 1.164897 |
| dronerf | 0.300000 | 0.964341 | 0.963488 | 1.241978 | 1.239366 |
| dronerf | 0.400000 | 0.769004 | 0.771181 | 1.267726 | 1.274468 |
| birdclef2023 | 0.000000 | 1.500000 | 1.500641 | 1.000000 | 1.000000 |
| birdclef2023 | 0.100000 | 1.303097 | 1.302786 | 1.084908 | 1.085273 |
| birdclef2023 | 0.200000 | 1.119933 | 1.122275 | 1.196398 | 1.197180 |
| birdclef2023 | 0.300000 | 0.952840 | 0.950727 | 1.297464 | 1.306052 |
| birdclef2023 | 0.400000 | 0.815328 | 0.814618 | 1.420369 | 1.417791 |
| esc50_animals | 0.000000 | 1.500000 | 1.501406 | 1.000000 | 1.000000 |
| esc50_animals | 0.100000 | 1.306394 | 1.310345 | 1.114493 | 1.115170 |
| esc50_animals | 0.200000 | 1.127797 | 1.130711 | 1.257057 | 1.260681 |
| esc50_animals | 0.300000 | 0.964244 | 0.965334 | 1.421780 | 1.420519 |
| esc50_animals | 0.400000 | 0.823765 | 0.825228 | 1.618695 | 1.623592 |


### c = 12

| Dataset | p | b/H E1 | b/H E2 | H_p/H_0 E1 | H_p/H_0 E2 |
| --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 6.000000 | 6.007921 | 1.000000 | 1.000000 |
| synthetic | 0.100000 | 4.803000 | 4.788292 | 1.127884 | 1.128463 |
| synthetic | 0.200000 | 3.712125 | 3.712428 | 1.313706 | 1.314169 |
| synthetic | 0.300000 | 2.587963 | 2.603103 | 1.650953 | 1.649792 |
| synthetic | 0.400000 | 1.479103 | 1.479275 | 2.456697 | 2.406258 |
| dronerf | 0.000000 | 6.000000 | 6.000000 | 1.000000 | 1.000000 |
| dronerf | 0.100000 | 4.500000 | 4.500000 | 1.211111 | 1.200000 |
| dronerf | 0.200000 | 3.556301 | 3.543391 | 1.334062 | 1.334265 |
| dronerf | 0.300000 | 2.403431 | 2.453497 | 1.601965 | 1.591899 |
| dronerf | 0.400000 | 1.524831 | 1.543090 | 2.333333 | 2.307831 |
| birdclef2023 | 0.000000 | 6.000000 | 6.002563 | 1.000000 | 1.000000 |
| birdclef2023 | 0.100000 | 4.765822 | 4.775844 | 1.139780 | 1.138297 |
| birdclef2023 | 0.200000 | 3.609603 | 3.608570 | 1.345651 | 1.344352 |
| birdclef2023 | 0.300000 | 2.547487 | 2.551386 | 1.653152 | 1.664035 |
| birdclef2023 | 0.400000 | 1.528019 | 1.528601 | 2.333275 | 2.342576 |
| esc50_animals | 0.000000 | 6.000000 | 6.005626 | 1.000000 | 1.000000 |
| esc50_animals | 0.100000 | 4.813895 | 4.818629 | 1.162767 | 1.164286 |
| esc50_animals | 0.200000 | 3.701823 | 3.689637 | 1.399074 | 1.398492 |
| esc50_animals | 0.300000 | 2.612384 | 2.611946 | 1.827549 | 1.828369 |
| esc50_animals | 0.400000 | 1.550467 | 1.550479 | 2.748672 | 2.759694 |


Raw-equivalent threshold inflation is in `threshold_diagnostics.csv` and `summary.csv`. For reference, the p = .4 values are below. They can exceed the normalized inflation because the contaminated input has a different normalization scale.

| c | Dataset | Raw-equivalent H ratio E1 | Raw-equivalent H ratio E2 |
| --- | --- | --- | --- |
| 2 | birdclef2023 | 1.563556 | 1.558852 |
| 2 | dronerf | 1.574877 | 1.565771 |
| 2 | esc50_animals | 1.551471 | 1.544114 |
| 2 | synthetic | 1.543173 | 1.537586 |
| 3 | birdclef2023 | 1.839750 | 1.842272 |
| 3 | dronerf | 1.950580 | 1.870613 |
| 3 | esc50_animals | 1.820908 | 1.816645 |
| 3 | synthetic | 1.821639 | 1.813774 |
| 12 | birdclef2023 | 3.926677 | 3.929219 |
| 12 | dronerf | 3.934868 | 3.871564 |
| 12 | esc50_animals | 3.869802 | 3.899175 |
| 12 | synthetic | 4.056580 | 4.056918 |


## Fixed-H control

F1 uses twice the median clean normalized E1 estimate over all training channels from stored split 0. F2 uses the corresponding E2 estimate. Both exclude every pilot ID and every split-0 test ID. This clean training calibration is the only non-pilot frame access for estimation; decomposition and error evaluation use only the specified pilots. Training IDs are saved in `fixed_H_training_ids.csv.gz`. H is fixed in normalized units per dataset and used unchanged for both the clean reference and contaminated input. This is a training-population control, not a clean per-channel oracle, and it is not an additional adoption candidate.

| Dataset | Training frames | Training channels | Fixed H F1 | Fixed H F2 |
| --- | --- | --- | --- | --- |
| synthetic | 1123 | 1123 | 1.172437 | 1.172606 |
| dronerf | 9487 | 18974 | 2.057010 | 2.033216 |
| birdclef2023 | 9512 | 9512 | 0.959920 | 0.959329 |
| esc50_animals | 3947 | 3947 | 0.517834 | 0.516582 |


| c | F1 beats both adaptive methods, /12 real cells | F2 beats both adaptive methods, /12 real cells |
| --- | --- | --- |
| 2 | 12 | 12 |
| 3 | 12 | 12 |
| 12 | 12 | 12 |


The next tables include synthetic and real datasets. A fixed method is marked as beating both only when its cell median is strictly smaller than both E1 and E2. At p = 0 all errors are exactly zero. Each method has its own clean reference, as in the primary estimand.

### c = 2

| Dataset | p | Error E1 | Error E2 | Error fixed F1 | Error fixed F2 | F1 beats both | F2 beats both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| synthetic | 0.100000 | 0.315366 | 0.315038 | 0.307780 | 0.307791 | yes | yes |
| synthetic | 0.200000 | 0.440450 | 0.440563 | 0.430730 | 0.430742 | yes | yes |
| synthetic | 0.300000 | 0.553211 | 0.552640 | 0.532222 | 0.532233 | yes | yes |
| synthetic | 0.400000 | 0.639419 | 0.639976 | 0.610763 | 0.610782 | yes | yes |
| dronerf | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| dronerf | 0.100000 | 0.637586 | 0.633487 | 0.627015 | 0.624333 | yes | yes |
| dronerf | 0.200000 | 0.888788 | 0.888645 | 0.872439 | 0.868980 | yes | yes |
| dronerf | 0.300000 | 1.069009 | 1.069764 | 1.049860 | 1.046094 | yes | yes |
| dronerf | 0.400000 | 1.266496 | 1.266784 | 1.229906 | 1.226409 | yes | yes |
| birdclef2023 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| birdclef2023 | 0.100000 | 0.741319 | 0.739976 | 0.670914 | 0.670612 | yes | yes |
| birdclef2023 | 0.200000 | 1.064039 | 1.063191 | 0.963662 | 0.963568 | yes | yes |
| birdclef2023 | 0.300000 | 1.266694 | 1.267340 | 1.147651 | 1.147338 | yes | yes |
| birdclef2023 | 0.400000 | 1.405759 | 1.403112 | 1.255696 | 1.255622 | yes | yes |
| esc50_animals | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| esc50_animals | 0.100000 | 0.156740 | 0.156766 | 0.155678 | 0.155719 | yes | yes |
| esc50_animals | 0.200000 | 0.234716 | 0.234766 | 0.221673 | 0.221834 | yes | yes |
| esc50_animals | 0.300000 | 0.300988 | 0.301745 | 0.279836 | 0.279844 | yes | yes |
| esc50_animals | 0.400000 | 0.354677 | 0.354063 | 0.325705 | 0.325644 | yes | yes |


### c = 3

| Dataset | p | Error E1 | Error E2 | Error fixed F1 | Error fixed F2 | F1 beats both | F2 beats both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| synthetic | 0.100000 | 0.382181 | 0.382206 | 0.375919 | 0.375937 | yes | yes |
| synthetic | 0.200000 | 0.552537 | 0.552570 | 0.534407 | 0.534434 | yes | yes |
| synthetic | 0.300000 | 0.699560 | 0.699292 | 0.656833 | 0.656849 | yes | yes |
| synthetic | 0.400000 | 0.823231 | 0.821583 | 0.748655 | 0.748671 | yes | yes |
| dronerf | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| dronerf | 0.100000 | 0.782000 | 0.780538 | 0.764750 | 0.760581 | yes | yes |
| dronerf | 0.200000 | 1.104651 | 1.104830 | 1.073324 | 1.067898 | yes | yes |
| dronerf | 0.300000 | 1.403513 | 1.405243 | 1.323408 | 1.318197 | yes | yes |
| dronerf | 0.400000 | 1.687734 | 1.690296 | 1.548515 | 1.542389 | yes | yes |
| birdclef2023 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| birdclef2023 | 0.100000 | 0.916625 | 0.915090 | 0.823978 | 0.824016 | yes | yes |
| birdclef2023 | 0.200000 | 1.327983 | 1.324718 | 1.189363 | 1.189021 | yes | yes |
| birdclef2023 | 0.300000 | 1.603531 | 1.603145 | 1.374288 | 1.373773 | yes | yes |
| birdclef2023 | 0.400000 | 1.774078 | 1.771699 | 1.514388 | 1.514154 | yes | yes |
| esc50_animals | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| esc50_animals | 0.100000 | 0.205634 | 0.205080 | 0.199473 | 0.199440 | yes | yes |
| esc50_animals | 0.200000 | 0.310496 | 0.311772 | 0.289075 | 0.288917 | yes | yes |
| esc50_animals | 0.300000 | 0.391158 | 0.390327 | 0.360523 | 0.360553 | yes | yes |
| esc50_animals | 0.400000 | 0.470342 | 0.470157 | 0.414344 | 0.414513 | yes | yes |


### c = 12

| Dataset | p | Error E1 | Error E2 | Error fixed F1 | Error fixed F2 | F1 beats both | F2 beats both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| synthetic | 0.100000 | 0.577196 | 0.575521 | 0.541720 | 0.541766 | yes | yes |
| synthetic | 0.200000 | 0.928111 | 0.929444 | 0.821952 | 0.821996 | yes | yes |
| synthetic | 0.300000 | 1.352039 | 1.349720 | 1.043987 | 1.044046 | yes | yes |
| synthetic | 0.400000 | 1.894249 | 1.887414 | 1.253282 | 1.253379 | yes | yes |
| dronerf | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| dronerf | 0.100000 | 1.174945 | 1.176343 | 1.126371 | 1.118844 | yes | yes |
| dronerf | 0.200000 | 2.088169 | 2.081946 | 1.694395 | 1.681500 | yes | yes |
| dronerf | 0.300000 | 3.074661 | 3.083175 | 2.230958 | 2.214471 | yes | yes |
| dronerf | 0.400000 | 3.544310 | 3.538095 | 2.534123 | 2.518373 | yes | yes |
| birdclef2023 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| birdclef2023 | 0.100000 | 1.473957 | 1.467615 | 1.250723 | 1.250205 | yes | yes |
| birdclef2023 | 0.200000 | 2.334096 | 2.331774 | 1.737791 | 1.736809 | yes | yes |
| birdclef2023 | 0.300000 | 2.990248 | 2.983940 | 2.105016 | 2.104118 | yes | yes |
| birdclef2023 | 0.400000 | 3.810787 | 3.807951 | 2.283457 | 2.282281 | yes | yes |
| esc50_animals | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no | no |
| esc50_animals | 0.100000 | 0.368030 | 0.367854 | 0.355909 | 0.355660 | yes | yes |
| esc50_animals | 0.200000 | 0.557800 | 0.557775 | 0.512284 | 0.512280 | yes | yes |
| esc50_animals | 0.300000 | 0.725851 | 0.727491 | 0.622423 | 0.622006 | yes | yes |
| esc50_animals | 0.400000 | 0.929684 | 0.928812 | 0.728315 | 0.728437 | yes | yes |


The fixed-H result is specific to this split-0 training calibration, shortened schedule, input normalization and component-error metric. It does not establish a classification improvement or justify using evaluation frames to set H. Full classification would recalibrate within each training split.

## Schedule and numerical validation

`h1 = .25`, `a = sqrt(2)`, `k_max = None`, `h_min = 15/L`, squared-triangle kernel, wrap boundary, `workers = 1` inside each decomposition. The actual pimf schedules, coarse to fine, are:

- L = 1024, h_min = 0.0146484375, 9 extracted components: 513, 365, 257, 183, 129, 93, 65, 47, 33.
- L = 2048, h_min = 0.00732421875, 11 extracted components: 1025, 727, 513, 365, 257, 183, 129, 93, 65, 47, 33.

The stored component order reverses these windows to fine-to-coarse, then appends the residual. The final scheduled radius is 16 samples, so the last pimf window is 33. The next radius would be about 11.314, below the 15-sample cutoff. The wrapper calls the repository's unmodified `NewtonSmoothAbs` through `pimf.IMF`. It exposes only H and h_min, preserving the production solver, driver, root tolerance and maximum iterations.

Validation at the original default schedule matched `newton_components` exactly, maximum absolute difference zero on all 9 clean checks spanning all datasets and both DroneRF channels. Fallback cases were checked on constant and sparse-jump scratch signals.

The full run completed 26,000 unique decompositions, producing 30,000 channel-condition-method rows and 318,000 extracted-slot errors. Clean p = 0 decompositions were reused across c. Every p = 0 component error is exactly zero. Failed solver rows: 0. Retries with cap 1000: 0. Maximum Newton iterations: 75. Maximum score residual: 1.58e-15. Maximum absolute reconstruction error: 9.09e-13. Smallest clean component norm: 0.00319. Estimator fallback rows: 0. Normalization fallback rows: 0. Loaded thread pools were limited to 1 thread. Summed worker CPU time was 1279.65 seconds.

Independent audit replays re-created 24 selected E1/E2 pipelines, chosen by the first pilot ID and pre-set grid points, through direct `pimf.IMF` calls. The largest per-slot error difference was 0. The audit also checked every pooled cell median from saved L2 numerators and denominators, all pilot ID sets, and the synthetic sigma-times-gain conversion. No errors or channels were silently dropped. These are descriptive pilot results, not a significance test.

## Reproduction

Exact commands, from this machine. All scripts are included under the scratch directory. `run_pilot.py` can resume completed channel files. To force a fresh independent run, first move the scratch `channel_results` directory and the generated aggregate files aside.

```bash
source /tmp/sigma_pilot/env.sh
$PY /tmp/sigma_pilot/validate.py > /tmp/sigma_pilot/validation.log 2>&1
$PY /tmp/sigma_pilot/run_pilot.py --workers 8 > /tmp/sigma_pilot/run.log 2>&1
$PY /tmp/sigma_pilot/audit_results.py > /tmp/sigma_pilot/audit.log 2>&1
$PY /tmp/sigma_pilot/build_report.py
```

`env.sh` sets the working directory to the study, `IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification`, the requested repository `.venv/bin/python`, all BLAS/OpenMP/Numba thread limits to 1, `PYTHONDONTWRITEBYTECODE=1`, and Numba/Matplotlib caches under scratch. It adds the study and scratch folders to `PYTHONPATH`.

To rebuild the aggregate tables without rerunning decompositions:

```bash
source /tmp/sigma_pilot/env.sh
$PY /tmp/sigma_pilot/run_pilot.py --aggregate-only
$PY /tmp/sigma_pilot/audit_results.py > /tmp/sigma_pilot/audit.log 2>&1
$PY /tmp/sigma_pilot/build_report.py
```

## Files

- `decision_summary.csv`, `decision_cells.csv`: the rule and its 36 real cells across c.
- `summary.csv`, `cell_comparison.csv`: all 60 dataset-by-p-by-c cells, all methods.
- `estimator_criteria.csv`: E1/E2 stability and synthetic accuracy.
- `threshold_diagnostics.csv`: median b/H, normalized and raw-equivalent H inflation, and H.
- `channel_metrics.csv.gz`, `slot_errors.csv.gz`: channel metrics and all extracted-slot errors, with clean norms and L2 differences.
- `fixed_H_calibration.csv`, `fixed_H_training_ids.csv.gz`, `fixed_H.json`: the two training-only controls.
- `validation.csv`, `independent_replays.csv`, `quality.json`, `solver_diagnostics.csv.gz`: validation and convergence evidence.
- `schedule.json`, `protocol.json`, `timing.json`, `versions.json`, `source_manifest.json`: reproducibility records.
- `source_snapshots/`: copies of the unchanged source modules used by the pilot.
- `channel_results/`: resumable per-channel results, with numerical diagnostics.
- `env.sh`, `pilot_core.py`, `validate.py`, `run_pilot.py`, `audit_results.py`, `build_report.py`: exact scripts.
