# dronerf_pilot_spectrogram: dronerf / 10class

2 fits on shared split IDs [0]; wall time 30.2 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                             |   mean |   std |
|:--------------------------------------------|-------:|------:|
| ('raw', 'comp', 'log_stft_cnn2d')           | 0.3650 |   nan |
| ('raw', 'comp+log_scale', 'log_stft_cnn2d') | 0.3738 |   nan |

Raw log-STFT reference, deterministic per-frame transform with no train/test-fitted moments.
Three Conv2d/GroupNorm/ReLU/MaxPool2d blocks, global mean/max pooling, optional train-standardized log_scale.
Same grouped inner validation, balanced cross entropy, AdamW, epoch selection and fresh outer refit as 1-D CNN.
Different architecture and representation: an exploratory time-frequency reference, not a decomposition pair.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
