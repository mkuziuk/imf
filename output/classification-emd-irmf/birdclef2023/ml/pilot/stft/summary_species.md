# cnn_stft_pilot: birdclef2023 / species

2 fits on shared split IDs [0]; wall time 49.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                             |   mean |   std |
|:--------------------------------------------|-------:|------:|
| ('raw', 'comp', 'compact_resnet')           | 0.4946 |   nan |
| ('raw', 'comp+log_scale', 'compact_resnet') | 0.4801 |   nan |

Raw log-STFT reference: a 2-D residual CNN, 3x3 kernels, widths 16/32/64, GroupNorm and frequency/time mean/max pooling. All frames are 93 ms. Fixed STFT parameters do not use training, validation or test labels.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
