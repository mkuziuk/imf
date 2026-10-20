# uas_phase3_pilot_cnn: uas_birds_60ghz / bird_drone

4 fits on shared split IDs [0]; wall time 16.0 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                       |   mean |   std |
|:------------------------------------------------------|-------:|------:|
| ('irmf_h2_a1.41', 'comp', 'compact_resnet')           | 0.9766 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'compact_resnet') | 0.9688 |   nan |
| ('raw', 'comp', 'compact_resnet')                     | 0.9844 |   nan |
| ('raw', 'comp+log_scale', 'compact_resnet')           | 1.0000 |   nan |

Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods apart from the input channel count; optional standardized log_scale joins the pooled head.
Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on all outer training samples for that many epochs. Test samples never select epochs or scale statistics.
Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may vary across devices or library versions. Training curves include selection and refit stages.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
