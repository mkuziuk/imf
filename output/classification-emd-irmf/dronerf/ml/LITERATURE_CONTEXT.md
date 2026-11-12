# DroneRF literature context

The present primary task is 10-class recognition of activity-gated 2,048-sample
frames, with segment-held-out repeated CV. L and H are nonsimultaneous views.
Probability pooling uses all selected frames from each held-out segment.
Training and test still share the BUI recording session.

[Lin et al., 2026](https://www.hanspub.org/journal/paperinformation?paperid=150367),
DOI 10.12677/csa.2026.169296, report a Raw/FFT/STFT three-branch network with
227 paired segments split 160/34/33 before window generation. Their abstract
specifies 32 windows per segment, each 2 x 8,192 samples. The combined training
configuration obtains 99.26% mean window balanced accuracy over three seeds,
versus 99.01% for cross entropy alone. All four configurations correctly
classify the 33 held-out segments after probability aggregation. This is the
binary Background/UAV task. It does not establish 99% ten-class recognition,
and its input, fixed holdout and training differ from this study. This source
does not verify the separate 2.5-ms-STFT reference mentioned in the task brief.

[Shulman, 2026](https://arxiv.org/html/2607.01025v1) reports AR-versus-Bebop
macro-F1 0.742 with random segment evaluation and 0.455 when holding out
recordings. The task and grouping differ from ours. It supports keeping
session generalization separate from performance on unseen segments of
already represented sessions.

These papers provide context rather than directly comparable score ceilings.
Source pages were checked on 2026-10-21 local time. No literature result
selected a method, fold or model budget in the present experiment.
