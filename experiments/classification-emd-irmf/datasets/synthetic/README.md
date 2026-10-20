# synthetic

Generated smoke-test dataset for the shared pipeline: prepare, validate,
decompose, features, classical baselines.

- Source: `prepare.py` in this folder (no download). Seed 20261020.
- License: CC0-1.0. Citation: none.
- Format: 1,500 frames, 1 channel, L = 1024, nominal fs = 1024 Hz.
- Task `signal` (`y_signal`), 5 balanced classes: `tone`, `am`, `fm`, `chirp`,
  `transient`. All classes share a carrier band of 0.03-0.12 cycles per sample and
  differ only in how the carrier is modulated (amplitude, frequency, linear sweep,
  or Gaussian wave packets).
- Each frame adds a random low-frequency background (harmonics 1-5 of the frame)
  and Gaussian noise with sigma 0.6-1.2 times the unit-std class signal.
- Groups: 150 generating "recordings" of 10 frames. A recording fixes the carrier,
  gain, background level and noise parameters. Its frames redraw phases,
  modulation details and noise.
- Noise regimes (`noise_regime` in meta, 10 recordings per class and regime):
  `gaussian`; `impulsive_one`, with one-sided exponential impulses (probability
  0.02-0.08, scale 3-8); `impulsive_two`, with the same impulses given random
  signs. The impulses follow `generate_observation` in `docs/imf_core.py`.
- Gain is log-uniform over two decades and independent of class, so `log_scale`
  should carry no class information.

Run `.venv/bin/python experiments/classification-emd-irmf/datasets/synthetic/prepare.py`
from the worktree root. Results go to `output/classification-emd-irmf/synthetic/`.
