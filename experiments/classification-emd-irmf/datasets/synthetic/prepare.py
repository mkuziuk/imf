"""Synthetic multiscale classification set in the prepared format (end-to-end smoke test).

Five classes share one carrier band and differ in how the carrier is modulated:
tone, am, fm, chirp, transient (Gaussian wave packets). Every frame also carries a
random low-frequency background and Gaussian noise. A "recording" (the CV group)
fixes the carrier, gain, background level and noise parameters; its frames vary
phases, modulation details and noise draws. Recordings come in three noise
regimes, balanced across classes:

  gaussian        Gaussian noise only
  impulsive_one   plus one-sided exponential impulses (contamination as in
                  docs/imf_core.py generate_observation, without the random sign)
  impulsive_two   plus two-sided exponential impulses (generate_observation as is)

Usage (from the worktree root):
  .venv/bin/python experiments/classification-emd-irmf/datasets/synthetic/prepare.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "synthetic"
SEED = 20261020
FRAME_LENGTH = 1024
FS = 1024.0
CLASSES = ("tone", "am", "fm", "chirp", "transient")
REGIMES = ("gaussian", "impulsive_one", "impulsive_two")
RECORDINGS_PER_CELL = 10
FRAMES_PER_RECORDING = 10
CARRIER_RANGE = (0.03, 0.12)
GAIN_DECADES = (-1.0, 1.0)
BACKGROUND_RANGE = (0.5, 1.5)
SIGMA_RANGE = (0.6, 1.2)
CONTAMINATION_PROB_RANGE = (0.02, 0.08)
CONTAMINATION_SCALE_RANGE = (3.0, 8.0)
DEFAULT_DATA_ROOT = "/Users/mikhail/Datasets/imf-classification"


def unit_std(values):
    values = values - values.mean()
    return values / values.std()


def background(t, rng):
    """Random low-frequency baseline: harmonics 1-5 of the frame with 1/k amplitudes."""
    out = np.zeros_like(t)
    for harmonic in range(1, 6):
        out += rng.normal(scale=1.0 / harmonic) * np.sin(2 * np.pi * harmonic * t + rng.uniform(0, 2 * np.pi))
    return unit_std(out)


def class_component(label, n, carrier, rng):
    """Unit-std class signal on n samples; carrier in cycles per sample."""
    index = np.arange(n)
    t = index / n
    f0 = carrier * rng.uniform(0.97, 1.03)
    phase = rng.uniform(0, 2 * np.pi)
    if label == "tone":
        x = np.sin(2 * np.pi * f0 * index + phase)
    elif label == "am":
        depth = rng.uniform(0.5, 0.9)
        rate = rng.uniform(2.0, 8.0)
        envelope = 1 + depth * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
        x = envelope * np.sin(2 * np.pi * f0 * index + phase)
    elif label == "fm":
        rate = rng.uniform(2.0, 8.0)
        deviation = rng.uniform(0.2, 0.4) * f0
        beta = deviation * n / rate
        x = np.sin(2 * np.pi * f0 * index + beta * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi)) + phase)
    elif label == "chirp":
        span = rng.uniform(0.3, 0.6) * f0 * rng.choice([-1.0, 1.0])
        inst = f0 - span / 2 + span * t
        x = np.sin(2 * np.pi * np.cumsum(inst) + phase)
    elif label == "transient":
        x = np.zeros(n)
        for _ in range(rng.integers(3, 7)):
            center = rng.uniform(0.05, 0.95) * n
            width = rng.uniform(20.0, 60.0)
            x += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((index - center) / width) ** 2) * np.sin(
                2 * np.pi * f0 * index + rng.uniform(0, 2 * np.pi)
            )
    else:
        raise ValueError(label)
    return unit_std(x)


def contamination(n, prob, scale, two_sided, rng):
    """generate_observation's impulsive term: Bernoulli(prob) mask times Exp(scale)."""
    mask = rng.random(n) < prob
    impulses = mask * rng.exponential(scale=scale, size=n)
    if two_sided:
        impulses *= rng.choice([-1, 1], size=n)
    return impulses


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    n_recordings = len(CLASSES) * len(REGIMES) * RECORDINGS_PER_CELL
    frames = np.empty((n_recordings * FRAMES_PER_RECORDING, 1, FRAME_LENGTH), dtype=np.float32)
    rows = []
    recording = 0
    t = np.arange(FRAME_LENGTH) / FRAME_LENGTH
    for label in CLASSES:
        for regime in REGIMES:
            for _ in range(RECORDINGS_PER_CELL):
                carrier = float(np.exp(rng.uniform(*np.log(CARRIER_RANGE))))
                gain = float(10 ** rng.uniform(*GAIN_DECADES))
                level = float(rng.uniform(*BACKGROUND_RANGE))
                sigma = float(rng.uniform(*SIGMA_RANGE))
                prob = float(rng.uniform(*CONTAMINATION_PROB_RANGE)) if regime != "gaussian" else 0.0
                scale = float(rng.uniform(*CONTAMINATION_SCALE_RANGE)) if regime != "gaussian" else 0.0
                for position in range(FRAMES_PER_RECORDING):
                    frame_sigma = sigma * rng.uniform(0.9, 1.1)
                    clean = class_component(label, FRAME_LENGTH, carrier, rng) + level * background(t, rng)
                    noisy = clean + rng.normal(scale=frame_sigma, size=FRAME_LENGTH)
                    if regime != "gaussian":
                        noisy += contamination(FRAME_LENGTH, prob, scale, regime == "impulsive_two", rng)
                    frame_id = len(rows)
                    frames[frame_id, 0] = gain * noisy
                    rows.append({
                        "frame_id": frame_id,
                        "group": f"rec{recording:04d}",
                        "y_signal": label,
                        "noise_regime": regime,
                        "source_file": f"synthetic/rec{recording:04d}",
                        "start_sample": position * FRAME_LENGTH,
                        "carrier": carrier,
                        "gain": gain,
                        "background_level": level,
                        "sigma": frame_sigma,
                        "contamination_prob": prob,
                        "contamination_scale": scale,
                    })
                recording += 1
    return frames, pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default=os.environ.get("IMF_CLS_DATA", DEFAULT_DATA_ROOT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    out_dir = Path(args.data_root) / NAME / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, meta = generate(args.seed)
    np.save(out_dir / "frames.npy", frames)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    info = {
        "name": NAME,
        "fs": FS,
        "frame_length": FRAME_LENGTH,
        "channels": ["signal"],
        "tasks": {"signal": "y_signal"},
        "primary_task": "signal",
        "n_frames": int(len(meta)),
        "n_groups": int(meta["group"].nunique()),
        "class_counts": {"signal": {k: int(v) for k, v in meta["y_signal"].value_counts().sort_index().items()}},
        "preparation": {
            "seed": args.seed,
            "classes": list(CLASSES),
            "noise_regimes": list(REGIMES),
            "recordings_per_class_and_regime": RECORDINGS_PER_CELL,
            "frames_per_recording": FRAMES_PER_RECORDING,
            "carrier_cycles_per_sample": list(CARRIER_RANGE),
            "gain_log10_range": list(GAIN_DECADES),
            "background_level_range": list(BACKGROUND_RANGE),
            "gaussian_sigma_range": list(SIGMA_RANGE),
            "contamination_prob_range": list(CONTAMINATION_PROB_RANGE),
            "contamination_scale_range": list(CONTAMINATION_SCALE_RANGE),
            "units": "class component has unit std; sigma, background and impulse scales are relative to it",
        },
        "source_url": "generated by experiments/classification-emd-irmf/datasets/synthetic/prepare.py",
        "license": "CC0-1.0 (synthetic data)",
        "citation": "none (synthetic)",
        "notes": (
            "Smoke-test dataset. Gains are log-uniform over two decades independently of class, so "
            "log_scale should carry no class information. noise_regime is balanced across classes and "
            "recorded per frame for subgroup analyses."
        ),
    }
    with open(out_dir / "dataset.json", "w") as handle:
        json.dump(info, handle, indent=2)
    print(f"wrote {out_dir}: frames {frames.shape}, {info['n_groups']} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
