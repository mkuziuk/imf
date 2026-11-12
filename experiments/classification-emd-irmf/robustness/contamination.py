"""Common random numbers for additive contamination before normalization."""
import numpy as np

SEED = 20261021
DATASETS = ('synthetic', 'dronerf', 'birdclef2023', 'esc50_animals')
LEVELS = (0.002, 0.01, 0.04)
KINDS = ('impulse1', 'impulse2', 'gauss')
CONDITIONS = ('clean',) + tuple(f'{kind}_{p:g}' for kind in KINDS for p in LEVELS)
AMPLITUDE = 10.0


def draws(dataset, frame_id, channel, length):
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, DATASETS.index(dataset), int(frame_id), int(channel)]))
    return (rng.random(length), rng.exponential(1.0, length),
            rng.choice([-1, 1], length), rng.normal(0.0, 1.0, length))


def clean_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(sigma if sigma > 0 else np.std(x))


def contaminate(x, dataset, frame_id, channel, condition):
    x = np.asarray(x, dtype=np.float64)
    if condition == 'clean':
        return x.copy()
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition}')
    kind, level = condition.rsplit('_', 1)
    p = float(level)
    u, e, s, z = draws(dataset, frame_id, channel, x.size)
    scale = AMPLITUDE * clean_sigma(x)
    noise = scale * np.sqrt(2 * p) * z if kind == 'gauss' else (u < p) * scale * e
    if kind == 'impulse2':
        noise = noise * s
    return x + noise


def subset_ids(dataset, n_frames, count=100):
    """Fixed pilot/figure frames; excluded from every classification partition."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, DATASETS.index(dataset), 991]))
    return np.sort(rng.choice(n_frames, min(count, n_frames), replace=False))
