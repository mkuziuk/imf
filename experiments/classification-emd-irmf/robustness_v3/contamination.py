"""Notebook-scale contamination with the earlier experiments' common draws."""
import numpy as np

from emdirmf.decompose import noise_sigma_hat
from robustness.contamination import DATASETS, draws, subset_ids

LEVELS = (0.1, 0.2, 0.3, 0.4)
CONDITIONS = ('clean',) + tuple(f'nb3_{p:g}' for p in LEVELS)
MATCHED = ('nb3_0.4',)
SCALE = 3.0


def condition_level(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return 0.0 if condition == 'clean' else float(condition.rsplit('_', 1)[1])


def clean_sigma(x):
    sigma, source = noise_sigma_hat(x)
    return 0.0 if source == 2 else sigma


def contaminate(x, dataset, frame_id, channel, condition):
    x = np.asarray(x, dtype=np.float64)
    p = condition_level(condition)
    if not p:
        return x.copy()
    u, e, s, _ = draws(dataset, frame_id, channel, x.size)
    return x + (u < p) * (SCALE * clean_sigma(x)) * e * s
