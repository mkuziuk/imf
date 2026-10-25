"""Laplace noise on a random fraction of samples, before normalization."""
import numpy as np

from robustness.contamination import DATASETS, clean_sigma, draws, subset_ids

LEVELS = (0.1, 0.2, 0.3, 0.4)
CONDITIONS = ('clean',) + tuple(f'laplace_{p:g}' for p in LEVELS)
MATCHED = ('laplace_0.4',)
SCALE = 1.0

__all__ = ['CONDITIONS', 'DATASETS', 'LEVELS', 'MATCHED', 'SCALE', 'clean_sigma',
           'condition_level', 'contaminate', 'subset_ids']


def condition_level(condition):
    return 0.0 if condition == 'clean' else float(condition.rsplit('_', 1)[1])


def contaminate(x, dataset, frame_id, channel, condition):
    """x + (u < p) * SCALE * sigma * e * s, the impulse run's u, e, s draws."""
    x = np.asarray(x, dtype=np.float64)
    if condition == 'clean':
        return x.copy()
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition {condition}')
    u, e, s, _ = draws(dataset, frame_id, channel, x.size)
    return x + (u < condition_level(condition)) * (SCALE * clean_sigma(x)) * e * s
