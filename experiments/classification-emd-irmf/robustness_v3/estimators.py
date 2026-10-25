"""Pre-specified H estimators and an immutable pilot decision."""
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.decompose import noise_sigma_hat
from robustness_v3.storage import digest, publish, save_json

PROVENANCE = Path(__file__).parent/'provenance'
DECISION = PROVENANCE/'h_decision.json'
PILOT = Path('/tmp/sigma_pilot/REPORT.md')


def sigma_iqr(y):
    d = np.diff(np.asarray(y, dtype=np.float64))
    q25, q75 = np.quantile(d, [0.25, 0.75], method='linear')
    sigma = float(q75 - q25) / (1.349 * math.sqrt(2))
    if sigma > 0:
        return sigma, 0
    sigma = float(np.mean(np.abs(d))) * math.sqrt(math.pi) / 2
    return (sigma, 1) if sigma > 0 else (1.0, 2)


def estimator(name):
    return {'E1': noise_sigma_hat, 'E2': sigma_iqr}[name]


def decide_cells(cells):
    """One real dataset x p row, four columns with paired E1/E2 criteria."""
    if len(cells) != 12 or cells[['dataset', 'p']].duplicated().any():
        raise ValueError('pilot must contain 12 distinct real dataset x p cells')
    expected = {(d, p) for d in ('dronerf', 'birdclef2023', 'esc50_animals') for p in (.1, .2, .3, .4)}
    if set(zip(cells.dataset, cells.p)) != expected:
        raise ValueError('wrong pilot cells')
    if not np.isfinite(cells[['l2_e1', 'l2_e2', 'scale_e1', 'scale_e2']]).all().all():
        raise ValueError('nonfinite pilot criteria')
    better = int((cells.l2_e2 < cells.l2_e1).sum())
    stable = int((cells.scale_e2 <= cells.scale_e1).sum())
    return dict(estimator='E2' if better >= 9 and stable >= 7 else 'E1',
                lower_l2_cells=better, no_worse_scale_cells=stable, total_cells=12,
                rule='c=3; lower median per-slot L2 in >=9/12; scale stability no worse in >=7/12')


PILOT_SHA256 = '9bb6f2c66f6da7a54090be2bf0c5cbe2aedf70f6adb12574a01eb86b07eb4e8b'
DECISION = PROVENANCE/'h_decision_v2.json'


def await_decision():
    """The dated amendment fixes E1 and dataset H before classification."""
    import json
    if DECISION.exists():
        value = json.loads(DECISION.read_text())
        if value['report_sha256'] != PILOT_SHA256 or value['estimator'] != 'E1':
            raise RuntimeError('amended H decision mismatch')
        return value
    if digest(PILOT) != PILOT_SHA256:
        raise RuntimeError('fixed-H pilot report SHA-256 mismatch')
    copy = PROVENANCE/'sigma_pilot_REPORT_v2.md'
    publish(copy, lambda f: f.write(PILOT.read_bytes()))
    record = dict(estimator='E1', report_sha256=PILOT_SHA256,
        lower_l2_cells=8, no_worse_scale_cells=4, total_cells=12,
        amendment_date='2026-10-22', mode='fixed dataset H in normalized units',
        calibration='2 * median E1 over all channels of 100 excluded clean normalized pilot frames',
        report_copy=copy.name, first_decision_sha256=digest(PROVENANCE/'h_decision.json'))
    save_json(DECISION, record)
    return record


def calibrate_h(frames, pilot_ids):
    from emdirmf.decompose import normalize_frame
    ids = np.asarray(pilot_ids, dtype=int)
    if len(ids) != 100 or len(np.unique(ids)) != 100:
        raise ValueError('fixed H requires exactly 100 distinct pilot frames')
    estimates = [noise_sigma_hat(normalize_frame(x)[0])[0]
                 for i in ids for x in frames[i]]
    return dict(H_d=float(2 * np.median(estimates)), n_frames=len(ids), n_channels=len(estimates),
                pilot_ids=ids.tolist(), normalized_clean_e1=estimates)
