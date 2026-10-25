"""Laplace-contaminated features through the impulse run's unchanged extraction path.

Workers serve every dataset. Each loads a dataset's frames, clean features and
clean stacks the first time a job needs them.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf import decompose as dec
from emdirmf.features import feature_columns, frame_features
from emdirmf.io import frames_fingerprint
from emdirmf.paths import data_root
from robustness.environment import verify
from robustness.pilot import METHODS
from robustness.processing import channel_stack
from robustness_laplace.contamination import CONDITIONS, SCALE, contaminate, subset_ids
from robustness_laplace.storage import digest, root, save_json, save_npz

CHUNK = 32
MANIFEST = 'manifest.json'
VERIFICATION = 'clean_verification.json'
PROTOCOL = Path(__file__).with_name('PROTOCOL_laplace.md')
_STATE = {}


def state(dataset):
    if dataset not in _STATE:
        base = data_root()/dataset
        frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
        clean, sd, stacks = {}, {}, {}
        for method in METHODS:
            table = pd.read_parquet(base/'features'/f'{method}.parquet')
            if not np.array_equal(table.frame_id, np.arange(len(frames))) or not table.decomp_ok.all():
                raise ValueError(f'{dataset}/{method}: invalid clean features')
            values = table[feature_columns(table.columns, include_log_scale=True)].to_numpy(dtype=np.float64)
            spread = values.std(axis=0)
            sd[method] = np.where(spread > 0, spread, 1.0)
            clean[method] = values
            if method != 'raw':
                stacks[method] = np.load(base/'decomp'/f'{method}.npy', mmap_mode='r', allow_pickle=False)
        _STATE[dataset] = dict(frames=frames, clean=clean, sd=sd, stacks=stacks,
                               subset=set(map(int, subset_ids(dataset, len(frames)))))
    return _STATE[dataset]


def extract(frame, dataset, frame_id, condition, method):
    """Same as robustness.processing.extract, with Laplace contamination."""
    normalized, log_scale, stacks, hits, errors = [], [], [], [], []
    for channel, x in enumerate(frame):
        y, scale = dec.normalize_frame(contaminate(x, dataset, frame_id, channel, condition))
        stack, hit, error = channel_stack(y, method)
        hits.append(hit)
        errors.append(error)
        if stack is None:
            stack = np.full((dec.n_slots(method, len(x)), len(x)), np.nan)
        stacks.append(stack if method == 'raw' else stack.astype(np.float32))
        normalized.append(y)
        log_scale.append(float(scale))
    stack = np.stack(stacks)
    features = frame_features(stack[None])[0].astype(np.float32)
    values = np.concatenate([features.astype(np.float64), log_scale])
    if not np.isfinite(values).all() and not any(errors):
        errors[0] = 'nonfinite stack/features'
    return values, stack, np.array(hits, dtype=bool), '; '.join(e for e in errors if e)


def verification_job(dataset, frame_id):
    s = state(dataset)
    rows = []
    for method in METHODS:
        values, _, hits, error = extract(s['frames'][frame_id], dataset, frame_id, 'clean', method)
        cached = s['clean'][method][frame_id]
        n_comp = len(values) - s['frames'].shape[1]
        tolerance = np.maximum(np.abs(np.spacing(cached[:n_comp].astype(np.float32))).astype(float), 1e-7)
        ok = (not error and np.all(np.abs(values[:n_comp]-cached[:n_comp]) <= tolerance)
              and np.allclose(values[n_comp:], cached[n_comp:], atol=1e-14, rtol=1e-14))
        rows.append(dict(dataset=dataset, frame_id=int(frame_id), method=method, ok=bool(ok),
                         exact=bool(np.array_equal(values, cached)),
                         max_absolute_error=float(np.max(np.abs(values-cached))), cap_hits=int(hits.sum())))
    return rows


def chunk_path(dataset, condition, start):
    return root(dataset)/'features'/condition/f'part_{start:05d}.npz'


def chunk_job(dataset, condition, start, stop):
    s = state(dataset)
    frames = s['frames']
    n_channels, length = frames.shape[1:]
    n = stop-start
    data = dict(frame_id=np.arange(start, stop), h_ratio=np.empty((n, n_channels)))
    figure_ids, figure_stacks = [], {m: [] for m in METHODS}
    wall = time.perf_counter()
    for method in METHODS:
        data[method+'__values'] = np.empty((n, s['clean'][method].shape[1]))
        data[method+'__ok'] = np.ones(n, dtype=bool)
        data[method+'__drift'] = np.empty((n, 2))
        data[method+'__l2'] = np.empty((n, n_channels, dec.n_slots(method, length)+1))
        data[method+'__cap'] = np.zeros((n, n_channels), dtype=bool)
        data[method+'__cpu'] = np.zeros(n)
        data[method+'__seconds'] = np.zeros(n)
        data[method+'__errors'] = np.full(n, '', dtype='U512')
    for local, frame_id in enumerate(range(start, stop)):
        for channel, x in enumerate(frames[frame_id]):
            yc, _ = dec.normalize_frame(x)
            yn, _ = dec.normalize_frame(contaminate(x, dataset, frame_id, channel, condition))
            data['h_ratio'][local, channel] = dec.noise_sigma_hat(yn)[0] / dec.noise_sigma_hat(yc)[0]
        figure = frame_id in s['subset']
        if figure:
            figure_ids.append(frame_id)
        for method in METHODS:
            cpu, started = time.process_time(), time.perf_counter()
            values, stack, cap, error = extract(frames[frame_id], dataset, frame_id, condition, method)
            clean = s['clean'][method][frame_id]
            clean_stack = (dec.normalize_frame(frames[frame_id])[0][:, None, :]
                           if method == 'raw' else s['stacks'][method][frame_id])
            clean_stack = np.asarray(clean_stack, dtype=np.float64)
            difference = np.asarray(stack, dtype=np.float64) - clean_stack
            slot_l2 = np.linalg.norm(difference, axis=-1) / np.maximum(np.linalg.norm(clean_stack, axis=-1), 1e-12)
            whole = np.linalg.norm(difference.reshape(n_channels, -1), axis=-1) / np.maximum(
                np.linalg.norm(clean_stack.reshape(n_channels, -1), axis=-1), 1e-12)
            drift = np.abs((values-clean)/s['sd'][method])
            n_comp = len(values)-n_channels
            data[method+'__values'][local] = values
            data[method+'__ok'][local] = not bool(error)
            data[method+'__errors'][local] = error
            data[method+'__cap'][local] = cap
            data[method+'__drift'][local] = (np.mean(drift[:n_comp]), np.mean(drift))
            data[method+'__l2'][local] = np.column_stack([slot_l2, whole])
            data[method+'__cpu'][local] = time.process_time()-cpu
            data[method+'__seconds'][local] = time.perf_counter()-started
            if figure:
                figure_stacks[method].append(stack)
    if figure_ids:
        for method in METHODS:
            path = root(dataset)/'figures'/condition/method/f'part_{start:05d}.npz'
            if not path.exists():
                save_npz(path, frame_id=np.array(figure_ids), stacks=np.stack(figure_stacks[method]).astype(np.float32))
    data['chunk_wall_seconds'] = np.array(time.perf_counter()-wall)
    save_npz(chunk_path(dataset, condition, start), **data)
    return dict(dataset=dataset, condition=condition, start=start)


def feature_jobs(dataset):
    n = len(np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False))
    return [(dataset, c, i, min(i+CHUNK, n)) for c in CONDITIONS[1:] for i in range(0, n, CHUNK)
            if not chunk_path(dataset, c, i).exists()]


def write_manifest(dataset):
    """Record parameters and source/input hashes; refuse to mix incompatible runs."""
    pimf = verify()
    base, out = data_root()/dataset, root(dataset)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    task = json.loads((base/'prepared/dataset.json').read_text())['primary_task']
    study = Path(__file__).parents[1]
    source = [Path(__file__), Path(__file__).with_name('contamination.py'), PROTOCOL,
              study/'robustness/contamination.py', study/'robustness/processing.py',
              study/'robustness/environment.py']
    source += [study/'emdirmf'/n for n in ('decompose.py', 'features.py', 'irmf_newton.py', 'parallel.py')]
    inputs = [base/'prepared/meta.parquet', base/'prepared/dataset.json', base/f'splits_{task}.json']
    inputs += [base/'features'/f'{m}.parquet' for m in METHODS]
    inputs += [base/'decomp'/f'{m}.json' for m in METHODS if m != 'raw']
    manifest = dict(version=1, dataset=dataset, shape=list(frames.shape), chunk=CHUNK,
                    conditions=list(CONDITIONS), scale=SCALE, methods=list(METHODS), pimf=pimf,
                    frames_fingerprint=frames_fingerprint(frames),
                    subset_ids=subset_ids(dataset, len(frames)).tolist(),
                    sources={str(p.relative_to(study)): digest(p) for p in source},
                    inputs={str(p.relative_to(base)): digest(p) for p in inputs})
    path = out/MANIFEST
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f'{dataset}: feature manifest mismatch; refusing to mix outputs')
    else:
        save_json(path, manifest)
    return manifest


def matrices(dataset):
    """Load numeric feature chunks, keeping the original frame indexing."""
    base, out = data_root()/dataset, root(dataset)
    clean, columns = {}, {}
    for method in METHODS:
        table = pd.read_parquet(base/'features'/f'{method}.parquet')
        columns[method] = feature_columns(table.columns, include_log_scale=True)
        clean[method] = table[columns[method]].to_numpy(dtype=np.float64)
    result = {'clean': clean}
    valid = {'clean': np.ones(len(next(iter(clean.values()))), dtype=bool)}
    for condition in CONDITIONS[1:]:
        result[condition] = {m: np.empty_like(a) for m, a in clean.items()}
        valid[condition] = np.ones_like(valid['clean'])
        seen = np.zeros_like(valid['clean'])
        for path in sorted((out/'features'/condition).glob('part_*.npz')):
            with np.load(path, allow_pickle=False) as z:
                ids = z['frame_id']
                if seen[ids].any():
                    raise AssertionError('duplicate frame chunk')
                seen[ids] = True
                for method in METHODS:
                    result[condition][method][ids] = z[method+'__values']
                    valid[condition][ids] &= z[method+'__ok']
        if not seen.all():
            raise RuntimeError(f'{dataset}: incomplete {condition} features')
    return result, valid, columns
