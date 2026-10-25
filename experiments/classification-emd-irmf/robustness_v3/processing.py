"""Fresh comp features with float64 raw inputs; only pilot IDs retain full decompositions."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf import decompose as dec
from emdirmf.features import feature_columns, feature_names, frame_features
from emdirmf.io import frames_fingerprint
from emdirmf.paths import data_root
from robustness.environment import verify
from robustness.processing import channel_stack as shared_channel_stack
from robustness_v3.construction import (WINDOWS, channel_stack, fold_emd, n_slots,
                                        native_ratio, parse_variant, variants)
from robustness_v3.contamination import CONDITIONS, SCALE, clean_sigma, contaminate, subset_ids
from robustness_v3.estimators import DECISION, calibrate_h
from robustness_v3.storage import check_disk, digest, publish, root, save_json, save_npz

CHUNK = 16
MANIFEST = 'manifest_v2.json'
VERIFICATION = 'clean_verification_v2.json'
PROTOCOL = Path(__file__).with_name('PROTOCOL_v3.md')
_STATE = {}


def state(dataset, *, clean=False):
    if dataset not in _STATE:
        frames = np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
        decision = json.loads(DECISION.read_text())
        _STATE[dataset] = dict(frames=frames, variants=variants(frames.shape[-1]),
            fixed_h=json.loads((root(dataset)/MANIFEST).read_text())['fixed_h']['H_d'],
            subset=set(map(int, subset_ids(dataset, len(frames)))))
    s = _STATE[dataset]
    if clean and 'clean' not in s:
        s['clean'] = {v: np.load(root(dataset)/'matrices/clean'/f'{v}.npy', mmap_mode='r', allow_pickle=False)
                      for v in s['variants']}
        s['sd'] = {v: np.load(root(dataset)/'matrices/clean'/f'{v}__sd.npy', allow_pickle=False)
                   for v in s['variants']}
    return s


def extract_all(frame, dataset, frame_id, condition, fixed_h):
    """Run PyEMD once per channel, then fold; preserve float64 raw samples."""
    vs = variants(frame.shape[-1])
    stacks, hits, errors = {v: [] for v in vs}, {v: [] for v in vs}, {v: [] for v in vs}
    cpu, wall = {v: 0.0 for v in vs}, {v: 0.0 for v in vs}
    h_ratio, b_h, h_values, sigma_values, sigma_sources = [], [], [], [], []
    for channel, x in enumerate(frame):
        y, log_scale = dec.normalize_frame(contaminate(x, dataset, frame_id, channel, condition))
        sigma, source = dec.noise_sigma_hat(y)
        h_values.append(fixed_h)
        sigma_values.append(sigma)
        sigma_sources.append(source)
        h_ratio.append(sigma / fixed_h)
        b_h.append(SCALE * clean_sigma(x) / np.exp(log_scale) / fixed_h)
        t, c = time.perf_counter(), time.process_time()
        emd, _, emd_error = shared_channel_stack(y, 'emd')
        emd_wall, emd_cpu = time.perf_counter()-t, time.process_time()-c
        for v in vs:
            method, k = parse_variant(v)
            t, c = time.perf_counter(), time.process_time()
            hit, error = False, ''
            if method == 'raw':
                stack = y[None]
            elif method == 'emd':
                stack, error = (None if emd is None else fold_emd(emd, k)), emd_error
            else:
                stack, hit, error = channel_stack(y, method, k, fixed_h)
            if stack is None:
                stack = np.full((n_slots(v), len(x)), np.nan)
            stacks[v].append(np.asarray(stack, dtype=np.float64 if method == 'raw' else np.float32))
            hits[v].append(hit)
            errors[v].append(error)
            wall[v] += time.perf_counter()-t + (emd_wall if v == 'emd_k8' else 0)
            cpu[v] += time.process_time()-c + (emd_cpu if v == 'emd_k8' else 0)
    result = {}
    for v in vs:
        stack = np.stack(stacks[v])
        t, c = time.perf_counter(), time.process_time()
        values = frame_features(stack[None])[0].astype(np.float32)
        error = '; '.join(e for e in errors[v] if e)
        if not np.isfinite(values).all() and not error:
            error = 'nonfinite stack/features'
        result[v] = dict(values=values, stack=stack, cap=np.array(hits[v], dtype=bool), error=error,
                         seconds=wall[v]+time.perf_counter()-t, cpu=cpu[v]+time.process_time()-c)
    return result, dict(h_ratio=np.array(h_ratio), b_over_h=np.array(b_h), h=np.array(h_values),
        sigma_hat=np.array(sigma_values), sigma_source=np.array(sigma_sources, dtype=np.int8))


def verification_job(dataset, frame_id):
    frames = np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    rows = []
    # Verification needs only EMD/raw and can run before the pilot decision.
    for method in ('emd', 'raw'):
        stacks = []
        for x in frames[frame_id]:
            y, _ = dec.normalize_frame(x)
            stack, _, error = shared_channel_stack(y, method)
            if error:
                raise RuntimeError(error)
            stacks.append(stack if method == 'raw' else stack.astype(np.float32))
        values = frame_features(np.stack(stacks)[None])[0].astype(np.float32)
        table = pd.read_parquet(data_root()/dataset/'features'/f'{method}.parquet')
        if not np.array_equal(table.frame_id, np.arange(len(frames))) or not table.decomp_ok.all():
            raise ValueError('invalid clean reference')
        cached = table[feature_columns(table.columns)].iloc[frame_id].to_numpy(dtype=np.float32)
        tolerance = np.maximum(np.abs(np.spacing(cached)).astype(float), 1e-7)
        ok = bool(np.all(np.abs(values.astype(float)-cached) <= tolerance))
        rows.append(dict(dataset=dataset, frame_id=int(frame_id), method=method, ok=ok,
                         exact=bool(np.array_equal(values, cached)),
                         max_absolute_error=float(np.max(np.abs(values.astype(float)-cached)))))
    return rows


def chunk_path(dataset, condition, start):
    return root(dataset)/'features'/condition/f'part_{start:05d}.npz'


def chunk_jobs(dataset, clean=False):
    n = len(np.load(data_root()/dataset/'prepared/frames.npy', mmap_mode='r', allow_pickle=False))
    conditions = ('clean',) if clean else CONDITIONS[1:]
    return [(dataset, i, min(i+CHUNK, n)) for i in range(0, n, CHUNK)
            if any(not chunk_path(dataset, c, i).exists() for c in conditions)]


def chunk_job(dataset, start, stop, *, clean=False):
    # Contamination cannot begin before an immutable decision and passing verification.
    if not json.loads((root(dataset)/VERIFICATION).read_text())['ok']:
        raise RuntimeError('clean reproduction failed')
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    if manifest['h_decision'] != json.loads(DECISION.read_text()):
        raise RuntimeError('H decision mismatch')
    s = state(dataset, clean=not clean)
    vs, frames = s['variants'], s['frames']
    conditions = ('clean',) if clean else CONDITIONS[1:]
    outputs = {c: dict(frame_id=np.arange(start, stop)) for c in conditions}
    retained = {c: {v: [] for v in vs} for c in conditions}
    figure_ids = [i for i in range(start, stop) if i in s['subset']]
    for c in conditions:
        for v in vs:
            outputs[c].update({v+'__'+key: [] for key in
                ('values', 'ok', 'errors', 'cap', 'drift', 'l2', 'cpu', 'seconds', 'h_ratio', 'b_over_h', 'h', 'sigma_hat', 'sigma_source')})
    begun = time.perf_counter()
    for frame_id in range(start, stop):
        reference, clean_diagnostic = extract_all(frames[frame_id], dataset, frame_id, 'clean', s['fixed_h'])
        if any(r['error'] for r in reference.values()):
            raise RuntimeError(f'{dataset}/{frame_id}: failed clean decomposition')
        for condition in conditions:
            current, diagnostic = ((reference, clean_diagnostic) if clean else
                extract_all(frames[frame_id], dataset, frame_id, condition, s['fixed_h']))
            for v, r in current.items():
                difference = r['stack'].astype(float)-reference[v]['stack'].astype(float)
                norms = np.maximum(np.linalg.norm(reference[v]['stack'].astype(float), axis=-1), 1e-12)
                l2 = np.linalg.norm(difference, axis=-1)/norms
                whole = np.linalg.norm(difference.reshape(len(difference), -1), axis=-1)/np.maximum(
                    np.linalg.norm(reference[v]['stack'].astype(float).reshape(len(difference), -1), axis=-1), 1e-12)
                drift = 0.0 if clean else float(np.mean(np.abs((r['values'].astype(float)-s['clean'][v][frame_id])/s['sd'][v])))
                row = dict(values=r['values'], ok=not bool(r['error']), errors=r['error'], cap=r['cap'],
                    drift=drift, l2=np.column_stack([l2, whole]), cpu=r['cpu'], seconds=r['seconds'], **diagnostic)
                for key, value in row.items():
                    outputs[condition][v+'__'+key].append(value)
                if frame_id in s['subset']:
                    retained[condition][v].append(r['stack'])
    for condition, data in outputs.items():
        for key in data:
            data[key] = np.asarray(data[key], dtype='U512' if key.endswith('__errors') else None)
        data['chunk_wall_seconds'] = np.array(time.perf_counter()-begun)
        for v in vs:
            if figure_ids:
                path = root(dataset)/'figures'/condition/v/f'part_{start:05d}.npz'
                if not path.exists():
                    save_npz(path, frame_id=np.array(figure_ids), stacks=np.stack(retained[condition][v]))
        path = chunk_path(dataset, condition, start)
        if not path.exists():
            save_npz(path, **data)
    check_disk()
    return dict(dataset=dataset, start=start, clean=clean)


def condition_matrices(dataset, condition):
    s = state(dataset)
    n, channels = s['frames'].shape[:2]
    values = {v: np.empty((n, channels*n_slots(v)*20), dtype=np.float32) for v in s['variants']}
    valid, seen = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    for path in sorted((root(dataset)/'features'/condition).glob('part_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            ids = z['frame_id']
            if np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids) or seen[ids].any():
                raise AssertionError('invalid or duplicate frame chunk')
            seen[ids] = True
            for v in s['variants']:
                values[v][ids] = z[v+'__values']
                valid[ids] &= z[v+'__ok'] & np.isfinite(z[v+'__values']).all(axis=1)
    if not seen.all():
        raise RuntimeError(f'{dataset}: incomplete {condition}')
    return values, valid


def materialize_condition(dataset, condition):
    values, valid = condition_matrices(dataset, condition)
    columns = {}
    for v, a in values.items():
        path = root(dataset)/'matrices'/condition/f'{v}.npy'
        if not path.exists():
            publish(path, lambda f, a=a: np.save(f, a, allow_pickle=False))
        columns[v] = feature_names(state(dataset)['frames'].shape[1], n_slots(v))
        if condition == 'clean':
            sd = a.astype(float).std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            path = path.with_name(v+'__sd.npy')
            if not path.exists():
                publish(path, lambda f, a=sd: np.save(f, a, allow_pickle=False))
    path = root(dataset)/'matrices'/condition/'valid.npy'
    if not path.exists():
        publish(path, lambda f: np.save(f, valid, allow_pickle=False))
    return columns


def write_manifest(dataset):
    pimf = verify()
    base, out, study = data_root()/dataset, root(dataset), Path(__file__).parents[1]
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    task = json.loads((base/'prepared/dataset.json').read_text())['primary_task']
    source = sorted(Path(__file__).parent.glob('*.py')) + [PROTOCOL]
    source += [study/'robustness'/n for n in ('contamination.py', 'processing.py', 'environment.py', 'storage.py', 'classification.py')]
    source += [study/'emdirmf'/n for n in ('decompose.py', 'features.py', 'irmf_newton.py', 'parallel.py', 'io.py', 'cv.py', 'stats.py', 'paths.py')]
    inputs = [base/'prepared/meta.parquet', base/'prepared/dataset.json', base/f'splits_{task}.json',
              base/'features/emd.parquet', base/'features/raw.parquet']
    decision = json.loads(DECISION.read_text())
    for k in WINDOWS[frames.shape[-1]]:
        native_ratio(frames.shape[-1], k)
    manifest = dict(version=2, experiment_version=3, dataset=dataset, shape=list(frames.shape), chunk=CHUNK,
        conditions=list(CONDITIONS), scale=SCALE, methods=list(variants(frames.shape[-1])),
        verification_file=VERIFICATION, raw_stack_dtype='float64', decomposition_stack_dtype='float32',
        fixed_h=calibrate_h(frames, subset_ids(dataset, len(frames))),
        pilot_report_sha256=decision['report_sha256'], h_decision=decision, h_decision_sha256=digest(DECISION), pimf=pimf,
        windows={str(k): list(w) for k, w in WINDOWS[frames.shape[-1]].items()},
        frames_fingerprint=frames_fingerprint(frames), subset_ids=subset_ids(dataset, len(frames)).tolist(),
        sources={str(p.relative_to(study)): digest(p) for p in source},
        inputs={str(p.relative_to(base)): digest(p) for p in inputs})
    source += [DECISION, DECISION.parent/decision['report_copy'], DECISION.parent/'h_decision.json']
    manifest['sources'].update({str(p.relative_to(study)): digest(p) for p in source})
    path = out/MANIFEST
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f'{dataset}: feature manifest mismatch')
    else:
        save_json(path, manifest)
    return manifest
