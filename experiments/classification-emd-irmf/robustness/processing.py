"""Stream contaminated channel stacks into identical shared float32 features."""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf import decompose as dec
from emdirmf.features import feature_columns, frame_features
from emdirmf.io import frames_fingerprint
from emdirmf.irmf_newton import newton_components, warm_up
from emdirmf.parallel import limit_threads, single_thread_env
from emdirmf.paths import data_root
from robustness.contamination import CONDITIONS, contaminate, subset_ids
from robustness.environment import verify
from robustness.pilot import METHODS
from robustness.storage import check_disk, digest, root, save_json, save_npz

CHUNK = 32
MANIFEST = "manifest_v2.json"
VERIFICATION = "clean_verification_v2.json"
_STATE = {}


def initialize(dataset):
    limit_threads()
    verify()
    warm_up()
    base = data_root()/dataset
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    clean, stats, stacks = {}, {}, {}
    for method in METHODS:
        table = pd.read_parquet(base/'features'/f'{method}.parquet')
        if not np.array_equal(table.frame_id, np.arange(len(frames))) or not table.decomp_ok.all():
            raise ValueError(f'{dataset}/{method}: invalid clean features')
        columns = feature_columns(table.columns, include_log_scale=True)
        values = table[columns].to_numpy(dtype=np.float64)
        sd = values.std(axis=0)
        stats[method] = np.where(sd > 0, sd, 1.0)
        clean[method] = values
        if method != 'raw':
            stacks[method] = np.load(base/'decomp'/f'{method}.npy', mmap_mode='r', allow_pickle=False)
    _STATE.clear()
    _STATE.update(dataset=dataset, frames=frames, clean=clean, sd=stats, stacks=stacks,
                  subset=set(map(int, subset_ids(dataset, len(frames)))))


def channel_stack(y, method):
    """Return stack, cap-hit flag, failure text; a failed stack is never accepted."""
    if method != METHODS[-1]:
        stack, _ = dec.decompose_signal(y, method)
        return stack, False, ''
    hit = False
    try:
        stack, info = newton_components(y)
        hit = max(info['stage_iterations']) >= 100
        if not hit:
            return stack, False, ''
    except RuntimeError as error:
        if 'Newton failed to converge' not in str(error):
            raise
        hit = True
    try:
        stack, info = newton_components(y, root_max_iter=1000)
        if not all(info['stage_converged']):
            raise RuntimeError('Newton failed to converge at cap 1000')
        return stack, hit, ''
    except RuntimeError as error:
        if 'Newton failed to converge' not in str(error):
            raise
        return None, hit, str(error)


def extract(frame, dataset, frame_id, condition, method):
    normalized = []
    log_scale = []
    stacks = []
    hits = []
    errors = []
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


def verification_job(frame_id):
    s = _STATE
    rows = []
    for method in METHODS:
        values, _, hits, error = extract(s['frames'][frame_id], s['dataset'], frame_id, 'clean', method)
        cached = s['clean'][method][frame_id]
        n_comp = len(values) - s['frames'].shape[1]
        # Compare component values in units of float32 representable rounding.
        tolerance = np.maximum(np.abs(np.spacing(cached[:n_comp].astype(np.float32))).astype(float), 1e-7)
        ok = (not error and np.all(np.abs(values[:n_comp]-cached[:n_comp]) <= tolerance)
              and np.allclose(values[n_comp:], cached[n_comp:], atol=1e-14, rtol=1e-14))
        rows.append(dict(frame_id=int(frame_id), method=method, ok=bool(ok),
                         exact=bool(np.array_equal(values, cached)),
                         max_absolute_error=float(np.max(np.abs(values-cached))), cap_hits=int(hits.sum())))
    return rows


def chunk_job(args):
    condition, start, stop = args
    s = _STATE
    frames = s['frames']
    n_channels, length = frames.shape[1:]
    n = stop-start
    data = dict(frame_id=np.arange(start, stop), h_ratio=np.empty((n, n_channels)))
    figure_ids = []
    figure_stacks = {m: [] for m in METHODS}
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
            yn, _ = dec.normalize_frame(contaminate(x, s['dataset'], frame_id, channel, condition))
            data['h_ratio'][local, channel] = dec.noise_sigma_hat(yn)[0] / dec.noise_sigma_hat(yc)[0]
        figure = frame_id in s['subset']
        if figure:
            figure_ids.append(frame_id)
        for method in METHODS:
            cpu, started = time.process_time(), time.perf_counter()
            values, stack, cap, error = extract(frames[frame_id], s['dataset'], frame_id, condition, method)
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
            path = root(s['dataset'])/'figures'/condition/method/f'part_{start:05d}.npz'
            if not path.exists():
                save_npz(path, frame_id=np.array(figure_ids), stacks=np.stack(figure_stacks[method]).astype(np.float32))
    data['chunk_wall_seconds'] = np.array(time.perf_counter()-wall)
    save_npz(root(s['dataset'])/'features'/condition/f'part_{start:05d}.npz', **data)
    return condition, start


def prepare(dataset, workers=6):
    if not 1 <= workers <= 6:
        raise ValueError('workers must be 1..6')
    expected_pimf = verify()
    base, out = data_root()/dataset, root(dataset)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    task = json.loads((base/'prepared/dataset.json').read_text())['primary_task']
    source = [Path(__file__), Path(__file__).with_name('contamination.py'),
              Path(__file__).with_name('environment.py')]
    source += [Path(__file__).parents[1]/'emdirmf'/n for n in
               ('decompose.py','features.py','irmf_newton.py','parallel.py')]
    inputs = [base/'prepared/meta.parquet', base/'prepared/dataset.json', base/f'splits_{task}.json']
    inputs += [base/'features'/f'{m}.parquet' for m in METHODS]
    inputs += [base/'decomp'/f'{m}.json' for m in METHODS if m != 'raw']
    manifest = dict(version=1, dataset=dataset, shape=list(frames.shape), chunk=CHUNK,
                    conditions=list(CONDITIONS), methods=list(METHODS), pimf=expected_pimf,
                    frames_fingerprint=frames_fingerprint(frames),
                    subset_ids=subset_ids(dataset, len(frames)).tolist(),
                    sources={str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in source},
                    inputs={str(p.relative_to(base)): digest(p) for p in inputs})
    path = out/MANIFEST
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Feature manifest mismatch; refusing to replace outputs')
    else:
        save_json(path, manifest)
    single_thread_env()
    limit_threads()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(dataset,)) as pool:
        verification = out/VERIFICATION
        if not verification.exists():
            ids = np.linspace(0, len(frames)-1, 50, dtype=int)
            rows = [r for block in pool.map(verification_job, map(int, ids)) for r in block]
            save_json(verification, dict(rows=rows, ok=all(r['ok'] for r in rows)))
        if not json.loads(verification.read_text())['ok']:
            # Recalculation is specified, but it must be implemented explicitly, never
            # mix a failed verification with old clean feature definitions.
            raise RuntimeError('Clean features do not reproduce; clean recomputation required before proceeding')
        print(f'{dataset}: clean verification passed', flush=True)
        for condition in CONDITIONS[1:]:
            jobs = [(condition, i, min(i+CHUNK, len(frames))) for i in range(0, len(frames), CHUNK)
                    if not (out/'features'/condition/f'part_{i:05d}.npz').exists()]
            count = 0
            for _, _ in pool.map(chunk_job, jobs):
                count += 1
                if count % 50 == 0:
                    check_disk()
                    print(f'{dataset} {condition}: {count}/{len(jobs)} chunks', flush=True)
            print(f'{dataset} {condition}: complete', flush=True)
    verify()
    run_path = out/'runs/features.json'
    if not run_path.exists():
        save_json(run_path, dict(wall_seconds=time.perf_counter()-started, workers=workers,
                               disk_bytes=check_disk(), pimf=expected_pimf))
    return manifest


def matrices(dataset):
    """Load numeric feature chunks, retaining original frame indexing."""
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
            raise RuntimeError(f'incomplete {condition} features')
    return result, valid, columns
