"""Validate the completed run, then write the report artifacts.

    $PY -m robustness_laplace.analyse [--out DIR]
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import frames_fingerprint, label_column, load_info, load_meta
from emdirmf.paths import OUTPUT_ROOT, data_root
from robustness.environment import START, verify
from robustness_laplace.analysis import DATASETS, FAMILY_SIZES, METRICS, load_scores, paired_tests
from robustness_laplace.classification import checkpoint, impulse_checkpoint
from robustness_laplace.processing import MANIFEST, VERIFICATION
from robustness_laplace.reporting import aggregate_representations, figures, readings, render_report
from robustness_laplace.run import runs_dir
from robustness_laplace.storage import digest, root, save_json

STUDY = Path(__file__).parents[1]


def validate_predictions(dataset):
    meta, info = load_meta(dataset), load_info(dataset)
    y = meta[label_column(info)].to_numpy(dtype=str)
    labels = sorted(set(y))
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    excluded = set(manifest['subset_ids'])
    splits = {s.split_id: s for s in load_splits(dataset)}
    valid = {c: np.load(root(dataset)/'matrices'/c/'valid.npy', allow_pickle=False) for c in manifest['conditions']}
    keep = np.ones(len(y), dtype=bool)
    keep[list(excluded)] = False
    n_predictions = 0
    for path in sorted((root(dataset)/'classification').rglob('complete.json')):
        value = json.loads(path.read_text())
        with np.load(path.with_name('predictions.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for row in value['rows']:
            c, sid = row['condition'], row['split_id']
            ids, pred = arrays[c+'__ids'], arrays[c+'__pred']
            if row['scenario'] == 'shift':
                clean_ids, clean_pred = arrays['clean__ids'], arrays['clean__pred']
            else:
                shift = checkpoint(dataset, 'shift', row['classifier'], row['method'], row['feature_set'], sid)
                with np.load(shift/'predictions.npz', allow_pickle=False) as base:
                    clean_ids, clean_pred = base['clean__ids'], base['clean__pred']
            baseline = dict(zip(clean_ids.tolist(), clean_pred.tolist()))
            check_row(row, y, labels, ids, pred, baseline)
            if set(map(int, ids)) & excluded or not np.isin(ids, splits[sid].test).all():
                raise AssertionError('pilot inclusion or test-fold mismatch')
            np.testing.assert_array_equal(ids, splits[sid].test[(keep & valid[c])[splits[sid].test]])
            train_condition = 'clean' if row['scenario'] == 'shift' else c
            if row['n_train'] != int((keep & valid[train_condition])[splits[sid].train].sum()):
                raise AssertionError('training population mismatch')
            n_predictions += len(ids)
    verification = json.loads((root(dataset)/VERIFICATION).read_text())
    if not verification['ok']:
        raise AssertionError('clean cache reproduction failed')
    for c in manifest['conditions'][1:]:
        for m in manifest['methods']:
            ids = []
            for p in (root(dataset)/'figures'/c/m).glob('part_*.npz'):
                with np.load(p, allow_pickle=False) as z:
                    ids.extend(z['frame_id'].tolist())
            if set(ids) != excluded or len(ids) != len(excluded):
                raise AssertionError('incomplete or duplicate retained subset')
    return dict(dataset=dataset, predictions_recomputed=n_predictions,
                clean_exact_checks=sum(r['exact'] for r in verification['rows']),
                clean_checks=len(verification['rows']),
                clean_max_absolute_error=max(r['max_absolute_error'] for r in verification['rows']),
                pilot_exclusions=len(excluded), grouped_splits_verified=len(splits),
                feature_manifest_sha256=digest(root(dataset)/MANIFEST),
                classification_manifest_sha256=digest(root(dataset)/'classification_manifest.json'))


def check_row(row, y, labels, ids, pred, baseline):
    """Recompute a row's metrics and its clean-baseline metrics from saved predictions."""
    if row['n_test'] != len(ids):
        raise AssertionError('test count mismatch')
    actual = split_metrics(y[ids], pred, labels)
    clean = split_metrics(y[ids], np.array([baseline[int(i)] for i in ids], dtype=str), labels)
    for metric in METRICS:
        if abs(actual[metric]-row[metric]) > 1e-12:
            raise AssertionError('prediction-score mismatch')
        if abs(clean[metric]-row['clean_'+metric]) > 1e-12:
            raise AssertionError('clean-baseline mismatch')


def check_provenance(dataset):
    """Current inputs, manifest links and the impulse references all match the run's record."""
    base = data_root()/dataset
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    for name, expected in manifest['inputs'].items():
        if digest(base/name) != expected:
            raise RuntimeError(f'{dataset}: input changed since the run: {name}')
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    if frames_fingerprint(frames) != manifest['frames_fingerprint']:
        raise RuntimeError(f'{dataset}: prepared frames changed since the run')
    feature_sha = digest(root(dataset)/MANIFEST)
    for path in (root(dataset)/'classification_manifest.json', root(dataset)/'matrices/complete.json'):
        if json.loads(path.read_text())['feature_manifest_sha256'] != feature_sha:
            raise RuntimeError(f'{dataset}: {path.name} does not point at the current feature manifest')
    impulse = json.loads((base/'robustness/manifest_v2.json').read_text())
    for key in ('inputs', 'subset_ids', 'frames_fingerprint', 'shape'):
        if impulse[key] != manifest[key]:
            raise RuntimeError(f'{dataset}: impulse run {key} differ; its C choices do not apply')
    impulse_sources = json.loads((base/'robustness/classification_manifest.json').read_text())['source_hashes']
    for name in ('baselines/run_classical.py', 'ml/run_trees.py', 'ml/common.py'):
        if impulse_sources[name] != digest(STUDY/name):
            raise RuntimeError(f'{dataset}: {name} differs from the impulse run')
    references = []
    for path in sorted((root(dataset)/'classification/shift').rglob('complete.json')):
        value = json.loads(path.read_text())
        if value['selection'] != 'impulse_run_choice':
            continue
        row = value['rows'][0]
        where = impulse_checkpoint(dataset, row['classifier'], row['method'], row['feature_set'], row['split_id'])
        references.append(f"{path.parent.relative_to(root(dataset))} {digest(where/'complete.json')} "
                          f"{digest(where/'predictions.npz')}")
    return dict(inputs_verified=len(manifest['inputs']), impulse_inputs_match=True,
                impulse_classifier_sources_match=True, reused_reference_checkpoints=len(references),
                reused_reference_digest=hashlib.sha256('\n'.join(references).encode()).hexdigest())


def run_record(datasets):
    """Completed run records that together cover the requested datasets."""
    records = [json.loads(p.read_text()) for p in sorted(runs_dir().glob('run_*.json'))]
    records = [r for r in records if set(r['datasets']) & set(datasets)]
    covered = set().union(*(set(r['datasets']) for r in records))
    if not set(datasets) <= covered:
        raise RuntimeError(f'no completed run record for {sorted(set(datasets) - covered)}')
    return dict(workers=max(r['workers'] for r in records), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUTPUT_ROOT/'robustness_laplace')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS),
                        help='a subset gives a preview, which must go to a non-default --out')
    args = parser.parse_args()
    partial = set(args.datasets) != set(DATASETS)
    if partial and args.out == OUTPUT_ROOT/'robustness_laplace':
        raise SystemExit('a dataset subset is a preview; pass a scratch --out')
    current = verify()
    if args.out.exists():
        raise FileExistsError(f'refusing to replace {args.out}')
    scores, fit_costs, stability, costs, diagnostics, failures, validation = [], [], [], [], [], [], []
    for d in args.datasets:
        print(f'validating {d}', flush=True)
        for provenance, field in ((MANIFEST, 'sources'), ('classification_manifest.json', 'source_hashes')):
            for filename, expected in json.loads((root(d)/provenance).read_text())[field].items():
                if digest(STUDY/filename) != expected:
                    raise RuntimeError(f'source changed since the run: {filename}')
        provenance = check_provenance(d)
        s, f = load_scores(d)
        scores.append(s)
        fit_costs.append(f)
        validation.append(validate_predictions(d) | provenance)
        st, co, di, fa = aggregate_representations(d)
        stability.append(st)
        costs.append(co)
        diagnostics.append(di)
        failures.append(fa)
    scores, fit_costs = pd.concat(scores, ignore_index=True), pd.concat(fit_costs, ignore_index=True)
    stability, costs = pd.concat(stability, ignore_index=True), pd.concat(costs, ignore_index=True)
    diagnostics, failures = pd.concat(diagnostics, ignore_index=True), pd.concat(failures, ignore_index=True)
    tests, vectors = paired_tests(scores, args.datasets)
    reading = readings(tests)
    run = None if partial else run_record(args.datasets)
    verify()
    out = args.out
    out.mkdir(parents=True)
    for name, table in [('per_split_scores.csv', scores), ('paired_tests.csv', tests), ('paired_vectors.csv', vectors),
                        ('primary_tests.csv', tests[tests.family == 'Primary']), ('stability_metrics.csv', stability),
                        ('costs.csv', costs), ('fit_costs.csv', fit_costs), ('solver_diagnostics.csv', diagnostics),
                        ('failed_frames.csv', failures), ('prespecified_reading.csv', reading)]:
        table.to_csv(out/name, index=False, mode='x')
    summary = scores.groupby(['dataset', 'scenario', 'classifier', 'feature_set', 'method', 'condition'])[
        list(METRICS)].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(k).rstrip('_') if isinstance(k, tuple) else k for k in summary.columns]
    summary.to_csv(out/'score_summary.csv', index=False, mode='x')
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'lightgbm', 'EMD-signal', 'numba', 'pimf'):
        versions[package] = importlib.metadata.version(package)
    stamp = datetime.now(timezone.utc).isoformat()
    save_json(out/'environment_audit.json', dict(start=json.loads(START.read_text()), final=dict(utc=stamp, pimf=current),
                                                 unchanged=True, versions=versions,
                                                 runs=run['records'] if run else None))
    save_json(out/'validation.json', dict(datasets=validation, families=tests.groupby('holm_family').size().to_dict(),
                                          expected_families=FAMILY_SIZES, all_prediction_metrics_recomputed=True,
                                          shift_classical_selection=fit_costs.groupby(['dataset', 'selection']).size()
                                          .rename('fits').reset_index().to_dict('records')))
    figures(out, scores, tests, stability)
    render_report(out, scores, tests, stability, costs, diagnostics, fit_costs, reading, run)
    print(out/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
