"""CSV summaries, figures and the amended protocol's research report."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustness_v3.analysis import CLASSICAL, FAMILY_SIZES, descriptive_tables, selections, selected_scores
from robustness_v3.construction import METHODS, n_slots, parse_variant
from robustness_v3.contamination import CONDITIONS, LEVELS
from robustness_v3.processing import MANIFEST, VERIFICATION
from robustness_v3.storage import disk_bytes, root

LABELS = dict(synthetic='Synthetic', dronerf='DroneRF', birdclef2023='BirdCLEF 2023', esc50_animals='ESC-50 animals')
NAMES = dict(raw='Raw', emd='EMD', linear='Linear IMF', irmf='IRMF')
COLORS = dict(raw='#898781', emd='#1baf7a', linear='#eb6834', irmf='#2a78d6')
ORDER = ('dronerf', 'birdclef2023', 'esc50_animals', 'synthetic')
TOP = CONDITIONS[-1]


def present(table):
    return [d for d in ORDER if d in set(table.dataset)]


def schedule_text(dataset, selected=None):
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    windows = manifest['windows']
    if selected is not None:
        ks = sorted(set(int(k) for k in selected[selected.method.isin(['linear', 'irmf'])].k))
        windows = {str(k): windows[str(k)] for k in ks}
    return '; '.join(f'K={k}: '+','.join(map(str, w)) for k, w in windows.items())


def aggregate_representations(dataset):
    stability, costs, diagnostics, failed = [], [], [], []
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    for condition in CONDITIONS:
        fields = ('drift', 'l2', 'cpu', 'seconds', 'cap', 'ok', 'h', 'h_ratio', 'sigma_hat', 'sigma_source', 'b_over_h')
        values = {v: {k: [] for k in fields} for v in manifest['methods']}
        for path in sorted((root(dataset)/'features'/condition).glob('part_*.npz')):
            with np.load(path, allow_pickle=False) as z:
                for v in manifest['methods']:
                    for key in fields:
                        values[v][key].append(z[v+'__'+key])
                    failed.extend(dict(dataset=dataset, condition=condition, variant=v, frame_id=int(i), error=str(e))
                                  for i, e, ok in zip(z['frame_id'], z[v+'__errors'], z[v+'__ok']) if not ok)
        for variant, fields in values.items():
            v = {k: np.concatenate(a) for k, a in fields.items()}
            method, k = parse_variant(variant)
            key = dict(dataset=dataset, condition=condition, variant=variant, method=method, k=k)
            ok = v['ok']
            measures = [('feature_drift', 'all', v['drift'][ok])]
            labels = (['raw'] if method == 'raw' else [f'component_{i}' for i in range(k)]+['residual'])
            measures += [('relative_l2', label, v['l2'][ok, :, i].ravel())
                         for i, label in enumerate(labels+['whole_stack'])]
            for measure, slot, a in measures:
                stability.append(dict(key, measure=measure, slot=slot, n=len(a),
                    median=float(np.median(a)) if len(a) else np.nan,
                    p95=float(np.quantile(a, .95)) if len(a) else np.nan))
            diag = dict(key, n_frames=len(ok), n_channels=v['h'].size,
                        newton_cap_channels=int(v['cap'].sum()), newton_cap_frames=int(v['cap'].any(axis=1).sum()),
                        failed_frames=int((~ok).sum()), sigma_fallback_channels=int((v['sigma_source'] != 0).sum()))
            for name in ('h', 'h_ratio', 'sigma_hat', 'b_over_h'):
                diag[name+'_median'] = float(np.median(v[name]))
                diag[name+'_p95'] = float(np.quantile(v[name], .95))
            diagnostics.append(diag)
            costs.append(dict(key, n_frames=len(ok), cpu_seconds=float(v['cpu'].sum()),
                frame_wall_seconds=float(v['seconds'].sum()), cpu_s_per_frame=float(v['cpu'].mean()),
                failed_frames=int((~ok).sum()), cap_channels=int(v['cap'].sum())))
    return pd.DataFrame(stability), pd.DataFrame(costs), pd.DataFrame(diagnostics), pd.DataFrame(
        failed, columns=['dataset', 'condition', 'variant', 'frame_id', 'error'])


def readings(tests):
    rows = []
    for dataset in present(tests):
        top = tests[(tests.dataset == dataset) & (tests.family == 'Primary') & (tests.condition == TOP)].iloc[0]
        significant = top.p_nb_holm < .05
        verdict = ('IRMF more robust' if significant and top.mean_diff > 0 else
                   'linear IMF more robust' if significant else 'does not reject; not equivalence')
        rows.append(dict(dataset=dataset, condition=TOP, irmf_k=top.irmf_k, linear_k=top.comparator_k,
                         did=top.mean_diff, ci_low=top.ci_low, ci_high=top.ci_high,
                         p_nb_holm=top.p_nb_holm, verdict=verdict, status=top.status))
    return pd.DataFrame(rows)


def figure_footer(fig, datasets, selected):
    lines = []
    for d in datasets:
        cell = selected[selected.dataset == d]
        ks = ', '.join(f'{r.classifier}/{r.method} K*={r.k}' for r in cell.itertuples())
        lines.extend([f'{LABELS[d]}: {ks}', 'Windows, coarse to fine: '+schedule_text(d, cell)])
    fig.text(.01, .01, '\n'.join(lines), fontsize=6.3, va='bottom', wrap=True)
    return min(.55, .08+.035*len(lines))


def figures(out, scores, tests, stability):
    folder = out/'figures'
    folder.mkdir()
    chosen = selections(scores)
    selected = selected_scores(scores)
    order = present(scores)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    for classifier in ('classical', 'lightgbm'):
        fig, axes = plt.subplots(len(order), 1, figsize=(11, 4.4*len(order)+1.5), squeeze=False)
        used = []
        for ax, d in zip(axes[:, 0], order):
            clf = CLASSICAL[d] if classifier == 'classical' else classifier
            sel = chosen[(chosen.dataset == d) & (chosen.classifier == clf)]
            used.append(sel)
            for r in sel.itertuples():
                cell = selected[(selected.dataset == d) & (selected.classifier == clf)
                                & (selected.variant == r.variant) & (selected.scenario == 'shift')]
                means = cell.groupby('condition').balanced_accuracy.mean().reindex(CONDITIONS)
                ax.plot([0, *LEVELS], 100*means, 'o-', label=f'{NAMES[r.method]}, K*={r.k}', color=COLORS[r.method])
            ax.set(title=f'{LABELS[d]} | {clf} | clean-selected K*', ylabel='Balanced accuracy (%)',
                   xlabel='Contaminated sample fraction', xticks=[0, *LEVELS])
            ax.legend(fontsize=8); ax.grid(alpha=.25)
        margin = figure_footer(fig, order, pd.concat(used))
        fig.tight_layout(rect=[0, margin, 1, 1])
        fig.savefig(folder/f'ba_curves_{classifier}.png', dpi=160)
        fig.savefig(folder/f'ba_curves_{classifier}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(len(order), 1, figsize=(11, 3.6*len(order)+2), squeeze=False)
    for ax, d in zip(axes[:, 0], order):
        cell = tests[(tests.dataset == d) & (tests.family == 'Primary')].set_index('condition').loc[list(CONDITIONS[1:])]
        for i, r in enumerate(cell.itertuples()):
            ax.errorbar(100*r.mean_diff, i, xerr=[[100*(r.mean_diff-r.ci_low)], [100*(r.ci_high-r.mean_diff)]],
                        fmt='o', capsize=3, mfc=COLORS['irmf'] if r.p_nb_holm < .05 else 'white', color=COLORS['irmf'])
        ax.axvline(0, color='.6'); ax.grid(axis='x', alpha=.25)
        ax.set(title=f'{LABELS[d]}, IRMF K*={cell.irmf_k.iloc[0]} vs linear K*={cell.comparator_k.iloc[0]}',
               yticks=range(4), yticklabels=CONDITIONS[1:], xlabel='DiD, BA percentage points, corrected 95% CI')
    fig.suptitle('Primary DiD; filled points: Holm p < .05. Synthetic family=18; real Primary family=12')
    margin = figure_footer(fig, order, chosen[chosen.classifier != 'lightgbm'])
    fig.tight_layout(rect=[0, margin, 1, .96]); fig.savefig(folder/'primary_did_forest.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(order), 1, figsize=(11, 3.6*len(order)+2), squeeze=False)
    for ax, d in zip(axes[:, 0], order):
        for r in chosen[(chosen.dataset == d) & (chosen.classifier == CLASSICAL[d])].itertuples():
            cell = stability[(stability.dataset == d) & (stability.variant == r.variant)
                             & (stability.measure == 'feature_drift')].set_index('condition').loc[list(CONDITIONS)]
            ax.plot([0, *LEVELS], cell['median'], 'o-', color=COLORS[r.method], label=f'{NAMES[r.method]}, K*={r.k}')
        ax.set(title=LABELS[d], ylabel='Median standardized feature drift', xlabel='Contaminated sample fraction')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    margin = figure_footer(fig, order, chosen[chosen.classifier != 'lightgbm'])
    fig.tight_layout(rect=[0, margin, 1, 1]); fig.savefig(folder/'feature_drift.png', dpi=160); plt.close(fig)


def render_report(out, scores, tests, stability, costs, diagnostics, fit_costs, reading, run):
    every, best, selected, chosen = descriptive_tables(scores)
    lines = ['# Classification robustness v3', '',
        'Protocol and amendment dated 2026-10-22. Noise is added before shared median/MAD normalization: '
        '`x + (u < p) * 3 noise_sigma_hat(x_clean_raw) * e * s`, with common random numbers and p=.1,.2,.3,.4. '
        'A constant channel receives no contamination. There is no Gaussian term.', '',
        'IRMF uses converged safeguarded Newton, tolerance 1e-12, cap 100 and cap-1000 retry. '
        'H is fixed in normalized units per dataset, twice the median E1 across every channel of its 100 '
        'excluded clean pilot frames. Those frames enter no classifier fit or score. E1 is retained by the '
        'pre-specified IQR rule, 8/12 lower-error and 4/12 no-worse-stability cells. '
        'Pilot report SHA-256: `9bb6f2c66f6da7a54090be2bf0c5cbe2aedf70f6adb12574a01eb86b07eb4e8b`.', '',
        'Features are the shared 20 per slot, comp only. Raw normalized samples stay float64; decomposition '
        'stacks are float32. EMD runs once per channel with 8 IMFs plus residue; lower K folds the remaining '
        'slots into the residue. IMF and IRMF keep the fine content below the 31-sample final window in '
        'their residual slot. Their extracted slots are stored fine to coarse, followed by the residual.', '',
        '## Fixed H and windows', '']
    for d in present(scores):
        manifest = json.loads((root(d)/MANIFEST).read_text())
        lines += [f'### {LABELS[d]}', '', f'H_d = {manifest["fixed_h"]["H_d"]:.15g}; '
                  f'{manifest["fixed_h"]["n_channels"]} pilot channels. EMD K=2,3,4,6,8. '
                  f'Linear and IRMF windows, coarse to fine: {schedule_text(d)}.', '']
    lines += ['## Clean-selected K*', '',
        'Each classifier selects its own K* by mean clean balanced accuracy over all 15 grouped splits, '
        'ties to smaller K. The same K* applies at every condition. Selecting on clean outer-fold scores '
        'makes clean scores slightly optimistic for every method alike. These are descriptive selection '
        'scores, without nested K selection.', '',
        chosen[['dataset', 'classifier', 'method', 'k', 'mean_clean_ba']].to_markdown(index=False, floatfmt='.6f'), '',
        '## Pre-specified reading at 40%', '', reading.to_markdown(index=False, floatfmt='.6f'), '',
        'DiD = [BA_IRMF*(p) - BA_comparator*(p)] - [BA_IRMF*(clean) - BA_comparator*(clean)]. '
        'Two-sided Nadeau-Bengio corrected t-tests use 15 paired splits. Wilcoxon is secondary. '
        'Holm families are Primary 12, S1 12, S3 24 and S5 6 over the three real datasets. '
        'Synthetic uses a separate family of 18. Partial previews pad absent family tests with p=1. '
        'A positive significant Primary DiD at 40% favors IRMF robustness; a negative significant one '
        'favors linear IMF. Failure to reject is not equivalence.', '',
        'Matched classical models train and test at 40%, at each method\'s classical K*. Each is scheduled '
        'only after that method\'s clean shift fits finish. Every classical fit runs the full shared grouped '
        'C grid afresh. LightGBM uses the shared inner early stopping and outer refit. No model is pickled.', '',
        '## Best-vs-best balanced accuracy', '',
        'Fractions, mean and sample SD over 15 splits. K* is fixed by clean scores.', '',
        selected.to_markdown(index=False, floatfmt='.6f'), '',
        '## All K and conditions', '', every.to_markdown(index=False, floatfmt='.6f'), '',
        '## Descriptive best K at each condition', '',
        'These maxima use the condition\'s own outer-fold scores and do not define the hypothesis tests.', '',
        best.to_markdown(index=False, floatfmt='.6f'), '',
        '## Paired tests', '', tests.to_markdown(index=False, floatfmt='.6f'), '',
        '## Solver and H diagnostics', '',
        '`h_ratio` means the input\'s E1 estimate divided by fixed H_d, not an H inflation factor. '
        '`b_over_h` converts raw b through the input\'s normalization scale before dividing by H_d. '
        'Per-frame, per-channel diagnostics and per-slot relative L2 are retained in the numeric chunks '
        'for every variant, including clean. Full stacks are retained only for the 100 pilot IDs.', '',
        diagnostics.to_markdown(index=False, floatfmt='.6f'), '',
        '## Verification and dated deviations', '',
        '2026-10-22: the first raw gate failed on 7/200 raw checks because the draft cast raw samples to '
        'float32. All 200 EMD checks passed exactly. The original `clean_verification.json` files remain '
        'unchanged. The amendment restores float64 raw samples. The passing checks are separately saved '
        'as `clean_verification_v2.json`, referenced by `manifest_v2.json`, version 2. '
        'No tolerance was relaxed and contamination began only after this new gate passed.', '',
        '2026-10-22: constant-channel noise is zero when the shared estimator returns its constant fallback '
        'source code 2. H calibration retains the shared E1 fallbacks. Fixed H uses the excluded pilots, '
        'as amended, rather than the pilot report\'s split-0 training calibration.', '']
    for d in present(scores):
        value = json.loads((root(d)/VERIFICATION).read_text())
        lines += [f'{LABELS[d]} verification: {sum(r["ok"] for r in value["rows"])}/{len(value["rows"])} pass, '
                  f'{sum(r["exact"] for r in value["rows"])}/{len(value["rows"])} exact.', '']
    lines += ['## Cost and reproduction', '', f'External v3 outputs currently occupy {disk_bytes()/1e9:.3f} GB of the 30 GB budget.', '',
        costs.to_markdown(index=False, floatfmt='.4f'), '',
        f'Fit CPU total: {fit_costs.fit_cpu_seconds.sum():.2f} s. '
        + (f'Completed run records: {json.dumps(run)}.' if run else 'Dataset-subset preview.'), '',
        'Run from `experiments/classification-emd-irmf` with `IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification` '
        'and OMP, OpenBLAS, MKL, VECLIB, NumExpr and Numba thread variables set to 1.', '',
        '```bash', '../../.venv/bin/python -m pytest -q robustness_v3/tests robustness_laplace/tests',
        '../../.venv/bin/python -m robustness_v3.run --workers 14 --datasets dronerf birdclef2023 esc50_animals --detach',
        '../../.venv/bin/python -m robustness_v3.analyse', '```', '',
        'Source/input hashes, fixed H calibration, cached-feature verification, every saved prediction metric '
        'and its clean baseline are checked by analyse before report publication. '
        'CSV score units are fractions. Figures use percentage points.', '']
    with (out/'REPORT.md').open('x') as f:
        f.write('\n'.join(lines))
