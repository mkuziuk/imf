"""Small CSVs, figures and a source-linked REPORT.md for the Laplace experiment."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emdirmf.decompose import normalize_frame, slot_labels
from emdirmf.paths import data_root
from robustness.pilot import METHODS
from robustness_laplace.analysis import CLASSICAL, DATASETS, FAMILY_SIZES, LINEAR, ROBUST
from robustness_laplace.contamination import CONDITIONS, LEVELS, MATCHED, SCALE, condition_level
from robustness_laplace.processing import MANIFEST, VERIFICATION
from robustness_laplace.storage import disk_bytes, root

LABELS = dict(synthetic='Synthetic', dronerf='DroneRF', birdclef2023='BirdCLEF 2023', esc50_animals='ESC-50 animals')
NAMES = {'raw': 'Raw', 'emd': 'EMD', LINEAR: 'Linear IMF', ROBUST: 'IRMF'}
COLORS = {'raw': '#898781', 'emd': '#1baf7a', LINEAR: '#eb6834', ROBUST: '#2a78d6'}
ORDER = ('dronerf', 'birdclef2023', 'esc50_animals', 'synthetic')
TOP = CONDITIONS[-1]


def present(table):
    """Datasets in display order, limited to those in a (possibly partial) result table."""
    return [d for d in ORDER if d in set(table.dataset)]


def pct_label(condition):
    return 'clean' if condition == 'clean' else f'{100*condition_level(condition):g}%'


def aggregate_representations(dataset):
    """All-frame descriptive medians and p95; the chunk NPZs keep individual values."""
    stability, costs, diagnostics, failed = [], [], [], []
    length = json.loads((data_root()/dataset/'prepared/dataset.json').read_text())['frame_length']
    for condition in CONDITIONS[1:]:
        values = {m: dict(drift=[], l2=[], cpu=[], seconds=[], cap=[], ok=[]) for m in METHODS}
        h_ratios = []
        for path in sorted((root(dataset)/'features'/condition).glob('part_*.npz')):
            with np.load(path, allow_pickle=False) as z:
                h_ratios.append(z['h_ratio'])
                for m in METHODS:
                    for k in values[m]:
                        values[m][k].append(z[m+'__'+k])
                    for i, error, ok in zip(z['frame_id'], z[m+'__errors'], z[m+'__ok']):
                        if not ok:
                            failed.append(dict(dataset=dataset, condition=condition, method=m,
                                               frame_id=int(i), error=str(error)))
        h = np.concatenate(h_ratios)
        robust_cap = np.concatenate(values[ROBUST]['cap'])
        robust_ok = np.concatenate(values[ROBUST]['ok'])
        diagnostics.append(dict(dataset=dataset, condition=condition, n_frames=len(h), n_channels=h.size,
                                h_ratio_median=float(np.median(h)), h_ratio_p95=float(np.quantile(h, .95)),
                                newton_cap_channels=int(robust_cap.sum()),
                                newton_cap_frames=int(robust_cap.any(axis=1).sum()),
                                newton_failed_frames=int((~robust_ok).sum())))
        for method in METHODS:
            v = {k: np.concatenate(a) for k, a in values[method].items()}
            ok = v['ok']
            for fs, i in [('comp', 0), ('comp+log_scale', 1)]:
                a = v['drift'][ok, i]
                stability.append(dict(dataset=dataset, condition=condition, method=method, measure='feature_drift',
                                      feature_set=fs, slot='all', n=len(a), median=float(np.median(a)),
                                      p95=float(np.quantile(a, .95))))
            for i, label in enumerate(slot_labels(method, length)+['whole_stack']):
                a = v['l2'][ok, :, i].ravel()
                stability.append(dict(dataset=dataset, condition=condition, method=method, measure='relative_l2',
                                      feature_set='components', slot=label, n=len(a), median=float(np.median(a)),
                                      p95=float(np.quantile(a, .95))))
            costs.append(dict(dataset=dataset, condition=condition, method=method,
                              stage='decomposition_features_stability', n_frames=len(ok),
                              cpu_seconds=float(v['cpu'].sum()), frame_wall_seconds=float(v['seconds'].sum()),
                              cpu_s_per_frame=float(v['cpu'].mean()), failed_frames=int((~ok).sum()),
                              cap_channels=int(v['cap'].sum())))
    return (pd.DataFrame(stability), pd.DataFrame(costs), pd.DataFrame(diagnostics),
            pd.DataFrame(failed, columns=['dataset', 'condition', 'method', 'frame_id', 'error']))


def readings(tests):
    rows = []
    for dataset in present(tests):
        cell = tests[(tests.dataset == dataset) & (tests.family == 'Primary')].set_index('condition')
        top = cell.loc[TOP]
        significant = bool(top.p_nb_holm < .05)
        verdict = ('IRMF more robust' if significant and top.mean_diff > 0 else
                   'linear IMF more robust' if significant else 'no significant difference')
        rows.append(dict(dataset=dataset, condition=TOP, did=top.mean_diff, ci_low=top.ci_low, ci_high=top.ci_high,
                         p_nb_holm=top.p_nb_holm, positive=bool(top.mean_diff > 0), holm_significant=significant,
                         irmf_more_robust=bool(significant and top.mean_diff > 0), verdict=verdict,
                         did_by_level=json.dumps({c: float(cell.loc[c].mean_diff) for c in CONDITIONS[1:]}),
                         status=top.status))
    return pd.DataFrame(rows)


def mean_ba(scores, dataset, classifier, feature_set, method, condition, scenario='shift'):
    cell = scores[(scores.dataset == dataset) & (scores.scenario == scenario) & (scores.classifier == classifier)
                  & (scores.feature_set == feature_set) & (scores.method == method) & (scores.condition == condition)]
    return 100 * cell.balanced_accuracy.mean()


def figures(out, scores, tests, stability):
    folder = out/'figures'
    folder.mkdir()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    x = [0, *LEVELS]
    order = present(tests)
    n = len(order)
    for classifier, fs, filename in (('classical', 'comp', 'ba_curves'),
                                     ('classical', 'comp+log_scale', 'ba_curves_classical_with_scale'),
                                     ('lightgbm', 'comp', 'ba_curves_lightgbm_comp'),
                                     ('lightgbm', 'comp+log_scale', 'ba_curves_lightgbm_with_scale')):
        fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.8), squeeze=False)
        axes = axes[0]
        for ax, dataset in zip(axes, order):
            clf = CLASSICAL[dataset] if classifier == 'classical' else classifier
            for method in METHODS:
                means = [mean_ba(scores, dataset, clf, fs, method, c) for c in CONDITIONS]
                ax.plot(x, means, 'o-', label=NAMES[method], color=COLORS[method], ms=3.5, lw=1.6)
            ax.set_title(f'{LABELS[dataset]} | {clf}')
            ax.set_xticks(x, [pct_label(c) for c in CONDITIONS])
            ax.grid(alpha=.25)
            ax.set_xlabel('Share of samples with Laplace noise')
        axes[0].set_ylabel('Balanced accuracy (%)')
        axes[0].legend(fontsize=8)
        label = 'fixed classical model' if classifier == 'classical' else 'LightGBM'
        fig.suptitle(f'Clean-trained models tested on contaminated data, {label}, {fs} features; fold means', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .93])
        fig.savefig(folder/(filename+'.png'), dpi=180)
        if filename == 'ba_curves':
            fig.savefig(folder/(filename+'.pdf'))
        plt.close(fig)
    p = tests[tests.family == 'Primary']
    fig, axs = plt.subplots(1, n, figsize=(3.5*n, 3.6), sharex=True, squeeze=False)
    axs = axs[0]
    for ax, dataset in zip(axs, order):
        cell = p[p.dataset == dataset].set_index('condition').loc[list(CONDITIONS[1:])]
        for i, (condition, r) in enumerate(cell.iterrows()):
            filled = r.p_nb_holm < .05
            ax.errorbar(100*r.mean_diff, i, xerr=[[100*(r.mean_diff-r.ci_low)], [100*(r.ci_high-r.mean_diff)]],
                        fmt='o', color='#2a78d6', mfc='#2a78d6' if filled else 'white', capsize=3, ms=5)
        ax.axvline(0, color='.6', lw=1)
        ax.set_title(LABELS[dataset])
        ax.grid(axis='x', alpha=.25)
        ax.set_xlabel('DiD, BA points')
        ax.set_yticks(range(len(cell)), [pct_label(c) for c in cell.index])
        ax.invert_yaxis()
    fig.suptitle('IRMF minus linear IMF gap change, clean-trained classical model, comp features\n'
                 f'95% corrected CIs; filled: Holm p < .05 (real datasets, family of {FAMILY_SIZES["Primary"]}; '
                 f'synthetic, its own family of {FAMILY_SIZES["synthetic"]})', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .86])
    fig.savefig(folder/'primary_did_forest.png', dpi=180)
    fig.savefig(folder/'primary_did_forest.pdf')
    plt.close(fig)
    fig, axs = plt.subplots(1, n, figsize=(3.5*n, 3.6), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, d in zip(axs, order):
        for m in METHODS:
            cell = stability[(stability.dataset == d) & (stability.method == m) & (stability.measure == 'feature_drift')
                             & (stability.feature_set == 'comp')].set_index('condition')
            ax.plot(LEVELS, cell.loc[list(CONDITIONS[1:]), 'median'], 'o-', label=NAMES[m], color=COLORS[m], ms=3.5)
        ax.set_title(LABELS[d])
        ax.set_xlabel('Share of samples with Laplace noise')
        ax.grid(alpha=.25)
    axs[0].set_ylabel('Median standardized feature drift')
    axs[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder/'feature_drift.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(n, 3, figsize=(13, 2.5*n), squeeze=False)
    for row, d in enumerate(order):
        manifest = json.loads((root(d)/MANIFEST).read_text())
        frame_id = manifest['subset_ids'][0]
        frames = np.load(data_root()/d/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
        clean_raw, _ = normalize_frame(frames[frame_id, 0])
        for col, m in enumerate(('raw', LINEAR, ROBUST)):
            for path in (root(d)/'figures'/TOP/m).glob('part_*.npz'):
                with np.load(path, allow_pickle=False) as z:
                    loc = np.flatnonzero(z['frame_id'] == frame_id)
                    if len(loc):
                        contaminated = z['stacks'][loc[0], 0, 0].copy()
                        break
            clean = clean_raw if m == 'raw' else np.load(data_root()/d/'decomp'/f'{m}.npy', mmap_mode='r',
                                                         allow_pickle=False)[frame_id, 0, 0]
            ax = axs[row, col]
            ax.plot(clean, lw=.7, color='.35', label='Clean')
            ax.plot(contaminated, lw=.7, alpha=.75, color=COLORS[m], label='Contaminated')
            ax.set_title(f'{LABELS[d]} frame {frame_id} | {NAMES[m]}' + (', finest component' if col else ', normalized'))
            if row == n-1:
                ax.set_xlabel('Sample')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f'Fixed-subset examples, channel 0, Laplace noise on {pct_label(TOP)} of samples')
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(folder/'component_examples.png', dpi=160)
    plt.close(fig)


def fmt_p(p):
    return f'{p:.2g}' if p >= 1e-3 else f'{p:.1e}'


def test_row(r):
    return f'{100*r.mean_diff:+.2f} | [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}] | {fmt_p(r.p_nb_holm)}'


def render_report(out, scores, tests, stability, costs, diagnostics, fit_costs, reading, run):
    ORDER = present(tests)
    verifications = [json.loads((root(d)/VERIFICATION).read_text()) for d in ORDER]
    exact = sum(r['exact'] for v in verifications for r in v['rows'])
    total = sum(len(v['rows']) for v in verifications)
    classical = fit_costs[(fit_costs.scenario == 'shift') & (fit_costs.classifier != 'lightgbm')]
    reused = int((classical.selection == 'impulse_run_choice').sum())
    trees = fit_costs[fit_costs.classifier == 'lightgbm']
    tree_match = int(trees.clean_matches_impulse_run.astype(bool).sum())
    primary = tests[tests.family == 'Primary']
    lines = ['# Laplace contamination and classification robustness', '']
    if run is None:
        lines += [f'Preview of {", ".join(LABELS[d] for d in ORDER)} only. This is not the protocol\'s '
                  'report: the full report needs all four datasets.', '']
    lines += ['## Question and design', '',
             'Does IRMF keep more of its classification performance than linear IMF when a growing share of '
             'samples carries Laplace noise? A positive difference-in-differences (DiD) means the IRMF minus '
             'linear IMF gap becomes more favorable under contamination. It does not mean contaminated accuracy '
             'exceeds clean accuracy.', '',
             'The [protocol](../../../experiments/classification-emd-irmf/robustness_laplace/PROTOCOL_laplace.md) '
             'was written before any contaminated result, and its SHA-256 is in every run manifest. It reuses '
             'the [impulse experiment](../robustness/REPORT.md) and changes the noise. With probability p a '
             f'sample receives Laplace(0, b) noise, b = {SCALE:g} robust SD of the clean channel '
             '(1.4826 MAD), for p = 0.1, 0.2, 0.3 and 0.4. There is no Gaussian term and no Gaussian control. '
             'The draws are the impulse run\'s, so the masks are nested across p and its two-sided spikes at '
             '4% sit on the same samples, at ten times the scale. Injection precedes the shared median/MAD '
             'normalization.', '',
             'Datasets, inputs, splits, the 100 excluded pilot frames per dataset, features and classifiers are '
             'unchanged. The real tasks are DroneRF 10class, BirdCLEF 2023 species and ESC-50 animals; synthetic '
             'is its own Holm family. The 60 GHz radar is excluded. IRMF uses the converged Numba Newton solver. '
             f'Clean features reproduced the cache on 50 frames per dataset ({exact}/{total} exactly). '
             '[Validation](validation.json).', '',
             'Shift fits train one clean model per split, input, feature set and classifier and predict every '
             'condition. To save time, the clean classical models were refit at the C the impulse run had '
             'selected on the same training rows, and accepted only if their clean predictions matched that '
             f'run\'s on every test frame. {reused} of {len(classical)} matched; the rest reran the full grid '
             'search. A match confirms the inputs and C. The solvers are deterministic (primal liblinear for '
             'the SVM, lbfgs for logistic regression), which is what makes the refit the same model. '
             f'LightGBM shift fits reran in full, and {tree_match} of {len(trees)} reproduced the '
             f'impulse run\'s clean predictions exactly. Matched fits train and test at p = {condition_level(MATCHED[0]):g} '
             'with the full grid. [Per-split scores](per_split_scores.csv), [fit costs](fit_costs.csv).', '',
             '## Primary results', '',
             f'The primary family holds the {FAMILY_SIZES["Primary"]} real-data DiDs below: IRMF minus linear IMF, '
             'clean-trained fixed classical model, comp features. Tests are two-sided Nadeau-Bengio corrected '
             't-tests with Holm correction across the family and marginal 95% corrected intervals. Wilcoxon '
             'tests are saved as secondary. Values are fold-paired percentage points. '
             '[Full tests](paired_tests.csv), [paired vectors](paired_vectors.csv).', '',
             '| Dataset | Laplace share | DiD, pp | 95% CI, pp | Holm p |', '|---|---|---:|---|---:|']
    for d in ORDER:
        cell = primary[primary.dataset == d].set_index('condition')
        for c in CONDITIONS[1:]:
            lines.append(f'| {LABELS[d]} | {pct_label(c)} | {test_row(cell.loc[c])} |')
    interim = ('' if tests.holm_complete.all() else ' Some families are incomplete in this preview, so Holm '
               'counts their missing members as p = 1 and never uses a smaller family than the protocol.')
    lines += ['', 'Synthetic p values are Holm-corrected within its own family of '
              f'{FAMILY_SIZES["synthetic"]} tests.' + interim, '', '![Primary DiDs](figures/primary_did_forest.png)', '',
              'The protocol reads IRMF as more robust on a dataset if the 40% DiD is positive with Holm p < .05, '
              'and linear IMF as more robust if it is negative with Holm p < .05. '
              '[Reading CSV](prespecified_reading.csv).', '']
    for _, r in reading.set_index('dataset').loc[list(ORDER)].reset_index().iterrows():
        lines.append(f'- {LABELS[r.dataset]}: {r.verdict}. DiD at 40% {100*r.did:+.2f} pp, '
                     f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}], Holm p {fmt_p(r.p_nb_holm)}.')
    lines += ['', '## Secondary results, exploratory', '',
              f'Each real-data family has its own Holm correction: S1 = {FAMILY_SIZES["S1"]}, S3 = {FAMILY_SIZES["S3"]}, '
              f'S4 = {FAMILY_SIZES["S4"]} and S5 = {FAMILY_SIZES["S5"]}. S1 compares IRMF with EMD; S3 uses LightGBM; '
              'S4 adds log_scale; S5 trains and tests at 40% relative to the clean-trained, clean-tested '
              'baseline. The impulse run\'s S2 tested Gaussian controls and has no counterpart.', '',
              '| Dataset | Family | Tests | Positive, Holm p < .05 | Negative, Holm p < .05 | DiD range, pp |',
              '|---|---|---:|---:|---:|---|']
    for d in ORDER:
        for family in ('S1', 'S3', 'S4', 'S5'):
            c = tests[(tests.dataset == d) & (tests.family == family)]
            pos = int(((c.mean_diff > 0) & (c.p_nb_holm < .05)).sum())
            neg = int(((c.mean_diff < 0) & (c.p_nb_holm < .05)).sum())
            lines.append(f'| {LABELS[d]} | {family} | {len(c)} | {pos} | {neg} | '
                         f'{100*c.mean_diff.min():+.2f} to {100*c.mean_diff.max():+.2f} |')
    lines += ['', 'Selected 40% contrasts, chosen by severity, not by significance:', '',
              '| Dataset | Contrast | DiD, pp | 95% CI, pp | Holm p |', '|---|---|---:|---|---:|']
    choices = [('S1', 'comp', 'shift', 'emd', 'classical', 'classical vs EMD'),
               ('S3', 'comp', 'shift', LINEAR, 'lightgbm', 'LightGBM vs linear IMF'),
               ('S3', 'comp', 'shift', 'emd', 'lightgbm', 'LightGBM vs EMD'),
               ('S4', 'comp+log_scale', 'shift', LINEAR, 'classical', 'classical + log_scale vs linear IMF'),
               ('S5', 'comp', 'matched', LINEAR, 'classical', 'matched training vs linear IMF'),
               ('S5', 'comp', 'matched', 'emd', 'classical', 'matched training vs EMD')]
    for d in ORDER:
        for family, fs, scenario, comp, clf, label in choices:
            a = tests[(tests.dataset == d) & (tests.family == family) & (tests.condition == TOP)
                      & (tests.feature_set == fs) & (tests.scenario == scenario) & (tests.comparator == comp)
                      & (tests.classifier == (CLASSICAL[d] if clf == 'classical' else clf))]
            lines.append(f'| {LABELS[d]} | {family}, {label} | {test_row(a.iloc[0])} |')
    lines += ['', '## Absolute classification performance', '',
              'Mean fold balanced accuracy, clean-trained fixed classical model, comp features. Descriptive. '
              'Every classifier and feature set is in [score_summary.csv](score_summary.csv).', '',
              '![BA curves](figures/ba_curves.png)', '',
              'More curves: [classical with log_scale](figures/ba_curves_classical_with_scale.png), '
              '[LightGBM comp](figures/ba_curves_lightgbm_comp.png), '
              '[LightGBM with log_scale](figures/ba_curves_lightgbm_with_scale.png).', '',
              '| Dataset | Input | ' + ' | '.join(f'{pct_label(c)} BA, %' for c in CONDITIONS) + ' |',
              '|---|---|' + '---:|' * len(CONDITIONS)]
    for d in ORDER:
        for m in METHODS:
            v = [mean_ba(scores, d, CLASSICAL[d], 'comp', m, c) for c in CONDITIONS]
            lines.append(f'| {LABELS[d]} | {NAMES[m]} | ' + ' | '.join(f'{x:.2f}' for x in v) + ' |')
    lines += ['', '## Representation stability', '',
              'Feature drift is the mean absolute feature change divided by each feature\'s clean all-frame SD. '
              'Relative L2 compares each contaminated stack with its clean cached stack within channels. Both are '
              'descriptive and never enter a classifier. EMD slots can mode-mix, so equally numbered EMD and IMF '
              'slots are not matched bands. [All conditions and slots](stability_metrics.csv).', '',
              '| Dataset | Input | Feature drift median / p95 at 40% | Whole-stack relative L2 median / p95 at 40% |',
              '|---|---|---|---|']
    for d in ORDER:
        for m in METHODS:
            a = stability[(stability.dataset == d) & (stability.method == m) & (stability.condition == TOP)]
            f = a[(a.measure == 'feature_drift') & (a.feature_set == 'comp')].iloc[0]
            l2 = a[(a.measure == 'relative_l2') & (a.slot == 'whole_stack')].iloc[0]
            lines.append(f'| {LABELS[d]} | {NAMES[m]} | {f["median"]:.3g} / {f.p95:.3g} | '
                         f'{l2["median"]:.3g} / {l2.p95:.3g} |')
    lines += ['', '![Feature drift](figures/feature_drift.png)', '',
              'The [component examples](figures/component_examples.png) use the first pre-selected pilot frame '
              'per dataset, channel 0, at 40%.', '',
              '## H, convergence and exclusions', '',
              'IRMF recomputes H from each contaminated channel. Ratios are channel-level medians. '
              '[Diagnostics](solver_diagnostics.csv), [failed frames](failed_frames.csv).', '',
              '| Dataset | Laplace share | Median H / clean H | p95 | Cap-100 retries | Failed frames |',
              '|---|---|---:|---:|---:|---:|']
    for d in ORDER:
        for _, r in diagnostics[diagnostics.dataset == d].iterrows():
            lines.append(f'| {LABELS[d]} | {pct_label(r.condition)} | {r.h_ratio_median:.3g} | {r.h_ratio_p95:.3g} | '
                         f'{r.newton_cap_channels} | {r.newton_failed_frames} |')
    feature_cpu = costs.cpu_seconds.sum()
    fit_cpu = fit_costs.fit_cpu_seconds.sum() + fit_costs.predict_cpu_seconds.sum()
    if run is None:
        timing = 'This preview does not report wall times; the full report does. '
    else:
        parts = [f'{" and ".join(LABELS[d] for d in r["datasets"])} in {r["wall_seconds"]/3600:.2f} h'
                 if r['wall_seconds'] > 360 else
                 f'{" and ".join(LABELS[d] for d in r["datasets"])} in {r["wall_seconds"]:.0f} s'
                 for r in sorted(run['records'], key=lambda r: r['started_utc'])]
        timing = (f'Each run used one pool of {run["workers"]} single-thread workers for its datasets. '
                  f'Wall times: {"; ".join(parts)}. ')
    lines += ['', '## Compute cost', '',
              timing + f'Measured CPU: {feature_cpu/3600:.2f} h for '
              f'decomposition, features and stability, {fit_cpu/3600:.2f} h for fitting and prediction. '
              f'External data occupy {disk_bytes()/1e9:.2f} GB. pimf is 0.4.0 and its source hashes matched the '
              'impulse run\'s record before analysis. [Feature costs](costs.csv), [environment](environment_audit.json).', '',
              '| Dataset | Feature CPU, h | Fit CPU, h | Fits |', '|---|---:|---:|---:|']
    for d in ORDER:
        f = fit_costs[fit_costs.dataset == d]
        lines.append(f'| {LABELS[d]} | {costs[costs.dataset == d].cpu_seconds.sum()/3600:.2f} | '
                     f'{(f.fit_cpu_seconds.sum()+f.predict_cpu_seconds.sum())/3600:.2f} | {len(f)} |')
    lines += ['', '## Scope', '',
              'The results hold for these prepared populations and this injected noise. They do not show '
              'robustness to natural noise or other corruptions. A positive DiD can coexist with lower absolute '
              'IRMF accuracy, so read it with the absolute curves.', '',
              '## Reproduction', '',
              '```bash', 'cd experiments/classification-emd-irmf',
              'export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification',
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
              'export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1',
              'PY=../../.venv/bin/python', '$PY -m pytest -q robustness_laplace/tests',
              '$PY -m robustness_laplace.run --workers 14 --datasets synthetic',
              '$PY -m robustness_laplace.run --workers 14 --datasets dronerf birdclef2023 esc50_animals --detach',
              '$PY -m robustness_laplace.analyse', '```', '',
              'Completed checkpoints resume and are never replaced. Full data, checkpoints and predictions stay '
              'in each dataset\'s external `robustness_laplace/` directory. CSVs keep fractions; tables show '
              'percentage points.']
    with (out/'REPORT.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
