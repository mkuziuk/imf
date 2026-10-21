"""Small CSVs, scientific figures and a source-linked plain-prose report."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emdirmf.decompose import normalize_frame, slot_labels
from emdirmf.paths import data_root
from robustness.analysis import CLASSICAL, DATASETS, LINEAR, METHODS, ROBUST, load_scores, paired_tests
from robustness.contamination import CONDITIONS, KINDS, LEVELS
from robustness.environment import START, verify
from robustness.processing import VERIFICATION
from robustness.storage import disk_bytes, root, save_json

LABELS = dict(synthetic='Synthetic*', dronerf='DroneRF', birdclef2023='BirdCLEF 2023', esc50_animals='ESC-50 animals')
METHOD_LABELS = dict(raw='Raw', emd='EMD', imf_linear_a1_41='Linear IMF')
NAMES = {'raw':'Raw', 'emd':'EMD', LINEAR:'Linear IMF', ROBUST:'IRMF'}
COLORS = dict(zip(METHODS, ['#777777','#d88b22','#2166ac','#b2182b']))


def aggregate_representations(dataset):
    """All-frame descriptive medians/p95; retained NPZs keep individual values."""
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
                    for i, error, ok in zip(z['frame_id'],z[m+'__errors'],z[m+'__ok']):
                        if not ok:
                            failed.append(dict(dataset=dataset,condition=condition,method=m,frame_id=int(i),error=str(error)))
        h = np.concatenate(h_ratios)
        robust_cap = np.concatenate(values[ROBUST]['cap'])
        robust_ok = np.concatenate(values[ROBUST]['ok'])
        diagnostics.append(dict(dataset=dataset,condition=condition,n_frames=len(h),n_channels=h.size,
             h_ratio_median=float(np.median(h)),h_ratio_p95=float(np.quantile(h,.95)),
             newton_cap_channels=int(robust_cap.sum()),newton_cap_frames=int(robust_cap.any(axis=1).sum()),
             newton_failed_frames=int((~robust_ok).sum())))
        for method in METHODS:
            v = {k: np.concatenate(a) for k,a in values[method].items()}
            ok = v['ok']
            for fs, i in [('comp',0),('comp+log_scale',1)]:
                a = v['drift'][ok,i]
                stability.append(dict(dataset=dataset,condition=condition,method=method,measure='feature_drift',
                  feature_set=fs,slot='all',n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95))))
            for i, label in enumerate(slot_labels(method,length)+['whole_stack']):
                a = v['l2'][ok,:,i].ravel()
                stability.append(dict(dataset=dataset,condition=condition,method=method,measure='relative_l2',
                  feature_set='components',slot=label,n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95))))
            costs.append(dict(dataset=dataset,condition=condition,method=method,stage='decomposition_features_stability',
              n_frames=len(ok),cpu_seconds=float(v['cpu'].sum()),frame_wall_seconds=float(v['seconds'].sum()),
              cpu_s_per_frame=float(v['cpu'].mean()),failed_frames=int((~ok).sum()),
              cap_channels=int(v['cap'].sum())))
    return pd.DataFrame(stability), pd.DataFrame(costs), pd.DataFrame(diagnostics), pd.DataFrame(failed,
        columns=['dataset','condition','method','frame_id','error'])


def readings(tests):
    rows=[]
    for dataset in DATASETS:
        primary=tests[(tests.dataset==dataset)&(tests.family=='Primary')&tests.condition.str.endswith('_0.04')]
        gauss=tests[(tests.dataset==dataset)&(tests.family=='S2')&(tests.contrast=='DiD')&(tests.condition=='gauss_0.04')]
        g=float(gauss.iloc[0].mean_diff)
        for _,r in primary.iterrows():
            rows.append(dict(dataset=dataset,condition=r.condition,did=r.mean_diff,gauss_did=g,
              positive=bool(r.mean_diff>0),holm_significant=bool(r.p_nb_holm<.05),
              gauss_smaller=bool(g<r.mean_diff),helps=bool(r.mean_diff>0 and r.p_nb_holm<.05 and g<r.mean_diff),
              status=r.status))
    return pd.DataFrame(rows)


def csv_new(path, table):
    table.to_csv(path,index=False,mode='x')


def figures(out, scores, tests, stability):
    folder=out/'figures'
    folder.mkdir()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for classifier, fs, filename in (
        ('classical','comp','ba_curves'),
        ('classical','comp+log_scale','ba_curves_classical_with_scale'),
        ('lightgbm','comp','ba_curves_lightgbm_comp'),
        ('lightgbm','comp+log_scale','ba_curves_lightgbm_with_scale'),
    ):
        fig, axes=plt.subplots(4,3,figsize=(13,12),sharex=True,sharey=True)
        for row,dataset in enumerate(DATASETS):
            for col,kind in enumerate(KINDS):
                ax=axes[row,col]
                for method in METHODS:
                    clf=CLASSICAL[dataset] if classifier=='classical' else classifier
                    cell=scores[(scores.dataset==dataset)&(scores.scenario=='shift')&
                        (scores.feature_set==fs)&(scores.classifier==clf)&(scores.method==method)]
                    cs=['clean']+[f'{kind}_{p:g}' for p in LEVELS]
                    means=[cell[cell.condition==c].balanced_accuracy.mean()*100 for c in cs]
                    ax.plot([0,*LEVELS],means,'o-',label=NAMES[method],color=COLORS[method],ms=3)
                ax.set_title(f'{LABELS[dataset]} | {kind}')
                ax.set_ylim(0,100)
                ax.grid(alpha=.2)
                if col==0: ax.set_ylabel('Balanced accuracy (%)')
                if row==3: ax.set_xlabel('Contamination p')
        axes[0,0].legend(ncol=2,fontsize=8)
        label='Fixed classical model' if classifier=='classical' else 'LightGBM'
        fig.suptitle(f'Clean-trained shift, {label}, {fs} features\n* Synthetic added after pre-registration at the user\'s request; descriptive fold means',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95])
        fig.savefig(folder/(filename+'.png'),dpi=180)
        if filename=='ba_curves': fig.savefig(folder/(filename+'.pdf'))
        plt.close(fig)
    p=tests[tests.family=='Primary']
    fig,axs=plt.subplots(1,4,figsize=(15,5.5),sharey=True,sharex=True)
    for ax,dataset in zip(axs,DATASETS):
        cell=p[p.dataset==dataset]
        for i,(_,r) in enumerate(cell.iterrows()):
            color='#2166ac' if r.condition.startswith('impulse1') else '#b2182b'
            ax.errorbar(100*r.mean_diff,i,xerr=[[100*(r.mean_diff-r.ci_low)],[100*(r.ci_high-r.mean_diff)]],
                        fmt='o',color=color,capsize=3,ms=5)
            if r.p_nb_holm<.05: ax.text(100*r.ci_high+.2,i,'*',va='center')
        ax.axvline(0,color='.5',lw=1);ax.set_title(LABELS[dataset]);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('DiD in BA percentage points')
        ax.set_yticks(range(len(cell)),cell.condition.tolist())
    axs[0].invert_yaxis()
    fig.suptitle('IRMF minus linear IMF, clean-trained shift, classical, comp\n95% corrected CIs; * Holm p < .05. Real-data family n=18; synthetic added family n=87')
    fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(folder/'primary_did_forest.png',dpi=180)
    fig.savefig(folder/'primary_did_forest.pdf');plt.close(fig)
    # Method stability shown against condition severity for all four datasets.
    fig,axs=plt.subplots(1,4,figsize=(14,3.6),sharey=True)
    for ax,d in zip(axs,DATASETS):
        for m in METHODS:
            cell=stability[(stability.dataset==d)&(stability.method==m)&(stability.measure=='feature_drift')&
               (stability.feature_set=='comp')&stability.condition.str.startswith('impulse2')]
            ax.plot(LEVELS,cell.set_index('condition').loc[[f'impulse2_{p:g}' for p in LEVELS],'median'],
                    'o-',label=NAMES[m],color=COLORS[m],ms=3)
        ax.set_title(LABELS[d]);ax.set_xlabel('Two-sided impulse p');ax.grid(alpha=.2)
    axs[0].set_ylabel('Median standardized feature drift');axs[-1].legend(fontsize=8)
    fig.suptitle("* Synthetic, added after pre-registration at the user's request", fontsize=10)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(folder/'feature_drift.png',dpi=180);plt.close(fig)
    # Prespecified first pilot ID, channel zero, no selection by model performance.
    fig,axs=plt.subplots(4,3,figsize=(13,10))
    for row,d in enumerate(DATASETS):
        manifest=json.loads((root(d)/'manifest_v2.json').read_text())
        frame_id=manifest['subset_ids'][0]
        frames=np.load(data_root()/d/'prepared/frames.npy',mmap_mode='r',allow_pickle=False)
        clean_raw,_=normalize_frame(frames[frame_id,0])
        for col,m in enumerate(('raw',LINEAR,ROBUST)):
            for path in (root(d)/'figures/impulse2_0.04'/m).glob('part_*.npz'):
                with np.load(path,allow_pickle=False) as z:
                    loc=np.flatnonzero(z['frame_id']==frame_id)
                    if len(loc):
                        contaminated=z['stacks'][loc[0],0,0].copy();break
            if m=='raw': clean=clean_raw
            else:
                clean=np.load(data_root()/d/'decomp'/f'{m}.npy',mmap_mode='r',allow_pickle=False)[frame_id,0,0]
            ax=axs[row,col];ax.plot(clean,lw=.7,color='.4',label='Clean')
            ax.plot(contaminated,lw=.7,alpha=.75,color=COLORS[m] if m!='raw' else '#d88b22',label='Contaminated')
            ax.set_title(f'{LABELS[d]} frame {frame_id} | {NAMES[m]}'+(' w5' if col else ' normalized'))
            if row==3: ax.set_xlabel('Sample')
    axs[0,0].legend(fontsize=8)
    fig.suptitle("Fixed-subset examples, channel 0, two-sided impulses at p=.04\n* Synthetic, added after pre-registration at the user's request")
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(folder/'component_examples.png',dpi=160);plt.close(fig)


def linked_primary_table(tests):
    lines=['| Dataset | Condition | DiD, pp | 95% corrected CI, pp | Holm p |',
           '|---|---|---:|---|---:|']
    selected=tests[tests.family=='Primary']
    for i,(_,r) in enumerate(selected.iterrows(),2):
        link=f'primary_tests.csv#L{i}'
        lines.append(f'| {LABELS[r.dataset]} | {r.condition} | [{100*r.mean_diff:+.3f}]({link}) | '
          f'[{100*r.ci_low:+.3f}, {100*r.ci_high:+.3f}]({link}) | [{r.p_nb_holm:.5g}]({link}) |')
    return '\n'.join(lines)


def render_report(out,scores,tests,stability,costs,diagnostics,fit_costs,reading,wall):
    primary=tests[tests.family=='Primary']
    verifications=[json.loads((root(d)/VERIFICATION).read_text()) for d in DATASETS]
    exact_checks=sum(r['exact'] for v in verifications for r in v['rows'])
    total_checks=sum(len(v['rows']) for v in verifications)
    restart_notes=[]
    parallel_path=root('dronerf')/'runs/parallel_schedule.json'
    parallel=json.loads(parallel_path.read_text())
    for dataset in DATASETS:
        for path in sorted((root(dataset)/'runs/restarts').glob('*.json')):
            record=json.loads(path.read_text())
            restart_notes.append(
                f'- A provider interruption left the {LABELS[dataset]} batch attached to an old '
                f'pipe. After {record["completed_fit_checkpoints"]} fit checkpoints, a detached '
                'watcher stopped its process group and restarted the batch with file-backed '
                f'output. The completed checkpoints retained their SHA-256 hashes. '
                f'The interruption added {record["classification_segment_wall_seconds"]:.2f} '
                f'wall s and {record["uncheckpointed_worker_cpu_seconds"]:.2f} uncheckpointed '
                f'worker CPU s; both are included in the cost section. '
                f'The file-backed [batch log]({record["detached"]["log"]}) is under the '
                f'dataset robustness directory. [Restart audit]({path}).')
    lines=['# Impulsive contamination and classification robustness','',
       '## Question and design','',
       'The experiment asks whether injected impulses improve the relative classification '
       'performance of converged IRMF against linear IMF. A positive difference-in-differences '
       'means the IRMF minus linear IMF gap becomes more favorable under contamination. '
       'It does not mean contaminated accuracy exceeds clean accuracy.','',
       'The [pre-registration](../../../experiments/classification-emd-irmf/robustness/PROTOCOL_robustness.md) '
       'was committed as `d44a287` before any contaminated classification result. The implementation '
       'and dated synthetic amendment were committed as `c2777d7`. Four inputs see the same '
       'prepared frames and common random numbers. The real tasks are DroneRF 10class, '
       'BirdCLEF 2023 species and ESC-50 animals, with synthetic signal added as specified below. '
       'The 60 GHz radar is excluded. A=10 and p is .002, .01 or .04. One-sided and two-sided '
       'exponential spikes share positions/magnitudes; Gaussian noise matches their expected '
       'squared-noise power. Injection precedes the shared median/MAD normalization.','',
       'The existing grouped five-fold, three-seed splits are preserved. The fixed 100 pilot '
       'frames per dataset are excluded from both training and testing in every condition. '
       'Analysis populations before solver exclusions are synthetic 1,400, DroneRF 11,900, '
       'BirdCLEF 11,900 and ESC-50 5,000. Synthetic was run end to end first. Clean cache '
       f'verification passed for all four methods on 50 frames per dataset '
       f'({exact_checks}/{total_checks} exactly equal, all others within float32 rounding). '
       '[Verification audit](validation.json).','',
       'Classical models reuse the exact shared runner pipelines: linear SVM on DroneRF/ESC-50, '
       'logistic regression on BirdCLEF/synthetic, with grouped inner selection over the shared '
       'C grids. LightGBM reuses the shared 500-round cap, early stopping and grouped training-only '
       'validation. Component-only features are primary; both classifiers also use comp+log_scale. '
       'Shift fits one clean model per split/input/feature set/classifier and predicts all ten '
       'conditions. Matched classical fits the three p=.04 conditions. All scaling, imputation '
       'and selection use training rows only. [Per-split scores](per_split_scores.csv), '
       '[fit costs and selected parameters via fit keys](fit_costs.csv).','',
       '## Primary results','',
       'The confirmatory family has the 18 real-data impulse DiDs below, with two-sided '
       'Nadeau-Bengio corrected t-tests, Holm correction across those 18, and marginal 95% '
       'corrected intervals. Wilcoxon and its own Holm correction are saved as secondary '
       'tests. Corrected variances use the observed per-split test/train frame-count ratios. '
       'All values are fold-paired. [Full tests](paired_tests.csv), [paired DiD vectors](paired_vectors.csv).','',
       'Synthetic* is "synthetic, added after pre-registration at the user\'s request". Its six '
       'primary counterparts appear alongside the real datasets, but its Holm correction '
       'spans all 87 synthetic primary/S1-S5 tests in a separate family. It changes none of '
       'the pre-registered real-data families.','', linked_primary_table(tests),'',
       '![Primary DiDs](figures/primary_did_forest.png)','',
       'The pre-specified reading requires a positive Holm-significant impulse DiD at p=.04 '
       'and a smaller matched-power Gaussian DiD. The following rows apply that rule literally. '
       'Synthetic has amended status, so its result is not confirmatory. [Reading CSV](prespecified_reading.csv).','']
    for d in DATASETS:
        cell=reading[reading.dataset==d]
        success=cell[cell.helps].condition.tolist()
        g=cell.iloc[0].gauss_did*100
        details='; '.join(f'{r.condition}: {r.did*100:+.3f} pp, Holm p={primary[(primary.dataset==d)&(primary.condition==r.condition)].iloc[0].p_nb_holm:.5g}' for _,r in cell.iterrows())
        lines.append(f'- {LABELS[d]}: rule '+('met for '+', '.join(success) if success else 'not met')+f'. {details}. Gaussian DiD at .04 is {g:+.3f} pp.')
    lines += ['', '## Secondary results, exploratory','',
       'Each real-data secondary family has its own Holm correction: S1=18, S2=27, S3=54, '
       'S4=108 and S5=36. Synthetic remains in its separate 87-test family. S1 compares '
       'IRMF with EMD; S2 tests Gaussian DiDs and impulse-minus-Gaussian DiDs; S3 uses '
       'LightGBM; S4 adds log_scale; S5 trains and tests with matched contamination relative '
       'to the clean-trained/clean-tested baseline. [All tests](paired_tests.csv).','',
       '| Dataset | Family | Tests | Positive Holm p < .05 | Negative Holm p < .05 | Mean DiD range, pp |',
       '|---|---|---:|---:|---:|---|']
    for d in DATASETS:
        for family in ('S1','S2','S3','S4','S5'):
            c=tests[(tests.dataset==d)&(tests.family==family)]
            pos=((c.mean_diff>0)&(c.p_nb_holm<.05)).sum();neg=((c.mean_diff<0)&(c.p_nb_holm<.05)).sum()
            lines.append(f'| {LABELS[d]} | {family} | {len(c)} | {pos} | {neg} | {100*c.mean_diff.min():+.2f} to {100*c.mean_diff.max():+.2f} |')
    lines += ['', 'Selected p=.04 contrasts below make the secondary families easier to inspect. '
       'They were selected by the pre-specified severity, not by statistical significance. '
       '[Source tests](paired_tests.csv).','',
       '| Dataset | Contrast | DiD, pp | 95% corrected CI, pp | Holm p |',
       '|---|---|---:|---|---:|']
    choices=[('S1','impulse2_0.04','comp','shift','emd','DiD',None),
             ('S2','impulse2_0.04','comp','shift',LINEAR,'impulse_minus_gauss_DiD',None),
             ('S3','impulse2_0.04','comp','shift',LINEAR,'DiD','lightgbm'),
             ('S4','impulse2_0.04','comp+log_scale','shift',LINEAR,'DiD','classical'),
             ('S5','impulse2_0.04','comp','matched',LINEAR,'DiD',None)]
    for d in DATASETS:
        for fam,c,fs,sc,comp,contrast,clf in choices:
            a=tests[(tests.dataset==d)&(tests.family==fam)&(tests.condition==c)&(tests.feature_set==fs)&
                    (tests.scenario==sc)&(tests.comparator==comp)&(tests.contrast==contrast)]
            if clf: a=a[a.classifier==(CLASSICAL[d] if clf=='classical' else clf)]
            r=a.iloc[0]
            label=f'{fam}, {r.classifier}, {NAMES[comp]}, {contrast}'
            lines.append(f'| {LABELS[d]} | {label} | {100*r.mean_diff:+.3f} | [{100*r.ci_low:+.3f}, {100*r.ci_high:+.3f}] | {r.p_nb_holm:.5g} |')
    lines += ['', '## Absolute classification performance','',
       'The curves show mean fold balanced accuracy for every input under clean training, '
       'the fixed classical family and comp features. They are descriptive. Clean models '
       'are reused across conditions. Every condition/classifier/scale variant is available '
       'in [per_split_scores.csv](per_split_scores.csv) and [score_summary.csv](score_summary.csv).','',
       '![BA curves](figures/ba_curves.png)','',
       'Additional full degradation curves: [classical with log_scale](figures/ba_curves_classical_with_scale.png), '
       '[LightGBM comp](figures/ba_curves_lightgbm_comp.png), '
       '[LightGBM with log_scale](figures/ba_curves_lightgbm_with_scale.png).','',
       '| Dataset | Input | Clean BA, % | One-sided .04 BA, % | Two-sided .04 BA, % | Gaussian .04 BA, % |',
       '|---|---|---:|---:|---:|---:|']
    for d in DATASETS:
        for m in METHODS:
            a=scores[(scores.dataset==d)&(scores.classifier==CLASSICAL[d])&(scores.feature_set=='comp')&(scores.scenario=='shift')&(scores.method==m)]
            v=[a[a.condition==c].balanced_accuracy.mean()*100 for c in ('clean','impulse1_0.04','impulse2_0.04','gauss_0.04')]
            lines.append(f'| {LABELS[d]} | {NAMES[m]} | '+' | '.join(f'{x:.3f}' for x in v)+' |')
    lines += ['', '## Representation stability','',
       'Feature drift is the mean absolute change divided by each feature\'s clean all-frame '
       'population SD, with SD=1 for constants. This descriptive standardization never '
       'enters the classifier. The relative L2 measures compare each contaminated stack '
       'with its own clean float32 cached stack, within channels, with denominator floor '
       '1e-12. Raw features use the shared float64 normalized samples. EMD slots can '
       'mode-mix; similarly numbered EMD/IMF slots are not matched frequency bands. '
       '[All conditions and slots](stability_metrics.csv).','',
       '| Dataset | Input | Feature drift median / p95, comp | Whole-stack relative L2 median / p95 |',
       '|---|---|---|---|']
    for d in DATASETS:
        for m in METHODS:
            a=stability[(stability.dataset==d)&(stability.method==m)&(stability.condition=='impulse2_0.04')]
            f=a[(a.measure=='feature_drift')&(a.feature_set=='comp')].iloc[0]
            l=a[(a.measure=='relative_l2')&(a.slot=='whole_stack')].iloc[0]
            lines.append(f'| {LABELS[d]} | {NAMES[m]} | {f["median"]:.4g} / {f.p95:.4g} | {l["median"]:.4g} / {l.p95:.4g} |')
    lines += ['', '![Feature drift](figures/feature_drift.png)','',
       'The [fixed-subset component examples](figures/component_examples.png) use the first '
       'pre-selected pilot ID per dataset, channel zero, for two-sided impulses at .04. '
       'These IDs were not selected by classification or drift results. Full stacks are '
       'retained only for the 100 fixed pilot/figure IDs per dataset/condition.','',
       '## H diagnostics, convergence and exclusions','',
       'H is recomputed per channel from the contaminated normalized frame. The ratios '
       'below are channel-level medians, not ratios of dataset-level medians. '
       '[All H ratios, retries and failures](solver_diagnostics.csv), [failed frames](failed_frames.csv).','',
       '| Dataset | Condition | Median H / clean H | p95 | Cap-100 channel retries | Final failed frames |',
       '|---|---|---:|---:|---:|---:|']
    for d in DATASETS:
        for _,r in diagnostics[(diagnostics.dataset==d)&diagnostics.condition.str.endswith('_0.04')].iterrows():
            lines.append(f'| {LABELS[d]} | {r.condition} | {r.h_ratio_median:.5g} | {r.h_ratio_p95:.5g} | {r.newton_cap_channels} | {r.newton_failed_frames} |')
    lines += ['', f'Across all full contaminated runs there were {int(diagnostics.newton_cap_channels.sum())} '
       f'cap-100 channel retries involving {int(diagnostics.newton_cap_frames.sum())} frame-condition pairs, '
       f'and {int(diagnostics.newton_failed_frames.sum())} final IRMF failed frame-condition pairs. '
       'Every accepted IRMF channel converged. The timing pilots had zero cap hits and '
       'zero failed channels. [Pilot measurements](../../../experiments/classification-emd-irmf/robustness/provenance/pilot_costs.csv).','',
       '## Compute cost and dependency audit','',
       'The real datasets overlapped after the dated scheduling change. DroneRF used six '
       'workers, BirdCLEF four and ESC-50 two, for at most 12 robustness workers '
       'total; each used one BLAS/OpenMP thread. Stage wall times below overlap, so '
       'their sum is total stage exposure rather than elapsed real-batch makespan. '
       'The table separates stage wall time from measured CPU summed over worker fits. '
       'Feature CPU includes decomposition, feature extraction and stack/drift comparison '
       'inside each method timer. It excludes worker initialization, I/O and the shared '
       'H-ratio calculation outside those timers. Classifier CPU includes fitting and '
       'prediction/scoring in the worker. No historical clean decomposition CPU is charged '
       'again. [Feature costs](costs.csv), [individual fit costs](fit_costs.csv), '
       '[stage wall times](wall_times.csv).','',
       '| Dataset | Feature wall, s | Classification wall, s | Feature measured CPU, s | Classifier measured CPU, s |',
       '|---|---:|---:|---:|---:|']
    for d in DATASETS:
        w=wall[wall.dataset==d].iloc[0]
        feature_cpu=costs[costs.dataset==d].cpu_seconds.sum()
        f=fit_costs[fit_costs.dataset==d]
        cls_cpu=f.fit_cpu_seconds.sum()+f.predict_cpu_seconds.sum()
        lines.append(f'| {LABELS[d]} | {w.feature_wall_seconds:.2f} | {w.classification_wall_seconds:.2f} | {feature_cpu:.2f} | {cls_cpu:.2f} |')
    total_wall=wall.feature_wall_seconds.sum()+wall.classification_wall_seconds.sum()
    total_cpu=costs.cpu_seconds.sum()+fit_costs.fit_cpu_seconds.sum()+fit_costs.predict_cpu_seconds.sum()
    restart_cpu=wall.uncheckpointed_worker_cpu_seconds.sum()
    lines += ['', f'Stage wall times sum to {total_wall:.2f} s; measured compute CPU sums to '
       f'{total_cpu:.2f} s. Restart records add {restart_cpu:.2f} worker CPU s from '
       'interrupted work, initialization and checkpoint publication outside completed-fit timers. '
       f'The measured total including that restart overhead is {total_cpu+restart_cpu:.2f} CPU s. '
       f'The timing pilots add {wall.pilot_wall_seconds.sum():.2f} wall s '
       f'and {wall.pilot_cpu_seconds.sum():.2f} measured CPU s. Stage totals exclude test-suite '
       'runs and report rendering. External robustness data occupy '
       f'{disk_bytes()/1e9:.3f} GB at analysis, below the 60 GB limit. '
       'The pinned pimf version is 0.4.0. Every top-level installed pimf Python file\'s '
       'SHA-256 matched the starting record immediately before final analysis. '
       '[Dependency audit](environment_audit.json). No shared module was edited.','',
       '## Deviations and scope','',
       '- On 2026-10-21, after pre-registration, the user added synthetic primary/S1-S5 '
       'tests and requested synthetic in every table/figure. They form a separate '
       '87-test Holm family and are labelled post-registration throughout.',
       '- The pre-registered timing rule applied all three permitted cuts in order. '
       'Matched LightGBM was first reduced to seed 0, then dropped, and matched classical '
       'was restricted to p=.04. All shift conditions and all 18 primary tests were '
       'retained. The initial conservative projection was 42.51 h and 18.27 h after '
       'cuts, using historical wall-time proxies and a DroneRF grid workload multiplier. '
       'Actual stage times are reported above; cuts were not reversed after seeing results.',
       '- The first clean-verification implementation rounded the raw control to '
       'float32 before feature extraction. This caused five of 50 synthetic checks '
       'to exceed one float32 rounding unit. It was corrected to the unchanged shared '
       'float64 raw path before any contaminated feature or classification result. '
       'The failed audit and its manifest remain preserved externally. Active provenance '
       'uses manifest_v2.json and clean_verification_v2.json. Numeric feature chunks '
       'contain all four methods in one file per frame block, rather than separate files '
       f'per method. {exact_checks}/{total_checks} active clean checks were exact; all '
       'passed the pre-specified float32-rounding criterion, so cached clean features '
       'were reused without recomputation.',
       '- An initial multiprocessing launch from stdin failed before computation. '
       'All successful runs use importable modules. No checkpoint or result was replaced.',
       *restart_notes,
       f'- On {parallel["utc"]}, at the user\'s request, BirdCLEF and ESC-50 were '
       'started concurrently with DroneRF to bring their results forward. The serial '
       f'launcher (PID {parallel["prior_serial_launcher_pid"]}) was stopped while its '
       f'DroneRF child (PID {parallel["preserved_dronerf_runner_pid"]}) continued. '
       'The old report waiter was replaced with one that checks each dataset completion. '
       'Dataset-specific locks prevent concurrent runs on the same dataset. The '
       'allocation was DroneRF six, BirdCLEF four and ESC-50 two single-thread '
       'workers. Within each dataset, all feature conditions precede shift classical '
       'fits, which precede LightGBM and matched fits. Only scheduling changed; '
       f'conditions, features, classifiers and splits did not. [Run record]({parallel_path}).',
       '- The shared current classical grids are used as requested. In particular, '
       'DroneRF uses the full shared SVM C grid, whereas the historical 10class report '
       'used fixed C=.1. The new clean scores are therefore the baselines for every '
       'new DiD, rather than historical scores.',
       '- The 100 pilot IDs per dataset are removed from both fitting and testing '
       'to implement the explicit pilot-frame exclusion. Their descriptive representation '
       'metrics and example stacks are retained, but they contribute to no classification '
       'test. Existing group partitions and split IDs remain unchanged.',
       '', 'The experiment is conditional on the prepared populations and injected noise '
       'model. It does not identify naturally occurring impulses, establish unseen-session '
       'DroneRF performance, or show robustness to other corruption mechanisms. Positive '
       'DiDs can coexist with lower absolute IRMF BA. Absolute degradation curves and '
       'Gaussian controls are needed to read the relative effects.','',
       '## Reproduction','',
       'From the checkout root, enter the study directory and use the pinned environment. '
       'Completed immutable checkpoints resume; existing output artifacts are never overwritten.','',
       '```bash', 'cd experiments/classification-emd-irmf',
       'export IMF_CLS_DATA=/Users/mikhail/Datasets/imf-classification',
       'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
       'export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1',
       'PY=../../.venv/bin/python', '$PY -m pytest -q tests robustness/tests',
       '$PY -m robustness.run --dataset synthetic --workers 6',
       "$PY -c 'from robustness.analysis import smoke; smoke()'",
       '$PY -m robustness.run_locked --dataset dronerf --workers 6',
       '$PY -m robustness.run_locked --dataset birdclef2023 --workers 4',
       '$PY -m robustness.run_locked --dataset esc50_animals --workers 2',
       '$PY -m robustness.analyse', '```','',
       'The timing pilot commands and protocol details are in '
       '[robustness/README.md](../../../experiments/classification-emd-irmf/robustness/README.md). '
       'All full data, numeric checkpoints and predictions remain under each dataset\'s '
       'external robustness directory. CSVs retain fractions; report tables and forests '
       'multiply score differences by 100 to show percentage points.']
    with (out/'REPORT.md').open('x') as f: f.write('\n'.join(lines)+'\n')
