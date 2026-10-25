#!/usr/bin/env python3
"""Build the compact, self-contained results browser and static paired plot."""

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emdirmf.paths import output_dir

OUT = output_dir("birdclef2023", "ml")
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
NAMES = ["Raw", "EMD", "Linear IMF", "IRMF", "Converged IRMF"]
MODELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "extratrees": "ExtraTrees",
          "minirocket": "MiniRocket", "multirocket": "MultiRocket", "compact_resnet": "1-D CNN",
          "spectrogram_cnn": "2-D STFT CNN"}
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def main():
    payload = {}
    for task in ("species", "passerine"):
        if not (OUT / f"all_folds_{task}.csv").exists():
            continue
        folds = pd.read_csv(OUT / f"all_folds_{task}.csv")
        recorded = pd.read_csv(OUT / f"recording_pooled_seeds_{task}.csv")
        recorded = recorded[recorded.complete_seed]
        scores = []
        for unit, table in (("frame", folds), ("recording", recorded)):
            for key, cell in table.groupby(["method", "classifier", "feature_set"]):
                n_folds = len(folds[(folds.method == key[0]) & (folds.classifier == key[1]) &
                                    (folds.feature_set == key[2])])
                scores.append(dict(zip(["method", "classifier", "feature_set"], key),
                                   unit=unit, n_folds=n_folds,
                                   **{m: float(cell[m].mean()) for m in METRICS}))
        reference = OUT / "stft_reference"
        if (reference / f"recording_pooled_seeds_{task}.csv").exists():
            for unit, filename in (("frame", "results_folds"), ("recording", "recording_pooled_seeds")):
                table = pd.read_csv(reference / f"{filename}_{task}.csv")
                frame_table = pd.read_csv(reference / f"results_folds_{task}.csv")
                for side, cell in table.groupby("feature_set"):
                    scores.append({"method": "raw", "classifier": "spectrogram_cnn", "feature_set": side,
                                   "unit": unit, "n_folds": int((frame_table.feature_set == side).sum()),
                                   **{m: float(cell[m].mean()) for m in METRICS}})
        pairs = pd.read_csv(OUT / f"paired_all_metrics_{task}.csv")
        pairs = pairs[(pairs.metric == "balanced_accuracy") &
                      (pairs.method_a.isin(METHODS[3:]) | pairs.method_b.isin(METHODS[3:]))]
        pairs = pairs[["classifier", "feature_set", "method_a", "method_b",
                       "mean_diff", "ci_low", "ci_high", "p_nb_holm"]].round(7)
        payload[task] = {"scores": scores, "pairs": pairs.replace({np.nan: None}).to_dict("records")}
    html = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font-family:var(--font-sans,system-ui);font-size:14px;color:var(--foreground,#182338);margin:0}p{line-height:1.5}small,.muted{color:var(--muted-foreground,#627087)}.controls{display:flex;gap:9px;flex-wrap:wrap;margin:16px 0}select{background:var(--card,#fff);color:inherit;border:1px solid var(--border,#cbd2db);border-radius:6px;padding:7px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{text-align:right;padding:9px 5px;border-bottom:1px solid var(--border,#dce2e9);white-space:nowrap}td:first-child,th:first-child{text-align:left}th{font-size:12px;color:var(--muted-foreground,#627087);font-weight:500}.winner{color:var(--accent,#246257);font-weight:700}h3{font-size:16px;margin:25px 0 8px}svg{width:100%;min-width:650px;height:300px}.foot{font-size:12px;margin-top:12px}.chart-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
</style></head><body>
<p><strong>BirdCLEF2023</strong> · 12,000 frames · recordist-grouped validation<br><small id="caption"></small></p>
<div class="controls"><select id="task" aria-label="Task"></select><select id="metric" aria-label="Metric"><option value="balanced_accuracy">Balanced accuracy</option><option value="macro_f1">Macro-F1</option><option value="accuracy">Accuracy</option></select><select id="unit" aria-label="Evaluation unit"><option value="frame">Frames</option><option value="recording">Recordings, mean confidence</option></select></div>
<div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p class="foot">Values are mean percentages, without / with log_scale. Frame scores average folds; recording scores pool out-of-fold recordings per seed. Counts describe training folds. Tree inputs have 20/180/360 component features for raw/EMD/IMF; log_scale adds one. Bold cells mark the largest mean within a model; model rankings are descriptive. The 2-D STFT CNN is a raw-only reference.</p>
<h3>Paired balanced-accuracy differences, frames</h3><div class="chart-controls"><select id="target" aria-label="Target method"><option value="irmf_h2_a1.41">IRMF</option><option value="irmf_h2_a1.41_conv">Converged IRMF</option></select><span>versus</span><select id="reference" aria-label="Reference method"><option value="emd">EMD</option><option value="imf_linear_a1.41">Linear IMF</option><option value="raw">Raw</option></select><select id="features" aria-label="Log-scale variant"><option value="comp">Without log_scale</option><option value="comp+log_scale">With log_scale</option></select></div><div class="scroll"><svg id="plot" viewBox="0 0 720 300" role="img" aria-label="Corrected paired balanced accuracy differences"></svg></div>
<p class="foot">95% Nadeau-Bengio corrected intervals. q is Holm-adjusted across all model, log-variant and method-pair comparisons in the task's balanced-accuracy table. Wilcoxon and all secondary metrics are in the CSV files. ROCKET recording confidence uses an uncalibrated softmax of decision scores; other runners use native model probabilities.</p>
<script>const DATA=__DATA__,METHODS=__METHODS__,NAMES=__NAMES__,MODELS=__MODELS__;
const byId=id=>document.getElementById(id),pct=x=>(100*x).toFixed(1),signed=x=>(x>=0?'+':'')+(100*x).toFixed(2),q=x=>x<.001?x.toExponential(1):x.toFixed(3);
byId('task').innerHTML=Object.keys(DATA).map(t=>`<option value="${t}">${t==='species'?'Species, primary':'Passerine, secondary'}</option>`).join('');
function draw(){const task=byId('task').value,metric=byId('metric').value,unit=byId('unit').value,d=DATA[task],models=Object.keys(MODELS).filter(m=>d.scores.some(s=>s.classifier===m));
byId('caption').textContent=task==='species'?'16 species · 93 ms per frame · paired methods on identical folds':'Binary passerine task · same recordist exclusion';
byId('head').innerHTML='<tr><th>Method</th>'+models.map(m=>{const n=d.scores.find(s=>s.classifier===m).n_folds;return `<th>${MODELS[m]}<br><small>${n} folds</small></th>`}).join('')+'</tr>';
byId('body').innerHTML=METHODS.map((method,i)=>'<tr><td>'+NAMES[i]+'</td>'+models.map(model=>{const values=['comp','comp+log_scale'].map(f=>{const s=d.scores.find(s=>s.method===method&&s.classifier===model&&s.feature_set===f&&s.unit===unit);if(!s)return '–';const maximum=Math.max(...d.scores.filter(x=>x.classifier===model&&x.feature_set===f&&x.unit===unit).map(x=>x[metric]));return `<span class="${s[metric]===maximum&&model!=='spectrogram_cnn'?'winner':''}">${pct(s[metric])}</span>`});return '<td>'+values.join(' / ')+'</td>'}).join('')+'</tr>').join('');
const target=byId('target').value,reference=byId('reference').value,features=byId('features').value,rows=models.map(model=>{const p=d.pairs.find(p=>p.classifier===model&&p.feature_set===features&&((p.method_a===target&&p.method_b===reference)||(p.method_b===target&&p.method_a===reference)));if(!p)return null;const sign=p.method_a===target?1:-1;return {model,mean:sign*p.mean_diff,low:sign===1?p.ci_low:-p.ci_high,high:sign===1?p.ci_high:-p.ci_low,q:p.p_nb_holm}}).filter(Boolean);
const left=120,right=500,limit=Math.max(.03,...rows.flatMap(r=>[Math.abs(r.low),Math.abs(r.high)]))*1.12,x=v=>left+(v+limit)/(2*limit)*(right-left),spacing=220/Math.max(rows.length,1);let svg=`<line x1="${x(0)}" y1="10" x2="${x(0)}" y2="245" stroke="var(--border,#ccd2da)"/>`;
rows.forEach((r,i)=>{const y=30+i*spacing,color=r.q<.05?'var(--accent,#21685c)':'var(--muted-foreground,#718096)';svg+=`<text x="0" y="${y+4}" font-size="13" fill="currentColor">${MODELS[r.model]}</text><line x1="${x(r.low)}" x2="${x(r.high)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="3"/><circle cx="${x(r.mean)}" cy="${y}" r="5" fill="${color}"/><text x="525" y="${y-2}" font-size="12" fill="currentColor">${signed(r.mean)} pp · q=${q(r.q)}</text><text x="525" y="${y+14}" font-size="11" fill="var(--muted-foreground,#718096)">[${signed(r.low)}, ${signed(r.high)}]</text>`});
[-limit,0,limit].forEach(v=>{svg+=`<text x="${x(v)}" y="272" text-anchor="middle" font-size="12" fill="currentColor">${signed(v)} pp</text>`});byId('plot').innerHTML=svg;
}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',draw));
let panels=0;for(const task of Object.keys(DATA)){byId('task').value=task;for(const metric of ['balanced_accuracy','macro_f1','accuracy']){byId('metric').value=metric;for(const unit of ['frame','recording']){byId('unit').value=unit;draw();if(byId('body').children.length!==5||byId('body').textContent.includes('NaN'))throw Error('Invalid score panel');panels++;}}for(const target of ['irmf_h2_a1.41','irmf_h2_a1.41_conv']){byId('target').value=target;for(const reference of ['emd','imf_linear_a1.41','raw']){byId('reference').value=reference;for(const features of ['comp','comp+log_scale']){byId('features').value=features;draw();if(byId('plot').innerHTML.includes('NaN'))throw Error('Invalid paired plot');panels++;}}}}
byId('task').value=Object.keys(DATA)[0];byId('metric').value='balanced_accuracy';byId('unit').value='frame';byId('target').value='irmf_h2_a1.41';byId('reference').value='emd';byId('features').value='comp';draw();console.log('Validated panels',panels,'methods',METHODS.length,'rows',byId('body').children.length);</script></body></html>'''
    for placeholder, value in (("__DATA__", payload), ("__METHODS__", METHODS), ("__NAMES__", NAMES), ("__MODELS__", MODELS)):
        html = html.replace(placeholder, json.dumps(value, separators=(",", ":")))
    (OUT / "report.html").write_text(html)
    pairs = pd.read_csv(OUT / "paired_all_metrics_species.csv")
    plot_models = {model: title for model, title in MODELS.items() if model in set(pairs.classifier)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, feature_set in zip(axes, ["comp", "comp+log_scale"]):
        for index, model in enumerate(plot_models):
            cell = pairs[(pairs.classifier == model) & (pairs.feature_set == feature_set) &
                         (pairs.metric == "balanced_accuracy") & (pairs.method_a == "emd") &
                         (pairs.method_b == "irmf_h2_a1.41")]
            if cell.empty:
                continue
            row = cell.iloc[0]
            delta, low, high = -100 * row.mean_diff, -100 * row.ci_high, -100 * row.ci_low
            axis.errorbar(delta, index, xerr=[[delta-low], [high-delta]], fmt="o", capsize=4,
                          color="#247364" if row.p_nb_holm < .05 else "#7c8593")
        axis.axvline(0, color="#b0b7bf", linewidth=1)
        axis.set_title("With log_scale" if feature_set.endswith("log_scale") else "Without log_scale")
        axis.set_xlabel("IRMF minus EMD balanced accuracy, pp")
        axis.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(len(plot_models)), list(plot_models.values()))
    axes[0].invert_yaxis()
    fig.suptitle("Species: corrected 95% paired intervals; green passes Holm")
    fig.tight_layout()
    fig.savefig(OUT / "paired_irmf_emd.png", dpi=180)
    plt.close(fig)
    importance = pd.read_csv(OUT / "scale_importance_species.csv")
    tree_models = ["lightgbm", "xgboost", "extratrees"]
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    for row, method in enumerate(METHODS[1:]):
        for column, model in enumerate(tree_models):
            axis = axes[row, column]
            cell = importance[(importance.method == method) & (importance.classifier == model) &
                              (importance.feature_set == "comp")].sort_values("scale_rank")
            axis.bar(np.arange(len(cell)), 100 * cell["mean"], color=["#8aa6b4", "#7d9b84", "#27786b", "#69518a"][row])
            axis.set_xticks(np.arange(len(cell)), cell.slot_label, rotation=90, fontsize=7)
            axis.set_ylabel(NAMES[row + 1] + ", %" if column == 0 else "%")
            if row == 0:
                axis.set_title(MODELS[model] + ": " + (cell.importance_type.iloc[0] if len(cell) else "pending"))
    fig.suptitle("Species: component-only importance in true fine-to-coarse order; descriptive within each model")
    fig.tight_layout()
    fig.savefig(OUT / "scale_importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
