#!/usr/bin/env python3
"""Make the self-contained, interactive result table used in the handoff."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output/classification-emd-irmf/esc50_animals/ml"
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"]
MODELS = ["lightgbm", "xgboost", "extratrees", "minirocket", "compact_resnet", "multirocket", "log_stft_cnn2d"]
TASKS = ["animal", "bird"]
FEATURES = ["comp", "comp+log_scale"]


def rows(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    scores = [[TASKS.index(row["task"]), FEATURES.index(row["feature_set"]),
               MODELS.index(row["classifier"]), METHODS.index(row["method"]), *[
                   round(100 * float(row[f"{metric}_mean"]), 3)
                   for metric in ("balanced_accuracy", "macro_f1", "accuracy")]]
              for row in rows("scores.csv")]
    pairs = [[TASKS.index(row["task"]), FEATURES.index(row["feature_set"]),
              MODELS.index(row["classifier"]), METHODS.index(row["method_a"]),
              METHODS.index(row["method_b"]), *[
                  float(row[column]) * (100 if column in ("mean_diff", "ci_low", "ci_high") else 1)
                  for column in ("mean_diff", "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm")],
              int(row["holm_family_size"])] for row in rows("paired_requested.csv")
             if row["metric"] == "balanced_accuracy"]
    costs = [[METHODS.index(row["method"]), float(row["seconds_per_frame"]), int(row["n_slots"])]
             for row in rows("decomposition_costs.csv")]
    document = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground);font:13px var(--font-sans)}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}label{display:flex;align-items:center;gap:5px}
select{color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:6px;padding:5px;font:inherit}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 6px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}th{font-weight:500;color:var(--muted-foreground);font-size:12px}
.winner{font-weight:700}.quiet{color:var(--muted-foreground);font-size:12px;line-height:1.5;margin:8px 0 16px}
h3{font-size:15px;font-weight:600;margin:20px 0 8px}.significant{color:var(--primary);font-weight:600}
.cost{max-width:440px}.negative{color:var(--destructive)}
</style></head><body>
<div class="controls"><label>Task <select id="task"><option>Animal</option><option>Bird</option></select></label>
<label>Input <select id="feature"><option>Without log scale</option><option>With log scale</option></select></label>
<label>Metric <select id="metric"><option>Balanced accuracy</option><option>Macro-F1</option><option>Accuracy</option></select></label></div>
<div class="scroll"><table id="scores"></table></div>
<p class="quiet">Percentages, means over uploader-held-out folds. Trees and MiniRocket use 15 folds. The 1-D CNN and MultiRocket use seed 0's five folds. Bold marks the largest mean in a column.</p>
<p id="spectrogram" class="quiet"></p>
<h3>Paired balanced accuracy</h3><label>Model <select id="model"></select></label>
<div class="scroll"><table id="pairs"></table></div>
<p class="quiet" id="inference"></p>
<h3>Cached decomposition cost</h3><table id="cost" class="cost"></table>
<p class="quiet">CPU seconds per frame. These are source-cache timings; all classifiers reuse the decompositions. Full metrics, clip scores, training times and validations are saved with the report.</p>
<script>
const scores=__SCORES__,pairs=__PAIRS__,costs=__COSTS__;
const method=['Raw','EMD','Linear IMF','IRMF · GD60','IRMF · converged'];
const models=['LightGBM','XGBoost','ExtraTrees','MiniRocket','1-D CNN','MultiRocket'];
const byId=id=>document.getElementById(id), fmt=value=>value.toFixed(2), sign=value=>(value>0?'+':'')+fmt(value);
const pval=value=>value<0.001?value.toExponential(1):value.toFixed(3);
byId('model').innerHTML=models.map((name,index)=>`<option value="${index}">${name}</option>`).join('');
function draw(){
 const task=byId('task').selectedIndex,feature=byId('feature').selectedIndex,metric=byId('metric').selectedIndex;
 const selected=scores.filter(row=>row[0]===task&&row[1]===feature), maximum=models.map((_,model)=>Math.max(...selected.filter(row=>row[2]===model).map(row=>row[4+metric])));
 byId('scores').innerHTML='<thead><tr><th>Method</th>'+models.map(name=>`<th>${name}</th>`).join('')+'</tr></thead><tbody>'+method.map((name,method)=>'<tr><td>'+name+'</td>'+models.map((_,model)=>{
 const row=selected.find(row=>row[2]===model&&row[3]===method);return row?`<td class="${row[4+metric]===maximum[model]?'winner':''}">${fmt(row[4+metric])}</td>`:'<td>-</td>'}).join('')+'</tr>').join('')+'</tbody>';
 const reference=selected.find(row=>row[2]===6);byId('spectrogram').textContent=reference?'Raw log-STFT 2-D CNN reference: '+fmt(reference[4+metric])+'%, five folds. Different architecture; no paired decomposition test.':'';
 const model=+byId('model').value, selectedPairs=pairs.filter(row=>row[0]===task&&row[1]===feature&&row[2]===model);
 byId('pairs').innerHTML='<thead><tr><th>A − B</th><th>Difference, pp</th><th>95% CI</th><th>t Holm p</th><th>Wilcoxon Holm p</th></tr></thead><tbody>'+selectedPairs.map(row=>`<tr><td>${method[row[3]]} − ${method[row[4]]}</td><td>${sign(row[5])}</td><td>[${sign(row[6])}, ${sign(row[7])}]</td><td class="${row[8]<.05?'significant':''}">${pval(row[8])}</td><td>${pval(row[9])}</td></tr>`).join('')+'</tbody>';
 byId('inference').textContent='Primary test: Nadeau-Bengio corrected resampled t-test. Wilcoxon is secondary. Holm covers '+(selectedPairs[0]?.[10]??100)+' tests per task and metric in this table. Corrected 95% intervals are not simultaneous intervals.';
}
byId('cost').innerHTML='<thead><tr><th>Method</th><th>CPU s/frame</th><th>Slots</th></tr></thead><tbody>'+costs.map(row=>`<tr><td>${method[row[0]]}</td><td>${row[1]===0?'0':row[1].toFixed(row[1]<.1?4:3)}</td><td>${row[2]}</td></tr>`).join('')+'</tbody>';
for(const id of ['task','feature','metric','model'])byId(id).addEventListener('change',draw);draw();
console.assert(byId('scores').querySelectorAll('tbody tr').length===5,'five method rows');
console.assert(byId('pairs').querySelectorAll('tbody tr').length===10,'ten paired contrasts');
console.log('Result overview ready',scores.length,'score cells',pairs.length,'paired rows');
</script></body></html>'''
    document = document.replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
    document = document.replace("__PAIRS__", json.dumps(pairs, separators=(",", ":")))
    document = document.replace("__COSTS__", json.dumps(costs, separators=(",", ":")))
    (OUT / "overview.html").write_text(document)
    print(OUT / "overview.html")


if __name__ == "__main__":
    main()
