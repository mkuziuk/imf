#!/usr/bin/env python3
"""Build a self-contained table explorer from the audited DroneRF results."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output/classification-emd-irmf/dronerf/ml"


def write_ablations(scores, pairs):
    """Keep convergence and view checks readable alongside the full CSVs."""
    selected = pairs[(pairs.task == "10class") & (pairs.metric == "balanced_accuracy") &
                     (pairs.inference_table == "primary") &
                     (pairs.method_a == "irmf_h2_a1.41_conv") &
                     pairs.method_b.isin(["irmf_h2_a1.41", "imf_linear_a1.41"])].copy()
    selected["contrast"] = selected.method_b.map({
        "irmf_h2_a1.41": "Conv minus GD60", "imf_linear_a1.41": "Conv minus linear"})
    selected["effect [95% CI]; Holm p"] = selected.apply(lambda r:
        f"{100*r.mean_diff:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]; {r.p_nb_holm:.4g}", axis=1)
    lines = ["# Primary-task ablations", "", "Balanced accuracy; differences in percentage points. "
             "Individual Nadeau–Bengio 95% intervals, with Holm p from the complete 100-test main table.", "",
             "## Solver convergence", "", selected[["classifier", "feature_set", "contrast",
             "effect [95% CI]; Holm p"]].to_markdown(index=False), ""]
    strata = pd.read_csv(OUT / "convergence_strata.csv")
    strata = strata[strata.method.isin(["imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv"])]
    strata = strata.groupby(["classifier", "feature_set", "method", "gd60_all_converged"]).agg(
        BA=("balanced_accuracy", "mean"), n_frames=("n_frames", "first")).reset_index()
    strata["BA %"] = (100 * strata.BA).map(lambda x: f"{x:.2f}")
    lines += ["Scores on the same GD60-converged/unconverged frame populations, pooled per seed "
              "and then averaged over available seeds. The 1,997 unconverged frames and 10,003 "
              "converged frames each contain all ten classes. These are descriptive associations; "
              "solver failure does not randomly assign frames to difficulty strata.", "",
              strata.drop(columns="BA").to_markdown(index=False), "", "## H sensitivity", ""]
    sensitivity = scores[(scores.task == "10class") & (scores.classifier == "lightgbm") &
                         ~scores.runner.str.contains("lonly") &
                         scores.method.isin(["irmf_h1_a1.41_conv", "irmf_h2_a1.41_conv", "irmf_h4_a1.41_conv"])].copy()
    sensitivity["BA %"] = (100 * sensitivity.balanced_accuracy_mean).map(lambda x: f"{x:.2f}")
    lines += [sensitivity.pivot(index="method", columns="feature_set", values="BA %").to_markdown(), "",
              "The six H-sensitivity contrasts form a separate Holm table in `paired_all_metrics.csv`.", "",
              "## L-only LightGBM", "", "The same 12,000 prepared frames, folds and capacities are retained. "
              "L-only therefore includes frames accepted by the original H-only activity gate. "
              "Family selection from outer-fold means is exploratory.", ""]
    bands = pd.read_csv(OUT / "paired_bands.csv")
    bands = bands[(bands.task == "10class") & (bands.metric == "balanced_accuracy")].copy()
    bands["L-only / L+H BA %"] = bands.apply(lambda r: f"{100*r.mean_a:.2f} / {100*r.mean_b:.2f}", axis=1)
    bands["L minus L+H [95% CI]; Holm p"] = bands.apply(lambda r:
        f"{100*r.mean_diff:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]; {r.p_nb_holm:.4g}", axis=1)
    lines += [bands[["method_a", "feature_set", "L-only / L+H BA %",
                     "L minus L+H [95% CI]; Holm p"]].to_markdown(index=False), "",
              "Holm covers all ten method/scale variants in the band table. "
              "The a=2 capacity controls are in `TABLES.md` and the separate 30-test capacity table "
              "in `PAIRS.md`. All three metrics remain available in the CSV exports.", ""]
    (OUT / "ABLATIONS.md").write_text("\n".join(lines))


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    pairs = pd.read_csv(OUT / "paired_requested.csv")
    write_ablations(scores, pairs)
    scores = scores[~scores.runner.str.contains("lonly")]
    segment = pd.read_csv(OUT / "segment_scores.csv")
    segment = segment[~segment.runner.str.contains("lonly")]
    pairs = pairs[(pairs.metric == "balanced_accuracy") & (pairs.inference_table == "primary") &
                  (pairs.task == "10class") & (pairs.classifier == "lightgbm")]
    score_columns = ["task", "method", "feature_set", "classifier", "balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"]
    pair_columns = ["task", "classifier", "method_a", "method_b", "feature_set", "mean_diff", "ci_low", "ci_high", "p_nb_holm"]
    detail = pd.read_csv(OUT / "all_folds.csv")
    detail = detail[~detail.runner.str.contains("lonly")]
    folds = detail.groupby(["task", "classifier"]).split_id.nunique().rename("folds").reset_index().to_dict("records")
    payload = {"frame": scores[score_columns].to_dict("records"), "segment": segment[score_columns].to_dict("records"),
               "pairs": pairs[pair_columns].to_dict("records"), "folds": folds}
    # JSON nulls keep the document valid when a one-seed segment SD is undefined.
    data = json.dumps(payload, allow_nan=True, separators=(",", ":")).replace("NaN", "null").replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DroneRF phase-3 results</title><style>
:root{color-scheme:light dark;font-family:var(--font-sans,system-ui,sans-serif);color:var(--foreground,CanvasText)}
body{margin:0;background:transparent}p{margin:0 0 14px;line-height:1.5}h2{font-size:17px;margin:22px 0 10px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:12px 0 16px}
label{font-size:12px;color:var(--muted-foreground,#64748b)}select{display:block;margin-top:5px;padding:7px 24px 7px 9px;
background:var(--secondary,Canvas);color:inherit;border:1px solid var(--border,#d4dbe2);border-radius:6px;font:inherit}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:10px 6px;text-align:right;border-bottom:1px solid var(--border,#d4dbe2);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-weight:600}th small{display:block;font-size:10px;font-weight:400;color:var(--muted-foreground,#64748b)}
.conv{background:color-mix(in srgb,var(--accent,#2563eb) 8%,transparent)}.muted{color:var(--muted-foreground,#64748b);font-size:12px}
.effects{display:grid;gap:9px;margin-top:12px}.effect{border-left:3px solid var(--border,#d4dbe2);padding:6px 10px;font-size:13px;line-height:1.45}
.effect.positive{border-color:var(--success,#168a55)}.effect.negative{border-color:var(--destructive,#be3f3f)}.effect strong{font-weight:600}.effect small{display:block;color:var(--muted-foreground,#64748b)}
button{background:transparent;color:var(--accent,#2563eb);border:0;padding:10px 0;cursor:pointer;text-align:left}
@media(max-width:500px){th,td{padding:9px 5px}table{font-size:11px}.controls{gap:8px}select{max-width:170px}}
</style></head><body>
<p id="intro"></p>
<div class="controls"><label>Task<select id="task"></select></label>
<label>Evaluation unit<select id="level"><option value="frame">Frames</option><option value="segment">Pooled segments</option></select></label>
<label>Metric<select id="metric"><option value="balanced_accuracy">Balanced accuracy</option><option value="macro_f1">Macro-F1</option><option value="accuracy">Accuracy</option></select></label></div>
<div class="scroll"><table id="scores"></table></div>
<button id="capacity" type="button">Show the a=2 capacity controls</button>
<p class="muted" id="details"></p>
<p class="muted" id="reference"></p>
<h2>Primary 10-class frame effects on balanced accuracy</h2>
<label>LightGBM feature set<select id="scale"><option value="comp">Without log scale</option><option value="comp+log_scale">With log scale</option></select></label>
<div class="effects" id="effects"></div>
<p class="muted" style="margin-top:16px">Differences are percentage points with individual Nadeau-Bengio 95% CIs. Holm covers the complete primary table for each task and metric; Wilcoxon is secondary. Selecting a best family from outer means is exploratory.</p>
<p class="muted">Fold counts are shown per model. Trees cap at 60 boosting rounds or 64 ExtraTrees trees. MiniRocket uses 1,008 features and five head passes. CNN uses width 8, at most eight epochs and common downsampling by two. L/H are nonsimultaneous views. Segment splits still share BUI sessions.</p>
<script>const data=PAYLOAD;
const names={raw:'Raw',emd:'EMD','imf_linear_a1.41':'Linear IMF','irmf_h2_a1.41':'IRMF GD60','irmf_h2_a1.41_conv':'IRMF converged','irmf_h2_a2_conv':'IRMF converged a=2','imf_linear_a2':'Linear IMF a=2'};
const models=[['lightgbm','LGBM'],['xgboost','XGB'],['extratrees','ExtraTrees'],['minirocket','MiniRocket'],['compact_resnet','CNN']];
const methods=Object.keys(names);let capacity=true;
const el=id=>document.getElementById(id), number=v=>v==null?'—':(100*v).toFixed(2);
for(const task of ['10class','4class','2class'])if(data.frame.some(r=>r.task===task)){let option=document.createElement('option');option.value=task;option.textContent=task.replace('class',' classes');el('task').append(option)}
function draw(){const task=el('task').value,level=el('level').value,metric=el('metric').value;
 const rows=data[level].filter(r=>r.task===task),families=models.filter(([c])=>rows.some(r=>r.classifier===c));
 el('intro').textContent=`${task.replace('class','-class')} DroneRF. Each cell gives percentages without / with log scale.`;
 let table='<thead><tr><th>Input</th>'+families.map(([c,n])=>{const f=data.folds.find(r=>r.task===task&&r.classifier===c);return `<th>${n}<small>${f?f.folds:'?'} folds</small></th>`}).join('')+'</tr></thead><tbody>';
 for(const method of methods.slice(0,capacity?7:5)){table+=`<tr class="${method==='irmf_h2_a1.41_conv'?'conv':''}"><td>${names[method]}</td>`;
  for(const [classifier]of families){const value=fs=>{const r=rows.find(r=>r.method===method&&r.classifier===classifier&&r.feature_set===fs);return r?number(r[metric+'_mean']):'—'};
   table+=`<td>${value('comp')} / ${value('comp+log_scale')}</td>`}table+='</tr>'}el('scores').innerHTML=table+'</tbody>';
 el('details').textContent=level==='frame'?'Means of held-out fold scores. Every method uses the same split IDs within a model.':'Mean native probabilities per held-out segment, then score all 227 segments per seed. Five folds correspond to one seed. This is a sensitivity analysis.';
 if(capacity)el('details').textContent+=' The reduced schedule has ten slots per view; EMD has nine.';
 const reference=fs=>rows.find(r=>r.method==='raw'&&r.classifier==='log_stft_cnn2d'&&r.feature_set===fs);
 const without=reference('comp'),withScale=reference('comp+log_scale');
 el('reference').textContent=without&&withScale?`Raw log-STFT 2-D CNN reference (5 folds): ${number(without[metric+'_mean'])} / ${number(withScale[metric+'_mean'])}%. A separate architecture and representation, shown descriptively.`:'';
 el('capacity').textContent=capacity?'Hide the a=2 capacity controls':'Show the a=2 capacity controls';
 const contrasts=[['irmf_h2_a1.41','emd'],['irmf_h2_a1.41_conv','irmf_h2_a1.41'],['irmf_h2_a1.41_conv','imf_linear_a1.41']];
 el('effects').innerHTML=contrasts.map(([a,b])=>{const r=data.pairs.find(r=>r.task==='10class'&&r.classifier==='lightgbm'&&r.feature_set===el('scale').value&&r.method_a===a&&r.method_b===b);if(!r)return '';
 const signed=x=>(x>=0?'+':'')+(100*x).toFixed(2);return `<div class="effect ${r.p_nb_holm<.05?(r.mean_diff>0?'positive':'negative'):''}"><strong>${names[a]} minus ${names[b]}: ${signed(r.mean_diff)} pp</strong><small>95% CI [${signed(r.ci_low)}, ${signed(r.ci_high)}]; Holm p=${r.p_nb_holm.toPrecision(3)}</small></div>`}).join('');
 console.log('Rendered',task,level,metric,rows.length,'rows',families.length,'model families');}
for(const id of ['task','level','metric','scale'])el(id).addEventListener('change',draw);
el('capacity').addEventListener('click',()=>{capacity=!capacity;draw()});draw();
</script></body></html>'''.replace("PAYLOAD", data)
    destination = OUT / "overview.html"
    destination.write_text(html)
    print(destination)


if __name__ == "__main__":
    main()
