"""A standalone, filterable display of the registered primary result tables."""

import json


def write_overview(destination, tests, best, reading, totals):
    primary = tests[(tests.variant == "classical/comp") & (tests.metric == "balanced_accuracy")]
    fields = ["dataset", "k", "contrast", "mean_diff", "ci_low", "ci_high", "p_nb_holm"]
    payload = {
        "a": primary[primary.family == "A"][fields].to_dict("records"),
        "c": primary[primary.family == "C"][fields].to_dict("records"),
        "best": best[["dataset", "classifier", "feature_set", "family", "k",
                      "balanced_accuracy", "n_features"]].to_dict("records"),
        "reading": reading.to_dict("records"),
        "wall_hours": totals["runner_wall_seconds"] / 3600,
        "cpu_hours": totals["measured_cpu_seconds"] / 3600,
        "calendar_hours": totals["elapsed_since_first_pilot_seconds"] / 3600,
    }
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    document = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Component-count results</title>
<style>
body{margin:0;color:var(--foreground,#20252b);font-family:var(--font-sans,system-ui);font-size:14px;line-height:1.5}
h2{font-size:20px;margin:24px 0 8px}h3{font-size:16px;margin:20px 0 6px}
p{margin:8px 0 14px}.muted{color:var(--muted-foreground,#5e6670)}
select{font:inherit;color:inherit;background:var(--background,white);border:1px solid var(--border,#ccd2d8);border-radius:6px;padding:6px 10px;margin-left:8px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--border,#d9dfe4);padding:7px 5px}
th{font-weight:650}.number{text-align:right;white-space:nowrap}.significant{font-weight:750;color:var(--foreground,#20252b)}
.metrics{display:flex;flex-wrap:wrap;gap:20px;margin:18px 0}.metric strong{font-size:22px;display:block}
.metric span{font-size:12px;color:var(--muted-foreground,#5e6670)}
@media(max-width:500px){table{font-size:11px}th,td{padding:6px 3px}.metrics{gap:14px}}
</style></head><body>
<h2>Component count and classification</h2>
<p>Primary tables use fixed classical models, <b>comp</b> features and 15 paired grouped splits per input.
The best-K table also includes LightGBM and <b>comp+log_scale</b>.
K counts extracted components; each input also has one residual.</p>
<label>Dataset<select id="dataset"><option value="all">All four datasets</option></select></label>
<div id="metrics" class="metrics"></div>
<h3>Pre-specified matched-count reading</h3>
<p class="muted">Both IRMF − EMD and linear IMF − EMD must be positive and Holm-significant at K=5 and K=6.</p>
<div id="reading" class="scroll"></div>
<h3>Family A: all 48 matched-count tests</h3>
<p class="muted">Estimates and marginal corrected 95% CIs are BA percentage points. Two-sided Nadeau–Bengio p values have Holm correction across all 48 tests. Bold p values are below 0.05.</p>
<div id="a" class="scroll"></div>
<h3>Best observed K</h3>
<p class="muted">Among each method's registered count grid and study default. These means are descriptive; K was selected on the same outer folds. Truncations are excluded.</p>
<div id="best" class="scroll"></div>
<h3>Family C: default-schedule truncation</h3>
<p class="muted">Truncation keeps the coarsest stages and folds all finer content into the residual. Holm spans eight tests per dataset. GD60 k_max=8 is the full pimf-default IRMF input.</p>
<div id="c" class="scroll"></div>
<p class="muted">CPU totals include model fits, block decomposition/feature work and native timing fits. They exclude normalization, input-array preparation, direct pilot verification, JIT/startup, tests and orchestration. Timed phase wall excludes input-array preparation; calendar elapsed includes work between phases. Full scores, secondary tests, padding, energies, costs and reproduction commands are in REPORT.md and its CSVs.</p>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const labels={synthetic:'Synthetic',dronerf:'DroneRF',birdclef2023:'BirdCLEF',esc50_animals:'ESC-50'};
const methods={linear:'Linear IMF',irmf:'IRMF',emd:'EMD'};
const selector=document.getElementById('dataset');
for(const [value,label] of Object.entries(labels)){const option=document.createElement('option');option.value=value;option.textContent=label;selector.append(option)}
const signed=v=>(v>=0?'+':'−')+Math.abs(v*100).toFixed(3);
const probability=v=>v===0?'0':v<.001?v.toExponential(3):v.toFixed(4);
function table(id,headers,rows){
 const host=document.getElementById(id);host.replaceChildren();
 const element=document.createElement('table'),head=document.createElement('thead'),line=document.createElement('tr');
 for(const header of headers){const cell=document.createElement('th');cell.textContent=header;line.append(cell)}head.append(line);element.append(head);
 const body=document.createElement('tbody');
 for(const row of rows){const tr=document.createElement('tr');for(const item of row){const td=document.createElement('td');td.textContent=item.text??item;if(item.cls)td.className=item.cls;tr.append(td)}body.append(tr)}
 element.append(body);host.append(element);
}
function filtered(rows){return rows.filter(row=>selector.value==='all'||row.dataset===selector.value)}
function comparisons(id){table(id,['Dataset','K','Contrast','Estimate, pp','95% CI, pp','Holm p'],filtered(data[id]).map(r=>[
 labels[r.dataset],r.k,r.contrast,{text:signed(r.mean_diff),cls:'number'},
 {text:'['+signed(r.ci_low)+', '+signed(r.ci_high)+']',cls:'number'},
 {text:probability(r.p_nb_holm),cls:'number'+(r.p_nb_holm<.05?' significant':'')}]));}
function draw(){
 table('reading',['Dataset','Criterion','Positive and significant'],filtered(data.reading).map(r=>[
 labels[r.dataset],r.survives_matched_counts?'Met':'Not met',r.positive_and_holm_significant+'/4']));
 comparisons('a');comparisons('c');
 table('best',['Dataset','Classifier','Feature set','Method','K','Mean BA, %','Features'],filtered(data.best).map(r=>[
 labels[r.dataset],{logreg:'Logistic',linsvm:'SVM',lightgbm:'LightGBM'}[r.classifier],r.feature_set,
 methods[r.family],r.k,{text:(r.balanced_accuracy*100).toFixed(3),cls:'number'},r.n_features]));
}
for(const [key,label] of [['wall_hours','Timed phase wall hours'],['cpu_hours','Measured CPU hours'],['calendar_hours','Calendar elapsed hours']]){
 const box=document.createElement('div');box.className='metric';const value=document.createElement('strong');value.textContent=data[key].toFixed(3);const caption=document.createElement('span');caption.textContent=label;box.append(value,caption);document.getElementById('metrics').append(box);
}
selector.addEventListener('change',draw);draw();
</script></body></html>'''
    with (destination / "overview.html").open("x") as handle:
        handle.write(document.replace("PAYLOAD", encoded))
