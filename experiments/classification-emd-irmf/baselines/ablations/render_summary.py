"""Build a self-contained browser view of the cross-dataset paired table."""

import json

import pandas as pd

from run_ablations import OUTPUT


def main():
    folder = OUTPUT / "uas_birds_60ghz/classical_ablations"
    frame = pd.read_csv(folder / "cross_dataset_all_metrics.csv", float_precision="round_trip")
    columns = ["dataset", "classifier", "metric", "feature_set", "contrast", "mean_diff",
               "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm", "holm_family_size"]
    view = frame[columns].copy()
    for column in ("mean_diff", "ci_low", "ci_high", "p_nb_holm", "p_wilcoxon_holm"):
        view[column] = view[column].map(lambda value: float(f"{value:.8g}"))
    records = view.values.tolist()
    html = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}body{margin:0;color:var(--foreground,#20242b);font:14px var(--font-sans,system-ui)}
    h2{font-size:20px;margin:0 0 8px;font-weight:650}p{line-height:1.5;margin:8px 0;color:var(--muted-foreground,#626975)}
    .controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 8px}label{font-size:12px;color:var(--muted-foreground,#626975)}
    select{display:block;max-width:100%;padding:7px 9px;margin-top:4px;border:1px solid var(--border,#ccd1d8);border-radius:6px;color:var(--foreground,#20242b);background:var(--background,transparent);font:inherit}
    table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted-foreground,#626975);font-weight:550}td,th{padding:8px 6px;border-bottom:1px solid var(--border,#ccd1d8)}td:first-child,th:first-child{padding-left:0}td:not(:first-child){white-space:nowrap}.scroll{overflow:auto}small{color:var(--muted-foreground,#626975)}
    </style></head><body><h2>Classical decomposition ablations</h2>
    <p>Paired differences on the same saved grouped folds. Positive values favor the first method. Each dataset keeps one classifier fixed across methods.</p><p id="pending"></p>
    <div class="controls"><label>Contrast<select id="contrast"></select></label>
    <label>Features<select id="features"><option value="comp">Without log_scale</option><option value="comp+log_scale">With log_scale</option></select></label>
    <label>Metric<select id="metric"><option value="balanced_accuracy">Balanced accuracy</option><option value="macro_f1">Macro-F1</option><option value="accuracy">Accuracy</option></select></label></div>
    <div class="scroll"><table><thead><tr><th>Dataset / fixed classifier</th><th>Δ pp [95% CI]</th><th>Corrected-t<br>Holm p</th><th>Wilcoxon<br>Holm p</th></tr></thead><tbody id="rows"></tbody></table></div>
    <p id="note"></p><small>Intervals use Nadeau-Bengio corrected fold uncertainty. Pooled estimates weight observed datasets equally and propagate their corrected variances; they do not establish an effect on unseen datasets. Classifier selection from phase-2 outer scores makes these analyses exploratory.</small>
    <script>const keys=__COLUMNS__;const data=__DATA__.map(row=>Object.fromEntries(keys.map((key,i)=>[key,row[i]])));
    const order=['uas_birds_60ghz','esc50_animals','birdclef2023','dronerf','pooled'];
    const names={uas_birds_60ghz:'UAS',esc50_animals:'ESC-50 animals',birdclef2023:'BirdCLEF species',dronerf:'DroneRF 10-class',pooled:'Pooled'};
    const missing=order.filter(d=>d!=='pooled'&&!data.some(r=>r.dataset===d));document.getElementById('pending').textContent=missing.length?'Pending: '+missing.map(d=>names[d]).join(', '):'';
    const contrasts=[...new Set(data.map(r=>r.contrast))];const c=document.getElementById('contrast');
    contrasts.forEach(s=>{let o=document.createElement('option');o.value=s;o.textContent=s;c.appendChild(o)});c.value='Q3 IRMF a2-EMD';
    const fmt=x=>(x>=0?'+':'')+(100*x).toFixed(2);const p=x=>x<.001?x.toExponential(1):x.toFixed(3);
    function draw(){const metric=document.getElementById('metric').value,features=document.getElementById('features').value;
    const rows=data.filter(r=>r.metric===metric&&r.feature_set===features&&r.contrast===c.value).sort((a,b)=>order.indexOf(a.dataset)-order.indexOf(b.dataset));
    document.getElementById('rows').innerHTML=rows.map(r=>`<tr><td>${names[r.dataset]} / ${r.classifier}</td><td>${fmt(r.mean_diff)} [${fmt(r.ci_low)}, ${fmt(r.ci_high)}]</td><td>${p(r.p_nb_holm)}</td><td>${p(r.p_wilcoxon_holm)}</td></tr>`).join('');
    document.getElementById('note').textContent=`Holm correction covers ${rows[0].holm_family_size} cross-dataset cells per metric, across all contrasts and both feature sets. Wilcoxon is secondary.`;
    console.log('rendered',rows.length,'datasets; all effect intervals finite',rows.every(r=>Number.isFinite(r.ci_low)&&Number.isFinite(r.ci_high)));}
    document.querySelectorAll('select').forEach(e=>e.addEventListener('change',draw));draw();</script></body></html>'''
    html = html.replace("__DATA__", json.dumps(records, allow_nan=False))
    html = html.replace("__COLUMNS__", json.dumps(columns))
    path = folder / "overview.html"
    path.write_text(html)
    print(path)


if __name__ == "__main__":
    main()
