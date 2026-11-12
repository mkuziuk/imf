#!/usr/bin/env python3
"""Render small radar phase-3 error, power, sensitivity and HTML tables."""
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "output/classification-emd-irmf/uas_birds_60ghz/ml"
METHODS = ["raw", "emd", "imf_linear_a1.41", "irmf_h2_a1.41", "irmf_h2_a1.41_conv",
           "irmf_h2_a2_conv", "imf_linear_a2", "irmf_h1_a1.41_conv", "irmf_h4_a1.41_conv"]
LABELS = ["Raw", "EMD", "Linear", "IRMF GD60", "IRMF converged", "IRMF a=2", "Linear a=2", "IRMF H=1", "IRMF H=4"]
MODELS = ["lightgbm", "xgboost", "extratrees", "minirocket", "multirocket", "compact_resnet"]


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    errors = pd.read_csv(OUT / "error_counts.csv")
    classes = pd.read_csv(OUT / "per_class_errors.csv")
    power = pd.read_csv(OUT / "power.csv")
    pairs = pd.read_csv(OUT / "paired_requested.csv")
    record_errors = pd.read_csv(OUT / "recording_errors.csv")
    frame_records = pd.read_csv(OUT / "frame_errors_by_recording.csv")
    lines = ["# Errors on held-out poses", "",
        "Each cell is evaluated once per seed, three times overall. Denominators are 2,352 cell evaluations or 147 recording evaluations per method/model/task/feature set. Counts are correlated, not independent samples.",
        "Groups ever wrong count distinct poses across all three seeds, out of 45. Recording aggregation averages probabilities; majority-vote alternatives are in error_counts.csv and recording_errors.csv.",
        "LightGBM is fixed below. Every classifier, class and recording is retained in the CSV files.", ""]
    for task in ("bird_drone", "target3"):
        for fs in ("comp", "comp+log_scale"):
            part = classes[(classes.task == task) & (classes.feature_set == fs) &
                           (classes.classifier == "lightgbm") & (classes.level == "frame")]
            table = part.pivot(index="method", columns="class", values="errors").reindex(METHODS)
            counts = errors[(errors.task == task) & (errors.feature_set == fs) &
                            (errors.classifier == "lightgbm")].set_index("method")
            table["cell-error poses /45"] = counts.groups_ever_frame_wrong
            table["recording errors /147"] = counts.probability_mean_errors
            table["record-error poses /45"] = counts.groups_ever_probability_mean_wrong
            lines += [f"## {task}, {fs}", "", table.to_markdown(), ""]
        frame = frame_records[(frame_records.task == task) & (frame_records.classifier == "lightgbm") &
                              (frame_records.method == "irmf_h2_a1.41")].copy()
        rec = record_errors[(record_errors.task == task) & (record_errors.classifier == "lightgbm") &
                            (record_errors.method == "irmf_h2_a1.41") &
                            (record_errors.aggregation == "probability_mean")]
        frame = frame.merge(rec[["feature_set", "group_recording", "errors"]],
                            on=["feature_set", "group_recording"], validate="one_to_one")
        frame = frame[(frame.n_frame_errors > 0) | (frame.errors > 0)]
        lines += [f"## {task}, recordings ever wrong for GD60 IRMF LightGBM", "",
                  frame[["group_recording", "group", "y_true", "feature_set", "n_frame_errors", "n_frame_evaluations", "errors"]].sort_values(
                      ["feature_set", "n_frame_errors"], ascending=[True, False]).to_markdown(index=False), ""]
    (OUT / "ERRORS.md").write_text("\n".join(lines))
    selected = power[(power.method_a == "irmf_h2_a1.41") & (power.method_b == "emd")].copy()
    columns = ["mean_diff", "se_corrected", "threshold_nominal", "mde80_nominal", "threshold_holm_first_step", "mde80_holm_first_step"]
    selected[columns] *= 100
    (OUT / "POWER.md").write_text("\n".join([
        "# Detectable balanced-accuracy differences", "",
        "GD60 IRMF minus EMD. All values below are percentage points. se_corrected uses the observed paired fold variance and mean held-out pose-group ratio, with 14 df. threshold is the difference needed for significance at the stated alpha, conditional on that variance. mde80 solves 80% power under the corresponding noncentral-t model.", "",
        "Nominal alpha is .05; Holm first step uses .05/120 and is a conservative threshold. The actual step-down rank depends on all tests. These plug-in sensitivities are not evidence of equivalence or a new-session power forecast. Zero paired variance leaves planning MDE undefined.", "",
        selected[["task", "feature_set", "classifier"] + columns].to_markdown(index=False, floatfmt=".3f"), "",
        "All requested contrasts, including capacity/H checks, are in power.csv. RUN_NOTES.md gives the formula and numerical integration method.", ""]))
    sensitivity = pairs[(pairs.inference_table == "sensitivity") & (pairs.metric == "balanced_accuracy")].copy()
    sensitivity["A - B"] = sensitivity.method_a + " - " + sensitivity.method_b
    sensitivity["delta [95% CI], pp"] = sensitivity.apply(lambda r: f"{100*r.mean_diff:+.2f} [{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]", axis=1)
    (OUT / "ABLATIONS.md").write_text("\n".join([
        "# Capacity and H sensitivity", "",
        "EMD and both a=2 methods have nine slots, 360 tree features, or 18 stack channels. The a=sqrt(2) methods have 16 slots, 640 tree features, or 32 stack channels. Width and transformed-feature budgets are held common. Newton tags converged in every channel. These checks match input count, not adaptive bandwidth locations.", "",
        "Holm covers 84 tests per task/metric in this table; Nadeau-Bengio is primary and Wilcoxon secondary. All mean scores are in TABLES.md, all metrics in SCORES_ALL_METRICS.md.", "",
        sensitivity[["task", "feature_set", "classifier", "A - B", "delta [95% CI], pp", "p_nb_holm", "p_wilcoxon_holm"]].to_markdown(index=False, floatfmt=".5g"), ""]))
    score_columns = ["task", "method", "feature_set", "classifier", "balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"]
    error_columns = ["task", "method", "feature_set", "classifier", "frame_errors", "groups_ever_frame_wrong", "probability_mean_errors", "groups_ever_probability_mean_wrong"]
    payload = {"scores": scores[score_columns].to_dict("records"), "errors": errors[error_columns].to_dict("records"),
               "methods": METHODS, "labels": LABELS, "models": MODELS}
    html = '''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans,system-ui);color:var(--foreground,#222);font-size:13px;line-height:1.45;margin:0}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 14px}label{display:grid;gap:3px;color:var(--muted-foreground,#666);font-size:12px}select{font:inherit;padding:6px;border:1px solid var(--border,#ddd);border-radius:5px;background:var(--background,#fff);color:var(--foreground,#222)}
.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px}th,td{padding:7px 6px;text-align:right;border-bottom:1px solid var(--border,#ddd);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted-foreground,#666)}h3{font-size:16px;font-weight:600;margin:18px 0 6px}p{margin:6px 0;color:var(--muted-foreground,#666)}.note{font-size:12px}.good{color:var(--success,#17834a)}.bad{color:var(--destructive,#b93636)}
</style></head><body>
<p>784 I/Q cell frames · 49 recordings · 45 pose groups · 15 paired folds · all nine inputs</p>
<div class="controls"><label>Task<select id="task"><option value="bird_drone">Bird vs UAS</option><option value="target3">Bird / Mavic / P3P</option></select></label><label>Scale feature<select id="scale"><option value="comp">Without log_scale</option><option value="comp+log_scale">With log_scale</option></select></label><label>Metric<select id="metric"><option value="balanced_accuracy_mean">Balanced accuracy</option><option value="macro_f1_mean">Macro-F1</option><option value="accuracy_mean">Accuracy</option></select></label></div>
<div class="table"><table id="scores"></table></div>
<p class="note">Percent, mean over 15 held-out folds. Colors mark descriptive differences from EMD. Mini/Multi use 1,008/672 features; CNN width 8, at most 20 epochs. Other model budgets are in RUN_NOTES.md.</p>
<h3>How many poses are ever wrong?</h3><div class="controls"><label>Classifier<select id="model"></select></label></div>
<div class="table"><table id="errors"></table></div>
<p class="note">A pose counts once if wrong in any of the three seeds. Cell errors are repeated out-of-fold errors / 2,352. Recording errors are probability-mean errors / 147. All data comes from one indoor campaign.</p>
<script>const DATA=__DATA__;
const names={lightgbm:'LGBM',xgboost:'XGB',extratrees:'ET',minirocket:'Mini',multirocket:'Multi',compact_resnet:'CNN'};
const byId=id=>document.getElementById(id);
DATA.models.forEach(m=>byId('model').add(new Option(names[m],m)));
function render(){const task=byId('task').value,scale=byId('scale').value,metric=byId('metric').value,model=byId('model').value;
const values=DATA.scores.filter(r=>r.task===task&&r.feature_set===scale),err=DATA.errors.filter(r=>r.task===task&&r.feature_set===scale&&r.classifier===model);
let header='<tr><th>Input</th>'+DATA.models.map(m=>'<th>'+names[m]+'</th>').join('')+'</tr>';
byId('scores').innerHTML=header+DATA.methods.map((m,i)=>'<tr><td>'+DATA.labels[i]+'</td>'+DATA.models.map(c=>{const v=values.find(r=>r.method===m&&r.classifier===c)[metric],base=values.find(r=>r.method==='emd'&&r.classifier===c)[metric];return '<td class="'+(v>base+.00005?'good':v<base-.00005?'bad':'')+'">'+(100*v).toFixed(2)+'</td>'}).join('')+'</tr>').join('');
byId('errors').innerHTML='<tr><th>Input</th><th>Cell errors</th><th>Poses /45</th><th>Recording errors</th><th>Poses /45</th></tr>'+DATA.methods.map((m,i)=>{const r=err.find(x=>x.method===m);return '<tr><td>'+DATA.labels[i]+'</td><td>'+r.frame_errors+'</td><td>'+r.groups_ever_frame_wrong+'</td><td>'+r.probability_mean_errors+'</td><td>'+r.groups_ever_probability_mean_wrong+'</td></tr>'}).join('');
console.log('Rendered',task,scale,metric,values.length,'score cells',err.length,'error cells');}
['task','scale','metric','model'].forEach(id=>byId(id).addEventListener('change',render));render();</script></body></html>'''.replace('__DATA__', json.dumps(payload, allow_nan=False))
    (OUT / "overview.html").write_text(html)
    print("Wrote ERRORS.md, POWER.md, ABLATIONS.md and overview.html")


if __name__ == "__main__":
    main()
