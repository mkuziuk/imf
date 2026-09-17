"""Offline scale controls and short explanations for the timing chart."""
from itertools import product

CSS = """
.scaling-controls{display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;margin:1.2rem 0}
.scale-options{display:flex;gap:.4rem;flex-wrap:wrap;border:0;padding:0;margin:0}
.scale-options legend{font-size:.85rem;color:var(--muted);padding:0;margin-bottom:.4rem}
.scaling-controls label{display:inline-flex;align-items:center;gap:.45rem;cursor:pointer}
.scale-options label{padding:.45rem .75rem;border:1px solid #59616c;border-radius:.5rem}
.scale-options label:has(input:checked){background:#3a4657;border-color:#8ab4f8}
.scaling-controls input{accent-color:#8ab4f8}
.scaling-controls input:focus-visible{outline:2px solid #8ab4f8;outline-offset:4px}
.scale-panel{margin:0}.scale-panel[hidden]{display:none}
.scale-description{min-height:1.6em}.fit-formula{font-size:1.2rem;color:#edf0f2}
.fit-table{min-width:0;width:100%}.fit-range{display:block;color:var(--muted);font-size:.8rem;margin-top:.2rem}
@media(max-width:600px){.fit-table th,.fit-table td{padding:.6rem .4rem}.fit-table{font-size:.82rem}.fit-table th{font-size:.76rem}.fit-range{font-size:.72rem}}
@media print{.scaling-controls{display:none}}
"""

SCRIPT = """<script>
(() => {
  const section = document.getElementById('scaling');
  if (!section) return;
  const descriptions = {
    seconds: {
      linear: 'Linear axes show the actual gaps in waiting time.',
      loglog: 'Both axes are logarithmic. The slope of each fitted line is its power p.'
    },
    relative: {
      linear: 'Each method starts at 1× at 1,000 points. A value of 100× means it takes 100 times as long as its own starting time.',
      loglog: 'Each method starts at 1× at 1,000 points. Both axes are logarithmic; the slope of each fitted line is its power p.'
    }
  };
  function updateChart() {
    const scale = section.querySelector('input[name="timing-scale"]:checked').value;
    const units = section.querySelector('input[name="timing-units"]:checked').value;
    const fit = section.querySelector('#show-scaling-fit').checked ? 'yes' : 'no';
    section.querySelectorAll('.scale-panel').forEach(panel => {
      panel.hidden = panel.dataset.scale !== scale || panel.dataset.fit !== fit || panel.dataset.units !== units;
    });
    section.querySelector('#scale-description').textContent = descriptions[units][scale];
    section.querySelector('#normalization-note').hidden = units !== 'relative';
  }
  section.querySelector('.scaling-controls').addEventListener('change', updateChart);
  updateChart();
})();
</script>"""


def render_scaling(result, frame, fitted, graphic):
    params = result["cases"][0]
    stages = [case["stages"] for case in result["cases"]]
    quantlet_max_n = max(run["n"] for run in result["runs"] if run["method"] == "quantlet")
    panels = []
    for units, scale, show_fit in product(['seconds', 'relative'], ['linear', 'loglog'], [True, False]):
        label = 'Linear axes' if scale == 'linear' else 'Logarithmic point and time axes'
        hidden = '' if units == 'relative' and scale == 'linear' and show_fit else ' hidden'
        image = graphic('scaling_' + ('relative_' if units == 'relative' else '') + scale + ('_fit' if show_fit else ''),
                        f'{label}: ' + ('time relative to each method at 1000 points' if units == 'relative' else 'full decomposition time in seconds') +
                        ' for QuantLet at one thread and my GD at one, four and twelve threads'
                        + (', with dashed fitted power laws within the measured ranges' if show_fit else ''))
        panels.append(f'<figure class="scale-panel" data-units="{units}" data-scale="{scale}" data-fit="{"yes" if show_fit else "no"}"{hidden}>{image}</figure>')
    endpoint = frame[frame.n == quantlet_max_n]
    gd_endpoint = endpoint[(endpoint.method == 'gd') & (endpoint.workers == 1)].iloc[0]
    q_endpoint = endpoint[endpoint.method == 'quantlet'].iloc[0]
    fit_rows = []
    common_rows = []
    for fit in fitted["fits"]:
        fit_rows.append(f'<tr><td>{fit["label"]}<span class="fit-range">Fit: {fit["n_min"]:,} to {fit["n_max"]:,} points</span></td>'
                        f'<td>{fit["power"]:.2f}</td><td>{fit["doubling_factor"]:.2f}×</td></tr>')
        common = fit["shared_range_fit"]
        common_rows.append(f'<tr><td>{fit["label"]}</td><td>{common["power"]:.2f}</td>'
                           f'<td>{common["doubling_factor"]:.2f}×</td><td>{common["r_squared_log"]:.4f}</td></tr>')
    gd_powers = [fit["power"] for fit in fitted["fits"] if fit["method"] == "gd"]
    interpretation = ('<p>The GD fits are close to power 2. Threads reduce the waiting time, '
                      'while the fitted proportional growth stays similar.</p>'
                      if all(1.85 <= power <= 2.15 for power in gd_powers) and
                      max(gd_powers) - min(gd_powers) < 0.15 else '')
    rows = []
    for n in result["sizes"]:
        group = frame[frame.n == n]
        times = []
        for method, workers in [("quantlet", 1), ("gd", 1), ("gd", 4), ("gd", 12)]:
            matching = group[(group.method == method) & (group.workers == workers)]
            times.append(float(matching.iloc[0].seconds) if len(matching) else None)
        rows.append(f'<tr><td>{n:,}</td><td>{int(group.iloc[0].stage_count)}</td>' +
                    ''.join(f'<td>{value:.3f}</td>' if value is not None else '<td>Not measured</td>' for value in times) + '</tr>')
    checks = (f'All {len(result["runs"])} runs passed the applicable output checks.'
              if result["metadata"]["all_output_checks_passed"] else 'Some runs failed the output checks.')
    extensions = result["metadata"].get("quantlet_extensions", [])
    if extensions:
        extension = extensions[-1]
        minutes = extension["elapsed_wall_seconds"] / 60
        extension_note = (f'I added {len(extension["completed_sizes"])} QuantLet measurements in {minutes:.1f} minutes. '
                          f'This extension had a {extension["budget_seconds"] / 60:g}-minute limit. ')
        if extension['status'] == 'time_limit':
            extension_note += 'The limit stopped the unfinished run, which is excluded. '
        extension_note += f'QuantLet ends at {quantlet_max_n:,} measured points; I have not timed larger sizes.'
    else:
        extension_note = ('I stopped further QuantLet calculations because the full sweep was projected '
                          'to exceed the 40-minute limit. Its line ends at the last completed measurement.')
    return f'''<section class="section" id="scaling"><h2>Time as the number of points grows</h2>
<p>I keep my defaults: sigma = {params['sigma']}, H = 2 × sigma = {params['H']}, the same signal and noise model, and my squared-triangular kernel. My GD covers 10 sizes. QuantLet uses one thread and now reaches {quantlet_max_n:,} points.</p>
<div class="scaling-controls">
<fieldset class="scale-options"><legend>Time display</legend>
<label><input type="radio" name="timing-units" value="seconds">Seconds</label>
<label><input type="radio" name="timing-units" value="relative" checked>Relative to 1,000</label>
</fieldset>
<fieldset class="scale-options"><legend>Chart scale</legend>
<label><input type="radio" name="timing-scale" value="linear" checked>Linear</label>
<label><input type="radio" name="timing-scale" value="loglog">Log–log</label>
</fieldset>
<label><input type="checkbox" id="show-scaling-fit" checked>Show fitted power laws</label>
</div>
<p id="scale-description" class="caption scale-description" aria-live="polite">Each method starts at 1× at 1,000 points. A value of 100× means it takes 100 times as long as its own starting time.</p>
{''.join(panels)}
<p class="caption">Markers show measurements. Dashed lines are fits and stop at the measured endpoints. Turn off fits to join the measurements directly. Select <strong>Log–log</strong> to see each fitted power law as a straight line.</p>
<p id="normalization-note">This view compares proportional growth. At {quantlet_max_n:,} points, my one-thread GD has grown {gd_endpoint.time_multiple:.0f}× and QuantLet {q_endpoint.time_multiple:.0f}×. GD still takes fewer seconds: {gd_endpoint.seconds:.0f} s versus {q_endpoint.seconds:.0f} s. Select <strong>Seconds</strong> to compare the waiting time.</p>
<h3>How much does doubling the signal cost?</h3>
<p>A power of 1 means twice as much time for twice as many points. A power of 2 means four times as much time.</p>
<div class="table-wrap"><table class="fit-table"><thead><tr><th>Solver and fitted range</th><th>Power p</th><th>Time for 2× points</th></tr></thead><tbody>{''.join(fit_rows)}</tbody></table></div>
{interpretation}
<p class="caption">These fits describe the measured range. There is one timing per size, so I cannot estimate run-to-run variation. The default windows grow with the signal and stages increase from {min(stages)} to {max(stages)}; this is the cost of the full default workflow.</p>
<details><summary>Fit method and a comparison over the same sizes</summary>
<p class="fit-formula">Time ≈ a × <span>(points / 1,000)</span><sup>p</sup></p>
<p>I fit a straight line to log time versus log points, giving every measured size equal weight. The chart uses each solver's full measured range. For a comparison over the same sizes, the table below fits all four curves only through {quantlet_max_n:,} points. R² measures how closely the points follow a straight line in this view; it does not prove a complexity bound.</p>
<p>The relative view divides each measured time by that method's measured time at 1,000 points. Dashed curves divide each original fit by its own fitted time at 1,000 points. Both start at 1; the fitted powers stay unchanged.</p>
<div class="table-wrap"><table><thead><tr><th>Solver</th><th>Power p, shared range</th><th>Time for 2× points</th><th>R², log space</th></tr></thead><tbody>{''.join(common_rows)}</tbody></table></div>
<p><a href="solver-speed-files/scaling_fits.csv">Fit coefficients CSV</a> · <a href="solver-speed-files/scaling_fits.json">Fit model and source-data hash</a></p></details>
<p class="caption">{extension_note}</p>
<details><summary>Exact times and size-sweep data</summary><div class="table-wrap"><table><thead><tr><th>Points</th><th>Stages</th><th>QuantLet, 1 thread · s</th><th>My GD, 1 thread · s</th><th>My GD, 4 threads · s</th><th>My GD, 12 threads · s</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>{checks} Scalar comparisons cover sizes through {quantlet_max_n:,}. Larger GD runs are checked for convergence, reconstruction and agreement across thread counts. Each configuration uses identical observations and windows at a given size. Measurements follow an untimed warmup at 1,000 points.</p>
<p><a href="solver-speed-files/scaling.json">Every measurement and source hash</a> · <a href="solver-speed-files/scaling_summary.csv">Summary CSV</a> · <a href="solver-speed-files/scaling_normalized.csv">Relative times CSV</a> · <a href="solver-speed-files/scaling_benchmark.py">Original size-sweep code</a> · <a href="solver-speed-files/extend_quantlet.py">QuantLet extension code</a> · <a href="solver-speed-files/scaling_charts.py">Fit and chart code</a> · <a href="solver-speed-files/scaling_report.py">Chart controls</a></p></details></section>'''
