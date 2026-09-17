"""Build the short, illustrated benchmark report from completed timing results."""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_support import ROOT, OUTPUT, kernel, summarize

COLORS = {"gd": "#8ab4f8", "quantlet": "#f0b56b"}
TEXT, MUTED, PANEL, GRID = "#edf0f2", "#b2b8c0", "#292c31", "#454a52"


def chart_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "figure.facecolor": PANEL, "axes.facecolor": PANEL,
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": TEXT,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.alpha": 0.5, "legend.facecolor": PANEL,
        "legend.edgecolor": GRID, "svg.fonttype": "path", "axes.titleweight": "normal",
    })


def build_report(result):
    assert result["metadata"].get("complete"), "Only completed benchmarks can be reported."
    assert all(run["workers"] == 1 for run in result["runs"] if run["method"] == "quantlet"), \
        "QuantLet results must be single-threaded."
    chart_style()
    frame = summarize(result)
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pngs = []
    svgs = {}

    def save(fig, name):
        svg_path = figures / (name + ".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        fig.savefig(figures / (name + ".png"), dpi=170, bbox_inches="tight")
        pngs.append(figures / (name + ".png"))
        svgs[name] = (figures / (name + ".svg")).read_text()
        plt.close(fig)

    def one(case, method, workers):
        return frame[(frame.case == case) & (frame.method == method) & (frame.workers == workers)].iloc[0]

    comparison_rows = []
    for case in result["cases"]:
        key = case["key"]
        fig, ax = plt.subplots(figsize=(7.2, 3.15), layout="constrained")
        specs = [("gd", 1), ("gd", 4), ("gd", 12), ("quantlet", 1)]
        rows = [one(key, method, workers) for method, workers in specs]
        seconds = np.array([row.seconds for row in rows])
        uncertainty = np.array([[row.seconds - row.minimum for row in rows],
                                [row.maximum - row.seconds for row in rows]])
        positions = np.arange(4)
        ax.barh(positions, seconds, color=[COLORS[method] for method, _ in specs], height=0.55)
        ax.errorbar(seconds, positions, xerr=uncertainty, fmt="none", ecolor=TEXT, capsize=3, linewidth=1)
        ax.set_yticks(positions, [f"{'My GD' if method == 'gd' else 'QuantLet'} · {workers} {'thread' if workers == 1 else 'threads'}"
                                 for method, workers in specs])
        ax.invert_yaxis()
        xmax = max(row.maximum for row in rows)
        ax.set_xlim(0, xmax * 1.23)
        for position, row in zip(positions, rows):
            ax.text(row.maximum + xmax * 0.035, position, f"{row.seconds:.3f} s", va="center", fontsize=12)
        ax.set_xlabel("Seconds for the whole decomposition · lower is better")
        ax.set_title(case["label"] + f" · {case['n']:,} points", loc="left", pad=12)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        save(fig, "times_" + key)

        fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
        group = frame[(frame.case == key) & (frame.method == "gd")].sort_values("workers")
        x = np.arange(len(group))
        ax.plot(x, group.speedup, color=COLORS["gd"], marker="o", linewidth=2, label="My GD")
        baseline = one(key, "gd", 1).seconds
        ax.fill_between(x, baseline / group.maximum, baseline / group.minimum, color=COLORS["gd"], alpha=0.14)
        best = group.loc[group.seconds.idxmin()]
        comparison_rows.append({"case": key, "label": case["label"],
                                "quantlet_single": one(key, "quantlet", 1).seconds,
                                "one_thread": baseline, "twelve_threads": one(key, "gd", 12).seconds,
                                "best_workers": int(best.workers),
                                "twelve_speedup": float(one(key, "gd", 12).speedup)})
        ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(x, group.workers)
        ax.set_xlabel("My GD worker threads")
        ax.set_ylabel("GD speed relative to one thread")
        ax.set_ylim(0, max(4.0, frame[frame.case == key].speedup.max() * 1.18))
        ax.set_title(case["label"], loc="left", pad=12)
        ax.yaxis.grid(True)
        ax.legend(loc="upper right")
        save(fig, "threads_" + key)

    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    u = np.linspace(-1, 1, 501)
    ax.plot(u, 1.5 * (1 - np.abs(u)) ** 2, color=COLORS["gd"], label="My kernel", linewidth=2.5)
    ax.plot(u, 0.75 * (1 - u ** 2), color=COLORS["quantlet"], label="QuantLet kernel", linewidth=2.5)
    ax.set(xlabel="Position inside the window", ylabel="Relative influence", title="My kernel concentrates more weight near the center")
    ax.set_xticks([-1, 0, 1], ["Left edge", "Center", "Right edge"])
    ax.set_ylim(0, 1.7)
    ax.legend()
    save(fig, "kernels")

    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    distance = np.linspace(0.001, 2, 501)
    for H, color, label in [(0.4, COLORS["gd"], "My GD: H = 0.4"),
                             (0.8, "#80cbc4", "Later notebook: H = 0.8"),
                             (1.0, COLORS["quantlet"], "QuantLet: H = 1.0")]:
        # Divide psi(e)/e by its limit at zero. Multiplying all weights by
        # this H-dependent constant leaves the fitted minimizer unchanged.
        relative_weight = math.sqrt(math.pi / 2) * H * np.array([
            math.erf(value / (math.sqrt(2) * H)) for value in distance]) / distance
        ax.plot(distance, relative_weight * 100, color=color, linewidth=2.5, label=label)
    ax.set(xlabel="Distance from the local fitted value", ylabel="Relative weight, %",
           title="Smaller H reduces an unusual value's weight sooner", ylim=(0, 105))
    ax.legend(loc="lower left")
    ax.yaxis.grid(True)
    save(fig, "robustness")

    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    for case, method in zip(result["cases"], ["gd", "quantlet"]):
        ax.plot(np.arange(1, case["stages"] + 1), np.array(case["bandwidths"]) * 200,
                marker="o", linewidth=2.5, color=COLORS[method], label=case["label"])
    ax.set(xlabel="Component number", ylabel="Window width as % of signal", title="Both start wide, then use smaller windows")
    ax.set_ylim(0, 57)
    ax.set_xticks(range(1, max(c["stages"] for c in result["cases"]) + 1))
    ax.legend()
    ax.yaxis.grid(True)
    save(fig, "windows")

    # Agreement figure uses the first component, where errors are largest here.
    example_case = result["cases"][0]
    data = np.load(OUTPUT / (example_case["key"] + "_example.npz"))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(data["t"], data["gd_components"][0], color=COLORS["gd"], linewidth=2.5, label="My GD")
    axes[0].plot(data["t"], data["quantlet_components"][0], color=COLORS["quantlet"], linewidth=1.7,
                 linestyle="--", label="QuantLet")
    axes[0].set(title="The fitted components agree", ylabel="First component")
    axes[0].legend()
    error = data["gd_components"][0] - data["quantlet_components"][0]
    axes[1].plot(data["t"], error * 1e6, color="#80cbc4", linewidth=1.3)
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set(xlabel="Position in the signal", ylabel="Difference\nin millionths")
    save(fig, "agreement")

    scaling = None
    scaling_path = OUTPUT / "scaling.json"
    if scaling_path.exists():
        candidate = json.loads(scaling_path.read_text())
        if candidate["metadata"].get("complete"):
            from scaling_charts import draw_scaling, save_fits
            scaling = candidate
            scaling_frame, scaling_fits = draw_scaling(scaling, save)
            save_fits(scaling_fits, OUTPUT)
            scaling_frame[["n", "method", "workers", "seconds", "baseline_seconds", "time_multiple"]].to_csv(
                OUTPUT / "scaling_normalized.csv", index=False)

    report_dir = ROOT / "tmp/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    earlier_comparison = ('<a href="quantlet-imf-comparison.html">Earlier mathematical comparison</a>'
                          if (report_dir / "quantlet-imf-comparison.html").exists() else '')
    evidence = report_dir / "solver-speed-files"
    evidence.mkdir(exist_ok=True)
    for source in [OUTPUT / "benchmark.json", OUTPUT / "benchmark_summary.csv",
                   Path(__file__).with_name("benchmark_support.py"), Path(__file__),
                   Path(__file__).with_name("report_style.css"),
                   Path(__file__).with_name("solver_speed_comparison.ipynb")]:
        if source.exists():
            shutil.copyfile(source, evidence / source.name)
    if scaling is not None:
        for source in [scaling_path, OUTPUT / "scaling_summary.csv",
                       Path(__file__).with_name("scaling_benchmark.py"),
                       Path(__file__).with_name("scaling_charts.py"),
                       Path(__file__).with_name("scaling_report.py"),
                       Path(__file__).with_name("extend_quantlet.py"),
                       OUTPUT / "scaling_fits.json", OUTPUT / "scaling_fits.csv",
                       OUTPUT / "scaling_normalized.csv"]:
            shutil.copyfile(source, evidence / source.name)

    def graphic(name, label):
        encoded = base64.b64encode(svgs[name].encode()).decode()
        return f'<img class="chart" src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(label, quote=True)}">'

    your_case = result["cases"][0]["key"]
    their_case = result["cases"][1]["key"]
    your_single = one(your_case, "gd", 1).seconds
    their_single = one(your_case, "quantlet", 1).seconds
    gd_choices = frame[(frame.case == your_case) & (frame.method == "gd")]
    best_gd = gd_choices.loc[gd_choices.seconds.idxmin()]
    single_thread_advantages = [one(case["key"], "quantlet", 1).seconds / one(case["key"], "gd", 1).seconds
                                for case in result["cases"]]
    twelve_thread_gains = [one(case["key"], "gd", 12).speedup for case in result["cases"]]
    all_runs = result["runs"] + (scaling["runs"] if scaling is not None else [])
    max_error = max(run["max_component_difference_from_scalar"] for run in all_runs
                    if run["max_component_difference_from_scalar"] is not None)
    max_thread_error = max(run["max_component_difference_from_single_thread"] for run in all_runs)
    speed_rows = ''.join(
        f'<tr><td>{row["label"]}</td><td>{row["quantlet_single"]:.3f} s</td>'
        f'<td>{row["one_thread"]:.3f} s</td><td>{row["twelve_threads"]:.3f} s</td>'
        f'<td>{row["best_workers"]}</td><td>{row["twelve_speedup"]:.2f}×</td></tr>'
        for row in comparison_rows)
    raw_rows = ''.join(
        f'<tr><td>{row.case}</td><td>{row.method}</td><td>{row.workers}</td><td>{row.seconds:.4f}</td>'
        f'<td>{row.minimum:.4f}–{row.maximum:.4f}</td></tr>' for row in frame.itertuples())
    css = Path(__file__).with_name("report_style.css").read_text()
    provenance = ('<p>These are retained measurements from the original timing session. '
                  'QuantLet results with more than one worker were removed. The JSON keeps '
                  'the source hashes recorded when the timings were measured.</p>'
                  if result["metadata"].get("scope_revision") else '')
    from scaling_report import CSS as SCALING_CSS, SCRIPT as SCALING_SCRIPT, render_scaling
    scaling_section = ''
    scaling_nav = ''
    if scaling is not None:
        scaling_nav = '<a href="#scaling">Signal length</a>'
        scaling_section = render_scaling(scaling, scaling_frame, scaling_fits, graphic)
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Single-threaded QuantLet compared with GD at several thread counts, with a simple parameter comparison.">
<title>IMF solver speed and settings</title><style>{css}
.layout{{grid-template-columns:12rem minmax(0,70rem);gap:2rem;width:min(100% - 2rem,86rem)}}
.hero{{padding:2rem}}h1{{max-width:25ch;font-size:clamp(2rem,4vw,3.5rem)}}.lede{{max-width:65ch}}
.grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}}.chart{{width:100%;height:auto;display:block}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem}}
.metric{{padding:1rem;background:#30343a;border-radius:.7rem}}.metric strong{{font-size:1.9rem;display:block;color:#80cbc4}}
.metric span{{font-size:.9rem;color:var(--muted)}}.caption{{font-size:.94rem;color:var(--muted);max-width:none}}
.table-wrap{{margin:1rem 0}}table{{font-size:.88rem;min-width:42rem}}th{{text-transform:none;letter-spacing:0;font-size:.85rem}}
.section p{{max-width:80ch}}.checks{{display:flex;flex-wrap:wrap;gap:1rem}}.checks p{{background:#30343a;padding:.7rem 1rem;border-radius:.6rem;margin:0}}
.flow{{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;margin:1rem 0}}.flow span{{padding:.8rem;border:1px solid #454a52;border-radius:.6rem}}.flow b{{color:#80cbc4}}
details{{margin-top:1rem}}summary{{cursor:pointer;color:#8ab4f8}}footer a{{margin-right:1rem}}
@media(max-width:1050px){{.layout{{display:block;max-width:72rem}}aside{{position:static;margin-bottom:1rem}}nav{{display:flex;flex-wrap:wrap}}}}
@media(max-width:780px){{.grid,.metric-grid{{grid-template-columns:1fr}}.hero,.section{{padding:1.1rem}}}}
@media print{{.grid{{grid-template-columns:1fr}}.section{{break-inside:auto}}.metric,.checks p{{background:#eee}}.metric strong{{color:#234}}.chart{{max-width:85%;margin:auto}}table,th,td{{color:#222;background:white}}details{{display:block}}}}
{SCALING_CSS}
</style></head><body><div class="layout"><aside aria-label="Report navigation"><p class="nav-title">Speed report</p><nav>
<a href="#summary">Result</a><a href="#times">Time taken</a>{scaling_nav}<a href="#speed">Why it is faster</a><a href="#threads">More threads</a><a href="#settings">Settings</a><a href="#accuracy">Same answer?</a><a href="#method">How we tested</a>
</nav></aside><main>
<header class="hero" id="summary"><p class="eyebrow">17 September 2026 · Apple M4 Max</p><h1>Which IMF solver finishes sooner?</h1>
<p class="lede">My gradient-descent solver was faster in these matched tests. I compare my GD at several thread counts with QuantLet's original single-threaded solver.</p>
<div class="metric-grid"><div class="metric"><strong>{their_single / your_single:.1f}×</strong><span>My GD's advantage at one thread in the original 1,000-point test</span></div>
<div class="metric"><strong>{best_gd.speedup:.1f}×</strong><span>My GD's best measured gain from threads in the original 1,000-point test</span></div>
<div class="metric"><strong>≈ {max_error * 1e6:.2f}</strong><span>Millionths: largest component difference where a scalar reference was measured</span></div></div></header>

<section class="section" id="times"><h2>Time for one complete decomposition</h2>
<p>Within each test, both solvers receive the same signal and settings. Shorter bars mean less waiting.</p>
<div class="grid">{graphic('times_' + your_case, 'Timing comparison using my settings')}{graphic('times_' + their_case, 'Timing comparison using QuantLet settings')}</div>
<p class="caption">Bars show the median of {result['repeats']} runs. Thin lines show the fastest and slowest run. These timings include all {result['cases'][0]['stages']} stages for my settings and all {result['cases'][1]['stages']} stages for QuantLet's settings. The panels use different signals and H values, so their difference is not a test of signal length alone.</p></section>

{scaling_section}

<section class="section" id="speed"><h2>Where the speed comes from</h2>
<ol>
<li>I process many windows together in NumPy, avoiding Python loops over individual fits.</li>
<li>I use gradient updates and lookup tables to evaluate the score, instead of repeatedly evaluating the full loss through scalar searches.</li>
<li>I distribute independent groups of windows across CPU cores.</li>
</ol>
<p>In the two fixed-size comparisons, the first two choices together produced a {min(single_thread_advantages):.1f}–{max(single_thread_advantages):.1f}× single-thread advantage. Using 12 threads added another {min(twelve_thread_gains):.1f}–{max(twelve_thread_gains):.1f}× speedup. I have not measured batching, gradient updates and lookup tables separately.</p>
<p class="caption">H, window sizes and stage count affect how much work is needed. Ground-truth comparisons and error analysis improve the evaluation, but do not make the solver faster.</p></section>

<section class="section" id="threads"><h2>More threads speed up my GD</h2>
<div class="grid">{graphic('threads_' + your_case, 'GD thread speedups using my settings')}{graphic('threads_' + their_case, 'GD thread speedups using QuantLet settings')}</div>
<p class="caption">These curves show only my GD. Above 1× is faster than its one-thread baseline. Shading shows the observed timing range.</p>
<p>My GD fits several groups of windows at once. QuantLet uses its original single-threaded solver throughout.</p>
<div class="table-wrap"><table><thead><tr><th>Test settings</th><th>QuantLet, 1 thread</th><th>My GD, 1 thread</th><th>My GD, 12 threads</th><th>Fastest GD thread count</th><th>GD gain at 12 threads</th></tr></thead><tbody>{speed_rows}</tbody></table></div>
<p class="caption">The fastest tested GD count is specific to these inputs and this machine.</p></section>

<section class="section" id="settings"><h2>The repositories start with different settings</h2>
<div class="grid">{graphic('kernels', 'Different kernel shapes')}{graphic('robustness', 'Smaller H reduces the relative weight of unusual values sooner')}</div>
<p>A window chooses which nearby points contribute. The kernel sets their weight by position. H reduces their weight when their values are unusual. In the H chart, 100% is the weight of a value very close to the local fit.</p>
<div style="max-width:48rem;margin:auto">{graphic('windows', 'Window width decreases by stage')}</div>
<div class="table-wrap"><table><thead><tr><th>Setting</th><th>My GD notebook</th><th>My later review notebook</th><th>QuantLet main script</th></tr></thead><tbody>
<tr><td>Signal length</td><td>1,000</td><td>1,000</td><td>2,001 including both endpoints</td></tr>
<tr><td>Kernel</td><td>Squared triangular</td><td>Squared triangular</td><td>Epanechnikov</td></tr>
<tr><td>First window width</td><td>About 50% of signal</td><td>About 50%</td><td>About 40%</td></tr>
<tr><td>Components</td><td>9</td><td>9</td><td>8</td></tr>
<tr><td>H</td><td>0.4</td><td>0.8</td><td>1.0</td></tr>
<tr><td>Noise</td><td>Gaussian SD 0.2 + signed exponential</td><td>Gaussian SD 0.4 + signed exponential</td><td>Default: replace 20% with zero</td></tr>
<tr><td>Contamination probability</td><td>20%</td><td>20%</td><td>20%</td></tr>
<tr><td>Exponential scale</td><td>0.2</td><td>0.6</td><td>Alternative: Gaussian mixture, SD 0.1 / 0.4 or 0.7</td></tr>
<tr><td>Stopping rule</td><td>Small update, at most 60 iterations</td><td>Same</td><td>Small search interval, at most 200 iterations</td></tr>
<tr><td>Default worker threads</td><td>12</td><td>12</td><td>1</td></tr>
</tbody></table></div>
<p class="caption">I tested my GD settings and QuantLet's default zero-replacement settings. Within each case both methods use that case's kernel and H. Both use an endpoint-exclusive grid, so QuantLet's case has 2,000 unique points. The later review column is a source comparison, not another timing case. My newer residual-distribution notebook instead uses 512 points, H = 0.2 and 400 repetitions.</p></section>

<section class="section" id="accuracy"><h2>The speed difference did not change the fitted answer</h2>
{graphic('agreement', 'Matching fitted components and their small numerical difference')}
<div class="checks"><p>{sum((run['max_component_difference_from_scalar'] is None or run['max_component_difference_from_scalar'] < 1e-4) and run['max_component_difference_from_single_thread'] < 1e-4 and run['unconverged_chunks'] == 0 for run in all_runs)} of {len(all_runs)} runs passed the applicable checks.</p><p>{sum(run['unconverged_chunks'] for run in all_runs)} unfinished GD chunks.</p><p>Largest thread-related change: {max_thread_error * 1e6:.2f} millionths.</p></div>
<p class="caption">The overlay shows the first component using my settings. Component checks cover agreement with the scalar solver where it was measured, and agreement across GD thread counts elsewhere. This does not rank signal-recovery quality under different settings.</p></section>

<section class="section" id="method"><h2>What the clock included</h2>
<div class="flow"><span>Prepare local windows</span><b>→</b><span>Fit windows</span><b>→</b><span>Subtract component</span><b>→</b><span>Next stage</span></div>
<p>I timed one whole signal, including window preparation and GD thread-pool startup. I excluded input generation, plots and population-reference diagnostics. Each solver had a warmup. The original tests used shuffled configuration order; the later extension ran only QuantLet in ascending size order. Other numerical-library thread counts were set to one.</p>
<p class="caption">Apple M4 Max, {result['metadata']['logical_cpus']} CPU cores, {round(result['metadata']['memory_bytes'] / 1024 ** 3)} GB RAM. Python {result['metadata']['python'].split()[0]}, NumPy {result['metadata']['numpy']}. The machine was not reserved exclusively. The fixed-size comparisons use three runs per configuration; the size sweep uses one. These tests do not establish a universal best thread count.</p>
<p>My current 12-thread GD setting is already close to the fastest result. QuantLet remains a single-threaded baseline.</p>
<details><summary>Exact timings and reproducibility files</summary><div class="table-wrap"><table><thead><tr><th>Case</th><th>Solver</th><th>Threads</th><th>Median seconds</th><th>Observed range</th></tr></thead><tbody>{raw_rows}</tbody></table></div>
<p><a href="solver-speed-files/solver_speed_comparison.ipynb">Executed notebook</a> · <a href="solver-speed-files/benchmark.json">Every timing and source hash</a> · <a href="solver-speed-files/benchmark_summary.csv">Summary CSV</a> · <a href="solver-speed-files/benchmark_support.py">Benchmark adapters</a> · <a href="solver-speed-files/report_builder.py">Report builder</a></p>
<p>GD keeps its original 60-iteration limit and relative-update tolerance of 0.000001. Scalar search keeps its absolute interval tolerance of 0.000001. All component differences must be below 0.0001. Their stopping rules differ, so numerical agreement is checked separately. Lookup-table creation and imports happen before timing. Minimum GD thread chunk size is 64 points.</p>
{provenance}
<p><a href="https://github.com/QuantLet/IRMF-Intrinsic-Robust-Multiscale-Filtering/blob/58d6762cccbdce3abe5e183ceaf9564bf46768ca/Simulations/imd_python_fixedvf.py">QuantLet source</a> · <a href="https://github.com/mkuziuk/imf/blob/d7d8a22372eec660060f744f813e5f91a37245e5/experiments/gd-irmf/gd_irmf.ipynb">My GD source</a></p>
</details></section>
<footer><a href="../">All reports</a>{earlier_comparison}Charts are embedded and work offline.</footer>
</main></div>{SCALING_SCRIPT}</body></html>'''
    if re.search(r'\{\{[^}]+\}\}', document):
        raise ValueError("Unfilled template token")
    destination = report_dir / "imf-solver-speed.html"
    destination.write_text(document)
    refresh_index(report_dir)
    return destination, pngs


def refresh_index(report_dir):
    cards = []
    for report in sorted(report_dir.glob("*.html")):
        if report.name.startswith('.'):
            continue
        title = re.search(r'<title>(.*?)</title>', report.read_text(), re.S)
        title = title.group(1) if title else html.escape(report.stem)
        cards.append(f'<a class="report" href="reports/{quote(report.name)}"><strong>{title}</strong><span>{html.escape(report.name)}</span></a>')
    (report_dir.parent / "index.html").write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>imf reports</title><style>body{color:#edf0f2;background:#202225;font:17px/1.6 system-ui;margin:3rem auto;padding:0 1.5rem;max-width:60rem}.report{display:block;background:#292c31;border:1px solid #454a52;border-radius:12px;padding:1.3rem;margin:1rem 0;color:#8ab4f8;text-decoration:none}.report span{display:block;color:#b2b8c0;font-size:.85rem}a:focus-visible{outline:2px solid #8ab4f8;outline-offset:4px}</style></head><body><h1>imf reports</h1>''' + ''.join(cards) + '</body></html>')


if __name__ == "__main__":
    result = json.loads((OUTPUT / "benchmark.json").read_text())
    print(build_report(result)[0])
