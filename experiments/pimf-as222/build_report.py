"""Build concise figures and LaTeX from completed measurements, without retiming."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/pimf-as222"
FIGURES = ROOT / "overleaf/figures/pimf-as222"
BLUE = "#0075b5"
ORANGE = "#d45e00"
GRAY = "#767c83"


def sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def make_figures(results):
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves = np.load(OUTPUT / "parallel_curves.npz")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253c50", "axes.edgecolor": "#a9b3bc",
        "xtick.color": "#42576b", "ytick.color": "#42576b",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(5, 2, figsize=(7.15, 8.2), sharex=True, sharey="row")
    for k in range(5):
        for column in range(2):
            ax = axes[k, column]
            ax.plot(curves["t"], curves["truth"][k], color=GRAY, lw=0.8, label="True sine")
            for name, color, style, label in [("pimf", BLUE, "-", f"pimf, {results['settings']['workers']} workers"), ("fft", ORANGE, "--", "AS222-style FFT")]:
                values = curves[name][:, k].mean(axis=0) if column == 0 else curves[name][0, k]
                ax.plot(curves["t"], values, color=color, lw=1.0, ls=style, label=label)
            ax.set_xlim(0, 1)
            ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
            ax.tick_params(labelsize=8)
            if column == 0:
                ax.set_ylabel(f"Component {k + 1}")
            ax.set_title(f"f = {results['settings']['frequencies'][k]}, h = {results['settings']['bandwidths'][k]:.5f}", fontsize=9, loc="left", pad=4)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    fig.text(0.08, 0.972, "Mean of 20 paired realizations", fontsize=11, color="#17354e")
    fig.text(0.56, 0.972, "First paired realization", fontsize=11, color="#17354e")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.945), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.06, top=0.89, hspace=0.40, wspace=0.17)
    fig.savefig(FIGURES / "components.pdf")
    fig.savefig(FIGURES / "components.png", dpi=180)
    plt.close(fig)

    timing = results["timing"]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    n = np.array([row["n"] for row in timing])
    for method, color, label in [("pimf_1", GRAY, "pimf, 1 worker"), ("pimf_max", BLUE, f"pimf, {results['settings']['workers']} workers"), ("fft", ORANGE, "AS222-style FFT")]:
        med = np.array([row[method + "_median_ms"] for row in timing])
        low = np.array([row[method + "_min_ms"] for row in timing])
        high = np.array([row[method + "_max_ms"] for row in timing])
        ax.plot(n, med, "o-", color=color, lw=1.5, ms=4, label=label)
        ax.fill_between(n, low, high, color=color, alpha=0.17)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n, [str(value) for value in n], fontsize=8)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Five-component runtime, ms")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES / "runtime.pdf")
    fig.savefig(FIGURES / "runtime.png", dpi=180)
    plt.close(fig)


def build_tex(r):
    timing = next(row for row in r["timing"] if row["n"] == 1000)
    accuracy_rows = []
    for row in r["pilot"]["summary"]:
        accuracy_rows.append(
            f"{row['component']} / {row['frequency']} & {row['pimf_fft_correlation']:.4f} & "
            f"{row['difference_rms']:.5f} & {100 * row['relative_difference']:.1f}\\% & "
            f"{row['pimf_true_correlation']:.4f} & {row['fft_true_correlation']:.4f} \\\\"
        )
    timing_rows = [f"{row['n']:,} & {row['pimf_1_median_ms']:.2f} & {row['pimf_max_median_ms']:.2f} & {row['fft_median_ms']:.3f} & {row['parallel_speedup']:.2f}$\\times$ \\\\" for row in r["timing"]]
    setup_rows = [f"{k + 1} & {f} & {w:.2f} & {h:.5f} \\\\" for k, (f, w, h) in enumerate(zip(r['settings']['frequencies'], r['settings']['amplitudes'], r['settings']['bandwidths'], strict=True))]
    check = r["validation"]
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm,top=18mm,bottom=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,xcolor,microtype,fancyhdr}
\usepackage[colorlinks=true,urlcolor=teal,linkcolor=teal]{hyperref}
\hypersetup{pdftitle={IRMF with pimf versus AS222-style FFT},pdfsubject={Reproduction of the supplied five-sine benchmark with pimf 0.4.0}}
\definecolor{navy}{HTML}{17354E}
\definecolor{teal}{HTML}{007C91}
\definecolor{muted}{HTML}{516578}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.3pt}
\fancyfoot[L]{\footnotesize\color{muted}pimf / AS222-style FFT \enspace | \enspace Five-sine reproduction}
\fancyfoot[R]{\footnotesize\color{muted}\thepage}
\newcommand{\heading}[1]{\vspace{7pt}{\large\color{teal}#1}\par\vspace{2pt}}
\newcommand{\pagetitle}[1]{{\LARGE\color{navy}#1}\par\vspace{4pt}}
\begin{document}
{\footnotesize\color{muted}RESEARCH COMPARISON \hfill 13 October 2026}\par\vspace{6pt}
\pagetitle{IRMF with \texttt{pimf} versus AS222-style FFT}
{\color{muted}The supplied five-sine benchmark, reproduced with \texttt{pimf 0.4.0}.}

\heading{Result}
The package reproduces all 25 source-report accuracy entries within their displayed rounding.
The FFT comparator still fails the agreement checkpoint in all 100 component pairs.
At $n=1000$, \texttt{pimf} takes \textbf{@@PIMFMS@@\,ms with @@WORKERS@@ workers}, versus
@@SERIALMS@@\,ms with one worker and @@FFTMS@@\,ms for FFT.
Using all cores slows this small problem; at $n=8000$, it gives a \textbf{@@LARGESPEED@@-fold speedup}.

\heading{Same experiment, packaged local solver}
Use $t_i=i/n$, $n=1000$, and
\[
Y_i=\sum_{j=1}^{5} w_j\sin(2\pi f_jt_i)+\varepsilon_i,
\qquad \varepsilon_i\overset{\mathrm{iid}}{\sim}N(0,0.2^2).
\]
\begin{minipage}[t]{0.45\linewidth}
\vspace{0pt}
\centering\small
\begin{tabular}{rrrr}
\toprule Component & $f_j$ & $w_j$ & $h_j$\\\midrule
@@SETUP@@
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.51\linewidth}
\vspace{0pt}
\small
20 paired realizations; seed 20261004; one validation draw first.
Classical Epanechnikov weights, periodic boundaries, robustness $H=1$.
Each method subtracts its own component before the next bandwidth.
No retuning or contamination is added.
\end{minipage}

\texttt{pimf} uses \texttt{IMF}, \texttt{SmoothAbs(1)}, and \texttt{Epanechnikov()}.
One public \texttt{decompose(..., h1=h, k\_max=1)} call per bandwidth preserves the
nongeometric schedule and its exact fractional widths. Settings are
\texttt{lookup=True}, \texttt{tol=1e-6}, \texttt{max\_iter=60}, and \texttt{workers=@@WORKERS@@}.

\heading{Component agreement and true-sine recovery}
\small
\begin{center}
\begin{tabular}{rrrrrr}
\toprule
Component / $f$ & $r(P,F)$ & RMS$(P-F)$ & Relative RMS & $r(P,T)$ & $r(F,T)$\\\midrule
@@ACCURACY@@
\bottomrule
\end{tabular}
\end{center}
\normalsize
$P$: \texttt{pimf}; $F$: FFT; $T$: corresponding weighted true sine.
Entries average metrics computed separately for each realization.
Relative RMS is $\operatorname{RMS}(P-F)/\operatorname{RMS}(P)$.
The inherited checkpoint requires $r(P,F)\geq0.995$ and relative RMS $\leq5\%$
in \emph{every} pair. It fails, so the conditional 1,000-run extension is not triggered.

\heading{Why changing the implementation does not close the gap}
Let $K$ be the common discrete kernel smoother and
$\psi_H(u)=\operatorname{erf}(u/(\sqrt{2}H))$.
Direct IRMF solves $\sum_iq_{ti}\psi_H(R_i-x_t)=0$.
The comparator instead uses
\[
p=KR,\qquad S=p+\frac{K\psi_H(R-p)}{K\psi'_H(R-p)}.
\]
Its scores are centered at observation-specific pilots $p_i$.
For a linear score, it reduces to $(2K-K^2)R$, whereas direct smoothing gives $KR$.
The discrepancy is in the estimator; FFT arithmetic is accurate.

\newpage
\pagetitle{Recovered components}
{\color{muted}Same observations and bandwidths; each method uses its own residual recursion.}
\begin{center}
\includegraphics[width=\linewidth]{figures/pimf-as222/components.pdf}
\end{center}
Component 2 has the largest relative disagreement, @@COMP2@@\% on average.
The mean curves expose persistent attenuation and leakage; the single-run curves show noise variation.
The correlations on page 1 are per-realization averages, not correlations of the mean curves.

\newpage
\pagetitle{Runtime, checks, and original sources}
\heading{Fresh timing measurements}
\begin{minipage}[c]{0.49\linewidth}
\includegraphics[width=\linewidth]{figures/pimf-as222/runtime.pdf}
\end{minipage}\hfill
\begin{minipage}[c]{0.48\linewidth}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrr}
\toprule $n$ & 1 worker & @@WORKERS@@ workers & FFT & Speedup\\\midrule
@@TIMING@@
\bottomrule
\end{tabular}
\par\vspace{4pt}Times in ms; speedup = 1-worker / @@WORKERS@@-worker time.
\end{minipage}

Five warmed repetitions per size, with all three configurations retimed in randomized order.
Timing includes five smoothing calls, thread-pool creation, and residual updates. Kernel weights
and FFT spectra are precomputed; simulation, setup, checks, and plotting are excluded.
Shading shows the observed range. Cached and uncached package outputs are bit-identical.

Apple M4 Max, 16 cores; Python 3.14.6; NumPy 2.5.4; SciPy 1.18.1.
Numerical-library threads remain at one. \texttt{workers=16} gives pool capacities of 4, 8,
and 16 for $n=250$, $500$, and $n\geq1000$, because \texttt{pimf} requires at least 64 samples
per chunk. Parallel overhead outweighs gains at small $n$; all measured FFT times remain lower.

\heading{Numerical checks}
\small
\begin{tabularx}{\linewidth}{@{}Xr@{}}
\toprule Check & Maximum absolute error\\\midrule
@@WORKERS@@-worker \texttt{pimf} versus 25 exact-score Brent roots & $@@ROOTERR@@$\\
FFT versus spatial evaluation of the same one-step formula & $@@FFTERR@@$\\
Package versus independently sampled kernel weights & $@@KERNELERR@@$\\
Default package versus tighter solver with score lookup disabled & $@@TIGHTERR@@$\\
Reconstruction across all 20 paired runs & $@@RECONERR@@$\\
One versus @@WORKERS@@ workers across all 20 paired runs & $@@PARALLELERR@@$\\
\bottomrule
\end{tabularx}
\par\vspace{3pt}
\normalsize
All 100 parallel package stages converged, with at most @@MAXITER@@ iterations.
Thread chunks stop independently, explaining small differences near solver tolerance.
The root and tight-solver checks use the separate validation realization.

\heading{What the original papers establish}
H\"ardle's AS222 paper is from \textbf{1987}; the \textbf{1988} paper is Dusoir's correction.
Equation 2 of AS222 gives the pilot-plus-score-ratio construction, using Huber scores and Gaussian
smoothing. Page 106 explicitly notes the changed score centers. Dusoir corrects initialization
of \texttt{WK(5:6,:)} and \texttt{M1=M/2}. Both full publisher PDFs were retrieved and checked.

This experiment preserves the supplied report's Gaussian-score, Epanechnikov-kernel adaptation,
with an unbinned periodic grid. It does not reproduce historical Fortran outputs or the original
contaminated random-design example. Modern kernel transforms are explicitly initialized and inverse
FFTs specify their length. Twenty Gaussian-noise realizations establish this comparison only;
they do not test resistance to outliers or automatic bandwidth selection.

\heading{Reproduce and inspect}
\small
The experiment README at \path{experiments/pimf-as222/README.md} gives the commands and
pinned dependencies. Raw paired metrics, timing repetitions, curves, solver diagnostics,
checks, and source hashes are in \path{output/pimf-as222/}.

\vspace{3pt}
\textbf{Sources.} Supplied comparison PDFs dated 4 and 8 October 2026.
W. H\"ardle, \emph{Resistant Smoothing Using the Fast Fourier Transform}, 1987, 36:104--111,
\href{https://doi.org/10.2307/2347850}{doi:10.2307/2347850}.
T. Dusoir, \emph{Remark AS R73}, 1988, 37:316--317,
\href{https://doi.org/10.2307/2347358}{doi:10.2307/2347358}.
\href{https://pypi.org/project/pimf/0.4.0/}{pimf 0.4.0, PyPI}.
The source reports' code and saved arrays were not supplied; matching their rounded tables
does not establish bitwise reproduction of their curves.
\end{document}
"""
    replacements = {
        "PIMFMS": f"{timing['pimf_max_median_ms']:.2f}",
        "SERIALMS": f"{timing['pimf_1_median_ms']:.2f}",
        "FFTMS": f"{timing['fft_median_ms']:.3f}",
        "WORKERS": str(r['settings']['workers']),
        "LARGESPEED": f"{next(row for row in r['timing'] if row['n'] == 8000)['parallel_speedup']:.2f}",
        "SETUP": "\n".join(setup_rows), "ACCURACY": "\n".join(accuracy_rows),
        "TIMING": "\n".join(timing_rows),
        "COMP2": f"{100 * r['pilot']['summary'][1]['relative_difference']:.1f}",
        "ROOTERR": sci(check['root_max_abs_error']),
        "FFTERR": sci(check['fft_spatial_max_abs_difference']),
        "KERNELERR": sci(check['kernel_max_abs_difference']),
        "TIGHTERR": sci(check['default_vs_tight_direct_max_abs']),
        "RECONERR": sci(r['pilot']['reconstruction_max_abs']),
        "MAXITER": str(r['pilot']['max_solver_iterations']),
        "PARALLELERR": sci(check['serial_parallel_max_abs_all_20_runs']),
    }
    for key, value in replacements.items():
        tex = tex.replace("@@" + key + "@@", value)
    assert "@@" not in tex
    (ROOT / "overleaf/pimf_as222_comparison.tex").write_text(tex)


def main():
    results = json.loads((OUTPUT / "results.json").read_text())
    parallel = json.loads((OUTPUT / "parallel_results.json").read_text())
    results = {**results, **parallel, "validation": {**results["validation"], **parallel["validation"]}}
    make_figures(results)
    build_tex(results)
    print("Built figures and overleaf/pimf_as222_comparison.tex")


if __name__ == "__main__":
    main()
