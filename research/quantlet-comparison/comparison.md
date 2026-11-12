Comparison of QuantLet IRMF and this imf repository, 17 September 2026

Both repositories implement the same family of recursive robust local-location smoothers. QuantLet contributes a useful population-reference simulation and score-linearization experiment. Your repository has a faster vectorized solver, more observation models and baselines, and substantially more work on recursive error propagation. Their published plots cannot directly establish better or worse performance than yours because the kernels, reference targets, signals, noise laws, bandwidths and loss parameters differ.

The QuantLet repository was cloned to [the sibling checkout](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering). This review pins QuantLet at `58d6762cccbdce3abe5e183ceaf9564bf46768ca` and imf at `d7d8a22372eec660060f744f813e5f91a37245e5`. It also reads the existing, untracked theory work under `research/prop41-lemma42`. No existing source, notebook or working-tree change was edited. The new comparison artifacts are in this directory.

**What is in each project.** QuantLet has 83 tracked files, largely figures. Its main executable is [Simulations/imd_python_fixedvf.py](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering/Simulations/imd_python_fixedvf.py:159). There is an older, shorter Python implementation in [result/imd_python_fixed.py](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering/result/imd_python_fixed.py:126), a related R implementation in [result/combine2.R](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering/result/combine2.R:1), and separate signal/loss illustration scripts. The checkout contains no manuscript, package configuration, dependency manifest or automated test suite. Its folder metadata describes the simulations, with the newer submission dated 27 August 2026. The mathematical comparison therefore uses its actual algorithms and your July 3 draft, rather than attributing an unseen theorem to QuantLet.

| Topic | QuantLet | Your imf repository |
|---|---|---|
| Main purpose | Monte Carlo component error, cumulative reconstruction error, pointwise QQ plots and linearization diagnostics | Method comparisons, solver implementation, contamination sweeps, clean-reference errors, operator-energy analysis and residual distributions |
| Methods exercised | Smooth robust-location recursion | Local mean, ordinary median, weighted median, direct-score GD and lookup-score GD |
| Signals | Two periodic sine signals | Sine/bump/trend signal, more complex generated signals and noise-only controls |
| Spatial kernel | Conventional Epanechnikov | Squared-triangular kernel, often named Epanechnikov in older code |
| Local solver | Scalar golden-section minimization in Python; `optimize` in R | Batched gradient descent, optional thread chunks, later batched Monte Carlo |
| Main component reference | Pooled expected-loss approximation from repeated noisy paths | The same decomposition applied to the clean signal |
| Current main simulation defaults | CLI `n=2000`, actually 2001 grid points, `m=1000`, 8 stages, `h1=0.2`, `H=1` | Notebook-specific; GD family often 1000 points, window 501 down to 31, $H=2\sigma$; latest residual experiment uses 512 points and 400 repetitions |

**The common mathematics.** Both use

$$
\widehat S_{k,t}=\arg\min_s\sum_u w_{k,t,u}\rho_H(r_{k,u}-s),
\qquad r_{k+1}=r_k-\widehat S_k,
$$

with

$$
\begin{aligned}
\rho_H(z)&=z\operatorname{erf}\!\left(\frac{z}{\sqrt{2}H}\right)
+\sqrt{\frac{2}{\pi}}H e^{-z^2/(2H^2)},\\[4pt]
\psi_H(z)&=2\Phi(z/H)-1,\\[4pt]
\psi'_H(z)&=\frac{\sqrt{2/\pi}}{H}e^{-z^2/(2H^2)}.
\end{aligned}
$$

These are algebraically identical in the two projects. QuantLet's factor $1/h$ in the kernel and your normalization of weights do not change the minimizer, since each scales the entire local objective by a positive constant. Both proceed from coarse smoothing to finer residual details. These IMFs are algorithmic components; they are not guaranteed to equal the sine waves or other ingredients used to synthesize the signal.

For normalized weights, the objective gradient has Lipschitz bound $\sqrt{2/\pi}/H$. Your positive-score update with step $0.95\,H/\sqrt{2/\pi}$ has the correct sign and a conservative step size. QuantLet's scalar search solves the same strictly convex local problem for positive H and nonempty positive weights. The optimization strategy is an implementation difference, not a new statistical estimator when all inputs and convergence criteria match.

**The kernel difference changes both statistics and spectral behavior.** QuantLet uses

$$
K_E(u)=\tfrac34(1-u^2)_+.
$$

Your notebooks follow the displayed formula on page 4 of `IMF.pdf`, proportional to $(1-\lvert u\rvert)_+^2$. After unit-integral normalization this is

$$
K_Q(u)=\tfrac32(1-|u|)_+^2.
$$

The PDF's prefactor $3/4$ integrates to $1/2$; that scalar disappears when fitting or normalizing weights. The shape difference remains. Direct integration gives:

| Unit-integral kernel | Second moment, integral of u²K | Energy, integral of K² |
|---|---:|---:|
| QuantLet Epanechnikov | 1/5 | 3/5 |
| Your squared triangular | 1/10 | 9/10 |

For a linear local mean at the same small bandwidth, the usual leading smooth-signal bias coefficient is half as large with your kernel, while its white-noise variance constant is 1.5 times as large. This is a bias/variance tradeoff, not a universal performance ordering. Robust recursive components require their own calculation.

There is also a theoretical consequence. On a proper regular periodic grid, the conventional Epanechnikov smoother can have negative Fourier eigenvalues. The diagnostic at 256 points and radius 50 gives $\lambda_{\min}(W)=-0.0808323$, hence $\lVert I-W\rVert_2=1.0808323$. Your kernel gives a minimum eigenvalue of `0.00059988` and $\lVert I-W\rVert_2=0.99940012$ in the same test. These are linear smoother calculations, not eigenvalues of a fixed robust operator.

Thus the positive-spectrum and residual-contraction claim in draft Lemma 2.2 cannot simply be transferred to their kernel. Row-stochasticity alone does not prove that claim. Analytically, the normalized continuous Fourier transforms are $3(\sin z-z\cos z)/z^3$ for Epanechnikov and $6(z-\sin z)/z^3$ for squared triangular; the former changes sign and the latter is nonnegative. This is a derived comparison, supported by the exact discrete check in [compare.py](/Users/mikhail/Projects/imf/research/quantlet-comparison/compare.py).

**Their error target differs from yours.** Their pooled component is

$$
S^{\mathrm{pool}}_{k,t}=\arg\min_s\sum_u K_{h_k}(t-u)
\frac1m\sum_{j=1}^m\rho_H(r^{(j)}_{k,u}-s).
$$

This approximates the expected-noisy-loss definition in draft Sections 2.4 and 3.2. It is not the average of the separately fitted components. They first update each path with its own fitted component, then pool the resulting residuals at the next stage. That is consistent with the population target based on the law of the recursively computed residuals. See [the pooling and update code](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering/Simulations/imd_python_fixedvf.py:235).

Most of your component-error notebooks instead compare $\widehat S_k(Y)$ with $\widehat S_k(X)$, using the same H and kernel on clean data. This measures departure from the clean algorithmic decomposition. It is a valid target, but generally differs from the draft's population IMF. Even under independent Gaussian noise, Gaussian convolution gives the exact first-stage identity

$$
\mathbb E\rho_H(a+\sigma Z)=\rho_{\sqrt{H^2+\sigma^2}}(a),\qquad Z\sim N(0,1).
$$

The expected-loss target therefore fits the clean signal with a wider loss parameter. For the two-sine signal, a common Epanechnikov bandwidth of `50/256`, `H=0.1` and `sigma=0.4`, the clean fit and exact Gaussian population fit differ by RMSE `0.06630`, with maximum difference `0.14385`. This example isolates the target difference without Monte Carlo error. A constant signal under symmetric noise is a special case where the targets agree. Later recursive noise is dependent and the same simple Gaussian identity generally no longer applies.

QuantLet's pooled reference is itself estimated using the same m paths whose errors it reports. It is not exact population truth. With quadratic loss this self-inclusion multiplies the component-error variance by $1-1/m$ relative to a known population target. The robust case has no universal exact factor. Our actual one-path run reports component error identically zero at all three stages, because the path and pooled objectives coincide, although the observations remain noisy. At `m=1000` the simple quadratic effect is small; the design still needs a reference-convergence check. An analytic first-stage target or a separate large reference ensemble would make calibration clearer.

**The observation models test different robustness questions.** QuantLet has two modes. Zero replacement uses $Y=(1-B)X$, with contamination probabilities 0.2 and 0.4. This produces noise $-BX$ with nonzero, signal-dependent mean. Its Gaussian mixture uses additive centered noise with component standard deviations 0.1 and either 0.4 or 0.7, with contamination probability 0.2. See [the generators](/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering/Simulations/imd_python_fixedvf.py:364).

Your early baseline and robust-gradient-descent notebooks use Gaussian noise plus one-sided exponential contamination. The GD-IRMF and later additive notebooks multiply exponential magnitudes by random signs. Your masked variants replace the signal at contaminated locations with Gaussian plus signed exponential noise. These are not identical to either QuantLet mode. In particular, the root README's one-sided description does not cover all current experiments. The newest [residual-distribution notebook](/Users/mikhail/Projects/imf/output/jupyter-notebook/residual-distribution-comparison.ipynb) describes signed contamination correctly and feeds both methods the same observations.

QuantLet's `H=1` is ten times the uncontaminated Gaussian standard deviation. Your later experiments usually choose $H=2\sigma$; the early robust notebook explores smaller multiples as well. Their fit is consequently closer to a local mean around small residuals. For a constant clean signal equal to 1 under 40% zero replacement, the exact population fit is approximately `0.60834` at H=1, and `0.90326` at H=0.1. A small estimation error around their population target does not imply recovery of the original value 1. Their separate cumulative-error plots are therefore necessary.

**Their linearization diagnostics are worth adapting.** The main script compares six approximations, including `pdf_linear`, which evaluates the score and curvature at the pooled local target. In notation with residual paths R and target S, its form is

$$
L_{j,t}=\frac{\sum_u w_{t,u}\{\psi_H(R_{j,u}-S_t)-\overline\psi_{u,t}\}}
{\sum_u w_{t,u}\overline{\psi'_H(R_{j,u}-S_t)}}.
$$

This has the positive sign required by the estimating equation. It agrees with the sign correction in your [Proposition 4.1 reconstruction](/Users/mikhail/Projects/imf/research/prop41-lemma42/IMF_Prop41_Lemma42.md:71), rather than the incorrect sign printed in the draft proposition. Despite its name, `pdf_linear` follows the expansion around S from Section 4; it is not a literal implementation of Section 3.1's evaluation at X. `L_pop` and related variants instead use scores of residual noise, dropping the local signal-minus-component offset, so their agreement needs testing.

In our small `n=48`, actually 49 points, `m=8`, three-stage Gaussian-mixture run, `pdf_linear` has correlations `0.99969`, `0.99986`, `0.99982` with component error. Its approximation RMSEs are `0.00133`, `0.000414`, `0.000587`. These are descriptive checks using shared estimated targets and curvature; they do not establish an asymptotic theorem or independent calibration. The source's stored summary pools locations and paths for correlation, while the QQ plots examine repetitions at a fixed location near t=0.4.

Your theory reconstruction explicitly restricts its result to a fixed location with stated curvature and concentration conditions. Neither the upstream experiments nor the incomplete proposition in `IMF.pdf` resolves general recursive dependence or proves a uniform-in-time robust bound. For sensitivity to the original data, a robust stage has Jacobian

$$
\begin{aligned}
(J_k)_{t,u}&=\frac{w_{k,t,u}\psi'_H(r_{k,u}-\widehat S_{k,t})}
{\sum_v w_{k,t,v}\psi'_H(r_{k,v}-\widehat S_{k,t})},\\[6pt]
D_Y\widehat S_k&=J_k(I-J_{k-1})\cdots(I-J_1).
\end{aligned}
$$

That product explains why linear commuting-filter arguments cannot be applied directly to robust recursion.

**Error scaling must name the quantity being scaled.** QuantLet computes `n_time*h_k*mean(component_error**2)` and a separate unscaled spatial supremum. It also computes those metrics for cumulative signal-recovery error. Your older notebooks divide component errors by `a**(k/2)` before reporting RMSE, and some use the same factor for sup errors. At fixed n under an exact geometric schedule, squaring your RMSE gives the same bandwidth rate up to a constant factor. Neither normalization forces identical constants across stages or justifies a flat supremum curve.

For the linear method, define

$$
A_k=(I-W_k)\cdots(I-W_1),\qquad B_k=W_kA_{k-1}.
$$

Then component noise is $B_k\varepsilon$, residual noise is $A_k\varepsilon$, and cumulative recovery error is $-A_kX+(I-A_k)\varepsilon$. They are different quantities. The first stage is a low-pass component; later stages have additional residual factors. Your [recursive-error audit](/Users/mikhail/Projects/imf/research/first-imf-recursive-error/executive-summary.md) and [energy-scaled notebook](/Users/mikhail/Projects/imf/experiments/real-vs-calculated/gd_imf_real_vs_calculated_energy_scaled.ipynb) already address this more directly than QuantLet's bandwidth-only plots. The draft's page-6 identification of component and residual error energies is incorrect, and Lemma 2.5 gives an order statement, not equal stage constants.

**The implementations agree when matched, but neither solver should be treated as infallible.** The reproducible comparison uses 256 points without a duplicated endpoint, radius 50, H=0.4 and the same seeded observations. We pass conventional Epanechnikov weights to your original GD function, then compare it with QuantLet's original scalar fitting function.

| Diagnostic | Result |
|---|---:|
| Maximum fit difference, same kernel and inputs | 0.00000585 |
| GD iterations | 38 of 60 |
| Maximum exact weighted-score residual at GD fit | 0.00000274 |
| Maximum differences over three matched recursive stages | 0.00000585, 0.000000601, 0.000000329 |
| RMSE after changing only the kernel | 0.09840 |
| Maximum change after changing only the kernel | 0.16883 |
| Scalar local-fit time, median of three runs | 0.136 s |
| Vectorized GD time, median of three runs | 0.0404 s |

GD was about 3.4 times faster in this small single-thread check. Its windows were already constructed; neither timed method computed population targets or diagnostics. This is not an end-to-end benchmark of the two repositories. Your direct erf approximation has maximum score error `1.39e-7`, and the lookup path `1.02e-6`, on the checked standardized-residual grid. Functions called `exact_score` in your notebook use an approximation to erf, so that name overstates precision.

The default 60 iterations are not sufficient for every input. A stress window of 101 values, with its central 41 values equal to 10 and the others zero, starts at unweighted median zero. With your squared-triangular weights and H=0.1, GD returns `4.24618` after 60 iterations; the scalar solution is `9.96704`. The final update is `0.07018`, well above the `0.00000525` stopping threshold. This does not explain the previously audited first-stage error spike, but it is a concrete reason to return convergence status and use a fallback when the cap is reached. Your traced routines expose iteration history, yet still return the estimate without a failure status.

**The periodic grids need alignment.** QuantLet creates `(0:n)/n`, including both 0 and 1, then copies the observations by shifts of one. Consequently, two independent endpoint observations occupy the same periodic coordinates and the local sampling mass increases near the seam. At n=64, the unnormalized first-stage kernel mass ranges from `63.9954` to `67.7454`; the Frobenius norm of the commutator of two linear smoothing matrices is `0.01553`. With `arange(64)/64`, the mass is constant and the commutator norm is `2.14e-16`. The endpoint issue changes the precise regular-design premise; it does not invalidate the existence of the fits. Their extension clipped to `[-0.5,1.5]` also requires care if h exceeds 0.5.

Your `np.pad(..., mode='wrap')` filters are shift-invariant in the sample index. However, many notebooks label the grid with both 0 and 1, and some generated signals include a nonperiodic trend. This is a different seam issue. For a matched periodic experiment, both projects should use the same endpoint-exclusive grid and physical half-bandwidth. A full window of 501 samples corresponds to radius 250; it is not a half-bandwidth of `501/n`.

**Code organization and reproducibility.** Your lookup-based core GD functions are copied across six notebooks; AST comparison confirms identical function bodies in the GD and real-vs-calculated variants. The residual-distribution notebook adds another batch implementation. Extracting shared kernels, scores, solvers and observation generators into a module would make comparisons and fixes more reliable. Notebook assertions and your numerical diagnostics provide useful checks, but the repository is not yet a tested installable library.

QuantLet's command-line simulation is convenient, and its NPZ export preserves many diagnostics. It also nests Python loops over stages, paths, targets and scalar objective evaluations; `np.vectorize(math.erf)` still calls Python for individual values. At the CLI defaults, retained dense float64 component, residual, diagnostic and cumulative arrays alone require about 1.8 GB per scenario, before temporary arrays. Running the default two-case pair retains both. Batching and making diagnostics optional would help. Older entry points have different defaults, so folder metadata alone is insufficient to reproduce a run. The main Matplotlib boxplot routines construct title strings but never apply them, and pairwise y-axis limits come only from the second case, which can clip the first case if it has larger values. Those are presentation issues separate from the estimator.

The existing unpaired linear-versus-robust panels in your older real-vs-calculated notebooks use different contamination conditions, so they are not a controlled method ranking. Your latest residual-distribution notebook fixes that design by sharing the observations, using 400 repetitions, distinguishing raw residual from clean-centered residual perturbation, and checking exact linear variance. It should be treated as part of your current work, not overlooked in favor of the older demos.

**What I would take from their work.** Keep your vectorized solver and operator-energy diagnostics. Add their expected-loss population target and target-centered score expansion as explicit optional diagnostics. Before claiming agreement or superiority, run both methods on identical observations with an explicitly named kernel, identical H and bandwidths, and three separately labeled outputs: population-component error, clean-decomposition error, and cumulative signal-recovery error. For Gaussian noise, use the analytic widened-H population objective at stage 1; for later stages, check convergence of an independent reference ensemble. Add solver convergence status and a bounded scalar fallback before broad parameter sweeps.

**What was actually run.** [compare.py](/Users/mikhail/Projects/imf/research/quantlet-comparison/compare.py) loads original upstream functions and extracts original function definitions from your GD notebook without executing its demos. It checks solver agreement, a three-stage matched recursion, score approximation, exact discrete spectra, periodic-grid weights, analytic population targets, the finite-iteration stress case, and the real upstream multi-path decomposition. Reconstruction error was at most `2.22e-16` in the small upstream checks. We also ran its CLI with `--n 48 --m 8 --K 3 --t 2 --tc 1`; it successfully wrote an NPZ file and five figures. The full 2000-by-1000 simulation and all existing notebooks were not rerun, and the R implementation was read rather than executed.

All numerical results are in [results.json](/Users/mikhail/Projects/imf/research/quantlet-comparison/results.json). Rerun from the imf root with:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/quantlet-comparison/compare.py
```

![Kernel shapes, matched solver differences, and discrete spectra](/Users/mikhail/Projects/imf/research/quantlet-comparison/comparison.png)
