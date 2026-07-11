## Summary

The first-stage spike is predicted by the exact finite-sample linear operators; it is not a numerical accident. With the notebook's periodic boundary, weights, window schedule, $n=1000$, and $\sigma=0.4$,

\[
e_1=W_1\varepsilon,
\qquad
e_k=W_k\prod_{j<k}(I-W_j)\varepsilon\quad(k>1).
\]

Stage 1 is a low-pass filter of the original noise. Every later component contains at least one residual/high-pass factor and is band-pass: it has zero DC gain and much less operator energy. An exact FFT calculation gives scaled recursive RMS $0.020182$ for stage 1 but a stage-2-to-9 mean of $0.006776$, a factor of $2.98$. The stored notebook values, $0.020249$ and $0.006776$-scale later values, are therefore the expected shape of this recursion.

The same calculation distinguishes the statement that is approximately true from the one that is not. For the single-pass filters $W_k\varepsilon$, exact RMS divided by $a^{k/2}$ has mean $0.020266$ and coefficient of variation $0.65\%$ over all nine stages. For the recursive components, the bandwidth rate remains useful, but its multiplicative operator-energy constant changes sharply between the initial low-pass term and subsequent detail terms. Scaling by bandwidth alone cannot make those constants equal.

Monte Carlo, boundary alternatives, and robust controls all support this explanation:

- The seed-777 linear stage-1 RMSE is at the 58.9th percentile of 5,000 iid-Gaussian simulations, so it is not an unusually bad realization.
- The current `wrap` boundary gives exactly constant pointwise variance at every location. Switching to `reflect` changes stage-1 domain RMS by only $+6.7\%$ and leaves the scaled first-stage/later-stage gap intact.
- Across 120 robust Monte Carlo trials, the scaled stage-1/later-stage ratio is $2.84$ with Gaussian noise only and $2.82$ with the notebook's $p=0.2$ contamination. Contamination raises error levels but does not create the first-stage discontinuity.
- The robust seed-777 fits converge in 5--11 iterations, far below `max_iter=60`; failed optimization is not the cause.

Numerically, the theory-compatible verdict is: $a^{k/2}$ is an excellent finite-sample normalizer for the implemented **single-pass RMS**, and it captures the order of later recursive RMS values, but it does not imply one common constant for all recursive components. It also does not flatten sup errors; the effective number and correlation of extrema change with bandwidth.

## Key Evidence

### Exact finite-$n$ variance

Under `boundary="wrap"`, every $W_k$ is a symmetric circulant convolution. Let $w_k(\omega_m)$ be its DFT multiplier and

\[
q_k(\omega_m)=w_k(\omega_m)\prod_{j<k}\{1-w_j(\omega_m)\}.
\]

For iid noise with covariance $\sigma^2I$, circulant symmetry and Parseval give

\[
\mathbb E\{\operatorname{RMSE}(e_k)^2\}
=\frac{\sigma^2}{n}\sum_m|q_k(\omega_m)|^2
=\sigma^2\|\ell_k\|_2^2,
\]

where $\ell_k$ is the periodic impulse response. This is an exact discrete moment, not an asymptotic approximation. The resulting values are:

| stage | window | exact single-pass RMS / $a^{k/2}$ | exact recursive RMS / $a^{k/2}$ | recursive variance / single-pass variance |
|---:|---:|---:|---:|---:|
| 1 | 501 | 0.020182 | 0.020182 | 1.000 |
| 2 | 355 | 0.020169 | 0.008278 | 0.168 |
| 3 | 251 | 0.020182 | 0.007020 | 0.121 |
| 4 | 177 | 0.020227 | 0.006663 | 0.109 |
| 5 | 125 | 0.020264 | 0.006502 | 0.103 |
| 6 | 89 | 0.020229 | 0.006356 | 0.099 |
| 7 | 63 | 0.020270 | 0.006365 | 0.099 |
| 8 | 45 | 0.020240 | 0.006303 | 0.097 |
| 9 | 31 | 0.020628 | 0.006720 | 0.106 |

Thus the first residual factor cuts stage-2 variance to $16.8\%$ of its single-pass counterpart; the full preceding-factor product leaves only about $9.7\%$--$12.1\%$ at later stages. This is much too large and systematic an effect to attribute to rounding or random variation.

The DC gains provide an even simpler diagnostic. Stage 1 has $q_1(0)=1$, while $q_k(0)=0$ for every $k>1$, because each normalized smoother satisfies $w_j(0)=1$. Stage 1 retains the coarse/constant noise mode; every later recursive component removes it. The operators are qualitatively different before any scalar normalization is applied.

An adjacent-factor ablation shows where most of the reduction enters. Stage 2's full operator is exactly $W_2(I-W_1)$. At stage 3 the full preceding product has $84.6\%$ of the standard deviation of the adjacent-only operator $W_3(I-W_2)$; this ratio approaches about $76\%$ later. The first residual subtraction creates the main discontinuity, and older residual factors provide additional suppression.

### Reproduction and Monte Carlo variability

The diagnostic implementation reproduces all notebook RMSEs to the displayed six decimal places. In particular:

- linear recursive raw RMSE: $0.024080, 0.010586, 0.009927, 0.010339, 0.013394, 0.018929, 0.024960, 0.026529, 0.033173$;
- robust contaminated recursive raw RMSE: $0.034567, 0.011224, 0.012278, 0.013467, 0.015742, 0.019772, 0.025637, 0.029977, 0.038813$.

For 5,000 fresh iid-Gaussian vectors, the exact linear stage-1 quantity is
$\sqrt{\mathbb E\operatorname{RMSE}^2}=0.024000$. The Monte Carlo mean of the RMSE itself is $0.022906$, which is slightly lower by Jensen's inequality because the coarse filter leaves few effective degrees of freedom. The seed-777 value $0.024080$ has percentile 58.9 and lies comfortably inside the 5--95% interval $0.012475$--$0.035661$. All nine recursive seed values lie inside their corresponding Monte Carlo 5--95% intervals. The later deviations above and below the exact curve are ordinary one-realization variability.

The sup metric behaves differently. In the same simulation, mean single-pass sup error divided by $a^{k/2}$ increases from $0.03779$ at stage 1 to $0.06180$ at stage 9 even though the exact pointwise/RMS scale is nearly flat. Shorter windows produce shorter correlation length and more effectively independent opportunities for a maximum. A bandwidth-only pointwise standard-error normalization is therefore not a constant-sup-error normalization.

### Boundary checks

With the notebook's periodic padding, the exact single-pass pointwise standard deviation is identical at all 1,000 positions for every stage; at stage 1 it is $0.02400013$ everywhere. There is no privileged numerical edge in the implemented linear experiment.

For comparison, an exact row-weight calculation under `reflect` gives stage-1 domain RMS $0.0256125$, only $1.067\times$ the periodic value. Reflection does increase endpoint standard deviation to $1.411\times$ the center value because reflected observations receive duplicate weights, but this is an alternative boundary condition, not what the notebook runs.

In 400 paired recursive simulations, the scaled stage-1-to-stage-2--9 mean ratio is $2.88$ under `wrap` and $3.09$ under `reflect`. The effect survives and slightly strengthens under reflection. Boundary handling can change local shapes and suprema, especially for the nonlinear robust signal near the periodic seam, but it does not explain the stage-1 RMS spike.

### Robust and contamination controls

The seed-777 robust Gaussian-only control, using the same clean reference and $H=0.8$, has scaled recursive RMSE

\[
0.021263, 0.007262, 0.005875, 0.005163, 0.005756,
0.006831, 0.007514, 0.006678, 0.007009.
\]

This is nearly the same pattern as the linear Gaussian-only sequence, so neither contamination nor robust nonlinearity is required. Adding contamination raises the robust seed's first value to (0.029067), but the later stage-2--9 mean is only (0.007357).

Across 120 independent trials:

| robust case | mean scaled stage 1 | mean scaled stages 2--9 | ratio |
|---|---:|---:|---:|
| Gaussian only | 0.018807 | 0.006627 | 2.84 |
| Gaussian + $p=0.2$, scale-0.6 contamination | 0.023112 | 0.008205 | 2.82 |

The robust map has no fixed global convolution operator, so these are empirical controls rather than an exact variance formula. They are nevertheless strong evidence that the same initialization-versus-residual distinction persists after local linearization. The seed-777 convergence audit found 5--7 iterations for the clean stages, 8--9 for Gaussian-only stages, and 9--11 for contaminated stages, with terminal updates near the specified tolerance. The first-stage error is not a `max_iter` or lookup-grid failure.

### Stage indexing and normalization

The schedule is approximately

\[
h_k=h_1a^{-(k-1)},\qquad a=\sqrt2,
\]

so a standard deviation proportional to $h_k^{-1/2}$ is naturally normalized by $a^{(k-1)/2}$ when stage 1 is the baseline. The notebook divides by $a^{k/2}$. These two normalizations differ by the same factor $a^{1/2}$ at every stage:

\[
\frac{z_k}{a^{k/2}}=a^{-1/2}\frac{z_k}{a^{(k-1)/2}}.
\]

Changing between them rescales the whole curve vertically and cannot create, remove, or reduce the relative stage-1 spike. Odd-window rounding explains the small residual variation in exact single-pass scaled RMS, particularly the final `45 -> 31` ratio of 1.452 rather than exactly $\sqrt2$. A fully schedule-aware single-pass normalizer is $\sigma\sqrt{\sum_jw_{k,j}^2}$; a recursive linear normalizer is the exact component-operator energy, not bandwidth alone.

## Method or Context Details

The runnable script duplicates the notebook's signal, observation, kernel, lookup score, linear recursion, and robust recursion without importing or modifying the notebook. It writes only under `research/first-imf-recursive-error/diagnostics/`.

The main checks were:

1. construct each periodic smoother's length-1,000 impulse response and DFT multiplier;
2. form both single-pass $W_k$ and recursive $W_k\prod_{j<k}(I-W_j)$ transfer functions;
3. use Parseval for exact iid-Gaussian integrated/pointwise variance;
4. compare the exact values with the seed-777 notebook outputs and 5,000 spectral Monte Carlo replicates;
5. compute exact boundary row energies for `wrap` and `reflect`, plus 400 paired recursive simulations;
6. run same-observation linear/robust and Gaussian/contaminated controls, plus 120 robust Monte Carlo replicates per observation regime.

The first-window weights have $\sum_jw_j^2=0.0036000384$, effective sample size $277.775$, and exact stage-1 standard deviation

\[
0.4\sqrt{0.0036000384}=0.0240001.
\]

This also explains why comparing against the unweighted heuristic $0.4/\sqrt{501}=0.01787$ would incorrectly label the stage-1 value as inflated. The implemented weights are proportional to $(1-|u|)^2_+$, not the conventional Epanechnikov $1-u^2$; the code and local theory note use the former consistently, but external constants for the conventional kernel would not transfer.

The exact quantity reported in `linear_operator_exact.csv` is $\sqrt{\mathbb E\operatorname{RMSE}^2}$, while `linear_monte_carlo.csv` separately reports the Monte Carlo mean and quantiles of realized RMSE. This distinction matters most at the coarsest, highly correlated stage.

## Sources

1. [`gd_imf_real_vs_calculated.ipynb`](../../../experiments/real-vs-calculated/gd_imf_real_vs_calculated.ipynb), current working-tree version: observation generator, weights, linear/robust filters, recursion, normalization, and stored output values.
2. `/Users/mikhail/Projects/imf/research/first-imf-recursive-error/diagnostics/run_numerical_diagnosis.py`: complete reproducible calculation.
3. Generated tables in the same diagnostics directory:
   - `linear_operator_exact.csv`;
   - `linear_monte_carlo.csv`;
   - `single_pass_boundary_exact.csv`;
   - `recursive_boundary_monte_carlo.csv`;
   - `seed777_method_controls.csv`;
   - `robust_monte_carlo.csv`;
   - `summary.json`.
4. `/Users/mikhail/Projects/imf/IMF.pdf`, especially the recursive operator definition and Lemma 2.5 as audited separately in `agents/theory.md`. The lemma gives an $\asymp(nh_k)^{-1}$ order for the recursive variance diagonal, not equality of stage constants.

No external web data or third-party numerical source was used for this memo.

## Contradictions or Uncertainty

- The linear calculation is exact for iid homoscedastic noise and periodic boundaries. Correlated or heteroscedastic noise changes the covariance calculation, though it does not remove the operator distinction between stage 1 and later stages.
- `sqrt(E[RMSE^2])`, `E[RMSE]`, one realized RMSE, pointwise standard deviation, and sup error are different quantities. A claim that “scaled error is constant” is not testable until the intended quantity is specified. The PDF's variance-diagonal order does not by itself give exact realized-RMSE or sup constancy.
- The robust Monte Carlo has 120 replicates, sufficient to establish the large qualitative gap but not to estimate tail probabilities precisely. Robust errors are signal- and residual-dependent, and no exact fixed-operator analogue of the FFT calculation was assumed.
- The robust clean-reference target in the notebook is a robust fit to clean data. A population robust theorem based on expected noisy loss may target a different quantity; the numerical controls do not resolve that theoretical mismatch.
- The reflection experiment is an ablation only. The actual notebook uses `wrap`, for which the linear variance is spatially constant. Robust wrap behavior near a nonperiodic signal seam can still affect local or sup errors.
- The last window ratio and odd rounding make any geometric normalizer approximate. These finite-schedule deviations are below the stage-1 operator-constant change by an order of magnitude.
- The one-off convergence audit used the same deterministic update and stopping rule but one worker. Local windows are independent, so worker chunking can change only tolerance-scale stopping, not the $10^{-2}$ error pattern.

## Open Questions

1. Which object is intended to be constant: pointwise SD, $\sqrt{\mathbb E\mathrm{RMSE}^2}$, expected realized RMSE, or sup error? The recommended diagnostic and normalization depend on this choice.
2. Should stage 1 be treated as a separate coarse/scaling component and constancy assessed only after the recursive filter bank reaches its stage-3-to-9 plateau? Exact scaled recursive RMS over stages 3--9 has coefficient of variation only $3.63\%$.
3. If all recursive linear stages must be standardized, should the plot divide by exact $\sigma\|\ell_k\|_2$ rather than by bandwidth? That would test the white-noise model instead of assuming equal operator constants.
4. For robust stages, should a parametric bootstrap or numerical Jacobian estimate the stage-specific sandwich/operator energy? Bandwidth alone cannot account for adaptive influence weights and residual composition.
5. Should the empirical comparison rerun linear and robust methods on identical Gaussian-only and contaminated observations in the main notebook presentation? The diagnostic controls show this separation is important for method comparisons, although it is not needed to explain the first-stage spike.
6. If the target metric is sup error, what multiplicity/correlation correction is intended? The Monte Carlo results rule out $a^{k/2}$ alone as a constant-sup normalizer.
