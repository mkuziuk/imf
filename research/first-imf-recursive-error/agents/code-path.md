## Summary

The large stage-1 recursive error is the expected output of the implemented decomposition, not evidence of an indexing, padding, or optimizer failure. The first recursive component is a low-pass location estimate of the original input. Every later component is a low-pass estimate of a residual from which all earlier low-frequency estimates have already been removed. In the linear path this difference is exact:

\[
e_1=W_1\varepsilon,\qquad
e_k=W_k(I-W_{k-1})\cdots(I-W_1)\varepsilon\quad(k\ge2).
\]

Thus stage 1 is a low-pass/scaling component, whereas stages 2 onward are band-pass/detail components. The factors \((I-W_j)\) strongly reduce the later filter gains and annihilate the constant mode. Dividing all of these different operators by the single-pass normalization \(a^{k/2}\) cannot make stage 1 and the later stages share one constant.

The notebook's own numbers support this diagnosis. For the linear run, stage-1 RMSE is `0.024080`, essentially equal to the exact white-noise prediction `0.024000` for the implemented weights. After division by \(a^{k/2}\), stage 1 is `0.020249`, while the median over stages 2--9 is `0.006663` (3.04 times smaller). For the robust run those values are `0.029067` and `0.007397` (3.93 times smaller). In contrast, the *linear single-pass* scaled RMSE is nearly flat over all stages (coefficient of variation 4.68%). This is the error family for which the usual \((nh)^{-1/2}\) heuristic is relevant.

No existing project file was changed in this audit. All numerical checks were run from the current working-tree version of [`gd_imf_real_vs_calculated.ipynb`](../../../experiments/real-vs-calculated/gd_imf_real_vs_calculated.ipynb).

## Key Evidence

- **Exact implemented linear operator.** In code cell `267ea198`, `local_linear_filter` constructs a wrap-padded, symmetric, normalized convolution. If \(m_k=2r_k+1\), then

  \[
  (W_kz)_i=\sum_{q=-r_k}^{r_k}w_{k,q}z_{(i+q)\bmod n},\qquad
  w_{k,q}=\frac{(1-|q|/r_k)^2_+}{\sum_{\ell=-r_k}^{r_k}(1-|\ell|/r_k)^2_+}.
  \]

  (`0.75` cancels under normalization.) `linear_imf_with_history` starts with \(R_0=z\), appends \(S_k=W_kR_{k-1}\), and then sets \(R_k=(I-W_k)R_{k-1}\). Consequently

  \[
  R_k=(I-W_k)\cdots(I-W_1)z,\qquad
  S_k=L_kz,\qquad L_k=W_k(I-W_{k-1})\cdots(I-W_1).
  \]

  With the same clean signal in both paths, `calculated_linear["imfs"] - real_linear["imfs"]` equals `linear_imf_with_history(y_no_contamination - x_clean)["imfs"]` to `1.41e-15` max absolute error.

- **Fourier explanation of the first-stage jump.** All \(W_k\) are symmetric circulant matrices under `boundary="wrap"`, so they commute and are diagonalized by the DFT. Their component transfer functions are

  \[
  \widehat L_1(\omega)=\widehat W_1(\omega),\qquad
  \widehat L_k(\omega)=\widehat W_k(\omega)\prod_{j<k}\bigl(1-\widehat W_j(\omega)\bigr).
  \]

  Because every normalized smoother has \(\widehat W_j(0)=1\), \(\widehat L_1(0)=1\), but \(\widehat L_k(0)=0\) for all \(k>1\). Numerically, the maximum DFT gain is `1.000` for \(L_1\), `0.384` for \(L_2\), and about `0.30--0.33` for \(L_3,\ldots,L_9\). A constant perturbation is therefore an immediate counterexample to recursive-error constancy: it appears wholly in stage 1 and exactly zero in every later stage.

- **Exact white-noise calculation.** For iid Gaussian noise with variance \(\sigma^2\), wrap/circulant symmetry gives

  \[
  \mathbb E\{e_k(i)^2\}=\sigma^2\lVert \ell_k\rVert_2^2,
  \]

  where \(\ell_k\) is the impulse response of \(L_k\). The following table compares (i) the observed linear recursive scaled RMSE, (ii) the expected scaled RMSE for a single pass \(W_k\varepsilon\), and (iii) the expected scaled RMSE for the actual recursive operator \(L_k\varepsilon\). These expectations use the exact discrete weights, not an asymptotic approximation.

  | stage | window | observed recursive / \(a^{k/2}\) | expected single-pass / \(a^{k/2}\) | expected recursive / \(a^{k/2}\) |
  |---:|---:|---:|---:|---:|
  | 1 | 501 | 0.020249 | 0.020182 | 0.020182 |
  | 2 | 355 | 0.007485 | 0.020169 | 0.008278 |
  | 3 | 251 | 0.005903 | 0.020182 | 0.007020 |
  | 4 | 177 | 0.005170 | 0.020227 | 0.006663 |
  | 5 | 125 | 0.005631 | 0.020264 | 0.006502 |
  | 6 | 89 | 0.006692 | 0.020229 | 0.006356 |
  | 7 | 63 | 0.007421 | 0.020270 | 0.006365 |
  | 8 | 45 | 0.006632 | 0.020240 | 0.006303 |
  | 9 | 31 | 0.006974 | 0.020628 | 0.006720 |

  The expected recursive column already contains the stage-1 discontinuity seen in the data. The realized values are reasonable one-seed fluctuations around it. The scaled linear single-pass values are flat because \(\sum_qw_{k,q}^2\) grows approximately as \(1/r_k\), while \(r_k\) shrinks by \(a\).

- **Stage 1 is exactly the single pass in both paths.** In cell `e1ea4c6c`, the recursive and single-pass errors at stage 1 are identical by construction; there are no prior residual factors yet. The divergence begins only at stage 2. Raw/scaled RMSEs are:

  | case | stage 1 raw / scaled | stage 2 raw / scaled | median scaled, stages 2--9 |
  |---|---:|---:|---:|
  | Gaussian only / linear | 0.024080 / 0.020249 | 0.010586 / 0.007485 | 0.006663 |
  | contaminated / robust | 0.034567 / 0.029067 | 0.011224 / 0.007936 | 0.007397 |

  Read-only cross-combinations preserve the pattern: robust with Gaussian noise only has a stage-1-to-later-median scaled ratio of `3.15`, and linear with contaminated noise has ratio `4.17`. The effect is therefore structural, not peculiar to robust optimization or to the selected contamination regime.

- **The sample mean is not the main linear explanation.** The realized Gaussian perturbation has mean `-0.005569`. Stage 1 preserves it and all later linear stages have mean zero to floating-point precision, but the mean accounts for only `5.35%` of stage-1 squared RMSE (`0.023427` centered RMSE versus `0.024080` total RMSE). Low-frequency passband energy, not merely a chance DC offset, causes most of the gap.

- **Exact robust recursion.** Let \(T_k\) be `local_robust_gd_filter` at the stage-\(k\) window, let \(x_{k-1}\) and \(y_{k-1}\) be the clean and observed residuals, and let \(\delta_{k-1}=y_{k-1}-x_{k-1}\). Cell `91eeb55c` implements

  \[
  e_k=T_k(x_{k-1}+\delta_{k-1})-T_k(x_{k-1}),\qquad
  \delta_k=\delta_{k-1}-e_k=\varepsilon-\sum_{j\le k}e_j.
  \]

  It is incorrect to replace this by \(T_k(\varepsilon)\): the robust map is nonlinear. For the intended exact smooth score (ignoring the negligible lookup and stopping errors), implicit differentiation at a local optimum gives adaptive influence weights

  \[
  \frac{\partial T_k(z)_i}{\partial z_{i+q}}
  =\frac{w_{k,q}\,\psi_H'(z_{i+q}-T_k(z)_i)}
  {\sum_\ell w_{k,\ell}\,\psi_H'(z_{i+\ell}-T_k(z)_i)},\qquad
  \psi_H'(r)=\frac{\sqrt{2/\pi}}{H}e^{-r^2/(2H^2)}.
  \]

  Each row sums to one, so the local linearization again passes constants at stage 1 and its residual map removes them. Later Jacobians are adaptive, evaluated on changing residuals, and do not commute. There is no fixed robust analogue of a single \(W_k\) to which the same constant can automatically be assigned.

- **Robust optimization is converged.** Neither robust run approaches `max_iter=60`: clean stages use 5--7 iterations and contaminated stages use 9--11. Reported final maximum updates are about `0.08e-6` to `1.03e-6`, consistent with the scale-adjusted `tol=1e-6` rule. The lookup validation in cell `644a1849` reports maximum score and exponential interpolation errors below `2e-6`. Optimizer or lookup failure cannot explain errors of order `1e-2`.

- **Scaling/indexing is internally consistent but only approximate.** Cell `b66696e1` sets \(a=\sqrt2\), enumerates stages from 1, and divides by \(a^{k/2}\). The schedule in cell `25b95378` is `501, 355, 251, 177, 125, 89, 63, 45, 31`, with radii `250, 177, 125, 88, 62, 44, 31, 22, 15`. A standard-error heuristic based on \(r_k\approx r_1a^{-(k-1)}\) naturally uses \(a^{(k-1)/2}\). The code's \(a^{k/2}\) differs only by the same constant factor \(a^{1/2}\) at every stage, so it cannot create the stage-1 outlier. Rounding/clipping introduces small further deviations; an exact single-pass normalizer would use \(\sqrt{\sum w_{k,q}^2}\), or at least \(\sqrt{r_1/r_k}\).

- **No boundary-indexing error was found.** `np.pad(..., mode="wrap")` followed by `sliding_window_view(..., window_size)` yields exactly `n=1000` centered windows. The first/last offsets have zero kernel weight, and every output index is aligned with the center of its row. Wrap makes the design circular, so apparent beginning/end effects in the linear error are just locations on a circle. For the robust path, the clean signal is not exactly periodic (`x[-1]-x[0] = -0.04954`), so wrap can alter adaptive fits near the seam; it can affect local shapes/suprema but does not explain the generic stage-1 jump.

- **Reference inputs are correctly paired within each case, but the cases are not comparable method controls.** Cell `b66696e1` compares clean versus Gaussian-only data for linear, but clean versus Gaussian-plus-contamination data for robust. `true_components` is never used as the reference: "real IMF" means the same algorithm applied to `x_clean`, not the known generative components. The reported robust error therefore mixes ordinary noise, contamination, and nonlinear signal-dependent effects, and it should not be numerically compared with the linear row as if only the contrast changed.

## Method or Context Details

**Window and boundary path.** `make_window_schedule` (`25b95378`) starts from `odd_ceiling(n / 2)`, treating that number as a *full point count*, then reduces it by approximately \(\sqrt2\). `local_linear_filter` and `local_robust_gd_filter` convert each count to `radius = window_size // 2`. On the normalized grid, the actual kernel half-bandwidths are \(r_k/(n-1)\): `0.25025, 0.17718, 0.12513, 0.08809, 0.06206, 0.04404, 0.03103, 0.02202, 0.01502`. The `relative_width` column shown in cell `25b95378` is instead the full window count divided by `n` (`0.501`, etc.). Confusing these two definitions changes theoretical constants by roughly a factor of two in bandwidth (\(\sqrt2\) in standard error), but not the special role of stage 1.

**Implemented kernel and effective sample size.** `epanechnikov_weights` (`644a1849`) uses \((1-|u|)^2_+\). At the first window, \(\sum w_q^2=0.0036000384\), so the effective sample size is only \(1/\sum w_q^2=277.775\), despite the nominal `501` points. Hence the exact stage-1 single-pass standard deviation is

\[
0.4\sqrt{0.0036000384}=0.0240001,
\]

which matches the observed `0.0240798`. A naive unweighted `0.4 / sqrt(501) = 0.01787` benchmark would wrongly make stage 1 look about 35% too large.

The function name is potentially misleading. The standard Epanechnikov kernel is \(\tfrac34(1-u^2)1_{|u|\le1}\), not \(\tfrac34(1-|u|)^2_+\). At the first window, the standard discrete Epanechnikov weights have \(\sum w_q^2=0.0024000192\); the current kernel's standard deviation is `1.22475` times larger. The repository notes and code use the current formula consistently, so this is a nomenclature/theory-alignment question rather than an internal code inconsistency. If an external theorem's constants assume the standard kernel, the current implementation does not satisfy that assumption.

**Metric path.** Cell `e1ea4c6c` forms `recursive_error = calculated_imf - real_imf`, then computes mean, MAE, RMSE, and sup norm. Sign is irrelevant for RMSE/sup and is consistent throughout. Array indexing is also consistent: human stage `k` indexes `imfs[k - 1]`, while normalization receives `k`. `error_value_frame` calls `scale_sup_error_by_stage` even for pointwise signed values, but that function is currently identical to `scale_error_by_stage`; this is a maintenance smell, not a present numerical bug.

The trend calls use `min_stage=1`, so the structurally different initialization dominates fitted slopes. For recursive scaled RMSE the fitted slope over stages 1--9 is `-8.52e-4` (linear) and `-1.401e-3` (robust); over stages 2--9 it is instead `+9.38e-5` and `+6.63e-5`. The plotting helpers already accept `min_stage`, but the current calls do not use it. This is an interpretation issue: including stage 1 hides the later near-plateau.

The same \(a^{k/2}\) divisor is used for RMSE and sup errors. Pointwise/RMSE variance scaling does not by itself imply constant finite-sample suprema; maxima depend on correlation length/effective multiplicity and generally require a different or logarithmically adjusted statement. The plotted increase of linear single-pass scaled sup error (`0.0382` at stage 1 to `0.0676` at stage 9) is therefore not evidence that the RMSE calculation is wrong.

**Robust numerical path.** `robust_gd_fit_windows` (`91eeb55c`) starts each row at its unweighted median, uses normalized kernel weights, updates by

\[
x^{(m+1)}=\operatorname{clip}\!\left[x^{(m)}+\frac{0.95H}{\sqrt{2/\pi}}
\sum_qw_q\psi_H(z_q-x^{(m)})\right],
\]

and stops on a chunk-wide maximum update. Chunking can change the final answer at roughly tolerance scale because chunks stop independently, and the tolerance threshold includes `max(abs(x))`; neither effect is remotely large enough to explain the reported component errors. Translation equivariance holds for the mathematical objective and for the iteration apart from this stopping-tolerance detail.

**Observation generation.** `generate_observation` (`25b95378`) adds Gaussian noise and then `mask * exponential * random_sign`. Thus the contamination used here is a *signed, symmetric exponential-magnitude* additive contamination, not one-sided exponential contamination, even though some project prose calls it one-sided. `centered_contamination=False` does not make the signed contamination have a nonzero population mean. This model mismatch may matter when matching a theorem's assumptions, but it does not produce the stage-1/later operator distinction.

**Checks that do not establish component validity.** Reconstruction errors below `1e-15` follow algebraically from repeated `residual = residual - imf`; they detect bookkeeping failures but cannot validate that the extracted components or their scaling match a theorem. The useful validation for the present question is the operator/impulse-response calculation above.

## Sources

- Current working-tree [`gd_imf_real_vs_calculated.ipynb`](../../../experiments/real-vs-calculated/gd_imf_real_vs_calculated.ipynb):
  - cell `25b95378`: `make_window_schedule`, `generate_observation`, parameters, and realized schedule;
  - cell `644a1849`: `epanechnikov_weights`, score/loss lookup, and interpolation validation;
  - cell `91eeb55c`: `robust_gd_fit_windows`, `local_robust_gd_filter`, and `robust_gd_imf_with_history`;
  - cell `267ea198`: `local_linear_filter` and `linear_imf_with_history`;
  - cell `b66696e1`: stage normalization, clean/observed pairings, and decomposition runs;
  - cell `e1ea4c6c`: recursive/single-pass error construction and printed summary;
  - cell `0503c494`: grid scaling and trend plotting helpers.
- `README.md`, especially "Robust Gradient-Descent IMF" and "Linear Operator Note", for the repository's stated recursion and wrap-commutativity interpretation.
- `ROBUST_GRADIENT_DESCENT.md`, sections 1, 3, 13, and 14, for the intended recursion, kernel formula, sequential robust stages, and the stated limit of linear commutativity.
- [`gd_imf_real_vs_calculated_review.tex`](../../../overleaf/gd_imf_real_vs_calculated_review.tex), "Error Definitions" and "Setup and Metrics", for the documented distinction between recursive and single-pass errors.
- Read-only numerical audit executed with the repository `.venv` by loading the named notebook cells directly. It checked exact component identities, impulse responses, DFT gains, theoretical iid Gaussian variances, convergence traces, boundary subsets, and cross-combinations of method/noise regime.

## Contradictions or Uncertainty

- The code path establishes that \(a^{k/2}\) approximately stabilizes *single-pass pointwise/RMSE noise*, and that it does not put the initialization component and recursive detail components on one common constant. Whether `IMF.pdf` proves a statement about single-pass estimators, recursive components only after initialization, a different norm, or an asymptotic bound must be settled from the exact theorem; the notebook itself does not cite a theorem number or reproduce its assumptions.
- The stage exponent may be indexed from zero or one in the theory. The code begins at `k=1`, although the schedule is naturally \(r_k\approx r_1a^{-(k-1)}\). This only changes all scaled values by one common factor and cannot explain the first-stage gap.
- The repository consistently calls \((1-|u|)^2_+\) "Epanechnikov", but standard references use \(1-u^2\). It is unclear whether the reference PDF intentionally defines the former kernel or whether the implementation was transcribed incorrectly.
- Robust finite-noise errors have no exact fixed linear operator. The Jacobian argument explains the same mechanism locally, while the observed cross-combination results show it empirically; it is not an exact distributional formula comparable to the linear white-noise variance calculation.
- All displayed empirical values come from one fixed seed (`777`) and `n=1000`. The exact operator norms are seed-independent, but realized RMSE/suprema and fitted slopes are not.
- Wrap is mathematically clean for the linear circulant analysis. For robust fits on a nonperiodic signal it makes the local Jacobian signal-dependent near the seam, so a theorem assuming an interior point, independent boundary correction, or a truly periodic signal may not apply there.

## Open Questions

- What exact theorem, proposition, or heuristic is the source of \(a^{k/2}\)? Its estimator, bandwidth definition, stage origin, norm (pointwise, \(L_2\), RMSE, or sup), and probability mode are needed before claiming constancy.
- Does that result treat the first/coarsest low-pass term separately from the later band-pass/detail terms? The implemented operators require such a distinction.
- Is the intended kernel really \((1-|u|)^2_+\), or should `epanechnikov_weights` use the standard \((1-u^2)_+\)?
- Is `window_size` intended to represent full support width or theoretical half-bandwidth \(h_k\)? The code uses the former while the asymptotic formula normally uses the latter.
- Should the empirical comparison center each realized noise vector or otherwise remove the coarse/DC component before comparing recursive detail errors? That would answer a different, potentially useful question, but should be explicit.
- Should linear and robust methods be rerun on the same observation regimes when comparing their error levels? The current notebook isolates neither contrast nor contamination.
- If uniform/sup error constancy is a goal, what logarithmic/effective-multiplicity normalization does the source result prescribe? Reusing the RMSE divisor is not justified by the code alone.
