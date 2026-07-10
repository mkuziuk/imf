## Answer

The large first recursive error is expected from the implemented decomposition; it is not evidence of a bug in indexing, wrap padding, lookup interpolation, or robust optimization.

For the linear method,

\[
e_1=W_1\varepsilon,\qquad
e_k=W_k\prod_{j<k}(I-W_j)\varepsilon\quad(k>1).
\]

Stage 1 is a low-pass estimate of the original noise. Every later stage contains residual/high-pass factors and is a band-pass detail with zero DC gain and much lower operator energy. Bandwidth scaling removes the main change in scale with \(h_k\), but it cannot remove the different filter-energy constant between the initial low-pass term and later detail terms.

The theory in `IMF.pdf` supports only an order statement. Lemma 2.5 asserts

\[
(\mathcal W_k\mathcal W_k^\top)_{tt}\asymp (nh_k)^{-1},
\qquad \mathcal W_k=W_k\prod_{j<k}(I-W_j),
\]

so scaled recursive pointwise SD/expected RMSE has the same order across stages. The symbol \(\asymp\) does not imply one equal constant. The exact stage-dependent constant is about ten times smaller in variance after the first-stage transient. The PDF does not prove constant robust recursive error or constant sup error.

## Key Evidence

- The exact finite-\(n\) linear operator calculation predicts scaled recursive RMSE `0.020182` at stage 1, `0.008278` at stage 2, and about `0.0063--0.0067` later. The notebook reports `0.020249`, `0.007485`, and the same later plateau.
- Exact scaled **single-pass** RMSE is nearly flat at about `0.02027` across all stages (coefficient of variation `0.65%`). This is the cleanest demonstration of the bandwidth rate.
- The seed-777 linear stage-1 RMSE is at the `58.9th` percentile of 5,000 Gaussian replications; all stage values lie within their 5--95% Monte Carlo intervals.
- Reflect rather than wrap boundaries changes stage-1 domain RMS by only `6.7%` and preserves the gap.
- Robust Gaussian-only control: scaled stage 1 is `0.021263`, versus mean `0.006511` over stages 2--9. With contamination these are `0.029067` and `0.007357`. Across 120 robust trials, the mean stage-1/tail ratio is about `2.84` without contamination and `2.82` with contamination.
- Robust fits converge in 5--11 iterations, far below the 60-iteration limit. Optimization failure is not a plausible explanation.
- The notebook's \(a^{k/2}\) divisor versus the natural \(a^{(k-1)/2}\) stage origin differs by one common factor only, so indexing cannot create the first-stage jump.
- `IMF.pdf` contains an incorrect intermediate equality between component error \(W_kA_{k-1}\varepsilon\) and residual noise \(A_k\varepsilon\). Lemma 2.5 returns to the correct composite component operator but is unproved and order-level.

The executed companion experiment is [`first-imf-recursive-error-diagnostics.ipynb`](../../output/jupyter-notebook/first-imf-recursive-error-diagnostics.ipynb). Full numerical outputs are in [`diagnostics/`](diagnostics/), with separate [`code-path`](agents/code-path.md), [`numerical`](agents/numerical-diagnosis.md), and [`theory`](agents/theory.md) memos.

## Confidence

**High** for the linear diagnosis. It follows from exact discrete operators, reproduces the notebook, and is supported by 5,000 Monte Carlo replications and a boundary ablation.

**High** that the robust stage-1 pattern is structural in this experiment, because it persists without contamination and across 120 trials, and the optimizer is converged.

**Moderate** for any formal robust-theory claim. The local PDF's robust section gives a one-step expansion, not a completed theorem for the recursive nonlinear composition.

## Risks and Gaps

- The local PDF is an unfinished July 3, 2026 draft: Lemma 2.5 has no proof or stated uniform constants, Proposition 4.1 is incomplete, and several formulas/references contain errors.
- The notebook's robust “real IMF” is the robust fit applied directly to clean data. The PDF's population target minimizes expected noisy loss. These targets generally differ; in the first-stage Gaussian check the mismatch RMSE is small (`0.000603`), but later contaminated stages are not covered by that calculation.
- The exact linear calculation assumes iid homoscedastic noise and the implemented periodic design. Correlated or heteroscedastic noise needs its full covariance.
- Pointwise/RMSE scaling does not normalize sup errors. Shorter correlation lengths create more effective opportunities for a maximum.
- The linear and robust headline cases use different observation regimes, so their absolute error levels are not a controlled method comparison.

## Decision Implications

1. Treat stage 1 as a separate coarse/scaling component. Assess recursive scaled-error stability over stages 2+ or, more cleanly here, stages 3--9; the exact linear scaled values over stages 3--9 have only `3.63%` coefficient of variation.
2. If the goal is a flat linear diagnostic, normalize recursive components by their exact operator SD, \(\sigma\sqrt{(\mathcal W_k\mathcal W_k^\top)_{tt}}\), rather than by bandwidth alone.
3. For robust recursive components, estimate a stage-specific scale with a parametric bootstrap or a numerical Jacobian/sandwich calculation.
4. Keep single-pass and recursive errors labeled separately. The former cleanly checks the bandwidth heuristic; the latter checks the complete filter bank.
5. Analyze sup error with simulated simultaneous quantiles or an explicit correlation/multiplicity correction, not the RMSE divisor alone.
6. If testing the PDF's robust theorem, construct its expected-noisy-loss population target rather than using only the clean same-contrast fit.

## Recommended Next Questions

- Is “constant scaled error” intended to mean pointwise SD, expected RMSE, one realized RMSE, or sup error?
- Should the intended theory explicitly exclude the initial coarse component or allow a stage-dependent equivalent-kernel constant?
- Is the robust validation target the PDF population IMF or the same-contrast clean-data decomposition used by the notebook?
- Can the author provide a proof and uniformity conditions for Lemma 2.5 and complete Proposition 4.1?
