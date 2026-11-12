# All-components theorem under strengthened assumptions

The follow-up [TeX proof](../../overleaf/irmf_all_components_bound.tex)
closes the propagation, induction, and population-target steps under explicit
additional conditions. It does not prove the original draft at its full
generality.

On the finite evaluation grid, the final result is

\[
 \max_t|\widetilde S_t^{(k)}-S_{{\rm pdf},t}^{(k)}|
 \le A_1\sqrt{z\alpha_k}+A_2z\alpha_k,\qquad k=1,\ldots,K,
 \quad z=x+\log(10nK),
\]

simultaneously with probability at least \(1-e^{-x}\).
Here \(\alpha_k\asymp1/(nh_k)\) gives the requested bandwidth rate.
The constants are independent of the stage when the deterministic curvature,
operator, schedule, and stability parameters remain bounded.

## Assumptions added or made quantitative

The earlier assumptions of positive population curvature and bounded
deterministic residual products are retained. The new stability condition is

\[
 C_BC_\alpha d<a-1,\qquad
 d=\max_{k,t}
 \frac{\sum_u a_{tu}^{(k)}
   \mathbb E|\rho''(c_{tu}^{(k)}+\varepsilon_u)
      -\mathbb E\rho''(c_{tu}^{(k)}+\varepsilon_u)|}{f_{kt}}.
\]

The quantity \(d\) is determined by the noise law, reference signal, and
weights. It does not assume anything about the unknown recursive remainder.
It measures relative variation of the local curvature.

The proof specifies every constant and an explicit positive threshold
\(\eta\), and requires \(z\max_k\alpha_k\le\eta\).
The theorem for the fixed-noise reference needs no noise moments if the
expected contrast is anchored. The paper-target comparison additionally
requires a finite fourth moment, a positive tail-integration cutoff, and
the stated target-localization inequalities. A sharper version retains
the explicit tail correction instead of absorbing it into the second-order term.

## What was proved

1. A complete inequality for the centered quadratic statistic. Random
   partitioning separates its two index sets, two symmetrizations introduce
   independent signs, and two elementary Rademacher moment bounds give
   \(16p\) times the Frobenius norm of a deterministic envelope.
   The argument covers different kernels at different index pairs.
2. A simultaneous bound on \(Z_k\Lambda_k\), including its diagonal.
   Its two terms have orders \(\alpha_{k-1}\) and
   \(z\sqrt{\alpha_k\alpha_{k-1}}\). Independence is across observations;
   the noise is reused across stages.
3. The propagation estimate

   \[
   \|\Phi_k\|_\infty\le
   \frac{4L_3}{\lambda}\|E_k\|_\infty^2+
   C_{\rm loc}\sqrt{z\alpha_k}\|E_k\|_\infty+
   \|Z_k\Lambda_k\|_\infty+
   \|Z_k\|_\infty\|\Theta_k\|_\infty.
   \]

   A normalization inequality for positive vectors controls Jacobian
   changes using curvature at the single-pass fit. Uniform positive
   curvature along the entire interpolation segment is unnecessary.
4. An induction whose constants close. Choose
   \(d<\delta<(a-1)/(C_BC_\alpha)\), so
   \(q=C_BC_\alpha\delta/(a-1)<1\).
   Set \(D=C_Bs_1(A_R+P(A))/(1-q)\) before starting the induction.
   The finest-scale restriction keeps the accumulated error in its
   prescribed neighborhood. This proves
   \(\|\Theta_{k+1}\|_\infty\le Dz\alpha_k\), rather than assuming it.
5. The expansion
   \(\|\widetilde S^{(k)}-S_\phi^{(k)}-L_k\|_\infty
   \le D(1+C_\alpha/a)z\alpha_k\) at all stages.
6. The target comparison. In the expected score,
   \(\mathbb E[g_{ktu}\Lambda_k(u)]\) contains only the diagonal
   observation term; all off-diagonal terms vanish by independence and
   centering. Truncated tail integration controls the other terms.
   The result bounds the deterministic target shift and yields the final
   all-components inequality above.
7. A nonempty regime with a growing number of stages. For a constant
   signal, Gaussian noise with \(H=2\sigma\), and successively refined
   block-average weights, \(C_B=2\), \(a=2\), and \(C_\alpha=1\).
   A direct second-moment calculation verifies the dispersion condition.
   For \(K\le c\log_2 n\), \(0<c<1\), every remaining sufficient condition
   holds for large enough \(n\). This example uses block weights; it is
   not a claim about all convolution or boundary designs.

## Check against the submitted Gaussian experiment

The [condition report](diagnostics/all_components_conditions.json) gives:

| Quantity | Value |
|---|---:|
| Curvature dispersion \(d\) | 0.1711180942 |
| Allowed dispersion threshold | 0.1951707444 |
| Chosen \(\delta\) | 0.1831444193 |
| Induction coefficient \(q\) | 0.9383804928 |
| Finest-scale threshold \(\eta\) | \(3.9993\times10^{-10}\) |
| Actual \(z\alpha_{\max}\), at 95% confidence | 1.4371360865 |

The dispersion condition passes. The sample-size condition fails by many
orders of magnitude, so this proof does **not** certify the \(n=1000\),
nine-stage experiment. The constants are conservative, especially the
quadratic-statistic and induction constants. Passing a simulation is not
a substitute for satisfying the theorem's assumptions.

The Gaussian dispersion formula uses normal CDFs after completing the square,
not numerical quadrature. Independent adaptive quadrature agrees within
\(5.5\times10^{-16}\).
The exact propagation split is also checked on the three Gaussian samples,
with maximum discrepancy below \(7.6\times10^{-17}\).
The finite-design operator calculations are floating-point checks, not
interval certificates.

The existing numerical audit was rerun in a temporary output directory.
All six CSVs match exactly, and its summary JSON differs only in elapsed
runtime. The new eight-page proof compiles without layout or reference
warnings; every page was visually inspected.

## Reproduction

    .venv/bin/python research/recursive-component-bounds/diagnostics/check_all_components_conditions.py
    .venv/bin/python research/recursive-component-bounds/diagnostics/check_recursive_bounds.py --archive ~/Downloads/artifacts.zip
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/imf-all-components-build overleaf/irmf_all_components_bound.tex

The first command reports failed applicability conditions as data. A failure
of a sufficient condition is not an assertion that the method or theorem
is false. Formula and algebra consistency checks still raise errors if they
fail.

## Remaining limits

Removing the curvature-dispersion condition, obtaining useful finite-sample
constants for the submitted experiment, and treating all boundary designs
remain open. The earlier Fourier hypotheses are sufficient conditions;
this proof does not establish them for every discrete kernel.
The theorem concerns the paper's population targets and the fixed-noise
reference, not the same-loss clean-data target in some notebooks.
