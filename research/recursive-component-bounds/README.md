# Review of the recursive component bounds

Start with the [five-page visual guide](../../overleaf/irmf_intuitive_guide/irmf_intuitive_guide.pdf).
It connects Proposition 4.1's notation to the propagation remainder,
quadratic cancellation, and induction closure, without worked arithmetic.
The earlier numerical example remains available through its
[reproduction script](build_concrete_guide.py).

Continue with the [12-page visual guide](../../overleaf/irmf_all_components_guide/irmf_all_components_guide.pdf)
for a worked explanation of the propagation remainder, quadratic cancellation,
induction, and population-target correction. Its
[editable LaTeX project](../../overleaf/irmf_all_components_guide/)
includes diagrams, plots from the audited results, and the theorem's applicability limits.

The follow-up [proof completion](PROOF_COMPLETION.md) proves the propagation
remainder, closes the induction, and compares the population targets under
an additional curvature-dispersion condition and explicit sample-size and
moment conditions. See [the new theorem](../../overleaf/irmf_all_components_bound.tex).
Its conservative finite-sample conditions do not certify the submitted
n=1000 experiment. The archive audit below remains valid for the original,
less restrictive statement.

Reviewed on 2026-10-31 against Downloads/artifacts.zip and the current repository.

**Original archive verdict: partial pass. The archive does not prove a bound for all
robust components relative to the paper's population targets.**
The exact decomposition and several supporting results are correct.
The propagation remainder, closure of the induction, and comparison of
targets still need proofs. The companion guide sometimes presents these
open steps as established conclusions.

The reviewed mathematical statement is
[the TeX note](../../overleaf/irmf_recursive_component_bounds.tex).
Its final corollary explicitly assumes the missing simultaneous propagation
bound. It is not an unconditional all-components theorem.

## Results accepted into the project

1. The exact decomposition
   \(e_k=L_k+\sum_{j\le k}T_{kj}(R_j-\Phi_j)\), with the signs and
   order of the noncommuting products checked.
2. Uniform linear residual and component operator bounds under the stated
   discrete Fourier assumptions K1--K4 and geometric scales. The new note
   supplies details of the derivative estimates and includes empty products.
   Verification of K1--K4 for every claimed sampled-kernel family is separate.
3. The perturbation lemma for population equivalent kernels when
   \(C_P\sum_k\tau_k\le1/2\), including the oscillation bound on each \(\tau_k\).
4. Simultaneous concentration of the leading terms by grouping contributions
   by independent observation. Stages reuse the same noise vector.
5. A self-contained single-pass Fisher remainder estimate with explicit
   local curvature and sample-size conditions. This replaces the abbreviated
   appeal to Proposition 4.1 in the new note; it does not change the existing
   Proposition 4.1 file.
6. A conditional component bound with explicit constants and failure
   probability, assuming a simultaneous bound on the actual propagation
   remainders. No bootstrap argument is claimed.

The corrected concentration result permits
\(\alpha_j\le C_\alpha\alpha_k a^{-(k-j)}\), rather than requiring
\(C_\alpha=1\). The rounded nine-stage schedule needs
\(C_\alpha=1.0045415812465368\); a conservative value is 1.00455.

## Findings in the submitted proof

Line numbers refer to the unmodified
[source note](source/irmf_contraction_note.tex) and
[source guide](source/irmf_contraction_guide.tex).

### 1. The propagation lemma is unfinished

Note lines 360--391 are a sketch. The quadratic Jacobian estimate needs a
positive lower bound on the empirical curvature along
\(X_\phi^{(k)}+\varepsilon-sE_k\), including the fitted location at each
\(s\in[0,1]\). A population curvature lower bound alone is insufficient.
Localization may supply this, but its event and smallness conditions
are not specified.

The off-diagonal quadratic statistic needs explicit coefficient and norm
bounds, centering of each observation's contribution, a diagonal-term
estimate, and a tail inequality with constants. Saying that two conditional
Bernstein inequalities give the desired order does not supply those steps.
The probability must cover all time points and stages on a single event.
The source acknowledges an omitted step, but more than one calculation
needs to be supplied.

### 2. The induction does not close as written

Note lines 410--418 assert that the propagation lemma's hypotheses follow
by induction. Its output constant depends on the input constants.
The bound for the \(\Theta\)-part includes \(2L_2 A_\Theta/\lambda\),
and the quadratic term depends on \(A_E^2\).
Substitution into the next bound for \(\Theta\) produces an inequality
involving \(A_\Theta\) on both sides. There is no coefficient estimate or
smallness condition that solves this inequality uniformly in the number
of stages.

For example, an estimate \(P_k\le Bz\alpha_{k-1}+D\|\Theta_k\|_\infty\),
followed by summation of previous \(P_j\), does not itself imply
\(P_k=O(z\alpha_{k-1})\) uniformly in \(k\). Geometric summation controls
known coefficients; it does not bound coefficients that depend recursively
on the same proposed constant. This is a proof gap, not a counterexample
to the intended theorem.

The proof also invokes a uniform bound on all bare products
\(A_\phi^{(m:l)}\), although the theorem explicitly assumes only a component
composite bound. One must assume/prove the additional bound or replace this
step. The propagation lemma gives a per-stage exceptional event; the theorem
needs a joint event.

The new corollary assumes the actual simultaneous propagation bound directly.

### 3. Transfer to the paper's targets remains unproved

Note lines 125--137 assert that the paper's targets and the fixed-\(\phi\)
targets differ at second order, then extend the theorem to those targets.
Gap 4, lines 498--503, correctly lists the same assertion as unproved.

Small \(E_k\) alone gives a first-order perturbation estimate. A
second-order conclusion requires cancellation of the linear expectation,
including dependence between \(E_k\) and the original noise, plus
curvature and integrability bounds. Monte Carlo differences do not prove
this cancellation or a uniform asymptotic rate. The guide's phrase
"equal numerically" at line 361 overstates the evidence.

The new note keeps \(\|S_\phi^{(k)}-S_{\rm pdf}^{(k)}\|_\infty\) explicit.
There is also no theorem here for the clean-data, same-\(\rho\) reference
used by some existing experiments.

### 4. The single-pass remainder needs quantitative hypotheses

Note lines 345--357 identify the correct deterministic-signal single pass.
Obtaining one constant \(A_R\) across stages still requires uniform
curvature, separation of \(\delta\) from one, and local sample-size
conditions. These should not be hidden in the word "hence".

The source's first coefficient is \(9/8\); the repository's Proposition 4.1
uses \(3/4\). The larger value is conservative under the same hypotheses,
so that difference is not itself an invalid bound.
The repository proposition also has an empirical strong-convexity
assumption. Strict convexity of the smoothed absolute loss does not imply
uniform positive curvature on the whole real line. The new proof establishes
the needed curvature in a small neighborhood instead.

### 5. Kernel and schedule applications need qualification

The linear Fourier argument is valid under K1--K4. The asserted applications
to sampled kernels lack a uniform discrete verification.
The continuous Epanechnikov zero near 4.49 alone does not prove the
corresponding statement for every sampled and normalized kernel.
The extra cross-scale decay remark at note line 274 is not proved there
and is not imported as a result.

The numerical schedule is rounded, so its radii and largest weights do not
form the exact geometric sequence assumed by the draft.
The \(C_\alpha\) correction fixes the concentration application for this
finite schedule. The Fourier lemma still assumes exact geometric scales;
the finite schedule's operator bounds are checked directly.

The archived diagnostics save only prefix residual norms, not the maximum
over every partial product needed for \(C_P\tau_\Sigma\le1/2\).
Some reported prefix norms also describe robust operators rather than the
linear operators defining \(C_P\).
The new diagnostic checks all required linear products and obtains
\(C_P=2.108655379927799\), giving
\(C_P\tau_\Sigma=0.49330952146146984\) for the Gaussian reference.
These are floating-point checks for one finite design, not certified
interval bounds or a uniform theorem over designs.

### 6. Other claims need qualification

- The expected-Jacobian telescoping at note lines 198--201 is correct.
  It does not identify the random leading term with one composite matrix
  applied to a common score vector. The sharp variance approximation remains open.
- "Fresh noise" in the guide is misleading. The exact decomposition uses
  the same original noise vector in every single pass.
- "Arbitrary law" needs a finite expected loss or an anchored expected
  contrast. For Cauchy noise, the unmodified expected smoothed absolute
  loss is infinite. The new note includes the anchored formulation.
- Dropping the linear Bernstein term requires a smallness condition on
  \(z\alpha_k\).
- A larger remainder at the finest contaminated window does not by itself
  prove failure of a sufficient condition of Proposition 4.1.
  The source does not evaluate that condition.

## Numerical and packaging review

1. The archived linear CSV is reproduced to every reported digit.
   The new check includes all partial products, including the longest one.
2. Rerunning the original Gaussian script for the first three seed-777
   samples reproduces the archived sup and RMS values within
   \(3.0\times10^{-14}\).
3. An independent implementation with a score-residual stopping rule
   reproduces the Gaussian sup values within \(2.1\times10^{-14}\).
   It checks the recursive identity to \(6.2\times10^{-17}\) and signal
   reconstruction to \(6.7\times10^{-16}\).
4. For the Gaussian reference, all checked population component composites
   have sup norm at most \(1+10^{-12}\); all partial population residual
   products have sup norm at most 2.112719. The tilt sum is 0.233945066.
5. Medians were recalculated from all four submitted NPZ files.
   At stage nine, relative sup remainders are 10.37%, 10.63%, 26.14%,
   and 41.57% for Gaussian, triple-amplitude Gaussian, and the two
   contamination scenarios, matching the reported rounded values.
   The full 300/200-repeat simulations and 400-repeat target calculation
   were not rerun. Their stored summaries remain empirical evidence.
   The 21-stage claims and the guide's figures were not independently regenerated.
6. The original operator loop evaluates the norm before updating the
   product and omits the longest product. Correcting this does not change
   the Gaussian maximum above.
7. The simulated \(\eta\) uses a sample curvature average because its
   population-denominator variable remains unset. The text defines a
   population denominator, so the implemented surrogate is different.
8. The contamination expectation uses 40-node Gauss--Laguerre quadrature,
   with no supplied quadrature error estimate. It is approximate.
   The target-check script always generates Gaussian observations even
   if its copied contamination arguments are set.
9. The source TeX contains unresolved artifact placeholders. The guide
   requests figure PDFs while the archive contains PNGs for those figures.
   The cited first- and second-component drafts are absent from both
   the archive and the current repository.
10. The existing first-component diagnostic was rerun with outputs
    redirected to a temporary directory. All six CSVs match the repository
    exactly. Its summary JSON matches except for elapsed runtime.

For the Bernstein convention in the new proof, see the primary
[Boucheron lecture slides](https://dornsife.usc.edu/steincolloquium/wp-content/uploads/sites/371/2023/09/boucheron.pdf)
on sub-gamma tails. The proof checks otherwise use the submitted definitions
and the repository's local sources.

## Reproduction and provenance

SciPy was already listed in requirements.txt but missing from the local
virtual environment; it was installed there for this review.
From the repository root:

    .venv/bin/python research/recursive-component-bounds/diagnostics/check_recursive_bounds.py

To compare with the original archived arrays:

    .venv/bin/python research/recursive-component-bounds/diagnostics/check_recursive_bounds.py --archive ~/Downloads/artifacts.zip

The default runs three Gaussian samples and writes
[verification.json](diagnostics/verification.json) and
[linear_partial_products.csv](diagnostics/linear_partial_products.csv).
Use --output-dir to keep reruns separate. The checked-in JSON includes
the optional archive comparison.

To compile without adding TeX build files to the repository:

    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/imf-recursive-bounds overleaf/irmf_recursive_component_bounds.tex

[source/](source/) contains unchanged submitted TeX, Python, and linear CSV
files for provenance. They are preliminary inputs, not approved project
proofs or maintained diagnostics. Their issues are intentionally preserved.
[manifest.json](source/manifest.json) records the archive hash and each
member's hash, size, and import status.
Submitted PDFs, figures, and binary Monte Carlo arrays were not imported.
No existing notebook or earlier proof was changed. The previously untracked
prop41-lemma42 and quantlet-comparison directories were left untouched.

## Gaps in the original statement

Items 1--3 below are resolved under the strengthened assumptions in
[the follow-up proof](PROOF_COMPLETION.md). They remain open at the
original statement's full generality.

1. Prove a simultaneous propagation-remainder estimate with explicit
   empirical-curvature localization and quadratic-statistic constants.
2. Close the induction with quantitative conditions ensuring constants
   stay bounded as the number of stages grows.
3. Prove the comparison with the paper's recursive population targets.
4. Verify discrete kernel hypotheses uniformly, or state the theorem for
   designs with separately established composite bounds.
5. Treat sharp variance constants, nonperiodic boundaries, and other
   clean-reference targets as separate extensions.
