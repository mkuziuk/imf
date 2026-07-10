## Verdict

The investigation meets the brief and supports a clear two-part verdict.

For the **linear decomposition**, the large first recursive error is expected from the implemented and theoretically defined composite operator. It is not caused by an indexing error, boundary misalignment, an unusual seed, or a failed calculation. The component-error operator is

\[
\mathcal W_k=W_kA_{k-1},\qquad
A_{k-1}=(I-W_{k-1})\cdots(I-W_1).
\]

At stage 1, \(\mathcal W_1=W_1\) is a low-pass smoother with DC gain one. At every later stage, the factors in \(A_{k-1}\) remove the already-extracted low-frequency content, so \(\mathcal W_k\) is a lower-energy band-pass/detail operator with DC gain zero. The exact finite-\(n\) calculation predicts scaled RMS 0.020182 at stage 1 and a stages-2--9 mean of 0.006776, a ratio of 2.978. The notebook realizes 0.020249 at stage 1 and a stages-2--9 mean of 0.006488, a ratio of 3.121. This is quantitative agreement, not a theory violation.

The relevant theoretical statement, Lemma 2.5 of IMF.pdf, **does apply to the calligraphic composite operator \(\mathcal W_k\)**. It is therefore too narrow to say that the theory's bandwidth rate concerns only the bare single-pass smoother \(W_k\). However, the lemma states only

\[
(\mathcal W_k\mathcal W_k^\top)_{tt}\asymp (nh_k)^{-1}.
\]

This is an asymptotic order statement. It does not assert equality, a common variance constant for all stages, convergence of one realized scaled RMSE curve, or a sup-norm result. The exact stage-dependent operator-energy constants differ by roughly a factor of ten in variance between stage 1 and the later plateau while remaining compatible with the same \((nh_k)^{-1}\) order. Thus “scaled error should be constant” is stronger than the cited theory.

For the **robust decomposition**, the evidence strongly supports the same initialization-versus-residual mechanism empirically, but there is no valid theorem in the supplied note for the notebook's recursive robust error. The robust section analyzes a one-step fit against an expected-noisy-loss population target, not the notebook's clean-fit reference, and it does not control the nonlinear residual composition. Proposition 4.1 is incomplete. The robust result should therefore be presented as a Monte Carlo and local-linearization finding, not as a consequence of Lemma 2.5.

No change to the decomposition code is warranted merely to remove the first-stage spike. The needed correction is interpretive: treat stage 1 as the coarse/scaling component, distinguish exact operator standardization from bandwidth-only scaling, and state precisely which error functional is being tested.

## Evidence Quality

| Evidence | Grade | Review |
|---|---|---|
| Algebraic derivation from linear_imf_with_history | **A / decisive** | The derivation is exact, directly follows the update order, and was numerically checked to 1.41e-15. It identifies the relevant object as \(\mathcal W_k=W_kA_{k-1}\). |
| Exact finite-\(n\) FFT/impulse-response calculation | **A / decisive** | It uses the actual weights, rounded windows, \(n=1000\), \(\sigma=0.4\), and periodic design. The predicted stage pattern closely matches the notebook. This is stronger for the present finite experiment than the unproved asymptotic lemma. |
| Reproduction of stored notebook metrics | **A** | All linear and robust RMSEs reproduce to the displayed precision. The companion notebook has execution counts 1, 2, 3 and no error outputs. |
| Linear Monte Carlo, 5,000 repetitions | **A-** | It separates \(\sqrt{E[\mathrm{RMSE}^2]}\) from \(E[\mathrm{RMSE}]\), places the seed-777 stage-1 value at percentile 0.589, and shows all realized stages inside their 5--95% intervals. It decisively rules out an exceptional seed. |
| Boundary diagnostics | **A- for linear; C+ for robust** | Periodic pointwise variance is exactly constant over position. Reflect ablations and 400 simulations preserve the gap. Robust seam effects are discussed but were not systematically simulated under alternative boundaries. |
| Robust controls and 120-repetition Monte Carlo | **B+** | Gaussian-only and contaminated trials both give a stage-1/tail ratio near 2.8, and convergence is well below the iteration cap. This is strong qualitative evidence, but 120 trials do not characterize tails and there is no exact robust operator. |
| Direct reading of IMF.pdf | **A for what the document says; C for theorem reliability** | Pages 5--6 visibly define \(\mathcal W_k\) and state Lemma 2.5 for that composite. But the lemma has no proof or uniformity statement, the surrounding page contains algebraic errors, and the robust proposition is unfinished. |
| Robust population-target check in the companion notebook | **B-** | The Gaussian first-stage \(H_{\rm eff}\) calculation finds a small but nonzero target discrepancy (RMSE 0.000603). It establishes that the targets are not identical in that case, but does not quantify later recursive, contaminated, or general-signal discrepancies. |
| Diagnostic implementation independence | **B+** | The exact linear calculation is independently formulated in Fourier form. The large Monte Carlo script reproduces notebook definitions manually, while the companion notebook also executes selected original cells read-only. Agreement reduces transcription risk, but the two numerical artifacts still share formulas and are not an external replication. |

The strongest chain of evidence is: exact code algebra \(\rightarrow\) exact operator energy \(\rightarrow\) stored-output reproduction \(\rightarrow\) Monte Carlo placement. The robust and theory claims should not be given the same evidentiary weight as that linear chain.

## Agreements and Disagreements

All three memos and the companion notebook agree on the central facts:

- stage 1 is \(W_1\varepsilon\), while later linear errors are \(W_kA_{k-1}\varepsilon\);
- the residual factors remove DC and sharply reduce later component energy;
- the first-stage spike is present in the exact finite-\(n\) operator, not generated by random variation;
- \(a^{k/2}\) and \(a^{(k-1)/2}\) differ only by one common vertical factor, so the stage-origin convention cannot create the spike;
- the actual theoretical half-bandwidth is \(R_k/(n-1)\), not window_size divided by \(n\);
- periodic boundary indexing is correct and does not create a privileged edge in the linear run;
- robust optimization converges, and contamination is not necessary for the qualitative pattern;
- a bandwidth-only normalization does not flatten sup errors;
- the implemented kernel matches the PDF's displayed \((1-|u|)^2_+\) formula but is not the conventionally named Epanechnikov kernel.

The main reconciliation concerns the scope of Lemma 2.5:

- agents/theory.md correctly establishes that Lemma 2.5 is written for the **composite** \(\mathcal W_k=W_kA_{k-1}\).
- Parts of agents/code-path.md emphasize that the familiar flat constant is naturally obtained for the **single-pass** \(W_k\varepsilon\). That numerical statement is correct, but any wording that suggests Lemma 2.5 itself applies only to \(W_k\) should be superseded by the theory memo.
- There is no substantive contradiction after making the distinction between **rate/order** and **equal constant**. The lemma assigns the composite operator the order \((nh_k)^{-1}\); the exact calculation shows that its multiplier \(C_k\) changes sharply at initialization. Both statements can be true.

Several numerical differences are changes of estimand rather than disagreements:

- 0.006776 is the **exact expected-RMS** stages-2--9 mean; 0.006488 is the **seed-777 realized** mean; 0.006420 is the seed-777 stages-4--9 mean. These should not be interchanged.
- The code-path memo's 4.68% single-pass scaled coefficient of variation is for one realized seed. The diagnostic summary's 0.65% is for the exact discrete standard-deviation curve. Both are correct.
- A stage-1/tail ratio of about 3.1--4.2 in selected seed/control combinations and about 2.82--2.84 across robust Monte Carlo means is expected because ratios of a single realization and ratios of across-trial means are different statistics.
- The numerical memo's summary phrase that the stored later values are at the “0.006776-scale” is imprecise: 0.006776 is the exact tail mean, whereas the stored tail mean is 0.006488. Its tables and later calculations make the distinction correctly.

The theory memo identifies genuine defects in the source note that the other memos do not contradict:

- page 6 equates component error \(W_kA_{k-1}\varepsilon\) with residual noise \(A_k\varepsilon\), which is false;
- Lemma 2.4 omits the \(\sigma^2\) covariance factor and needs uncorrelated noise, not merely equal marginal variances;
- the definition \(\varepsilon:=X-S\) conflicts with the subsequent correct \(\varepsilon=Y-X\);
- the robust proposition and its probability bound are incomplete.

These defects lower confidence in extrapolating the note, but they do not undermine the exact code-level explanation.

## Gaps and Failure Modes

- **Lemma 2.5 is not a proof of constancy.** It has no proof and does not state whether its comparison constants are uniform in \(k\), \(t\), or a growing number of stages. Even if its constants were uniform, an approximately tenfold variance-constant difference can still satisfy a coarse \(\asymp\) bound. It cannot justify equality of scaled errors.
- **Finite sample versus asymptotics.** The last radius is only 15. The equivalent-kernel approximation is least credible there. The exact discrete operator calculation avoids this problem for the current linear run, but the theory claim remains asymptotic.
- **The error functional remains underspecified in the original claim.** Pointwise SD, \(\sqrt{E[\mathrm{RMSE}^2]}\), \(E[\mathrm{RMSE}]\), one realized RMSE, and sup error are distinct. A constant claim must name one.
- **No formal robust recursion result.** After stage 1, robust residuals are data-dependent and their noise is neither independent nor identically distributed. A one-step M-estimator expansion does not control \(J_k\prod_{j<k}(I-J_j)\).
- **Robust reference mismatch.** The notebook compares a noisy robust fit with the same robust procedure on clean data. The PDF's population target minimizes expected noisy loss. The small Gaussian stage-1 discrepancy does not show that this mismatch remains small under contamination or recursion.
- **Limited robust generalization.** The robust Monte Carlo fixes one signal, \(n=1000\), \(H=0.8\), one bandwidth factor, one contamination law, and 120 trials. The mechanism is persuasive for this notebook, not universally quantified.
- **No robust boundary ablation.** A nonperiodic clean signal is wrapped. The linear variance result is unaffected, but robust adaptive weights near the seam may change local and sup errors.
- **Sup-error normalization is unresolved.** The simulations show scaled single-pass sup means increasing from roughly 0.0378 to 0.0618. No simultaneous-band, entropy, or effective-multiplicity correction was derived.
- **Terminology can transfer wrong constants.** The code/PDF kernel is internally consistent but not conventional Epanechnikov. External variance constants for \(3(1-u^2)/4\) do not apply. The current kernel gives a stage-1 effective sample size 277.775, not the nominal 501.
- **Observation regimes differ between displayed methods.** Linear uses Gaussian-only data; robust uses Gaussian plus contamination. This does not cause the shared stage-1 pattern, but it prevents a clean numerical method comparison in the original presentation.
- **Contamination description is inconsistent.** The generator multiplies exponential magnitudes by random signs, so it is symmetric signed additive contamination, not one-sided contamination as some prose suggests.
- **Reconstruction is a weak validation.** Near-zero reconstruction error follows tautologically from subtracting each recorded component. It does not validate component interpretation or theoretical scaling.
- **Stored-artifact review limitation.** The diagnostic script is reproducible and its generated tables are internally consistent, but this review inspected rather than reran the full 160-second Monte Carlo. The executed companion notebook was checked for complete execution and errors.

## Confidence

- **Linear causal diagnosis: 0.99 (very high).** Exact algebra and exact finite-\(n\) variance independently force the observed qualitative and quantitative behavior.
- **No indexing, periodic-boundary, seed, or optimizer cause: 0.97 (very high).** Direct alignment checks, exact periodic variance, Monte Carlo percentiles, and convergence traces agree.
- **Interpretation of Lemma 2.5 as order rather than constant equality: 0.97 (very high).** The calligraphic operator and \(\asymp\) wording are visible in the primary document. Confidence is lower in the lemma's mathematical validity because it is unproved.
- **Robust mechanism for this notebook: 0.90 (high).** Gaussian-only controls, contaminated controls, Monte Carlo, and local linearization agree.
- **Formal robust theoretical coverage: 0.20 (low).** The supplied note does not prove the needed recursive clean-reference result.
- **Generalization to other signals, \(H\), schedules, and boundaries: 0.70 (moderate).** The operator principle is general, but the measured constants and robust ratios are experiment-specific.

Overall confidence in the decision “do not treat the first-stage spike as a code bug or a violation of the cited order result” is **very high**.

## Decision-Relevant Findings

1. **Do not alter the recursion to force a flat curve.** That would change the decomposition to satisfy an unsupported diagnostic expectation.
2. **State the theory correctly.** Lemma 2.5 concerns \(\mathcal W_k\), but only its \((nh_k)^{-1}\) order. It does not promise one common scaled-error constant.
3. **Separate stage 1 in interpretation.** It is the coarse low-pass/scaling term; stages 2+ are details. For the exact linear curve, stages 3--9 already have only 3.63% coefficient of variation after bandwidth scaling, but around a constant about three times below stage 1.
4. **Use the right standardizer for the question.**
   - To test white-noise calibration of linear recursive components, divide by the exact \(\sigma\|\ell_k\|_2\), or equivalently by \(\sigma\sqrt{(\mathcal W_k\mathcal W_k^\top)_{tt}}\) under wrap.
   - To illustrate only the bandwidth rate, keep \(a^{k/2}\) but label it an order normalization with stage-dependent constants.
   - Do not reuse the same claim for sup error.
5. **Treat the exponent offset as cosmetic.** \(a^{k/2}\) versus \(a^{(k-1)/2}\) changes all values by \(a^{-1/2}\). A schedule-aware display can use \(\sqrt{R_1/R_k}\), but this will not remove the first-stage gap.
6. **Do not claim a robust theorem from this PDF.** For robust validation, first choose the target—expected-noisy-loss population fit or clean-data fit—and then use a bootstrap or numerical-Jacobian/sandwich calibration for the recursive operator.
7. **Compare methods on matched inputs.** If linear-versus-robust performance is a goal, show both under Gaussian-only and contaminated observations, as the diagnostics already do.
8. **Correct surrounding interpretation before constants.** window_size is full support count, theoretical \(h_k\) is half-width, the current kernel is nonstandardly named, and the simulated contamination is signed. These do not cause the spike, but they can invalidate borrowed formulas or labels.

## Next Questions

1. Which quantity is the intended “scaled error”: pointwise SD, expected squared RMSE, expected realized RMSE, one realized RMSE, or a supremum?
2. Is stage 1 intended to be judged against later details, or explicitly reported as a separate coarse/scaling component?
3. Can the author provide a proof of Lemma 2.5 and specify whether its constants are uniform in stage and location?
4. For the robust comparison, which population target is scientifically intended: the PDF's expected-noisy-loss minimizer or the notebook's clean-data robust fit?
5. Should a follow-up diagnostic estimate robust stage energies with a parametric bootstrap, an automatic/numerical Jacobian, or both?
6. How stable are the robust stage ratios across \(n\), \(H/\sigma\), signal shape, contamination probability/scale, bandwidth factor, and nonperiodic boundary treatments?
7. If sup error matters, what simultaneous confidence-band or effective-multiplicity normalization should replace \(a^{k/2}\) alone?
8. Should future presentation replace “Epanechnikov” with the exact kernel formula and replace “one-sided contamination” with the actual signed contamination model?
