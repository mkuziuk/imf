## Summary

The theory does **not** say that the notebook's scaled recursive error must be one constant at every IMF stage.

The source of the scaling is `IMF.pdf`, page 3, \(h_{k+1}=h_k/a\), together with Lemma 2.5, page 6:

\[
\mathcal W_k=W_kA_{k-1},\qquad
(\mathcal W_k\mathcal W_k^\top)_{tt}\asymp (nh_k)^{-1}.
\]

Here \(\mathcal W_k\), not \(W_k\), is the full recursive component operator. For white noise, \(e_k=\mathcal W_k\varepsilon\), so the lemma implies only

\[
\operatorname{sd}\{e_k(t)\}\asymp \sigma (nh_k)^{-1/2}
=\sigma(nh_1)^{-1/2}a^{(k-1)/2}.
\]

Thus bandwidth scaling gives a common **order**, not equality or a stage-independent variance constant. The note uses \(\asymp\), gives no proof, and does not specify uniform constants in \(k\).

The first-stage spike is expected from the full operator. At stage 1, \(\mathcal W_1=W_1\), a low-pass smoother of raw noise. For \(k>1\),

\[
\mathcal W_k=W_k\prod_{j<k}(I-W_j),
\]

so the previous residual subtractions add high-pass factors and make later components band-pass. Scaling by \(h_k^{-1/2}\) removes dilation, but not this large change in filter energy.

An exact finite-\(n\) calculation for the notebook's linear filters predicts this almost perfectly. With \(n=1000\), wrap boundaries, the actual windows, \(a=\sqrt2\), and \(\sigma=0.4\), the predicted recursive RMSE divided by \(a^{k/2}\) is 0.02018 at stage 1 and about 0.0063--0.0067 after the transient, a factor of about 3.1. The stored notebook output is 0.020249 at stage 1 and has a stage-4-to-9 mean of 0.006420, also a factor of about 3.15. The large first linear recursive error is therefore not a violation of the operator theory.

There is no theorem in the note proving scaled-error constancy for recursive robust IMFs. The robust part gives a one-step M-estimator expansion, merely defines the later recursion, and has an incomplete Proposition 4.1. Also, the notebook's robust “real IMF” is a robust fit to clean data, whereas the PDF's population target minimizes the **expected noisy contrast**. These generally differ. The robust first-stage effect is consistent with the same low-pass-versus-band-pass mechanism after local linearization, but that is an inference, not a result proved in `IMF.pdf`.

## Key Evidence

### Lemmas 2.3--2.5

Pages 5--6 define

\[
A_k=(I-W_k)\circ\cdots\circ(I-W_1),\qquad
\mathcal W_k=W_kA_{k-1}.
\]

Lemma 2.3 gives the true linear component \(S_k=\mathcal W_kX\), and immediately afterward the note gives

\[
\widetilde S_k-S_k=\mathcal W_k\varepsilon .
\]

Therefore, under \(\operatorname{Var}(\varepsilon)=\sigma^2I\),

\[
\operatorname{Var}\{\widetilde S_k(t)-S_k(t)\}
=\sigma^2(\mathcal W_k\mathcal W_k^\top)_{tt}.
\]

Lemma 2.5 states that the diagonal is of order \((nh_k)^{-1}\). It does not state an exact constant, convergence of a scaled curve, or a realized-error result.

There is an important mistake in the paragraph between Lemmas 2.4 and 2.5. It equates component error with residual noise:

\[
E\|\widetilde S_k-S_k\|^2=E\|\varepsilon^{(k)}\|^2.
\]

This is generally false: component error is \(W_kA_{k-1}\varepsilon\), while residual noise is \(A_k\varepsilon\). Lemma 2.5 returns to the correct component operator \(\mathcal W_k\).

### Exact reason the hidden constant changes

On the notebook's periodic regular grid, all \(W_k\) are circulant. If \(w_k(\omega_m)\) is the Fourier multiplier of \(W_k\), then

\[
q_k(\omega_m)
=w_k(\omega_m)\prod_{j<k}\{1-w_j(\omega_m)\},
\]

and

\[
\operatorname{Var}\{e_k(t)\}
=\frac{\sigma^2}{n}\sum_m|q_k(\omega_m)|^2.
\]

The stage-1 product is empty. For every later stage, \(w_j(0)=1\), so the multiplier vanishes at zero frequency. The first component is low-pass; the others are band-pass.

In an equivalent-kernel approximation,

\[
\operatorname{Var}\{e_k(t)\}
\sim \frac{\sigma^2}{nh_k}C_k,
\]

where, up to Fourier convention,

\[
C_k=\frac1{2\pi}\int|\widehat L(z)|^2
\prod_{\ell=1}^{k-1}|1-\widehat L(a^\ell z)|^2\,dz,
\qquad L=K/\!\int K.
\]

The rate \(1/(nh_k)\) is shared; \(C_k\) is not. For the displayed/code kernel
\(K(u)=\tfrac34(1-|u|)^2_+\), the one-pass/stage-1 constant is

\[
C_1=\int L^2=0.9.
\]

For the exact discrete recursive filters, \(C_k=(nh_k)\|q_k\|^2\) is about 0.09 after stages 3--4, roughly one tenth of \(C_1\). The corresponding standard deviations differ by about \(\sqrt{10}\) after bandwidth scaling.

### Exact discrete check

For a periodic unit impulse \(\delta\), I reproduced the notebook's weights and used

\[
q_k=W_kr_{k-1},\qquad r_k=r_{k-1}-q_k,\qquad r_0=\delta.
\]

Circulant symmetry yields
\(\mathbb E\{\operatorname{RMSE}(e_k)^2\}=\sigma^2\|q_k\|^2\).
The PDF half-width is \(h_k=R_k/(n-1)\), with
\(R_k=(\texttt{window\_size}_k-1)/2\).

| k | window / radius | \(C_k\) recursive | predicted recursive scaled RMS | predicted one-pass scaled RMS |
|---:|---:|---:|---:|---:|
| 1 | 501 / 250 | 0.900911 | 0.020182 | 0.020182 |
| 2 | 355 / 177 | 0.151774 | 0.008278 | 0.020169 |
| 3 | 251 / 125 | 0.109011 | 0.007020 | 0.020182 |
| 4 | 177 / 88 | 0.097770 | 0.006663 | 0.020227 |
| 5 | 125 / 62 | 0.092771 | 0.006502 | 0.020264 |
| 6 | 89 / 44 | 0.088974 | 0.006356 | 0.020229 |
| 7 | 63 / 31 | 0.088901 | 0.006365 | 0.020270 |
| 8 | 45 / 22 | 0.087500 | 0.006303 | 0.020240 |
| 9 | 31 / 15 | 0.095902 | 0.006720 | 0.020628 |

The one-pass scaled RMS is flat. The recursive scaled RMS necessarily has a high first stage and then a lower plateau.

The current stored outputs agree:

- linear recursive scaled RMSE: 0.020249 at stage 1, then 0.007485, 0.005903, 0.005170, 0.005631, 0.006692, 0.007421, 0.006632, 0.006974;
- linear one-pass scaled RMSE: 0.020249, 0.019819, 0.019297, 0.018604, 0.018025, 0.018129, 0.019129, 0.019913, 0.020827;
- robust recursive scaled RMSE: 0.029067 at stage 1 versus a stage-2-to-9 mean of 0.007357.

### Indexing, norm, and bandwidth

The PDF starts at \(k=1\) and uses \(h_{k+1}=h_k/a\), hence

\[
\sqrt{h_1/h_k}=a^{(k-1)/2}.
\]

The notebook divides by \(a^{k/2}\). This differs by the same constant \(a^{1/2}\) at every stage, so it changes only vertical scale and cannot create or remove the stage-1 spike. With odd-window rounding, the exact finite-schedule normalizer is
\(\sqrt{h_1/h_k}=\sqrt{R_1/R_k}\).

The PDF's \(h_k\) is a half-width in design units. The notebook's `window_size` is the full number of grid points. The first half-width is \(250/999\), not 0.501.

Page 4 lists squared \(L_2\) error and sup error as desired metrics. Lemma 2.5 is only a pointwise variance-diagonal statement. On the periodic grid it implies the order of the **expected squared RMSE**, but not equality of one realized RMSE. It provides no completed sup-norm theorem. A supremum also has a multiplicity/correlation factor, so \(a^{-k/2}\) alone need not flatten it.

### Robust Section 4 and target mismatch

Pages 9--10 define the one-step population target

\[
S_t=\arg\min_x\sum_u E\rho(Y_u-x)K_h(t-u)
\]

and give the Fisher-type approximation

\[
\widetilde S_t-S_t
\approx F_t^{-1}\sum_u
\{\rho'(Y_u-S_t)-E\rho'(Y_u-S_t)\}K_h(t-u).
\]

For independent observations and nondegenerate local curvature, this suggests a one-step \(1/\sqrt{nh}\) rate with a sandwich constant depending on score variance, curvature, \(H\), signal variation, and the noise law.

Section 3.2 only defines recursive \(Y^{(k)}\), \(X^{(k)}\), \(\varepsilon^{(k)}\), and \(\phi_k\). Section 4 then analyzes a single fit again. It never controls the nonlinear residual-map composition. Proposition 4.1 is visibly unfinished: its displayed right-hand side is blank (`\le ,`), and the probability bound's constants/conditions are incomplete.

The notebook's robust clean reference is

\[
\arg\min_x\sum_u\rho(X_u^{(k)}-x)K_{h_k}(t-u),
\]

whereas the PDF target is

\[
\arg\min_x\sum_uE\rho(X_u^{(k)}+\varepsilon_u^{(k)}-x)K_{h_k}(t-u).
\]

For quadratic loss and zero-mean noise the expectation adds an \(x\)-independent term, so the targets coincide. For the smoothed absolute loss they generally do not, except in special cases such as a locally constant signal with suitable symmetry. Hence the PDF's robust expansion is not directly a theorem for the notebook's clean-fit-versus-noisy-fit error.

A plausible local-linearization is

\[
e_k\approx J_k\prod_{j<k}(I-J_j)\varepsilon,
\]

which has the same stage-1 low-pass/later band-pass structure. The note does not derive or bound this Jacobian product.

## Method or Context Details

The local primary document is `IMF.pdf`, *Intrinsic robust multiscale filtering*, Vladimir Spokoiny (WIAS/HU Berlin), dated July 3, 2026, 21 pages. SHA-256:
`bfec11fb67f0348edf4e57b4174f27e0f21e92190c27d6c3ec06af4a9deafc6e`.
It has no journal, DOI, report number, or arXiv identifier in its metadata.

The linear rate interpretation requires assumptions that are not all stated beside Lemma 2.5:

- quadratic loss/local mean;
- regular design with \(nh_k\) effective points and a fixed nondegenerate kernel;
- independent or uncorrelated homoscedastic noise, \(\operatorname{Var}(\varepsilon)=\sigma^2I\);
- controlled boundaries, preferably the notebook's regular periodic case;
- \(nh_k\) large for an equivalent-kernel approximation;
- fixed \(k\), unless an additional uniform-in-\(k\) theorem is supplied.

Equal marginal variances alone do not imply \(\sigma^2I\). The exact discrete calculation avoids asymptotic error at the notebook's smallest radius of 15.

| Quantity | Support in the note | Does scaling force one constant? |
|---|---|---|
| Linear one-pass pointwise SD | Standard \(1/\sqrt{nh_k}\) kernel rate | Approximately, under exact dilation |
| Linear recursive pointwise SD / expected RMSE | Lemma 2.5 gives the same order | No; \(C_k\) changes |
| Linear sup error | No completed bound | No |
| Robust one-step fit vs PDF population target | Heuristic/Fisher expansion | Only an order heuristic with changing sandwich constant |
| Robust recursive component | Recursion defined, no rate theorem | No |
| Robust noisy fit vs notebook clean fit | Different target from PDF | No |

## Sources

1. Vladimir Spokoiny, *Intrinsic robust multiscale filtering*, July 3, 2026, [local PDF](/Users/mikhail/Projects/imf/IMF.pdf), pages 3--11. Public project copy: [GitHub](https://github.com/mkuziuk/imf/blob/main/IMF.pdf).
2. [`gd_imf_real_vs_calculated.ipynb`](/Users/mikhail/Projects/imf/gd_imf_real_vs_calculated.ipynb), current working-tree version, especially cells `a7f99780`, `25b95378`, `91eeb55c`, `267ea198`, `b66696e1`, and `e1ea4c6c`.
3. [`gd_imf_real_vs_calculated_review.tex`](/Users/mikhail/Projects/imf/gd_imf_real_vs_calculated_review.tex), Sections 1--2. This is secondary project documentation, not an independent theorem source.
4. Vladimir Spokoiny, *Sharp bounds in perturbed smooth optimization*, arXiv:2505.02002: [primary arXiv record](https://arxiv.org/abs/2505.02002), [DOI](https://doi.org/10.48550/arXiv.2505.02002). This supports the generic optimization appendix, not Lemma 2.5.
5. Vladimir Spokoiny, *Estimation and inference in error-in-operator model*, arXiv:2504.11834: [primary arXiv record](https://arxiv.org/abs/2504.11834). This is the identifier incorrectly printed for *Sharp bounds...* in `IMF.pdf`.
6. [Vladimir Spokoiny's WIAS publication list](https://wias-berlin.de/people/spokoiny/publications.jsp), checked July 9, 2026.

## Contradictions or Uncertainty

1. Lemma 2.5 has no proof or stated uniformity in \(t,k,h\), or growing \(k_{\max}\).
2. Page 6 confuses component error \(W_kA_{k-1}\varepsilon\) with residual noise \(A_k\varepsilon\) in an expected-\(L_2\) equality.
3. Lemma 2.4 omits the \(\sigma^2\) factor: under white noise it should be
   \(\Sigma_k=\sigma^2A_kA_k^\top\). Equal marginal variances are insufficient without zero covariances.
4. Page 6 first prints “\(\varepsilon:=X-S\)” and later correctly defines \(\varepsilon=Y-X\); the former is evidently a typo.
5. Robust Proposition 4.1 is incomplete and cannot serve as a quantitative sup-error theorem.
6. No recursive robust variance theorem or clean-reference comparison theorem is proved.
7. The reference “*Sharp bounds...*, arxiv.org/2504.11834” is wrong; the correct identifier is 2505.02002.
8. Exact-title searches and the WIAS list did not reveal a separate public publication record for *Intrinsic robust multiscale filtering* as of July 9, 2026. I verified only the local/project GitHub copy; another private or unindexed version may exist.
9. The note calls \(\tfrac34(1-|u|)^2_+\) Epanechnikov; the conventional Epanechnikov formula is \(\tfrac34(1-u^2)_+\). The notebook matches the note's displayed formula, so this is not the cause, but it changes numerical constants.

## Open Questions

1. Can the author supply a proof of Lemma 2.5 and clarify whether its constants are uniform in \(k\) and include stage 1?
2. Is the intended robust target the expected-noisy-loss population IMF or the notebook's same-contrast clean-data IMF?
3. Is “constant scaled error” intended for pointwise SD, expected RMSE, realized RMSE, or supremum? The note supports none as an exact constant.
4. Should the first coarse low-pass component be compared with later band-pass components, or should scale constancy be assessed only after a burn-in?
5. For a robust theorem, can one control \(J_k\prod_{j<k}(I-J_j)\), including residual dependence, curvature changes, contamination, and numerical optimization error?
6. If a flat finite-sample diagnostic is wanted, should it use exact operator energy
   \(\sigma\sqrt{(\mathcal W_k\mathcal W_k^\top)_{tt}}\) for the linear case and a bootstrap/Jacobian analogue for the robust case rather than bandwidth alone?
