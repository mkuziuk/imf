# Prop 4.1 (reconstructed) and Lemma 4.2 (proof)

Source: Spokoiny, *Intrinsic robust multiscale filtering*, draft July 3, 2026 (`IMF.pdf`), §4 and Appendix A.7.
Scope: theory only. Notation below fixes OCR/layout corruption in the PDF.
**Patched** after the third Codex pass (`tmp/VERIFICATION.md`): (SC) on $Y$, argmins over $\Upsilon$, explicit $C_2$, probability "at least."

---

## Setup (from (4.1)–(4.2) and §3.1)

Fix bandwidth $h$ and kernel $K$, write $K_h(y)=K(y/h)$. Observations $Y_u=X_u+\varepsilon_u$. Write $\mathcal{X}=\sigma(X_v:v)$ for the full design. Contrast $\rho$ (e.g. smoothed absolute loss $\rho_H$ from Lemma 3.1–3.2).

**Noise given the design.** Conditionally on $\mathcal{X}$, the errors $(\varepsilon_u)$ are independent and identically distributed, with the same law as the generic $\varepsilon$ used to define
$$
\phi(z)\;=\;\mathbb{E}\bigl[\rho(z+\varepsilon)\bigr]
=\mathbb{E}\bigl[\rho(z+\varepsilon_u)\mid\mathcal{X}\bigr].
$$
(If the design is deterministic this is ordinary i.i.d. noise.) All expectations in $S_t$, $f$, and $\nabla_t$ below are **noise expectations given $\mathcal{X}$**.

**Parameter domain.** Fix a compact convex set $\Upsilon\subset\mathbb{R}$. Both local M-fits are defined **over $\Upsilon$**, not over all of $\mathbb{R}$:
$$
\begin{aligned}
\widetilde S_t
&=\arg\inf_{x\in\Upsilon}\sum_u\rho(Y_u-x)\,K_h(t-u),\\
S_t
&=\arg\inf_{x\in\Upsilon}\sum_u\mathbb{E}\bigl[\rho(Y_u-x)\mid\mathcal{X}\bigr]\,K_h(t-u)
=\arg\inf_{x\in\Upsilon}\sum_u\phi(X_u-x)\,K_h(t-u).
\end{aligned}
$$
Assume $S_t\in\operatorname{int}(\Upsilon)$, so the first-order condition $f'(S_t)=0$ holds. (If one prefers the unrestricted problems on $\mathbb{R}$, it is enough to prove that both unrestricted minimizers lie in $\operatorname{int}(\Upsilon)$ and then restrict; for strictly convex $\rho_H$ the unrestricted minimizers are unique, so the two formulations coincide as soon as those points sit in $\operatorname{int}(\Upsilon)$.)

Define the population and empirical local criteria
$$
f(x)=\sum_u\phi(X_u-x)\,K_h(t-u),\qquad
f_\circ(x)=\sum_u\rho(Y_u-x)\,K_h(t-u),
$$
so $x^\ast=S_t=\arg\inf_{x\in\Upsilon}f$ and $x^\circ=\widetilde S_t=\arg\inf_{x\in\Upsilon}f_\circ$. Both $x^\ast$ and $f$ are $\mathcal{X}$-measurable; $x^\circ$ and $f_\circ$ depend on $Y$.

Chain rule in the arg-variable $x$:
$$
\begin{aligned}
f'(x)&=-\sum_u\phi'(X_u-x)\,K_h(t-u),&
f_\circ'(x)&=-\sum_u\rho'(Y_u-x)\,K_h(t-u),\\
f''(x)&=\sum_u\phi''(X_u-x)\,K_h(t-u),&
f_\circ''(x)&=\sum_u\rho''(Y_u-x)\,K_h(t-u).
\end{aligned}
$$
At the interior population minimizer, $f'(x^\ast)=0$. Also
$$
\sum_u\mathbb{E}\bigl[\rho'(Y_u-x^\ast)\mid\mathcal{X}\bigr]\,K_h(t-u)
=\sum_u\phi'(X_u-x^\ast)\,K_h(t-u)=0.
$$

As in the draft,
$$
\begin{aligned}
F_t&=\sum_u\phi''(X_u-S_t)\,K_h(t-u)=f''(x^\ast),\\
\nabla_t&=\sum_u\bigl\{\rho'(Y_u-S_t)-\mathbb{E}\bigl[\rho'(Y_u-S_t)\mid\mathcal{X}\bigr]\bigr\}\,K_h(t-u),\\
D_t&=F_t^{1/2}.
\end{aligned}
$$
Thus $\nabla_t=-f_\circ'(x^\ast)$. Precisely, $F_t=N_t(S_t)$ from (3.3). The draft shorthand "$F=N_t$" with $N_t=N_t(X_t)$ is automatic under constant curvature (including quadratic $\rho$, where $\phi''\equiv 1$), and can also hold accidentally when $S_t=X_t$; it is not an identity for general $\rho$.

Write $F=f''(x^\ast)=F_t$, $F_\circ=f_\circ''(x^\ast)$, $D=F^{1/2}$. Let $U=U_r(x^\ast)=\{x:\lvert D(x-x^\ast)\rvert\le r\}$ be the local elliptic neighborhood from Appendix A; we take $U\subset\Upsilon$.

---

## Proposition 4.1 (cleaned statement)

**Sign.** Theorem A.9 / (A.8) controls the Newton residual
$$
x^\circ-x^\ast+F_\circ^{-1}f_\circ'(x^\ast)
=\widetilde S_t-S_t-F_\circ^{-1}\nabla_t.
$$
After replacing $F_\circ$ by $F$, the residual matching (A.8) is $\widetilde S_t-S_t-F^{-1}\nabla_t$. The PDF display uses $\widetilde S_t-S_t+F_t^{-1}\nabla_t$; with $\nabla_t$ as defined above, that is the **wrong sign**. (Equivalently keep "$+$" if one redefines $\nabla_t:=f_\circ'(S_t)$.)

**Standing assumptions (not part of $\Omega_t$).**
Assume $K\ge 0$, $N_h(t)>0$, and $F=f''(x^\ast)\ge c_\ast N_h(t)>0$ (so $D\asymp\sqrt{N_h(t)}$). These three are $\mathcal{X}$-measurable.

**(SC)** (strong convexity of $f_\circ$ on $\Upsilon$). Require
$$
\mu(Y,t,\Upsilon)
:=\inf_{x\in\Upsilon}f_\circ''(x)
=\inf_{x\in\Upsilon}\sum_u\rho''(Y_u-x)\,K_h(t-u)
\ge\mu_0>0.
$$
This is a statement about the **empirical** Hessian, hence about $Y_u-x$, not $X_u-x$. It is **not** $\mathcal{X}$-measurable in general: the modulus $\mu(Y,t,\Upsilon)$ depends on $Y$. For $\rho_H$, the bound holds **pathwise** for every finite realization: $K\ge 0$, $N_h(t)>0$, $\rho_H''(z)>0$ for every finite $z$, and $\rho_H''$ continuous on compact $\Upsilon$ give $\mu(Y,t,\Upsilon)>0$ with no probability cost. The numerical value of $\mu$ may still depend on $Y$. This is *not* a uniform modulus on all of $\mathbb{R}$, where $\rho_H''(z)\to 0$ as $\lvert z\rvert\to\infty$.

**Proposition 4.1 (reconstructed; fixed $t$).**
Fix $t\in[0,1]$. Assume Lemma 4.2 at this $t$, **(SC)**, and the numerical conditions **(A.9-num)** below, which are **standing** (checked from $C_1,C_2,C_3$ after the concentration event, not folded into $\Omega_t$ as extra random constraints). Let $\Omega_t(x)$ be the concentration event of Lemma 4.2, so
$$
\mathbb{P}\bigl(\Omega_t(x)\bigr)\ge 1-e^{-x}.
$$
On $\Omega_t(x)$, writing $\delta=C_2/D$,
$$
D\Bigl\lvert\widetilde S_t-S_t-F^{-1}\nabla_t\Bigr\rvert
\;\le\;
\frac{3\,C_1^2\,C_3}{4\,D\,(1-\delta)^3}
\;+\;
\frac{\delta\,C_1}{1-\delta}.
$$

**(A.9-num).** On $\Omega_t(x)$ one has $\lvert\nabla_t\rvert/D\le C_1$ and $\lvert F-F_\circ\rvert/D\le C_2$, hence $b_\circ\le C_1/(1-\delta)$ once $\delta<1$. Require
$$
\delta=\frac{C_2}{D}<1,\qquad
\kappa^2=\frac{1}{1-\delta},\qquad
r\ge\frac{3C_1}{2(1-\delta)},\qquad
\kappa^2\cdot\frac{C_3}{D}\cdot\frac{C_1}{1-\delta}<\frac{4}{9}.
$$
These imply $D^2\le\kappa^2 F_\circ$, $r\ge 3b_\circ/2$, and $\kappa^2\tau_3 b_\circ<4/9$ with $\tau_3=C_3/D$. They are hypotheses on $(C_1,C_2,C_3,D,r)$, not extra random events. If they fail, the proposition is silent; they are not used to shrink $\mathbb{P}(\Omega_t(x))$. ($C_1,C_2$ are $\mathcal{X}$-measurable Bernstein thresholds as written below.)

**Uniform-in-$t$ (honest statement).**
The display is for a **fixed** $t$. The PDF's $\sup_{t\in[0,1]}$ is not obtained from a fixed-$t$ Bernstein argument. A simultaneous event needs an empirical-process argument in the joint index $t$, plus regularity of $K$ in $t$.

**Derivation (corrected constants).**
From Lemma 4.2(i) and $F=D^2$,
$$
\frac{\lvert F_\circ-F\rvert}{F}\le\delta:=\frac{C_2}{D},
\qquad
F_\circ\ge(1-\delta)F,\qquad
\frac{F}{F_\circ}\le\frac{1}{1-\delta}.
$$
With $b_\circ=\lvert D\,F_\circ^{-1}f_\circ'(x^\ast)\rvert$ and $f'(x^\ast)=0$,
$$
b_\circ=\frac{D}{F_\circ}\lvert\nabla_t\rvert\le\frac{C_1}{1-\delta}.
$$
Theorem A.9 / (A.8) with $\tau_3=C_3/D$ gives, writing $R_\circ=x^\circ-x^\ast+F_\circ^{-1}f_\circ'(x^\ast)$,
$$
\frac{F_\circ}{D}\lvert R_\circ\rvert
\le\frac{3\tau_3}{4}\,b_\circ^2
=\frac{3C_3}{4D}\,b_\circ^2,
$$
hence
$$
\lvert R_\circ\rvert
\le\frac{3C_3}{4F_\circ}\,b_\circ^2
\le\frac{3C_3}{4D^2(1-\delta)}\,b_\circ^2
\le\frac{3C_1^2 C_3}{4D^2(1-\delta)^3}.
$$
Replacing $F_\circ^{-1}$ by $F^{-1}$ costs
$$
\bigl\lvert(F_\circ^{-1}-F^{-1})f_\circ'(x^\ast)\bigr\rvert
\le\frac{\delta}{1-\delta}\,F^{-1}\lvert f_\circ'(x^\ast)\rvert
\le\frac{\delta}{1-\delta}\,\frac{C_1}{D}.
$$
Multiply by $D$ and use $f_\circ'(x^\ast)=-\nabla_t$ to get the displayed scaled bound.

---

## Lemma 4.2 (cleaned statement)

**Lemma 4.2 (fixed $t$).**
Fix $t\in[0,1]$. There exists a random set $\Omega_t(x)$, depending only on the score and Hessian concentration, with
$$
\mathbb{P}\bigl(\Omega_t(x)\bigr)\ge 1-e^{-x},
$$
and constants $C_1,C_2$ (using the shifted level $y=x+\log 4$ below) together with deterministic $C_3,C_4$, such that on $\Omega_t(x)$:

**(i)** Score and Hessian concentration at $x^\ast$:
$$
\frac{\lvert f'(x^\ast)-f_\circ'(x^\ast)\rvert}{D}
=\frac{1}{D}\Biggl\lvert\sum_u\bigl\{\rho'(Y_u-x^\ast)-\mathbb{E}[\rho'(Y_u-x^\ast)\mid\mathcal{X}]\bigr\}K_h(t-u)\Biggr\rvert
\le C_1,
$$
$$
\frac{\lvert f''(x^\ast)-f_\circ''(x^\ast)\rvert}{D}
=\frac{1}{D}\Biggl\lvert\sum_u\bigl\{\rho''(Y_u-x^\ast)-\mathbb{E}[\rho''(Y_u-x^\ast)\mid\mathcal{X}]\bigr\}K_h(t-u)\Biggr\rvert
\le C_2.
$$
Equivalently $\lvert\nabla_t\rvert/D\le C_1$ and $\lvert F-F_\circ\rvert/D\le C_2$. Explicitly, with $M_j=\lVert\rho^{(j)}\rVert_\infty$, $w_u=K_h(t-u)$, $K_\infty=\lVert K\rVert_\infty$, and $y=x+\log 4$,
$$
v_j=M_j^2\sum_u w_u^2,\qquad
b_j=2M_j K_\infty,\qquad
C_j=\frac{\sqrt{2v_j\,y}+b_j y/3}{D},\qquad j=1,2.
$$

**(ii)** Third-derivative control for $(T3^\ast)$ of $f_\circ$ on $U$ (deterministic; not a concentration event):
$$
\sup_{x\in U}\frac{1}{D^2}\Biggl\lvert\sum_u\rho'''(Y_u-x)\,K_h(t-u)\Biggr\rvert
\le C_3
:=\frac{\lVert\rho'''\rVert_\infty\,N_h(t)}{D^2}.
$$
In particular $\tau_3\le C_3/D$ works in $(T3^\ast)$ for $f_\circ$ in the $D$-metric.

**(iii)** Hessian continuity of $f$ on $U$ (deterministic; unused for A.9):
$$
\sup_{x\in U}\frac{1}{D^2}\Biggl\lvert\sum_u\bigl(\phi''(X_u-x)-\phi''(X_u-x^\ast)\bigr)\,K_h(t-u)\Biggr\rvert
\le C_4
:=\frac{r\,\lVert\phi'''\rVert_\infty\,N_h(t)}{D^3}.
$$

---

## Proof of Lemma 4.2

### Standing assumptions

1. **Independence / law.** Conditionally on $\mathcal{X}$, $(\varepsilon_u)$ are i.i.d. with the law defining $\phi$. Then
$$
\mathbb{E}\bigl[\rho^{(j)}(Y_u-x^\ast)\mid\mathcal{X}\bigr]=\phi^{(j)}(X_u-x^\ast),
$$
because $x^\ast$ is $\mathcal{X}$-measurable. Conditional independence of the $\varepsilon_u$ alone is not enough without matching the law of $\phi$.
2. **Contrast regularity.** $\rho\in C^3$ with $\rho',\rho'',\rho'''$ bounded. For $\rho_H$: $\lvert\rho'\rvert\le 1$, $0\le\rho''\le 2/(H\sqrt{2\pi})$, $\lvert\rho'''\rvert\le 2/(H^2\sqrt{2\pi})$.
3. **Kernel / curvature.** $K\ge 0$ bounded; $N_h(t)\asymp nh$; $F\ge c_\ast N_h(t)>0$. These are $\mathcal{X}$-measurable.
4. **(SC)** and **(A.9-num)** as in Proposition 4.1. These are standing, not events. **(SC)** is a pathwise lower bound on $f_\circ''$ (depends on $Y$). **(A.9-num)** is a condition on the $\mathcal{X}$-measurable thresholds $C_1,C_2,C_3,D,r$.

### What $\Omega_t(x)$ is

Only the two Bernstein events of Steps 1–2 (score and Hessian). Steps 3–4 are deterministic. **(SC)** and **(A.9-num)** are hypotheses. The level $x>0$ is the exponential-tail parameter (clash with the arg-variable $x$). This is a **fixed-$t$** event.

### Step 1 — score

Conditionally on $\mathcal{X}$, set
$$
Z_u=\bigl(\rho'(Y_u-x^\ast)-\mathbb{E}[\rho'(Y_u-x^\ast)\mid\mathcal{X}]\bigr)\,K_h(t-u).
$$
Then $\mathbb{E}[Z_u\mid\mathcal{X}]=0$, the $Z_u$ are conditionally independent, and $\sum_u Z_u=\nabla_t$. Boundedness: $\lvert Z_u\rvert\le 2M_1 w_u$ with $M_1=\lVert\rho'\rVert_\infty$. Variance proxy $v_1=M_1^2\sum_u w_u^2$. Bernstein: for all $y\ge 0$,
$$
\mathbb{P}\Bigl(\Bigl\lvert\sum_u Z_u\Bigr\rvert\ge\sqrt{2v_1\,y}+\tfrac{b_1}{3}y\Bigm|\,\mathcal{X}\Bigr)\le 2e^{-y},
$$
with increment $b_1=2M_1 K_\infty$.

### Step 2 — Hessian

The same argument with $M_2=\lVert\rho''\rVert_\infty$, $v_2=M_2^2\sum_u w_u^2$, $b_2=2M_2 K_\infty$, and the same two-sided failure $2e^{-y}$.

### Union bound (factor $2$ from two-sided tails)

There are $m=2$ two-sided Bernstein inequalities, hence $2m=4$ elementary exponential tails. The union failure probability is at most $4e^{-y}$. Take
$$
y=x+\log 4=x+\log(2m).
$$
Then $4e^{-y}=e^{-x}$, so the intersection $\Omega_t(x)$ of Steps 1–2 satisfies $\mathbb{P}(\Omega_t(x))\ge 1-e^{-x}$. On that event,
$$
C_1=\frac{\sqrt{2v_1\,y}+b_1 y/3}{D},\qquad
C_2=\frac{\sqrt{2v_2\,y}+b_2 y/3}{D}.
$$

### Step 3 — $C_3$ (deterministic)

$$
\Biggl\lvert\sum_u\rho'''(Y_u-x)\,K_h(t-u)\Biggr\rvert
\le\lVert\rho'''\rVert_\infty N_h(t)
\quad\text{for all }x,
$$
hence $C_3=\lVert\rho'''\rVert_\infty N_h(t)/D^2$ on $U$. No probability cost.

### Step 4 — $C_4$ (deterministic; unused for A.9)

Mean value theorem: $\lvert\phi''(X_u-x)-\phi''(X_u-x^\ast)\rvert\le\lVert\phi'''\rVert_\infty\lvert x-x^\ast\rvert$, so for $x\in U_r(x^\ast)$
$$
C_4=\frac{r\,\lVert\phi'''\rVert_\infty\,N_h(t)}{D^3}.
$$

### Closing

$\Omega_t(x)$ is the intersection of Steps 1–2 only. Its probability is **at least** $1-e^{-x}$ by the $y=x+\log 4$ shift. Proposition 4.1 then applies Theorem A.9 on $\Omega_t(x)$ **provided** **(SC)** and **(A.9-num)** hold for those $C_j$.

---

## Gaps still present in the draft PDF (unchanged paper issues)

1. Prop 4.1 RHS blank; filled with the $(1-\delta)^3$ bound.
2. Sign typo on the Newton residual.
3. Lemma 4.2 OCR / blank (iii).
4. $C_j$ and $\Omega$ undefined; A.9 numerics not checked.
5. PDF writes $\sup_t$ with no chaining.
6. "$F=N_t$" should be $F_t=N_t(S_t)$.
7. §4 is one-step only.

---

## What is proved here vs still open

- **Proved (fixed $t$, minimizers over $\Upsilon$, under (SC) and (A.9-num)):** Lemma 4.2(i) on an event of probability $\ge 1-e^{-x}$; (ii)–(iii) deterministically; Prop 4.1 via Theorem A.9 with the $(1-\delta)^3$ constant and explicit $C_1,C_2$.
- **Still open in the note:** uniform-in-$t$ Prop 4.1; recursive multi-step IMF theorem.
