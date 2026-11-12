"""Leading-term (linearised) model for IRMF components k>=2: Monte Carlo test.

Reference path (phi-path): the IRMF recursion run on the clean signal with the
averaged contrast phi = E rho_H(. + eps) = rho_s, s = sqrt(H^2 + sigma^2) (exact for
Gaussian eps).  Stage 1 of this path is Note 1's population target S_t.

Per stage k, exact identity  e_k = l_k + R_k - Wo_k E_k - Phi_k  with
  e_k   = S~_k - So_k                        (component error, sup norm reported)
  l_k   = F^{-1} nabla_k (single-pass leading term on raw noise, Note 1)
  R_k   = single-pass remainder               (Prop 4.1 applied to signal ro_k)
  E_k   = sum_{j<k} e_j                       (accumulated fitting error)
  Phi_k = M_k(ro+eps) - M_k(ro+eps-E_k) - Wo_k E_k   (Jacobian fluctuation + quadratic)
Linear leading term  L_k = l_k - Wo_k G_k,  G_{k+1} = G_k + L_k  (G_1 = 0).
Run from the imf repo root:  python <path>/irmf_leading_term_mc.py [reps] [p_contam] [scale]
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import erf

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0         # contamination prob (additive)
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0     # contamination scale / sigma
AMP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0     # signal amplitude multiplier
SIGMA, H, SEED = 0.4, 0.8, 777
NB = "experiments/real-vs-calculated/gd_imf_real_vs_calculated_limits_noise_only.ipynb"

def load_signal():
    nb = json.load(open(NB)); ns = {"__name__": "nbx"}
    for i in [3, 5, 7, 9, 10]:
        exec(compile("".join(nb["cells"][i]["source"]), f"cell{i}", "exec"), ns)
        ns["display"] = lambda *a, **k: None; plt.close("all")
    return ns["x_clean"], ns["window_sizes"], ns["n"]

x, WS, n = load_signal(); K = len(WS); x = AMP * x
IDX, WGT = [], []
for w in WS:
    R = w // 2; d = np.arange(-R, R + 1)
    wt = np.maximum(0, 1 - np.abs(d / R)) ** 2; wt /= wt.sum()
    IDX.append((np.arange(n)[:, None] + d[None, :]) % n); WGT.append(wt)

psi  = lambda r, s: erf(r / (np.sqrt(2) * s))
rho2 = lambda r, s: np.sqrt(2 / np.pi) / s * np.exp(-0.5 * (r / s) ** 2)

def msmooth(y, k, s, tol=1e-13, it=100):
    """argmin_z sum_u rho_s(y_u - z) w_{u,t} for all t; safeguarded Newton."""
    Yw = y[IDX[k]]; w = WGT[k]
    lo, hi = Yw.min(1).copy(), Yw.max(1).copy(); z = Yw @ w
    for _ in range(it):
        f = (psi(Yw - z[:, None], s) * w).sum(1)          # decreasing in z
        lo = np.where(f > 0, z, lo); hi = np.where(f < 0, z, hi)
        fp = (rho2(Yw - z[:, None], s) * w).sum(1)
        zn = z + f / fp
        bad = (zn <= lo) | (zn >= hi); zn[bad] = 0.5 * (lo + hi)[bad]
        if np.max(np.abs(zn - z)) < tol: z = zn; break
        z = zn
    return z

# ---------- deterministic phi-path, population kernels Wo_k ----------
SPHI = np.sqrt(H ** 2 + SIGMA ** 2) if P == 0 else None
def phi_derivs(c):
    """E psi_H(c+eps), E rho_H''(c+eps) for the noise law (Gaussian or contaminated)."""
    if P == 0:
        return psi(c, SPHI), rho2(c, SPHI)
    # mixture: eps = G + B*C, C = scale*sigma*Exp*sign ; numerical quadrature on C
    a = (1 - P) * psi(c, SPHI0); b = (1 - P) * rho2(c, SPHI0)
    for node, wq in zip(LAG_X, LAG_W):          # C = +-scale*sigma*Exp(1), Gauss-Laguerre
        for sg in (1.0, -1.0):
            g = sg * SCALE * SIGMA * node
            a = a + P * 0.5 * wq * psi(c + g, SPHI0); b = b + P * 0.5 * wq * rho2(c + g, SPHI0)
    return a, b
LAG_X, LAG_W = np.polynomial.laguerre.laggauss(40)
SPHI0 = np.sqrt(H ** 2 + SIGMA ** 2)

def phi_smooth(y, k, it=100):
    Yw = y[IDX[k]]; w = WGT[k]; z = Yw @ w
    for _ in range(it):
        a, b = phi_derivs(Yw - z[:, None])
        zn = z + (a * w).sum(1) / (b * w).sum(1)
        if np.max(np.abs(zn - z)) < 1e-13: z = zn; break
        z = zn
    return z

ro, So, Wo, Wlin, Fk = [], [], [], [], []
r = x.copy()
for k in range(K):
    s = phi_smooth(r, k); ro.append(r.copy()); So.append(s)
    _, c2 = phi_derivs(r[IDX[k]] - s[:, None])
    A = np.zeros((n, n)); B = np.zeros((n, n))
    rows = np.repeat(np.arange(n), len(WGT[k]))
    np.add.at(A, (rows, IDX[k].ravel()), (c2 * WGT[k]).ravel())
    np.add.at(B, (rows, IDX[k].ravel()), np.tile(WGT[k], n))
    Fk.append(A.sum(1)); Wo.append(A / A.sum(1, keepdims=True)); Wlin.append(B)
    r = r - s

# ---------- operator diagnostics ----------
I = np.eye(n); diag = {"tilt_inf": [], "IminusWo_2": [], "prodWo_2": [], "prodWo_inf": [],
                       "compWo_inf_max": [], "prodW_2": [], "compW_inf_max": []}
Pk, Pl = I.copy(), I.copy()
for k in range(K):
    diag["tilt_inf"].append(np.abs(Wo[k] - Wlin[k]).sum(1).max())
    diag["IminusWo_2"].append(np.linalg.norm(I - Wo[k], 2))
    diag["prodWo_2"].append(np.linalg.norm(Pk, 2)); diag["prodWo_inf"].append(np.abs(Pk).sum(1).max())
    diag["prodW_2"].append(np.linalg.norm(Pl, 2))
    # max over j<=k of ||Wo_k prod_{j<i<k}(I-Wo_i)||_inf  (j=k-1 -> Wo_k itself)
    Q, Ql, best, bestl = I.copy(), I.copy(), 1.0, 1.0
    for i in range(k - 1, -1, -1):
        best = max(best, np.abs(Wo[k] @ Q).sum(1).max()); bestl = max(bestl, np.abs(Wlin[k] @ Ql).sum(1).max())
        Q = Q @ (I - Wo[i]); Ql = Ql @ (I - Wlin[i])
    diag["compWo_inf_max"].append(best); diag["compW_inf_max"].append(bestl)
    Pk = (I - Wo[k]) @ Pk; Pl = (I - Wlin[k]) @ Pl

# ---------- Monte Carlo ----------
keys = ["e", "L", "eta", "linW", "eL", "R", "Phi", "WD"]
sup = {q: np.zeros((REPS, K)) for q in keys}; rms = {q: np.zeros((REPS, K)) for q in keys}
rng = np.random.default_rng(SEED)
c_eta = None
for rep in range(REPS):
    eps = SIGMA * rng.normal(size=n)
    if P > 0:
        eps = eps + (rng.random(n) < P) * SCALE * SIGMA * rng.exponential(size=n) * rng.choice([-1., 1.], n)
    y = x + eps; rr = y.copy(); E = np.zeros(n); G = np.zeros(n)
    eta = psi(eps, H); eta = eta / (rho2(eps, H).mean() if c_eta is None else c_eta)
    Aeta = eta.copy(); Alin = eta.copy()
    for k in range(K):
        St = msmooth(rr, k, H); e = St - So[k]
        sp = msmooth(ro[k] + eps, k, H) - So[k]                 # single pass on raw noise
        l = (psi(ro[k][IDX[k]] + eps[IDX[k]] - So[k][:, None], H) * WGT[k]).sum(1) / Fk[k]
        Rk = sp - l
        WE = Wo[k] @ E; Phi = (sp - e) - WE                      # propagation minus Wo E
        L = l - Wo[k] @ G
        WD = Wo[k] @ (E - G)
        etam = Wo[k] @ Aeta; Aeta = Aeta - etam                  # Wo_k prod(I-Wo) eta
        linm = Wlin[k] @ Alin; Alin = Alin - linm                # W_k prod(I-W) eta
        for q, v in zip(keys, [e, L, etam, linm, e - L, Rk, Phi, WD]):
            sup[q][rep, k] = np.abs(v).max(); rms[q][rep, k] = np.sqrt((v ** 2).mean())
        rr = rr - St; E = E + e; G = G + L
tag = (f"p{P:g}_s{SCALE:g}" if P > 0 else "gauss") + (f"_amp{AMP:g}" if AMP != 1 else "")
np.savez(f"leading_term_mc_{tag}.npz", **{f"sup_{q}": sup[q] for q in keys},
         **{f"rms_{q}": rms[q] for q in keys}, **{k_: np.array(v) for k_, v in diag.items()},
         windows=np.array(WS), sigma=SIGMA, H=H, P=P, scale=SCALE)
print("windows", WS)
for k_, v in diag.items(): print(f"{k_:15s}", np.round(v, 4))
print("median over reps of sup-norm, by stage:")
for q in keys: print(f"  {q:5s}", np.round(np.median(sup[q], 0), 4))
print("median sup ratio |e-L|/|e|:", np.round(np.median(sup["eL"] / sup["e"], 0), 3))
print("median rms ratio |e-L|/|e|:", np.round(np.median(rms["eL"] / rms["e"], 0), 3))
