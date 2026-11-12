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


# ---------- IMF.pdf true IMFs: S^(k)_t = argmin sum_u a E rho(Y^(k)_u - x), expectation over reps ----------
REPS2 = 400
rng = np.random.default_rng(4242)
Yk = np.zeros((K, REPS2, n), dtype=np.float32)
for rep in range(REPS2):
    rr = x + SIGMA * rng.normal(size=n)
    for k in range(K):
        Yk[k, rep] = rr; rr = rr - msmooth(rr, k, H)
def mean_target(Ys, k, it=30):
    """argmin_x sum_u a_{u,t} mean_rep rho_H(Ys[rep,u]-x), safeguarded Newton, chunked over t."""
    w = WGT[k]; out = np.empty(n)
    for c0 in range(0, n, 50):
        idx = IDX[k][c0:c0+50]                     # (c, win)
        Yw = Ys[:, idx].astype(np.float64)         # (R, c, win)
        z = Yw.mean(0) @ w
        for _ in range(it):
            d = Yw - z[None, :, None]
            f = (psi(d, H) * w).sum(2).mean(0); fp = (rho2(d, H) * w).sum(2).mean(0)
            zn = z + f / fp
            if np.max(np.abs(zn - z)) < 1e-12: z = zn; break
            z = zn
        out[c0:c0+50] = z
    return out
res = []
for k in range(K):
    full = mean_target(Yk[k], k); h1 = mean_target(Yk[k, :REPS2//2], k); h2 = mean_target(Yk[k, REPS2//2:], k)
    d = full - So[k]; se = np.abs(h1 - h2) / 2           # MC error of the full-sample target
    res.append((k+1, WS[k], np.abs(d).max(), np.sqrt((d**2).mean()), np.sqrt((se**2).mean()), np.abs(se).max()))
    print(f"stage {k+1} win {WS[k]:3d}: sup|S_pdf-S_phi|={np.abs(d).max():.4f} rms={np.sqrt((d**2).mean()):.4f}  (MC se rms {np.sqrt((se**2).mean()):.4f}, max {np.abs(se).max():.4f})", flush=True)
np.save("imf_pdf_target_vs_phi.npy", np.array(res))
