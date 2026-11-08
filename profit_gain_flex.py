"""Section 5.2: location of the platform's profit gain under the flexible commission.
For every point of the flexible grid, the profit path is evaluated at intensities 0, 0.0005, ..., 0.25.
Reports the intensity at which profit peaks, the size of the gain, and the first intensity at which
profit is below its zero-intensity level. Writes flex_gain.pkl (read by observations.py)."""
import numpy as np, sympy as sp, pickle, pandas as pd
from model import h, l, c, r, lam, agency

S = {'HA': agency(h), 'LA': agency(l)}
F = {nm: {k: sp.lambdify((h, l, c, r, lam), S[nm][k], 'numpy') for k in ['piR', 'Omega']} for nm in S}
Q = pickle.load(open('quartic.pkl', 'rb'))['quartic']
coef = {nm: [sp.lambdify((h, l, c, lam), cf, 'numpy') for cf in sp.Poly(Q[nm], r).all_coeffs()] for nm in Q}


def rbar(nm, hv, lv, cv, la):
    if nm == 'HA':
        return (cv * (hv - 1) + 2 - la * (2 - cv)) / (cv * (hv + 1) + 2 - 2 * la)
    return ((4 - 2 * la) - cv * (2 + lv - la)) / ((4 - 2 * la) - cv * lv)


def rhat(hv, cv):
    return hv * (4 - cv * (2 * hv + 1)) / 2


def rstar(nm, hv, lv, cv, LG):
    """Optimal commission on the intensity grid LG: root of the quartic below the incumbent's bound."""
    cfs = np.array([np.broadcast_to(np.asarray(cf(hv, lv, cv, LG), dtype=float), LG.shape) for cf in coef[nm]]).T
    mon = cfs[:, 1:] / cfs[:, 0:1]
    C = np.zeros((len(LG), 4, 4)); C[:, 0, :] = -mon; C[:, 1, 0] = 1; C[:, 2, 1] = 1; C[:, 3, 2] = 1
    roots = np.linalg.eigvals(C)
    ub = rbar(nm, hv, lv, cv, LG)
    if nm == 'HA':
        ub = np.minimum(ub, rhat(hv, cv))
    out = np.full(len(LG), np.nan)
    for i in range(len(LG)):
        cand = [x.real for x in roots[i] if abs(x.imag) < 1e-9 and 0 < x.real < ub[i]]
        if cand:
            out[i] = max(cand, key=lambda x: F[nm]['Omega'](hv, lv, cv, x, LG[i]))
    return out


LG = np.linspace(0, 0.25, 501)
H = np.round(np.arange(1.10, 2.001, 0.05), 3); L = np.round(np.arange(0.50, 0.951, 0.05), 3)
rows = []
for hv in H:
    cb = 2 / (1 + hv)
    cs = list(np.round(np.arange(0.1, cb - 1e-9, 0.025), 4))
    if cb - 0.005 > cs[-1]:
        cs.append(round(cb - 0.005, 4))
    for cv in cs:
        for nm, lvs in [('HA', [0.9]), ('LA', L)]:  # the high-end objects do not depend on l
            for lv in lvs:
                rs = rstar(nm, hv, lv, cv, LG)
                pr = np.asarray(F[nm]['piR'](hv, lv, cv, rs, LG), dtype=float)
                ip = int(np.nanargmax(pr))
                below = np.where((LG > 0) & (pr < pr[0]))[0]
                rows.append((nm, hv, lv, cv, LG[ip], pr[ip] / pr[0] - 1, LG[below[0]] if len(below) else np.nan))
d = pd.DataFrame(rows, columns=['tier', 'h', 'l', 'c', 'lam_peak', 'gain', 'lam_end'])
d.to_pickle('flex_gain.pkl')
for nm in ['HA', 'LA']:
    x = d[d.tier == nm]
    print(f'{nm}: points {len(x)}, largest peak intensity {x.lam_peak.max():.4f}, largest gain {100 * x.gain.max():.4f}%, '
          f'smallest gain {100 * x.gain.min():.2e}%, profit below its zero-intensity level from at most {x.lam_end.max():.4f}, '
          f'points without such an intensity on [0, 0.25]: {int(x.lam_end.isna().sum())}')
